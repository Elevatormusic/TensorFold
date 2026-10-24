"""The TFE draft head as the Zig engine runs it (zig/src/families/dsv41/eagle.zig), in torch on the CPU: the reference
for the engine's self-test vectors, and a cross-check against the training code's chain (eagle_train.py chain_forward).

The engine's order of work for one stream (pending token at position q0, real rows below q0):
  real row p (p < q0):  g_p = RMSNorm(fc(taps_p)); x = w_in [RMSNorm(embed(x_{p+1})) ; RMSNorm(g_p)]; the layer with
                        RoPE position p; its key and value go into the stream's ring at p.
  draft step 1:         the real row q0 - 1's output o_1 -> logits -> draft d_1 (top-1).
  draft step j > 1:     position q0 + j - 2, token d_{j-1}, g = o_{j-1}; its key and value go into the ring too.
  attention of a row at position P: the ring rows at positions max(lo, P - 128) .. P (lo: the stream's first row).
Activations are bf16 between the operations (fp32 inside each), as the engine's kernels and cuBLAS calls give them.

  python3 eagle_ref.py cross HEAD.pt --train-py PATH/eagle_train.py        the chain equals chain_forward's
  python3 eagle_ref.py vectors HEAD.tfe OUT.tfev --embed EMBED.bf16 [--T 160] [--steps 5]   self-test vectors
"""

import argparse
import importlib.util
import json
import math
import struct

import torch
import torch.nn.functional as F

from eagle_convert import read

WINDOW = 128


class Ref:
    """The converted head's tensors (bf16) and its math; `rnd`: round to bf16 between the operations."""

    def __init__(self, head: dict, t: dict, embed: torch.Tensor, rnd: bool = True):
        self.h, self.t, self.embed, self.rnd = head, {k: v.float() for k, v in t.items()}, embed.float(), rnd
        self.eps = head["eps"]

    def r(self, x):
        return x.to(torch.bfloat16).float() if self.rnd else x

    def rms(self, x, w):
        return self.r(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w)

    def lin(self, x, name):
        return self.r(x @ self.t[name].T)

    def rope(self, x, pos: torch.Tensor):
        """RoPE on the first rope_dim features of each head (halves rotated), positions pos [rows]."""

        rd, theta = self.h["rope_dim"], self.h["rope_theta"]
        half = rd // 2
        f = theta ** (-torch.arange(half, dtype=torch.float32) / half)
        a = pos.float()[:, None] * f[None]
        cos, sin = a.cos()[:, None], a.sin()[:, None]
        x1, x2 = x[..., :half], x[..., half:rd]
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., rd:]], -1)
        return self.r(out)

    def fuse(self, taps):
        """taps [rows, L, D] in the engine's order -> g [rows, D]."""

        if self.h["tap_norm"]:
            taps = self.rms(taps, self.t["tap_norms"])
        return self.rms(self.lin(taps.flatten(-2), "fc.weight"), self.t["fc_norm"])

    def feedback(self, o):
        return self.rms(o, self.t["fb_norm"]) if self.h["tap_norm"] else o

    def layer_in(self, ids, g):
        e = self.r(self.embed[ids])
        x = self.lin(torch.cat([self.rms(e, self.t["e_norm"]), self.rms(g, self.t["g_norm"])], -1), "w_in.weight")
        hn = self.rms(x, self.t["attn_norm"])
        H, hd = self.h["heads"], self.h["head_dim"]
        q = self.lin(hn, "wq.weight").view(-1, H, hd)
        kv = self.lin(hn, "wkv.weight")
        return x, q, kv[:, :hd], kv[:, hd:]

    def layer_out(self, x, att):
        x = self.r(x + self.lin(att.flatten(-2), "wo.weight"))
        gu = self.lin(self.rms(x, self.t["mlp_norm"]), "w13.weight")
        g, u = gu.chunk(2, -1)
        y = self.r(F.silu(g) * u)
        return self.r(x + self.lin(y, "w2.weight"))

    def logits(self, o):
        """fp32 logits (the engine's up projection writes fp32)."""

        d = self.lin(self.rms(o, self.t["out_norm"]), "down.weight")
        return d @ self.t["up.weight"].T

    def attend(self, q, keys, vals):
        """q [H, hd]; keys / vals [n, hd] -> [H, hd] (fp32 scores, softmax, fp32 sum, bf16 out)."""

        s = (q @ keys.T) * self.h["head_dim"] ** -0.5
        return self.r(s.softmax(-1) @ vals)


def chain(ref: Ref, taps, ids, q0: int, steps: int, lo: int = 0):
    """The engine's chain for one stream: taps [>= q0, L, D], ids [>= q0 + 1] (ids[q0]: the pending token), real rows
    lo .. q0 - 1. -> (drafts [steps], o [steps, D], p_top1 [steps], logits [steps, V])."""

    P = torch.arange(lo, q0)
    g = ref.fuse(taps[lo:q0])
    x, q, k, v = ref.layer_in(ids[lo + 1:q0 + 1], g)
    q, k = ref.rope(q, P), ref.rope(k[:, None], P)[:, 0]
    ring_k = {int(p): k[i] for i, p in enumerate(P)}
    ring_v = {int(p): v[i] for i, p in enumerate(P)}
    att = []
    for i, p in enumerate(P.tolist()):
        win = list(range(max(lo, p - WINDOW), p + 1))
        att.append(ref.attend(q[i], torch.stack([ring_k[w] for w in win]), torch.stack([ring_v[w] for w in win])))
    o_all = ref.layer_out(x, torch.stack(att))
    o = o_all[-1:]
    drafts, outs, probs, lgs = [], [], [], []
    for j in range(1, steps + 1):
        if j > 1:
            pos = q0 + j - 2
            x, qj, kj, vj = ref.layer_in(torch.tensor([drafts[-1]]), ref.feedback(o))
            qj, kj = ref.rope(qj, torch.tensor([pos])), ref.rope(kj[:, None], torch.tensor([pos]))[:, 0]
            ring_k[pos], ring_v[pos] = kj[0], vj[0]
            win = list(range(max(lo, pos - WINDOW), pos + 1))
            att = ref.attend(qj[0], torch.stack([ring_k[w] for w in win]), torch.stack([ring_v[w] for w in win]))
            o = ref.layer_out(x, att[None])
        lg = ref.logits(o)[0]
        d = int(lg.argmax())
        drafts.append(d)
        outs.append(o[0])
        probs.append(float(lg.softmax(-1)[d]))
        lgs.append(lg)
    return drafts, torch.stack(outs), probs, torch.stack(lgs)


def load_train(path: str):
    spec = importlib.util.spec_from_file_location("eagle_train", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fake_inputs(L: int, D: int, V: int, T: int, seed: int):
    """Random taps with row scales like the target's (a spread of 0.05 to 20) and random token ids."""

    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.empty(T, L, 1).uniform_(math.log(0.05), math.log(20.0), generator=g))
    taps = (torch.randn(T, L, D, generator=g) * scale).to(torch.bfloat16)
    ids = torch.randint(0, V, (T,), generator=g)
    return taps, ids


def cross(a) -> None:
    """chain() without rounding against eagle_train.chain_forward (fp32, the same bf16-valued weights and inputs)."""

    import tempfile

    import eagle_convert

    tr = load_train(a.train_py)
    ck = torch.load(a.head, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    hm = tr.make_head({**cfg, "layers": len(cfg["tap_layers"])})
    hm.load_state_dict(ck["state"])
    with torch.no_grad():
        for p in hm.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    hm.eval()
    head, tensors = eagle_convert.convert(ck)
    D, V, L = head["dim"], head["vocab"], len(head["taps"])
    g = torch.Generator().manual_seed(7)
    embed = (torch.randn(V, D, generator=g) * 0.02).to(torch.bfloat16)
    ref = Ref(head, tensors, embed, rnd=False)
    perm = [head["train_taps"].index(t) for t in head["taps"]]                 # engine order <- training order
    T, S = a.T, a.steps
    taps, ids = fake_inputs(L, D, V, T + S, 3)
    worst = 0.0
    out = []
    for q0 in a.q0:
        d, o, _, _ = chain(ref, taps.float(), ids, q0, S)
        ids_tf = ids.clone()
        ids_tf[q0 + 1:q0 + S] = torch.tensor(d[:S - 1])                       # chain_forward's tokens = our drafts
        taps_train = torch.empty_like(taps)
        taps_train[:, perm] = taps                                            # back to the training order
        n = q0 + S
        with torch.no_grad():
            outs = tr.chain_forward(hm, embed.float(), taps_train[:n].float(), ids_tf[:n], torch.arange(n),
                                    torch.zeros(n, dtype=torch.long), S, hidden=True)
        t = q0 - 1
        for j in range(S):
            err = float((outs[j][t] - o[j]).abs().max() / o[j].abs().max())
            worst = max(worst, err)
            out.append({"q0": q0, "step": j + 1, "rel_err": round(err, 7)})
    print(json.dumps({"cross": "chain vs chain_forward", "worst_rel_err": worst, "cases": out}))
    if worst > 1e-4:
        raise SystemExit("MISMATCH")


MAGIC_V = b"TFEVECS1"


def vectors(a) -> None:
    """Self-test vectors for the engine: cases of one stream with the pending token at q0 (real rows 0 .. q0 - 1) and
    the expected drafts, outputs and top-1 probabilities of `steps` steps. --embed: the target's embedding (raw bf16
    [vocab, dim], the checkpoint's embed.weight bytes), the table the engine's head reads."""

    head, tensors = read(a.head)
    D, V, L = head["dim"], head["vocab"], len(head["taps"])
    T, S = a.T, a.steps
    taps, ids = fake_inputs(L, D, V, T + S, 11)
    embed = torch.frombuffer(bytearray(open(a.embed, "rb").read()), dtype=torch.int16).view(torch.bfloat16).view(V, D)
    ref = Ref(head, tensors, embed, rnd=True)
    cases = []
    for q0 in a.q0:
        d, o, p, lg = chain(ref, taps.float(), ids, q0, S)
        top2 = lg.topk(2, -1).values
        cases.append({"q0": q0, "drafts": d, "p_top1": p, "gap": (top2[:, 0] - top2[:, 1]).tolist(), "o": o})
        print(json.dumps({"q0": q0, "drafts": d, "p_top1": [round(x, 5) for x in p]}))
    meta = {"T": T, "steps": S, "dim": D, "layers": L, "taps": head["taps"], "vocab": V,
            "q0": [c["q0"] for c in cases], "drafts": [c["drafts"] for c in cases],
            "p_top1": [c["p_top1"] for c in cases], "gap": [c["gap"] for c in cases]}
    blobs = [taps.view(torch.int16).numpy().tobytes(), ids.to(torch.int32).numpy().tobytes(),
             torch.cat([c["o"] for c in cases]).float().numpy().tobytes()]
    meta["blobs"] = ["taps bf16 [T + steps, L, D]", "ids int32 [T + steps]", "o fp32 [cases, steps, D]"]
    meta["sizes"] = [len(b) for b in blobs]
    text = json.dumps(meta).encode()
    with open(a.out, "wb") as f:
        f.write(MAGIC_V + struct.pack("<Q", len(text)) + text)
        for b in blobs:
            f.write(b)
    print(json.dumps({"out": a.out, "cases": len(cases), "bytes": 16 + len(text) + sum(meta["sizes"])}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cross", "vectors"])
    p.add_argument("head")
    p.add_argument("out", nargs="?")
    p.add_argument("--train-py", default=None)
    p.add_argument("--embed", default=None, help="vectors: the target embedding, raw bf16 [vocab, dim]")
    p.add_argument("--T", type=int, default=160)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--q0", type=int, nargs="+", default=[1, 2, 64, 129, 130, 160])
    a = p.parse_args()
    cross(a) if a.mode == "cross" else vectors(a)


if __name__ == "__main__":
    main()
