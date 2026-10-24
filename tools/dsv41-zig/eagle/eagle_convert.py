"""A TFE draft head checkpoint (eagle_train.py's HEAD.pt: {"cfg", "state"}) as the file the Zig engine loads
(TF_DS_DRAFTER=eagle TF_DS_EAGLE_HEAD=FILE): every tensor in bf16, row-major, after one JSON header.

  python3 eagle_convert.py HEAD.pt OUT.tfe [--target-head PREFIX]

--target-head: a two_stage head's rescoring table, the target's own vocabulary head from eagle_collect.py --head-dump
(PREFIX-r0.pt, PREFIX-r1.pt: {"rank", "wt" [dim, vocab / 2], "norm"}), as tensor "target_head" bf16 [vocab, dim]
(the ranks' halves in rank order). The header's "target_head" names the files, their sha256 and shapes.

File: b"TFEHEAD1", u64 little-endian header length, the JSON header (padded with spaces so the data starts at a
multiple of 4096), then the tensors (each at a multiple of 256 bytes from the data start). Header fields: the head's
shape (dim, heads, kv_heads, head_dim, inter, rank, vocab, window, rope_dim, rope_theta, eps), tap_norm, qk_norm,
head_type ("lowrank": up(down(o)) picks the draft; "two_stage": its top cand_k ids rescored by the target's own head),
cand_k, `taps` (the
tap layers in the engine's order: layers ascending, the final hidden state 1000 last) and `tensors` {name: {offset,
shape, dtype}}.

The engine writes its taps in that order. The training order (cfg["tap_layers"], the order of --layers) can differ, so
fc's input column blocks (and tap_norms' rows) are put in the engine's order here; fc's output is the same.
Only the base head (eagle_train.py Head) is supported; a "wide" head is refused.
"""

import argparse
import glob
import hashlib
import json
import os
import struct

import torch

FINAL = 1000
MAGIC = b"TFEHEAD1"
NAMES = ("fc_norm", "e_norm", "g_norm", "attn_norm", "mlp_norm", "out_norm", "fc.weight", "w_in.weight", "wq.weight",
         "wkv.weight", "wo.weight", "w13.weight", "w2.weight", "down.weight", "up.weight")


def engine_order(taps: list[int]) -> list[int]:
    """The engine's tap order: layers ascending, the final state last."""

    return sorted(t for t in taps if t != FINAL) + ([FINAL] if FINAL in taps else [])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def target_head(prefix: str, vocab: int, dim: int) -> tuple[torch.Tensor, dict]:
    """The head dump files of PREFIX as one bf16 [vocab, dim] table and their record."""

    paths = sorted(glob.glob(f"{prefix}-r*.pt"))
    dumps = [(p, torch.load(p, map_location="cpu", weights_only=True)) for p in paths]
    dumps.sort(key=lambda x: x[1]["rank"])
    if [d["rank"] for _, d in dumps] != list(range(len(dumps))) or not dumps:
        raise SystemExit(f"head dump ranks of {prefix}: {[d['rank'] for _, d in dumps]}")
    for p, d in dumps:
        if d["wt"].shape[0] != dim:
            raise SystemExit(f"{p}: wt {tuple(d['wt'].shape)}, want [{dim}, *]")
    lm = torch.cat([d["wt"].T for _, d in dumps]).contiguous().to(torch.bfloat16)
    if tuple(lm.shape) != (vocab, dim):
        raise SystemExit(f"target head {tuple(lm.shape)}, want ({vocab}, {dim})")
    rec = {"source": "eagle_collect.py --head-dump", "shape": [vocab, dim],
           "files": [{"name": os.path.basename(p), "rank": d["rank"], "wt": list(d["wt"].shape), "sha256": sha256(p)}
                     for p, d in dumps],
           "sha256_bf16": hashlib.sha256(lm.view(torch.int16).numpy().tobytes()).hexdigest()}
    return lm, rec


def convert(ck: dict, target: str | None = None) -> tuple[dict, dict]:
    """-> (header, {name: bf16 tensor}) of a checkpoint; `target`: the head dump prefix (a two_stage head)."""

    cfg = ck["cfg"]
    if cfg.get("kind", "base") != "base":
        raise SystemExit(f"head kind {cfg.get('kind')!r} is not supported (base only)")
    sd = ck["state"]
    taps = list(cfg["tap_layers"])
    if len(taps) != cfg["layers"] or len(set(taps)) != len(taps):
        raise SystemExit(f"bad tap_layers {taps} for {cfg['layers']} layers")
    order = engine_order(taps)
    perm = [taps.index(t) for t in order]
    D = cfg["dim"]
    out = {}
    for n in NAMES:
        out[n] = sd[n].float()
    fc = out["fc.weight"].view(D, len(taps), D)
    out["fc.weight"] = fc[:, perm].reshape(D, len(taps) * D)
    tap_norm = bool(cfg.get("tap_norm", False))
    if tap_norm:
        out["tap_norms"] = sd["tap_norms"].float()[perm]
        out["fb_norm"] = sd["fb_norm"].float()
    qk_norm = bool(cfg.get("qk_norm", False))
    if qk_norm:
        out["q_norm"] = sd["q_norm"].float()
        out["k_norm"] = sd["k_norm"].float()
    head_type = cfg.get("head_type", "lowrank")
    cand_k = int(cfg.get("cand_k", 0))
    if head_type not in ("lowrank", "two_stage") or (head_type == "two_stage") != (cand_k > 0):
        raise SystemExit(f"head_type {head_type!r} with cand_k {cand_k} is not supported")
    if (head_type == "two_stage") != (target is not None):
        raise SystemExit("a two_stage head needs --target-head (and only a two_stage head takes it)")
    rec = None
    if target is not None:
        out["target_head"], rec = target_head(target, cfg["vocab"], D)
    hd = cfg["head_dim"]
    if out["wkv.weight"].shape != (2 * hd, D) or out["wq.weight"].shape != (cfg["heads"] * hd, D):
        raise SystemExit("unexpected attention shapes")
    head = {"format": 1, "kind": "base", "dim": D, "heads": cfg["heads"], "kv_heads": 1, "head_dim": hd,
            "inter": cfg["inter"], "rank": cfg["rank"], "vocab": cfg["vocab"], "window": cfg["window"],
            "rope_dim": cfg["rope_dim"], "rope_theta": 10000.0, "eps": 1e-6, "tap_norm": tap_norm, "qk_norm": qk_norm,
            "head_type": head_type, "cand_k": cand_k, "taps": order,
            "train_taps": taps}
    if rec is not None:
        head["target_head"] = rec
    return head, {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}


def write(path: str, head: dict, tensors: dict) -> None:
    table, off = {}, 0
    for k, t in tensors.items():
        off = (off + 255) // 256 * 256
        table[k] = {"offset": off, "shape": list(t.shape), "dtype": "bf16"}
        off += t.numel() * 2
    head = {**head, "tensors": table, "data_bytes": off}
    text = json.dumps(head, separators=(",", ":")).encode()
    start = (len(MAGIC) + 8 + len(text) + 4095) // 4096 * 4096
    text += b" " * (start - len(MAGIC) - 8 - len(text))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(text)) + text)
        pos = 0
        for k, t in tensors.items():
            o = table[k]["offset"]
            f.write(b"\0" * (o - pos))
            f.write(t.view(torch.int16).numpy().tobytes())
            pos = o + t.numel() * 2


def read(path: str) -> tuple[dict, dict]:
    """The header and the tensors (bf16) of a converted file."""

    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != MAGIC:
        raise ValueError("not a TFE head file")
    n = struct.unpack("<Q", b[8:16])[0]
    head = json.loads(b[16:16 + n])
    start = 16 + n
    out = {}
    for k, e in head["tensors"].items():
        cnt = 1
        for s in e["shape"]:
            cnt *= s
        raw = torch.frombuffer(bytearray(b[start + e["offset"]:start + e["offset"] + 2 * cnt]), dtype=torch.int16)
        out[k] = raw.view(torch.bfloat16).view(*e["shape"])
    return head, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("head")
    p.add_argument("out")
    p.add_argument("--target-head", default=None, help="a two_stage head: the head dump prefix (PREFIX-r*.pt)")
    a = p.parse_args()
    ck = torch.load(a.head, map_location="cpu", weights_only=True)
    head, tensors = convert(ck, a.target_head)
    write(a.out, head, tensors)
    print(json.dumps({"out": a.out, "taps": head["taps"], "train_taps": head["train_taps"],
                      "head_type": head["head_type"], "cand_k": head["cand_k"],
                      "target_head": head.get("target_head"),
                      "params": sum(t.numel() for k, t in tensors.items() if k != "target_head")}))


if __name__ == "__main__":
    main()
