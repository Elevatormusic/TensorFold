"""A TFE draft head checkpoint (eagle_train.py's HEAD.pt: {"cfg", "state"}) as the file the Zig engine loads
(TF_DS_DRAFTER=eagle TF_DS_EAGLE_HEAD=FILE): every tensor in bf16, row-major, after one JSON header.

  python3 eagle_convert.py HEAD.pt OUT.tfe

File: b"TFEHEAD1", u64 little-endian header length, the JSON header (padded with spaces so the data starts at a
multiple of 4096), then the tensors (each at a multiple of 256 bytes from the data start). Header fields: the head's
shape (dim, heads, kv_heads, head_dim, inter, rank, vocab, window, rope_dim, rope_theta, eps), tap_norm, `taps` (the
tap layers in the engine's order: layers ascending, the final hidden state 1000 last) and `tensors` {name: {offset,
shape, dtype}}.

The engine writes its taps in that order. The training order (cfg["tap_layers"], the order of --layers) can differ, so
fc's input column blocks (and tap_norms' rows) are put in the engine's order here; fc's output is the same.
Only the base head (eagle_train.py Head) is supported; a "wide" head is refused.
"""

import json
import struct
import sys

import torch

FINAL = 1000
MAGIC = b"TFEHEAD1"
NAMES = ("fc_norm", "e_norm", "g_norm", "attn_norm", "mlp_norm", "out_norm", "fc.weight", "w_in.weight", "wq.weight",
         "wkv.weight", "wo.weight", "w13.weight", "w2.weight", "down.weight", "up.weight")


def engine_order(taps: list[int]) -> list[int]:
    """The engine's tap order: layers ascending, the final state last."""

    return sorted(t for t in taps if t != FINAL) + ([FINAL] if FINAL in taps else [])


def convert(ck: dict) -> tuple[dict, dict]:
    """-> (header, {name: bf16 tensor}) of a checkpoint."""

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
    hd = cfg["head_dim"]
    if out["wkv.weight"].shape != (2 * hd, D) or out["wq.weight"].shape != (cfg["heads"] * hd, D):
        raise SystemExit("unexpected attention shapes")
    head = {"format": 1, "kind": "base", "dim": D, "heads": cfg["heads"], "kv_heads": 1, "head_dim": hd,
            "inter": cfg["inter"], "rank": cfg["rank"], "vocab": cfg["vocab"], "window": cfg["window"],
            "rope_dim": cfg["rope_dim"], "rope_theta": 10000.0, "eps": 1e-6, "tap_norm": tap_norm, "taps": order,
            "train_taps": taps}
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
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    head, tensors = convert(ck)
    write(sys.argv[2], head, tensors)
    print(json.dumps({"out": sys.argv[2], "taps": head["taps"], "train_taps": head["train_taps"],
                      "params": sum(t.numel() for t in tensors.values())}))


if __name__ == "__main__":
    main()
