"""Validate a LoRA adapter at default strength 1.0 without loading base model weights.

Print its modules and table bytes, and optionally write tensorfold_adapter.json. Runtime checks the configured strength;
an adapter with a compensating strength can be valid even when this default-strength check refuses it.

  python3 adapter_check.py ADAPTER_DIR MODEL_DIR [--tp 2] [--slots 2] [--write-manifest] [--json]

``--slots`` is the number of adapters the server will hold (the tables keep a slot for each and one for the base model).
"""

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tensorfold.families.deepseek_v41.config import Cfg  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import lora_format as lf  # noqa: E402
from tensorfold.families.deepseek_v41.cuda.lora_place import span  # noqa: E402


def rank_bytes(spec: lf.AdapterSpec, dims: lf.Dims, world: int, slots: int) -> int:
    """Table bytes for this adapter's module shapes, including each slot's int32 rank and float32 scale."""

    if slots < 1:
        raise ValueError("--slots must be positive (the number of configured adapters)")
    total = 0
    for m in spec.modules:
        width, out = dims.shape(m.kind)
        lo, hi = span(width, 0, world)
        total += (slots + 1) * (lf.padded_rank(m.rank) * (hi - lo + out) + 2) * 4
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter")
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--write-manifest", action="store_true", help="write tensorfold_adapter.json from the files")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    model = pathlib.Path(args.model)
    cfg = Cfg.read(model)
    dims = lf.Dims.of(cfg)
    config = model / "config.json"
    sha = hashlib.sha256(config.read_bytes()).hexdigest()
    try:
        spec = lf.inspect_folder(args.adapter, dims, config_sha=sha)
        nbytes = rank_bytes(spec, dims, args.tp, args.slots)
        for module in spec.modules:
            lf.effective_scale(module.scale, 1.0, module.name)
        with lf.snapshot_weights(spec.weights, spec.sha256) as (source, _):
            parsed = lf.read_header(source)
            for module in spec.modules:                              # the checks read the bytes whose hash we report
                lf.read_factors(spec, module, parsed, source=source)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    report = spec.describe()
    report["base"] = {"model_type": "deepseek_v41", "hidden_size": dims.dim, "num_hidden_layers": dims.n_layers,
                      "config_sha256": sha}
    report["bytes_per_rank"] = nbytes
    report["checked_strength"] = 1.0
    if args.write_manifest:
        manifest = {"format": lf.FORMAT, "base": report["base"], "files": {lf.WEIGHTS: {"sha256": spec.sha256}},
                    "modules": {m.name: {"rank": m.rank} for m in spec.modules},
                    "provenance": dict(spec.manifest.get("provenance") or {})}
        (pathlib.Path(args.adapter) / lf.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"ok: {len(spec.modules)} modules, weights sha256 {spec.sha256[:16]}")
    print("checked strength: 1.0 (default; runtime checks the configured strength)")
    for m in spec.modules:
        print(f"  {m.name}: rank {m.rank}, alpha {m.alpha:g}, scale {m.scale:g}")
    print(f"float32 tables on a rank of {args.tp}, {args.slots} adapter(s) held: "
          f"{report['bytes_per_rank'] / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
