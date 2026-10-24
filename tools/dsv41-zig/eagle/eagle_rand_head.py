"""A random TFE head checkpoint in eagle_train.py's format (HEAD.pt: {"cfg", "state"}), to test the converter, the
reference and the engine on a head shape before a trained head exists. Norm weights are drawn from 0.5 to 1.5, so a
norm in the wrong place shows.

  python3 eagle_rand_head.py PATH/eagle_train.py OUT.pt --taps 26 2 37 1000 [--tap-norm] [--qk-norm] [--window 511]
      [--cand-k 128]

--taps: the training order (the converter puts them in the engine's order); 1000 is the final hidden state.
Then: eagle_ref.py cross OUT.pt --train-py PATH/eagle_train.py (the reference against the training code), and
eagle_convert.py + eagle_ref.py vectors for the engine's self-test (TF_DS_EAGLE_CHECK).
"""

import argparse
import importlib.util

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train_py")
    p.add_argument("out")
    p.add_argument("--taps", type=int, nargs="+", required=True)
    p.add_argument("--tap-norm", action="store_true")
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--cand-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("eagle_train", a.train_py)
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    torch.manual_seed(a.seed)
    cfg = {"layers": len(a.taps), "tap_norm": a.tap_norm, "heads": 32, "head_dim": 128, "inter": 4096, "rank": 256,
           "dim": 5120, "vocab": 129280, "qk_norm": a.qk_norm, "cand_k": a.cand_k}
    if a.window is not None:
        cfg["window"] = a.window
    h = tr.make_head(cfg)
    with torch.no_grad():
        for _, w in h.named_parameters():
            if w.dim() == 1 or (w.dim() == 2 and w.shape[0] == len(a.taps)):   # the norms (tap_norms is [taps, dim])
                w.copy_(torch.rand_like(w) + 0.5)
    out = dict(h.cfg)
    out["tap_layers"] = list(a.taps)
    print(out)
    torch.save({"cfg": out, "state": h.state_dict()}, a.out)


if __name__ == "__main__":
    main()
