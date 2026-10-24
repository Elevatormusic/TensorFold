"""Acceptance by position for each drafted batch of a phase, from the engine's TF_DS_ACCEPT_LOG lines (cumulative).

  python3 tfe_seg.py ENGINE_HEAD.log

The drafted batches run in a fixed order (tfe_bench.py: per category 1 stream, 4 streams t1, 4 streams t0; draft
false between them). A drafted batch is a run of lines where the windows count rises; a draft-false batch between two
drafted ones leaves it flat for many rounds. Runs split at a flat stretch of 60 rounds or more.
"""

import json
import sys

ORDER = [(c, s, t) for c in ("prose", "code", "tools") for s, t in ((1, "t1"), (4, "t1"), (4, "t0"))]


def main():
    rows = []
    for line in open(sys.argv[1], errors="replace"):
        k = line.find('{"accept_by_pos"')
        if k >= 0:
            try:
                rows.append(json.loads(line[k:].replace("{ ", "[").replace(" }", "]"))["accept_by_pos"])
            except ValueError:
                pass
    segs = []
    start = prev = None
    flat_from = None
    for r in rows:
        if prev is None:
            prev = r
            continue
        if r["windows"] > prev["windows"]:
            if start is None:
                start = prev
            flat_from = None
            last = r
        elif start is not None:
            flat_from = flat_from or prev
            if r["rounds"] - flat_from["rounds"] >= 60:
                segs.append((start, last))
                start = None
                flat_from = None
        prev = r
    if start is not None:
        segs.append((start, last))
    print(f"{len(segs)} drafted runs (want {len(ORDER)} for a full phase, 3 for --quick)")
    for i, (a, b) in enumerate(segs):
        asked = [x - y for x, y in zip(b["asked"], a["asked"])]
        kept = [x - y for x, y in zip(b["kept"], a["kept"])]
        pos = [round(k / n, 2) if n else None for k, n in zip(kept, asked)][:5]
        lab = ORDER[i] if len(segs) == len(ORDER) else (i,)
        print(json.dumps({"batch": lab, "rounds": b["rounds"] - a["rounds"], "windows": b["windows"] - a["windows"], "asked": asked[:5], "kept": kept[:5],
                          "accept_by_pos": pos, "kept_per_window": round(sum(kept) / max(1, b["windows"] - a["windows"]), 3)}))


if __name__ == "__main__":
    main()
