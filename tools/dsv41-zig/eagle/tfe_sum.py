"""Summary of a TFE window's bench.jsonl: per phase and batch the tokens, aggregate and median decode tok/s, drafted
and accepted; and token_sha of every (category, prompt, sampling) across the phases and draft settings.

  python3 tfe_sum.py BENCH.jsonl
"""

import collections
import json
import sys


def main():
    reqs, batches = [], []
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r.get("kind") == "req":
            reqs.append(r)
        elif r.get("kind") == "batch":
            batches.append(r)
    print("phase | cat | streams | sampling | drafts | tokens | agg tok/s | median tok/s | drafted | accepted | acc rate")
    for b in batches:
        t = "t0" if b["temp"] == 0 else "t1"
        acc = round(b["accepted"] / b["drafted"], 3) if b["drafted"] else None
        print(f'{b["phase"]} | {b["cat"]} | {b["streams"]} | {t} | {"on" if b["draft"] else "off"} | {b["tokens"]} | '
              f'{b["agg_tps"]:.1f} | {b["median_tps"]:.1f} | {b["drafted"]} | {b["accepted"]} | {acc}')
    shas = collections.defaultdict(set)
    for r in reqs:
        t = "t0" if r["temp"] == 0 else "t1"
        shas[(r["cat"], r["i"], t)].add(r.get("token_sha"))
    bad = {k: v for k, v in shas.items() if len(v) != 1 or None in v}
    print(f"token_sha across all phases and draft settings: {len(shas) - len(bad)}/{len(shas)} groups one value")
    for k, v in bad.items():
        print("  differs:", k, sorted(map(str, v)))
    fails = [r for r in reqs if not r.get("ok")]
    print(f"failed requests: {len(fails)}")


if __name__ == "__main__":
    main()
