"""TFE head live check client (stdlib only). Runs one phase against the engine (TFE_URL, default 127.0.0.1:8899) and
writes JSON lines.

  python3 -I tfe_bench.py PHASE OUT.jsonl [--quick]

Per category (prose, code, tools) and stream count (1, 4): the requests at the served sampling (temperature 1.0,
top-k 20, a fixed seed each), drafts on; then the same requests with "draft": false (token_sha must be equal). Also one
greedy pass (temperature 0) of each prompt, drafts on and off. Per batch: the server decode tok/s of each reply, the
batch's tokens over its wall time, and the engine's accept_by_pos counters (docker logs of rank 0) before and after.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

URL = os.environ.get("TFE_URL", "http://127.0.0.1:8899/v1/chat/completions")
MODEL = "DeepSeek-v4.1-Flash-1017-TF"
CONTAINER = os.environ.get("TFE_CONTAINER", "tfebench")  # rank 0's container (its stderr has the accept lines)

PROSE = [
    "Write a 350-word short story about a lighthouse keeper who finds a message in a bottle. Use plain prose.",
    "Explain to a high-school student how vaccines train the immune system. About 300 words, no lists.",
    "Describe a walk through a busy morning market in a coastal town, in about 300 words.",
    "Write a 300-word essay on why public libraries still matter.",
]
CODE = [
    "Write a Python function that parses an ISO 8601 duration string (like P3DT4H5M) into seconds, with tests using unittest.",
    "Write a C function that reverses a singly linked list in place, plus a small main() that tests it.",
    "Write a TypeScript class for an LRU cache with get and put in O(1), with comments.",
    "Write a bash script that finds the 10 largest files under a directory and prints their sizes in human-readable form.",
]
TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Search flights between two airports on a date.",
        "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                                                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                                                        "passengers": {"type": "integer"}},
                       "required": ["origin", "destination", "date"]}}},
]
TOOLS = [
    [{"role": "user", "content": "What is the weather in Lisbon and in Oslo right now, in Celsius? Call the tools."}],
    [{"role": "user", "content": "Find flights from SFO to JFK on 2026-11-14 for 2 passengers."}],
    [{"role": "user", "content": "What is the weather in Tokyo?"},
     {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\", \"unit\": \"c\"}"}}]},
     {"role": "tool", "tool_call_id": "call_1", "content": "{\"city\": \"Tokyo\", \"temp_c\": 17, \"sky\": \"light rain\", \"wind_kmh\": 12, \"humidity\": 81}"},
     {"role": "user", "content": "Summarize that for me and suggest what to wear today, in about 120 words."}],
    [{"role": "user", "content": "Book-planning: first check the weather in Rome, then search flights from LHR to FCO on 2026-12-03 for 1 passenger."}],
]
CATS = {"prose": [[{"role": "user", "content": p}] for p in PROSE],
        "code": [[{"role": "user", "content": p}] for p in CODE],
        "tools": TOOLS}
MAXTOK = {"prose": 512, "code": 512, "tools": 256}


def accept_line():
    try:
        out = subprocess.run(["docker", "logs", "--tail", "2000", CONTAINER], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=30).stdout
    except Exception:
        return None
    last = None
    for line in out.decode("utf-8", "replace").splitlines():
        k = line.find('{"accept_by_pos"')
        if k >= 0:
            try:
                last = json.loads(line[k:].replace("{ ", "[").replace(" }", "]"))["accept_by_pos"]
            except ValueError:
                pass
    return last


def one(cat, i, msgs, draft, temp, seed):
    body = {"model": MODEL, "messages": msgs, "max_tokens": MAXTOK[cat], "seed": seed, "draft": draft}
    if temp is not None:
        body["temperature"] = temp
    if cat == "tools":
        body["tools"] = TOOLS_SPEC
    t0 = time.time()
    rec = {"cat": cat, "i": i, "draft": draft, "temp": temp, "seed": seed}
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        rt = r.get("tensorfold") or r.get("runtime") or {}
        sp = r.get("speculative") or {}
        ch = r["choices"][0]
        rec.update(ok=True, wall=time.time() - t0, tokens=r["usage"]["completion_tokens"],
                   tps=rt.get("tokens_per_second"), ttft=rt.get("time_to_first_token"), token_sha=rt.get("token_sha"),
                   drafted=sp.get("drafted"), accepted=sp.get("accepted"), rounds=sp.get("rounds"),
                   finish=ch.get("finish_reason"), tool_calls=len(ch["message"].get("tool_calls") or []),
                   head=(ch["message"].get("content") or "")[:80], keys=sorted(r.keys()))
    except Exception as e:  # a failed request is a line too
        rec.update(ok=False, wall=time.time() - t0, error=repr(e)[:300])
    return rec


def batch(cat, streams, draft, temp, out, phase):
    msgs = CATS[cat]
    recs = [None] * len(msgs)
    a0 = accept_line()
    t0 = time.time()
    if streams == 1:
        for i, m in enumerate(msgs):
            recs[i] = one(cat, i, m, draft, temp, 1000 + i)
    else:
        th = []
        for i, m in enumerate(msgs):
            def run(i=i, m=m):
                recs[i] = one(cat, i, m, draft, temp, 1000 + i)
            th.append(threading.Thread(target=run))
        for t in th:
            t.start()
        for t in th:
            t.join()
    wall = time.time() - t0
    time.sleep(1.5)
    a1 = accept_line()
    for r in recs:
        out.write(json.dumps({"phase": phase, "kind": "req", "streams": streams, **r}) + "\n")
    toks = sum(r.get("tokens") or 0 for r in recs)
    acc = None
    if a0 and a1:
        acc = {"rounds": a1["rounds"] - a0["rounds"], "windows": a1["windows"] - a0["windows"],
               "asked": [x - y for x, y in zip(a1["asked"], a0["asked"])],
               "kept": [x - y for x, y in zip(a1["kept"], a0["kept"])]}
    elif a1:
        acc = {"cumulative": a1}
    tps = sorted(r["tps"] for r in recs if r.get("tps"))
    line = {"phase": phase, "kind": "batch", "cat": cat, "streams": streams, "draft": draft, "temp": temp,
            "ok": sum(1 for r in recs if r.get("ok")), "n": len(recs), "tokens": toks, "wall": wall,
            "agg_tps": toks / wall if wall > 0 else 0, "median_tps": tps[len(tps) // 2] if tps else None,
            "drafted": sum(r.get("drafted") or 0 for r in recs), "accepted": sum(r.get("accepted") or 0 for r in recs),
            "accept_by_pos": acc}
    out.write(json.dumps(line) + "\n")
    out.flush()
    print(json.dumps({k: line[k] for k in ("cat", "streams", "draft", "temp", "ok", "tokens", "agg_tps", "median_tps", "drafted", "accepted")}), flush=True)
    return recs


def main():
    phase, path = sys.argv[1], sys.argv[2]
    quick = "--quick" in sys.argv
    with open(path, "a") as out:
        shas = {}
        for cat in ("prose", "code", "tools"):
            for streams in ((1,) if quick else (1, 4)):
                for draft in (True, False):
                    for r in batch(cat, streams, draft, None, out, phase):
                        shas.setdefault((cat, r["i"], "t1"), {})[(streams, draft)] = r.get("token_sha")
            for draft in (True, False):
                for r in batch(cat, 4, draft, 0, out, phase):
                    shas.setdefault((cat, r["i"], "t0"), {})[(4, draft)] = r.get("token_sha")
        same = sum(1 for v in shas.values() if len(set(v.values())) == 1 and None not in v.values())
        verdict = {"phase": phase, "kind": "sha", "groups": len(shas), "all_equal": same,
                   "diff": [f"{k[0]}{k[1]}{k[2]}:" + ",".join(f"{s}{'d' if d else 'n'}={h}" for (s, d), h in v.items())
                            for k, v in shas.items() if len(set(v.values())) != 1 or None in v.values()]}
        out.write(json.dumps(verdict) + "\n")
        print(json.dumps(verdict), flush=True)
        print("PHASE_DONE " + phase, flush=True)


if __name__ == "__main__":
    main()
