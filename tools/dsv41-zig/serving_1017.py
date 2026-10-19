#!/usr/bin/env python3
"""serving_1017.py MODEL_DIR [--template FILE] [--check]: make a DeepSeek-V4.1-Flash-1017 EXL3 pack servable by the
engine at this branch. Standard library only, no GPU; it reads the pack's headers and about 130 MB of its tensors and
changes no downloaded file.

MODEL_DIR is a download of a 1017 EXL3 pack (ours 4.0 bpw, LibertAI/Mia 2.9 bpw, a community 2.8 bpw). The script
adds up to two files to it:

- serving-1017.safetensors: the MTP markov head under the names the engine reads, where the pack's converter wrote
  them differently. The 1017 converter writes mtp.N.markov_head.markov_w1.weight / markov_w2.weight; the engine reads
  mtp.N.markov_head.embed.weight / head.weight. The two are byte copies, so the file costs 132 MB and nothing is
  recomputed. A pack that already has the engine's names (a pack this script readied, or a community pack that kept
  the checkpoint's own names) gets no shim. The engine (and the Python line's rank-cache writer) maps every
  *.safetensors file in the folder in name order, a later file winning; serving-1017.safetensors sorts after model-*,
  layer-* and native.safetensors, so its names replace the converter's.
  The Engram q / k tables need no shim: the 1017 packs hold them in bf16, and both the engine (checkpoint.zig widens
  bf16 -> f32 exactly) and the Python line (.to(torch.float32)) read them as f32. The markov head's bf16 (the 1017
  checkpoint's own dtype; some packs keep it) needs none either: the engine converts it to the fp16 its kernel reads
  at load (weights.zig), exactly.
- chat_template.jinja: this repository's tools/dsv41-zig/serving/chat_template.jinja (sha256
  b581d19d13f5c678d72ae197781c016c8331174cf69118017f20dadbb7e2e6a4, checked against DeepSeek's reference renderings).
  The 1017 checkpoint ships no chat template and the server needs one for chat requests.

--check changes nothing. It exits 0 when the files in MODEL_DIR equal what the script would write, and 1 otherwise.
Either way it prints the sha256 of the files it would write (and of the ones present).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys

OUT = "serving-1017.safetensors"
TEMPLATE = "chat_template.jinja"
MARKOV = re.compile(r"^(mtp\.\d+)\.markov_head\.(?:markov_w([12])|(embed|head))\.weight$")
ENGRAM_QK = re.compile(r"^layers\.\d+\.engram\.[qk]_weight$")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def entries(model_dir):
    """Every tensor of the folder by name, the way the engine maps it: files in name order, a later file winning."""
    out = {}
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, name)
        base, h = header(path)
        for k, v in h.items():
            out[k] = (path, base, v)
    return out


def read_tensor(entry):
    path, base, info = entry
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        data = f.read(b - a)
    if len(data) != b - a:
        sys.exit(f"{path}: short read (an incomplete download?)")
    return data


def encode(tensors):
    """the file's bytes: tensors is a list of (name, dtype, shape, bytes) in file order"""
    hdr, off = {}, 0
    for name, dt, shape, data in tensors:
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hdr["__metadata__"] = {"format": "pt"}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    return [struct.pack("<Q", len(hb)), hb] + [t[3] for t in tensors]


def sha256_of(parts):
    s = hashlib.sha256()
    for p in parts:
        s.update(p)
    return s.hexdigest()


def sha256_file(path):
    s = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            s.update(block)
    return s.hexdigest()


def plan_shim(model_dir):
    """The markov tensors the pack needs under the engine's names, in name order; ([], []) when it needs none.

    Returns (parts, names, notes): the safetensors bytes to write, the tensor names they hold, and what the pack's
    dtypes mean for the engine (printed as notes)."""
    have = entries(model_dir)
    if not have:
        sys.exit(f"no *.safetensors in {model_dir}: is MODEL_DIR a 1017 EXL3 pack?")
    shim, notes, markov, engram = {}, [], False, False
    for name in sorted(have):
        m = MARKOV.match(name)
        if m:
            markov = True
            stem, conv, plain = m.group(1), m.group(2), m.group(3)
            engine = f"{stem}.markov_head.{'embed' if (conv == '1' or plain == 'embed') else 'head'}.weight"
            path, base, info = have[name]
            if engine.endswith("head.weight") and info["dtype"] == "BF16":
                notes.append(f"{name} is bf16: the engine converts it to fp16 at load (weights.zig)")
            if name != engine:
                if engine in shim:
                    sys.exit(f"two tensors would write {engine}")
                shim[engine] = (info["dtype"], info["shape"], read_tensor((path, base, info)))
            continue
        if ENGRAM_QK.match(name):
            engram = True
            info = have[name][2]
            if info["dtype"] != "F32":
                notes.append(f"{name} is {info['dtype']}: the engine widens it to f32 at load (checkpoint.zig)")
    if not markov or not engram:
        sys.exit(f"{model_dir}: expected the MTP markov head and the Engram q / k of a 1017 pack")
    names = sorted(shim)
    if not names:
        return [], [], notes
    return encode([(n, *shim[n]) for n in names]), names, notes


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--template", default=os.path.join(here, "serving", TEMPLATE))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = a.model_dir
    parts, names, notes = plan_shim(d)
    tpl = open(a.template, "rb").read()
    tpl_sha = hashlib.sha256(tpl).hexdigest()
    shim_p = os.path.join(d, OUT)
    tpl_p = os.path.join(d, TEMPLATE)
    shim_sha = sha256_of(parts) if parts else None
    if a.check:
        ok = True
        got = sha256_file(shim_p) if os.path.isfile(shim_p) else "absent"
        want = shim_sha if shim_sha else "absent"
        print(f"{OUT}: {got} ({'ok' if got == want else 'expected ' + want})")
        ok = ok and got == want
        got = sha256_file(tpl_p) if os.path.isfile(tpl_p) else "absent"
        print(f"{TEMPLATE}: {got} ({'ok' if got == tpl_sha else 'expected ' + tpl_sha})")
        ok = ok and got == tpl_sha
        sys.exit(0 if ok else 1)
    if parts:
        tmp = shim_p + ".partial"
        with open(tmp, "wb") as f:
            for p in parts:
                f.write(p)
        os.replace(tmp, shim_p)
        print(f"{OUT}: {', '.join(names)}")
        print(f"{OUT}: {shim_sha}")
    else:
        print(f"{OUT}: not needed (the pack already has the engine's markov names)")
    for note in notes:
        print(f"note: {note}")
    if os.path.abspath(a.template) != os.path.abspath(tpl_p):
        shutil.copyfile(a.template, tpl_p)
    print(f"{TEMPLATE}: {tpl_sha}")


if __name__ == "__main__":
    main()
