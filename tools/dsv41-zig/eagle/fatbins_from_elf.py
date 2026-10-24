"""The kernel fatbins a Zig CUDA binary embeds (zig/src/cuda/kernels.zig: one `kernels.Blob("fatbin_<name>").bytes`
symbol each), written out as <name>.fatbin: the input of `zig build -Dfatbins=DIR`, so a build reuses another build's
kernels byte for byte (no nvcc). The binary must keep its symbols (no -Dstrip).

  python3 fatbins_from_elf.py BINARY OUT_DIR
"""

import os
import re
import struct
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    b = open(sys.argv[1], "rb").read()
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if b[:4] != b"\x7fELF" or b[4] != 2 or b[5] != 1:
        raise SystemExit("not a 64-bit little-endian ELF")
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    secs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)
        secs.append((name, typ, addr, off, size, link, entsize))
    symtab = [s for s in secs if s[1] == 2]
    if not symtab:
        raise SystemExit("no symbol table (a stripped binary)")
    _, _, _, off, size, link, entsize = symtab[0]
    stroff = secs[link][3]
    pat = re.compile(rb'kernels\.Blob\("fatbin_([a-z0-9_]+)"')
    got = {}
    for k in range(size // entsize):
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", b, off + k * entsize)
        end = b.index(b"\0", stroff + st_name)
        m = pat.match(b[stroff + st_name:end])
        if not m or st_shndx >= len(secs):
            continue
        sec = secs[st_shndx]
        data = b[sec[3] + (st_value - sec[2]):][:st_size]
        if struct.unpack_from("<I", data, 0)[0] != 0xBA55ED50:
            raise SystemExit(f"{m.group(1).decode()}: not a fatbin")
        name = m.group(1).decode()
        open(os.path.join(out, f"{name}.fatbin"), "wb").write(data)
        got[name] = len(data)
    for n, s in sorted(got.items()):
        print(f"{n} {s}")
    print(f"{len(got)} fatbins")


if __name__ == "__main__":
    main()
