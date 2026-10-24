"""The TFE head's few-row linears, attention and top-1 timed on any CUDA GPU (CUDA events, the median of 50 runs):
the GB/s of each weight read. Driver API through ctypes, as eagle_kernels_test.py.

  python3 eagle_kernels_time.py EAGLE.fatbin
"""

import ctypes
import json
import statistics
import sys

import torch

import eagle_kernels_test as kt  # (runs its checks first: a broken kernel is not timed)

cu = kt.cu


def timed(f, grid, block, args, reps=50):
    vals = []
    for kind, v in args:
        vals.append({"p": ctypes.c_uint64, "ll": ctypes.c_longlong, "i": ctypes.c_int, "f": ctypes.c_float}[kind](v))
    arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    g = grid + (1,) * (3 - len(grid))
    e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
    kt.ok(cu.cuEventCreate(ctypes.byref(e0), 0), "event")
    kt.ok(cu.cuEventCreate(ctypes.byref(e1), 0), "event")
    out = []
    for _ in range(reps + 3):
        kt.ok(cu.cuEventRecord(e0, None), "record")
        kt.ok(cu.cuLaunchKernel(f, g[0], g[1], g[2], block, 1, 1, 0, None, arr, None), "launch")
        kt.ok(cu.cuEventRecord(e1, None), "record")
        kt.ok(cu.cuEventSynchronize(e1), "sync")
        ms = ctypes.c_float()
        kt.ok(cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1), "elapsed")
        out.append(ms.value * 1e3)
    return statistics.median(out[3:])


gen = torch.Generator().manual_seed(1)
res = {}
for name, (n, k) in {"w_in": (5120, 10240), "wq": (4096, 5120), "wo": (5120, 4096), "w13": (8192, 5120),
                     "up": (129280, 256), "fc": (5120, 15360)}.items():
    Wb = (torch.randn(n, k, generator=gen) * 0.02).to(torch.bfloat16)
    q8 = Wb.float().to(torch.float8_e4m3fn)
    s8 = torch.ones(n)
    for rows in (1, 4):
        x = torch.randn(rows, k, generator=gen).to(torch.bfloat16)
        o = torch.zeros(rows, n, dtype=torch.bfloat16)
        px, po, ps = kt.dev_t(x), kt.dev_t(o), kt.dev_t(s8)
        for f8 in (False, True):
            pw = kt.dev_t(q8.view(torch.uint8) if f8 else Wb)
            eb = 1 if f8 else 2
            G = min(32, k * eb // 16)
            grid = (n + 8 * (32 // G) - 1) // (8 * (32 // G))
            if f8:
                us = timed(kt.fn("tf_eagle_gemv_fp8_kernel"), (grid,), 256, [("p", pw.value), ("p", ps.value), ("i", n), ("i", k), ("p", px.value), ("ll", k), ("i", rows), ("p", po.value), ("ll", n), ("i", 0), ("i", G)])
            else:
                us = timed(kt.fn("tf_eagle_gemv_bf16_kernel"), (grid,), 256, [("p", pw.value), ("i", n), ("i", k), ("p", px.value), ("ll", k), ("i", rows), ("p", po.value), ("ll", n), ("i", 0), ("i", G)])
            res[f"{name}_r{rows}_{'fp8' if f8 else 'bf16'}"] = f"{us:.1f} us {n * k * eb / us / 1e3:.0f} GB/s"
            cu.cuMemFree_v2(pw)
        for t in (px, po, ps):
            cu.cuMemFree_v2(t)

# attention: 1 row and 4 rows, window 128 and 511
H, hd = 32, 128
for W in (128, 511):
    ring = W + 16
    K = torch.randn(4 * ring + 1, hd, generator=gen).to(torch.bfloat16)
    V = torch.randn(4 * ring + 1, hd, generator=gen).to(torch.bfloat16)
    for N in (1, 4):
        q = torch.randn(N, H * hd, generator=gen).to(torch.bfloat16)
        args = [kt.dev_t(t) for t in (q, K, V, torch.arange(N) * ring, torch.full((N,), 5000), torch.zeros(N, dtype=torch.long),
                                       torch.zeros(N, H * hd, dtype=torch.bfloat16))]
        us = timed(kt.fn("tf_eagle_attn_kernel"), (N, H), 128, [("p", args[0].value), ("ll", H * hd), ("p", args[1].value), ("p", args[2].value), ("i", 1), ("p", args[3].value), ("p", args[4].value), ("p", args[5].value), ("i", ring), ("i", W), ("i", H), ("f", hd ** -0.5), ("p", args[6].value), ("ll", H * hd)])
        res[f"attn_w{W}_r{N}"] = f"{us:.1f} us"

# top-1 of 129280 logits: one block against the split
Vv = 129280
for N in (1, 4):
    lg = torch.randn(N, Vv, generator=gen)
    pl, pi, pc = kt.dev_t(lg), kt.dev_t(torch.zeros(N, dtype=torch.long)), kt.dev_t(torch.zeros(N))
    res[f"top1_one_r{N}"] = f"{timed(kt.fn('tf_eagle_top1_kernel'), (N,), 1024, [('p', pl.value), ('ll', Vv), ('i', Vv), ('p', 0), ('ll', 0), ('p', pi.value), ('p', pc.value), ('f', 1.0), ('f', 0.0)]):.1f} us"
    pm, pz, pq = kt.dev_t(torch.zeros(N, 64)), kt.dev_t(torch.zeros(N, 64)), kt.dev_t(torch.zeros(N, 64, dtype=torch.int32))
    a = timed(kt.fn("tf_eagle_top1_part_kernel"), (N, 64), 256, [("p", pl.value), ("ll", Vv), ("i", Vv), ("p", pm.value), ("p", pz.value), ("p", pq.value)])
    b = timed(kt.fn("tf_eagle_top1_join_kernel"), (N,), 32, [("p", pm.value), ("p", pz.value), ("p", pq.value), ("i", 64), ("p", pi.value), ("p", pc.value), ("f", 1.0), ("f", 0.0)])
    res[f"top1_split_r{N}"] = f"{a:.1f} + {b:.1f} us"
print(json.dumps(res, indent=1))
