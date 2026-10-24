"""The TFE head's row kernels (zig/kernels/cuda/dsv41/eagle.cu) against torch on the CPU, on any CUDA GPU (the
fatbin built for it: nvcc -fatbin -arch=sm_XX eagle.cu). Driver API through ctypes; one line a kernel.

  python3 eagle_kernels_test.py EAGLE.fatbin
"""

import ctypes
import math
import sys

import torch
import torch.nn.functional as F

cu = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")


def ok(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: CUresult {r}")


ok(cu.cuInit(0), "cuInit")
dev = ctypes.c_int()
ok(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
ok(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "retain")
ok(cu.cuCtxSetCurrent(ctx), "current")
mod = ctypes.c_void_p()
img = open(sys.argv[1], "rb").read()
ok(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(img)), "load")


def fn(name):
    f = ctypes.c_void_p()
    ok(cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), name)
    return f


def dev_t(t: torch.Tensor):
    t = t.contiguous()
    p = ctypes.c_uint64()
    n = t.numel() * t.element_size()
    ok(cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(max(n, 4))), "alloc")
    ok(cu.cuMemcpyHtoD_v2(p, ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(n)), "h2d")
    return p


def back(p, like: torch.Tensor):
    out = torch.empty_like(like).contiguous()
    ok(cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), p, ctypes.c_size_t(out.numel() * out.element_size())), "d2h")
    return out


def launch(f, grid, block, args):
    vals = []
    for kind, v in args:
        vals.append({"p": ctypes.c_uint64, "ll": ctypes.c_longlong, "i": ctypes.c_int, "f": ctypes.c_float}[kind](v))
    arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
    g = grid + (1,) * (3 - len(grid))
    ok(cu.cuLaunchKernel(f, g[0], g[1], g[2], block, 1, 1, 0, None, arr, None), "launch")
    ok(cu.cuCtxSynchronize(), "sync")


r = lambda x: x.to(torch.bfloat16).float()
gen = torch.Generator().manual_seed(0)
res = {}

# rmsnorm: rows with a stride, into the second half of a wider row
R, D = 7, 5120
x = (torch.randn(R, 2 * D, generator=gen) * 3).to(torch.bfloat16)
w = (torch.rand(D, generator=gen) + 0.5).to(torch.bfloat16)
o = torch.zeros(R, 2 * D, dtype=torch.bfloat16)
px, pw, po = dev_t(x), dev_t(w), dev_t(o)
launch(fn("tf_eagle_rmsnorm_kernel"), (R,), 512, [("p", px.value), ("ll", 2 * D), ("p", pw.value), ("p", po.value + D * 2), ("ll", 2 * D), ("i", D), ("f", 1e-6)])
got = back(po, o)[:, D:].float()
xf = x[:, :D].float()
want = r(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())
res["rmsnorm"] = float((got - want).abs().max() / want.abs().max())

# rope: 32 heads x 128, the first 64 rotated, positions up to 300K
H, hd, rd = 32, 128, 64
q = torch.randn(R, H * hd, generator=gen).to(torch.bfloat16)
pos = torch.tensor([0, 1, 5, 127, 128, 4096, 300001], dtype=torch.long)
pq, pp = dev_t(q), dev_t(pos)
launch(fn("tf_eagle_rope_kernel"), (R,), 256, [("p", pq.value), ("ll", H * hd), ("i", H), ("i", hd), ("i", rd), ("p", pp.value), ("f", 10000.0)])
got = back(pq, q).float().view(R, H, hd)
half = rd // 2
f = 10000.0 ** (-torch.arange(half, dtype=torch.float32) / half)
a = pos.float()[:, None] * f[None]
c, s = a.cos()[:, None], a.sin()[:, None]
qf = q.float().view(R, H, hd)
want = torch.cat([qf[..., :half] * c - qf[..., half:rd] * s, qf[..., :half] * s + qf[..., half:rd] * c, qf[..., rd:]], -1)
res["rope"] = float((got - r(want)).abs().max())

# attention: 3 streams' rings of 144 rows (+1 sink), windows clipped by lo and by 128
ring, W, slots = 144, 128, 3
K = torch.randn(slots * ring + 1, hd, generator=gen).to(torch.bfloat16)
V = torch.randn(slots * ring + 1, hd, generator=gen).to(torch.bfloat16)
rows = [(0, 10, 0), (1, 200, 50), (2, 140, 0), (1, 201, 50), (0, 0, 0)]  # (slot, position, lo)
N = len(rows)
qa = torch.randn(N, H * hd, generator=gen).to(torch.bfloat16)
base = torch.tensor([sl * ring for sl, _, _ in rows])
pz = torch.tensor([p for _, p, _ in rows])
lo = torch.tensor([l for _, _, l in rows])
out = torch.zeros(N, H * hd, dtype=torch.bfloat16)
args = [dev_t(t) for t in (qa, K, V, base, pz, lo, out)]
launch(fn("tf_eagle_attn_kernel"), (N, H), 128, [("p", args[0].value), ("ll", H * hd), ("p", args[1].value), ("p", args[2].value), ("i", 1), ("p", args[3].value), ("p", args[4].value), ("p", args[5].value), ("i", ring), ("i", W), ("i", H), ("f", hd ** -0.5), ("p", args[6].value), ("ll", H * hd)])
got = back(args[6], out).float()
err = 0.0
for i, (sl, p, l) in enumerate(rows):
    idx = torch.tensor([sl * ring + t % ring for t in range(max(l, p - W), p + 1)])
    sc = (qa[i].float().view(H, hd) @ K[idx].float().T) * hd ** -0.5
    want = sc.softmax(-1) @ V[idx].float()
    err = max(err, float((got[i].view(H, hd) - r(want)).abs().max()))
res["attn"] = err
ring2, W2 = 527, 511
K2 = torch.randn(2 * ring2 + 1, hd, generator=gen).to(torch.bfloat16)
V2 = torch.randn(2 * ring2 + 1, hd, generator=gen).to(torch.bfloat16)
rows2 = [(0, 600, 0), (1, 700, 400), (0, 511, 0), (1, 5, 0)]
N2 = len(rows2)
qb = torch.randn(N2, H * hd, generator=gen).to(torch.bfloat16)
out2 = torch.zeros(N2, H * hd, dtype=torch.bfloat16)
args2 = [dev_t(t) for t in (qb, K2, V2, torch.tensor([sl * ring2 for sl, _, _ in rows2]), torch.tensor([p for _, p, _ in rows2]),
                            torch.tensor([l for _, _, l in rows2]), out2)]
launch(fn("tf_eagle_attn_kernel"), (N2, H), 128, [("p", args2[0].value), ("ll", H * hd), ("p", args2[1].value), ("p", args2[2].value), ("i", 1), ("p", args2[3].value), ("p", args2[4].value), ("p", args2[5].value), ("i", ring2), ("i", W2), ("i", H), ("f", hd ** -0.5), ("p", args2[6].value), ("ll", H * hd)])
got = back(args2[6], out2).float()
err = 0.0
for i, (sl, p, l) in enumerate(rows2):
    idx = torch.tensor([sl * ring2 + t % ring2 for t in range(max(l, p - W2), p + 1)])
    sc = (qb[i].float().view(H, hd) @ K2[idx].float().T) * hd ** -0.5
    want = sc.softmax(-1) @ V2[idx].float()
    err = max(err, float((got[i].view(H, hd) - r(want)).abs().max()))
res["attn511"] = err

# silu * up
I = 4096
gu = torch.randn(N, 2 * I, generator=gen).to(torch.bfloat16)
y = torch.zeros(N, I, dtype=torch.bfloat16)
pg, py = dev_t(gu), dev_t(y)
launch(fn("tf_eagle_silu_mul_kernel"), (min((N * I + 255) // 256, 65535),), 256, [("p", pg.value), ("i", I), ("p", py.value), ("ll", N * I)])
got = back(py, y).float()
want = r(r(F.silu(gu[:, :I].float())) * gu[:, I:].float())
res["silu_mul"] = float((got - want).abs().max())

# top-1 and its confidence
Vv = 129280
lg = torch.randn(N, Vv, generator=gen) * 4
lg[2, 777] = 40.0
lg[3, 5] = lg[3].max() + 1e-3
ids = torch.zeros(N, dtype=torch.long)
conf = torch.zeros(N)
pl, pi, pc = dev_t(lg), dev_t(ids), dev_t(conf)
launch(fn("tf_eagle_top1_kernel"), (N,), 1024, [("p", pl.value), ("ll", Vv), ("i", Vv), ("p", 0), ("ll", 0), ("p", pi.value), ("p", pc.value), ("f", 1.0), ("f", 0.0)])
gi, gc = back(pi, ids), back(pc, conf)
p1 = lg.softmax(-1).max(-1)
pr = p1.values.clamp(1e-7, 1 - 1e-7)
res["top1_ids_equal"] = bool((gi == p1.indices).all())
res["top1_conf"] = float((gc - torch.log(pr / (1 - pr))).abs().max())
parts = 64
pm, pz, pq = torch.zeros(N, parts), torch.zeros(N, parts), torch.zeros(N, parts, dtype=torch.int32)
ppm, ppz, ppq = dev_t(pm), dev_t(pz), dev_t(pq)
ids3, conf3 = torch.zeros(N, dtype=torch.long), torch.zeros(N)
pi3, pc3 = dev_t(ids3), dev_t(conf3)
launch(fn("tf_eagle_top1_part_kernel"), (N, parts), 256, [("p", pl.value), ("ll", Vv), ("i", Vv), ("p", ppm.value), ("p", ppz.value), ("p", ppq.value)])
launch(fn("tf_eagle_top1_join_kernel"), (N,), 32, [("p", ppm.value), ("p", ppz.value), ("p", ppq.value), ("i", parts), ("p", pi3.value), ("p", pc3.value), ("f", 1.0), ("f", 0.0)])
res["top1_split_ids_equal"] = bool((back(pi3, ids3) == p1.indices).all())
res["top1_split_conf"] = float((back(pc3, conf3) - torch.log(pr / (1 - pr))).abs().max())

# two_stage: top-k candidates (ties at the k-th value by the lower id), their exact logits, top-1 through the map
for k in (128, 512):
    lk = (torch.randn(N, Vv, generator=gen) * 3).to(torch.bfloat16).float()   # bf16 values: many ties
    lk[1, :] = 0.5                                                            # a row of one value
    cand = torch.full((N, k), -1, dtype=torch.long)
    pk, pcand = dev_t(lk), dev_t(cand)
    launch(fn("tf_eagle_topk_kernel"), (N,), 1024, [("p", pk.value), ("ll", Vv), ("i", Vv), ("i", k), ("p", pcand.value), ("ll", k)])
    got = back(pcand, cand).sort(-1).values
    order = torch.sort(lk, stable=True, dim=-1, descending=True).indices[:, :k]   # stable: equal values by lower id
    res[f"topk{k}_sets_equal"] = bool((got == order.sort(-1).values).all())
Dk, k = 5120, 128
Wt = (torch.randn(Vv, Dk, generator=gen) * 0.02).to(torch.bfloat16)
xo = torch.randn(N, Dk, generator=gen).to(torch.bfloat16)
cand = torch.randint(0, Vv, (N, k), generator=gen)
ex = torch.zeros(N, k)
pw, px, pcd, pex = dev_t(Wt), dev_t(xo), dev_t(cand), dev_t(ex)
f = fn("tf_eagle_rescore_kernel")
vals = [ctypes.c_uint64(pw.value), ctypes.c_int(Dk), ctypes.c_uint64(px.value), ctypes.c_longlong(Dk), ctypes.c_uint64(pcd.value),
        ctypes.c_longlong(k), ctypes.c_int(k), ctypes.c_uint64(pex.value), ctypes.c_longlong(k)]
arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
ok(cu.cuLaunchKernel(f, N, (k + 7) // 8, 1, 256, 1, 1, Dk * 4, None, arr, None), "launch")
ok(cu.cuCtxSynchronize(), "sync")
gx = back(pex, ex)
want = torch.einsum("nkd,nd->nk", Wt.float()[cand], xo.float())
res["rescore"] = float((gx - want).abs().max() / want.abs().max())
ids2, conf2 = torch.zeros(N, dtype=torch.long), torch.zeros(N)
pi2, pc2 = dev_t(ids2), dev_t(conf2)
launch(fn("tf_eagle_top1_kernel"), (N,), 1024, [("p", pex.value), ("ll", k), ("i", k), ("p", pcd.value), ("ll", k), ("p", pi2.value), ("p", pc2.value), ("f", 1.0), ("f", 0.0)])
res["top1_map_ids_equal"] = bool((back(pi2, ids2) == cand.gather(1, gx.argmax(-1, keepdim=True))[:, 0]).all())



def fp8_rows(Wb: torch.Tensor):
    """The converter's rule: s = amax(|W|) / 448 a row (1 where 0), q = e4m3 RNE of W / s."""

    wf = Wb.float()
    sc = wf.abs().amax(1) / 448.0
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    return (wf / sc[:, None]).to(torch.float8_e4m3fn), sc


gw = 0.0
for (n_, k_) in ((5120, 4096), (4096, 5120), (512, 15360), (129280, 256), (256, 5120), (136, 10240)):
    Wb = (torch.randn(n_, k_, generator=gen) * 0.02).to(torch.bfloat16)
    Wb[3] = 0
    q8, s8 = fp8_rows(Wb)
    for rows_ in (1, 4, 16):
        xg = torch.randn(rows_, k_ + 8, generator=gen).to(torch.bfloat16)        # rows k + 8 apart
        for f8 in (False, True):
            wd = q8.float() * s8[:, None] if f8 else Wb.float()
            want = xg[:, :k_].float() @ wd.T
            base_ = torch.randn(rows_, n_ + 8, generator=gen).to(torch.bfloat16)
            for mode in (0, 1, 2):
                o_ = torch.zeros(rows_, n_ + 8, dtype=torch.float32) if mode == 1 else base_.clone()
                pw_ = dev_t(q8.view(torch.uint8) if f8 else Wb)
                ps_ = dev_t(s8)
                px_, po_ = dev_t(xg), dev_t(o_)
                eb = 1 if f8 else 2
                G = min(32, k_ * eb // 16)
                per = 32 // G
                grid = (n_ + 8 * per - 1) // (8 * per)
                if f8:
                    launch(fn("tf_eagle_gemv_fp8_kernel"), (grid,), 256, [("p", pw_.value), ("p", ps_.value), ("i", n_), ("i", k_), ("p", px_.value), ("ll", k_ + 8), ("i", rows_), ("p", po_.value), ("ll", n_ + 8), ("i", mode), ("i", G)])
                else:
                    launch(fn("tf_eagle_gemv_bf16_kernel"), (grid,), 256, [("p", pw_.value), ("i", n_), ("i", k_), ("p", px_.value), ("ll", k_ + 8), ("i", rows_), ("p", po_.value), ("ll", n_ + 8), ("i", mode), ("i", G)])
                got = back(po_, o_).float()
                w_ = want if mode != 2 else base_[:, :n_].float() + want
                if mode == 1:
                    e_ = float((got[:, :n_] - w_).abs().max() / w_.abs().max())
                else:
                    e_ = float((got[:, :n_] - r(w_)).abs().max() / w_.abs().max())
                e_ = max(e_, float((got[:, n_:] - o_[:, n_:].float()).abs().max()))       # the gap columns stay
                gw = max(gw, e_)
                for t_ in (pw_, ps_, px_, po_):
                    cu.cuMemFree_v2(t_)
res["gemv"] = gw

print(res)
bad = res["rmsnorm"] > 1e-2 or res["rope"] > 0.05 or res["attn"] > 0.02 or res["silu_mul"] > 0.05 or \
    not res["top1_ids_equal"] or res["top1_conf"] > 1e-3 or not res["topk128_sets_equal"] or     not res["topk512_sets_equal"] or res["rescore"] > 1e-4 or not res["top1_map_ids_equal"] or \
    res["attn511"] > 0.02 or not res["top1_split_ids_equal"] or res["top1_split_conf"] > 1e-3 or res["gemv"] > 1e-2
print("FAIL" if bad else "PASS")
