// The TFE draft head (zig/src/families/dsv41/eagle.zig): its row kernels and its few-row linears. Ours. Each one
// computes in fp32 and rounds to bf16 where tools/dsv41-zig/eagle/eagle_ref.py rounds (the reference of the
// engine's self-test). Raw device pointers, the caller's stream.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {

__device__ __forceinline__ float bf(const __nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float ninf() { return -__int_as_float(0x7f800000); }

__device__ __forceinline__ float block_sum(float v, float* sh) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.f;
    if (warp == 0)
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ float block_max(float v, float* sh) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : ninf();
    if (warp == 0)
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

}  // namespace

// out[r] = bf16(x[r] * rsqrt(mean(x[r]^2) + eps) * w), a row a block; x rows ldx apart, out rows ldo apart (elements).
extern "C" __global__ void tf_eagle_rmsnorm_kernel(const __nv_bfloat16* x, long long ldx, const __nv_bfloat16* w,
                                                   __nv_bfloat16* out, long long ldo, int d, float eps) {
    __shared__ float sh[32];
    const __nv_bfloat16* row = x + (long long)blockIdx.x * ldx;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = bf(row[i]);
        ss += v * v;
    }
    const float r = rsqrtf(block_sum(ss, sh) / (float)d + eps);
    __nv_bfloat16* o = out + (long long)blockIdx.x * ldo;
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = __float2bfloat16_rn(bf(row[i]) * r * bf(w[i]));
}

// RoPE in place on the first rd features of each of `heads` heads (hd apart) of row r (rows ld apart) at position
// pos[r]: the halves rotated, frequency theta^(-i / (rd / 2)). A row a block.
extern "C" __global__ void tf_eagle_rope_kernel(__nv_bfloat16* x, long long ld, int heads, int hd, int rd,
                                                const long long* pos, float theta) {
    const int half = rd / 2;
    const float p = (float)pos[blockIdx.x];
    __nv_bfloat16* row = x + (long long)blockIdx.x * ld;
    for (int t = threadIdx.x; t < heads * half; t += blockDim.x) {
        const int h = t / half, i = t % half;
        const float f = powf(theta, -((float)i / (float)half));
        const float a = p * f;
        const float c = cosf(a), s = sinf(a);
        __nv_bfloat16* q = row + h * hd;
        const float x1 = bf(q[i]), x2 = bf(q[i + half]);
        q[i] = __float2bfloat16_rn(x1 * c - x2 * s);
        q[i + half] = __float2bfloat16_rn(x1 * s + x2 * c);
    }
}

// The window attention of one row and one query head (grid rows x heads, 128 threads; head_dim 128): the keys and
// values of the row's stream ring (K, V bf16 [ring rows, kv_heads * 128], the stream's rows from base[r]) at the
// positions max(lo[r], pos[r] - window) .. pos[r] (ring row base + position % ring), fp32 scores times `scale`, softmax,
// the fp32 sum of the values, bf16 out [rows, heads * 128] (rows ldo apart). A thread a key for the scores; for the
// values each warp sums a quarter of the keys (a lane 4 features), then the 4 warps' sums are added.
extern "C" __global__ void __launch_bounds__(128) tf_eagle_attn_kernel(
    const __nv_bfloat16* q, long long ldq, const __nv_bfloat16* K, const __nv_bfloat16* V, int kv_heads,
    const long long* base, const long long* pos, const long long* lo, int ring, int window, int heads, float scale,
    __nv_bfloat16* out, long long ldo) {
    constexpr int HD = 128;
    __shared__ float qs[HD];
    __shared__ float sc[512];
    __shared__ int rw[512];
    __shared__ float sh[32];
    __shared__ float4 part[4][32];
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const long long ldk = (long long)kv_heads * HD;
    const long long P = pos[r];
    const long long first = max(lo[r], P - (long long)window);
    const int n = (int)(P - first + 1);
    if (n <= 0 || n > 512) {  // no key (a caller's fault): zeros, not garbage
        out[(long long)r * ldo + h * HD + t] = __float2bfloat16_rn(0.f);
        return;
    }
    qs[t] = bf(q[(long long)r * ldq + h * HD + t]);
    const long long b = base[r];
    for (int j = t; j < n; j += blockDim.x) rw[j] = (int)(b + (first + j) % ring);
    __syncthreads();
    for (int j = t; j < n; j += blockDim.x) {
        const uint4* k4 = reinterpret_cast<const uint4*>(K + (long long)rw[j] * ldk + kvh * HD);
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            const uint4 u = k4[c];
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(p2[e]);
                s += qs[c * 8 + e * 2] * f.x + qs[c * 8 + e * 2 + 1] * f.y;
            }
        }
        sc[j] = s * scale;
    }
    __syncthreads();
    float m = ninf();
    for (int j = t; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m, sh);
    float z = 0.f;
    for (int j = t; j < n; j += blockDim.x) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        z += e;
    }
    z = block_sum(z, sh);
    const int lane = t & 31, warp = t >> 5;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int j = warp; j < n; j += 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(V + (long long)rw[j] * ldk + kvh * HD + lane * 4);
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&u);
        const float2 f0 = __bfloat1622float2(p2[0]), f1 = __bfloat1622float2(p2[1]);
        const float w = sc[j];
        acc.x += w * f0.x;
        acc.y += w * f0.y;
        acc.z += w * f1.x;
        acc.w += w * f1.y;
    }
    part[warp][lane] = acc;
    __syncthreads();
    const float* pf = reinterpret_cast<const float*>(part);  // part[w] as floats: feature lane * 4 + c = t
    const float v = pf[t] + pf[HD + t] + pf[2 * HD + t] + pf[3 * HD + t];
    out[(long long)r * ldo + h * HD + t] = __float2bfloat16_rn(v / z);
}

// SwiGLU: out [rows, inter] = bf16(bf16(silu(g)) * u) of gu [rows, 2 * inter] ([g ; u] a row), as torch's bf16 ops.
extern "C" __global__ void tf_eagle_silu_mul_kernel(const __nv_bfloat16* gu, int inter, __nv_bfloat16* out,
                                                    long long count) {
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < count; t += (long long)gridDim.x * blockDim.x) {
        const long long r = t / inter, i = t % inter;
        const float g = bf(gu[r * 2 * inter + i]), u = bf(gu[r * 2 * inter + inter + i]);
        const float sg = bf(__float2bfloat16_rn(g / (1.f + expf(-g))));
        out[t] = __float2bfloat16_rn(sg * u);
    }
}

// A draft step's token and confidence, a row a block (1024 threads): the largest logit of [n] fp32 (rows ld apart; a
// tie goes to the lower token id), its softmax probability p over the n, and conf = a * log(p / (1 - p)) + b (the lane
// core reads sigmoid(conf) as the draft's survival). map: null (logit i is token i) or the row's token ids int64 [n]
// (rows ldm apart: a two_stage head's candidates). ids [rows] int64, conf [rows] fp32.
extern "C" __global__ void __launch_bounds__(1024) tf_eagle_top1_kernel(const float* logits, long long ld, int n,
                                                                        const long long* map, long long ldm,
                                                                        long long* ids, float* conf, float a, float b) {
    __shared__ float sv[32];
    __shared__ long long si[32];
    __shared__ float sh[32];
    constexpr long long none = 0x7fffffffffffffffLL;
    const float* row = logits + (long long)blockIdx.x * ld;
    const long long* mrow = map ? map + (long long)blockIdx.x * ldm : nullptr;
    float bv = ninf();
    long long bi = none;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        const long long id = mrow ? mrow[i] : i;
        if (v > bv || (v == bv && id < bi)) {
            bv = v;
            bi = id;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const long long oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        bv = lane < nw ? sv[lane] : ninf();
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const long long oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            sv[0] = bv;
            si[0] = bi;
        }
    }
    __syncthreads();
    const float m = sv[0];
    const long long best = si[0];
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += expf(row[i] - m);
    z = block_sum(z, sh);
    if (threadIdx.x == 0) {
        const float p = 1.f / z;
        const float q = fminf(fmaxf(p, 1e-7f), 1.f - 1e-7f);
        ids[blockIdx.x] = best == none ? 0 : best;
        conf[blockIdx.x] = a * logf(q / (1.f - q)) + b;
    }
}

// The top-1 of long rows in two kernels (the low-rank head's 129280 logits): tf_eagle_top1_part_kernel (grid rows x
// parts, 256 threads) takes each part's largest logit (a tie: the lower id) and its sum of exp(logit - that max);
// tf_eagle_top1_join_kernel (grid rows, 32 threads) joins the parts and writes what tf_eagle_top1_kernel writes. The
// part buffers: pm, pz fp32 and pi int32 [rows, parts].
extern "C" __global__ void __launch_bounds__(256) tf_eagle_top1_part_kernel(const float* logits, long long ld, int n,
                                                                             float* pm, float* pz, int* pi) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float sh[32];
    constexpr int none = 0x7fffffff;
    const int parts = gridDim.y, per = (n + parts - 1) / parts;
    const int lo = blockIdx.y * per, hi = min(n, lo + per);
    const float* row = logits + (long long)blockIdx.x * ld;
    float bv = ninf();
    int bi = none;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = row[i];
        if (v > bv) {  // (ids rise in a thread's loop: a tie keeps the lower id)
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        bv = lane < nw ? sv[lane] : ninf();
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            sv[0] = bv;
            si[0] = bi;
        }
    }
    __syncthreads();
    const float m = sv[0];
    const int best = si[0];
    float z = 0.f;
    if (best != none)
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) z += expf(row[i] - m);
    z = block_sum(z, sh);
    if (threadIdx.x == 0) {
        const long long at = (long long)blockIdx.x * parts + blockIdx.y;
        pm[at] = m;
        pz[at] = z;
        pi[at] = best;
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_eagle_top1_join_kernel(const float* pm, const float* pz,
                                                                           const int* pi, int parts, long long* ids,
                                                                           float* conf, float a, float b) {
    constexpr int none = 0x7fffffff;
    const long long at = (long long)blockIdx.x * parts;
    const int lane = threadIdx.x;
    float bv = ninf();
    int bi = none;
    for (int j = lane; j < parts; j += 32) {
        const float v = pm[at + j];
        const int id = pi[at + j];
        if (id != none && (v > bv || (v == bv && id < bi))) {
            bv = v;
            bi = id;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    float z = 0.f;
    for (int j = lane; j < parts; j += 32)
        if (pi[at + j] != none) z += pz[at + j] * expf(pm[at + j] - bv);
    for (int off = 16; off > 0; off >>= 1) z += __shfl_xor_sync(0xffffffffu, z, off);
    if (lane == 0) {
        const float p = 1.f / z;
        const float q = fminf(fmaxf(p, 1e-7f), 1.f - 1e-7f);
        ids[blockIdx.x] = bi == none ? 0 : bi;
        conf[blockIdx.x] = a * logf(q / (1.f - q)) + b;
    }
}

namespace {

// 16 weights of a 16-byte chunk as fp32: 8 bf16 (F8 false, the first 8 of w set) or 16 e4m3.
template <bool F8>
__device__ __forceinline__ void chunk_w(const uint4 u, float* w) {
    if constexpr (F8) {
        const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&u);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2(p[e], __NV_E4M3));
            const float2 f = __half22float2(h);
            w[2 * e] = f.x;
            w[2 * e + 1] = f.y;
        }
    } else {
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(p[e]);
            w[2 * e] = f.x;
            w[2 * e + 1] = f.y;
        }
    }
}

// y[r, o] = scale[o] * sum_i x[r, i] * W[o, i] for rows r < rows <= 16 (fp32 sums): a group of G lanes an output
// (G = 32, or the 16-byte chunks of a row when fewer), a lane every G-th chunk. mode 0: out bf16 = y; 1: out fp32 = y;
// 2: out bf16 = out + y (one rounding, as cuBLAS's beta 1).
template <bool F8>
__device__ __forceinline__ void gemv_rows(const uint8_t* W, const float* scale, int n, int k, const __nv_bfloat16* x,
                                          long long ldx, int rows, void* out, long long ldo, int mode, int G) {
    constexpr int E = F8 ? 16 : 8;  // the weights of a chunk
    const int chunks = k / E;
    const long long ldw = (long long)k * (F8 ? 1 : 2);
    const int lane = threadIdx.x & 31, sub = lane % G;
    const int per_warp = 32 / G;
    const long long warps = (long long)gridDim.x * (blockDim.x >> 5);
    for (long long o = ((long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * per_warp + lane / G; ;
         o += warps * per_warp) {
        const bool live = o < n;
        if (__all_sync(0xffffffffu, !live)) break;
        float acc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) acc[r] = 0.f;
        if (live) {
            const uint4* wr = reinterpret_cast<const uint4*>(W + o * ldw);
#pragma unroll 2
            for (int c = sub; c < chunks; c += G) {
                float w[16];
                chunk_w<F8>(__ldg(wr + c), w);
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    if (r < rows) {
                        const uint4* xr = reinterpret_cast<const uint4*>(x + r * ldx + (long long)c * E);
#pragma unroll
                        for (int h = 0; h < E / 8; ++h) {
                            const uint4 u = __ldg(xr + h);
                            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const float2 f = __bfloat1622float2(p[e]);
                                acc[r] = __fmaf_rn(w[h * 8 + 2 * e], f.x, acc[r]);  // (FMAs: the build has
                                acc[r] = __fmaf_rn(w[h * 8 + 2 * e + 1], f.y, acc[r]);  // --fmad=false)
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < rows)
                for (int off = G / 2; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
        }
        if (live && sub == 0) {
            const float sc = scale ? scale[o] : 1.f;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r < rows) {
                    const float y = acc[r] * sc;
                    if (mode == 1) {
                        reinterpret_cast<float*>(out)[r * ldo + o] = y;
                    } else {
                        __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out) + r * ldo + o;
                        *ob = __float2bfloat16_rn(mode == 2 ? bf(*ob) + y : y);
                    }
                }
            }
        }
    }
}

}  // namespace

// The TFE head's linears for a few rows: x bf16 [rows <= 16, k] (rows ldx apart, 16-byte aligned) @ W [n, k]^T
// (row-major, bf16 or e4m3 with an fp32 scale an output row: W[o] = scale[o] * q[o]); see gemv_rows. 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_eagle_gemv_bf16_kernel(const uint8_t* W, int n, int k,
                                                                            const __nv_bfloat16* x, long long ldx,
                                                                            int rows, void* out, long long ldo,
                                                                            int mode, int G) {
    gemv_rows<false>(W, nullptr, n, k, x, ldx, rows, out, ldo, mode, G);
}

extern "C" __global__ void __launch_bounds__(256) tf_eagle_gemv_fp8_kernel(const uint8_t* W, const float* scale, int n,
                                                                           int k, const __nv_bfloat16* x, long long ldx,
                                                                           int rows, void* out, long long ldo, int mode,
                                                                           int G) {
    gemv_rows<true>(W, scale, n, k, x, ldx, rows, out, ldo, mode, G);
}

namespace {

// An fp32's order as an unsigned key: a larger float, a larger key.
__device__ __forceinline__ unsigned int fkey(const float v) {
    const unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace

// A two_stage head's candidates, a row a block (1024 threads): the k largest of [n] fp32 logits (rows ld apart), a tie
// at the k-th value taken by the lower token ids, as int64 ids into cand [rows, k] (rows ldc apart; the order inside a
// row is not fixed). A radix select on the keys (4 passes of 8 bits), then the keys above the k-th and the first of
// the equal ones in id order.
extern "C" __global__ void __launch_bounds__(1024) tf_eagle_topk_kernel(const float* logits, long long ld, int n, int k,
                                                                        long long* cand, long long ldc) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_left, s_out, s_eq;
    __shared__ int wsum[32];
    const float* row = logits + (long long)blockIdx.x * ld;
    long long* out = cand + (long long)blockIdx.x * ldc;
    if (threadIdx.x == 0) {
        s_prefix = 0u;
        s_left = (unsigned int)k;
        s_out = 0u;
        s_eq = 0u;
    }
    unsigned int mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned int key = fkey(row[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int left = s_left, b = 255u;
            for (;; --b) {
                if (hist[b] >= left || b == 0u) break;
                left -= hist[b];
            }
            s_prefix = prefix | (b << shift);
            s_left = left;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned int kth = s_prefix;
    const unsigned int take_eq = s_left;  // the equal keys to take (the others: above kth)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (fkey(row[i]) > kth) out[atomicAdd(&s_out, 1u)] = i;
    }
    // the equal keys in id order: a chunk of blockDim ids at a time, each thread's rank among the equal ones before it
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int base = 0; base < n; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const bool eq = i < n && fkey(row[i]) == kth;
        const unsigned int bal = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) wsum[warp] = __popc(bal);
        __syncthreads();
        int before = 0;
        for (int w = 0; w < warp; ++w) before += wsum[w];
        before += __popc(bal & ((1u << lane) - 1u));
        const unsigned int taken = s_eq;
        if (eq && taken + (unsigned int)before < take_eq) out[(unsigned int)k - take_eq + taken + before] = i;
        __syncthreads();
        if (threadIdx.x == 0) {
            int all = 0;
            for (int w = 0; w < nw; ++w) all += wsum[w];
            s_eq = taken + (unsigned int)all;
        }
        __syncthreads();
        if (s_eq >= take_eq) break;
    }
}

// A two_stage head's exact logits: out[r, j] = W[cand[r, j]] . x[r] (W the target head bf16 [vocab, d], x bf16 rows
// ldx apart, fp32 sums), fp32 out [rows, k] (rows ldo apart). Grid rows x ceil(k / 8), 256 threads: a warp a candidate.
extern "C" __global__ void __launch_bounds__(256) tf_eagle_rescore_kernel(const __nv_bfloat16* W, int d,
                                                                          const __nv_bfloat16* x, long long ldx,
                                                                          const long long* cand, long long ldc, int k,
                                                                          float* out, long long ldo) {
    extern __shared__ float xs[];
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < d; i += blockDim.x) xs[i] = bf(x[(long long)r * ldx + i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int j = blockIdx.y * 8 + warp;
    if (j >= k) return;
    const __nv_bfloat16* w = W + cand[(long long)r * ldc + j] * (long long)d;
    float s = 0.f;
    for (int i = lane * 2; i < d; i += 64) {
        const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(w + i);
        s += xs[i] * __low2float(p) + xs[i + 1] * __high2float(p);
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) out[(long long)r * ldo + j] = s;
}
