// The TFE draft head (zig/src/families/dsv41/eagle.zig): its row kernels around the cuBLAS GEMMs. Ours. Each one
// computes in fp32 and rounds to bf16 where tools/dsv41-zig/eagle/eagle_ref.py rounds (the reference of the
// engine's self-test). Raw device pointers, the caller's stream.
#include <cstdint>
#include <cuda_bf16.h>

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
// the fp32 sum of the values, bf16 out [rows, heads * 128] (rows ldo apart).
extern "C" __global__ void __launch_bounds__(128) tf_eagle_attn_kernel(
    const __nv_bfloat16* q, long long ldq, const __nv_bfloat16* K, const __nv_bfloat16* V, int kv_heads,
    const long long* base, const long long* pos, const long long* lo, int ring, int window, int heads, float scale,
    __nv_bfloat16* out, long long ldo) {
    constexpr int HD = 128;
    __shared__ float qs[HD];
    __shared__ float sc[512];
    __shared__ float sh[32];
    const int r = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const long long ldk = (long long)kv_heads * HD;
    const long long P = pos[r];
    const long long first = max(lo[r], P - (long long)window);
    const int n = (int)(P - first + 1);
    if (n <= 0 || n > 512) {  // no key (a caller's fault): zeros, not garbage
        out[(long long)r * ldo + h * HD + threadIdx.x] = __float2bfloat16_rn(0.f);
        return;
    }
    qs[threadIdx.x] = bf(q[(long long)r * ldq + h * HD + threadIdx.x]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int j = warp; j < n; j += 4) {
        const long long row = base[r] + (first + j) % ring;
        const __nv_bfloat16* k = K + row * ldk + kvh * HD;
        float s = 0.f;
        for (int i = lane; i < HD; i += 32) s += qs[i] * bf(k[i]);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float m = ninf();
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m, sh);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        z += e;
    }
    z = block_sum(z, sh);
    float acc = 0.f;
    for (int j = 0; j < n; ++j) {
        const long long row = base[r] + (first + j) % ring;
        acc += sc[j] * bf(V[row * ldk + kvh * HD + threadIdx.x]);
    }
    out[(long long)r * ldo + h * HD + threadIdx.x] = __float2bfloat16_rn(acc / z);
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
