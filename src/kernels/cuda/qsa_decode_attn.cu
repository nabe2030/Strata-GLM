// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__device__ __forceinline__ void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(base);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = __half2float(__ushort_as_half(blk->d));
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE, bool LANE_CELL = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    if constexpr (LANE_CELL) {
        // S25 (STRATA_ATTN_LANECELL=1): thread t scores cell t % 64 for heads 3 (t / 64) .. +2 by itself - the same
        // 8-dimension partial per former lane (the identical expression), then the former warp_sum's butterfly as
        // lane 0 ran it (a[l] += a[l + o] for o = 16 .. 1), so every score is bit-identical without 12 shuffle
        // reductions per cell
        constexpr int HPT = G * CHUNK / THREADS;   // heads per thread (3)
        const int c = t % CHUNK, h0 = (t / CHUNK) * HPT;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int j = 0; j < HPT; ++j) sp[h0 + j][c] = -FLT_MAX;
        } else {
            float a[HPT][32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], l * 8, k8);
#pragma unroll
                for (int j = 0; j < HPT; ++j) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h0 + j][l * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h0 + j][l * 8 + 4]);
                    float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                    a[j][l] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < HPT; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int l = 0; l < o; ++l) a[j][l] = a[j][l] + a[j][l + o];
                sp[h0 + j][c] = a[j][0] * scale;
            }
        }
    } else
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// PR #540 (sskver): the same attention with fewer shuffles and less shared-memory traffic, bit for bit the kernel
// above (same operands, same order); +7% prompt speed on a V100, where this kernel reads every prompt chunk.  It
// uses more registers (80 vs 38 on sm_70), so on CUDA it runs on cards below sm_75 only, and exists only in the
// experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON).  HIP: built for wave32 cards (not the gfx906 wave64 backend) and
// run on gfx103x (RDNA2), with 8 cells per step and the butterfly's lane exchanges in registers (DPP / permlanex16)
// instead of ds_bpermute: there the score loop is bound by the LDS pipe (the q reads and ds_bpermute share it), not by
// occupancy - 8 cells per step at 7 waves/SIMD beat 1 cell at 9.  Every other card keeps exactly the kernel above.
#if defined(STRATA_EXPERIMENTAL_SM60) || (defined(__HIPCC__) && !defined(STRATA_HIP_GFX906))
#define STRATA_ATTN_PRE75_BUILT 1
#endif
#if defined(STRATA_ATTN_PRE75_BUILT)
// A butterfly step's lane exchange, `M` = 1, 2, 4, 8 or 16.  DPP (HIP, wave32): in registers - DPP row_xmask within a
// 16-lane row, v_permlanex16 across the two rows - instead of ds_bpermute; the same value moves, so every sum is
// unchanged bit for bit.
template <bool DPP, int M> __device__ __forceinline__ float shfl_x(float v) {
#if defined(__HIPCC__)
    if constexpr (DPP) {
        const int x = __float_as_int(v);
        if constexpr (M == 16) return __int_as_float(__builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
        else return __int_as_float(__builtin_amdgcn_mov_dpp(x, 0x160 | M, 0xF, 0xF, false));   // row_xmask:M
    }
#endif
    return __shfl_xor_sync(0xffffffffu, v, M);
}
// warp_sum / warp_max with the same pairing order (xor 16, 8, 4, 2, 1)
template <bool DPP> __device__ __forceinline__ float warp_sum_x(float v) {
    v += shfl_x<DPP, 16>(v); v += shfl_x<DPP, 8>(v); v += shfl_x<DPP, 4>(v); v += shfl_x<DPP, 2>(v); v += shfl_x<DPP, 1>(v);
    return v;
}
template <bool DPP> __device__ __forceinline__ float warp_max_x(float v) {
    v = fmaxf(v, shfl_x<DPP, 16>(v)); v = fmaxf(v, shfl_x<DPP, 8>(v)); v = fmaxf(v, shfl_x<DPP, 4>(v));
    v = fmaxf(v, shfl_x<DPP, 2>(v)); v = fmaxf(v, shfl_x<DPP, 1>(v));
    return v;
}
// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5.  Lane l ends holding head head_of_lane(l); lanes l and l^1 agree.
template <bool DPP> __device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + shfl_x<DPP, 16>(hi ? part[i] : part[i + 8]);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + shfl_x<DPP, 8>(hi ? r8[i] : r8[i + 4]);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + shfl_x<DPP, 4>(hi ? r4[i] : r4[i + 2]);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + shfl_x<DPP, 2>(hi2 ? r2[0] : r2[1]);
    return r1 + shfl_x<DPP, 1>(r1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own thread index), per KV format.
template <int KV_MODE>
__device__ __forceinline__ float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + d]));
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = __half2float(__ushort_as_half(blk->d));
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

template <int KV_MODE, int NC_ = 2, bool DPP_ = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads (dead once the scores are done)
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's 12 as three float4.  They are first written after the
    // __syncthreads that ends the score phase, the last reader of sq, so they share its storage (64 x 12 of its 256 x 12
    // floats) and the block's shared memory stays what it was before the cell-major layout.
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per step
    // (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts the traffic
    // that bounds this phase (24 x 16 B per lane per cell, against the FMAs) by NC; every dot product is the same
    // expression as for a single cell, so the scores are unchanged.
    constexpr int NC = NC_;   // CUDA 2 (4 measured slower on a V100: the registers cost occupancy); HIP gfx103x 8
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12<DPP_>(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12<DPP_>(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x<DPP_>(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x<DPP_>(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    // One cell's 12 weights (three broadcast float4 loads) into the 12 accumulators; the cells are folded in
    // ascending order, so the sums are the plain loop's, bit for bit.
    const auto fold = [&](int c, float v) {
        const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    };
    int c = 0;
    // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
    // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
#endif  // STRATA_ATTN_PRE75_BUILT

__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// ================================ glm5-next: MLA (absorbed) sparse attention, decode ==============================
// The same split-K flash-decode as attn_chunk_kernel (chunks of 64 selected cells -> per-chunk max / exp-sum / acc ->
// merge), for GLM-5.3-Flash's DSA: ONE 512-value latent per cell serves as K and V for all 64 query heads (MQA 64:1,
// llama.cpp b11461 glm5-next.cpp:917-950).  64 heads x 512 floats of q do not fit in shared memory, so a block takes
// a group of MGH heads (grid.y = head group); the latent rows are re-read per group from L2.  Pool: fp16 rows of 512
// at (page * 1 + 0) * page_size + slot (n_kv_heads = 1).  `scale` is the caller's (GLM: 1/sqrt(256), not 1/sqrt(512)).
constexpr int MHD = 512;            // latent dim
constexpr int MGH = 16;             // query heads per block
constexpr int MPL = MHD / 32;       // latent values per lane in the score pass (16)

// The latent row's formats: fp16 (512 x 2 bytes) or q4_0 (16 ggml blocks of 32: an fp16 scale + 16 bytes of nibbles,
// value i < 16 in the low nibble of qs[i], i >= 16 in the high nibble of qs[i - 16]; 288 bytes), 2026-10-08.
constexpr int MQ4_ROW = (MHD / QK4_0) * (int) sizeof(block_q4_0);
template <bool Q4>
__device__ __forceinline__ void mla_load8(const void* pool, long long row, int d0, float* out) {
    if constexpr (!Q4) {
        const uint4 raw = *reinterpret_cast<const uint4*>((const uint16_t*) pool + row * MHD + d0);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h2[j]); out[2 * j] = f.x; out[2 * j + 1] = f.y; }
    } else {
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((const uint8_t*) pool + row * MQ4_ROW) + d0 / QK4_0;
        const float d = __half2float(__ushort_as_half(blk->d));
        const int rem = d0 % QK4_0;                       // 0, 8, 16 or 24
        const uint8_t* bytes = blk->qs + (rem & 15);
        if (rem < 16) {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}
template <bool Q4>
__device__ __forceinline__ float mla_load1(const void* pool, long long row, int i) {
    if constexpr (!Q4) {
        return __half2float(__ushort_as_half(((const uint16_t*) pool)[row * MHD + i]));
    } else {
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((const uint8_t*) pool + row * MQ4_ROW) + i / QK4_0;
        const int r = i % QK4_0;
        const int nib = r < 16 ? (blk->qs[r] & 0x0F) : (blk->qs[r - 16] >> 4);
        return (float) (nib - 8) * __half2float(__ushort_as_half(blk->d));
    }
}

template <bool Q4>
__global__ void __launch_bounds__(THREADS) mla_chunk_kernel(const float* __restrict__ q, const void* __restrict__ pool,
                                                            const int32_t* __restrict__ page_table,
                                                            const int32_t* __restrict__ ids,
                                                            const int32_t* __restrict__ step, int n_head, int page_size,
                                                            float scale, float* __restrict__ part_acc,
                                                            float* __restrict__ part_m, float* __restrict__ part_l,
                                                            int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) n_head * MHD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[MGH][MHD];  // 32 KB: this group's query heads
    __shared__ float sp[MGH][CHUNK];              // scores, then probabilities
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, hg = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int h0 = hg * MGH;
    if (n_here <= 0) {
        if (t < MGH) { part_m[(size_t) chunk * n_head + h0 + t] = -FLT_MAX; part_l[(size_t) chunk * n_head + h0 + t] = 0.0f; }
        return;
    }
    for (int i = t; i < MGH * MHD; i += THREADS) sq[i / MHD][i % MHD] = q[(size_t) h0 * MHD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) page_table[cell / page_size];
            if (page >= 0) r = page * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: a warp per cell, each lane 16 of the 512 latent values (two 8-wide fp16 loads), used for MGH heads
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < MGH) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k16[MPL];
#pragma unroll
        for (int half = 0; half < 2; ++half)   // dims lane*8 + [0,8) and 256 + lane*8 + [0,8)
            mla_load8<Q4>(pool, srow[c], half * 256 + lane * 8, k16 + half * 8);
#pragma unroll
        for (int h = 0; h < MGH; ++h) {
            float s = 0.f;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][half * 256 + lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][half * 256 + lane * 8 + 4]);
                const float* k8 = k16 + half * 8;
                s += k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                     k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            }
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    for (int h = warp; h < MGH; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[(size_t) chunk * n_head + h0 + h] = m; part_l[(size_t) chunk * n_head + h0 + h] = l; }
    }
    __syncthreads();
    // values (= the same latent): thread t owns dims t and t + 256 for the MGH heads
    float acc0[MGH], acc1[MGH];
#pragma unroll
    for (int h = 0; h < MGH; ++h) { acc0[h] = 0.0f; acc1[h] = 0.0f; }
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        const float v0 = mla_load1<Q4>(pool, srow[c], t);
        const float v1 = mla_load1<Q4>(pool, srow[c], 256 + t);
#pragma unroll
        for (int h = 0; h < MGH; ++h) { acc0[h] = fmaf(sp[h][c], v0, acc0[h]); acc1[h] = fmaf(sp[h][c], v1, acc1[h]); }
    }
#pragma unroll
    for (int h = 0; h < MGH; ++h) {
        float* a = part_acc + ((size_t) chunk * n_head + h0 + h) * MHD;
        a[t] = acc0[h];
        a[256 + t] = acc1[h];
    }
}

__global__ void __launch_bounds__(MHD) mla_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks, int n_head,
                                                        float* __restrict__ out, long long scratch_stride) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    out += (size_t) blockIdx.y * (size_t) n_head * MHD;
    const int h = blockIdx.x, d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(size_t) c * n_head + h]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float m = part_m[(size_t) c * n_head + h];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[(size_t) c * n_head + h], w, L);
        acc = fmaf(part_acc[((size_t) c * n_head + h) * MHD + d], w, acc);
    }
    out[(size_t) h * MHD + d] = L > 0.0f ? acc / L : 0.0f;
}
#if defined(STRATA_ATTN_PRE75_BUILT)
// CUDA: the current device is below sm_75; HIP: it is gfx103x (per device: a layer split can mix cards).
// STRATA_ATTN_PRE75=0 turns PR #540's kernel off (A/B); on HIP =1 also runs it on another wave32 card.
bool pre75_attn() {
    static const int env = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e == nullptr ? -1 : e[0] == '0' ? 0 : e[0] == '1' ? 1 : -1;
    }();
    static int cc[64] = {};
    int dev = 0;
    if (env == 0) return false;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
#if defined(__HIPCC__)
    if (cc[dev] == 0) {   // 2: gfx103x, 1: another card
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        cc[dev] = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 2 : 1;
    }
    return env == 1 || cc[dev] == 2;
#else
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
#endif
}
#if defined(__HIPCC__)
#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M, 8, true> : attn_chunk_kernel<M>)
#else
#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
#endif
#else
#define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
#endif

}  // namespace

uint64_t mla_decode_attn_scratch_floats(int64_t cap, int64_t n_head) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) n_head * (MHD + 2) + 64;
}

namespace {
// ---- 2026-10-10: the same chunk on the tensor cores (the prompt path's MLA sub-batches; WMMA 16x16x16, FP16 operands,
// F32 accumulators).  The 16 heads' queries and the chunk's 64 latents (q4_0 / fp16 -> fp16) in shared memory;
// S = Q . L^T (4 warps, a 16-cell tile each), the softmax as mla_chunk_kernel's (the same part_m / part_l), P (fp16)
// . L into part_acc straight from the fragments (8 warps, four 16-dim tiles each).  Not bitwise the CUDA-core kernel
// (fp16 operands, another summation order); the merge is the same.  ~86 KB of dynamic shared memory.
constexpr size_t kMlaTcSmem = (size_t) MGH * MHD * 2 + (size_t) CHUNK * MHD * 2 + (size_t) MGH * CHUNK * 4 +
                              (size_t) MGH * CHUNK * 2 + (size_t) CHUNK * 8;
template <bool Q4>
__global__ void __launch_bounds__(THREADS) mla_chunk_tc_kernel(const float* __restrict__ q, const void* __restrict__ pool,
                                                               const int32_t* __restrict__ page_table,
                                                               const int32_t* __restrict__ ids,
                                                               const int32_t* __restrict__ step, int n_head, int page_size,
                                                               float scale, float* __restrict__ part_acc,
                                                               float* __restrict__ part_m, float* __restrict__ part_l,
                                                               int n_chunks, int cap, long long scratch_stride) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sQ = reinterpret_cast<__half*>(smem);                                   // [MGH][MHD]
    __half* sL = sQ + MGH * MHD;                                                    // [CHUNK][MHD]
    float* sS = reinterpret_cast<float*>(sL + CHUNK * MHD);                         // [MGH][CHUNK]
    __half* sP = reinterpret_cast<__half*>(sS + MGH * CHUNK);                       // [MGH][CHUNK]
    long long* srow = reinterpret_cast<long long*>(sP + MGH * CHUNK);               // [CHUNK]
    q += (size_t) blockIdx.z * (size_t) n_head * MHD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, hg = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int h0 = hg * MGH;
    if (n_here <= 0) {
        if (t < MGH) { part_m[(size_t) chunk * n_head + h0 + t] = -FLT_MAX; part_l[(size_t) chunk * n_head + h0 + t] = 0.0f; }
        return;
    }
    for (int i = t; i < MGH * MHD; i += THREADS) sQ[i] = __float2half(q[(size_t) h0 * MHD + i]);
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) page_table[cell / page_size];
            if (page >= 0) r = page * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    for (int i = t; i < CHUNK * (MHD / 8); i += THREADS) {   // the latents, 8 values a step, zero for empty cells
        const int c = i / (MHD / 8), d0 = (i % (MHD / 8)) * 8;
        float v[8];
        if (srow[c] >= 0) mla_load8<Q4>(pool, srow[c], d0, v);
        else
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = 0.0f;
        __half* dst = sL + c * MHD + d0;
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = __float2half(v[k]);
    }
    __syncthreads();
    if (warp < CHUNK / 16) {   // S[:, 16 warp .. 16 warp + 16) = Q . L^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
        for (int k = 0; k < MHD; k += 16) {
            wmma::load_matrix_sync(fa, sQ + k, MHD);
            wmma::load_matrix_sync(fb, sL + (warp * 16) * MHD + k, MHD);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(sS + warp * 16, acc, CHUNK, wmma::mem_row_major);
    }
    __syncthreads();
    for (int h = warp; h < MGH; h += WARPS) {   // mla_chunk_kernel's softmax over the chunk's cells
        const bool va = lane < n_here && srow[lane] >= 0, vb = lane + 32 < n_here && srow[lane + 32] >= 0;
        const float a = va ? sS[h * CHUNK + lane] * scale : -FLT_MAX, b = vb ? sS[h * CHUNK + lane + 32] * scale : -FLT_MAX;
        const float m = warp_max(fmaxf(a, b));
        const float ea = va ? __expf(a - m) : 0.0f;
        const float eb = vb ? __expf(b - m) : 0.0f;
        sP[h * CHUNK + lane] = __float2half(ea);
        sP[h * CHUNK + lane + 32] = __float2half(eb);
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[(size_t) chunk * n_head + h0 + h] = m; part_l[(size_t) chunk * n_head + h0 + h] = l; }
    }
    __syncthreads();
    // part_acc[h][d] = sum_c P[h][c] L[c][d]: 32 tiles of 16 dims, four a warp
    float* acc_out = part_acc + ((size_t) chunk * n_head + h0) * MHD;
    for (int nt = warp; nt < MHD / 16; nt += WARPS) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
#pragma unroll
        for (int k = 0; k < CHUNK; k += 16) {
            wmma::load_matrix_sync(fa, sP + k, CHUNK);
            wmma::load_matrix_sync(fb, sL + k * MHD + nt * 16, MHD);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(acc_out + nt * 16, acc, MHD, wmma::mem_row_major);
    }
}

// The same with 32 heads a block (2026-10-10): the chunk's latents decoded once for twice the heads (the 16-head
// kernel decoded them four times a query and chunk).  S and P overlay the queries' room once Q . L^T is done; 98,816
// bytes of dynamic shared memory (sm_120 allows 101,376).  8 warps: Q . L^T one 16x16 tile each, the softmax 4 heads
// each, P . L 8 tiles each.
constexpr int MGH2 = 32;
constexpr size_t kMlaTc32Smem = (size_t) MGH2 * MHD * 2 + (size_t) CHUNK * MHD * 2 + (size_t) CHUNK * 8;
template <bool Q4>
__global__ void __launch_bounds__(THREADS) mla_chunk_tc32_kernel(const float* __restrict__ q, const void* __restrict__ pool,
                                                                 const int32_t* __restrict__ page_table,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ step, int n_head, int page_size,
                                                                 float scale, float* __restrict__ part_acc,
                                                                 float* __restrict__ part_m, float* __restrict__ part_l,
                                                                 int n_chunks, int cap, long long scratch_stride) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sQ = reinterpret_cast<__half*>(smem);                                   // [MGH2][MHD]
    __half* sL = sQ + MGH2 * MHD;                                                   // [CHUNK][MHD]
    long long* srow = reinterpret_cast<long long*>(sL + CHUNK * MHD);               // [CHUNK]
    float* sS = reinterpret_cast<float*>(smem);                                     // [MGH2][CHUNK] over sQ (after Q . L^T)
    __half* sP = reinterpret_cast<__half*>(sS + MGH2 * CHUNK);                      // [MGH2][CHUNK]
    q += (size_t) blockIdx.z * (size_t) n_head * MHD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, hg = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int h0 = hg * MGH2;
    if (n_here <= 0) {
        if (t < MGH2) { part_m[(size_t) chunk * n_head + h0 + t] = -FLT_MAX; part_l[(size_t) chunk * n_head + h0 + t] = 0.0f; }
        return;
    }
    for (int i = t; i < MGH2 * MHD; i += THREADS) sQ[i] = __float2half(q[(size_t) h0 * MHD + i]);
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) page_table[cell / page_size];
            if (page >= 0) r = page * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    for (int i = t; i < CHUNK * (MHD / 8); i += THREADS) {
        const int c = i / (MHD / 8), d0 = (i % (MHD / 8)) * 8;
        float v[8];
        if (srow[c] >= 0) mla_load8<Q4>(pool, srow[c], d0, v);
        else
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = 0.0f;
        __half* dst = sL + c * MHD + d0;
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = __float2half(v[k]);
    }
    __syncthreads();
    // S tile (hm, cn) = Q[16 hm ..][:] . L[16 cn ..][:]^T, one tile a warp; stored once every warp is done with sQ
    const int hm = warp >> 2, cn = warp & 3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
    wmma::fill_fragment(sacc, 0.0f);
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
        for (int k = 0; k < MHD; k += 16) {
            wmma::load_matrix_sync(fa, sQ + (hm * 16) * MHD + k, MHD);
            wmma::load_matrix_sync(fb, sL + (cn * 16) * MHD + k, MHD);
            wmma::mma_sync(sacc, fa, fb, sacc);
        }
    }
    __syncthreads();
    wmma::store_matrix_sync(sS + (hm * 16) * CHUNK + cn * 16, sacc, CHUNK, wmma::mem_row_major);
    __syncthreads();
    for (int h = warp; h < MGH2; h += WARPS) {
        const bool va = lane < n_here && srow[lane] >= 0, vb = lane + 32 < n_here && srow[lane + 32] >= 0;
        const float a = va ? sS[h * CHUNK + lane] * scale : -FLT_MAX, b = vb ? sS[h * CHUNK + lane + 32] * scale : -FLT_MAX;
        const float m = warp_max(fmaxf(a, b));
        const float ea = va ? __expf(a - m) : 0.0f;
        const float eb = vb ? __expf(b - m) : 0.0f;
        sP[h * CHUNK + lane] = __float2half(ea);
        sP[h * CHUNK + lane + 32] = __float2half(eb);
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[(size_t) chunk * n_head + h0 + h] = m; part_l[(size_t) chunk * n_head + h0 + h] = l; }
    }
    __syncthreads();
    float* acc_out = part_acc + ((size_t) chunk * n_head + h0) * MHD;
    for (int tile = warp; tile < 2 * (MHD / 16); tile += WARPS) {   // (head half, 16-dim tile)
        const int th = tile / (MHD / 16), nt = tile % (MHD / 16);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
#pragma unroll
        for (int k = 0; k < CHUNK; k += 16) {
            wmma::load_matrix_sync(fa, sP + (th * 16) * CHUNK + k, CHUNK);
            wmma::load_matrix_sync(fb, sL + k * MHD + nt * 16, MHD);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(acc_out + (size_t) (th * 16) * MHD + nt * 16, acc, MHD, wmma::mem_row_major);
    }
}

template <bool Q4>
void mla_decode_attn_any(const float* q_abs, const void* latent_pool, const int32_t* page_table, int page_size,
                         const int32_t* ids, const int32_t* steps, int64_t cap, int n_head, float scale,
                         float* scratch, float* out, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (n_head % MGH != 0 || cap <= 0 || !scratch || !ids || !steps || !page_table || !latent_pool || n_q > 65535) {
        std::fprintf(stderr, "mla_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) mla_decode_attn_scratch_floats(cap, n_head);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * n_head * MHD;
    float* part_l = part_m + (size_t) n_chunks * n_head;
    cudaStream_t st = (cudaStream_t) stream;
    // the tensor-core chunk for the prompt path's sub-batches (>= 16 queries; the decode's windows, captured in graphs,
    // keep the CUDA-core kernel).  STRATA_MLA_TC=0: the CUDA-core kernel everywhere
    static const bool tc_on = [] { const char* v = std::getenv("STRATA_MLA_TC"); return !(v && v[0] == '0'); }();
    // per device (step2b): the shared-memory opt-in belongs to the device, and a layer split runs this on several
    static bool tc_ready_dev[64] = {};
    static bool tc_tried_dev[64] = {};
    static int tc32_ready_dev[64];
    static const bool tc32_init = [] { for (int& v : tc32_ready_dev) v = -1; return true; }();
    (void) tc32_init;
    int tdev = 0;
    cudaGetDevice(&tdev);
    if (tdev < 0 || tdev >= 64) tdev = 0;
    bool& tc_ready = tc_ready_dev[tdev];
    if (tc_on && n_q >= 16 && !tc_tried_dev[tdev]) {
        tc_tried_dev[tdev] = true;
        tc_ready = cudaFuncSetAttribute(mla_chunk_tc_kernel<Q4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kMlaTcSmem) == cudaSuccess;
        cudaGetLastError();
    }
    // the 32-head kernel where its shared memory fits (STRATA_MLA_TC32=0: the 16-head one)
    static const bool tc32_on = [] { const char* v = std::getenv("STRATA_MLA_TC32"); return !(v && v[0] == '0'); }();
    int& tc32_ready = tc32_ready_dev[tdev];
    if (tc32_ready < 0 && tc_on && n_q >= 16) {
        tc32_ready = tc32_on && n_head % MGH2 == 0 &&
                     cudaFuncSetAttribute(mla_chunk_tc32_kernel<Q4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kMlaTc32Smem) == cudaSuccess;
        cudaGetLastError();
    }
    // STRATA_MLA_TC_CHECK=1 (a test): the first calls run the CUDA-core kernel too and print how far the outputs are apart
    static const bool tc_check = [] { const char* v = std::getenv("STRATA_MLA_TC_CHECK"); return v && v[0] == '1'; }();
    static int tc_checks = 0;
    if (tc_check && tc_on && n_q >= 16 && tc_ready && tc_checks < 6) {
        ++tc_checks;
        const size_t on = (size_t) n_q * n_head * MHD;
        float* ref = nullptr;
        cudaMalloc(&ref, on * 4);
        mla_chunk_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH), (unsigned) n_q), THREADS, 0, st>>>(
            q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
            (int) cap, stride);
        mla_merge_kernel<<<dim3((unsigned) n_head, (unsigned) n_q), MHD, 0, st>>>(part_acc, part_m, part_l, n_chunks, n_head, ref, stride);
        if (tc32_ready == 1)
            mla_chunk_tc32_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH2), (unsigned) n_q), THREADS, kMlaTc32Smem, st>>>(
                q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
                (int) cap, stride);
        else
            mla_chunk_tc_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH), (unsigned) n_q), THREADS, kMlaTcSmem, st>>>(
                q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
                (int) cap, stride);
        mla_merge_kernel<<<dim3((unsigned) n_head, (unsigned) n_q), MHD, 0, st>>>(part_acc, part_m, part_l, n_chunks, n_head, out, stride);
        std::vector<float> a(on), b(on);
        cudaMemcpyAsync(a.data(), ref, on * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(b.data(), out, on * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        cudaFree(ref);
        double d2 = 0, n2 = 0;
        for (size_t i = 0; i < on; ++i) { const double d = (double) a[i] - b[i]; d2 += d * d; n2 += (double) a[i] * a[i]; }
        std::fprintf(stderr, "mla tc check: %d queries, rel L2 %.5f\n", (int) n_q, n2 > 0 ? std::sqrt(d2 / n2) : 0.0);
        return;
    }
    if (tc_on && n_q >= 16 && tc_ready && tc32_ready == 1)
        mla_chunk_tc32_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH2), (unsigned) n_q), THREADS, kMlaTc32Smem, st>>>(
            q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
            (int) cap, stride);
    else if (tc_on && n_q >= 16 && tc_ready)
        mla_chunk_tc_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH), (unsigned) n_q), THREADS, kMlaTcSmem, st>>>(
            q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
            (int) cap, stride);
    else
        mla_chunk_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH), (unsigned) n_q), THREADS, 0, st>>>(
            q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
            (int) cap, stride);
    mla_merge_kernel<<<dim3((unsigned) n_head, (unsigned) n_q), MHD, 0, st>>>(part_acc, part_m, part_l, n_chunks, n_head,
                                                                            out, stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mla_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

void mla_decode_attn_batch(const float* q_abs, const uint16_t* latent_pool, const int32_t* page_table, int page_size,
                           const int32_t* ids, const int32_t* steps, int64_t cap, int n_head, float scale,
                           float* scratch, float* out, int64_t n_q, void* stream) {
    mla_decode_attn_any<false>(q_abs, latent_pool, page_table, page_size, ids, steps, cap, n_head, scale, scratch, out,
                               n_q, stream);
}

void mla_decode_attn_batch_q4(const float* q_abs, const uint8_t* latent_q4, const int32_t* page_table, int page_size,
                              const int32_t* ids, const int32_t* steps, int64_t cap, int n_head, float scale,
                              float* scratch, float* out, int64_t n_q, void* stream) {
    mla_decode_attn_any<true>(q_abs, latent_q4, page_table, page_size, ids, steps, cap, n_head, scale, scratch, out,
                              n_q, stream);
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
    // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
    static const bool lane_cell = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
#define STRATA_ATTN_LC(M) attn_chunk_kernel<M, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (lane_cell) {
        if (kv_mode == 3) STRATA_ATTN_LC(3);
        else if (kv_mode == 2) STRATA_ATTN_LC(2);
        else if (kv_mode == 1) STRATA_ATTN_LC(1);
        else STRATA_ATTN_LC(0);
    } else
#undef STRATA_ATTN_LC
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK(3)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK(2)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK(1)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
        STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks,
                                                                                  attn, stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv);
    cudaStream_t st = (cudaStream_t) stream;
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK(3)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK(2)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK(1)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else
        STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    attn_merge_kernel<<<(unsigned) s.n_head, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
