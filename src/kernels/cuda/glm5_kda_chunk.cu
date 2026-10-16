// src/kernels/cuda/glm5_kda_chunk.cu - see include/strata/kernels/glm5_kda_chunk.hpp.
//
// The prepare and recurrence kernels are ported from Infernix (https://github.com/Wallawalla47/Infernix,
// src/ops/linear_attention/kimi_delta_attention/chunked/prepare.cu and recurrence.cu, and the helpers of
// src/ops/common/{mma,math,memory,warp}.cuh), Copyright the Infernix authors, licensed under the Apache License 2.0
// (http://www.apache.org/licenses/LICENSE-2.0).  Changes for this engine (2026-10-09): q/k/v are read as F32 from the
// conv output (q and k already L2-normalised, so the prepare does not normalise them again), the decay is this engine's
// natural-log gate per key dimension and beta its sigmoid (both computed by glm5_kda_gate_beta_chunk, the window's
// expressions), the state is [key][head][value], and the output is F32.
#include "strata/kernels/glm5_kda_chunk.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int S = 128;   // KDA head / state dimension

// ================================ the sub-batch conv, gates, v copy and output norm ================================

__device__ __forceinline__ float conv_input(const float* q, const float* k, const float* v, int DI, int t, int c) {
    return c < DI ? q[(size_t) t * DI + c] : c < 2 * DI ? k[(size_t) t * DI + c - DI] : v[(size_t) t * DI + c - 2 * DI];
}

// gdn_conv_l2_multi_kernel's arithmetic per token and channel, the inputs from the three planes
__global__ void __launch_bounds__(S) conv_chunk_kernel(const float* __restrict__ hist, const float* __restrict__ q,
                                                       const float* __restrict__ k, const float* __restrict__ v,
                                                       const float* __restrict__ w, float* __restrict__ h, int DI,
                                                       int qk_heads, float eps) {
    __shared__ float part[S / 32];
    const int C = 3 * DI;
    const int t = blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;   // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : conv_input(q, k, v, DI, src - 3, c);
    }
    const float x = conv_input(q, k, v, DI, t, c);
    const float sum = win[0] * w[c * 4] + win[1] * w[c * 4 + 1] + win[2] * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

// gdn_conv_commit_kernel with n = T
__global__ void conv_commit_chunk_kernel(float* __restrict__ hist, const float* __restrict__ q,
                                         const float* __restrict__ k, const float* __restrict__ v, int DI, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 3 * DI) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = T + j;   // the last three of [hist(3) | x_0..x_{T-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : conv_input(q, k, v, DI, src - 3, c);
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

// kda_gate_beta_multi_kernel, any number of tokens
__global__ void gate_beta_chunk_kernel(const float* __restrict__ f, const float* __restrict__ dt_b,
                                       const float* __restrict__ ssm_a, float lower_bound,
                                       const float* __restrict__ beta_logit, float* __restrict__ gate,
                                       float* __restrict__ beta, int h_v) {
    const int t = blockIdx.y, head = blockIdx.x, i = threadIdx.x;   // S threads
    const size_t o = ((size_t) t * h_v + head) * S + i;
    const float val = (f[o] + dt_b[head * S + i]) * ssm_a[head];
    const float nv = -1.0f * val;
    gate[o] = lower_bound * (1.0f / (1.0f + expf(-nv)));
    if (i == 0) beta[(size_t) t * h_v + head] = 1.0f / (1.0f + expf(-beta_logit[(size_t) t * h_v + head]));
}

__global__ void v_bf16_kernel(const float* __restrict__ h, int C, int DI, __nv_bfloat16* __restrict__ out, int T) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * DI) return;
    const size_t t = i / DI, j = i % DI;
    out[i] = __float2bfloat16_rn(h[t * C + 2 * DI + j]);
}

// the window kernel's output: oc * rsqrt(mean(oc^2) + eps) * gamma[col] * sigmoid(z)
__global__ void __launch_bounds__(S) out_norm_kernel(float* __restrict__ y, const float* __restrict__ z,
                                                     const float* __restrict__ gamma, float eps, int h_v) {
    __shared__ float part[S / 32];
    const int head = blockIdx.x, t = blockIdx.y, col = threadIdx.x;
    const size_t o = ((size_t) t * h_v + head) * S + col;
    const float oc = y[o];
    float sq = oc * oc;
    for (int s = 16; s > 0; s >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, s);
    if ((col & 31) == 0) part[col >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / (float) S + eps);
    y[o] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-z[o])));
}

// ================================ Infernix's chunked KDA (Apache-2.0), adapted ================================

constexpr int kStateDim = 128;
constexpr int kChunkSize = 16;
constexpr float kLog2E = 1.4426950408889634F;

__host__ __device__ inline int chunk_count(int tokens) { return (tokens + kChunkSize - 1) / kChunkSize; }

struct alignas(256) Chunk {
    __nv_bfloat16 kd[kChunkSize * kStateDim];
    __nv_bfloat16 qd[kChunkSize * kStateDim];
    __nv_bfloat16 kr[kChunkSize * kStateDim];
    float gamma[kStateDim];
    float solve[kChunkSize * kChunkSize];
    float mqk[kChunkSize * kChunkSize];
};
static_assert(sizeof(Chunk) == 14848, "the packet layout");

__host__ __device__ constexpr int vector_index(int row, int col) { return row * kStateDim + (col ^ (row * 8)); }
__host__ __device__ constexpr int restored_index(int row, int col) {
    const int physical = (row & ~7) + (row & 3) * 2 + ((row >> 2) & 1);
    return vector_index(physical, col);
}
__host__ __device__ constexpr int square_index(int row, int col) { return row * kChunkSize + (col ^ (((row >> 1) & 3) * 4)); }

template <class V, class T> __device__ __forceinline__ V load_vec(const T* p) { return *reinterpret_cast<const V*>(p); }
template <class T, class V> __device__ __forceinline__ void store_vec(T* p, V v) { *reinterpret_cast<V*>(p) = v; }
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                                         unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                                              unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float exp2_approx_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(__float_as_uint(hi)), "r"(__float_as_uint(lo)));
    return out;
}
__device__ __forceinline__ void unpack_bf16x2_to_fp32_bits(unsigned packed, unsigned& low, unsigned& high) {
    low = packed << 16;
    high = packed & 0xffff0000U;
}
__device__ __forceinline__ void cp_async_cg16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_cg16_zfill(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" : : "r"(smem_addr(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int G> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" : : "n"(G)); }

struct Args {
    const float* h;            // [T][C]: q at [0, DI), k at [DI, 2 DI) - both L2-normalised - v at [2 DI, 3 DI)
    int C;
    const float* gate;         // [T][h_v][128] natural-log decay per key dimension
    const float* beta;         // [T][h_v] (the sigmoid)
    const __nv_bfloat16* v;    // [T][h_v][128]
    float* state;              // [128 key][h_v][128 value]
    float* out;                // [T][h_v][128]
    int value_heads, tokens;
    float scale;
};

struct PrepareShared {
    Chunk packet;
    __nv_bfloat16 ki[kChunkSize * kStateDim];
    float prefix[kChunkSize * kStateDim];
    float last[kStateDim];
    float beta[kChunkSize];
    float lower[kChunkSize * kChunkSize];
};

__global__ __launch_bounds__(256, 2) void prepare_kernel(Args args, Chunk* __restrict__ workspace) {
    __shared__ PrepareShared sm;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int head = blockIdx.x % args.value_heads;
    const int chunk = blockIdx.x / args.value_heads;
    const int start = chunk * kChunkSize;
    const int length = min(kChunkSize, args.tokens - start);
    const int DI = args.value_heads * kStateDim;
    const int row = tid / 16;
    const int col = (tid & 15) * 8;

    // this engine: q and k as F32 from the conv output, already L2-normalised (Infernix normalises BF16 q/k here)
    float q[8]{}, k[8]{};
    if (row < length) {
        const float* qp = args.h + (size_t) (start + row) * args.C + head * kStateDim + col;
        const float* kp = qp + DI;
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            const float4 a = load_vec<float4>(qp + i), b = load_vec<float4>(kp + i);
            q[i] = a.x; q[i + 1] = a.y; q[i + 2] = a.z; q[i + 3] = a.w;
            k[i] = b.x; k[i + 1] = b.y; k[i + 2] = b.z; k[i + 3] = b.w;
        }
    }

    if (tid < kChunkSize)   // this engine: beta is the sigmoid already
        sm.beta[tid] = tid < length ? args.beta[(size_t) (start + tid) * args.value_heads + head] : 0.0F;
    if (tid < kStateDim) {   // this engine: the decay's natural log per key dimension, to log2 for exp2
        float sum = 0;
#pragma unroll
        for (int t = 0; t < kChunkSize; ++t) {
            if (t < length) sum += args.gate[((size_t) (start + t) * args.value_heads + head) * kStateDim + tid] * kLog2E;
            sm.prefix[t * kStateDim + tid] = sum;
        }
        sm.last[tid] = sum;
        sm.packet.gamma[tid] = exp2_approx_ftz(sum);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float g = sm.prefix[row * kStateDim + col + i];
        const float decay = exp2_approx_ftz(g);
        const int index = vector_index(row, col + i);
        sm.packet.kd[index] = __float2bfloat16_rn(k[i] * decay);
        sm.packet.qd[index] = __float2bfloat16_rn(q[i] * decay);
        sm.ki[index] = __float2bfloat16_rn(k[i] * exp2_approx_ftz(-g));
        sm.packet.kr[restored_index(row, col + i)] = __float2bfloat16_rn(k[i] * exp2_approx_ftz(sm.last[col + i] - g));
    }
    __syncthreads();

    // two independent 16x16 Gram products, BF16 operands with FP32 accumulation
    if (tid < 64) {
        const auto* a = tid < 32 ? sm.packet.kd : sm.packet.qd;
        float acc[2][4]{};
#pragma unroll
        for (int offset = 0; offset < kStateDim; offset += 16) {
            unsigned af[4];
            const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
            const int ac = offset + (lane >> 4) * 8;
            ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(a + vector_index(ar, ac)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1], smem_addr(sm.ki + vector_index(n * 8 + (lane & 7), offset + ((lane >> 3) & 1) * 8)));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = lane / 4 + (i / 2) * 8;
                const int c = n * 8 + (lane & 3) * 2 + (i & 1);
                if (tid < 32) sm.lower[r * kChunkSize + c] = r > c ? sm.beta[r] * acc[n][i] : 0;
                else sm.packet.mqk[square_index(r, c)] = r >= c ? acc[n][i] : 0;
            }
        }
    }
    __syncthreads();

    // each lane owns a column of the FP32 solve; a row is published only after every column consumed the earlier rows
    if (tid < 32) {
        const int c = lane & 15;
#pragma unroll
        for (int r = 0; r < kChunkSize; ++r) {
            float value = r == c ? 1.0F : 0.0F;
#pragma unroll
            for (int j = 0; j < r; ++j) value = fmaf(-sm.lower[r * kChunkSize + j], sm.packet.solve[square_index(j, c)], value);
            if (lane < 16) sm.packet.solve[square_index(r, c)] = value;
            __syncwarp();
        }
    }
    __syncthreads();
    sm.packet.solve[square_index(row, tid & 15)] *= sm.beta[tid & 15];   // solve = (I + L)^-1 diag(beta)
    __syncthreads();

    // an exact bit permutation: the consumer loads four TF32 A operands with one LDSM.x2
    const int at = vector_index(row, col);
#pragma unroll
    for (int which = 0; which < 2; ++which) {
        auto* plane = which == 0 ? sm.packet.kd : sm.packet.qd;
        const uint4 p = load_vec<uint4>(plane + at);
        const uint4 native = {(p.x & 0xffffU) | (p.z << 16), (p.x >> 16) | (p.z & 0xffff0000U),
                              (p.y & 0xffffU) | (p.w << 16), (p.y >> 16) | (p.w & 0xffff0000U)};
        store_vec(plane + at, native);
    }
    __syncthreads();

    auto* dst = reinterpret_cast<uint4*>(workspace + (size_t) head * chunk_count(args.tokens) + chunk);
    const auto* src = reinterpret_cast<const uint4*>(&sm.packet);
#pragma unroll
    for (int i = tid; i < (int) (sizeof(Chunk) / 16); i += 256) dst[i] = src[i];
}

template <int DV> inline constexpr int kThreads = DV >= 32 ? 256 : 128;

template <int Width> __device__ __forceinline__ int value_index(int row, int col) {
    if constexpr (Width == 8) return row * Width + col;
    if constexpr (Width == 16) return row * Width + (col ^ (((row >> 1) & 1) * 8));
    return row * Width + (col ^ ((row & 3) * 8));
}

__device__ __forceinline__ void load_vector_tf32_a(const __nv_bfloat16* p, int k, int lane, unsigned (&a)[4]) {
    unsigned low, high;
    ldmatrix_x2(low, high, smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k)));
    unpack_bf16x2_to_fp32_bits(low, a[0], a[2]);
    unpack_bf16x2_to_fp32_bits(high, a[1], a[3]);
}
__device__ __forceinline__ void load_vector_bf16_a(const __nv_bfloat16* p, int k, int lane, unsigned (&a)[4]) {
    ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 8)));
}
__device__ __forceinline__ void load_restored_bf16_a(const __nv_bfloat16* p, int key, int lane, unsigned (&a)[4]) {
    ldmatrix_x4_t(a[0], a[1], a[2], a[3], smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8, key + ((lane >> 3) & 1) * 8)));
}
template <int DV>
__device__ __forceinline__ void load_bf16_b(const float* p, int k, int n, int lane, unsigned& b0, unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = pack_bf16x2(p[value_index<DV>(k + t, col)], p[value_index<DV>(k + t + 4, col)]);
    b1 = pack_bf16x2(p[value_index<DV>(k + t + 8, col)], p[value_index<DV>(k + t + 12, col)]);
}

template <int DV> struct RecurrenceShared {
    struct alignas(256) Stage {
        Chunk packet;
        __nv_bfloat16 value[kChunkSize * DV];
    } stage[2];
    float snapshot[kStateDim * DV];
    float delta[kChunkSize * DV];
};

template <int DV>
__device__ __forceinline__ void load_stage(typename RecurrenceShared<DV>::Stage& stage, const Args& args,
                                           const Chunk* workspace, int head, int value_start, int chunk, int tid) {
    const auto* src = reinterpret_cast<const uint4*>(workspace + (size_t) head * chunk_count(args.tokens) + chunk);
    auto* dst = reinterpret_cast<uint4*>(&stage.packet);
#pragma unroll
    for (int i = tid; i < (int) (sizeof(Chunk) / 16); i += kThreads<DV>) cp_async_cg16(dst + i, src + i);
    constexpr int vectors = kChunkSize * DV / 8;
#pragma unroll
    for (int i = tid; i < vectors; i += kThreads<DV>) {
        const int row = i / (DV / 8), col = i % (DV / 8) * 8;
        const int token = chunk * kChunkSize + row;
        const int valid_token = token < args.tokens ? token : 0;
        const size_t off = ((size_t) valid_token * args.value_heads + head) * kStateDim + value_start + col;
        cp_async_cg16_zfill(stage.value + row * DV + col, args.v + off, token < args.tokens ? 16 : 0);
    }
    cp_commit();
}

template <int DV>
__device__ __forceinline__ void store_residual(float* delta, const __nv_bfloat16* value, const float (&prediction)[4],
                                               int n, int lane) {
    const int row = lane / 4, col = n * 8 + (lane & 3) * 2;
    const float2 v0 = __bfloat1622float2(load_vec<__nv_bfloat162>(value + row * DV + col));
    const float2 v1 = __bfloat1622float2(load_vec<__nv_bfloat162>(value + (row + 8) * DV + col));
    store_vec(delta + value_index<DV>(row, col), make_float2(v0.x - prediction[0], v0.y - prediction[1]));
    store_vec(delta + value_index<DV>(row + 8, col), make_float2(v1.x - prediction[2], v1.y - prediction[3]));
}

template <int DV>
__device__ __forceinline__ void matrix_delta(float (&acc)[4], const float* matrix, const float* delta, int n, int lane) {
    const int g = lane / 4, t = lane & 3;
#pragma unroll
    for (int k = 0; k < kChunkSize; k += 8) {
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(matrix + square_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 4)));
        mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3],
                      __float_as_uint(delta[value_index<DV>(k + t, n * 8 + g)]),
                      __float_as_uint(delta[value_index<DV>(k + t + 4, n * 8 + g)]));
    }
}

// this engine: the output F32
__device__ __forceinline__ void store_output(const Args& args, float (&acc)[4], int chunk, int head, int value_start,
                                             int n, int lane) {
    const int row = lane / 4, col = value_start + n * 8 + (lane & 3) * 2;
    const int t0 = chunk * kChunkSize + row;
    const size_t o0 = ((size_t) t0 * args.value_heads + head) * kStateDim + col;
    if (t0 < args.tokens) store_vec(args.out + o0, make_float2(acc[0] * args.scale, acc[1] * args.scale));
    if (t0 + 8 < args.tokens)
        store_vec(args.out + o0 + (size_t) 8 * args.value_heads * kStateDim, make_float2(acc[2] * args.scale, acc[3] * args.scale));
}

// this engine: state [key][head][value] (Infernix: [head][value][key])
__device__ __forceinline__ size_t state_at(const Args& args, int head, int key, int value) {
    return ((size_t) key * args.value_heads + head) * kStateDim + value;
}

template <int DV>
__global__ __launch_bounds__(kThreads<DV>, (DV >= 32 ? 2 : 4)) void chunk_recurrence_kernel(Args args, const Chunk* __restrict__ workspace) {
    constexpr int kWarps = kThreads<DV> / 32;
    constexpr int NT = DV / 8;
    constexpr int MW = NT < kWarps ? kWarps / NT : 1;
    constexpr int MPW = (kStateDim / 16) / MW;
    constexpr int NPW = NT > kWarps ? NT / kWarps : 1;
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm = *reinterpret_cast<RecurrenceShared<DV>*>(backing);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const int head = blockIdx.x / (kStateDim / DV);
    const int value_start = blockIdx.x % (kStateDim / DV) * DV;
    const int n_start = warp % NT;
    const int m_start = NT < kWarps ? warp / NT : 0;
    const int chunks = chunk_count(args.tokens);

    // MMA accumulators describe S^T[key, value]
    float state[NPW][MPW][4];
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key = (m_start + mi * MW) * 16 + g;
            const int value = value_start + n * 8 + t * 2;
            state[ni][mi][0] = args.state[state_at(args, head, key, value)];
            state[ni][mi][1] = args.state[state_at(args, head, key, value + 1)];
            state[ni][mi][2] = args.state[state_at(args, head, key + 8, value)];
            state[ni][mi][3] = args.state[state_at(args, head, key + 8, value + 1)];
        }
    }
    load_stage<DV>(sm.stage[0], args, workspace, head, value_start, 0, tid);
    if (chunks > 1) load_stage<DV>(sm.stage[1], args, workspace, head, value_start, 1, tid);

    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk + 1 < chunks) cp_wait<1>();
        else cp_wait<0>();
#pragma unroll
        for (int ni = 0; ni < NPW; ++ni) {
            const int n = n_start + ni * kWarps;
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key = (m_start + mi * MW) * 16 + g;
                const int col = n * 8 + t * 2;
                store_vec(sm.snapshot + value_index<DV>(key, col), make_float2(state[ni][mi][0], state[ni][mi][1]));
                store_vec(sm.snapshot + value_index<DV>(key + 8, col), make_float2(state[ni][mi][2], state[ni][mi][3]));
            }
        }
        __syncthreads();
        const auto& stage = sm.stage[chunk & 1];
        const Chunk& p = stage.packet;
        float history[NPW][4]{};

        if constexpr (NT >= kWarps) {
            float prediction[NPW][4]{};
#pragma unroll
            for (int k = 0; k < kStateDim; k += 16) {
                unsigned ka0[4], ka1[4], qa[4];
                load_vector_tf32_a(p.kd, k, lane, ka0);
                load_vector_tf32_a(p.kd, k + 8, lane, ka1);
                load_vector_bf16_a(p.qd, k, lane, qa);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    const int n = warp + ni * kWarps;
                    const float b0 = sm.snapshot[value_index<DV>(k + t, n * 8 + g)];
                    const float b1 = sm.snapshot[value_index<DV>(k + t + 4, n * 8 + g)];
                    const float b2 = sm.snapshot[value_index<DV>(k + t + 8, n * 8 + g)];
                    const float b3 = sm.snapshot[value_index<DV>(k + t + 12, n * 8 + g)];
                    mma_tf32_bits(prediction[ni][0], prediction[ni][1], prediction[ni][2], prediction[ni][3], ka0[0],
                                  ka0[1], ka0[2], ka0[3], __float_as_uint(b0), __float_as_uint(b1));
                    mma_tf32_bits(prediction[ni][0], prediction[ni][1], prediction[ni][2], prediction[ni][3], ka1[0],
                                  ka1[1], ka1[2], ka1[3], __float_as_uint(b2), __float_as_uint(b3));
                    mma_bf16(history[ni][0], history[ni][1], history[ni][2], history[ni][3], qa[0], qa[1], qa[2], qa[3],
                             pack_bf16x2(b0, b1), pack_bf16x2(b2, b3));
                }
            }
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) store_residual<DV>(sm.delta, stage.value, prediction[ni], warp + ni * kWarps, lane);
        } else {
            if (warp < 2 * NT) {
                const bool query = warp >= NT;
                const int n = warp % NT;
                if (query) {
#pragma unroll
                    for (int k = 0; k < kStateDim; k += 16) {
                        unsigned af[4], b0, b1;
                        load_vector_bf16_a(p.qd, k, lane, af);
                        load_bf16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                        mma_bf16(history[0][0], history[0][1], history[0][2], history[0][3], af[0], af[1], af[2], af[3], b0, b1);
                    }
                } else {
#pragma unroll
                    for (int k = 0; k < kStateDim; k += 8) {
                        unsigned af[4];
                        load_vector_tf32_a(p.kd, k, lane, af);
                        mma_tf32_bits(history[0][0], history[0][1], history[0][2], history[0][3], af[0], af[1], af[2], af[3],
                                      __float_as_uint(sm.snapshot[value_index<DV>(k + t, n * 8 + g)]),
                                      __float_as_uint(sm.snapshot[value_index<DV>(k + t + 4, n * 8 + g)]));
                    }
                    store_residual<DV>(sm.delta, stage.value, history[0], n, lane);
                }
            }
        }
        __syncthreads();

        // each value tile reads only its own R columns, so R can become Delta in place
#pragma unroll
        for (int n = warp; n < NT; n += kWarps) {
            float delta[4]{};
            matrix_delta<DV>(delta, p.solve, sm.delta, n, lane);
            const int col = n * 8 + t * 2;
            __syncwarp();
            store_vec(sm.delta + value_index<DV>(g, col), make_float2(delta[0], delta[1]));
            store_vec(sm.delta + value_index<DV>(g + 8, col), make_float2(delta[2], delta[3]));
        }
        __syncthreads();

        if constexpr (NT >= kWarps) {
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int n = warp + ni * kWarps;
                matrix_delta<DV>(history[ni], p.mqk, sm.delta, n, lane);
                store_output(args, history[ni], chunk, head, value_start, n, lane);
            }
        } else if (warp >= NT && warp < 2 * NT) {
            const int n = warp - NT;
            matrix_delta<DV>(history[0], p.mqk, sm.delta, n, lane);
            store_output(args, history[0], chunk, head, value_start, n, lane);
        }

        // S^T = diag(gamma) S^T + Kr^T Delta; the state stays an FP32 accumulator
        unsigned delta_b[NPW][2];
#pragma unroll
        for (int ni = 0; ni < NPW; ++ni) load_bf16_b<DV>(sm.delta, 0, n_start + ni * kWarps, lane, delta_b[ni][0], delta_b[ni][1]);
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key = (m_start + mi * MW) * 16 + g;
            const float g0 = p.gamma[key], g1 = p.gamma[key + 8];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                state[ni][mi][0] *= g0;
                state[ni][mi][1] *= g0;
                state[ni][mi][2] *= g1;
                state[ni][mi][3] *= g1;
            }
            unsigned af[4];
            load_restored_bf16_a(p.kr, key - g, lane, af);
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni)
                mma_bf16(state[ni][mi][0], state[ni][mi][1], state[ni][mi][2], state[ni][mi][3], af[0], af[1], af[2], af[3],
                         delta_b[ni][0], delta_b[ni][1]);
        }
        __syncthreads();
        if (chunk + 2 < chunks) load_stage<DV>(sm.stage[chunk & 1], args, workspace, head, value_start, chunk + 2, tid);
    }
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key = (m_start + mi * MW) * 16 + g;
            const int value = value_start + n * 8 + t * 2;
            args.state[state_at(args, head, key, value)] = state[ni][mi][0];
            args.state[state_at(args, head, key, value + 1)] = state[ni][mi][1];
            args.state[state_at(args, head, key + 8, value)] = state[ni][mi][2];
            args.state[state_at(args, head, key + 8, value + 1)] = state[ni][mi][3];
        }
    }
}

template <int DV> void launch_recurrence(const Args& args, const Chunk* workspace, cudaStream_t stream) {
    constexpr int shared_bytes = (int) sizeof(RecurrenceShared<DV>);
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int adev = 0;
    cudaGetDevice(&adev);
    if (adev < 0 || adev >= 64) adev = 0;
    if (!attr[adev]) {
        cudaFuncSetAttribute(chunk_recurrence_kernel<DV>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        attr[adev] = true;
    }
    chunk_recurrence_kernel<DV><<<args.value_heads * (kStateDim / DV), kThreads<DV>, shared_bytes, stream>>>(args, workspace);
}

// Infernix's measured choice on SM120a: the narrowest stripe for small grids, else the fewest weighted waves
int chunk_value_tile(int value_heads, int sm_count) {
    if ((int64_t) value_heads * 16 <= sm_count) return 8;
    int best = 64;
    int64_t best_work = INT64_MAX;
    const int tiles[3][2] = {{64, 34}, {32, 20}, {16, 13}};
    for (const auto& tc : tiles) {
        const int64_t blocks = (int64_t) value_heads * (kStateDim / tc[0]);
        const int64_t work = (blocks + sm_count - 1) / sm_count * tc[1];
        if (work < best_work) { best = tc[0]; best_work = work; }
    }
    return best;
}

}  // namespace

std::size_t glm5_kda_chunk_workspace_bytes(int h_v, int tokens) {
    return (std::size_t) h_v * chunk_count(tokens) * sizeof(Chunk);
}

void glm5_kda_conv_chunk(float* hist, const float* q, const float* k, const float* v, const float* conv_w, float* h,
                         int DI, int qk_heads, float eps, int T, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    conv_chunk_kernel<<<dim3((unsigned) (3 * DI / S), (unsigned) T), S, 0, s>>>(hist, q, k, v, conv_w, h, DI, qk_heads, eps);
    conv_commit_chunk_kernel<<<(unsigned) ((3 * DI + 255) / 256), 256, 0, s>>>(hist, q, k, v, DI, T);
    check("glm5_kda_conv_chunk");
}

void glm5_kda_gate_beta_chunk(const float* f, const float* dt_b, const float* ssm_a, float lower_bound,
                              const float* beta_logit, float* gate, float* beta, int h_v, int T, void* stream) {
    gate_beta_chunk_kernel<<<dim3((unsigned) h_v, (unsigned) T), S, 0, (cudaStream_t) stream>>>(f, dt_b, ssm_a, lower_bound,
                                                                                                beta_logit, gate, beta, h_v);
    check("glm5_kda_gate_beta_chunk");
}

void glm5_kda_chunk(const float* h, int C, const float* gate, const float* beta, float* state, float* y, int h_v, int T,
                    uint16_t* v_bf16, void* workspace, int sm_count, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const int DI = h_v * kStateDim;
    v_bf16_kernel<<<(unsigned) (((size_t) T * DI + 255) / 256), 256, 0, s>>>(h, C, DI, (__nv_bfloat16*) v_bf16, T);
    Args a{h, C, gate, beta, (const __nv_bfloat16*) v_bf16, state, y, h_v, T, 1.0f / sqrtf((float) kStateDim)};
    Chunk* ws = (Chunk*) workspace;
    prepare_kernel<<<(unsigned) ((int64_t) h_v * chunk_count(T)), 256, 0, s>>>(a, ws);
    switch (chunk_value_tile(h_v, sm_count)) {
        case 64: launch_recurrence<64>(a, ws, s); break;
        case 32: launch_recurrence<32>(a, ws, s); break;
        case 16: launch_recurrence<16>(a, ws, s); break;
        default: launch_recurrence<8>(a, ws, s); break;
    }
    check("glm5_kda_chunk");
}

void glm5_kda_out_norm(float* y, const float* z, const float* gamma, float eps, int h_v, int T, void* stream) {
    out_norm_kernel<<<dim3((unsigned) h_v, (unsigned) T), S, 0, (cudaStream_t) stream>>>(y, z, gamma, eps, h_v);
    check("glm5_kda_out_norm");
}

}  // namespace strata::kernels
