// W1 GDN prefill fix (2026-11-14): the llama b2766e3 mmq port (the int8 matmul).
// Q8_0 prequant activation (x arrives f32 - the affine op promotes it) x Q8_1 weight (converted from the mlx G64 4-bit
// affine, v = q*scale + bias, NO -8 centering - verified against
// mx.dequantize element-exact 2026-11-14) via WMMA i32 16x16x16 iu8 signed (RDNA4). Adapted from llama.cpp
// ggml-cuda/mmq.{cuh,cu} (RDNA3-5 / RDNA4 config, gfx1201).
//
// Kernel contract (llama b2766e3, verified against the mul_mat_q kernel and
// quantize_mmq_q8_1 in the reference tree):
//   - x in global memory: prequantized blocks (block_q8_0 here; the dp4a
//     loader reads only the d scale and the qs int8s, min is unused).
//   - y in global memory: block_q8_1_mmq, 144B per 128 values. For x=Q8_0 the
//     D4 layout applies (float d4[4] = one f32 scale per 32 values, then
//     int8 qs[128]). The reference converts the weights to Q8_1 with a
//     pre-kernel; here the source is the mlx G64 format, so mmq_quant_g64_q8_1
//     decodes it directly (no f32 round trip).
//   - stride_row_x is in x-block units (K/32); ncols_* is in element units.
//   - dst is f32 (the dp4a write-back is float); the caller converts to bf16.
#pragma once
#define GGML_USE_HIP 1
#include "common.cuh"
#include "unary.cuh"
#include "mmq.cuh"
#include "mmq-vec-dot.cuh"
#include "mmq.cu"

#include <hip/hip_bfloat16.h>

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/fast_primitives.h"

namespace rocm {

// f32 -> block_q8_0. One thread per 32-element subblock.
// (the affine quantized_matmul op promotes x to the scales dtype = f32).
// llama q8_0: d = amax/127, qs = round(v/d).
__global__ void mmq_quant_f32_q8_0(const float* __restrict__ x,
                                   block_q8_0* __restrict__ y, int K,
                                   int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int kb = K / 32;
  const int n = (int)(idx / kb);
  const int k = (int)(idx % kb);
  const float* xr = x + (int64_t)n * K + k * 32;
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    v[i] = xr[i];
    amax = fmaxf(amax, fabsf(v[i]));
  }
  block_q8_0 b{};
  if (amax == 0.f) {
    __half hv = __float2half(1.f);
    b.d = *reinterpret_cast<const uint16_t*>(&hv);
#pragma unroll
    for (int i = 0; i < 32; ++i) b.qs[i] = 0;
  } else {
    const float d_inv = 127.f / amax;
    __half hv = __float2half(1.f / d_inv);
    b.d = *reinterpret_cast<const uint16_t*>(&hv);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int q = (int)rintf(v[i] * d_inv);
      q = max(-127, min(127, q));
      b.qs[i] = (int8_t)q;
    }
  }
  y[idx] = b;
}

// mlx G64 4-bit affine -> block_q8_1_mmq (the D4 layout). One thread per
// 32-element subblock. mlx packing (qmm.hip): element e of an 8-element u32
// is (w >> 4*e) & 0xF; value = (raw - 8) * scale + bias.
__global__ void mmq_quant_g64_q8_1(const uint32_t* __restrict__ w,
                                   const float* __restrict__ scales,
                                   const float* __restrict__ biases,
                                   char* __restrict__ y, int N, int K,
                                   int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int kb = K / 32;
  const int n = (int)(idx / kb);
  const int k = (int)(idx % kb);
  const uint32_t* wr = w + (int64_t)n * (K / 8) + (int64_t)k * 4;
  const int g = k >> 1;  // 64-element group (two 32-element subblocks)
  const float sc = scales[(int64_t)n * (K / 64) + g];
  const float bz = biases[(int64_t)n * (K / 64) + g];
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t p = wr[i];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[i * 8 + e] = (float)((p >> (4 * e)) & 0xFu) * sc + bz;
      amax = fmaxf(amax, fabsf(v[i * 8 + e]));
    }
  }
  float d = 1.f;
  int8_t q[32];
  if (amax == 0.f) {
#pragma unroll
    for (int i = 0; i < 32; ++i) q[i] = 0;
  } else {
    const float d_inv = 127.f / amax;
    d = 1.f / d_inv;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int t = (int)rintf(v[i] * d_inv);
      t = max(-127, min(127, t));
      q[i] = (int8_t)t;
    }
  }
  const int b128 = k >> 2;
  const int sub = k & 3;
  // the kernel reads y K-block-major: y[k_block][n] (ib = k_block*ne1 + n)
  char* yb = y + ((int64_t)b128 * N + n) * 144;
  *reinterpret_cast<float*>(yb + sub * 4) = d;
  int8_t* qp = reinterpret_cast<int8_t*>(yb + 16 + sub * 32);
#pragma unroll
  for (int i = 0; i < 32; ++i) qp[i] = q[i];
}

// Transpose + convert: the mmq kernel writes dst in the ggml layout (m-inner:
// dst[j*M + i], j = N direction, i = M direction). MLX arrays are n-inner
// (row-major M x N: out[i*N + j]). 32x32 tiled transpose, 32x8 block.
template <typename OutT>
__global__ void mmq_tcopy(const float* __restrict__ in, OutT* __restrict__ out,
                          int M, int N) {
  __shared__ float tile[33][32];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m0 = blockIdx.x * 32;  // M direction
  const int n0 = blockIdx.y * 32;  // N direction
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    tile[ty * 4 + j][tx] = in[(n0 + ty * 4 + j) * M + (m0 + tx)];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[(m0 + ty * 4 + j) * N + (n0 + tx)] = (OutT)tile[tx][ty * 4 + j];
  }
}

// GFX1201-PERSISTENT-SCRATCH (build 47, 2026-11-15): the pool-bypass
// per-call hipMalloc/hipFree costs whole-device syncs (hipFree on ROCm).
// Grow-only scratch, mutex-guarded (allocation decisions only - the GEMMs
// stay async on the caller stream). The mmq path is single-stream in this
// engine (the graph eval stream); the record battery catches a divergence
// if the engine ever runs mmq on two streams.
inline struct mmq_scratch_t {
  std::mutex mtx;
  void* p[8] = {nullptr};
  size_t cap[8] = {0};
  void* get(int i, size_t bytes) {
    if (cap[i] < bytes) {
      if (p[i]) CUDA_CHECK(hipFree(p[i]));
      CUDA_CHECK(hipMalloc(&p[i], bytes));
      cap[i] = bytes;
    }
    return p[i];
  }
} g_mmq_scratch;
inline void* mmq_scratch_get(int i, size_t bytes) {
  std::lock_guard<std::mutex> lk(g_mmq_scratch.mtx);
  return g_mmq_scratch.get(i, bytes);
}

// Host launcher: per call convert x (bf16) -> Q8_0 and w (mlx G64) -> Q8_1,
// run the mmq (int8 dp4a) into f32; when out is bf16 the f32 temp is
// converted back (out dtype is f32 in the mlx graph - no conversion).
static void mmq_q8_0_g64(const float* x, const uint32_t* w,
                         const float* scales, const float* biases,
                         void* out, bool dst_f32, int M, int N, int K,
                         hipStream_t stream) {
  ggml_backend_cuda_context ctx(0);  // the device 0 (the gfx1201)
  const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;  // gfx1201
  const size_t smpbo = ggml_cuda_info().devices[0].smpbo;  // the max LDS per block
  const bool fallback = ggml_cuda_mmq_needs_fallback(M);
  {
    static std::atomic<bool> mmq_cfg_dbg_done{false};
    if (mmq_cfg_dbg_done.exchange(true) == false) {
      fprintf(stderr, "[mmq-cfg] cc=%d smpbo=%zu M=%d fallback=%d N=%d\n",
              cc, smpbo, M, (int)fallback, N);
      for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config cfg =
            ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, fallback, cc, GGML_PREC_Q8);
        fprintf(stderr, "[mmq-cfg]   J=%d type=%d nthreads=%d I=%d nbytes=%zu\n",
                J, (int)cfg.type, cfg.nthreads, cfg.I,
                mmq_get_nbytes_shared(cfg, cc));
      }
    }
  }

  const int64_t x_total = (int64_t)M * K / 32;
  const int64_t y_total = (int64_t)N * K / 32;
  // GFX1201-POOL-BYPASS (kept, build 54): per-call hipMalloc/hipFree. The
  // persistent-scratch variant (builds 47-53) exposed INTERMITTENT mmq
  // kernel-loss on gfx1201/ROCm 7.2 (3run r3 stale-dst signature 1.424 ~=
  // sqrt(2) of the prior call; build 50 lost 4/4 first calls). The device
  // sync in hipFree masks the loss - same disease family as the banked
  // async-pool write-loss. The sync-free TF/s (24.7/29.6) is SUSPECT until
  // the loss is RCA'd; the banked numbers carry the sync.
  void* xq8 = nullptr;
  void* yq81 = nullptr;
  CUDA_CHECK(hipMalloc(&xq8, (size_t)x_total * sizeof(block_q8_0)));
  CUDA_CHECK(hipMalloc(&yq81, (size_t)y_total * 36));
  float* dst = reinterpret_cast<float*>(out);  // PATCH81: GEMM writes out directly (no f32 temp)


  const int tb = 256;
  mmq_quant_f32_q8_0<<<(x_total + tb - 1) / tb, tb, 0, stream>>>(
      x, reinterpret_cast<block_q8_0*>(xq8), K, x_total);
  mmq_quant_g64_q8_1<<<(y_total + tb - 1) / tb, tb, 0, stream>>>(
      w, scales, biases, reinterpret_cast<char*>(yq81), N, K, y_total);

  int nthreads_best = 0;
  int J_best = ggml_cuda_mmq_get_J_best(GGML_TYPE_Q8_0, N, fallback, cc,
                                        smpbo, GGML_PREC_Q8, &nthreads_best);
  if (const char* je = std::getenv("MLX_MMQ_J")) {
    const int jv = std::atoi(je);
    if (jv > 0) J_best = jv;  // build 56: config sweep knob (default = heuristic)
  }
  mmq_args args = {reinterpret_cast<const char*>(xq8), GGML_TYPE_Q8_0,
                   reinterpret_cast<const int*>(yq81), nullptr, nullptr,
                   dst,
                   nullptr, K, M, N, K / 32, N, M,
                   1, 1, 0, 0, 0,
                   1, 1, 0, 0, 0,
                   N, J_best, {}};
  args.dst_bf16 = !dst_f32;  // PATCH81
  mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
  CUDA_CHECK(hipGetLastError());

  // PATCH81: tcopy killed - the GEMM wrote out directly. hipFree is a device
  // sync on ROCm - it waits for the GEMM before freeing (kernel-loss mask kept).
  CUDA_CHECK(hipFree(xq8));
  CUDA_CHECK(hipFree(yq81));
}

// ============================================================================
// R6 recipe item 1 (2026-11-14): the GLU-fused mmq variant (the llama dp4a
// has_fusion port). The weights (mlx G64 4-bit affine) convert to Q8_0
// (32-group, d as half) into the x slot (up + gate as two buffers); the
// activations convert to the Q8_1-mmq layout (K-block-major 144B/128 rows)
// into the y slot. The kernel loads 8-row up/gate alternating tiles from
// the two Q8_0 weight buffers (ggml_cuda_mmq_x_row has_fusion) and the dp4a
// write_back pairs (i, i+8) in a 32-lane row via __shfl and writes the
// fused GLU (swiglu = up * silu(gate)). dst = (M, N) row-major (no
// transpose) -> plain tcopy. Gated by MLX_GLU_MMQ=1 (DEFAULT OFF).

// mlx G64 -> block_q8_0 (32-group, d as half). One thread per 32-elem block.
__global__ void mmq_quant_g64_q8_0(const uint32_t* __restrict__ w,
                                   const float* __restrict__ scales,
                                   const float* __restrict__ biases,
                                   block_q8_0* __restrict__ y, int N, int K,
                                   int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int kb = K / 32;
  const int n = (int)(idx / kb);
  const int k = (int)(idx % kb);
  const uint32_t* wr = w + (int64_t)n * (K / 8) + (int64_t)k * 4;
  const int g = k >> 1;  // 64-elem group (two 32-elem blocks)
  const float sc = scales[(int64_t)n * (K / 64) + g];
  const float bz = biases[(int64_t)n * (K / 64) + g];
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t p = wr[i];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[i * 8 + e] = (float)((p >> (4 * e)) & 0xFu) * sc + bz;
      amax = fmaxf(amax, fabsf(v[i * 8 + e]));
    }
  }
  block_q8_0 b{};
  if (amax == 0.f) {
    __half hv = __float2half(1.f);
    b.d = *reinterpret_cast<const uint16_t*>(&hv);
  } else {
    const float d_inv = 127.f / amax;
    __half hv = __float2half(1.f / d_inv);
    b.d = *reinterpret_cast<const uint16_t*>(&hv);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int t = (int)rintf(v[i] * d_inv);
      t = max(-127, min(127, t));
      b.qs[i] = (int8_t)t;
    }
  }
  y[idx] = b;
}

// f32 activations -> Q8_1-mmq layout (144B per 128 values, K-block-major
// y[k_block][m]). One thread per (m, k128).
// P1 plan-B (2026-11-16): templated input (f32 or bf16) - the in-model
// activations are bf16 (static_cast is the identity for f32).
template <typename InT>
__global__ void mmq_quant_in_q8_1_mmq(const InT* __restrict__ x,
                                     char* __restrict__ y, int M, int K,
                                     int64_t total, bool ds4) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int kb128 = K / 128;
  const int m = (int)(idx / kb128);
  const int k128 = (int)(idx % kb128);
  const InT* xr = x + (int64_t)m * K + (int64_t)k128 * 128;
  float v[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) v[i] = static_cast<float>(xr[i]);
  char* yb = y + ((int64_t)k128 * M + m) * 144;
#pragma unroll
  for (int sub = 0; sub < 4; ++sub) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[sub * 32 + i]));
    float d = 1.f;
    int8_t q[32];
    if (amax == 0.f) {
      d = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) q[i] = 0;
    } else {
      const float d_inv = 127.f / amax;
      d = 1.f / d_inv;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        int t = (int)rintf(v[sub * 32 + i] * d_inv);
        t = max(-127, min(127, t));
        q[i] = (int8_t)t;
      }
    }
    if (ds4) {
      // DS4 layout (the Q4_K/Q5_K/Q6_K mmq class, upstream
      // quantize_mmq_q8_1): half2 (d, sum of the 32 raw values) per 32 -
      // vec_dot_q4_K_q8_1_impl_mmq reads the scale as half2 and the high
      // half is the raw-sum min-correction term.
      float sum32 = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) sum32 += v[sub * 32 + i];
      *reinterpret_cast<half2*>(yb + sub * 4) = __floats2half2_rn(d, sum32);
    } else {
      *reinterpret_cast<float*>(yb + sub * 4) = d;
    }
    int8_t* qp = reinterpret_cast<int8_t*>(yb + 16 + sub * 32);
#pragma unroll
    for (int i = 0; i < 32; ++i) qp[i] = q[i];
  }
}

// plain (M, N) f32 -> OutT copy (no transpose; the fused dst is already
// out-layout).
template <typename OutT>
__global__ void mmq_tcopy_plain(const float* __restrict__ in,
                                OutT* __restrict__ out, int M, int N) {
  __shared__ float tile[33][32];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m0 = blockIdx.x * 32;  // M direction
  const int n0 = blockIdx.y * 32;  // N direction
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    tile[ty * 4 + j][tx] = in[(m0 + ty * 4 + j) * N + (n0 + tx)];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[(m0 + ty * 4 + j) * N + (n0 + tx)] = (OutT)tile[tx][ty * 4 + j];
  }
}

// Identity (M,N) -> (M,N) copy (the GLU fused GEMM writes dst as
// (M_act, N_weight) row-major - no transpose; mmq_tcopy_plain above is
// a (M,N)-in transpose used by the plain path's (N,M) dst layout).
template <typename OutT>
__global__ void mmq_tcopy_id(const float* __restrict__ in,
                             OutT* __restrict__ out, int M, int N) {
  __shared__ float tile[33][32];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m0 = blockIdx.x * 32;
  const int n0 = blockIdx.y * 32;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    tile[ty * 4 + j][tx] = in[(m0 + ty * 4 + j) * N + (n0 + tx)];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[(m0 + ty * 4 + j) * N + (n0 + tx)] = (OutT)tile[ty * 4 + j][tx];
  }
}

// Build 56: vectorized identity copy. The GLU dst layout is (M_act, N_weight)
// row-major -> elementwise f32 -> OutT; no transpose, no shared memory.
// The scalar 32x32-staged version ran 301MB (A shape) at ~51 GB/s (profile
// 11-14: tcopy 5.92ms of the 33ms call); float4 loads target the ~600 GB/s
// class. Rounding: __floats2bfloat162_rn = RNE, same as the (OutT) cast.
template <typename OutT>
__global__ void mmq_tcopy_id_vec(const float* __restrict__ in,
                                 OutT* __restrict__ out, size_t n4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 f4 = reinterpret_cast<const float4*>(in)[i];
  if constexpr (std::is_same<OutT, __hip_bfloat16>::value) {
    const __hip_bfloat162 lo = __float22bfloat162_rn(make_float2(f4.x, f4.y));
    const __hip_bfloat162 hi = __float22bfloat162_rn(make_float2(f4.z, f4.w));
    uint2 pack;
    pack.x = *reinterpret_cast<const unsigned*>(&lo);
    pack.y = *reinterpret_cast<const unsigned*>(&hi);
    reinterpret_cast<uint2*>(out)[i] = pack;
  } else {
    reinterpret_cast<float4*>(out)[i] = f4;
  }
}

// Host launcher: the fused GLU GEMM. x (M,K) f32, w_up/w_gate (N,K) G64.
static void mmq_glu_g64(const float* x,
                        const char* w_up, const float* su, const float* bu,
                        const char* w_gate, const float* sg, const float* bg,
                        void* out, bool dst_f32, int M, int N, int K,
                        hipStream_t stream) {
  ggml_backend_cuda_context ctx(0);
  const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
  const size_t smpbo = ggml_cuda_info().devices[0].smpbo;

  const int64_t w_total = (int64_t)N * K / 32;
  const int64_t x_total = (int64_t)M * K / 32;
  const int64_t x_tiles = (int64_t)M * (K / 128);  // 128-elem tiles (kernel total)
  // GFX1201-POOL-BYPASS (kept, build 54): see the plain launcher comment -
  // the persistent scratch exposed the intermittent kernel-loss.
  void* wq8_u = nullptr;
  void* wq8_g = nullptr;
  void* xq81 = nullptr;
  CUDA_CHECK(hipMalloc(&wq8_u, (size_t)w_total * sizeof(block_q8_0)));
  CUDA_CHECK(hipMalloc(&wq8_g, (size_t)w_total * sizeof(block_q8_0)));
  CUDA_CHECK(hipMalloc(&xq81, (size_t)x_total * 36));
  float* dst = reinterpret_cast<float*>(out);  // PATCH81: GEMM writes out directly (no f32 temp)

  hipEvent_t ev[6];
  for (int ei = 0; ei < 6; ei++) CUDA_CHECK(hipEventCreate(&ev[ei]));
  static std::atomic<int> prof_calls{0};  // PATCH82: steady-state prof - fire on the 11th call
  const bool prof = (prof_calls.fetch_add(1) == 10);
  if (prof) CUDA_CHECK(hipEventRecord(ev[0], stream));
  const int tb = 256;
  mmq_quant_g64_q8_0<<<(w_total + tb - 1) / tb, tb, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(w_up), su, bu,
      reinterpret_cast<block_q8_0*>(wq8_u), N, K, w_total);
  if (prof) CUDA_CHECK(hipEventRecord(ev[1], stream));
  mmq_quant_g64_q8_0<<<(w_total + tb - 1) / tb, tb, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(w_gate), sg, bg,
      reinterpret_cast<block_q8_0*>(wq8_g), N, K, w_total);
  if (prof) CUDA_CHECK(hipEventRecord(ev[2], stream));
  mmq_quant_in_q8_1_mmq<float><<<(x_tiles + tb - 1) / tb, tb, 0, stream>>>(
      x, reinterpret_cast<char*>(xq81), M, K, x_tiles, /*ds4=*/false);
  if (prof) CUDA_CHECK(hipEventRecord(ev[3], stream));


  int nthreads_best = 0;
  // J is the y-slot tile = the activations (M rows).
  int J_best = ggml_cuda_mmq_get_J_best(GGML_TYPE_Q8_0, M, false, cc,
                                        smpbo, GGML_PREC_Q8,
                                        &nthreads_best);

  if (const char* je = std::getenv("MLX_MMQ_J")) { int jf = std::atoi(je); if (jf >= 8 && jf <= 128 && jf % 8 == 0) J_best = jf; }
  if (const char* je = std::getenv("MLX_MMQ_J")) {
    const int jv = std::atoi(je);
    if (jv > 0) J_best = jv;  // build 56: config sweep knob (default = heuristic)
  }
  {
    static std::atomic<bool> cfg80{false};
    if (cfg80.exchange(true) == false) {
      const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J_best, false, cc, GGML_PREC_Q8);
      fprintf(stderr, "[cfg80] M=%d N=%d K=%d cc=%d J_best=%d cfg.J=%d cfg.I=%d cfg.nthreads=%d nbytes=%zu I_dst=%d ntx=%d nty=%d\n",
              M, N, K, cc, J_best, c.J, c.I, c.nthreads,
              mmq_get_nbytes_shared(c, cc), c.I/2,
              (int)(( (size_t)M + c.J - 1) / c.J), (int)(( (size_t)N + c.I/2 - 1) / (c.I/2)));
      fprintf(stderr, "[ptr80] wq8_u=%p wq8_g=%p xq81=%p dst=%p out=%p sizes: wu=%zu wg=%zu x=%zu dst=%zu\n",
              (void*)wq8_u, (void*)wq8_g, (void*)xq81, (void*)dst, (void*)out,
              (size_t)w_total * sizeof(block_q8_0), (size_t)w_total * sizeof(block_q8_0),
              (size_t)x_total * 36, (size_t)M * N * 4);
    }
  }
  mmq_args args = {reinterpret_cast<const char*>(wq8_u), GGML_TYPE_Q8_0,
                   reinterpret_cast<const int*>(xq81), nullptr, nullptr,
                   dst,
                   nullptr, K, N, M, K / 32, M, N,
                   1, 1, 0, 0, 0,
                   1, 1, 0, 0, 0,
                   M, J_best, {}};
  args.fusion.gate = wq8_g;
  args.fusion.glu_op = GGML_GLU_OP_SWIGLU;
  args.dst_bf16 = !dst_f32;  // PATCH81
  mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
  CUDA_CHECK(hipGetLastError());
  if (prof) CUDA_CHECK(hipEventRecord(ev[4], stream));
  if (prof) CUDA_CHECK(hipEventRecord(ev[5], stream));  // PATCH81: tcopy removed, ev[5] at gemm end
  if (prof) {
    CUDA_CHECK(hipEventSynchronize(ev[5]));
    float ms[5];
    hipEventElapsedTime(&ms[0], ev[0], ev[1]);
    hipEventElapsedTime(&ms[1], ev[1], ev[2]);
    hipEventElapsedTime(&ms[2], ev[2], ev[3]);
    hipEventElapsedTime(&ms[3], ev[3], ev[4]);
    hipEventElapsedTime(&ms[4], ev[4], ev[5]);
    const double total = ms[0]+ms[1]+ms[2]+ms[3]+ms[4];  // PATCH81: ms (was seconds labeled ms)
    fprintf(stderr, "[prof-glu] M=%d N=%d K=%d wconv_u=%.3f wconv_g=%.3f xpreq=%.3f gemm=%.3f tcopy=%.3f ms | total=%.3f ms | gemm-only=%.1f TF/s (4MNK) | full=%.1f TF/s (4MNK)\n",
            M, N, K, (double)ms[0], (double)ms[1], (double)ms[2], (double)ms[3], (double)ms[4],
            (double)total, 4.0*M*N*K/(ms[3]/1000.0)/1e12, 4.0*M*N*K/(total/1000.0)/1e12);
  }
  for (int ei = 0; ei < 6; ei++) CUDA_CHECK(hipEventDestroy(ev[ei]));

  // PATCH81: tcopy killed - hipFree is a device sync on ROCm, waits for the GEMM.
  CUDA_CHECK(hipFree(wq8_u));
  CUDA_CHECK(hipFree(wq8_g));
  CUDA_CHECK(hipFree(xq81));
}

// ============================================================================
// Build 57 (R6 recipe, the kernel-level lever): the Q4_K weight path.
//
// The per-stage profile (build 55) showed the GEMM kernel at 31.0 TF/s
// (4MNK) owning 80% of the call, and the kernel + config + vec-dot sources
// are byte-identical to the anchor build's tree (29c720ea). The remaining
// structural difference vs the R6 recipe (their b2766e3 prefill): the
// WEIGHT TYPE - ours = Q8_0 (8.25 bits/weight through the LDS tile), theirs
// = Q4_K (4.5 bits/weight, in-SRAM dequant, the mmq Q4_K loader/vec_dot are
// the stock ones, already instantiated). This repacks the mlx G64 4-bit
// affine weights into block_q4_K (144B/256 = 4.5 bits/weight) per call:
// nibbles pass through EXACT (G64 value = q*s+b, q uint 0..15); the per-32
// scale/min ride the q4_K grid (4-bit scale + 2-bit min per 32 sub-block +
// superblock d/dmin halves) - the scale byte layout unpack_scales_q45_K
// reads: byte c (c<4) = sc_c | mn_c<<4 | mn_{c+4}<<6, byte 4+c = mn_c,
// byte 8+c = sc_{c+4} | mn_{c+4}<<4. One 256-thread block per 256 weights.
// P1 plan-B (2026-11-16): templated scale/bias dtype (f32 or bf16) - the
// in-model G64 affine scales/biases are bf16 (the R6 recipe dequants
// v = q*scale + bias in f32; static_cast is the identity for f32).
template <typename ST>
__global__ void mmq_repack_g64_q4k(const uint32_t* __restrict__ w,
                                   const ST* __restrict__ s,
                                   const ST* __restrict__ b,
                                   block_q4_K* __restrict__ out) {
  const int64_t e0 = (int64_t)blockIdx.x * 256;
  __shared__ float s32[8], m32[8];
  __shared__ uint8_t qs128[128];
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  block_q4_K* outb = out + blockIdx.x;  // build 62g: one block_q4_K per blockIdx
  // warp = one 32-sub-block; 4 threads per G64 u32 word, 1 nibble each.
  const int w_idx = lane / 8;
  const int nib = lane % 8;
  const int sb = (int)(e0 / 64) + warp / 2;  // the G64-64 index for this 32
  const uint32_t wv = w[e0 / 8 + (int64_t)warp * 4 + w_idx];
  const int q = (int)((wv >> (4 * nib)) & 0xF);
  const float sv = static_cast<float>(s[sb]);
  const float bv = b ? static_cast<float>(b[sb]) : 0.f;
  const float v = (float)q * sv + bv;
  float vmin = v, vmax = v;
  #pragma unroll
  for (int o = 16; o > 0; o /= 2) {
    vmin = fminf(vmin, __shfl_xor(vmin, o));
    vmax = fmaxf(vmax, __shfl_xor(vmax, o));
  }
  if (lane == 0) {
    s32[warp] = (vmax - vmin) / 15.0f;
    m32[warp] = -vmin;
  }
  __syncthreads();
  if (warp == 0) {
    float dmax = 0.0f, mmax = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      dmax = fmaxf(dmax, s32[c]);
      mmax = fmaxf(mmax, m32[c]);
    }
    const float d_super = dmax > 0.0f ? dmax / 63.0f : 1.0f;
    const float dmin_super = mmax > 0.0f ? mmax / 63.0f : 1.0f;
    uint8_t sc8[8], mn8[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      sc8[c] = (uint8_t) min(63, (int)roundf(s32[c] / d_super));
      mn8[c] = (uint8_t) min(63, (int)roundf(m32[c] / dmin_super));
    }
    if (lane < 4) {
      // qkx2 Q4_K scale layout (get_scale_min_k4 / unpack_scales_q45_K):
      // bytes 0-3: sc0-3 (6b) | sc4-7 hi2; bytes 4-7: mn0-3 (6b) | mn4-7 hi2;
      // bytes 8-11: sc4-7 lo4 | mn4-7 lo4.
      outb->scales[lane] = (uint8_t)((sc8[lane] & 0x3F) | (((sc8[lane + 4] >> 4) & 0x03) << 6));
      outb->scales[4 + lane] = (uint8_t)((mn8[lane] & 0x3F) | (((mn8[lane + 4] >> 4) & 0x03) << 6));
      outb->scales[8 + lane] = (uint8_t)((sc8[lane + 4] & 0x0F) | ((mn8[lane + 4] & 0x0F) << 4));
    }
    if (lane == 0) {
      outb->dm = __floats2half2_rn(d_super, dmin_super);  // (d, dmin)
    }
  }
  __syncthreads();
  // sub-block requant (build 62): the dequant scale is per-32
  // ((vmax-vmin)/15, -vmin) but the G64 nibble is group-64 relative,
  // so passthrough mis-scales sub-blocks whose min/max differ from the
  // group's. Re-round on the sub-block grid (the standard Q4_K encoding).
  const float s32c = s32[warp];
  const float m32c = m32[warp];  // = -vmin
  const int q4k = (s32c > 0.f)
      ? (int) rintf((v + m32c) / s32c)
      : 0;
  const int q4kc = max(0, min(15, q4k));
  // P1B4-STDQS (2026-11-17, audit vs 29c720eaf quantize_row_q4_K_ref): the
  // CANONICAL q4_K qs layout - per 64 group, 32 bytes:
  //   byte 32*g4 + l = (elem 64g4+32+l << 4) | elem 64g4+l, l = 0..31.
  // The reference WMMA two-line staging (loader P1B4-C, verbatim from
  // 29c720eaf) unpermutes exactly this layout to natural K lanes (mma word
  // w = elements 4w..4w+3) - the D1 battery-verified reference combination.
  // The build-63f per-subblock (elem 2j, 2j+1) pack is a different layout:
  // every loader variant tried against it mispairs (AUDIT3: candidate B
  // RELERR 1.9 c=-0.004; B97-era raw-stream 8.9x c=0.01).
  // Two-phase shared pack (no byte |= race: phase 1 writes every byte once,
  // phase 2 reads it back, ORs the partner warp's nibble, stores).
  if ((warp % 2) == 0) {
    qs128[32 * (warp / 2) + lane] = (uint8_t)q4kc;
  }
  __syncthreads();
  if ((warp % 2) == 1) {
    qs128[32 * (warp / 2) + lane] =
        (uint8_t)(qs128[32 * (warp / 2) + lane] | (q4kc << 4));
  }
  __syncthreads();
  if (warp == 0) {
    for (int l = lane; l < 128; l += 32) {
      outb->qs[l] = qs128[l];
    }
  }
}

// GEMM-1 3b-2' A/B (jimmy 2026-11-19): transpose-requant. G64 (N,K)
// K-fastest source -> FILE-order canonical q4_K blocks (K, N/256, 144):
// file block (k, nblock) = 256 consecutive n at fixed k (the kqfg op's w
// layout; the mmq repack is the transposed grouping, storage order). Same
// requant math + qkx2/qs pack as mmq_repack_g64_q4k (the per-32
// sub-blocks run over the n axis here; per-n-row G64 scale/bias - no
// per-warp share). One 256-thread block per file block; one-time fill,
// identity-cached by the caller.
template <typename ST>
__global__ void kqfg_ab_transpose_requant(const uint32_t* __restrict__ w,
                                          const ST* __restrict__ s,
                                          const ST* __restrict__ b,
                                          block_q4_K* __restrict__ out,
                                          int N, int K) {
  const int nblock = blockIdx.x;
  const int k = blockIdx.y;
  const int tid = threadIdx.x;
  const int n = nblock * 256 + tid;
  __shared__ float s32[8], m32[8];
  __shared__ uint8_t qs128[128];
  const int warp = tid / 32;
  const int lane = tid % 32;
  block_q4_K* outb = out + (size_t)nblock + (size_t)k * (N / 256);
  // source element (n, k): G64 nibble + per-n-row 64-group scale/bias
  const uint32_t wv = w[(size_t)n * (K / 8) + (k / 8)];
  const int q = (int)((wv >> (4 * (k % 8))) & 0xF);
  const int64_t sb = (int64_t)n * (K / 64) + (k / 64);
  const float sv = static_cast<float>(s[sb]);
  const float bv = b ? static_cast<float>(b[sb]) : 0.f;
  const float v = (float)q * sv + bv;
  float vmin = v, vmax = v;
  #pragma unroll
  for (int o = 16; o > 0; o /= 2) {
    vmin = fminf(vmin, __shfl_xor(vmin, o));
    vmax = fmaxf(vmax, __shfl_xor(vmax, o));
  }
  if (lane == 0) {
    s32[warp] = (vmax - vmin) / 15.0f;
    m32[warp] = -vmin;
  }
  __syncthreads();
  if (warp == 0) {
    float dmax = 0.0f, mmax = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      dmax = fmaxf(dmax, s32[c]);
      mmax = fmaxf(mmax, m32[c]);
    }
    const float d_super = dmax > 0.0f ? dmax / 63.0f : 1.0f;
    const float dmin_super = mmax > 0.0f ? mmax / 63.0f : 1.0f;
    uint8_t sc8[8], mn8[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      sc8[c] = (uint8_t) min(63, (int)roundf(s32[c] / d_super));
      mn8[c] = (uint8_t) min(63, (int)roundf(m32[c] / dmin_super));
    }
    if (lane < 4) {
      outb->scales[lane] = (uint8_t)((sc8[lane] & 0x3F) | (((sc8[lane + 4] >> 4) & 0x03) << 6));
      outb->scales[4 + lane] = (uint8_t)((mn8[lane] & 0x3F) | (((mn8[lane + 4] >> 4) & 0x03) << 6));
      outb->scales[8 + lane] = (uint8_t)((sc8[lane + 4] & 0x0F) | ((mn8[lane + 4] & 0x0F) << 4));
    }
    if (lane == 0) {
      outb->dm = __floats2half2_rn(d_super, dmin_super);
    }
  }
  __syncthreads();
  const float s32c = s32[warp];
  const float m32c = m32[warp];  // = -vmin
  const int q4k = (s32c > 0.f)
      ? (int) rintf((v + m32c) / s32c)
      : 0;
  const int q4kc = max(0, min(15, q4k));
  if ((warp % 2) == 0) {
    qs128[32 * (warp / 2) + lane] = (uint8_t)q4kc;
  }
  __syncthreads();
  if ((warp % 2) == 1) {
    qs128[32 * (warp / 2) + lane] =
        (uint8_t)(qs128[32 * (warp / 2) + lane] | (q4kc << 4));
  }
  __syncthreads();
  if (warp == 0) {
    for (int l = lane; l < 128; l += 32) {
      outb->qs[l] = qs128[l];
    }
  }
}

// Host launcher: the Q4_K-weight GLU fused mmq (the R6 recipe weight path).
// x (M,K) f32 -> Q8_1-mmq activations (y slot); the G64 weights -> block_q4_K
// (x slot, up + gate); the mmq Q4_K x Q8_1 dp4a kernel with the has_fusion
// loader (the 8-row up/gate interleave) + the fused GLU writeback.
static void mmq_glu_q4k_g64(const float* x,
                            const char* w_up, const float* su, const float* bu,
                            const char* w_gate, const float* sg, const float* bg,
                            void* out, bool dst_f32, int M, int N, int K,
                            hipStream_t stream) {
  ggml_backend_cuda_context ctx(0);
  const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
  const size_t smpbo = ggml_cuda_info().devices[0].smpbo;

  const int64_t w_total256 = (int64_t)N * K / 256;  // q4_K blocks per weight
  const int64_t x_total = (int64_t)M * K / 32;
  const int64_t x_tiles = (int64_t)M * (K / 128);
  void* wq4_u = nullptr;
  void* wq4_g = nullptr;
  void* xq81 = nullptr;
  CUDA_CHECK(hipMalloc(&wq4_u, (size_t)w_total256 * sizeof(block_q4_K)));
  CUDA_CHECK(hipMalloc(&wq4_g, (size_t)w_total256 * sizeof(block_q4_K)));
  CUDA_CHECK(hipMalloc(&xq81, (size_t)x_total * 36));
  float* dst = reinterpret_cast<float*>(out);  // PATCH81: GEMM writes out directly (no f32 temp)

  hipEvent_t ev[6];
  for (int ei = 0; ei < 6; ei++) CUDA_CHECK(hipEventCreate(&ev[ei]));
  static std::atomic<int> prof_call4k{0};
  const bool prof = (prof_call4k.fetch_add(1) == 4);  // PATCH79: 5th call (1-4 warm the GLU kernels)
  if (prof) CUDA_CHECK(hipEventRecord(ev[0], stream));

  const int tb = 256;
  auto stage4k = [](const char* st, hipError_t rc) {
    if (rc != hipSuccess) fprintf(stderr, "[stage4k] %s rc=%d\n", st, (int)rc);
  };
  // PATCH81: the f32 memset of the temp is gone (no temp; the GEMM writes every element).
  mmq_repack_g64_q4k<float><<<(int)w_total256, tb, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(w_up), su, bu,
      reinterpret_cast<block_q4_K*>(wq4_u));
  stage4k("repack_u", hipGetLastError());
  if (prof) CUDA_CHECK(hipEventRecord(ev[1], stream));
  mmq_repack_g64_q4k<float><<<(int)w_total256, tb, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(w_gate), sg, bg,
      reinterpret_cast<block_q4_K*>(wq4_g));
  stage4k("repack_g", hipGetLastError());
  if (prof) CUDA_CHECK(hipEventRecord(ev[2], stream));
  mmq_quant_in_q8_1_mmq<float><<<(x_tiles + tb - 1) / tb, tb, 0, stream>>>(
      x, reinterpret_cast<char*>(xq81), M, K, x_tiles, /*ds4=*/true);
  stage4k("xpreq", hipGetLastError());
  if (prof) CUDA_CHECK(hipEventRecord(ev[3], stream));
  {
    static const bool dbg4k = [] { const char* e = std::getenv("MLX_4KDBG"); return e != nullptr && e[0] == '1'; }();
    static std::atomic<bool> rdmp{false};
    if (dbg4k && rdmp.exchange(true) == false) {  // PATCH79: gated
      fprintf(stderr, "[align] w_up=%zu su=%zu bu=%zu x=%zu wq4u=%zu wq4g=%zu xq81=%zu out=%zu (mod256)\n",
              (size_t)w_up % 256, (size_t)su % 256, (size_t)bu % 256, (size_t)x % 256,
              (size_t)wq4_u % 256, (size_t)wq4_g % 256, (size_t)xq81 % 256, (size_t)out % 256);
      CUDA_CHECK(hipDeviceSynchronize());
      auto hexd = [](const void* p, size_t n) {
        std::vector<unsigned char> h(n);
        hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost);
        for (size_t k = 0; k < n; k++) fprintf(stderr, "%02x", h[k]);
      };
      fprintf(stderr, "[rdmp] wq4u0:"); hexd(wq4_u, 144); fprintf(stderr, "\n");
      fprintf(stderr, "[rdmp] wq4u101:"); hexd((char*)wq4_u + 101*144, 144); fprintf(stderr, "\n");
      fprintf(stderr, "[rdmp] wq4g0:"); hexd(wq4_g, 144); fprintf(stderr, "\n");
      fprintf(stderr, "[rdmp] xq81_k0:"); hexd(xq81, 128); fprintf(stderr, "\n");
      fprintf(stderr, "[rdmp] xq81_k1:"); hexd((char*)xq81 + (size_t)M * 144, 128); fprintf(stderr, "\n");
      { std::vector<float> h(128); hipMemcpy(h.data(), x, 128*4, hipMemcpyDeviceToHost);
        fprintf(stderr, "[rdmp] x0:"); for (int k = 0; k < 128; k += 8) fprintf(stderr, "%.6g ", (double)h[k]); fprintf(stderr, "\n"); }
      { std::vector<unsigned char> h(128); hipMemcpy(h.data(), w_up, 128, hipMemcpyDeviceToHost);
        fprintf(stderr, "[rdmp] wu0:"); for (size_t k = 0; k < 128; k++) fprintf(stderr, "%02x", h[k]); fprintf(stderr, "\n"); }
      { std::vector<float> h(4); hipMemcpy(h.data(), su, 16, hipMemcpyDeviceToHost);
        fprintf(stderr, "[rdmp] su0:"); for (int k = 0; k < 4; k++) fprintf(stderr, "%.9g ", (double)h[k]); fprintf(stderr, "\n"); }
      { std::vector<float> h(4); hipMemcpy(h.data(), bu, 16, hipMemcpyDeviceToHost);
        fprintf(stderr, "[rdmp] bu0:"); for (int k = 0; k < 4; k++) fprintf(stderr, "%.9g ", (double)h[k]); fprintf(stderr, "\n"); }
      for (size_t i = 0; i < (size_t)N; i++) { fprintf(stderr, "[rdmpa] wu%zu:", i); hexd((char*)wq4_u + i*(size_t)(K/256)*144, 144); fprintf(stderr, "\n"); }
      for (size_t i = 0; i < (size_t)N; i++) { fprintf(stderr, "[rdmpa] wg%zu:", i); hexd((char*)wq4_g + i*(size_t)(K/256)*144, 144); fprintf(stderr, "\n"); }
      for (size_t m = 0; m < (size_t)M; m++) { fprintf(stderr, "[rdmpa] xa%zu:", m); hexd((char*)xq81 + m*(size_t)144, 144); fprintf(stderr, "\n"); }
    }
  }

  int nthreads_best = 0;
  // J is the y-slot tile = the activations (M rows).
  int J_best = ggml_cuda_mmq_get_J_best(GGML_TYPE_Q4_K, M, false, cc,
                                        smpbo, GGML_PREC_Q8, &nthreads_best);

  if (const char* je = std::getenv("MLX_MMQ_J")) { int jf = std::atoi(je); if (jf >= 8 && jf <= 128 && jf % 8 == 0) J_best = jf; }
  if (J_best > 32 && M >= 1024) J_best = 32;  // PATCH80 (round4k 2026-11-15): J=32 beats the get_J_best=128 pick on the dp4a Q4_K path for large M: a1 gemm-only 39.5 vs 44.4 ms (20.9 vs 18.6 TF/s @~3.2GHz), bench 17.6 vs 16.7 (a1) / 13.6 vs 11.9 (a2); J=96 = GPU hang + memory fault (excluded). Small M already gets J<=32 from get_J_best.
  {
    static std::atomic<bool> cfg4k{false};
    if (cfg4k.exchange(true) == false) {
      const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, J_best, false, cc, GGML_PREC_Q8);
      fprintf(stderr, "[cfg4k] M=%d N=%d K=%d cc=%d J_best=%d nthreads_best=%d cfg.type=%d cfg.J=%d cfg.I=%d cfg.nthreads=%d nbytes=%zu I_dst=%d ntx=%d nty=%d\n",
              M, N, K, cc, J_best, nthreads_best, (int)c.type, c.J, c.I, c.nthreads,
              mmq_get_nbytes_shared(c, cc), c.I/2,
              (int)(( (size_t)M + c.J - 1) / c.J), (int)(( (size_t)N + c.I/2 - 1) / (c.I/2)));
    }
  }
  mmq_args args = {reinterpret_cast<const char*>(wq4_u), GGML_TYPE_Q4_K,
                   reinterpret_cast<const int*>(xq81), nullptr, nullptr,
                   dst,
                   nullptr, K, N, M, K / 256, M, N,
                   1, 1, 0, 0, 0,
                   1, 1, 0, 0, 0,
                   M, J_best, {}};
  args.fusion.gate = wq4_g;
  args.fusion.glu_op = GGML_GLU_OP_SWIGLU;
  args.dst_bf16 = !dst_f32;  // PATCH81
  mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
  { hipError_t rc = hipGetLastError();
    if (rc != hipSuccess) { fprintf(stderr, "[stage4k] mmq rc=%d\n", (int)rc);
      CUDA_CHECK(hipGetLastError()); } }
  if (prof) CUDA_CHECK(hipEventRecord(ev[4], stream));

  // PATCH81: tcopy launch removed (the GEMM wrote out directly).
  stage4k("tcopy", hipGetLastError());
  if (prof) {
    CUDA_CHECK(hipEventRecord(ev[5], stream));
    CUDA_CHECK(hipEventSynchronize(ev[5]));
    float ms[5];
    hipEventElapsedTime(&ms[0], ev[0], ev[1]);
    hipEventElapsedTime(&ms[1], ev[1], ev[2]);
    hipEventElapsedTime(&ms[2], ev[2], ev[3]);
    hipEventElapsedTime(&ms[3], ev[3], ev[4]);
    hipEventElapsedTime(&ms[4], ev[4], ev[5]);
    const double total = ms[0]+ms[1]+ms[2]+ms[3]+ms[4];  // PATCH79: ms (was seconds labeled ms)
    fprintf(stderr, "[prof-glu4k] M=%d N=%d K=%d repack_u=%.3f repack_g=%.3f xpreq=%.3f gemm=%.3f tcopy=%.3f ms | total=%.3f ms | gemm-only=%.1f TF/s (4MNK) | full=%.1f TF/s (4MNK)\n",
            M, N, K, (double)ms[0], (double)ms[1], (double)ms[2], (double)ms[3], (double)ms[4],
            (double)total, 4.0*M*N*K/(ms[3]/1000.0)/1e12, 4.0*M*N*K/(total/1000.0)/1e12);
  }
  for (int ei = 0; ei < 6; ei++) CUDA_CHECK(hipEventDestroy(ev[ei]));

  {
    static const bool dbg4k2 = [] { const char* e = std::getenv("MLX_4KDBG"); return e != nullptr && e[0] == '1'; }();
    if (dbg4k2 && dst_f32) {  // PATCH81: f32 out only (bf16 out is written in-kernel now)
      std::vector<float> h((size_t)M*N);
      hipMemcpy(h.data(), dst, (size_t)M*N*4, hipMemcpyDeviceToHost);
      long nz = 0, first = -1;
      for (long m = 0; m < M; m++) for (long ii = 0; ii < N; ii++)
        if (h[(size_t)m*N + ii] != 0.f) { nz++; if (first < 0) first = (long)m*N + ii; }
      fprintf(stderr, "[dstmap] M=%d N=%d nz=%ld/%ld first=%ld rows:", M, N, nz, (long)M*N, first);
      for (long m = 0; m < M; m++) {
        long rn = 0; for (long ii = 0; ii < N; ii++) if (h[(size_t)m*N + ii] != 0.f) rn++;
        if (rn) fprintf(stderr, " m%ld=%ld", m, rn);
      }
      fprintf(stderr, " | col0nz=");
      for (long ii = 0; ii < (N < 64 ? N : 64); ii++) if (h[ii] != 0.f) fprintf(stderr, " c%ld", ii);
      fprintf(stderr, " | row0[:8]=");
      for (long ii = 0; ii < 8; ii++) fprintf(stderr, "%.4g ", (double)h[ii]);
      fprintf(stderr, "\n");
    }
  }
  CUDA_CHECK(hipFree(xq81));
  // PATCH81: no f32 temp to free.
}

// P1 plan-B port (2026-11-16): plain (non-GLU) Q4_K-weight mmq launcher -
// the reference ggml R6 recipe (q8_1 x q8_1 INT8 WMMA, in-order K lanes,
// I=128/J=128/nthreads=256, smem 57856B) on in-model G64-affine Q4_K.
// Orientation = the GLU launcher minus fusion: x slot = the Q4_K weights
// (N rows), y slot = the Q8_1 activations (M rows, in-order DS4), dst
// row-major (M, N) [the dst_bf16 store - the in-model WMMA out is bf16].
// Per-call weight repack + activation quant (the established W1 GDN fix
// structure; 1,992 dispatches per 4K pass). The non-fallback config needs
// N % I == 0 (I = 128 for the J = 128 row; the dispatch gate enforces
// N % 128 == 0). M is clamped per-tile by the j_max writeback clamp.
// (namespace-scope, external linkage: called from qmm.hip's dispatch.)
// P1B20 (KPROF2 fix line, 2026-11-17): persistent mmq scratch cache.
// Mechanism (scratch path read first, per the fix-line gate): mmq_q4k_g64
// allocates+releases wq4 (N*K/256*320B repack buffer) and xq81 (M*K/128*144B
// activation staging buffer) on EVERY dispatch. 1992 dispatches per 4K pass:
//   - raw hipMalloc/hipFree (ALLOC=0): hipFree = whole-device sync = the 3.2x
//     e2e drag (GEMM-only ~12% of the raw wall, per KPROF2);
//   - pool malloc_async (default): the pool retains the high-water mark =
//     3/3 HBM-full on neon 16GB (18,874,368B request at the first prefill
//     eval, GPU idle, VRAM 0% between runs) + state-dependent OOM on helium.
// Fix: persistent raw-hipMalloc buffers keyed (N,K,stream)/(M,K,stream),
// LRU-evicted past the cap - no per-dispatch alloc, no free, outside the pool
// (no HWM retention). MLX_ROCM_MMQ_Q4K_SCRATCH=0 restores the legacy path.
// Plan-D (2026-11-18, 3.7K lane): per-shape cap on the repack content
// cache. MLX_ROCM_MMQ_REPACK_MAX_SHAPES = maximum number of cached
// repacked weight buffers (the P1B23 LRU/drop-oldest eviction already
// orders the entries; this adds a count bound). MLX_ROCM_MMQ_REPACK_MAX_BYTES
// = optional byte-budget override for the same cache (the t3309 default
// is the 2GB cap; the P1B23 8GB stays reachable via the override, e.g.
// =8589934592, for A/B). Both unset = the shrunk defaults (2GB byte cap,
// unlimited shapes). The P1B24 repack-ON steady retention is what the cap
// bounds (~max_shapes x per-weight bytes, drop-oldest).
static size_t mmq_repack_max_shapes() {
  static const size_t v = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_REPACK_MAX_SHAPES");
    if (!e) {
      return static_cast<size_t>(0);
    }
    long v = std::atol(e);
    return v > 0 ? static_cast<size_t>(v) : static_cast<size_t>(0);
  }();
  return v;
}
static size_t mmq_repack_max_bytes() {
  static const size_t v = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_REPACK_MAX_BYTES");
    if (!e) {
      return static_cast<size_t>(0);
    }
    // F1 fix (t3171 review): plain atol only accepted a bare integer; add
    // case-insensitive SI (1000-base) k/m/g suffixes so production arms can
    // use a suffixed byte cap (8g = 8*1000^3, 500M = 500*1000^2). Unknown
    // suffix or non-numeric = 0 (the legacy default path).
    char* endp = nullptr;
    unsigned long val = std::strtoul(e, &endp, 10);
    if (endp != nullptr && *endp != '\0') {
      switch (*endp) {
        case 'k':
        case 'K':
          val *= 1000UL;
          break;
        case 'm':
        case 'M':
          val *= 1000UL * 1000UL;
          break;
        case 'g':
        case 'G':
          val *= 1000UL * 1000UL * 1000UL;
          break;
        default:
          return static_cast<size_t>(0);
      }
    }
    return val > 0 ? static_cast<size_t>(val) : static_cast<size_t>(0);
  }();
  return v;
}
// t3309 (t3307 card lever (b)): the g_xq81_cache byte cap, env-overridable
// (the pre-shrink 1GB stays reachable via =1073741824). Same SI (1000-
// base) suffix parsing as mmq_repack_max_bytes (F1 fix, t3171 review).
static size_t mmq_xq81_max_bytes() {
  static const size_t v = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_XQ81_MAX_BYTES");
    if (!e) {
      return static_cast<size_t>(0);
    }
    char* endp = nullptr;
    unsigned long val = std::strtoul(e, &endp, 10);
    if (endp != nullptr && *endp != '\0') {
      switch (*endp) {
        case 'k':
        case 'K':
          val *= 1000UL;
          break;
        case 'm':
        case 'M':
          val *= 1000UL * 1000UL;
          break;
        case 'g':
        case 'G':
          val *= 1000UL * 1000UL * 1000UL;
          break;
        default:
          return static_cast<size_t>(0);
      }
    }
    return val > 0 ? static_cast<size_t>(val) : static_cast<size_t>(0);
  }();
  return v;
}
struct mmq_scratch_cache {
  std::mutex mtx;
  std::unordered_map<uint64_t, std::pair<void*, size_t>> m;
  std::deque<uint64_t> lru;  // back = MRU
  size_t total = 0;
  size_t cap;
  // Plan-D (2026-11-18, 3.7K lane): 0 = unlimited (legacy); otherwise
  // the maximum number of cached shapes (eviction drops the LRU-oldest
  // entry first, i.e. the deque front).
  size_t max_entries;
  explicit mmq_scratch_cache(size_t c, size_t me = 0) : cap(c), max_entries(me) {}
};
// T3322 (W3.1.1, stamp adjudication cloudy 2026-11-21): the card-class
// detection hook (the HIP device-query size class; read through the
// existing ggml_cuda_info() shim - HIP_VISIBLE_DEVICES-aware, device 0
// = the visible card; the same hipGetDeviceProperties family as
// device.cpp/allocator.cpp). <20GB class (the neon 16GB) -> the
// HOST-PINNED scratch + bounded P1B23 (the 265055a pre-fix shape: the
// kernel-facing scratch VA is pinned host memory, bounded 8GB/1GB,
// release OFF). >=20GB (the 20-32 middle + the helium 32GB) -> the
// DEVICE scratch + the W4 2GB/256MB shape (the official 293.3/8.15
// vehicle; the t3305 device pointer, release OFF). The A/B override
// MLX_ROCM_MMQ_SCRATCH_LOC=host|device arms a specific path (the gate
// use; default = the card class, arm-neutral, no env needed).
static bool mmq_scratch_host() {
  static const bool host = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_SCRATCH_LOC");
    if (e != nullptr) {
      if (e[0] == 'h' || e[0] == 'H') return true;
      if (e[0] == 'd' || e[0] == 'D') return false;
    }
    const unsigned long long gb =
        (unsigned long long)(ggml_cuda_info().devices[0].total_vram >> 30);
    return gb < 20;
  }();
  return host;
}
// T3322 (W3.1.1): the card-class-matched scratch alloc/free. Host class:
// pinned host (hipHostMalloc/hipFreeHost - NOT the t3116 sync class,
// which is the device hipFree whole-device sync; hipFreeHost does not
// sync the device). Device class: raw HBM (hipMalloc/hipFree), the t3305
// shape.
static bool mmq_scratch_alloc(void** p, size_t bytes) {
  if (mmq_scratch_host())
    return hipHostMalloc(p, bytes) == hipSuccess && *p != nullptr;
  return hipMalloc(p, bytes) == hipSuccess && *p != nullptr;
}
static void mmq_scratch_free(void* p) {
  if (p == nullptr) return;
  if (mmq_scratch_host())
    CUDA_CHECK(hipFreeHost(p));
  else
    CUDA_CHECK(hipFree(p));
}
// T3322 (W3.1.1, stamp adjudication): the PER-BOX DEFAULT cap. The env
// hook (unchanged, the A/B reference) wins when set; otherwise the card
// class selects: host class (card <20GB) -> the 265055a bounded P1B23
// 8GB (wq4) / 1GB (xq81); device class (card >=20GB) -> the W4 2GB /
// 256MB caps (the official 293.3/8.15 vehicle). The W3.1 8GB/1GB and
// W4 2GB/256MB values stay reachable via MLX_ROCM_MMQ_REPACK_MAX_BYTES /
// MLX_ROCM_MMQ_XQ81_MAX_BYTES as before (SI parse unchanged). The shape
// count honors MLX_ROCM_MMQ_REPACK_MAX_SHAPES (default: unlimited).
static size_t mmq_wq4_default_cap() {
  return mmq_scratch_host() ? (8ULL << 30) : (2ULL << 30);
}
static size_t mmq_xq81_default_cap() {
  return mmq_scratch_host() ? (1ULL << 30) : (268435456);
}
static mmq_scratch_cache g_wq4_cache(
    mmq_repack_max_bytes() ? mmq_repack_max_bytes() : mmq_wq4_default_cap(),
    mmq_repack_max_shapes());
static mmq_scratch_cache g_xq81_cache(
    mmq_xq81_max_bytes() ? mmq_xq81_max_bytes() : mmq_xq81_default_cap());
// GEMM-1 3b-2' A/B v2: file-order (K, N/256, 144) transpose-requant cache.
static mmq_scratch_cache g_bfile_cache(8ULL << 30);   // 8GB
static bool mmq_scratch_enabled() {
  static const bool en = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_Q4K_SCRATCH");
    return !(e != nullptr && e[0] == '0');
  }();
  return en;
}
static uint64_t mmq_key3(int a, int b, hipStream_t st) {
  uint64_t h = (uint64_t)(uint32_t)a * 1315423911ull;
  h ^= (uint64_t)(uint32_t)b * 2654435761ull;
  h ^= (uint64_t)(uintptr_t)st * 2246822519ull;
  return h;
}
// P1B23 (repack content cache, 2026-11-17, cloudy ruling 21:3xZ): the
// repacked q4_K buffer is a pure function of the weight identity
// (w, scales, biases, N, K, dtype). Key the wq4 buffer by that identity and
// skip the mmq_repack_g64_q4k launch on a content-valid hit. P1B20's
// (N,K,stream) key shared one buffer across all weights of equal shape, so
// the content was stale across weights and the repack ran every call (1608
// per 4K pass = 37.1% of the measure wall, P1B22). Cost: one repack buffer
// per unique weight = the full repack set resident (~5.5GB class;
// HBM-fit pre-registered: CH512 arm < 1200 t/s = LRU/fit class named with
// raw bytes; CH1024 HBM-full death = THE LINE). The 8GB LRU cap holds the
// full set for qwen3-9b. MLX_ROCM_MMQ_Q4K_REPACK=0 restores the P1B20 key
// + always-repack.
static bool mmq_repack_cache_enabled() {
  static const bool en = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_Q4K_REPACK");
    return !(e != nullptr && e[0] == '0');
  }();
  return en;
}
static uint64_t mmq_wkey(const void* w, const void* sc, const void* bi, int N,
                         int K, hipStream_t st, bool bf16) {
  uint64_t h = (uint64_t)(uintptr_t)w * 1315423911ull;
  h ^= (uint64_t)(uintptr_t)sc * 2654435761ull;
  h ^= (uint64_t)(uintptr_t)bi * 97531ull;
  h ^= (uint64_t)(uint32_t)N * 1597334671ull;
  h ^= (uint64_t)(uint32_t)K * 3938420197ull;
  h ^= (uint64_t)(uintptr_t)st * 2246822519ull ^
       (bf16 ? 0xA5A5A5ull : 0x5A5A5Aull);
  return h;
}
// Content-valid peek: existing same-size buffer for key -> ptr + LRU
// promotion; miss or size mismatch -> nullptr (no allocation, no mutation
// beyond LRU order).
static void* mmq_scratch_peek(mmq_scratch_cache& c, uint64_t key, size_t need) {
  if (!mmq_scratch_enabled()) return nullptr;
  std::lock_guard<std::mutex> lk(c.mtx);
  auto it = c.m.find(key);
  if (it == c.m.end() || it->second.second < need) return nullptr;
  auto pos = std::find(c.lru.begin(), c.lru.end(), key);
  if (pos != c.lru.end()) { c.lru.erase(pos); c.lru.push_back(key); }
  return it->second.first;
}
// Hit (or fresh/evicting fill) -> true + *out set. Miss -> false, *out untouched.
static bool mmq_scratch_try(mmq_scratch_cache& c, uint64_t key, size_t need, void** out) {
  if (!mmq_scratch_enabled()) return false;
  std::lock_guard<std::mutex> lk(c.mtx);
  auto it = c.m.find(key);
  if (it != c.m.end() && it->second.second >= need) {
    auto pos = std::find(c.lru.begin(), c.lru.end(), key);
    if (pos != c.lru.end()) { c.lru.erase(pos); c.lru.push_back(key); }
    *out = it->second.first;
    return true;
  }
  if (it != c.m.end()) {  // grow in place
    mmq_scratch_free(it->second.first);
    c.total -= it->second.second;
    c.m.erase(it);
  }
  // Plan-D: the per-shape cap evicts the LRU-oldest entries first
  // (drop-oldest = deque front) when the cache holds max_entries shapes.
  while (!c.lru.empty() &&
         (c.total + need > c.cap ||
          (c.max_entries > 0 && c.lru.size() >= c.max_entries))) {
    uint64_t old = c.lru.front();
    c.lru.pop_front();
    auto oit = c.m.find(old);
    if (oit == c.m.end()) continue;
    mmq_scratch_free(oit->second.first);
    c.total -= oit->second.second;
    c.m.erase(oit);
  }
  void* p = nullptr;
  if (!mmq_scratch_alloc(&p, need)) return false;
  c.m[key] = {p, need};
  c.total += need;
  c.lru.push_back(key);
  *out = p;
  return true;
}
// T3317 (W3.1, pin ruling 2026-11-21; T3322 W3.1.1: release-ON GATED, see
// mmq_release_per_chunk below): per-chunk release of the
// device-resident mmq scratch (the g_wq4_cache + g_xq81_cache raw
// hipMalloc buffers) at the end of each prefill chunk. The persistent
// P1B20/P1B23 caches + the LRU evict-oldest structure + the kqfg shim
// stay; the release empties both caches once per chunk. Trigger = the
// weight-identity sequence wrap (the first per-chunk weight re-touched),
// prefill chunks only (M >= 256 - the harness prefill-chunk class, 256/
// 512; the single 4K forward touches each weight once = no wrap = no
// release; decode M < 256 is excluded), the identity-keyed repack-cache
// path only (REPACK=0's shared (N,K) key repeats within a chunk - the
// wrap would misfire; the release stays off there).
// DESIGN (the t3116 sync-class gate): the release adds NO per-dispatch
// syncs - it runs ONCE per chunk: a single hipDeviceSynchronize (drain
// the chunk) + a hipFree loop over the released entries (device idle, the
// internal free-syncs are cheap). Per-chunk is ~1/8 the t3116 1992-syncs
// per 4K pass count class, bounded + A/B measurable:
// MLX_ROCM_MMQ_RELEASE_PER_CHUNK=0 restores the persistent-only behavior
// (the A/B reference). The W4 caps stay reachable via the REPACK/XQ81
// env hooks.
static bool mmq_release_per_chunk() {
  static const bool en = [] {
    // T3322 (W3.1.1, stamp adjudication): release-ON is GATED - OFF by
    // default on every box. The host path (card <20GB, the 265055a
    // pre-fix shape) is release-OFF regardless of the env. The device
    // path arms on the explicit MLX_ROCM_MMQ_RELEASE_PER_CHUNK=1 only
    // (W3.1's default-ON is withdrawn; the use is gated on the OOB fix
    // + the both-box token checks). =0 = the explicit persistent A/B.
    if (mmq_scratch_host()) return false;
    const char* e = std::getenv("MLX_ROCM_MMQ_RELEASE_PER_CHUNK");
    return e != nullptr && e[0] == '1';
  }();
  return en;
}
struct mmq_chunk_epoch {
  bool active = false;
  int m = 0;
  uint64_t first_key = 0;
};
static std::mutex g_epoch_mtx;
static std::unordered_map<void*, mmq_chunk_epoch> g_epochs;
// T3317: the per-chunk release. One hipDeviceSynchronize (the drain; the
// ONLY explicit sync) + a hipFree loop (device idle = the internal
// free-syncs are cheap, not the t3116 per-dispatch stall class). Emits the
// proof hook under MLX_MMQ_CFG (the [pcr] line: one per chunk, nbuf = the
// released entry count, syncs=1).
static void mmq_release_scratch(int* nbuf_out) {
  int nbuf = 0;
  hipError_t rc = hipDeviceSynchronize();
  if (rc != hipSuccess)
    fprintf(stderr, "[pcr] drain rc=%d\n", (int)rc);
  auto release = [&](mmq_scratch_cache& c) {
    std::lock_guard<std::mutex> lk(c.mtx);
    for (auto& kv : c.m) {
      mmq_scratch_free(kv.second.first);
      ++nbuf;
    }
    c.m.clear();
    c.lru.clear();
    c.total = 0;
  };
  release(g_wq4_cache);
  release(g_xq81_cache);
  if (nbuf_out) *nbuf_out = nbuf;
  if (const char* cje = std::getenv("MLX_MMQ_CFG"))
    fprintf(stderr, "[pcr] chunk_release nbuf=%d syncs=1\n", nbuf);
}
static void mmq_chunk_epoch_step(hipStream_t stream, int M, uint64_t key) {
  std::lock_guard<std::mutex> lk(g_epoch_mtx);
  auto& e = g_epochs[(void*)stream];
  if (!e.active) {
    e.active = true;
    e.m = M;
    e.first_key = key;
  } else if (e.m == M && key == e.first_key) {
    e.active = false;
    e.first_key = 0;
    mmq_release_scratch(nullptr);
  } else if (e.m != M) {
    // M changed (the chunk size switched, e.g. a single 4K forward before
    // the chunked prefill): restart the epoch, no release (the prior
    // epoch never completed its wrap).
    e.active = true;
    e.m = M;
    e.first_key = key;
  }
}
// T3317: the prefill->decode boundary release. The chunk-wrap release
// fires at the START of the next prefill chunk, so the LAST chunk's
// scratch would stay resident into the decode phase (the w31sync gate:
// gdn_fused_step_tape launch failed at decode under the HBM wall while
// the W4 2GB-cap control decoded fine - the 8GB wq4 + 1GB xq81 scratch
// of the last chunk plus the 14.3GB weights plus the 4K KV exceeded
// HBM). The design intent of the per-chunk release is that the scratch
// is GONE by decode time: the first M < 256 dispatch (decode, M=1..8
// incl. the MTP K=3 verify M=4) after an active prefill epoch releases
// the scratch at the prefill->decode boundary. Same sync class as the
// chunk-wrap (one hipDeviceSynchronize + hipFree loop, not per
// dispatch). Cold decode (no prior prefill epoch) is a no-op.
static void mmq_prefill_to_decode_step(hipStream_t stream) {
  std::lock_guard<std::mutex> lk(g_epoch_mtx);
  auto& e = g_epochs[(void*)stream];
  if (e.active) {
    e.active = false;
    e.first_key = 0;
    mmq_release_scratch(nullptr);
  }
}
// T3317: the mx.clear_cache() hook (allocator.cpp). The gate harness
// clears before decode (T3298_CLEAR_BEFORE_DECODE), and the GDN fused
// step tape is the FIRST decode op - a first-decode-dispatch release
// (mmq_prefill_to_decode_step) is too late (w31sync dev2: the tape
// launch failed under the HBM wall, 99% VRAM peak vs the W4 2GB-cap
// control's 81%): the LAST prefill chunk's 8GB wq4 + 1GB xq81 scratch
// must be GONE before the tape launches. When RELEASE_PER_CHUNK is on,
// clear_cache() releases the raw scratch + resets every chunk epoch.
// When =0 (the A/B reference) clear_cache() keeps the W4 behavior
// (persistent caches; only the driver-pool trim).
void mmq_clear_scratch_on_cache_clear() {
  if (!mmq_release_per_chunk())
    return;
  std::lock_guard<std::mutex> lk(g_epoch_mtx);
  for (auto& kv : g_epochs)
    kv.second.active = false;
  mmq_release_scratch(nullptr);
}
// GEMM-1 3b-2' A/B hook entry (defined in kquant_file_gemm.hip;
// cloudy ruling 3b-2', 2026-11-18).
// GEMM-1 3b-2' TRIO: the six in-model weight classes (M=512 basis;
// LEG2 cfgq4k composition). Representative dumps: one tensor per class.
static inline bool mmq_ab_class(int N, int K) {
  return (N == 1024 && K == 4096) || (N == 4096 && K == 4096) ||
         (N == 8192 && K == 4096) || (N == 12288 && K == 4096) ||
         (N == 4096 && K == 12288) || (N == 248320 && K == 4096);
}
static inline int mmq_ab_class_idx(int N, int K) {
  if (N == 1024 && K == 4096) return 0;
  if (N == 4096 && K == 4096) return 1;
  if (N == 8192 && K == 4096) return 2;
  if (N == 12288 && K == 4096) return 3;
  if (N == 4096 && K == 12288) return 4;
  if (N == 248320 && K == 4096) return 5;
  return -1;
}
extern "C" bool kqfg_ab_gemm(const void* x, bool x_bf16, const uint8_t* w,
                             void* out, bool dst_f32, int M, int K, int NBS,
                             hipStream_t stream);
void mmq_q4k_g64(const void* x, bool x_bf16, const void* w,
                       const void* scales, const void* biases, void* out,
                       bool dst_f32, int M, int N, int K,
                       hipStream_t stream) {
  ggml_backend_cuda_context ctx(0);
  const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
  const size_t smpbo = ggml_cuda_info().devices[0].smpbo;

  const int64_t w_total256 = (int64_t)N * K / 256;  // q4_K blocks per weight
  const int64_t x_tiles = (int64_t)M * (K / 128);
  // GEMM-1 3b-2' A/B hook v2 (jimmy 2026-11-19): MLX_ROCM_MMQ_Q4K_KQFILE=1
  // -> swap the GEMM op ONLY. The W1 kqfg op needs FILE-order 144B blocks
  // (256 consecutive n at fixed k); the G64 storage is (N, K) K-fastest
  // (layout test: storage-order cos 0.998 vs file-order cos -0.000), so a
  // transpose-requant fill (kqfg_ab_transpose_requant) produces the
  // file-order buffer, cached per weight identity (LRU 8GB, same
  // structure as the repack cache; one-time fill lands in the WARM pass)
  // and feeds the W1 kernel (k-order f32 accumulation, in-kernel dequant,
  // in-kernel RNE output cast). The wq4 repack + xq81 quant + the ported
  // mmq are all skipped in this branch (HBM class <= the control arm).
  // Default off = bit-identical mmq path (the A/B control self-check vs
  // leg-2 CTRL).
  static const bool kqfile_env = [] {
    const char* e = std::getenv("MLX_ROCM_MMQ_Q4K_KQFILE");
    return e != nullptr && e[0] == '1';
  }();
  if (kqfile_env) {
    const size_t need_bf = (size_t)w_total256 * 144;
    const uint64_t bf_key = mmq_wkey(w, scales, biases, N, K, stream, x_bf16);
    const bool bf_valid =
        mmq_scratch_peek(g_bfile_cache, bf_key, need_bf) != nullptr;
    void* bfile = nullptr;
    if (!mmq_scratch_try(g_bfile_cache, bf_key, need_bf, &bfile) || !bfile) {
      fprintf(stderr, "[kqfg-ab] bfile alloc fail M=%d N=%d K=%d\n", M, N, K);
      return;
    }
    if (!bf_valid) {
      dim3 gridbf(N / 256, K, 1);
      if (x_bf16) {
        kqfg_ab_transpose_requant<hip_bfloat16><<<gridbf, 256, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(w),
            reinterpret_cast<const hip_bfloat16*>(scales),
            reinterpret_cast<const hip_bfloat16*>(biases),
            reinterpret_cast<block_q4_K*>(bfile), N, K);
      } else {
        kqfg_ab_transpose_requant<float><<<gridbf, 256, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(w),
            reinterpret_cast<const float*>(scales),
            reinterpret_cast<const float*>(biases),
            reinterpret_cast<block_q4_K*>(bfile), N, K);
      }
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[kqfg-ab] bfile fill rc=%d M=%d N=%d K=%d\n",
                (int)hipGetLastError(), M, N, K);
        return;
      }
      // KQFG-AB-DUMP (v4/TRIO): one-time bfile byte dump per 6-class
      // representative (S2 input) for CPU verification of the
      // transpose-requant kernel.
      // Curation (persona scrub): the dump dir is env-var-driven - the
      // KQFG_AB_DUMP value is the dir when path-like ("/" or "."-relative),
      // the CWD otherwise (value = 1 or any bare token arms it).
      const char* kqdump_env = std::getenv("KQFG_AB_DUMP");
      if (kqdump_env != nullptr && mmq_ab_class(N, K)) {
        static bool bfdumped[6] = {false, false, false, false, false, false};
        int ci = mmq_ab_class_idx(N, K);
        if (ci >= 0 && !bfdumped[ci]) {
          bfdumped[ci] = true;
          void* hd = nullptr;
          if (hipMallocHost(&hd, need_bf) == hipSuccess && hd) {
            hipMemcpyAsync(hd, bfile, need_bf, hipMemcpyDeviceToHost, stream);
            if (hipStreamSynchronize(stream) == hipSuccess) {
              char fn[256];
              const char* dump_dir = (kqdump_env[0] == '/' || kqdump_env[0] == '.') ? kqdump_env : ".";
              snprintf(fn, sizeof(fn), "%s/bfiledump-N%dk%d.bin", dump_dir, N, K);
              FILE* f = fopen(fn, "wb");
              if (f) {
                fwrite(hd, 1, need_bf, f);
                fclose(f);
              }
              fprintf(stderr, "[kqfg-ab] bfile dumped %zu bytes N=%d K=%d\n",
                      need_bf, N, K);
            }
            hipFreeHost(hd);
          }
        }
      }
    }
    const bool ok = kqfg_ab_gemm(x, x_bf16, (const uint8_t*)bfile, out,
                                  dst_f32, M, K, (int)(N / 256), stream);
    if (!ok) {
      fprintf(stderr, "[kqfg-ab] launch fail M=%d N=%d K=%d\n", M, N, K);
    }
    return;
  }


  // P1B4-SCRATCH-FIX (2026-11-17, RCA t3116 accepted 05:53Z/05:56Z, op
  // order 06:1xZ): route the per-dispatch scratch through the hardened
  // rocm allocator pool. Raw hipMalloc/hipFree per call bypassed the pool
  // (hipFree = whole-device sync; HSA async-signal pool edge under churn);
  // 1992 dispatches per 4K pass stalled the in-model P arm (GPU 41 MHz
  // idle, rc=124). MLX_ROCM_MMQ_Q4K_ALLOC=0 restores raw hipMalloc (A/B).
  static const int alloc_raw = [] {
    const char* ae = std::getenv("MLX_ROCM_MMQ_Q4K_ALLOC");
    return (ae != nullptr && atoi(ae) == 0) ? 1 : 0;
  }();
  // T3326 (W3.1.2, cloudy ruling 2026-11-21 ~11:5xZ, msg 6xf2zr3XzeQY):
  // the card <20GB host-class miss path is RESTORED to the f5e7cd88
  // allocator-pool shape - the per-call hipHostMalloc/hipFreeHost branch
  // (T3322 W3.1.1) is removed, so the host class falls into the pool
  // else branch (Buffer pair + malloc_async + raw pointer; scope-end
  // RAII release; no per-call hipFreeHost). The device-class miss path
  // is byte-identical (the 27B fit fix stands). The class-matched caps,
  // the [das] emit, and the release policy are unchanged. alloc_raw (the
  // explicit Q4K_ALLOC=0 A/B) stays the raw device hipMalloc on every
  // card.
  // T3322 (W3.1.1): the scratch-location policy + per-box default proof
  // hook - one [das] line under MLX_MMQ_CFG on the first dispatch (the
  // class + the default caps + the release state + the card VRAM; the
  // canary is T3322-DAS-20261121).
  {
    static bool das_emitted = false;
    if (!das_emitted && std::getenv("MLX_MMQ_CFG")) {
      das_emitted = true;
      fprintf(stderr,
              "[das] scratch=%s cap_wq4=%zu cap_xq81=%zu release=%s "
              "vram_gib=%llu (T3322-W3111-DAS)\n",
              mmq_scratch_host() ? "host" : "device", g_wq4_cache.cap,
              g_xq81_cache.cap, mmq_release_per_chunk() ? "on" : "off",
              (unsigned long long)(ggml_cuda_info().devices[0].total_vram >> 30));
    }
  }
  mlx::core::allocator::Buffer wq4_b{nullptr}, xq81_b{nullptr};
  void* wq4 = nullptr;
  void* xq81 = nullptr;
  // P1B20: cache hit -> no per-dispatch alloc, no free at scope end. Miss -> legacy path.
  // P1B23: wq4 key = weight identity when the repack cache is enabled; the
  // cached buffer's CONTENT is then valid and the repack kernel is skipped
  // below. REPACK=0 -> P1B20 (N,K) key, always repack (exact old behavior).
  const bool repack_cache = mmq_repack_cache_enabled();
  const uint64_t wq4_key = repack_cache
                               ? mmq_wkey(w, scales, biases, N, K, stream, x_bf16)
                               : mmq_key3(N, K, stream);
  const size_t need_wq4 = (size_t)w_total256 * sizeof(block_q4_K);
  // T3322 (W3.1.1, the release-path OOB fix, the stamp adjudication):
  // the release trigger runs BEFORE the validity peek. Free-after-last-use:
  // the release's drain sync (hipDeviceSynchronize) completes every prior
  // consumer of the released buffers before the hipFree loop, and the
  // post-release peek sees the emptied cache, so the repack runs on the
  // fresh buffer before the GEMM consumes it. The W3.1 order bug: the
  // peek was pre-release, so a wrap-trigger dispatch read a stale
  // wq4_valid and skipped the repack on the freshly-allocated
  // (uninitialized) buffer - the 9B neon OOB warm-argmax 248044 (all 4
  // DEFAULT stamp rounds; the 27B battery was rate-data-only). No
  // per-dispatch syncs: the release fires only on the sequence wrap.
  if (mmq_release_per_chunk() && repack_cache) {
    if (M >= 256)
      mmq_chunk_epoch_step(stream, (int)M, wq4_key);
    else
      mmq_prefill_to_decode_step(stream);
  }
  const bool wq4_valid = repack_cache &&
                         (mmq_scratch_peek(g_wq4_cache, wq4_key, need_wq4) != nullptr);
  const bool from_cache =
      mmq_scratch_try(g_wq4_cache, wq4_key, need_wq4, &wq4) &&
      mmq_scratch_try(g_xq81_cache, mmq_key3(M, K, stream),
                      (size_t)x_tiles * 144, &xq81);
  if (!from_cache) {
    if (alloc_raw) {
      CUDA_CHECK(hipMalloc(&wq4, (size_t)w_total256 * sizeof(block_q4_K)));
      CUDA_CHECK(hipMalloc(&xq81, (size_t)x_tiles * 144));
    } else {
      wq4_b = mlx::core::rocm::allocator().malloc_async(
          (size_t)w_total256 * sizeof(block_q4_K), 0, (void*)stream);
      xq81_b = mlx::core::rocm::allocator().malloc_async((size_t)x_tiles * 144, 0, (void*)stream);
      // P1B4-DEVICE-SCRATCH (t3305, root-cause fix for the t3304 host-wall):
      // hand the kernels the DEVICE (HBM) pointer. wq4/xq81 are GPU-written
      // + GPU-read and are never CPU-read, so raw_ptr() was semantically
      // wrong here: it staged a full-size pinned host shadow per scratch
      // buffer (hipDeviceSynchronize + D2H round-trip per dispatch, shadows
      // persisting in the pool). The device pointer removes the pinned
      // shadows (the ~26G+ pinned wall), the D2H/H2D round-trips, and the
      // per-dispatch syncs (the ~8% sync-cost class) in one change.
      // alloc_raw (MLX_ROCM_MMQ_Q4K_ALLOC=0) = the raw hipMalloc/hipFree
      // A/B reference (kept, NOT the default - hipFree = whole-device sync,
      // t3116).
      wq4 = static_cast<mlx::core::rocm::RocmBuffer*>(wq4_b.ptr())->data;
      xq81 = static_cast<mlx::core::rocm::RocmBuffer*>(xq81_b.ptr())->data;
    }
  }

  const int tb = 256;
  // T3326 (W3.1.2): the fresh-scratch free handles alloc_raw only (the
  // raw device hipFree A/B). The pool path frees via the RAII Buffers
  // at scope end (both classes).
  auto free_fresh = [&]() {
    if (alloc_raw) {
      CUDA_CHECK(hipFree(wq4));
      CUDA_CHECK(hipFree(xq81));
    }
  };
  if (std::getenv("MLX_MMQ_DBG")) {
    fprintf(stderr, "[mmq-dbg] M=%ld N=%ld K=%ld wq4=%p(%ldB) xq81=%p(%ldB) out=%p(%ldB) w=%p x=%p\n",
            (long)M, (long)N, (long)K, wq4, (long)w_total256*sizeof(block_q4_K),
            xq81, (long)x_tiles*144, out, (long)M*N*(dst_f32?4:2), w, x);
  }

  auto stage4k = [](const char* st, hipError_t rc) {
    if (rc != hipSuccess) fprintf(stderr, "[stage4k] %s rc=%d\n", st, (int)rc);
  };
  // P1B23: skip the repack when the cached wq4 content is valid for this
  // weight identity. !from_cache always implies a fresh/legacy wq4 buffer
  // (the legacy path reassigns wq4), so the repack must run then even if a
  // valid cached entry exists under wq4_key.
  if (!wq4_valid || !from_cache) {
    if (x_bf16) {
      mmq_repack_g64_q4k<hip_bfloat16><<<(int)w_total256, tb, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(w),
          reinterpret_cast<const hip_bfloat16*>(scales),
          reinterpret_cast<const hip_bfloat16*>(biases),
          reinterpret_cast<block_q4_K*>(wq4));
    } else {
      mmq_repack_g64_q4k<float><<<(int)w_total256, tb, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(w),
          reinterpret_cast<const float*>(scales),
          reinterpret_cast<const float*>(biases),
          reinterpret_cast<block_q4_K*>(wq4));
    }
  }
  stage4k("repack", hipGetLastError());
  // GEMM-1 3b-2' TRIO: one-time wq4 byte dump per 6-class representative
  // (MMQ_AB_DUMP=1) = S1 input for the standalone trio. Dump after the
  // repack stage: the buffer is content-valid (fresh or cache-hit).
  // Curation (persona scrub): the dump dir is env-var-driven (see the
  // KQFG_AB_DUMP block above for the value semantics).
  const char* wdump_env = std::getenv("MMQ_AB_DUMP");
  if (wdump_env != nullptr && mmq_ab_class(N, K)) {
    static bool wdumped[6] = {false, false, false, false, false, false};
    int ci = mmq_ab_class_idx(N, K);
    if (ci >= 0 && !wdumped[ci]) {
      wdumped[ci] = true;
      void* hd = nullptr;
      if (hipMallocHost(&hd, need_wq4) == hipSuccess && hd) {
        hipMemcpyAsync(hd, wq4, need_wq4, hipMemcpyDeviceToHost, stream);
        if (hipStreamSynchronize(stream) == hipSuccess) {
          char fn[256];
          const char* dump_dir = (wdump_env[0] == '/' || wdump_env[0] == '.') ? wdump_env : ".";
          snprintf(fn, sizeof(fn), "%s/wq4dump-N%dk%d.bin", dump_dir, N, K);
          FILE* f = fopen(fn, "wb");
          if (f) {
            fwrite(hd, 1, need_wq4, f);
            fclose(f);
          }
          fprintf(stderr, "[mmq-ab] wq4 dumped %zu bytes N=%d K=%d\n",
                  need_wq4, N, K);
        }
        hipFreeHost(hd);
      }
    }
  }
  if (x_bf16) {
    mmq_quant_in_q8_1_mmq<hip_bfloat16><<<(x_tiles + tb - 1) / tb, tb, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(x),
        reinterpret_cast<char*>(xq81), M, K, x_tiles, /*ds4=*/true);
  } else {
    mmq_quant_in_q8_1_mmq<float><<<(x_tiles + tb - 1) / tb, tb, 0, stream>>>(
        reinterpret_cast<const float*>(x),
        reinterpret_cast<char*>(xq81), M, K, x_tiles, /*ds4=*/true);
  }
  stage4k("xpreq", hipGetLastError());

  int nthreads_best = 0;
  // J is the y-slot tile = the activations (M rows).
  int J_best = ggml_cuda_mmq_get_J_best(GGML_TYPE_Q4_K, M, false, cc,
                                       smpbo, GGML_PREC_Q8, &nthreads_best);
  if (const char* je = std::getenv("MLX_MMQ_J")) {
    int jf = std::atoi(je);
    if (jf >= 8 && jf <= 128 && jf % 8 == 0) J_best = jf;
  }
  const int I_val = ggml_cuda_mmq_get_I(GGML_TYPE_Q4_K, J_best, false, cc,
                                       GGML_PREC_Q8);
  // The non-fallback config: full I-tiles of the N (x slot) rows.
  if (J_best < 8 || N % I_val != 0) {
    if (alloc_raw && !from_cache) free_fresh();
    return;  // pool path: RAII release at scope end; P1B20 cached buffers persist
  }
  if (const char* cje = std::getenv("MLX_MMQ_CFG")) {
    const ggml_cuda_mmq_config c =
        ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, J_best, false, cc,
                                GGML_PREC_Q8);
    fprintf(stderr, "[cfgq4k] M=%d N=%d K=%d cc=%d J=%d nthreads=%d cfg.I=%d cfg.nthreads=%d nbytes=%zu nty=%d ntx=%d\n",
            M, N, K, cc, J_best, nthreads_best, c.I, c.nthreads,
            mmq_get_nbytes_shared(c, cc), (int)((M + J_best - 1) / J_best),
            (int)((N + c.I - 1) / c.I));
  }
  float* dst = reinterpret_cast<float*>(out);
  mmq_args args = {reinterpret_cast<const char*>(wq4), GGML_TYPE_Q4_K,
                   reinterpret_cast<const int*>(xq81), nullptr, nullptr,
                   dst, nullptr, K, N, M, K / 256, M, N,
                   1, 1, 0, 0, 0,
                   1, 1, 0, 0, 0,
                   M, J_best, {}};
  args.dst_bf16 = !dst_f32;
  mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
  hipError_t rc = hipGetLastError();
  if (rc != hipSuccess) {
    fprintf(stderr, "[stage4k] mmq rc=%d\n", (int)rc);
    CUDA_CHECK(hipGetLastError());
  }
  stage4k("mmq", hipGetLastError());
  if (alloc_raw && !from_cache) free_fresh();
  // pool path: the Buffers release to the allocator at scope end
  // P1B20: cached scratch persists across dispatches (no free)
}

// GEMM-1 3b-2' STANDALONE TRIO entry (jimmy, cloudy RULING v3 2026-11-18
// 17:4xZ): raw GEMM on caller-provided weight bytes (wq4 = S1 our
// orientation, bfile = S2 file orientation) through the RECORD launcher
// (J128 / CFG path) + the xq81 quant. The host orientation (ne0, ne1, nb0)
// is the only variable: S1 (ne0=K, ne1=N, nb0=K/256) vs S2 (ne0=N, ne1=K,
// nb0=N/256) on the same weight bytes. No repack, no weight cache; xq81
// staged via the persistent g_xq81_cache (record behavior: first dispatch
// per (M,K,stream) fills, the rest hit). [abq4k] log under MLX_MMQ_CFG.
// Returns false on the config fallback or a launch error.
static volatile char MMQ_AB_TRIO_CANARY[] = "MMQ-ABTRIO-20261119";
extern "C" const volatile char* mlx_rocm_mmq_abtrio_canary() {
  return MMQ_AB_TRIO_CANARY;
}
extern "C" bool mmq_ab_gemm(const void* x, bool x_bf16, const void* wbuf,
                            void* out, bool dst_f32, int M, int N, int K,
                            int ne0, int ne1, int nb0, hipStream_t stream) {
  ggml_backend_cuda_context ctx(0);
  const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
  const size_t smpbo = ggml_cuda_info().devices[0].smpbo;
  const int64_t x_tiles = (int64_t)M * (K / 128);
  mlx::core::allocator::Buffer xq81_b{nullptr};
  void* xq81 = nullptr;
  const bool from_cache =
      mmq_scratch_try(g_xq81_cache, mmq_key3(M, K, stream),
                      (size_t)x_tiles * 144, &xq81) &&
      xq81 != nullptr;
  if (!from_cache) {
    xq81_b = mlx::core::rocm::allocator().malloc_async(
        (size_t)x_tiles * 144, 0, (void*)stream);
    // P1B4-DEVICE-SCRATCH (t3305): xq81 is GPU-written (mmq_quant_in_q8_1_mmq)
    // + GPU-read and never CPU-read - raw_ptr() would stage a pinned host
    // shadow for this scratch (the t3304 host-wall lever). Device pointer.
    xq81 = static_cast<mlx::core::rocm::RocmBuffer*>(xq81_b.ptr())->data;
  }
  const int tb = 256;
  if (x_bf16) {
    mmq_quant_in_q8_1_mmq<hip_bfloat16><<<(x_tiles + tb - 1) / tb, tb, 0,
                                       stream>>>(
        reinterpret_cast<const hip_bfloat16*>(x),
        reinterpret_cast<char*>(xq81), M, K, x_tiles, /*ds4=*/true);
  } else {
    mmq_quant_in_q8_1_mmq<float><<<(x_tiles + tb - 1) / tb, tb, 0, stream>>>(
        reinterpret_cast<const float*>(x),
        reinterpret_cast<char*>(xq81), M, K, x_tiles, /*ds4=*/true);
  }
  if (hipGetLastError() != hipSuccess) return false;
  int nthreads_best = 0;
  int J_best = ggml_cuda_mmq_get_J_best(GGML_TYPE_Q4_K, M, false, cc, smpbo,
                                        GGML_PREC_Q8, &nthreads_best);
  if (const char* je = std::getenv("MLX_MMQ_J")) {
    int jf = std::atoi(je);
    if (jf >= 8 && jf <= 128 && jf % 8 == 0) J_best = jf;
  }
  const int I_val = ggml_cuda_mmq_get_I(GGML_TYPE_Q4_K, J_best, false, cc,
                                        GGML_PREC_Q8);
  if (J_best < 8 || N % I_val != 0) return false;
  if (const char* cje = std::getenv("MLX_MMQ_CFG")) {
    fprintf(stderr, "[abq4k] M=%d N=%d K=%d ne0=%d ne1=%d nb0=%d J=%d I=%d\n",
            M, N, K, ne0, ne1, nb0, J_best, I_val);
  }
  float* dst = reinterpret_cast<float*>(out);
  mmq_args args = {reinterpret_cast<const char*>(wbuf), GGML_TYPE_Q4_K,
                   reinterpret_cast<const int*>(xq81), nullptr, nullptr,
                   dst, nullptr, ne0, ne1, M, nb0, M, N,
                   1, 1, 0, 0, 0,
                   1, 1, 0, 0, 0,
                   M, J_best, {}};
  args.dst_bf16 = !dst_f32;
  mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
  return hipGetLastError() == hipSuccess;
}

// P1 plan-B build-hygiene canary (host odr-use in mmq-host.hip: volatile
// array, non-foldable load - the pattern of the B97T2FIX / G1 probes).
static volatile char P1B_MMQ_PORT_CANARY[] = "P1B4-STDQS-20261117";
extern "C" const volatile char* mlx_rocm_mmq_p1b_canary() {
  return P1B_MMQ_PORT_CANARY;
}
// P1B23 canary (repack content cache, 2026-11-17): host volatile array,
// odr-used; verify by strings|grep on the built lib.
static volatile char P1B23_REPACK_CANARY[] = "P1B23-REPACK-20261117";
extern "C" const volatile char* mlx_rocm_mmq_p1b23_canary() {
  return P1B23_REPACK_CANARY;
}
// Plan-D (2026-11-18, 3.7K lane): per-shape repack cap canary - verify by
// strings|grep on the built lib (harness-readable gate presence proof).
static volatile char PLAN_D_REPACKCAP_CANARY[] = "PLAND-REPACKCAP-20261118";
extern "C" const volatile char* mlx_rocm_plan_d_repackcap_canary() {
  return PLAN_D_REPACKCAP_CANARY;
}
// T3317 (W3.1): per-chunk release canary - verify by strings|grep on the
// built lib (the new release marker alongside REPACK_MAX_BYTES /
// XQ81_MAX_BYTES / Q4K_ALLOC). The MLX_MMQ_CFG [pcr] emit is the
// sync/dispatch proof hook (one line per chunk: nbuf + syncs=1).
static volatile char T3317_PCRL_CANARY[] = "T3317-PCREL-20261121";
extern "C" const volatile char* mlx_rocm_mmq_t3317_pcrel_canary() {
  return T3317_PCRL_CANARY;
}
// T3322 (W3.1.1, stamp adjudication): the device-aware scratch-location
// policy + per-box default canary (verify by strings|grep on the built
// lib; the T3317-PCREL class - host volatile array, non-foldable; the
// [das] emit under MLX_MMQ_CFG is the proof hook).
static volatile char T3322_DAS_CANARY[] = "T3322-DAS-20261121";
extern "C" const volatile char* mlx_rocm_mmq_t3322_das_canary() {
  return T3322_DAS_CANARY;
}
// T3322 (W3.1.1): the release-path OOB fix canary (free-after-last-use:
// the release trigger now runs BEFORE the wq4 validity peek, so a
// release-trigger dispatch re-repacks on the fresh buffer before the GEMM
// consumes it; the W3.1 order read the pre-release peek = a stale
// wq4_valid = a skipped repack on the uninitialized fresh buffer = the
// 9B neon OOB warm-argmax 248044).
static volatile char T3322_OOBFIX_CANARY[] = "T3322-OOBFIX-20261121";
extern "C" const volatile char* mlx_rocm_mmq_t3322_oobfix_canary() {
  return T3322_OOBFIX_CANARY;
}
// T3326 (W3.1.2, cloudy ruling 2026-11-21 ~11:5xZ, msg 6xf2zr3XzeQY):
// the card <20GB host-class miss-path RESTORE canary (the per-call
// hipHostMalloc/hipFreeHost branch is removed; the host class falls
// into the allocator-pool shape - Buffer pair + malloc_async + raw
// pointer, scope-end RAII release; the device-class miss path is
// byte-identical). Verify by strings|grep on the built lib (the marker
// alongside the T3322-DAS / T3322-OOBFIX / T3317-PCREL class).
static volatile char T3326_RESTORE_CANARY[] = "T3326-RESTORE-20261121";
extern "C" const volatile char* mlx_rocm_mmq_t3326_restore_canary() {
  return T3326_RESTORE_CANARY;
}

}  // namespace rocm

// MmulQGlu (R6 recipe item 1): the GLU-fused mmq variant.
namespace mlx::core::fast {

bool MmulQGlu::use_fallback(int M, int N, int K, Stream s) {
  if (s.device != Device::gpu) {
    return true;
  }
  static const bool env = [] {
    const char* e = std::getenv("MLX_GLU_MMQ");
    return e != nullptr && e[0] == '1';
  }();
  if (!env) {
    return true;
  }
  // the plain tcopy needs exact (M/32, N/32) grids; the mmq layouts need
  // K % 8 == 0 (G64 raw) and N % 128 == 0 (I_dst tiles, Q8_0 rows).
  if (K % 8 != 0 || N % 128 != 0 || M % 32 != 0 || M <= 8) {
    return true;
  }
  return false;
}

void MmulQGlu::eval_gpu(const std::vector<array>& inputs,
                        std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  auto& x = inputs[0];
  auto& wu = inputs[1];
  auto& su = inputs[2];
  auto& bu = inputs[3];
  auto& wg = inputs[4];
  auto& sg = inputs[5];
  auto& bg = inputs[6];
  auto& out = outputs[0];

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), enc));

  enc.set_input_array(x);
  enc.set_input_array(wu);
  enc.set_input_array(su);
  enc.set_input_array(bu);
  enc.set_input_array(wg);
  enc.set_input_array(sg);
  enc.set_input_array(bg);
  enc.set_output_array(out);

  const int K = (int)x.shape(-1);
  const int M = (int)x.shape(-2);
  const int N = (int)wu.shape(0);

  // Build 57: MLX_GLU_Q4K=1 -> the Q4_K weight path (the R6 recipe); the
  // G64->q4_K repack needs K % 256 == 0 (the q4_K block).
  static const bool q4k_env = [] {
    const char* e = std::getenv("MLX_GLU_Q4K");
    return e != nullptr && e[0] == '1';
  }();
  enc.launch_kernel([&](hipStream_t stream) {
    if (q4k_env && K % 256 == 0) {
      ::rocm::mmq_glu_q4k_g64(
          gpu_ptr<float>(x), gpu_ptr<const char>(wu), gpu_ptr<float>(su),
          gpu_ptr<float>(bu), gpu_ptr<const char>(wg), gpu_ptr<float>(sg),
          gpu_ptr<float>(bg), gpu_ptr<float>(out), true, M, N, K,
          static_cast<hipStream_t>(stream));
      return;
    }
    ::rocm::mmq_glu_g64(
        gpu_ptr<float>(x), gpu_ptr<const char>(wu), gpu_ptr<float>(su),
        gpu_ptr<float>(bu), gpu_ptr<const char>(wg), gpu_ptr<float>(sg),
        gpu_ptr<float>(bg), gpu_ptr<float>(out), true, M, N, K,
        static_cast<hipStream_t>(stream));
  });
}

}  // namespace mlx::core::fast
