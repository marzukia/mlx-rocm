// Copyright © 2025 Apple Inc.
//
// lm_head_int2: int2 row-quarter-split g128-asymmetric dequant GEMV (M=1)
// for the MTP draft head. Replaces the 4-bit lm_head (weight traffic per
// draft call) with a K/4-byte packed read.
//
// Packing (quarter-split, ported from radiance): for a row of K elements,
// packed byte p = j (j in 0..K/4-1) holds four 2-bit planes for x indices
// j, K/4+j, K/2+j, 3K/4+j. Element e lives in packed byte e mod (K/4),
// plane e / (K/4). Group G (0..K/128-1) owns elements [G*128, (G+1)*128):
// quarter q = G/8, local group gl = G%8, i.e. plane q of packed bytes
// [gl*128, ... +128). Quantization: per-group asymmetric, s =
// max((hi-lo)/3, 1e-8), z = clamp(round(-lo/s), 0, 3),
// q = clamp(round(w/s + z), 0, 3). The op takes the per-group
// coefficients scale_g = s, zs_g = z*s so the dequant
// algebra per group is out += dot_g * scale_g - sx_g * zs_g where
// dot_g = sum_k x_k * q_k and sx_g = sum_k x_k (host-precomputed).
//
// One thread per output column (N = 248320). After its column's logit,
// each 64-column subblock runs an 8-round shared-memory max selection
// and emits the per-block top-8 (value, col) candidate pairs so the
// caller can re-rank the global top-32 with the EXACT weights.
//
// The op is a plain add-on (no env gate needed for correctness; the MTP
// draft bench is the only caller). CPU: use_fallback -> throw (no chain
// fallback: the int2 packed weights only exist on the caller side).

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <math.h>

namespace mlx::core {
namespace rocm {

// u16 -> f32 (identical to the chain's bf2f; ROCm has no
// __bfloat162float intrinsic).
__device__ __forceinline__ float li2_bf2f16(unsigned short h) {
  union {
    unsigned short u;
    __hip_bfloat16 b;
  } c;
  c.u = h;
  return static_cast<float>(c.b);
}

__global__ void lm_head_int2_kernel(
    const __hip_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const float* __restrict__ scale,
    const float* __restrict__ zs,
    const float* __restrict__ sx,
    float* __restrict__ logits,
    float* __restrict__ cand_val,
    int* __restrict__ cand_col,
    int N,
    int K,
    int NG) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N)
    return;
  const int Q = K >> 2;      // quarter length in elements (K/4)
  const int GQ = Q >> 7;    // 128-groups per quarter
  float acc = 0.0f;
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    for (int gl = 0; gl < GQ; ++gl) {
      const int G = q * GQ + gl;
      const uint8_t* p = wq + (size_t)col * Q + (size_t)gl * 128;
      const __hip_bfloat16* xg = x + (size_t)q * Q + (size_t)gl * 128;
      float d = 0.0f;
      #pragma unroll 8
      for (int j = 0; j < 128; ++j) {
        unsigned b = p[j];
        d += li2_bf2f16(reinterpret_cast<const unsigned short*>(xg)[j]) *
             (float)((b >> (2 * q)) & 3);
      }
      acc += d * scale[(size_t)col * NG + G] - sx[G] * zs[(size_t)col * NG + G];
    }
  }
  float out = acc;
  logits[col] = out;

  // ---- per-64-column top-8 selection (shared-memory max rounds) ----
  __shared__ float sh[256];
  __shared__ int shi[256];
  __shared__ float wmax[8];
  __shared__ int wmaxi[8];
  __shared__ float sres[32];
  __shared__ int sresi[32];
  const int sub = threadIdx.x >> 6;
  const int l = threadIdx.x & 63;
  const int base = blockIdx.x * 256 + sub * 64;
  sh[sub * 64 + l] = out;
  shi[sub * 64 + l] = l;
  for (int r = 0; r < 8; ++r) {
    float v = sh[sub * 64 + l];
    int i = shi[sub * 64 + l];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_xor(v, off);
      int oi = __shfl_xor(i, off);
      if (ov > v) {
        v = ov;
        i = oi;
      }
    }
    if (l == 0) {
      wmax[sub * 2 + 0] = v;
      wmaxi[sub * 2 + 0] = i;
    }
    if (l == 32) {
      wmax[sub * 2 + 1] = v;
      wmaxi[sub * 2 + 1] = i;
    }
    __syncthreads();
    v = wmax[sub * 2 + 0];
    i = wmaxi[sub * 2 + 0];
    if (wmax[sub * 2 + 1] > v) {
      v = wmax[sub * 2 + 1];
      i = wmaxi[sub * 2 + 1];
    }
    sres[sub * 8 + r] = v;
    sresi[sub * 8 + r] = i;
    if (l == i)
      sh[sub * 64 + l] = -INFINITY;
    __syncthreads();
  }
  if (l == 0) {
    for (int r = 0; r < 8; ++r) {
      cand_val[(blockIdx.x * 4 + sub) * 8 + r] = sres[sub * 8 + r];
      cand_col[(blockIdx.x * 4 + sub) * 8 + r] = base + sresi[sub * 8 + r];
    }
  }
}

int launch_lm_head_int2(
    const void* x,
    const void* wq,
    const void* scale,
    const void* zs,
    const void* sx,
    void* logits,
    void* cand_val,
    void* cand_col,
    int N,
    int K,
    int NG,
    hipStream_t stream) {
  dim3 block(256);
  dim3 grid((N + 255) / 256);
  rocm::lm_head_int2_kernel<<<grid, block, 0, stream>>>(
      (const __hip_bfloat16*)x,
      (const uint8_t*)wq,
      (const float*)scale,
      (const float*)zs,
      (const float*)sx,
      (float*)logits,
      (float*)cand_val,
      (int*)cand_col,
      N,
      K,
      NG);
  return (int)hipGetLastError();
}

}  // namespace rocm
}  // namespace mlx::core

// ---- eval_gpu implementation ----
namespace mlx::core::fast {

void LMHeadInt2::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  auto& x = inputs[0];
  auto& wq = inputs[1];
  auto& scale = inputs[2];
  auto& zs = inputs[3];
  auto& sx = inputs[4];
  auto& logits = outputs[0];
  auto& cand_val = outputs[1];
  auto& cand_col = outputs[2];

  int N = (int)wq.shape(0);
  int K = (int)x.shape(1);
  int NG = (int)sx.shape(0);

  logits.set_data(mlx::core::rocm::malloc_async(logits.nbytes(), encoder));
  cand_val.set_data(
      mlx::core::rocm::malloc_async(cand_val.nbytes(), encoder));
  cand_col.set_data(
      mlx::core::rocm::malloc_async(cand_col.nbytes(), encoder));

  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = rocm::launch_lm_head_int2(
        gpu_ptr<void>(x),
        gpu_ptr<void>(wq),
        gpu_ptr<void>(scale),
        gpu_ptr<void>(zs),
        gpu_ptr<void>(sx),
        gpu_ptr<void>(logits),
        gpu_ptr<void>(cand_val),
        gpu_ptr<void>(cand_col),
        N,
        K,
        NG,
        stream);
    if (rc != 0) {
      throw std::runtime_error("lm_head_int2: launch failed");
    }
  });
}

}  // namespace mlx::core::fast
