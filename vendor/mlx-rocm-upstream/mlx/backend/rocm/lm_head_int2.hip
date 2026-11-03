// Copyright © 2025 Apple Inc.
//
// lm_head_int2: int2 g128-asymmetric dequant GEMV (M=1) for the MTP draft
// head. Replaces the full bf16 lm_head (248320 x 4096, 2.03 GiB of weight
// traffic per draft call) with a 0.27 GiB int2 packed read.
//
// Packing (quarter-split, ported from radiance): for a 128-group, byte j
// holds four 2-bit planes for x indices j, j+Q, j+2Q, j+3Q (Q = K/4):
//   v0 = b & 3 ; v1 = (b >> 2) & 3 ; v2 = (b >> 4) & 3 ; v3 = (b >> 6) & 3
// Quantization: per-group asymmetric, s = max((hi-lo)/3, 1e-8),
// z = clamp(round(-lo/s), 0, 3), q = clamp(round(w/s + z), 0, 3).
// The op takes the PRE-MULTIPLIED per-group coefficients
//   scale_g = 4*s,  zs_g = (4+z)*s
// so the dequant algebra per group is
//   out += dot_g * scale_g - sx_g * zs_g
// where dot_g = sum_k x_k * q_k and sx_g = sum_k x_k (host-precomputed;
// algebraically this equals sum_g s_g * sum_k x_k (q_k - z_g) exactly —
// see REPORT-mtp-fix-round1.md for the derivation and rounding notes).
//
// One thread per output column (N = 248320, 970 blocks of 256 threads).
// After its column's logit, each 64-column subblock runs an 8-round
// shared-memory max selection and emits the per-block top-8
// (value, col) candidate pairs so the caller can re-rank the global
// top-32 with the EXACT bf16 weights (bit-exact for those 32 columns).
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

namespace mlx::core {
namespace rocm {

__global__ void lm_head_int2_kernel(
    const __nv_bfloat16* __restrict__ x,
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
  const int Q = K >> 2;
  float out = 0.0f;
  for (int g = 0; g < NG; ++g) {
    const uint8_t* p = wq + (size_t)col * Q + (size_t)g * 32;
    const __nv_bfloat16* xg = x + (size_t)g * 128;
    float d = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < 32; ++j) {
      // Byte (j*4..j*4+3): plane q reads x index (j*4+q), j*4+Q, j*4+2Q,
      // j*4+3Q of the group.
      const int j0 = j * 4;
      unsigned b0 = p[j0];
      unsigned b1 = p[j0 + 1];
      unsigned b2 = p[j0 + 2];
      unsigned b3 = p[j0 + 3];
      d += __bfloat162float(xg[j0 * 4 + 0]) * (float)(b0 & 3);
      d += __bfloat162float(xg[j0 * 4 + 1]) * (float)((b0 >> 2) & 3);
      d += __bfloat162float(xg[j0 * 4 + 2]) * (float)((b0 >> 4) & 3);
      d += __bfloat162float(xg[j0 * 4 + 3]) * (float)((b0 >> 6) & 3);
      d += __bfloat162float(xg[j0 * 4 + 0 + Q]) * (float)((b1) & 3);
      d += __bfloat162float(xg[j0 * 4 + 1 + Q]) * (float)((b1 >> 2) & 3);
      d += __bfloat162float(xg[j0 * 4 + 2 + Q]) * (float)((b1 >> 4) & 3);
      d += __bfloat162float(xg[j0 * 4 + 3 + Q]) * (float)((b1 >> 6) & 3);
      d += __bfloat162float(xg[j0 * 4 + 0 + 2 * Q]) * (float)(b2 & 3);
      d += __bfloat162float(xg[j0 * 4 + 1 + 2 * Q]) * (float)((b2 >> 2) & 3);
      d += __bfloat162float(xg[j0 * 4 + 2 + 2 * Q]) * (float)((b2 >> 4) & 3);
      d += __bfloat162float(xg[j0 * 4 + 3 + 2 * Q]) * (float)((b2 >> 6) & 3);
      d += __bfloat162float(xg[j0 * 4 + 0 + 3 * Q]) * (float)(b3 & 3);
      d += __bfloat162float(xg[j0 * 4 + 1 + 3 * Q]) * (float)((b3 >> 2) & 3);
      d += __bfloat162float(xg[j0 * 4 + 2 + 3 * Q]) * (float)((b3 >> 4) & 3);
      d += __bfloat162float(xg[j0 * 4 + 3 + 3 * Q]) * (float)((b3 >> 6) & 3);
    }
    out += d * scale[(size_t)col * NG + g] - sx[g] * zs[(size_t)col * NG + g];
  }
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
      (const __nv_bfloat16*)x,
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

}  // namespace mlx::core

// ---- eval_gpu implementation ----
namespace mlx::core::fast {

void LmHeadInt2::eval_gpu(
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
