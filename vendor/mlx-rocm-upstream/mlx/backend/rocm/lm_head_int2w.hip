// Copyright © 2025 Apple Inc.
//
// lm_head_int2_wide: VECTORIZED (wave-wide) int2 g128-asymmetric dequant GEMV (M=1) for the MTP draft head. The R4c scalar port (lm_head_int2.hip) is 18.7 ms/row (19.75x the 4-bit head); this kernel mirrors the 4-bit tiled QMV structure (one 32-lane wave per output column, 64 columns per 2048-thread block, x staged f32 in LDS once per tile, streaming weight loads, per-group fused scale algebra) with the quarter-split 2-bit packing decoded per element (shift+and) fused straight into the MAC loop — no scalar dequant pre-pass.
//
// Packing (quarter-split, identical to the scalar port): packed byte p of a row of K elements holds four 2-bit planes for element indices j = p, K/4 + p, K/2 + p, 3K/4 + p (plane q = p / (K/4) of that byte). Group G = 8q + gl (gl = p >> 7) owns elements [G*128, (G+1)*128) of quarter q. Per-group algebra (scale_g = s, zs_g = z*s, sx_g host-precomputed): acc += d_g * scale_g - sx_g * zs_g where d_g = sum over the group's 128 elements of x_k * q_k. With 32 lanes per column and 32 j per lane per quarter, group G = 8q + (lane >> 2) is exactly the 4-lane group lane>>2 == gl (4 lanes x 32 j = 128 j), reduced with __shfl_xor offsets 2, 1 (the 4-lane groups are 4-aligned).
//
// Per lane per column: 4 iterations x (2 u32 = 8 packed bytes, 8-byte aligned, streaming) = 32 weight bytes (half the 4-bit kernel's 64), 128 FMA + 128 per-element decodes (one FMA + one decode per weight, same per-weight instruction count as the 4-bit kernel), 8 fused group ops (one per quarter — the 4-bit kernel's count), the 4-lane group reductions, and the final 5-step __shfl_xor warp reduction.
//
// Candidate emission transcribes the scalar port's 8-round per-64-column top-8 shared-memory max selection verbatim (64 shared entries, wmax[2] merge, offsets 16..1, strict '>' so the lower index wins on ties, 8 rounds), so the re-rank contract (top-32 candidates re-scored with exact weights by the caller) is unchanged.
//
// Shape requirements (host guards, traced from this kernel): N % 64 == 0 (one 64-column block per tile; a partial last block would run the selection over dead columns and write past the [N/64, 8] candidate arrays) and K % 512 == 0 (Q = K/4 % 32 == 0 so the 4-lane groups do not cross group boundaries; K = 4096 -> 8 quarters x 8 groups per quarter, all 32 groups covered). M = 1 only (the bench reshapes x to [1, K]).

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <math.h>

namespace mlx::core {
namespace rocm {

__global__ __launch_bounds__(2048) void lm_head_int2w_kernel(
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
    int NG,
    int n_tiles) {
  const int Q = K >> 2;  // quarter length in elements (K/4)
  const int lane = threadIdx.x;
  const int gl = lane >> 2;  // 4-lane group; group G = 8*q + gl
  const int l3 = lane & 3;
  const int tid = threadIdx.y * 32 + lane;

  extern __shared__ float x_shared[];

  __shared__ float sh[64];
  __shared__ int shi[64];
  __shared__ float wmax[2];
  __shared__ int wmaxi[2];
  __shared__ float sres[8];
  __shared__ int sresi[8];

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
  const int col = tile * 64 + threadIdx.y;
  // ---- x staging (f32, once per tile, shared by the 64 column-warps) ----
  __syncthreads();
  for (int i = tid; i < K; i += 2048)
    x_shared[i] = static_cast<float>(x[i]);
  __syncthreads();

  const uint8_t* wr = wq + (size_t)col * Q;
  const float* sr = scale + (size_t)col * NG;
  const float* zr = zs + (size_t)col * NG;

  float acc = 0.0f;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  #pragma unroll
  for (int it = 0; it < 4; ++it) {
    // This lane's 8 packed bytes for this iteration: 8 consecutive j of the
    // lane's 32 j per quarter (32 j = 8 bytes), 4 iterations cover the 32 j.
    const int j0 = 128 * gl + 32 * l3 + 8 * it;
    const uint2 wv = *reinterpret_cast<const uint2*>(wr + j0);
    const uint32_t w0 = wv.x;
    const uint32_t w1 = wv.y;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const uint32_t byte =
          (b < 4) ? ((w0 >> (8 * b)) & 0xFFu) : ((w1 >> (8 * (b - 4))) & 0xFFu);
      const int j = j0 + b;
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        // Plane q of packed byte j0 + b = element q*Q + (j0 + b); the decode
        // (byte >> 2q) & 3 is fused into the FMA (no pre-pass).
        d[q] += x_shared[q * Q + j] * static_cast<float>((byte >> (2 * q)) & 3u);
      }
    }
  }
  // Per quarter: 4-lane group reduction (offsets 2, 1; the 4-lane groups are
  // 4-aligned so the offsets stay in-group), then the fused group algebra.
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1)
      d[q] += __shfl_xor(d[q], off);
    const int G = 8 * q + gl;
    acc += d[q] * sr[G] - sx[G] * zr[G];
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off);

  if (lane == 0)
    logits[col] = acc;

  // ---- per-64-column top-8 selection: the scalar port's 8-round shared-memory
  // max selection transcribed verbatim onto this block's 64 columns (threads
  // 0..63 = columns 0..63, the same thread l -> column l layout as the scalar
  // sub-block; strict '>' keeps the lower index on ties). ----
  const int l = tid;  // 0..2047; the selection group is l in 0..63
  if (threadIdx.y < 2 && lane == 0) {
    sh[l] = acc;
    shi[l] = l;
  }
  for (int r = 0; r < 8; ++r) {
    float v = 0.0f;
    int i = 0;
    if (threadIdx.y < 2) {
      v = sh[l];
      i = shi[l];
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(v, off);
        int oi = __shfl_xor(i, off);
        if (ov > v) {
          v = ov;
          i = oi;
        }
      }
    }
    if (threadIdx.y == 0 && lane == 0) {
      wmax[0] = v;
      wmaxi[0] = i;
    }
    if (threadIdx.y == 1 && lane == 0) {
      wmax[1] = v;
      wmaxi[1] = i;
    }
    __syncthreads();
    v = wmax[0];
    i = wmaxi[0];
    if (wmax[1] > v) {
      v = wmax[1];
      i = wmaxi[1];
    }
    if (threadIdx.y < 2) {
      sres[r] = v;
      sresi[r] = i;
      if (l == i)
        sh[l] = -INFINITY;
    }
    __syncthreads();
  }
  if (threadIdx.y == 0 && lane == 0) {
    for (int r = 0; r < 8; ++r) {
      cand_val[tile * 8 + r] = sres[r];
      cand_col[tile * 8 + r] = tile * 64 + sresi[r];
    }
  }
  }  // grid-stride over column tiles
}

int launch_lm_head_int2w(
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
    int max_blocks,
    hipStream_t stream) {
  dim3 block(32, 64);
  int n_tiles = N / 64;
  int grid_y = (max_blocks > 0 && max_blocks < n_tiles) ? max_blocks : n_tiles;
  size_t lds = (size_t)K * 4;
  rocm::lm_head_int2w_kernel<<<dim3(1, grid_y), block, lds, stream>>>(
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
      NG,
      n_tiles);
  return (int)hipGetLastError();
}

}  // namespace rocm
}  // namespace mlx::core

// ---- eval_gpu implementation ----
namespace mlx::core::fast {

void LMHeadInt2W::eval_gpu(
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
  // One 2048-thread block per CU (the 4-bit tiled kernel's 768-thread blocks
  // run 2-3 per CU; this is the full-CU equivalent).
  int max_blocks = d.num_cus();

  logits.set_data(mlx::core::rocm::malloc_async(logits.nbytes(), encoder));
  cand_val.set_data(
      mlx::core::rocm::malloc_async(cand_val.nbytes(), encoder));
  cand_col.set_data(
      mlx::core::rocm::malloc_async(cand_col.nbytes(), encoder));

  encoder.launch_kernel([&](hipStream_t stream) {
    int rc = rocm::launch_lm_head_int2w(
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
        max_blocks,
        stream);
    if (rc != 0) {
      throw std::runtime_error("lm_head_int2_wide: launch failed");
    }
  });
}

}  // namespace mlx::core::fast
