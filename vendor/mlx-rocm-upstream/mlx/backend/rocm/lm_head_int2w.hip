// Copyright © 2025 Apple Inc.
//
// lm_head_int2_wide: VECTORIZED (wave-wide) int2 g128-asymmetric dequant GEMV (M=1) for the MTP draft head. The R4c scalar port (lm_head_int2.hip) is 18.7 ms/row (19.75x the 4-bit head); this kernel mirrors the 4-bit tiled QMV structure (one 32-lane wave per output column, 32 columns per 1024-thread block, x staged f32 in LDS once per tile, streaming weight loads, per-group fused scale algebra) with the quarter-split 2-bit packing decoded per element (shift+and) fused straight into the MAC loop - no scalar dequant pre-pass.
//
// Packing (quarter-split, identical to the scalar port): packed byte p of a row of K elements holds four 2-bit planes for element indices j = p, K/4 + p, K/2 + p, 3K/4 + p (plane q = p / (K/4) of that byte). Group G = 8q + gl (gl = p >> 7) owns elements [G*128, (G+1)*128) of quarter q. Per-group algebra (scale_g = s, zs_g = z*s, sx_g host-precomputed): acc += d_g * scale_g - sx_g * zs_g where d_g = sum over the group's 128 elements of x_k * q_k. With 32 lanes per column and 32 j per lane per quarter, group G = 8q + (lane >> 2) is exactly the 4-lane group lane>>2 == gl (4 lanes x 32 j = 128 j), reduced with __shfl_xor offsets 2, 1 (the 4-lane groups are 4-aligned).
//
// Per lane per column: 4 iterations x (2 u32 = 8 packed bytes, 8-byte aligned, streaming) = 32 weight bytes (half the 4-bit kernel's 64), 128 FMA + 128 per-element decodes (one FMA + one decode per weight, same per-weight instruction count as the 4-bit kernel), 8 fused group ops (one per quarter - the 4-bit kernel's count), the 4-lane group reductions, and the final 5-step __shfl_xor warp reduction.
//
// Block shape: (32, 32) = 1024 threads (one wave per column; AMD workgroups cap at 1024 threads, so the 2048-thread design is not launchable on gfx1201). 32 columns per tile, N/32 tiles, grid-stride over the CU-bounded grid; 2 blocks per CU (2048 threads, 32.5 KB LDS on the 64 KB CU budget). x staged f32 in dynamic LDS (K floats = 16 KB at K=4096) once per block, then this lane's 128 x-values (32 j per quarter x 4 quarters, 32 contiguous per quarter) are preloaded into VGPRs before the tile loop - mirroring the 4-bit tiled kernel's x_local staging, so the FMA loop does 0 LDS x-reads (the x row is fixed per call, so the staging is hoisted out of the grid-stride loop entirely).
//
// Candidate emission: per-32-column top-8, the scalar port's 8-round shared-memory max selection re-cast onto 32 entries with the single-selection warp (threadIdx.y == 0) - 8 rounds of 5-step __shfl_xor warp reduce-max (offsets 16..1, strict '>' so the lower index wins ties), the winner removed with -INFINITY in the pool, sres[8] in shared. Per-32 top-8 strictly refines per-64 top-8 (each per-64 top-8 set is contained in the union of its two per-32 top-8 sets), so the re-rank contract (top-32 candidates re-scored with exact weights by the caller) holds at least as strongly; the candidate arrays are [N/32, 8].
//
// Shape requirements (host guards, traced from this kernel): N % 32 == 0 (one 32-column block per tile; a partial last block would run the selection over dead columns and write past the [N/32, 8] candidate arrays) and K % 512 == 0 (Q = K/4 % 128 == 0 so the 4-lane 128-groups do not cross group boundaries; K = 4096 -> 8 quarters x 8 groups per quarter, all 32 groups covered). M = 1 only (the bench reshapes x to [1, K]).

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <math.h>

namespace mlx::core {
namespace rocm {

__global__ __launch_bounds__(1024) void lm_head_int2w_kernel(
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
  const int ty = threadIdx.y;  // the column wave
  const int gl = lane >> 2;  // 4-lane group; group G = 8*q + gl
  const int l3 = lane & 3;

  extern __shared__ float x_shared[];
  __shared__ float sh[32];
  __shared__ float sres[8];
  __shared__ int sresi[8];

  // ---- x staging (f32, once per block; the x row is fixed per call and
  // shared by the 32 column-warps across all grid-stride tiles) ----
  for (int i = lane; i < K; i += 1024)
    x_shared[i] = static_cast<float>(x[i]);
  __syncthreads();

  // ---- Register-resident x: this lane's 32 j per quarter x 4 quarters =
  // 128 x-values (mirrors the 4-bit tiled kernel's x_local staging; the
  // per-lane j block 128*gl + 32*l3 + [0..32) is 32 contiguous elements per
  // quarter). Loaded once before the tile loop: the FMA loop does 0 LDS
  // x-reads. ----
  float xv[4][32];
  #pragma unroll
  for (int q = 0; q < 4; ++q)
    #pragma unroll
    for (int j = 0; j < 32; ++j)
      xv[q][j] = x_shared[q * Q + 128 * gl + 32 * l3 + j];

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int col = tile * 32 + ty;
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
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          // Plane q of packed byte j0 + b = element q*Q + (j0 + b); the decode
          // (byte >> 2q) & 3 is fused into the FMA (no pre-pass); the x
          // operand is register-resident (xv[q][b] = x_shared[q*Q + 128*gl +
          // 32*l3 + 8*it + b], preloaded before the tile loop).
          d[q] += xv[q][b] * static_cast<float>((byte >> (2 * q)) & 3u);
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

    // ---- per-32-column top-8 selection: the scalar port's 8-round shared-memory
    // max selection re-cast onto 32 entries, run by the single selection warp
    // (ty == 0); every wave stages its column's logit into sh first. ----
    if (lane == 0)
      sh[ty] = acc;
    __syncthreads();
    if (ty == 0) {
      #pragma unroll
      for (int r = 0; r < 8; ++r) {
        // Per round: re-read the pool (the previous round's winner was
        // removed with -INFINITY), 5-step __shfl_xor reduce-max (offsets
        // 16..1, strict '>' so the lower index wins ties), record, remove.
        float v = sh[lane];
        int i = lane;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          float ov = __shfl_xor(v, off);
          int oi = __shfl_xor(i, off);
          if (ov > v) {
            v = ov;
            i = oi;
          }
        }
        sres[r] = v;
        sresi[r] = i;
        if (lane == i)
          sh[lane] = -INFINITY;
      }
    }
    __syncthreads();
    if (ty == 0 && lane == 0) {
      for (int r = 0; r < 8; ++r) {
        cand_val[tile * 8 + r] = sres[r];
        cand_col[tile * 8 + r] = tile * 32 + sresi[r];
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
  dim3 block(32, 32);
  int n_tiles = N / 32;
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
  // 2 blocks per CU (1024 threads each = the full 2048-thread CU budget;
  // 32.5 KB LDS each on the 64 KB budget).
  int max_blocks = d.num_cus() * 2;

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
