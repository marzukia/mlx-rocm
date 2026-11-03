// Copyright © 2025 Apple Inc.
//
// lm_head_int2_wide: VECTORIZED (wave-wide) int2 g128-asymmetric dequant GEMV (M=1) for the MTP draft head. The R4c scalar port (lm_head_int2.hip) is 18.7 ms/row (19.75x the 4-bit head); this kernel mirrors the 4-bit tiled QMV structure (one 32-lane wave per output column, 32 columns per 1024-thread block, x staged f32 in LDS once per tile, streaming weight loads, per-group fused scale algebra) with the quarter-split 2-bit packing decoded per element (shift+and) fused straight into the MAC loop - no scalar dequant pre-pass.
//
// Packing (quarter-split, identical to the scalar port): packed byte p of a row of K elements holds four 2-bit planes for element indices j = p, K/4 + p, K/2 + p, 3K/4 + p (plane q = p / (K/4) of that byte). Group G = 8q + gl (gl = p >> 7) owns elements [G*128, (G+1)*128) of quarter q. Per-group algebra (scale_g = s, zs_g = z*s, sx_g host-precomputed): acc += d_g * scale_g - sx_g * zs_g where d_g = sum over the group's 128 elements of x_k * q_k. With 32 lanes per column and 32 j per lane per quarter, group G = 8q + (lane >> 2) is exactly the 4-lane group lane>>2 == gl (4 lanes x 32 j = 128 j), reduced with __shfl_xor offsets 2, 1 (the 4-lane groups are 4-aligned).
//
// Per lane per column: 4 quarters x 2 sub-iterations x (4 u32 = 16 packed bytes, 4-byte aligned, streaming) = 32 weight bytes (half the 4-bit kernel's 64), 128 FMA + 128 per-element decodes (one FMA + one decode per weight, same per-weight instruction count as the 4-bit kernel), 8 fused group ops (one per quarter - the 4-bit kernel's count), the 4-lane group reductions, and the final 5-step __shfl_xor warp reduction. The x operand is streamed from LDS per 16-j sub-iteration (4 x float4 = 16 contiguous floats, the 4-bit kernel's x_local pattern: 128 LDS x-reads per lane per column, which the 4-bit kernel runs at 184.4 ns/column); the 16-value live window keeps the kernel at ~<= 96 VGPR/lane -> 2 blocks/CU (the v2 128-value preload cost 192 VGPR -> 1 block/CU).
//
// Block shape: (32, 32) = 1024 threads (one wave per column; AMD workgroups cap at 1024 threads, so the 2048-thread design is not launchable on gfx1201). 32 columns per tile, N/32 tiles, grid-stride over the CU-bounded grid; 2 blocks per CU (2048 threads, 32.5 KB LDS on the 64 KB CU budget). x staged f32 in dynamic LDS (K floats = 16 KB at K=4096) once per block, then read from LDS in 16-contiguous-float windows inside the FMA loop (the x row is fixed per call, so the staging itself is hoisted out of the grid-stride loop). The FMA order per lane per quarter is byte-identical to the v2 preload (bytes in ascending j, plane q innermost), so the per-lane accumulation sequence - and the bit-identity with the scalar port - is unchanged.
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
    int n_tiles) {
  // cand_val and cand_col are one backing allocation (the op allocates
  // them together, cand_col immediately after cand_val). The col stores
  // index the SAME base pointer with a 32-bit offset (n_tiles*8 + i);
  // forming a separate 64-bit col base would make the compiler allocate
  // it its own SGPR pair (the 24-SGPR/wave budget for two blocks per
  // CU is tight). NG is fixed at 32 by the int2 layout (4 chunks x 8
  // scale groups).
  const int K = 4096;
  const int NG = 32;
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
  // shared by the 32 column-warps across all grid-stride tiles). The
  // layout is bank-spread: k2 = p*1024 + 32*j + 4*gl + l3 for
  // k = p*1024 + 128*gl + 32*l3 + j, so a 32-lane load at fixed (c, b)
  // reads words 4*gl + l3 (32 distinct banks, one LDS cycle) instead of
  // 32 words at the same 4 banks (an 8-way conflict per load).
  for (int i = lane; i < K; i += 1024) {
    const int k2 = (i & 0xff80) | ((i & 31) << 5)
                   | (((i >> 7) & 7) << 2) | ((i >> 5) & 3);
    x_shared[k2] = static_cast<float>(x[i]);
  }
  __syncthreads();

  // ---- Register-pressure fix (v3): the x operand is streamed from LDS in
  // 4-j chunks (4 scalar loads per 2-bit plane chunk p, the bank-free
  // deinterleaved layout above) instead of the v2 128-value preload
  // (192 VGPR -> 1 block/CU). The c loop is a runtime loop (8 iterations
  // per column) so the compiler cannot schedule all 128 x loads in flight;
  // peak live x window = 16 floats. The weight is a 32-byte per-column
  // preload (2 x 16B, non-volatile, L1-hot) selected per 4-j chunk.
  // (A per-chunk volatile 4-byte load measures 1149 us/row; the plain
  // per-chunk u32 load makes the compiler re-issue the whole 32-byte
  // window twice per chunk - 8x redundant traffic, the 9.6 ms/row
  // regression; this preload + select keeps exactly 32 bytes per column.) The FMA order per
  // lane is byte-identical to the v2 preload: j ascending 0..31, plane p
  // innermost, each plane p of byte j paired with x[p*1024 + j] exactly
  // as v2, and the 4 plane reductions + scale run once per column after
  // all 128 FMA (v2's structure). Bit-identity by construction. ----

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int col = tile * 32 + ty;
    const uint8_t* wr = wq + (size_t)col * Q;
    // 32-byte weight window: loaded once per column (2 x 16B), selected
    // per 4-j chunk below.
    const uint4 wv0 = *reinterpret_cast<const uint4*>(wr);
    const uint4 wv1 = *reinterpret_cast<const uint4*>(wr + 16);
    const float* sr = scale + (size_t)col * NG;
    const float* zr = zs + (size_t)col * NG;

    float acc = 0.0f;
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = 0; c < 8; ++c) {
      const int j0 = 128 * gl + 32 * l3 + 4 * c;
      uint32_t w = wv0.x;
      w = (c == 1) ? wv0.y : w;
      w = (c == 2) ? wv0.z : w;
      w = (c == 3) ? wv0.w : w;
      w = (c == 4) ? wv1.x : w;
      w = (c == 5) ? wv1.y : w;
      w = (c == 6) ? wv1.z : w;
      w = (c == 7) ? wv1.w : w;
      #pragma unroll
      for (int p = 0; p < 4; ++p) {
        // Bank-spread x: k2 = p*1024 + 32*j + 4*gl + l3 (j = 4c + b).
        const int base = (p << 10) | ((4 * c) << 5);
        float x0 = x_shared[base + l3 + 4 * gl];
        d[p] += x0 * static_cast<float>((w >> (8 * 0 + 2 * p)) & 3u);
        float x1 = x_shared[base + 32 + l3 + 4 * gl];
        d[p] += x1 * static_cast<float>((w >> (8 * 1 + 2 * p)) & 3u);
        float x2 = x_shared[base + 64 + l3 + 4 * gl];
        d[p] += x2 * static_cast<float>((w >> (8 * 2 + 2 * p)) & 3u);
        float x3 = x_shared[base + 96 + l3 + 4 * gl];
        d[p] += x3 * static_cast<float>((w >> (8 * 3 + 2 * p)) & 3u);
      }
    }
    // Per plane chunk: 4-lane group reduction (offsets 2, 1; the 4-lane
    // groups are 4-aligned so the offsets stay in-group), then the fused
    // group algebra. Runs once per column, after all 128 FMA (v2's order).
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

    if (lane == 0) {
      // Route the store address through a 64-bit VGPR local: this lane's
      // col is uniform, and a direct `logits[col] = acc` makes the compiler
      // park the 64-bit address in an SGPR pair (s[24:25]) - 2 SGPR that
      // push the kernel past the 24-SGPR/wave budget for 2 blocks/CU.
      const uint64_t la = reinterpret_cast<uint64_t>(logits) + (uint64_t)col * 4;
      *reinterpret_cast<float*>(la) = acc;
    }

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
      // Explicit stores (no r loop): a runtime-indexed loop makes the
      // compiler build each round's 64-bit address in a fresh SGPR pair
      // (s[24:25]: 2 SGPR past the 24-SGPR/wave budget for 2 blocks/CU).
      const int t8 = tile * 8;
      cand_val[t8 + 0] = sres[0];
      cand_val[t8 + 1] = sres[1];
      cand_val[t8 + 2] = sres[2];
      cand_val[t8 + 3] = sres[3];
      cand_val[t8 + 4] = sres[4];
      cand_val[t8 + 5] = sres[5];
      cand_val[t8 + 6] = sres[6];
      cand_val[t8 + 7] = sres[7];
      int* col_base = cand_col;
      const int co = t8;
      col_base[co + 0] = tile * 32 + sresi[0];
      col_base[co + 1] = tile * 32 + sresi[1];
      col_base[co + 2] = tile * 32 + sresi[2];
      col_base[co + 3] = tile * 32 + sresi[3];
      col_base[co + 4] = tile * 32 + sresi[4];
      col_base[co + 5] = tile * 32 + sresi[5];
      col_base[co + 6] = tile * 32 + sresi[6];
      col_base[co + 7] = tile * 32 + sresi[7];
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
    void* cand,
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
      (float*)cand,
      (int*)cand_col,
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
  // The kernel's FMA layout is fixed-shape (4 x 1024-element 2-bit chunks);
  // the K argument was dropped from the kernel, so enforce it here.
  if (K != 4096) {
    throw std::invalid_argument("lm_head_int2_wide: K must be 4096");
  }
  if (NG != 32) {
    throw std::invalid_argument("lm_head_int2_wide: NG must be 32");
  }
  // 2 blocks per CU (1024 threads each = the full 2048-thread CU budget;
  // 32.5 KB LDS each on the 64 KB budget).
  int max_blocks = d.num_cus() * 2;

  logits.set_data(mlx::core::rocm::malloc_async(logits.nbytes(), encoder));
  cand_val.set_data(mlx::core::rocm::malloc_async(cand_val.nbytes(), encoder));
  cand_col.set_data(mlx::core::rocm::malloc_async(cand_col.nbytes(), encoder));

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
