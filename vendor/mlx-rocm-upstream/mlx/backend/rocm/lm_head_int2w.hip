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
// Shape requirements (host guards, traced from this kernel): N % 32 == 0 (one 32-column block per tile; a partial last block would run the selection over dead columns and write past the [N/32, 8] candidate arrays) and K % 512 == 0 (Q = K/4 % 128 == 0 so the 128-groups do not cross group boundaries; K = 4096 -> 8 quarters x 8 groups per quarter (4-lane groups, 32 j/lane) and K = 5120 -> 8 quarters x 10 groups per quarter (16-lane groups x 8 j, 2 groups per 32-lane pass, 5 passes - the 4->5 8-j chunk restructure; 40 j/lane, 40-byte weight window, 20.48 KB LDS, 2 blocks/CU). M = 1 only (the bench reshapes x to [1, K]).

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


// ---- K-generalization (27B): the K = 5120 / NG = 40 (Q = 1280 = 10 128-groups
// per quarter) variant. The 4-lane 128-group of the K = 4096 kernel cannot
// partition the 32-lane wave (10 groups x 4 lanes = 40 slots), so the group
// structure moves to 16-lane groups x 8 j (16 x 8 = 128 per group; 2 groups
// per 32-lane pass, 5 passes = 10 groups per quarter; 4 quarters = all 40
// groups covered). Per lane per column: 20 passes x 8 j = 160 FMA
// (5 k-passes x 4 quarters x 8 j).
//
// Weight access (per the ground-truth packer layout - byte p holds element
// p + q*1280 at bit 2q; group G = 10q + g10 owns elements [G*128, (G+1)*128)
// = quarter q = G/10, g10 = G%10, bytes 128*g10..128*g10+127 at bit 2q):
// lane (l8, gs) covers 128-group g10 = 2k + gs (k 0..4, gs 0/1); its 8 j =
// 128*g10 + 8*l8 + b (b 0..7), read as 2 x 32-bit streaming loads per pass
// (bytes 128*g10 + 8*l8 + 4*b0..7, 8-byte aligned) - one per 4-j half. x_b
// pairs with the 2-bit slot at shift 2q + 8*(b/4) + 2*(b%4) (b 0..3 -> word
// 0, b 4..7 -> word 1; the K = 4096 kernel's x_b/slot-b pairing). No
// preload window: the 2 words are loaded per pass (40 loads = 160 B per
// lane per column, L2-cached).
//
// x staging (20.48 KB LDS = 5120 x f32, 2 blocks/CU on the 64 KB CU budget):
// bank-scrambled layout - element q*1280 + 128*g10 + 8*l8 + b maps to
// k2 = q*1280 + 128*g10 + 4*(l8 + 16*(b&1)) + ((b>>1)&3), so a 32-lane read
// at fixed (b, q) spans 16 words per 16-lane group (banks l8 + 16*(b&1) mod
// 32 = 16 distinct; the 2 groups share the 16 banks = 2-way, 2 LDS cycles
// per read instruction) instead of 32 lanes on 1 bank (16-way). Writes are
// 2-way (2 cycles, one-shot).
//
// Same op/contract/quantization algebra as the scalar port and the K = 4096
// kernel (per 128-group: d*sr[G] - sx[G]*zr[G] with the host-precomputed sx;
// the per-32-column top-8 candidate emission is unchanged); only the
// eval_gpu guard + the group/chunk structure move. The per-lane acc
// accumulation order is pass-major (k outer, q inner: G order
// {gs, 2+gs, ...} then {10+gs, 12+gs, ...}) - a new shape (the scalar port's
// q-outer/G-ascending order is not bit-matchable from a 16-lane 5-group
// partition; same non-bit-exact class as the K = 4096 kernel's 8-group x
// 4-lane partition); the per-group dequant algebra is the same.
// 128-entry staging permutation (see kernel5120 comment). Table form is
// load-bearing: the ROCm 7.2 clang (22.x) mis-compiles the closed form
// 4*(l8 + 16*(b&1)) + ((b>>1)&3) when l8/b are bit-field extracts of one
// thread id (verified on gfx1201: the SASS strength-reduction produces a
// non-bijective index; the table + (r & 0xFF80) form is exact).
__constant__ const int PI128[128] = {
  0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71,
  8, 72, 9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
  16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
  24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95,
  32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103,
  40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111,
  48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54, 118, 55, 119,
  56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127
};

__global__ __launch_bounds__(1024) void lm_head_int2w_kernel5120(
    const hip_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const float* __restrict__ scale,
    const float* __restrict__ zs,
    const float* __restrict__ sx,
    float* __restrict__ logits,
    float* __restrict__ cand_val,
    int* __restrict__ cand_col,
    int n_tiles) {
  const int K = 5120;
  const int NG = 40;
  const int Q = K >> 2;  // 1280 quarter elements
  const int lane = threadIdx.x;
  const int ty = threadIdx.y;
  const int l8 = lane & 15;   // 16-lane group member (x 8 j = 128 j group)
  const int gs = lane >> 4;   // 16-lane group slot in the 32-lane wave (0/1)
  const int tid = lane + ty * 32;  // 2-D flat thread id (0..1023)

  extern __shared__ float x_shared[];
  __shared__ float sh[32];
  __shared__ float sres[8];
  __shared__ int sresi[8];

  // ---- x staging (f32, once per block; the x row is fixed per call and
  // shared by the 32 column-warps across all grid-stride tiles). Bank-
  // scrambled layout (see header): element e = q*1280 + 128*g10 + 8*l8 + b
  // -> k2 = q*1280 + 128*g10 + 4*(l8 + 16*(b&1)) + ((b>>1)&3).
  for (int i = tid; i < K; i += 1024) {
    const int q = i / 1280;
    const int r = i - q * 1280;
    x_shared[q * 1280 + (r & 0xFF80) + PI128[r & 127]] =
        static_cast<float>(x[i]);
  }
  __syncthreads();

  // ---- Per-pass VGPR streaming (one 128-group per pass): 8 j x 4 quarters
  // = 32 FMA per pass; the 2 weight words stream from HBM/L2 per pass
  // (8-byte aligned, no preload window). Peak live window = 4 x floats +
  // 2 weight words + dc[4] + acc (20 passes x 32 FMA = 160 FMA per lane per
  // column; the x operands are LDS-streamed in 32 read instructions).
  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int col = tile * 32 + ty;
    const uint8_t* wr = wq + (size_t)col * Q;
    const float* sr = scale + (size_t)col * NG;
    const float* zr = zs + (size_t)col * NG;

    float acc = 0.0f;
    float dc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < 5; ++k) {
      // pass k = 128-group g10 = 2k + gs: 8 j = 128*g10 + 8*l8 + b (b 0..7),
      // 2 x 4-j halves (word 0 = b 0..3, word 1 = b 4..7).
      const int g10 = 2 * k + gs;
      const uint8_t* wp = wr + 128 * g10 + 8 * l8;
      const uint32_t w0 = *reinterpret_cast<const uint32_t*>(wp);
      const uint32_t w1 = *reinterpret_cast<const uint32_t*>(wp + 4);
      dc[0] = dc[1] = dc[2] = dc[3] = 0.0f;
      #pragma unroll
      for (int b = 0; b < 8; ++b) {
        const uint32_t w = (b < 4) ? w0 : w1;
        const int xb = PI128[8 * l8 + b];
        const float x0 = x_shared[g10 * 128 + xb];
        const float x1 = x_shared[1280 + g10 * 128 + xb];
        const float x2 = x_shared[2560 + g10 * 128 + xb];
        const float x3 = x_shared[3840 + g10 * 128 + xb];
        // Byte b of the 8-byte window (b 0..3 = word 0, b 4..7 = word 1);
        // quarter q is bit 2q within that byte: word shift 8*(b&3) + 2*q.
        const int s2 = 8 * (b & 3);
        dc[0] += x0 * static_cast<float>((w >> s2) & 3u);
        dc[1] += x1 * static_cast<float>((w >> (s2 + 2)) & 3u);
        dc[2] += x2 * static_cast<float>((w >> (s2 + 4)) & 3u);
        dc[3] += x3 * static_cast<float>((w >> (s2 + 6)) & 3u);
      }
      // 16-lane group reduction (offsets 2, 1 within the 4-lane sub-groups,
      // then 8, 4 across them; the 16-lane groups are 16-aligned so all
      // offsets stay in-group), then the fused group algebra.
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        dc[q] += __shfl_xor(dc[q], 2);
        dc[q] += __shfl_xor(dc[q], 1);
        dc[q] += __shfl_xor(dc[q], 8);
        dc[q] += __shfl_xor(dc[q], 4);
        const int G = 10 * q + g10;
        acc += dc[q] * sr[G] - sx[G] * zr[G];
      }
    }
    // After the per-pass 16-lane group reduction, acc is broadcast across
    // the 16-lane group: the other 5 g10 groups (the 40-group total = 5 per
    // 16-lane group) live in the partner 16-lane group, so a single
    // cross-group step combines them (a 5-step 32-lane butterfly here would
    // multiply by 16 - it sums 16 copies of the already-reduced group sum).
    acc += __shfl_xor(acc, 16);

    if (lane == 0) {
      // Route the store address through a 64-bit VGPR local (the K = 4096
      // kernel's 24-SGPR/wave 2-blocks/CU fix).
      const uint64_t la = reinterpret_cast<uint64_t>(logits) + (uint64_t)col * 4;
      *reinterpret_cast<float*>(la) = acc;
    }

    // ---- per-32-column top-8 selection: unchanged from the K = 4096 kernel.
    if (lane == 0)
      sh[ty] = acc;
    __syncthreads();
    if (ty == 0) {
      #pragma unroll
      for (int r = 0; r < 8; ++r) {
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
  if (K == 5120 && NG == 40) {
    rocm::lm_head_int2w_kernel5120<<<dim3(1, grid_y), block, lds, stream>>>(
        (const hip_bfloat16*)x,
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
  if (K != 4096 || NG != 32) {
    return -1;
  }
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
  // K-generalization: K = 4096 (NG = 32, the 4-lane-group kernel) or
  // K = 5120 (NG = 40, the 16-lane-group 5120 kernel).
  if ((K != 4096 || NG != 32) && (K != 5120 || NG != 40)) {
    throw std::invalid_argument(
        "lm_head_int2_wide: K must be 4096 (NG 32) or 5120 (NG 40)");
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
