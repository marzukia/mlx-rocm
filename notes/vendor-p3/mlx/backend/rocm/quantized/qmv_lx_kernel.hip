// Copyright © 2025 Apple Inc.
//
// K1: int8-activation v_dot4 QMV for 4/8-bit affine decode (M<=8).
//
// The default qmv_wide_kernel re-reads the whole x row from L2 for every
// output column (N x K x 2B of L2 traffic per call) and pays a bf16->f32
// cvt + f32 FMA per weight. This kernel:
//   1. stages x in LDS as int8 (per-GROUP_SIZE scale dx + per-chunk exact
//      sum Sxc, quantized in-block by the block's warps — no extra launch),
//   2. streams the packed weights with 128-bit loads,
//   3. accumulates with v_dot4 (int8 x uint8, __builtin_amdgcn_sudot4),
//      one per 4 products instead of 4 FMA + 4 cvt + 4 shift/and,
//   4. groups COLS output columns per block (1 warp per column) so the x
//      row is staged once per COLS columns and launch/tail overhead is
//      amortized over COLS columns.
//
// Math (affine dequant w = s*q + b, x ~= dx*q_x):
//   sum_k x[k] * (s[g(k)]*q_w[k] + b[g(k)])
//     = sum_chunks s * dx * dot(q_x, q_w) + b * Sxc
// where a chunk is the WPER (=128/BITS) consecutive weights of one lane;
// Sxc is the exact (bf16) sum of x over that chunk.
//
// A WPER-weight chunk never spans two groups for GROUP_SIZE in {32,64,128}
// (WPER <= 32 <= GROUP_SIZE), so per-chunk group tracking is exact and the
// bias is counted exactly once per chunk.
//
// Requires: K % 32 == 0, K % GROUP_SIZE == 0, wave32.
// Gated by MLX_QMV_LX (opt-in A/B against the wide default).

#include "mlx/backend/rocm/quantized/qdequant.hpp"
#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename ScaleT, int GROUP_SIZE, int BITS, int COLS>
__global__ void __launch_bounds__(32 * COLS) qmv_lx_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int WPER = 128 / BITS; // weights per lane per iteration
  int lane = threadIdx.x;
  int warp = threadIdx.y;
  int col = blockIdx.x * COLS + warp;
  int row = blockIdx.y;
  if (col >= N || row >= M)
    return;
  int ng = K / GROUP_SIZE;

  // LDS: int8 x [K] | dx [ng] | Sxc [K/WPER]
  extern __shared__ uint8_t smem[];
  int8_t* sxq = reinterpret_cast<int8_t*>(smem);
  float* sdx = reinterpret_cast<float*>(smem + static_cast<size_t>(K));
  float* sSxc = sdx + ng;

  const T* xr = x + static_cast<size_t>(row) * K;

  // ---- Phase 1: per-group max + exact sum (one warp per group) ----
  for (int g = warp; g < ng; g += COLS) {
    float mx = 0.0f, sm = 0.0f;
    const T* gp = xr + g * GROUP_SIZE;
#pragma unroll
    for (int j = 0; j < GROUP_SIZE / 32; ++j) {
      float v = static_cast<float>(gp[lane + 32 * j]);
      mx = fmaxf(mx, fabsf(v));
      sm += v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor(mx, o));
      sm += __shfl_xor(sm, o);
    }
    float d = mx / 127.0f;
    if (d == 0.0f)
      d = 1.0f;
    if (lane == 0) {
      sdx[g] = d;
    }
  }
  __syncthreads();

  // ---- Phase 1b: quantize to int8 + per-chunk exact sum. Each thread owns
  // one contiguous WPER-value chunk (chunks never span a group). ----
  int tid = threadIdx.x + threadIdx.y * 32;
  int nchunks = K / WPER;
  for (int c = tid; c < nchunks; c += 32 * COLS) {
    int base = c * WPER;
    int g = base / GROUP_SIZE;
    float inv = 1.0f / sdx[g];
    float sm = 0.0f;
#pragma unroll
    for (int j = 0; j < WPER; ++j) {
      float v = static_cast<float>(xr[base + j]);
      sm += v;
      int q = __float2int_rn(v * inv);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      // Phase 2 pairs the low-nibble stream (even-indexed weights) with one
      // uint32 of sxq and the high-nibble stream (odd weights) with the next,
      // so sxq must hold [even x | odd x] per 8-value group for 4-bit;
      // 8-bit weights pair naturally and keep natural order.
      int dst = j;
      if constexpr (BITS == 4) {
        int m = j & 7;
        int group8 = (j & ~7);
        dst = (m & 1) ? (group8 + 4 + ((m - 1) >> 1)) : (group8 + (m >> 1));
      }
      sxq[base + dst] = static_cast<int8_t>(q);
    }
    sSxc[c] = sm;
  }
  __syncthreads();

  // ---- Phase 2: one warp per column, v_dot4 over the LDS-staged x ----
  const uint8_t* wrow = w + static_cast<size_t>(col) * (K * BITS) / 8;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;

  float acc = 0.0f;
  int cur_g = -1;
  int gdot = 0;
  float s = 0.0f, b = 0.0f, dxv = 0.0f;

  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    int g = k / GROUP_SIZE;
    if (g != cur_g) {
      cur_g = g;
      s = static_cast<float>(scr[g]);
      b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
      dxv = sdx[g];
      gdot = 0;
    }
    uint4 wv = *reinterpret_cast<const uint4*>(wrow + (k * BITS) / 8);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    if constexpr (BITS == 4) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t wd = wp[q];
        uint32_t ew = wd & 0x0F0F0F0F; // 4 int8 (0..15)
        uint32_t ow = (wd >> 4) & 0x0F0F0F0F;
        const uint32_t* xp = reinterpret_cast<const uint32_t*>(sxq + k + q * 8);
        gdot = __builtin_amdgcn_sudot4(true, xp[0], false, ew, gdot, false);
        gdot = __builtin_amdgcn_sudot4(true, xp[1], false, ow, gdot, false);
      }
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t wd = wp[q]; // 4 uint8 weights
        uint32_t xv = *reinterpret_cast<const uint32_t*>(sxq + k + q * 4);
        gdot = __builtin_amdgcn_sudot4(true, xv, false, wd, gdot, false);
      }
    }
    // Chunk flush: scale term over this chunk's dot, bias over this chunk's
    // exact x-sum (counted exactly once).
    acc += s * dxv * static_cast<float>(gdot) + b * sSxc[k / WPER];
    gdot = 0;
  }
  // All 32 lanes hold partial sums over disjoint K slices of this column:
  // reduce across the warp before writing, or one lane slice becomes the
  // whole output.
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_xor(acc, o);
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc);
}

} // namespace rocm

} // namespace mlx::core
