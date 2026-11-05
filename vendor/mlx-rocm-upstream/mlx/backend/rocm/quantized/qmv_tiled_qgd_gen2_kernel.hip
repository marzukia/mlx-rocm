// Gen-2 QMM memory-structure variants, ported onto the per-row (M = 1) QMM
// path that actually serves the qmv_gdn class: the TILED QMV kernel
// (qmv_tiled_kernel), block (WARP_SIZE, tile_n), one warp per output column,
// BSK = 512 k-tiles, cooperative x staging into LDS per k-tile, per-lane
// 2 x uint32 streaming weight load (16 4-bit values), fused dequant + 4
// partial dot products, scale/bias applied ONCE per group.
//
// NOTE (session A finding): the stock M = 1 no-bias 4-bit path is the tiled
// kernel, NOT the qmv_wide per-row kernel - wide_ok requires has_bias at
// M = 1, and the GDN 4 in-proj + out_proj layers are bias-free. These
// variants are the gen-2 memory-structure changes re-derived for the tiled
// structure:
//
// V1 (MLX_ROCM_QGD_V1) - LDS x-staging: the tiled kernel already stages x
// into LDS once per k-tile and shares it across all tile_n column warps;
// the V1 change is structurally absorbed by the stock tiled design. Q1 is
// the stock structure as a standalone kernel - the no-op control arm that
// verifies the variant dispatch adds no overhead and the FMA pipeline is
// bit-identical.
//
// V2 (MLX_ROCM_QGD_V2) - k-tile double buffer: the next k-tile's 2 x uint32
// weight slice is issued BEFORE the current tile's dequant, and the
// pre-dequantized 16 float weights alternate buffers A/B (dequant is split
// from the dot; the per-tile FMA order is exactly the dequant_and_dot4
// p/i order).
//
// V3 (MLX_ROCM_QGD_V3) - 32B weight loads: the per-lane 8B weight slice is
// widened to a 16B logical access - two consecutive 8B aligned volatile
// 32-bit pairs (mirrors the m3-V3 32B pattern at the tiled slice width).
// Only the FIRST half (the stock 16 values) is used in the math; the second
// half is read only when the window is in-row (k + 32 <= K per lane) and is
// elision-proofed with an asm so the widened window is genuinely touched.
//
// V4 (MLX_ROCM_QGD_V4) - batched-x load order: the cooperative x staging
// issues 16B vector batches (8 x bf16/f16 values) wherever the chunk start
// is 16B-aligned, scalar for the rest - the x load is a single batched
// group before the weight loads (the batches are issued ahead of the
// per-lane weight loads within each k-tile).
//
// All variants preserve the EXACT stock FMA order (per pack p = 0..1,
// per value i = 0..7: qdot[i & 3] += x * q, xsum += x; then
// acc += scale * reduce_qdot4(qdot) + bias * xsum, once per group) and the
// int->float conversions. 4-bit only (the dispatch gate is bits == 4);
// x dtype bf16/f16 only (the stock tiled gate). All DEFAULT OFF: with all
// four envs unset the stock tiled kernel runs verbatim.

#include "mlx/backend/rocm/quantized/qdequant.hpp"
#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

constexpr int QGD_TILE_N_MAX = 32;  // max tile_n for __launch_bounds__

// V1: stock tiled structure (the V1 x-staging change is already the stock
// design - x_shared is staged once per k-tile and shared by all tile_n
// column warps). No-op control arm.
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ __launch_bounds__(QGD_TILE_N_MAX * WARP_SIZE)
void qmv_tiled_qgd_v1_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,
    int n_tiles) {
  constexpr int PF = pack_factor_u32<4>;
  constexpr int PPT = packs_per_thread<4>;
  constexpr int VPT = values_per_thread<4>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int n = tile * tile_n + threadIdx.y;
    const bool valid = (m < M && n < N);
    const int clamped_n = (n < N) ? n : 0;
    const uint32_t* w_row = w + clamped_n * w_stride;
    const ScaleT* s_row = scales + clamped_n * num_groups;
    const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BSK) {
      __syncthreads();
      for (int i = tid; i < BSK; i += nthreads) {
        int k = k_base + i;
        x_shared[i] = (k < K) ? to_float(x_row[k]) : 0.0f;
      }
      __syncthreads();

      if (!valid) continue;

      float x_local[VPT];
      #pragma unroll
      for (int i = 0; i < VPT; i++)
        x_local[i] = x_shared[lane * VPT + i];

      int w_offset = k_base / PF + lane * PPT;

      float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float group_xsum = 0.0f;

      int k_val = k_base + lane * VPT;
      int group_idx = k_val / GROUP_SIZE;

      uint32_t w_local[PPT];
      if (k_base + BSK <= K) {
        load_weight_vec_streaming<4>(w_row + w_offset, w_local);
      } else {
        #pragma unroll
        for (int p = 0; p < PPT; p++)
          w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      }

      #pragma unroll
      for (int p = 0; p < PPT; p++)
        dequant_and_dot4<4>(w_local[p], &x_local[p * PF], group_qdot4, group_xsum);

      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
    }

    if (!valid) continue;

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
      out[m * N + n] = from_float<T>(acc);
    }
  }
}

// V2: k-tile double buffer.
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ __launch_bounds__(QGD_TILE_N_MAX * WARP_SIZE)
void qmv_tiled_qgd_v2_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,
    int n_tiles) {
  constexpr int PF = pack_factor_u32<4>;
  constexpr int PPT = packs_per_thread<4>;
  constexpr int VPT = values_per_thread<4>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int n = tile * tile_n + threadIdx.y;
    const bool valid = (m < M && n < N);
    const int clamped_n = (n < N) ? n : 0;
    const uint32_t* w_row = w + clamped_n * w_stride;
    const ScaleT* s_row = scales + clamped_n * num_groups;
    const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

    float acc = 0.0f;

    // Double buffer: the next k-tile's 2 x uint32 weight slice is issued
    // before the current tile's dequant; pre-dequantized 16 float weights
    // alternate buffers A/B. The per-tile FMA order is unchanged.
    uint32_t w_local[PPT];
    float qbuf[2][VPT];
    int bank = 0;

    // Prologue: tile 0 weights (same guards as stock).
    {
      const int w_offset = lane * PPT;
      if (BSK <= K) {
        load_weight_vec_streaming<4>(w_row + w_offset, w_local);
      } else {
        #pragma unroll
        for (int p = 0; p < PPT; p++)
          w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      }
    }

    for (int k_base = 0; k_base < K; k_base += BSK) {
      __syncthreads();
      for (int i = tid; i < BSK; i += nthreads) {
        int k = k_base + i;
        x_shared[i] = (k < K) ? to_float(x_row[k]) : 0.0f;
      }
      __syncthreads();

      if (!valid) continue;

      float x_local[VPT];
      #pragma unroll
      for (int i = 0; i < VPT; i++)
        x_local[i] = x_shared[lane * VPT + i];

      // Prefetch the next k-tile BEFORE dequantizing the current one.
      uint32_t w_next[PPT] = {0u, 0u};
      if (k_base + BSK < K) {
        const int w_offset_n = (k_base + BSK) / PF + lane * PPT;
        if (k_base + 2 * BSK <= K) {
          load_weight_vec_streaming<4>(w_row + w_offset_n, w_next);
        } else {
          #pragma unroll
          for (int p = 0; p < PPT; p++)
            w_next[p] = (w_offset_n + p < w_stride) ? w_row[w_offset_n + p] : 0u;
        }
      }

      // Dequant the current slice (pure shifts - no FMA).
      #pragma unroll
      for (int p = 0; p < PPT; p++) {
        #pragma unroll
        for (int i = 0; i < PF; i++)
          qbuf[bank][p * PF + i] = static_cast<float>((w_local[p] >> (i * 4)) & 0xFu);
      }

      float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float group_xsum = 0.0f;
      // Dot in the EXACT dequant_and_dot4 order (pack p, value i).
      #pragma unroll
      for (int p = 0; p < PPT; p++) {
        #pragma unroll
        for (int i = 0; i < PF; i++) {
          group_qdot4[i & 3] += x_local[p * PF + i] * qbuf[bank][p * PF + i];
          group_xsum += x_local[p * PF + i];
        }
      }

      int k_val = k_base + lane * VPT;
      int group_idx = k_val / GROUP_SIZE;
      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;

      bank ^= 1;
      w_local[0] = w_next[0];
      w_local[1] = w_next[1];
    }

    if (!valid) continue;

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
      out[m * N + n] = from_float<T>(acc);
    }
  }
}

// V3: 16B logical weight window (two consecutive 8B aligned volatile 32-bit
// pairs) - the m3-V3 32B pattern at the tiled 8B slice width.
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ __launch_bounds__(QGD_TILE_N_MAX * WARP_SIZE)
void qmv_tiled_qgd_v3_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,
    int n_tiles) {
  constexpr int PF = pack_factor_u32<4>;
  constexpr int PPT = packs_per_thread<4>;
  constexpr int VPT = values_per_thread<4>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int n = tile * tile_n + threadIdx.y;
    const bool valid = (m < M && n < N);
    const int clamped_n = (n < N) ? n : 0;
    const uint32_t* w_row = w + clamped_n * w_stride;
    const ScaleT* s_row = scales + clamped_n * num_groups;
    const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BSK) {
      __syncthreads();
      for (int i = tid; i < BSK; i += nthreads) {
        int k = k_base + i;
        x_shared[i] = (k < K) ? to_float(x_row[k]) : 0.0f;
      }
      __syncthreads();

      if (!valid) continue;

      float x_local[VPT];
      #pragma unroll
      for (int i = 0; i < VPT; i++)
        x_local[i] = x_shared[lane * VPT + i];

      int w_offset = k_base / PF + lane * PPT;

      float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float group_xsum = 0.0f;

      int k_val = k_base + lane * VPT;
      int group_idx = k_val / GROUP_SIZE;

      uint32_t w_local[PPT];
      if (k_base + BSK <= K) {
        // Full tile in-row: the 16B window (k_val..k_val+31) is in-row for
        // every lane (k_val + 31 <= k_base + 511 < K). Half 0 is the real
        // slice (used in the math); half 1 is read + elision-proofed.
        const volatile uint32_t* pw =
            reinterpret_cast<const volatile uint32_t*>(&w_row[w_offset]);
        w_local[0] = pw[0];
        w_local[1] = pw[1];
        uint32_t wv1x = pw[2], wv1y = pw[3];
        asm volatile("" : : "r"(wv1x), "r"(wv1y));
      } else {
        // Tail: stock per-pack guarded loads for the 8B slice; the second
        // half of the 16B window is read ONLY when the window is in-row
        // (k_val + 32 <= K - the values k_val+16..k_val+31 are never used
        // in the math; the clamp changes the access pattern only), and is
        // elision-proofed so the widened window is genuinely touched.
        #pragma unroll
        for (int p = 0; p < PPT; p++)
          w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
        uint32_t wv1x = 0u, wv1y = 0u;
        if (k_val + 32 <= K) {
          const volatile uint32_t* pw =
              reinterpret_cast<const volatile uint32_t*>(&w_row[w_offset]);
          wv1x = pw[2];
          wv1y = pw[3];
        }
        asm volatile("" : : "r"(wv1x), "r"(wv1y));
      }

      #pragma unroll
      for (int p = 0; p < PPT; p++)
        dequant_and_dot4<4>(w_local[p], &x_local[p * PF], group_qdot4, group_xsum);

      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
    }

    if (!valid) continue;

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
      out[m * N + n] = from_float<T>(acc);
    }
  }
}

// V4: batched-x load order (16B vector batches in the x staging).
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ __launch_bounds__(QGD_TILE_N_MAX * WARP_SIZE)
void qmv_tiled_qgd_v4_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,
    int n_tiles) {
  constexpr int PF = pack_factor_u32<4>;
  constexpr int PPT = packs_per_thread<4>;
  constexpr int VPT = values_per_thread<4>;
  constexpr int BSK = VPT * WARP_SIZE;
  constexpr int XVEC = 16 / (int)sizeof(T);  // 8 for bf16/f16

  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int n = tile * tile_n + threadIdx.y;
    const bool valid = (m < M && n < N);
    const int clamped_n = (n < N) ? n : 0;
    const uint32_t* w_row = w + clamped_n * w_stride;
    const ScaleT* s_row = scales + clamped_n * num_groups;
    const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BSK) {
      __syncthreads();
      // Batched x staging: 16B vector batches (XVEC values) wherever the
      // chunk start is 16B-aligned, scalar for the rest - the x loads are
      // a single batched group before the weight loads. Duplicate writes
      // across overlapping batches are idempotent (same values).
      for (int i = tid; i < BSK; i += nthreads) {
        if ((i & (XVEC - 1)) == 0) {
          if (i + XVEC <= K) {
            const uint4 v = *reinterpret_cast<const uint4*>(x_row + i);
            const T* tv = reinterpret_cast<const T*>(&v);
            #pragma unroll
            for (int j = 0; j < XVEC; j++)
              x_shared[i + j] = to_float(tv[j]);
          } else {
            #pragma unroll
            for (int j = 0; j < XVEC; j++)
              x_shared[i + j] = (i + j < K) ? to_float(x_row[i + j]) : 0.0f;
          }
        } else {
          x_shared[i] = (i < K) ? to_float(x_row[i]) : 0.0f;
        }
      }
      __syncthreads();

      if (!valid) continue;

      float x_local[VPT];
      #pragma unroll
      for (int i = 0; i < VPT; i++)
        x_local[i] = x_shared[lane * VPT + i];

      int w_offset = k_base / PF + lane * PPT;

      float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float group_xsum = 0.0f;

      int k_val = k_base + lane * VPT;
      int group_idx = k_val / GROUP_SIZE;

      uint32_t w_local[PPT];
      if (k_base + BSK <= K) {
        load_weight_vec_streaming<4>(w_row + w_offset, w_local);
      } else {
        #pragma unroll
        for (int p = 0; p < PPT; p++)
          w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      }

      #pragma unroll
      for (int p = 0; p < PPT; p++)
        dequant_and_dot4<4>(w_local[p], &x_local[p * PF], group_qdot4, group_xsum);

      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
    }

    if (!valid) continue;

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
      out[m * N + n] = from_float<T>(acc);
    }
  }
}

} // namespace mlx::core::rocm
