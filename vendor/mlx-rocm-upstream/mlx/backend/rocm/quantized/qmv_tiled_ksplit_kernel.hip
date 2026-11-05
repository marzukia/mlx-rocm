// K-split (M = 1) parallelism variant of the stock tiled QMV kernel, for the
// qmv_gdn class. Chartered by the gen2 ceiling analysis (t2040): the stock
// tiled launch (512 blocks, each grid-striding 2 column tiles x a long
// per-tile k-chain with 2 x uint32 per-lane streaming loads and NO software
// pipelining) puts ~128 KB of weight traffic in flight vs the ~273 KB
// needed to saturate the 546 GB/s HBM floor at ~500 ns HBM latency (546 GB/s
// x 500 ns). This variant is the chartered parallelism change: 4x K-splits.
//
// Grid: dim3(M, n_tiles, 4) - 4x the block count (4096 blocks at N = 8192,
// tile_n = 8), ONE column tile per block, NO grid-stride. Block k = blockIdx.z
// covers K values [k * k_per_split, (k + 1) * k_per_split): for K = 8192 that
// is 4 BSK = 512 k-tiles per block (vs 16 per stock column tile), i.e. the
// per-block serial k-chain is 4x shorter and blocks retire 4x faster.
//
// Per-split accumulation is the STOCK per-tile order exactly: for each
// BSK = 512 k-tile in [k0, k1) the per-lane 2 x uint32 streaming weight load +
// dequant_and_dot4 (pack p, value i order) + scale * reduce_qdot4 + bias *
// xsum applied once per group - so each partial equals the stock `acc` after
// the same tile sequence starting from zero. The 4x partials [M, N, 4]
// (float) are reduced by a second kernel with a per-column left fold
// ((p0 + p1) + p2) + p3. Because the re-association differs from the stock
// single accumulator, the set-env output is CLOSE but NOT bit-identical to
// stock (a few ULPs at most for 4-bit group-64 data); the unset-env path runs
// the stock tiled kernel verbatim and is bit-identical by construction.
//
// Alignment contract (enforced by the host gate): the split base
// k_per_split = K / 4 must be >= 512 (one full BSK tile per split) and a
// multiple of 512 (k-tile/pack boundaries stay aligned; every split then is
// an exact multiple of BSK, so the stock fast path covers every tile - no
// tail branch). K = 8192 -> base 2048 = 4 x BSK (the chartered class);
// K = 2048 -> base 512 = 1 x BSK. Other K fall back to stock on the host.
//
// 4-bit only (the dispatch gate is bits == 4); x dtype bf16/f16 (the stock
// tiled gate). DEFAULT OFF: MLX_ROCM_QGD_KSPLIT unset -> stock tiled verbatim.

#include "mlx/backend/rocm/quantized/qdequant.hpp"
#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

constexpr int KSP_TILE_N_MAX = 32; // max tile_n for __launch_bounds__
constexpr int KSP_S = 4; // chartered 4x K-splits

// Per-split partials writer. One column tile per block (no grid-stride); the
// k loop covers only this split's K range [k0, k1). The per-tile body is the
// stock qmv_tiled_kernel 4-bit body verbatim (same BSK = 512, same x staging,
// same per-lane load, same dequant/FMA order).
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ __launch_bounds__(KSP_TILE_N_MAX * WARP_SIZE)
void qmv_tiled_ksplit_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint32_t* __restrict__ w,    // [N, K/pack_factor] as uint32
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    float* __restrict__ partials,      // [M, N, KSP_S]
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,
    int k_per_split) {
  constexpr int PF = pack_factor_u32<4>;
  constexpr int PPT = packs_per_thread<4>;
  constexpr int VPT = values_per_thread<4>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int m = blockIdx.x;
  const int split = blockIdx.z;
  const int tile = blockIdx.y;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;
  // The host gate guarantees k_per_split % BSK == 0, so every k_base below
  // is a multiple of BSK (and of PF) - the stock fast path is valid for all
  // tiles of this split.
  const int k0 = split * k_per_split;
  const int k1 = k0 + k_per_split;

  const int n = tile * tile_n + threadIdx.y;
  const bool valid = (m < M && n < N);
  const int clamped_n = (n < N) ? n : 0;
  const uint32_t* w_row = w + clamped_n * w_stride;
  const ScaleT* s_row = scales + clamped_n * num_groups;
  const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

  float acc = 0.0f;

  for (int k_base = k0; k_base < k1; k_base += BSK) {
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
    // Fast path covers the whole split by the alignment contract; the
    // bounded fallback is kept for structural parity with the stock kernel.
    if (k_base + BSK <= K) {
      load_weight_vec_streaming<4>(w_row + w_offset, w_local);
    } else {
      #pragma unroll
      for (int p = 0; p < PPT; p++) {
        w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      }
    }

    #pragma unroll
    for (int p = 0; p < PPT; p++) {
      dequant_and_dot4<4>(w_local[p], &x_local[p * PF], group_qdot4, group_xsum);
    }

    float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
    float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
    acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
  }

  if (!valid) return;

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    partials[(m * N + n) * KSP_S + split] = acc;
  }
}

// Per-column left fold over the KSP_S float partials:
// ((p0 + p1) + p2) + p3 -> out (bf16/f16). Fixed order - deterministic.
template <typename T>
__global__ void qmv_tiled_ksplit_reduce_kernel(
    const float* __restrict__ partials,
    T* __restrict__ out,
    int M,
    int N) {
  const int m = blockIdx.y;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float acc = partials[(m * N + i) * KSP_S + 0];
    #pragma unroll
    for (int s = 1; s < KSP_S; s++)
      acc += partials[(m * N + i) * KSP_S + s];
    out[m * N + i] = from_float<T>(acc);
  }
}

} // namespace mlx::core::rocm
