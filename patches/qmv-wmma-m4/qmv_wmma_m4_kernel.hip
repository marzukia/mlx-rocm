// t1339: WMMA (rocwmma) M<=4 QMV kernel for gfx1201 (RDNA4) — K=3 MTP verify.
//
// One block = one wave (32 threads) = one 16-column n-tile x all M rows.
// The 16x16 4-bit weight tile is dequantized once per k-tile (the
// per-group scale/bias is constant within the 16-wide tile, since
// group_size >= 32 > 16) into the fp16 B fragment staging; A = x tile.
// Two k-tiles per iteration on two independent f32 accumulators
// (ILP = 2). The NEXT iteration's global data is prefetched into
// registers while the current iteration's LDS/fragment/mma work runs,
// and both x and the packed weight tile are 16B vector loads (the v1
// scalar strided 4B/8B loads + unpipelined latency chain measured
// 6-9x slow of the rows reference; this is the pipelined v3 form).
//
// Thread mapping (lane = 32): n = lane>>1 (0..15, the weight row /
// output column), h = lane&1 (half of the 16-wide tile):
//   x:  tile t at kt0: 8 consecutive values at x[n*K + kt0 + 8h]
//       (one 16B vector; valid rows m < M only, M <= 4, rest zeroed)
//   w:  the 32-wide k-pair: 4 consecutive words at
//       w[col * row_words + kt/2 + 4h] (one 16B vector); h = 0 decodes
//       tile 0 (words 0..3 -> kk 0..15), h = 1 decodes tile 1 (words
//       4..7 -> kk 16..31).
//
// The 16x16x16 f32-accumulate wmma is native for bf16 AND fp16 inputs
// (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32); the dequant rounds to
// fp16 (10-bit mantissa — bf16's 8-bit mantissa missed the 1e-2 bar)
// and x (bf16/f16) converts to fp16 near-exactly.
//
// Precision: STATISTICAL (not row-exact): fp16 dequant rounding +
// f32-accumulate wmma vs the rows path's f32 scalar FMA. Gate:
// maxdiff <= 1e-2 vs the rows reference, finite.
//
// Dispatch (qmm.hip): MLX_QMV_WMMA=1 (DEFAULT OFF), bits_ == 4,
// M = 1..4, N % 16 == 0, K % 16 == 0, x NOT float32 (the affine op
// promotes x to the scales dtype, so f32 scales reach here as f32 x
// and decline — bf16 scales keep x bf16, the model layout).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <type_traits>

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#include <rocwmma/rocwmma.hpp>
#define ROCM_QMV_WMMA_M4 1
#else
#define ROCM_QMV_WMMA_M4 0
#endif

namespace mlx::core::rocm {

template <typename T, typename ST, int GROUP_SIZE>
__global__ void __launch_bounds__(32)
qmv_wmma_m4_kernel(
    const T* __restrict__ x, const uint32_t* __restrict__ w,
    const ST* __restrict__ scales, const T* __restrict__ biases,
    T* __restrict__ out, int M, int N, int K, bool has_bias)
{
#if ROCM_QMV_WMMA_M4
  const int lane = threadIdx.x & 31;
  const int n0 = static_cast<int>(blockIdx.x) * 16;
  const int n = lane >> 1;        // 0..15 — the weight row / output col index
  const int h = lane & 1;         // half of the 16-wide tile
  const int col = n0 + n;         // N % 16 == 0 -> col < N always
  const int row_words = K >> 3;  // packed 4-bit weight row (words: 8 nibbles / word)
  const int ng = K / GROUP_SIZE;

  __shared__ __half A[2][16][16];  // x tiles (t, m, k)
  __shared__ __half B[2][16][16];  // w dequant tiles (t, n, k)
  __shared__ float Cs[16][16];

  rocwmma::fragment<
      rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major>
      fA[2];
  rocwmma::fragment<
      rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major>
      fB[2];
  rocwmma::fragment<
      rocwmma::accumulator, 16, 16, 16, float>
      fC[2];

  for (int e = 0; e < fC[0].num_elements; ++e) {
    fC[0].x[e] = 0.f;
    fC[1].x[e] = 0.f;
  }

  // x (bf16 or f16, 2B elements) -> fp16 (the wmma input class). One
  // 32-bit register holds TWO 16-bit values: xh(v) = low, xh(v>>16) = high.
  auto xh = [&](uint32_t v) -> __half {
    if constexpr (std::is_same_v<T, __half>) {
      return *reinterpret_cast<const __half*>(&v);
    } else {
      return __float2half(static_cast<float>(
          *reinterpret_cast<const hip_bfloat16*>(&v)));
    }
  };
  // stage one 32-bit register (2 x values) into tile t at kk off, +1.
  // xvt: main loop — 16 consecutive kk per thread (off 0..14, step 2).
  // xv:  tail — 8 kk per thread at 8h..8h+7 (off 0..6, step 2).
  auto xvt = [&](uint32_t v, int t, int off) {
    A[t][n][off] = xh(v);
    A[t][n][off + 1] = xh(v >> 16);
  };
  auto xv = [&](uint32_t v, int t, int off) {
    A[t][n][8 * h + off] = xh(v);
    A[t][n][8 * h + off + 1] = xh(v >> 16);
  };

  // w nibble -> s*w + b -> fp16.
  auto bval = [&](uint32_t word, int j, float s, float b) {
    return __float2half(s * static_cast<float>((word >> (4 * j)) & 0xF) + b);
  };

  // 16 consecutive x values (32B = two 16B vectors) for the tile at kt0:
  // row m = n; h = 0 -> tile 0 (values kt0..kt0+15), h = 1 -> tile 1.
  // Rows m >= M load nothing (LDS stays zero).
  auto load_x8 = [&](int kt0) -> uint4 {
    if (n < M) {
      return *reinterpret_cast<const uint4*>(
          x + static_cast<size_t>(n) * K + kt0 + 8 * h);
    }
    return make_uint4(0u, 0u, 0u, 0u);
  };
  auto load_x16 = [&](int kt0) -> uint4 {
    if (n < M) {
      const uint4* p = reinterpret_cast<const uint4*>(
          x + static_cast<size_t>(n) * K + kt0 + 16 * h);
      return p[0];
    }
    return make_uint4(0u, 0u, 0u, 0u);
  };
  auto load_x16b = [&](int kt0) -> uint4 {
    if (n < M) {
      const uint4* p = reinterpret_cast<const uint4*>(
          x + static_cast<size_t>(n) * K + kt0 + 16 * h);
      return p[1];
    }
    return make_uint4(0u, 0u, 0u, 0u);
  };

  // 2 consecutive packed words (one 8B vector) for the 32-wide k-pair
  // starting at kt0: words kt0/8 + 2h .. +1 (8 nibbles per word;
  // h = 0 -> tile 0's 16 values, h = 1 -> tile 1's 16 values).
  auto load_w2 = [&](int kt0) -> uint2 {
    return *reinterpret_cast<const uint2*>(
        w + static_cast<size_t>(col) * row_words + (kt0 >> 3) + 2 * h);
  };

  const int kt_last = (K / 32) * 32;  // full 32-pairs only
  uint4 xa = load_x16(0);
  uint4 xb = load_x16b(0);
  uint2 wv = load_w2(0);
  float s0 = static_cast<float>(scales[static_cast<size_t>(col) * ng + 0]);
  float b0 =
      has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + 0])
               : 0.0f;
  float s1 = static_cast<float>(scales[static_cast<size_t>(col) * ng + 16 / GROUP_SIZE]);
  float b1 =
      has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + 16 / GROUP_SIZE])
               : 0.0f;

  for (int kt = 0; kt < kt_last; kt += 32) {
    // 1. Stage the prefetched registers into LDS: each thread owns 16
    // consecutive kk of ITS tile (h = 0 -> A[0], h = 1 -> A[1]).
    xvt(xa.x, h, 0);
    xvt(xa.y, h, 2);
    xvt(xa.z, h, 4);
    xvt(xa.w, h, 6);
    xvt(xb.x, h, 8);
    xvt(xb.y, h, 10);
    xvt(xb.z, h, 12);
    xvt(xb.w, h, 14);
    if (h == 0) {
      // h = 0 decodes tile 0 (2 words -> 16 values).
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        B[0][n][j] = bval(wv.x, j, s0, b0);
        B[0][n][8 + j] = bval(wv.y, j, s0, b0);
      }
    } else {
      // h = 1 decodes tile 1 (2 words -> 16 values).
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        B[1][n][j] = bval(wv.x, j, s1, b1);
        B[1][n][8 + j] = bval(wv.y, j, s1, b1);
      }
    }
    __syncthreads();

    // 2. Fragment loads (LDS -> registers).
    rocwmma::load_matrix_sync(fA[0], &A[0][0][0], 16);
    rocwmma::load_matrix_sync(fB[0], &B[0][0][0], 16);
    rocwmma::load_matrix_sync(fA[1], &A[1][0][0], 16);
    rocwmma::load_matrix_sync(fB[1], &B[1][0][0], 16);

    // 3. Prefetch the next k-pair (overlaps the mma + barrier below).
    if (kt + 32 < kt_last) {
      const int g0 = (kt + 32) / GROUP_SIZE;
      const int g1 = (kt + 48) / GROUP_SIZE;
      s0 = static_cast<float>(scales[static_cast<size_t>(col) * ng + g0]);
      b0 =
          has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g0])
                   : 0.0f;
      s1 = static_cast<float>(scales[static_cast<size_t>(col) * ng + g1]);
      b1 =
          has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g1])
                   : 0.0f;
      xa = load_x16(kt + 32);
      xb = load_x16b(kt + 32);
      wv = load_w2(kt + 32);
    }

    // 4. Two independent mma chains (ILP = 2).
    rocwmma::mma_sync(fC[0], fA[0], fB[0], fC[0]);
    rocwmma::mma_sync(fC[1], fA[1], fB[1], fC[1]);
    __syncthreads();
  }

  // Tail: K % 32 != 0 -> one final 16-wide tile (K % 16 == 0 at dispatch).
  if (kt_last < K) {
    const int kt = kt_last;
    const int g = kt / GROUP_SIZE;
    const float s = static_cast<float>(scales[static_cast<size_t>(col) * ng + g]);
    const float b =
        has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g])
                 : 0.0f;
    // one word per thread (8 nibbles = 8 values): word kt/8 + h
    const uint32_t w1 = w[static_cast<size_t>(col) * row_words + (kt >> 3) + h];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      B[0][n][8 * h + j] = bval(w1, j, s, b);
    const uint4 xvt = load_x8(kt);
    xv(xvt.x, 0, 0);
    xv(xvt.y, 0, 2);
    xv(xvt.z, 0, 4);
    xv(xvt.w, 0, 6);
    __syncthreads();
    rocwmma::load_matrix_sync(fA[0], &A[0][0][0], 16);
    rocwmma::load_matrix_sync(fB[0], &B[0][0][0], 16);
    rocwmma::mma_sync(fC[0], fA[0], fB[0], fC[0]);
    __syncthreads();
  }

  // Merge the two accumulators, f32 -> LDS -> T-convert -> global.
  for (int e = 0; e < fC[0].num_elements; ++e)
    fC[0].x[e] += fC[1].x[e];
  rocwmma::store_matrix_sync(&Cs[0][0], fC[0], 16, rocwmma::mem_row_major);
  __syncthreads();
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    // True 16x16x16 f32 acc fragment layout (probe-verified on gfx1201):
    // lane L, element e -> row 8*(L>>4)+e, col L&15 (a lane owns 8
    // consecutive rows of one column).
    const int m = 8 * (lane >> 4) + e;
    const int c = lane & 15;
    if (m < M)
      out[static_cast<size_t>(m) * N + n0 + c] = static_cast<T>(Cs[m][c]);
  }
#endif
}

}  // namespace mlx::core::rocm
