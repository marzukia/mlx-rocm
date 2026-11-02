// t1428: WMMA (rocwmma) M=4 QMV kernel for gfx1201 (RDNA4) — K=3 MTP
// verify, v4c M=4-native (zero-LDS, direct-fragment, SW-pipelined).
//
// v1: LDS-staged, 1-wave. Correct; parity-or-better isolated, +25-44us
// full-stream. v2/v3: multi-wave + barrier-free; losses monotonically in
// waves. v4 (ILP=2): unit 24/24, but lost every isolated class
// (K=4096 N=12288: 379us vs rows 215). v4b (128-k super-chunks, 16B
// vector loads, ILP=8): identical timing (~380us) — the mma inline-asm
// fences the compiler from overlapping chunk i+1 loads with chunk i mma,
// so each 32-chunk pays its full load latency: K-scan 45/56/95/383us
// at K=512/1024/2048/4096 (per-chunk latency, not bandwidth).
//
// v4c (this file): explicit 3-stage register pipeline at 32-chunk
// granularity, position-indexed buffers B[0..2]. Iteration i (kt = 32i)
// processes B[i%3], which was loaded two iterations ago; B[(i+1)%3]
// holds chunk i+1; B[(i+2)%3] is refilled with chunk i+2 (or, in the
// closing window, set to the last chunk) BEFORE the two mma of chunk i
// execute — so while the mma run, the global loads of chunks i+1 and
// i+2 are already in flight. Per lane per 32-chunk: B = 1 x 16B vector
// (4 consecutive 32-bit words = 2 tiles x word_half; scalar 4B fallback
// when the row base is not 16B-aligned), A = 2 x 16B vectors on the 8
// live lanes (one per 16-wide tile; 24 dead lanes zero), scale/bias =
// 2 x 2B. Two mma per chunk on two independent f32 accumulators (ILP = 2;
// the mma unit saturates ~10ns/mma at ILP >= 4 per the 1-wave probe, so
// mma cost hides under the load latency, not the bottleneck).
//
// K%32 == 16 tail: a single extra 16-wide tile (1 mma) — v4b read 16 k
// out of range here; fixed.
//
// Fragment layouts (probe-verified on gfx1201): A (matrix_a row_major)
// lane L, element e -> (row, k) = (L&15, 8*(L>>4)+e) — at M=4 only
// lanes 0..3 (k 0..7) / 16..19 (k 8..15) are live. B (matrix_b
// col_major) lane L, element e -> (k, n) = (8*(L>>4)+e, L&15) — the 8
// nibbles of one 32-bit word are the 8 elements of one tile half. ACC
// lane L, element e -> row 8*(L>>4)+e, col L&15 — M<=4 rows sit on
// lanes 0..15 elements 0..M-1: independent per-thread stores, no
// barrier.
//
// Precision: STATISTICAL (fp16 dequant, 10-bit mantissa; x bf16/f16
// converts near-exactly). Gate: maxdiff <= 1e-2 vs the rows reference.
// Dispatch (qmm.hip): MLX_QMV_WMMA=1 (DEFAULT OFF), bits_ == 4, M = 4
// ONLY (M<=3 falls through to rows/scalar — M=1 is an inherent loss:
// 15/16 rows dead in the 16-row mma), N % 16 == 0, K % 16 == 0, x NOT
// float32 (f32 scales keep x bf16 in the model layout). Block size:
// MLX_QMV_WMMA_WAVES (1/2/4, dispatch default 2 — v4 has no inter-wave
// coupling; waves per block are pure scheduling granularity; v2 showed
// monotonically worse full-stream results at 2/4; the v4 round measured
// 1/2/4 explicitly); the launch falls back to the largest power-of-2
// wave count that divides N/16.
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

template <typename T, typename ST, int GROUP_SIZE, int WAVES>
__global__ void __launch_bounds__(32 * WAVES)
qmv_wmma_m4_kernel(
    const T* __restrict__ x, const uint32_t* __restrict__ w,
    const ST* __restrict__ scales, const T* __restrict__ biases,
    T* __restrict__ out, int M, int N, int K, bool has_bias)
{
#if ROCM_QMV_WMMA_M4
  const int g = threadIdx.x >> 5;  // 0..WAVES-1 — the wave in this block
  const int lane = threadIdx.x & 31;
  const int n0 = static_cast<int>(blockIdx.x) * (16 * WAVES) + g * 16;
  const int col = n0 + (lane & 15);   // B fragment n-index of this lane
  const int word_half = (lane >> 4);  // 0: k 0..7 of the tile, 1: k 8..15
  const int row_words = K >> 3;  // packed 4-bit weight row (words)
  const int ng = K / GROUP_SIZE;
  const bool a_live = (lane < 4) || (lane >= 16 && lane < 20);
  const int a_row = lane & 15;  // 0..3 for the live lanes
  const bool b_vec =
      (static_cast<size_t>(col) * row_words & 3) == 0;  // 16B-aligned B
  const bool b16 = (K & 31) == 16;  // trailing single 16-wide tile

  rocwmma::fragment<
      rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major>
      fA0, fA1;
  rocwmma::fragment<
      rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major>
      fB0, fB1;
  rocwmma::fragment<
      rocwmma::accumulator, 16, 16, 16, float>
      fC0, fC1;
  for (int e = 0; e < 4; ++e) {
    fC0.x[e] = 0.f;
    fC1.x[e] = 0.f;
  }

  // x (bf16 or f16, 2B elements) -> fp16 (the wmma input class).
  auto xh = [&](const T& v) -> __half {
    if constexpr (std::is_same_v<T, __half>) {
      return v;
    } else {
      return __float2half(static_cast<float>(v));
    }
  };

  // One 32-k chunk of raw data: scale/bias for its two 32-sub-chunks,
  // the lane's 4 B words (2 tiles x word_half), and the 8 A 2B values
  // per 16-wide tile (tile 0 = k kt..kt+15, tile 1 = kt+16..kt+31) on
  // the live lanes. A is stored as two 16B vectors (4 32-bit slots each,
  // one 2B value per slot, little-endian low half). When the chunk's
  // upper half runs past K (K%32 == 16, last chunk) the OOR A values
  // are zeroed (B x 0 = 0 in the mma) and its scale index clamped.
  struct Ch {
    float sc[2], bc[2];
    uint32_t bw[4];
    uint32_t av[2][4];  // [tile][2B-pair], low 16 bits = first element
  };
  auto load_ch = [&](Ch& d, int kt) {
    const int si0 = kt / GROUP_SIZE;
    int si1 = (kt + 32) / GROUP_SIZE;
    if (si1 >= ng)
      si1 = ng - 1;
    d.sc[0] = static_cast<float>(scales[static_cast<size_t>(col) * ng + si0]);
    d.bc[0] = has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + si0]) : 0.0f;
    d.sc[1] = static_cast<float>(scales[static_cast<size_t>(col) * ng + si1]);
    d.bc[1] = has_bias ? static_cast<float>(biases[static_cast<size_t>(col) * ng + si1]) : 0.0f;
    const bool short_t = (kt + 32 > K);  // upper 16 k out of range
    const uint32_t* pw = w + static_cast<size_t>(col) * row_words + (kt >> 3);
    if (b_vec && !short_t) {
      const uint4 t4 = reinterpret_cast<const uint4*>(pw)[0];
      d.bw[0] = t4.x;
      d.bw[1] = t4.y;
      d.bw[2] = t4.z;
      d.bw[3] = t4.w;
    } else if (b_vec) {
      d.bw[0] = pw[0];
      d.bw[1] = pw[1];
      d.bw[2] = pw[0];  // OOR words clamped; A is zeroed there
      d.bw[3] = pw[1];
    } else {
      d.bw[0] = pw[0];
      d.bw[1] = pw[1];
      d.bw[2] = short_t ? pw[0] : pw[2];
      d.bw[3] = short_t ? pw[1] : pw[3];
    }
    if (a_live) {
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        if (short_t && t == 1) {
          d.av[1][0] = 0;
          d.av[1][1] = 0;
          d.av[1][2] = 0;
          d.av[1][3] = 0;
          continue;
        }
        const uint4 t4 = reinterpret_cast<const uint4*>(
            x + static_cast<size_t>(a_row) * K + kt + 16 * t +
            8 * word_half)[0];
        const uint2* p2 = reinterpret_cast<const uint2*>(&t4);
        d.av[t][0] = p2[0].x;
        d.av[t][1] = p2[0].y;
        d.av[t][2] = p2[1].x;
        d.av[t][3] = p2[1].y;
      }
    } else {
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          d.av[t][i] = 0;
    }
  };

  // Dequant chunk d's two 16-tiles into fA0/fA1, fB0/fB1, then the two
  // mma (ILP = 2). The loads of the NEXT chunks are already in flight
  // (issued before this block in the loop), so the mma overlap them.
  auto run_ch = [&](const Ch& d) {
    const float s0 = d.sc[0];
    const float b0 = d.bc[0];
    const float s1 = d.sc[1];
    const float b1 = d.bc[1];
#pragma unroll
    for (int e = 0; e < 8; ++e)
      fB0.x[e] = __float2half(
          s0 * static_cast<float>(
                  ((e >> 3) ? d.bw[1] : d.bw[0]) >> (4 * (e & 7))) &
              0xF) +
          b0);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      fB1.x[e] = __float2half(
          s1 * static_cast<float>(
                  ((e >> 3) ? d.bw[3] : d.bw[2]) >> (4 * (e & 7))) &
              0xF) +
          b1);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t lo = d.av[t][2 * h + 0];
        const uint32_t hi = d.av[t][2 * h + 1];
        auto* fa = (t ? &fA1 : &fA0);
        fa->x[4 * h + 0] = xh(reinterpret_cast<const T*>(&lo)[0]);
        fa->x[4 * h + 1] = xh(reinterpret_cast<const T*>(&lo)[1]);
        fa->x[4 * h + 2] = xh(reinterpret_cast<const T*>(&hi)[0]);
        fa->x[4 * h + 3] = xh(reinterpret_cast<const T*>(&hi)[1]);
      }
    }
    rocwmma::mma_sync(fC0, fA0, fB0, fC0);
    rocwmma::mma_sync(fC1, fA1, fB1, fC1);
  };

  // 3-stage pipeline, position-indexed.
  Ch buf[3];
  load_ch(buf[0], 0);
  const int k32 = (K / 32) * 32;
  if (k32 >= 32)
    load_ch(buf[1], 32);
  if (k32 >= 64)
    load_ch(buf[2], 64);
  for (int i = 0; i * 32 < k32; ++i) {
    // Refill the far slot with chunk i+2 (or, in the closing window,
    // keep the last chunk there so the index advance makes it current).
    if (i * 32 + 64 < k32) {
      load_ch(buf[(i + 2) % 3], i * 32 + 64);
    } else if (i * 32 + 32 < k32) {
      buf[(i + 2) % 3] = buf[(i + 1) % 3];
    }
    run_ch(buf[i % 3]);
  }
  // K%32 == 16 tail: one 16-wide tile (1 mma on fC1).
  if (b16) {
    const int kt = k32;
    const int sg0 = kt / GROUP_SIZE;
    const int sg = (sg0 >= ng) ? ng - 1 : sg0;  // clamp: K%32==16 tail, kt = K-16
    const float s =
        static_cast<float>(scales[static_cast<size_t>(col) * ng + sg]);
    const float b = has_bias
                       ? static_cast<float>(
                             biases[static_cast<size_t>(col) * ng + sg])
                       : 0.0f;
    const uint32_t* pw =
        w + static_cast<size_t>(col) * row_words + (kt >> 3);
    const uint32_t w0 = pw[0];
    const uint32_t w1 = pw[1];
    if (a_live) {
      const uint4 t4 = reinterpret_cast<const uint4*>(
          x + static_cast<size_t>(a_row) * K + kt + 8 * word_half)[0];
      const uint2* p2 = reinterpret_cast<const uint2*>(&t4);
      fA1.x[0] = xh(reinterpret_cast<const T*>(&p2[0].x)[0]);
      fA1.x[1] = xh(reinterpret_cast<const T*>(&p2[0].x)[1]);
      fA1.x[2] = xh(reinterpret_cast<const T*>(&p2[0].y)[0]);
      fA1.x[3] = xh(reinterpret_cast<const T*>(&p2[0].y)[1]);
      fA1.x[4] = xh(reinterpret_cast<const T*>(&p2[1].x)[0]);
      fA1.x[5] = xh(reinterpret_cast<const T*>(&p2[1].x)[1]);
      fA1.x[6] = xh(reinterpret_cast<const T*>(&p2[1].y)[0]);
      fA1.x[7] = xh(reinterpret_cast<const T*>(&p2[1].y)[1]);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        fA1.x[e] = 0;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e)
      fB1.x[e] = __float2half(
          s * static_cast<float>(
                  ((e >> 3) ? w1 : w0) >> (4 * (e & 7))) &
              0xF) +
          b);
    rocwmma::mma_sync(fC1, fA1, fB1, fC1);
  }

  // Sum the accumulators -> fC0, direct acc -> global. Probe-verified
  // acc layout: lane L, element e -> row 8*(L>>4)+e, col L&15; for M = 4
  // only rows 0..3 (lanes 0..15, elements 0..M-1) are valid —
  // independent per-thread stores.
  for (int e = 0; e < 4; ++e)
    fC0.x[e] += fC1.x[e];
  if (lane < 16) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (e < M)
        out[static_cast<size_t>(e) * N + n0 + lane] = static_cast<T>(fC0.x[e]);
  }
#endif
}

}  // namespace mlx::core::rocm
