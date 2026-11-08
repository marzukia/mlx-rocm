// Copyright © 2025 Apple Inc.

// Multi-row dense wide-load QMV (4/8-bit affine, decode, M = 1..4).
//
// One weight pass for up to 4 output rows (method-correction doc 2026-10-28,
// section 4 step 1, lever 1): each warp accumulates the block's rows in
// separate accumulators, in the EXACT tile order of qmv_wide_kernel, so every
// row is bit-identical to the per-row kernel (lossless gate: maxdiff 0.0).
// x is staged in LDS once per block (lever 2) instead of re-reading it from
// L2 for each column; 16 cols per block (lever 3) to cut tail/scheduling
// effects on the large shapes.
//
// WEIGHT REUSE (run-825): the dequantized weight tile is computed ONCE per
// k-tile (shift+mask+cvt hoisted out of the row loop into a register array)
// and MAC'd against ALL M rows, instead of re-dequantizing per row. The
// per-row math keeps the exact FMA order of qmv_wide_kernel, so every row
// stays bit-identical to the per-row kernel.
//
// LX HOIST (run-852): the per-row x-chunk-sum lx = sum_j sx[m*K + k0+j] is
// COLUMN-INVARIANT (independent of the output column n), but the run-825 kernel
// recomputed it for every one of the block's 16 columns (16x redundancy) and
// for every row (Mx redundancy). It is now computed ONCE per (row, k-tile)
// by the lane whose threadIdx.y equals the row (the 32-term 4-bit / 16-term
// 8-bit FADD chain in the EXACT order of the run-825 per-row lx: q-major,
// j-minor over the chunk, same values from sx), written to a 256-float
// DOUBLE-BUFFERED LDS buffer (buffer parity = k-iteration parity),
// __syncthreads, and read by all lanes. The double buffer is required for
// correctness: with a single buffer, thread (c, m) could overwrite
// s_lx[m][c] in k-iteration k+1 while another wavefront (c, r') is still
// reading s_lx[m][c] in k-iteration k's m-loop (different threadIdx.y =
// different wavefront = different speed; no barrier between the m-loop
// read and the next hoist write). The double buffer makes iteration k
// (buffer p) and k+1 (buffer p^1) use disjoint buffers, and the in-k
// barrier B_{k+1} orders iteration k's read before iteration k+2's write
// to buffer p. The per-row ALU increment
// drops from 66 to 34 ops (32 FMA + 2 FMA); the per-row math keeps the exact
// FMA order of qmv_wide_kernel (same chunk order, same inner unroll, same
// acc += s*la + b*lx), so every row stays bit-identical to the per-row
// kernel (lossless gate: maxdiff 0.0). The in-k-loop __syncthreads requires
// uniform k-iteration counts across lanes: the dispatch gate enables the
// hoist only when K % (32 * WPER) == 0 (K % 1024 == 0 for 4-bit,
// K % 512 == 0 for 8-bit); shapes that do not satisfy it run the run-825 path
// (LXHOIST = false, no in-k barrier).
//
// A/B (run-889): LXHOIST is a template parameter selected at dispatch by
// MLX_QMV_MR_LX (default 1 = hoist on; 0 = run-825 path). Each instantiation
// compiles only its m-loop path (if constexpr), so the A/B baseline
// (LXHOIST = false) is codegen-equivalent to the run-825 kernel (the x-staging
// barrier and the dispatch structure are the only shared code). This makes
// the gain gate (method doc section 2: gain >= 3% to count) measurable in
// one session, free of the cross-run drift recorded in REPORT.md section 6.1
// (17% on an unchanged kernel between back-to-back runs).
//
// Grid: (ceil(N/16), ceil(M/4)); block: (32, 16).
// LDS: min(4, M) * K * sizeof(T) bytes (32 KB at K = 4096, M = 4) + 1 KB
// (s_lx, double-buffered, always declared; used only by the hoist path).
// The 32 KB gate bounds the x-staging only; the 1 KB s_lx sits on top
// (33 KB total at the max shape, well within the 64 KB per-block limit).
//
// Dispatch gate (qmm.hip): 4/8-bit affine with bias, transposed, non-batched,
// M <= 4, K % 32 == 0, GROUP_SIZE in {32,64,128}, bf16/fp16 x, wave32,
// LDS fits (<= 32 KB). The hoist additionally requires K % (32 * WPER) == 0
// (K % 1024 == 0 for 4-bit, K % 512 == 0 for 8-bit); the MR dispatch gate
// itself is unchanged by the hoist. M = 5..8 keeps the per-row
// qmv_wide_kernel.
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE, int BITS, bool LXHOIST>
__global__ void __launch_bounds__(1024) qmv_wide_mr_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int MAXM = 4;
  constexpr int WPER = 128 / BITS;
  // Shared memory as a byte buffer (extern __shared__ is a single symbol;
  // declaring it as T[] conflicts across T instantiations). Cast to T*.
  extern __shared__ uint8_t sx_bytes[];
  T* sx = reinterpret_cast<T*>(sx_bytes);  // [nrows][K]
  // LX HOIST (run-852): per-row, per-lane x-chunk-sum buffer (256 floats =
  // 1 KB, DOUBLE-BUFFERED on k-iteration parity). Written once per (row,
  // k-tile) by the thread (threadIdx.x == lane, threadIdx.y == row); read
  // by all threads after __syncthreads(). The double buffer avoids the
  // single-buffer race (thread (c, m) overwriting s_lx[m][c] in k+1 while
  // another wavefront still reads it in k's m-loop). Declared
  // unconditionally (1 KB, always fits); used only by the hoist path.
  __shared__ float s_lx[2][MAXM][32];
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int m0 = blockIdx.y * MAXM;
  int nrows = M - m0;
  if (nrows > MAXM)
    nrows = MAXM;
  if (nrows < 0)
    nrows = 0;
  bool valid_col = (col < N);
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = (K * BITS) / 8;
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;

  // Stage x once per block (16B vector loads; K % 32 == 0 is gated, so
  // K % (16/sizeof(T)) == 0). All threads reach __syncthreads() — the
  // col guard is applied after it, not as an early return.
  {
    constexpr int VEC = 16 / sizeof(T);  // 8 for bf16/fp16
    int tid = threadIdx.y * 32 + lane;
    int nthreads = blockDim.y * 32;
    int nvec = K / VEC;
    for (int idx = tid; idx < nrows * nvec; idx += nthreads) {
      int r = idx / nvec, v = idx % nvec;
      *reinterpret_cast<uint4*>(&sx[r * K + v * VEC]) =
          *reinterpret_cast<const uint4*>(
              &x[static_cast<size_t>(m0 + r) * K + v * VEC]);
    }
  }
  __syncthreads();

  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    acc[m] = 0.0f;
  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    // Double-buffer parity (uniform across lanes: k - lane*WPER is a
    // multiple of 32*WPER, so the division is exact and lane-invariant).
    int p = ((k - lane * WPER) / (32 * WPER)) & 1;
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * BITS) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // WEIGHT REUSE (run-825): dequantize the weight tile ONCE per k-tile and
    // reuse it across all M rows, instead of re-dequantizing (shift+mask+
    // cvt) per row. The dequantized values are bit-identical to the per-row
    // dequant (same exact int->float conversion), and the per-row math below
    // keeps the EXACT FMA order of qmv_wide_kernel (same chunk order, same
    // inner unroll, same acc += s*la + b*lx), so every row stays bit-identical
    // to the per-row kernel (lossless gate: maxdiff 0.0). x from LDS (same
    // values as the per-column global reads).
    constexpr int WN = 128 / BITS;  // 32 (4-bit) / 16 (8-bit) dequant/ lane
    float wn[WN];
    if constexpr (BITS == 4) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wn[q * 4 + j] = static_cast<float>((wp[q] >> (8 * j)) & 0xFF);
    }
    if constexpr (LXHOIST) {
      // LX HOIST (run-852): compute the column-invariant per-row x-chunk-sum
      // ONCE per (row, k-tile). The FADD chain is the EXACT order of the
      // run-825 per-row lx (q-major, j-minor over the 32/16-value chunk, same
      // values from sx), so the value is bit-identical to the run-825 per-row
      // lx. The thread (lane, row) writes s_lx[row][lane]; all threads read
      // it after the barrier. When !has_bias the consumer multiplies by
      // b == 0.0f, so 0.0f * (finite chain) == 0.0f == the run-825
      // (unaccumulated) lx — the result is bit-identical in both cases.
      {
        int ly = threadIdx.y;
        if (ly < nrows) {
          float lx = 0.0f;
          if constexpr (BITS == 4) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int j = 0; j < 8; ++j)
                lx += static_cast<float>(sx[ly * K + k + q * 8 + j]);
          } else {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int j = 0; j < 4; ++j)
                lx += static_cast<float>(sx[ly * K + k + q * 4 + j]);
          }
          s_lx[p][ly][lane] = lx;
        }
      }
      __syncthreads();
    }
    if constexpr (LXHOIST) {
      for (int m = 0; m < MAXM; ++m) {
        if (m >= nrows)
          break;
        float la = 0.0f;
        if constexpr (BITS == 4) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int kb = k + q * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j)
              la += static_cast<float>(sx[m * K + kb + j]) * wn[q * 8 + j];
          }
        } else {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int kb = k + q * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j)
              la += static_cast<float>(sx[m * K + kb + j]) * wn[q * 4 + j];
          }
        }
        acc[m] += s * la + b * s_lx[p][m][lane];
      }
    } else {
      // run-825 path (A/B baseline): per-row lx accumulated in the same inner
      // loop (EXACT run-825 order: la FMA then lx FADD per value, guarded by
      // has_bias), so this instantiation is codegen-equivalent to the run-825
      // kernel.
      for (int m = 0; m < MAXM; ++m) {
        if (m >= nrows)
          break;
        float la = 0.0f, lx = 0.0f;
        if constexpr (BITS == 4) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int kb = k + q * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              float xv = static_cast<float>(sx[m * K + kb + j]);
              la += xv * wn[q * 8 + j];
              if (has_bias)
                lx += xv;
            }
          }
        } else {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int kb = k + q * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              float xv = static_cast<float>(sx[m * K + kb + j]);
              la += xv * wn[q * 4 + j];
              if (has_bias)
                lx += xv;
            }
          }
        }
        acc[m] += s * la + b * lx;
      }
    }
  }
  for (int m = 0; m < MAXM; ++m) {
    if (m >= nrows)
      break;
    float a = acc[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down(a, o);
    if (lane == 0 && valid_col)
      out[static_cast<size_t>(m0 + m) * N + col] = static_cast<T>(a);
  }
}

} // namespace mlx::core::rocm
