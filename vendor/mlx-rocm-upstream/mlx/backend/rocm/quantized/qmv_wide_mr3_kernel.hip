// Copyright © 2025 Apple Inc.

// t2011 H3: multi-row dense wide-load QMV WITHOUT x-staging, shared weight
// dequant (4-bit affine, decode, M = 2..4).
//
// The per-row qmv_wide_kernel (the small-N M >= 2 fallback of the current
// dispatch) re-reads and re-dequantizes the weight tile once per output
// row: at M = 3 the 4-bit FMA:dequant ratio is 1:1 per row and the weight
// stream is touched 3x. This kernel dequantizes each weight tile ONCE per
// k-tile (shift+mask+cvt into a register array) and MACs it against ALL
// rows of the block, so the FMA:dequant ratio is M:1 (3:1 at M = 3).
// There is no x staging: the x rows (M x K x sizeof(T), L2-resident) are
// read directly from global with 16B vector loads, and the kernel uses no
// shared memory at all (no LDS co-residency cap; the 32-column-per-4-row
// block shape is the MR morphology with the 24 KB staging removed).
//
// Bit-identity contract: the per-row math is the EXACT qmv_wide_kernel
// traversal per (row, column) — k tiles in ascending order, la q-major /
// j-minor FMA order, per-value lx FADD (guarded by has_bias, in the same
// position as per-row), the same int->float conversions, the same
// `acc += s*la + b*lx` update; only the dequantized weight values are
// shared (the same static_cast<float> of the same shift/mask — a value
// that does not depend on the row), so every row output is bit-identical
// to the per-row kernel (lossless gate: maxdiff 0.0).
//
// Grid: (ceil(N/16), ceil(M/4)); block: (32, 16). LDS: none.
//
// Dispatch (qmm.hip): ONLY replaces the per-row wide fallback (the
// staged-MR / mr2 / rows gates all failed), M = 2..4, 4-bit, K % 32 == 0
// (already in the wide gate). Opt-in: MLX_ROCM_QMM_M3 (DEFAULT OFF — when
// unset the per-row wide kernel runs verbatim, zero behavior change).
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(1024) qmv_wide_mr3_kernel(
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
  constexpr int WPER = 32;  // 4-bit only
  constexpr int MCOLS = 16;
  int lane = threadIdx.x;
  int col = blockIdx.x * MCOLS + threadIdx.y;
  int m0 = blockIdx.y * MAXM;
  int nrows = M - m0;
  if (nrows > MAXM)
    nrows = MAXM;
  if (nrows < 0)
    nrows = 0;
  // Same early-return semantics as the per-row kernel (row/col guard
  // BEFORE any weight read — no OOB at col >= N, the MR12 defect class).
  if (nrows == 0 || col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = K / 2;  // 4-bit
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;

  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    acc[m] = 0.0f;
  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // WEIGHT REUSE (t2011 H3): dequantize the weight tile ONCE per k-tile
    // and MAC it against ALL rows of the block (FMA:dequant = M:1, 3:1
    // at M = 3), instead of the per-row kernel's per-row re-dequant
    // (1:1 per row, 3x dequant work and 3x weight stream).
    float wn[32];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
    // x read: 16B vector loads (16 / sizeof(T) values each); the values
    // and the q-major / j-minor order are identical to the per-row scalar
    // traversal, so the per-row FMA order is unchanged.
    constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
    constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m >= nrows)
        break;
      uint4 xv4[8] = {};
#pragma unroll
      for (int r = 0; r < NU4; ++r)
        xv4[r] = *reinterpret_cast<const uint4*>(
            &x[static_cast<size_t>(m0 + m) * K + k + r * VEC]);
      const T* px = reinterpret_cast<const T*>(xv4);
      // Per-row math (EXACT qmv_wide_kernel order): la FMA then per-value
      // lx FADD (guarded by has_bias, same position), acc += s*la + b*lx.
      float la = 0.0f, lx = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        int kb = k + q * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float xv = static_cast<float>(px[q * 8 + j]);
          la += xv * wn[q * 8 + j];
          if (has_bias)
            lx += xv;
        }
      }
      acc[m] += s * la + b * lx;
    }
  }
  for (int m = 0; m < MAXM; ++m) {
    if (m >= nrows)
      break;
    float a = acc[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down(a, o);
    if (lane == 0)
      out[static_cast<size_t>(m0 + m) * N + col] = static_cast<T>(a);
  }
}

} // namespace mlx::core::rocm
