// Copyright © 2025 Apple Inc.

// F3-M4X (VARIANT B, per RCA-r1 section 3.8 mechanism): the
// qmv_wide_mr3_kernel with the per-row 16B x-tile loads software-
// pipelined across the k-loop. The base kernel issues each row's
// NLOAD 16B loads and immediately consumes them (load-use dependent
// per k-tile: the exposed per-row 16B x-load latency is the named
// candidate for the M=4 GEMM running below its shared-decode design
// point). This variant double-buffers the x tiles: while the current
// k-tile is being MAC'd, the NEXT k-tile's row tiles are issued, so
// the load latency overlaps the FMA work of the current tile.
//
// Bit-identity contract: the per-row math is the EXACT
// qmv_wide_mr3_kernel traversal — same k-tile order (ascending), same
// la q-major / j-minor FMA order, same per-value lx FADD (guarded by
// has_bias, same position), same int->float conversions, same
// `acc += s*la + b*lx` update, same 5-stage __shfl_down reduction,
// same T cast. Only the x-load timing changes; the values are
// identical, so every row output is bit-identical to the base mr3
// kernel (which holds the per-row qmv_wide contract: maxdiff 0.0).
//
// Register cost: the double buffer holds 2 x NLOAD x 16B per row
// (bf16: NLOAD = 4; 64 VGPR per buffer). Launch at M = 4, bf16 x
// only (the f32 NLOAD = 8 form would spill; the f32 arm keeps the
// base kernel).
//
// Grid: (ceil(N/16), ceil(M/4)); block: (32, 16). LDS: none.
// Dispatch (qmm.hip, the m3ok branch): ONLY when MLX_ROCM_M4PREF=1
// (DEFAULT OFF — the base mr3 kernel runs verbatim when unset),
// M = 4, 4-bit, x bf16. M = 2/3 dispatch untouched (the K=2 / M=3
// record path is invariant).
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(1024) qmv_wide_mr3_pref_kernel(
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
  constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
  constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
  const int STEP = 32 * WPER;
  // Double-buffered x tiles (software pipeline across the k-loop).
  uint4 cur[MAXM][NU4];
  uint4 nxt[MAXM][NU4];
  for (int m = 0; m < MAXM; ++m)
    if (m < nrows)
#pragma unroll
      for (int r = 0; r < NU4; ++r)
        cur[m][r] = *reinterpret_cast<const uint4*>(
            &x[static_cast<size_t>(m0 + m) * K + lane * WPER + r * VEC]);
  for (int k = lane * WPER; k < K; k += STEP) {
    const bool have_next = (k + STEP) < K;
    // Issue the NEXT k-tile's row loads now (they overlap the FMA work
    // below); the FMA math itself is unchanged.
    if (have_next)
#pragma unroll
      for (int m = 0; m < MAXM; ++m)
        if (m < nrows)
#pragma unroll
          for (int r = 0; r < NU4; ++r)
            nxt[m][r] = *reinterpret_cast<const uint4*>(
                &x[static_cast<size_t>(m0 + m) * K + k + STEP + r * VEC]);
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    float wn[32];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m >= nrows)
        break;
      const T* px = reinterpret_cast<const T*>(cur[m]);
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
    if (!have_next)
      break;
#pragma unroll
    for (int m = 0; m < MAXM; ++m)
#pragma unroll
      for (int r = 0; r < NU4; ++r)
        cur[m][r] = nxt[m][r];
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
