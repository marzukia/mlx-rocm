// Copyright © 2025 Apple Inc.

// Multi-row dense wide-load QMV WITHOUT x-staging (4/8-bit affine, decode,
// M = 1..4). Codegen-identical to qmv_wide_mr_kernel in every register
// path (WEIGHT REUSE + optional LX HOIST, same FMA order as
// qmv_wide_kernel per row) except the x source: x is read DIRECTLY from
// global (L2-served) instead of being staged into LDS once per block.
//
// Purpose (run-t1964): the staged MR kernel's x staging is capped at
// 32 KB (qmm.hip dispatch gate), so K > 4096 shapes (the 4-bit Qwen3-9B
// down_proj: K = 12288, N = 4096, 25.2 MB packed per layer) fall back to
// the PER-ROW qmv_wide_kernel at M > 1 - the weight is re-read M times
// (3x at M = 3). This variant has no x staging, so the 32 KB cap does not
// apply: one weight pass for up to 4 rows at ANY K. The x rows are tiny
// (M x K x 2 B, L2-resident), so the direct reads are L2-served; the
// per-row math (dequantize the weight tile ONCE per k-tile and MAC across
// all M rows; the exact qmv_wide_kernel FMA order per row; the optional
// column-invariant lx hoist, same FADD chain order) is unchanged, so every
// row stays bit-identical to the per-row kernel (same values, same FMA
// order as the staged MR kernel, whose x-staged values are the same
// global values).
//
// Grid: (ceil(N/mcols), ceil(M/4)); block: (32, mcols) (mcols from
// blockDim.y, launched at 16). LDS: 1 KB (s_lx, double-buffered, used
// only by the hoist path).
//
// Dispatch (qmm.hip): ONLY on the per-row fallback (the staged-MR gate
// failed on the 32 KB LDS cap, i.e. min(4, M) * K * 2 > 32 KB), M <= 4,
// K % 32 == 0 (already in the wide gate), GROUP_SIZE in {32,64,128},
// 4/8-bit affine with bias, transposed, non-batched, bf16/fp16 x, wave32.
// Opt-in: MLX_QMV_MR_NOSTAGE (DEFAULT OFF - when unset the per-row wide
// kernel runs verbatim, zero behavior change). The hoist additionally
// requires K % (32 * WPER) == 0 (K % 1024 == 0 for 4-bit, K % 512 == 0
// for 8-bit), selected by MLX_QMV_MR_LX (default 1 = hoist on) exactly as
// the staged MR path.
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE, int BITS, bool LXHOIST>
__global__ void __launch_bounds__(1024) qmv_wide_mr_nostage_kernel(
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
  // LX HOIST buffer (256 floats = 1 KB, DOUBLE-BUFFERED on k-iteration
  // parity), same as the staged MR kernel: iteration k (buffer p) and
  // k+1 (buffer p^1) use disjoint buffers; the in-k barrier orders k's
  // read before k+2's write to buffer p.
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

  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    acc[m] = 0.0f;
  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    int p = ((k - lane * WPER) / (32 * WPER)) & 1;
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * BITS) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    constexpr int WN = 128 / BITS;  // 32 (4-bit) / 16 (8-bit) dequant/lane
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
      {
        int ly = threadIdx.y;
        if (ly < nrows) {
          float lx = 0.0f;
          if constexpr (BITS == 4) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int j = 0; j < 8; ++j)
                lx += static_cast<float>(
                    x[(m0 + ly) * K + k + q * 8 + j]);
          } else {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int j = 0; j < 4; ++j)
                lx += static_cast<float>(
                    x[(m0 + ly) * K + k + q * 4 + j]);
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
              la += static_cast<float>(
                       x[(m0 + m) * K + kb + j]) *
                  wn[q * 8 + j];
          }
        } else {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int kb = k + q * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j)
              la += static_cast<float>(
                       x[(m0 + m) * K + kb + j]) *
                  wn[q * 4 + j];
          }
        }
        acc[m] += s * la + b * s_lx[p][m][lane];
      }
    } else {
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
              float xv =
                  static_cast<float>(x[(m0 + m) * K + kb + j]);
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
              float xv =
                  static_cast<float>(x[(m0 + m) * K + kb + j]);
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
