// t1253: multi-row verify QMV — the MTPLX "msg" morphology ported from Apple Metal (mtplx verify_kernels) to gfx1201/HIP.
//
// t1858 (kernel fix round 1): the base kernel now decodes each column's
// 4/8-bit weight tile ONCE per k-tile and MACs it against all M rows
// (row-shared x-tiles + column-shared decode; the t1334 hoist kernel kept
// the shared decode but re-read x per column). Per-row math is unchanged
// (q-major/j-minor FMA order, same lx chain, same reduction) so the
// row-exact contract holds: output row r == quantized_matmul(x[r:r+1]) bit
// for bit, and identical to the pre-t1858 base kernel per row. The
// per-row inline-decode form (base) re-decoded the weight tile M times;
// this is a ~2x ALU cut at the same HBM traffic (one 16B weight load per
// column per k-tile, one x-element load per row per k-tile).
//
// Metal -> HIP mapping: Metal simdgroup (32 threads, SIMD32)  -> HIP wave32 (32-thread wave) Metal threadgroup (6 simdgroups)      -> HIP workgroup (6 waves, 192 threads) threadgroup memory / barriers         -> NONE: the msg morphology carries no shared memory and no barriers; each wave is fully independent pack-interleaved 32-bit weight loads  -> uint4 (128-bit) loads, lane-strided
//
// Morphology:
//
// Row-exact contract (oMLX row_exact_qmv; t1253 steal #2): the per-row arithmetic transcribes the one-row qmv_wide_kernel traversal verbatim — per (row m, k-tile): la = 0; lx = 0; for q: for j: la += xn[j] * wn[q*V + j] (same q-major, j-minor FMA order, same int->float conversions, same `acc += s*la + b*lx` update), then the same 5-stage __shfl_down reduction and the same T cast. Only the weight decode is shared across rows (bit-identical values: same shift+mask+cvt chain), so row r of the output equals quantized_matmul(x[r:r+1]) bit for bit while each weight block is decoded once. Test: one-row reference diff (RUN card, probe 2). When !has_bias, b == 0.0f and 0.0f * (finite lx chain) == 0.0f == the unaccumulated lx of the per-row kernel — bit-identical in both cases (same argument as the run-852 LX hoist).
//
// Register estimate (RDNA4 / gfx1201, 256 VGPR per thread): base t1253 form: acc[16] + xn[32] + ws[4] (transient) + pointers/addresses (~20) ~= 100 VGPRs. t1858 form: acc[16] + xall[4][32] + wdec[32] (transient) + lx[4] + pointers ~= 190 VGPRs -> 2 waves per 2 slots (full 8-slot CU at 100 VGPR vs 4 at 190); the 2x ALU cut per k-tile outweighs the occupancy halving when the base is decode-ALU-bound (the round-1 hypothesis; the A/B bench decides). 192-thread block, 0 LDS, 8 wave slots per CU.
//
// Dispatch (qmm.hip, inside the wave32 wide_ok branch): 4/8-bit affine with bias, transposed, non-batched, M = 2..4, N >= 16384 (the large-N band; below it the per-row path wins on block count — lab data in the docs), K % 32 == 0, GROUP_SIZE in {32,64,128}, bf16/fp16 x, wave32. Opt-in: MLX_ROCM_QMV_ROWS=1 (DEFAULT OFF — zero behavior change when unset); band floor: MLX_ROCM_QMV_ROWS_MIN_N (default 16384). M = 1 (eager decode) and M = 5..8 keep the per-row qmv_wide path.
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>
__global__ void __launch_bounds__(192)
qmv_rows_msg_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint8_t* __restrict__ w,     // [N, K*BITS/8] packed
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int MAXM = 4;
  constexpr int BN = 4;
  constexpr int WARPS = 6;
  constexpr int WPER = 128 / BITS;   // 32 (4-bit) / 16 (8-bit)
  constexpr int V = 32 / BITS;       // values per 32-bit word (8 / 4)
  constexpr int NVEC = WPER / 8;     // 16B loads per row k-tile (4 / 2)
  constexpr int ES = (int)sizeof(T);  // element bytes (2 f16/bf16, 4 f32)
  constexpr int EPU = 16 / ES;       // elements per 16B uint4 load
  constexpr int NLOAD = WPER * ES / 16;  // uint4 loads per row k-tile

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int col0 = blockIdx.x * (WARPS * BN) + wid * BN;

  const int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS) / 8;

  float acc[MAXM][BN];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
#pragma unroll
    for (int c = 0; c < BN; ++c)
      acc[m][c] = 0.0f;

  // t1858 (kernel fix round 1): the base kernel re-decoded each column's
  // 4/8-bit weight tile once PER ROW (M=4: 4x redundant shift+mask+cvt —
  // the decode is ~3x the FMA, so ~3/4 of the weight-decode ALU was wasted
  // and the 16B weight was re-loaded M times from L1/L2). This rewrite takes
  // the t1334 hoist's shared decode AND the base's shared x: per k-tile each
  // row's x-tile is loaded + converted ONCE (shared across all BN columns,
  // identical values and lx chain to the base), then per column the 16B
  // weight tile is loaded and decoded ONCE into wdec (the same shift+mask+
  // cvt chain, bit-identical values) and MAC'd against all M rows. The
  // t1334 hoist kernel re-read x per column (M=4: 4x redundant 16B loads of
  // the M*K*E element buffer); this layout reads each x element once and each
  // weight element once. Register cost: xall[4][32] + acc[16] + wdec[32]
  // ~= 190 VGPR (fits the 256 cap; the t1334 hoist proves a 32-element
  // decode array compiles cleanly on the 10.1 toolchain, unlike the t1281
  // 10.0/clang-23 defect in the per-row inline position). Per-k-tile VALU
  // (M=4, 4-bit): ~1148 (4x(32 cvt + 31 lx) + 4x96 decode + 4x4x32 FMA) vs
  // the base's ~2300 (4x(63 + 4x128)) — a 2x ALU cut with the same HBM
  // traffic (one 16B weight load per column per k-tile).
  // Row-exact: per (row m, column c, k-tile) la = sum over q (outer), j
  // (inner) of xall[m][q*V + j] * wdec[q*V + j] — the same q-major/j-minor
  // FMA order over bit-identical values as the base's inline decode, and lx
  // is the same j-minor chain over the same 32 values; acc update
  // (s*la + b*lx), s/b loads, and the 5-stage shfl reduction are unchanged,
  // so output row r == the base kernel's row r bit for bit.
  float xall[MAXM][WPER];
  float lx[MAXM];
  float wdec[WPER];
  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    const int g = k / GROUP_SIZE;
    // Row pass: each row's x-tile is loaded ONCE per k-tile (shared across all
    // BN columns; same type-generic 16B load + cvt chain as the base) and its
    // lx chain is the base's verbatim j-minor sum over the same values.
    for (int m = 0; m < MAXM; ++m) {
      if (m >= M)
        break;
      const T* xp = &x[static_cast<size_t>(m) * K + k];
      uint4 v4[8];
#pragma unroll
      for (int q = 0; q < NLOAD; ++q)
        v4[q] = *reinterpret_cast<const uint4*>(xp + q * EPU);
      const T* px = reinterpret_cast<const T*>(v4);
      float lx_acc = 0.0f;
#pragma unroll
      for (int j = 0; j < WPER; ++j) {
        xall[m][j] = static_cast<float>(px[j]);
        lx_acc += xall[m][j];
      }
      lx[m] = lx_acc;
    }
    // Column pass: decode each column's weight tile ONCE (shared across all
    // rows; 128-bit pack-interleaved load, adjacent lanes read adjacent 16B
    // chunks — coalescing is load-bearing) and MAC against all rows.
    for (int c = 0; c < BN; ++c) {
      const int col = col0 + c;
      if (col >= N)
        continue;
      const float s = load_scale_value<ScaleT, GROUP_SIZE, true>(
          scales[static_cast<size_t>(col) * ng + g]);
      const float b = has_bias
          ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g])
          : 0.0f;
      const uint4 wv = *reinterpret_cast<const uint4*>(
          &w[static_cast<size_t>(col) * row_bytes + (k * BITS) / 8]);
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
      // Same shift+mask+cvt chain as the base's inline decode, materialized
      // once per column (bit-identical values; the t1334 hoist kernel uses
      // this exact 32-element form and is bit-exact on the 10.1 toolchain).
      if constexpr (BITS == 4) {
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            wdec[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            wdec[q * 4 + j] = static_cast<float>((wp[q] >> (8 * j)) & 0xFF);
      }
      // Per-row math: the base's q-major/j-minor FMA order, identical values.
      for (int m = 0; m < MAXM; ++m) {
        if (m >= M)
          break;
        float la = 0.0f;
        if constexpr (BITS == 4) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int j = 0; j < 8; ++j)
              la += xall[m][q * 8 + j] * wdec[q * 8 + j];
        } else {
#pragma unroll
          for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int j = 0; j < 4; ++j)
              la += xall[m][q * 4 + j] * wdec[q * 4 + j];
        }
        acc[m][c] += s * la + b * lx[m];
      }
    }
  }
  // Same 5-stage shfl_down reduction + T cast as the per-row kernel.
  for (int m = 0; m < MAXM; ++m) {
    if (m >= M)
      break;
    for (int c = 0; c < BN; ++c) {
      const int col = col0 + c;
      if (col >= N)
        continue;
      float a = acc[m][c];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        a += __shfl_down(a, o);
      if (lane == 0)
        out[static_cast<size_t>(m) * N + col] = static_cast<T>(a);
    }
  }
}


// t1334: decode-hoist M<=4 verify QMV — same msg morphology / row-exact contract as qmv_rows_msg_kernel, but the per-column 4/8-bit weight tile is decoded ONCE per (column, k-tile) and shared across all M rows (column-outer, row-inner). The base kernel re-decodes the weight tile once per (row, column) — 4x redundant across rows; here the decode is ~3x the FMA, so hoisting removes ~3/4 of the weight-decode ALU. The x-tile is re-read per column (M=4: 4x, L2-resident, 32 KB buffer); the decode is the load-bearing ALU save. Row-exact: per-row math is the same q-major/j-minor FMA order over the same decoded values, so output row r == quantized_matmul(x[r:r+1]) to rounding. OPT-IN ONLY: MLX_ROCM_QMV_ROWS_HOIST=1 (DEFAULT OFF; base kernel unchanged when unset).
template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>
__global__ void __launch_bounds__(192)
qmv_rows_msg_hoist_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M, int N, int K, bool has_bias) {
  constexpr int MAXM = 4;
  constexpr int BN = 4;
  constexpr int WARPS = 6;
  constexpr int WPER = 128 / BITS;
  constexpr int EPU = 16 / (int)sizeof(T);
  constexpr int NLOAD = WPER * (int)sizeof(T) / 16;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int col0 = blockIdx.x * (WARPS * BN) + wid * BN;
  const int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS) / 8;
  float acc[MAXM][BN];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
#pragma unroll
    for (int c = 0; c < BN; ++c)
      acc[m][c] = 0.0f;
  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    const int g = k / GROUP_SIZE;
    for (int c = 0; c < BN; ++c) {
      const int col = col0 + c;
      if (col >= N)
        continue;
      const float s = load_scale_value<ScaleT, GROUP_SIZE, true>(
          scales[static_cast<size_t>(col) * ng + g]);
      const float b = has_bias
          ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g])
          : 0.0f;
      const uint4 wv = *reinterpret_cast<const uint4*>(
          &w[static_cast<size_t>(col) * row_bytes + (k * BITS) / 8]);
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
      // Decode this column's weight tile ONCE (shared across all rows). 32-element decoded array: the same int->float cvt chain the base kernel runs inline, materialized here so the per-row FMA does 32 FMA with no cvt (the ~3x-decode ALU save).
      float wdec[WPER];
      if constexpr (BITS == 4) {
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            wdec[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            wdec[q * 4 + j] = static_cast<float>((wp[q] >> (8 * j)) & 0xFF);
      }
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
        if (m >= M)
          break;
        float xn[WPER];
        {
          const T* xp = &x[static_cast<size_t>(m) * K + k];
          uint4 v4[8];
#pragma unroll
          for (int q = 0; q < NLOAD; ++q)
            v4[q] = *reinterpret_cast<const uint4*>(xp + q * EPU);
          const T* px = reinterpret_cast<const T*>(v4);
#pragma unroll
          for (int j = 0; j < WPER; ++j)
            xn[j] = static_cast<float>(px[j]);
        }
        float lx = 0.0f;
#pragma unroll
        for (int j = 0; j < WPER; ++j)
          lx += xn[j];
        float la = 0.0f;
#pragma unroll
        for (int j = 0; j < WPER; ++j)
          la += xn[j] * wdec[j];
        acc[m][c] += s * la + b * lx;
      }
    }
  }
  for (int m = 0; m < MAXM; ++m) {
    if (m >= M)
      break;
    for (int c = 0; c < BN; ++c) {
      const int col = col0 + c;
      if (col >= N)
        continue;
      float a = acc[m][c];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        a += __shfl_down(a, o);
      if (lane == 0)
        out[static_cast<size_t>(m) * N + col] = static_cast<T>(a);
    }
  }
}

} // namespace mlx::core::rocm
