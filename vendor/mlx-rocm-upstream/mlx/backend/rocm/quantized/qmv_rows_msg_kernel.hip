// t1253: multi-row verify QMV — the MTPLX "msg" morphology ported from Apple Metal (mtplx verify_kernels) to gfx1201/HIP.
//
// Metal -> HIP mapping: Metal simdgroup (32 threads, SIMD32)  -> HIP wave32 (32-thread wave) Metal threadgroup (6 simdgroups)      -> HIP workgroup (6 waves, 192 threads) threadgroup memory / barriers         -> NONE: the msg morphology carries no shared memory and no barriers; each wave is fully independent pack-interleaved 32-bit weight loads  -> uint4 (128-bit) loads, lane-strided
//
// Morphology:
//
// Row-exact contract (oMLX row_exact_qmv; t1253 steal #2): the per-row arithmetic transcribes the one-row qmv_wide_kernel traversal verbatim — per (row m, k-tile): la = 0; lx = 0; for q: for j: la += xn[j] * wn[q*V + j] (same q-major, j-minor FMA order, same int->float conversions, same `acc += s*la + b*lx` update), then the same 5-stage __shfl_down reduction and the same T cast. Only the weight decode is shared across rows (bit-identical values: same shift+mask+cvt chain), so row r of the output equals quantized_matmul(x[r:r+1]) bit for bit while each weight block is decoded once. Test: one-row reference diff (RUN card, probe 2). When !has_bias, b == 0.0f and 0.0f * (finite lx chain) == 0.0f == the unaccumulated lx of the per-row kernel — bit-identical in both cases (same argument as the run-852 LX hoist).
//
// t1858 round 2 (verify-forward bandwidth fix): the k-tile prefetch form.
// The round-1 hypothesis (decode-ALU-bound; hoist the per-row decode, hold
// the shared x-tile in VGPR) measured WRONG: the decode hoist is a wash
// (t1334 hoist 15.86 vs base 16.09 ms mlp, same box/wheel) and the shared
// xall[4][32] VGPR form spilled (128 VGPR for x alone; measured 2.5x
// SLOWER, 113.8 vs 45.0 ms fused). The k loop has a runtime trip count
// (K % 32 == 0 is the only dispatch constraint; 8 iterations at K = 4096)
// so the 16B w/x loads are latency-bound: base loads each w tile moments
// before its FMA. Round 2 issues ALL of this k-tile's 16B loads up front
// (xv[MAXM][NLOAD] + wv[BN]; 20 loads at 4-bit) so they are in flight
// concurrently while the first tiles' cvt/FMA work runs. Values, load
// order, FMA order (q-major, j-minor), the lx chain, the s/b loads, the
// acc update and the 5-stage shfl reduction are unchanged -> bit-identical
// to the pre-t1858 base kernel. Register cost: xv 16 uint4 (64 VGPR at
// 4-bit) + wv 4 (16) + sb/bb 8 + xn[32] (transient) + acc[16] + pointers
// ~= 150-160 VGPR -> 2 waves per 8 wave slot (16 waves/CU, full).
//
// Register estimate (RDNA4 / gfx1201, 256 VGPR per thread): base t1253 form: acc[16] + xn[32] + ws[4] (transient) + pointers/addresses (~20) ~= 100 VGPRs. t1858 r2 prefetch form: xv[4][4] + wv[4] + sb/bb[4][2] + xn[32] + acc[16] + pointers ~= 150-160 VGPRs -> 2 waves per slot (16 waves per CU, full 8-slot occupancy at 2/slot). 192-thread block, 0 LDS.
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

  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    const int g = k / GROUP_SIZE;
  // t1858 round 2: k-tile prefetch. Issue every 16B load of this k-tile
  // (the x tiles of the live rows, the s/b scalars and the weight tiles
  // of the live columns) BEFORE any of the k-tile's cvt/FMA work, so the
  // loads are in flight concurrently (the base form loaded each w tile
  // moments before its FMA and, with the runtime K trip count — 8
  // iterations at K = 4096, no unroll — the 16B loads were latency-bound).
  // The loaded values, the guards, the cvt chains, the FMA order
  // (q-major, j-minor), the lx chain and the acc update are unchanged,
  // so the output is bit-identical to the base kernel.
  uint4 xv[MAXM][NLOAD];
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    if (m >= M)
      break;
    const T* xp = &x[static_cast<size_t>(m) * K + k];
#pragma unroll
    for (int q = 0; q < NLOAD; ++q)
      xv[m][q] = *reinterpret_cast<const uint4*>(xp + q * EPU);
  }
  float sb[BN];
  float bb[BN];
  uint4 wv[BN];
#pragma unroll
  for (int c = 0; c < BN; ++c) {
    if (col0 + c >= N)
      continue;
    const int col = col0 + c;
    sb[c] = load_scale_value<ScaleT, GROUP_SIZE, true>(
        scales[static_cast<size_t>(col) * ng + g]);
    bb[c] = has_bias
        ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g])
        : 0.0f;
    // 128-bit pack-interleaved weight load (coalescing is load-bearing: adjacent lanes read adjacent 16B chunks).
    wv[c] = *reinterpret_cast<const uint4*>(
        &w[static_cast<size_t>(col) * row_bytes + (k * BITS) / 8]);
  }
  for (int m = 0; m < MAXM; ++m) {
    if (m >= M)
      break;
    // Type-generic x-tile cvt: NLOAD 16B uint4 loads covering WPER elements of any 2/4-byte T (f16/bf16: 8 elems/load; f32: 4). 2-byte path is bit-identical to the original (EPU=8, NLOAD=WPER/8).
    const T* px = reinterpret_cast<const T*>(xv[m]);
    float xn[WPER];
#pragma unroll
    for (int j = 0; j < WPER; ++j)
      xn[j] = static_cast<float>(px[j]);
    // Per-row x-chunk sum: the qmv_wide_kernel lx chain (q-major, j-minor, same values). Accumulated unconditionally; when !has_bias the consumer multiplies by b == 0.0f (bit-identical, see header).
    float lx = 0.0f;
#pragma unroll
    for (int j = 0; j < WPER; ++j)
      lx += xn[j];
    for (int c = 0; c < BN; ++c) {
      if (col0 + c >= N)
        continue;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv[c]);
      // Weight words: keep the four 32-bit pack-interleaved words in a 4-element array (ws) instead of a 32-element decoded float array (wn[V]), and run the identical shift+mask+cvt chain inline in the MAC loop. The decoded values are bit-identical to a standalone decode (same int->float conversion, same order) and the FMA order is unchanged (q-major, j-minor). The 4-element form is required to dodge a codegen defect in the ROCm 10.0 / clang 23.0.0rocm7.14 toolchain: the 32-element wn[V] array in this position qnan-poisons the acc initialization and deletes the entire k-loop body (all 18 instantiations compile to a no-op storing qnan). Probe matrix (t1281): 1-word/8-value and 2-word/16-value decode arrays survive at -O3; the 32-element 4-word form degenerates regardless of 16B vs 4B weight load and regardless of whether xn is an array or scalar loads.
      uint32_t ws[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        ws[q] = wp[q];
      // Per-row math: the qmv_wide_kernel inner loop, verbatim order (inline decode == same cvt chain as the shared decode; FMA order q-major, j-minor, identical values).
      float la = 0.0f;
      if constexpr (BITS == 4) {
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            la += xn[q * 8 + j] * static_cast<float>((ws[q] >> (4 * j)) & 0xF);
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            la += xn[q * 4 + j] * static_cast<float>((ws[q] >> (8 * j)) & 0xFF);
      }
      acc[m][c] += sb[c] * la + bb[c] * lx;
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
