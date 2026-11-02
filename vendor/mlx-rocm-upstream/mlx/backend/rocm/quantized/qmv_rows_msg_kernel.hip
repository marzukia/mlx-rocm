// t1253: multi-row verify QMV — the MTPLX "msg" morphology ported from
// Apple Metal (mtplx verify_kernels) to gfx1201/HIP.
//
// Metal -> HIP mapping (full doc: docs/qmv-rows-msg.md):
//   Metal simdgroup (32 threads, SIMD32)  -> HIP wave32 (32-thread wave)
//   Metal threadgroup (6 simdgroups)      -> HIP workgroup (6 waves, 192 threads)
//   threadgroup memory / barriers         -> NONE: the msg morphology carries
//                                           no shared memory and no barriers;
//                                           each wave is fully independent
//   pack-interleaved 32-bit weight loads  -> uint4 (128-bit) loads, lane-strided
//
// Morphology:
//   * Grid (ceil(N / 24), 1); block 192 threads = 6 waves. Each wave owns
//     BN = 4 consecutive output columns; the block covers 24 columns. The
//     waves share nothing — they ride in one workgroup purely to give the
//     scheduler fewer, heavier units (the reference's own note).
//   * Each wave computes ALL M rows (M <= 4) in an explicit MAXM x BN
//     accumulator array (16 accumulators at M = 4, under the 24-
//     accumulator/thread register ceiling recorded in the reference).
//   * Full-K reduction, lane-strided: k = lane*WPER stepping 32*WPER, with
//     128-bit (uint4 = 32 nibbles) pack-interleaved weight loads — 512 B
//     coalesced per wave-instruction (the load-bearing coalescing note).
//   * Each 4/8-bit weight tile (32 / 16 values) is decoded ONCE per
//     (column, k-tile) and MAC'd against ALL M rows: the weight stream is
//     read exactly once regardless of M (M = 4 costs one pass + activation
//     overhead, not four passes).
//   * Scales/biases: per-column scalar loads per k-tile (L1-served, exactly
//     as in the per-row qmv_wide path).
//   * x (M x K; 32 KB bf16 at M = 4, K = 4096) is read with per-element
//     scalar loads per (row, col, k-tile) — the reference's own load
//     shape (L2-resident buffer; the 4 columns re-read L1-cached
//     values, exactly as in the per-row path).
//
// Row-exact contract (oMLX row_exact_qmv; t1253 steal #2): the per-row
// arithmetic transcribes the one-row qmv_wide_kernel k-tile body VERBATIM
// — per (row m, col c, k-tile): the reference's exact source shape:
// scalar per-element x loads (`static_cast<float>(xp[kb + j])`, no 16B
// vector staging), a single `la` / `lx` accumulator pair, the identical
// shift+mask+cvt decode chain (`(wd >> (4*j)) & 0xF` ->
// static_cast<float>), the identical q-major j-minor FMA order
// (`la += xv * cvt`, `if (has_bias) lx += xv` — lx is accumulated only
// when has_bias, exactly as in the reference, so the compiled FP op
// sequence per row is the same contraction the reference gets), and the
// same `acc += s * la + b * lx` scalar-accumulator update. The only
// shared state is the 128-bit pack-interleaved WEIGHT tile (the
// reference's own load shape), read once per (col, k-tile) and reused
// across all M rows — row r of the output therefore equals
// quantized_matmul(x[r:r+1]) bit for bit while each weight block is
// decoded once. t1297: the t1281 ws[4] form was 1-2 ULP off because the
// 16B x staging / 2D acc scratch context changed the compiled FP
// association; this restructure removes that shape. Test: one-row
// reference diff (RUN card, probe 2).
//
// Register estimate (RDNA4 / gfx1201, 256 VGPR per thread):
//   acc[16] + la/lx/s/b + wv[4] (transient per (m,c)) + pointers/addresses
//   ~= 40 VGPRs -> full occupancy headroom; 192-thread block = 6 waves,
//   0 LDS, 8 wave slots per CU.
//
// Dispatch (qmm.hip, inside the wave32 wide_ok branch): 4/8-bit affine with
// bias, transposed, non-batched, M = 2..4, N >= 16384 (the large-N band;
// below it the per-row path wins on block count — lab data in the docs),
// K % 32 == 0, GROUP_SIZE in {32,64,128}, bf16/fp16 x, wave32. Opt-in:
// MLX_ROCM_QMV_ROWS=1 (DEFAULT OFF — zero behavior change when unset);
// band floor: MLX_ROCM_QMV_ROWS_MIN_N (default 16384). M = 1 (eager
// decode) and M = 5..8 keep the per-row qmv_wide path.
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
    for (int m = 0; m < MAXM; ++m) {
      if (m >= M)
        break;
      // Per-row scalar x base: the reference's `xr` (row start);
      // per-element scalar loads below, exactly the reference shape
      // (no 16B x staging — the staging array is the codegen-shape
      // trigger t1297 diagnosed; the x buffer is L2-resident across
      // blocks and the 4 columns re-read L1-cached values, as in the
      // reference where every (row, col) block loads its own x-tile).
      const T* xp = &x[static_cast<size_t>(m) * K];
      for (int c = 0; c < BN; ++c) {
        const int col = col0 + c;
        if (col >= N)
          continue;
        const float s = load_scale_value<ScaleT, GROUP_SIZE, true>(
            scales[static_cast<size_t>(col) * ng + g]);
        const float b = has_bias
            ? static_cast<float>(biases[static_cast<size_t>(col) * ng + g])
            : 0.0f;
        // 128-bit pack-interleaved weight load (coalescing is
        // load-bearing: adjacent lanes read adjacent 16B chunks) — the
        // reference's own load shape; the four words are reused across
        // all M rows (same tile, same inline decode chain; the t1281
        // defect was keyed on the 32-element decode array, which is
        // absent here by construction).
        const uint4 wv = *reinterpret_cast<const uint4*>(
            &w[static_cast<size_t>(col) * row_bytes + (k * BITS) / 8]);
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
        // Per-row math: the qmv_wide_kernel k-tile body, VERBATIM shape
        // (scalar x loads, single la/lx accumulators, identical
        // shift+mask+cvt chain, q-major j-minor FMA order, conditional
        // lx exactly as the reference, same `acc += s*la + b*lx`
        // scalar-accumulator update).
        float la = 0.0f, lx = 0.0f;
        if constexpr (BITS == 4) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t wd = wp[q];
            const int kb = k + q * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const float xv = static_cast<float>(xp[kb + j]);
              la += xv * static_cast<float>((wd >> (4 * j)) & 0xF);
              if (has_bias)
                lx += xv;
            }
          }
        } else {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t wd = wp[q];
            const int kb = k + q * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float xv = static_cast<float>(xp[kb + j]);
              la += xv * static_cast<float>((wd >> (8 * j)) & 0xFF);
              if (has_bias)
                lx += xv;
            }
          }
        }
        acc[m][c] += s * la + b * lx;
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

} // namespace mlx::core::rocm
