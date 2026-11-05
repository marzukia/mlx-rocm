// Copyright © 2025 Apple Inc.

// t2031 gen2 probe: next-generation mlp-class GEMM variants on the
// qmv_wide_mr3_kernel (the adopted M=3 shared-dequant QMM kernel,
// REPORT-qmm-m3.md ADOPT; the mlp class is 8.407 ms at 59.2% of the
// 546 GB/s HBM floor per the fold re-gate, REPORT-fold-qmm-hybrid.md).
//
// Four variants, each dispatched inside the same !mr_ok per-row fallback
// branch (4-bit, M in [2,4]) that the stock mr3 kernel occupies, each
// behind its own DEFAULT-OFF env (MLX_ROCM_QMM_M3_V1..V4; all unset =
// the stock mr3 kernel runs verbatim). Every variant keeps the bit-
// identity contract of the stock kernel: the per-(row, column) FMA
// traversal is unchanged (k tiles in ascending order, la q-major /
// j-minor FMA order, per-value lx FADD guarded by has_bias in the same
// position, the same int->float conversions, the same acc += s*la + b*lx
// update, the same 32-lane shfl_down butterfly and lane-0 store), and
// the same early return (col >= N) before any weight read. Only the
// memory-access structure changes.
//
// Grid: (ceil(N/16), ceil(M/4)); block: (32, 16) = 512 threads = 16 warps.
//
// V1 (MLX_ROCM_QMM_M3_V1) HOISTED la/lx (LDS x-staging): the per-row
// x loads are staged ONCE per k-tile into LDS per warp, shared across
// all 16 cols of the warp (the stock kernel re-reads the same x values
// from global/L2 once per col: 16x redundant x traffic). One row at a
// time (1024 values per warp), __syncthreads after each stage. LDS =
// 16 warps x 1024 values x sizeof(T): 32 KB (bf16/half) / 64 KB (f32).
//
// V2 (MLX_ROCM_QMM_M3_V2) K-TILE DOUBLE BUFFER: software pipeline within
// the warp - the weight load of k-tile i+1 is issued at the top of
// iteration i (before the FMA of tile i) and dequantized into the
// alternating (A/B) weight register buffer AFTER the FMA of tile i, so
// the next tile's load latency is hidden under the current tile's x
// loads + FMA + dequant.
//
// V3 (MLX_ROCM_QMM_M3_V3) 32B WEIGHT LOADS: the 16B weight vector load
// is widened to a 32B logical access (two consecutive 16B aligned loads;
// the second half is accessed and elision-proofed but its value is not
// used - bit-identity). Alignment verified for the 4-bit layout with the
// sidecar weight rows as stored (no reordering): the k-tile of lane l at
// iteration i sits at byte (k*4)/8 with k = 32*l + 1024*i, i.e. 16B-
// aligned for all lanes (64B-aligned for even lanes), and the 32B window
// [offset, offset+32) stays inside the row (max offset 2*(K-32) + 16B
// load end < K/2 row bytes for K % 32 == 0).
//
// V4 (MLX_ROCM_QMM_M3_V4) LOAD-PATTERN VARIANT: the per-row x loads of
// the k-tile are hoisted into registers and issued as ONE batched group
// (all rows, 16 x 16B) BEFORE the weight load, instead of the stock
// per-row interleave (x loads after the weight dequant, one row at a
// time). Same stored layout, same per-(row, col) FMA order - only the
// memory issue order changes (x batch ahead of the weight stream, for
// HBM row-buffer locality).
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(512) qmv_wide_mr3_v1_kernel(
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
  // LDS: 16 warp slabs of 1024 T values (32 lanes x WPER), one row at a
  // time. Launched with 16 * 1024 * sizeof(T) bytes (32 KB bf16/half,
  // 64 KB f32; the dispatch gate caps it at 65536).
  // Dynamic shared memory as a byte buffer (extern __shared__ is a
  // single symbol per module and cannot be redeclared with a different
  // type across kernels/instantiations - the stock mr kernel's sx_bytes
  // idiom and qmv_lx's smem already own those names).
  extern __shared__ uint8_t m3v1_lsmem[];
  T* xs = reinterpret_cast<T*>(m3v1_lsmem);
  int lane = threadIdx.x;
  int col = blockIdx.x * MCOLS + threadIdx.y;
  int m0 = blockIdx.y * MAXM;
  int nrows = M - m0;
  if (nrows > MAXM)
    nrows = MAXM;
  if (nrows < 0)
    nrows = 0;
  // Same early return as the stock kernel (BEFORE any weight read).
  if (nrows == 0 || col >= N)
    return;
  int warp = threadIdx.y;  // block (32, 16): 16 warps, warp = threadIdx.y
  T* xsw = &xs[static_cast<size_t>(warp) * (32 * WPER)];
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = K / 2;  // 4-bit
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;
  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    acc[m] = 0.0f;
  // Uniform trip count (the stock kernel's per-lane k < K bound becomes a
  // per-iteration guard so the in-loop __syncthreads is block-uniform).
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
  constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
  constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
  for (int i = 0; i < niter; ++i) {
    int k = lane * WPER + i * 32 * WPER;
    bool cur = k < K;
    int g = 0;
    float s = 0.0f, b = 0.0f;
    float wn[32] = {};
    if (cur) {
      g = k / GROUP_SIZE;
      s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
      b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
      uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
      // Stock shared-dequant: the weight tile is dequantized ONCE per
      // k-tile and MACed against all rows (FMA:dequant = M:1).
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
    }
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m >= nrows)
        break;
      // V1: stage this row's 32 k values into LDS once per k-tile, shared
      // across all 16 cols of the warp (the stock kernel re-reads the
      // same x values from global/L2 once per col: 16x redundant x
      // traffic). The staged values are bit-identical copies of the
      // stock kernel's 16B vector loads.
      if (cur) {
        uint4 xv4[8];
#pragma unroll
        for (int r = 0; r < NU4; ++r)
          xv4[r] = *reinterpret_cast<const uint4*>(
              &x[static_cast<size_t>(m0 + m) * K + k + r * VEC]);
        const T* px = reinterpret_cast<const T*>(xv4);
#pragma unroll
        for (int j = 0; j < WPER; ++j)
          xsw[lane * WPER + j] = px[j];
      }
      __syncthreads();
      if (cur) {
        // Per-row math (EXACT stock order): la FMA then per-value lx
        // FADD (guarded by has_bias), acc += s*la + b*lx - the values
        // now come from LDS instead of the per-col global re-read.
        float la = 0.0f, lx = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float xv = static_cast<float>(xsw[lane * WPER + q * 8 + j]);
            la += xv * wn[q * 8 + j];
            if (has_bias)
              lx += xv;
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
    if (lane == 0)
      out[static_cast<size_t>(m0 + m) * N + col] = static_cast<T>(a);
  }
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(512) qmv_wide_mr3_v2_kernel(
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
  // Same early return as the stock kernel (BEFORE any weight read).
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
  // V2: K-tile double buffer (software pipeline within the warp). Two
  // alternating weight register buffers (A/B); the weight LOAD of tile
  // i+1 is issued at the top of iteration i (ahead of the FMA of tile
  // i) and its dequant lands in the buffer AFTER the FMA of tile i, so
  // the next tile's load latency is hidden under the current tile's x
  // loads + FMA + dequant. Per-(row, column) FMA order is the stock
  // order (ascending k tiles, la q-major / j-minor, lx per-value).
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
  constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
  constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
  float wnA[32];
  float wnB[32];
  bool cur = (lane * WPER) < K;
  if (cur) {
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[((lane * WPER) * 4) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wnA[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
  }
  for (int i = 0; i < niter; ++i) {
    int k = lane * WPER + i * 32 * WPER;
    int kn = k + 32 * WPER;
    bool nxt = kn < K;
    // Prefetch the NEXT tile's 16B weight load (issued before the FMA of
    // the current tile; its latency hides under the FMA + dequant below).
    uint4 wv_next;
    if (nxt)
      wv_next = *reinterpret_cast<const uint4*>(&wr[(kn * 4) / 8]);
    if (cur) {
      int g = k / GROUP_SIZE;
      float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
      float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
      const float* wn = (i & 1) ? wnB : wnA;
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
        if (m >= nrows)
          break;
        // Per-row math (EXACT stock order): 16B x vector loads, la FMA
        // (q-major / j-minor) then per-value lx FADD (guarded by
        // has_bias), acc += s*la + b*lx.
        uint4 xv4[8];
#pragma unroll
        for (int r = 0; r < NU4; ++r)
          xv4[r] = *reinterpret_cast<const uint4*>(
              &x[static_cast<size_t>(m0 + m) * K + k + r * VEC]);
        const T* px = reinterpret_cast<const T*>(xv4);
        float la = 0.0f, lx = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
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
      // Dequant the NEXT tile AFTER the FMA of the current tile into the
      // alternating buffer (tile i+2 reuses the buffer freed by tile i).
      if (nxt) {
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv_next);
        float* wnd = (i & 1) ? wnA : wnB;
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            wnd[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
      }
    }
    cur = nxt;
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

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(512) qmv_wide_mr3_v3_kernel(
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
  // Same early return as the stock kernel (BEFORE any weight read).
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
  // V3: 32B weight loads. The 16B weight vector load is widened to a 32B
  // logical access: two consecutive 16B aligned loads (the 32B window is
  // 32B-aligned for even lanes and 16B-aligned for odd lanes; the 4-bit
  // layout keeps both halves in-row - no weight reordering, the sidecar
  // is read exactly as stored). Only the FIRST half (the stock 32
  // values) is used in the math; the second half is accessed with an
  // elision-proofing asm so the 32B window is genuinely touched.
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
  constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
  constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
  for (int i = 0; i < niter; ++i) {
    int k = lane * WPER + i * 32 * WPER;
    if (k >= K)
      continue;
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    // 32B window: [offset, offset+32) - two consecutive 16B aligned loads
    // (per-lane 32-bit volatile loads; the per-lane 16B alignment of the
    // 4-bit layout holds, the window is 32B-aligned for even lanes).
    const volatile uint32_t* pw = reinterpret_cast<const volatile uint32_t*>(
        &wr[(k * 4) / 8]);
    uint4 wv;  // half 0 (the stock 32 values, used in the math)
    wv.x = pw[0];
    wv.y = pw[1];
    wv.z = pw[2];
    wv.w = pw[3];
    // The 32B window [offset, offset+32) overruns the row end (and the
    // buffer) by up to 16B when k + 64 > K (the stock 16B load fits
    // exactly at the last tile: (K-32)/2 + 16 = K/2). Clamp the second
    // half to in-row windows only (values k+32..k+63 are never used in
    // the math - bit-identity - the clamp changes access pattern only).
    bool full32 = (k + 64) <= K;
    uint32_t wv1x = 0, wv1y = 0, wv1z = 0, wv1w = 0;
    if (full32) {
      wv1x = pw[4], wv1y = pw[5], wv1z = pw[6], wv1w = pw[7];  // half 1 (accessed, elision-proofed, unused)
    }
    // Input-only (read-write "r" constraints are unlowerable in AMDGPU
    // codegen: "illegal VGPR to SGPR copy").
    asm volatile("" : : "r"(wv1x), "r"(wv1y), "r"(wv1z), "r"(wv1w));
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // Stock shared-dequant: the weight tile is dequantized ONCE per
    // k-tile and MACed against all rows (FMA:dequant = M:1).
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
      // Per-row math (EXACT stock order): 16B x vector loads, la FMA
      // (q-major / j-minor) then per-value lx FADD (guarded by
      // has_bias), acc += s*la + b*lx.
      uint4 xv4[8];
#pragma unroll
      for (int r = 0; r < NU4; ++r)
        xv4[r] = *reinterpret_cast<const uint4*>(
            &x[static_cast<size_t>(m0 + m) * K + k + r * VEC]);
      const T* px = reinterpret_cast<const T*>(xv4);
      float la = 0.0f, lx = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
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

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(512) qmv_wide_mr3_v4_kernel(
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
  // Same early return as the stock kernel (BEFORE any weight read).
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
  // V4: load-pattern variant. The per-row x loads of the k-tile are
  // hoisted into registers and issued as ONE batched group (all active
  // rows, 16 x 16B for bf16) BEFORE the weight load, instead of the
  // stock per-row interleave (x loads after the weight dequant, one row
  // at a time). Same stored layout, same per-(row, column) FMA order -
  // only the memory issue order changes (x batch ahead of the weight
  // stream, for HBM row-buffer locality).
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
  constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
  constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
  for (int i = 0; i < niter; ++i) {
    int k = lane * WPER + i * 32 * WPER;
    if (k >= K)
      continue;
    // Batched x hoist: all rows' 32 k values, issued before the weight
    // load (values identical to the stock 16B vector loads).
    uint4 xv4[MAXM][8];
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m >= nrows)
        break;
#pragma unroll
      for (int r = 0; r < NU4; ++r)
        xv4[m][r] = *reinterpret_cast<const uint4*>(
            &x[static_cast<size_t>(m0 + m) * K + k + r * VEC]);
    }
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // Stock shared-dequant: the weight tile is dequantized ONCE per
    // k-tile and MACed against all rows (FMA:dequant = M:1).
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
      // Per-row math (EXACT stock order): la FMA (q-major / j-minor)
      // then per-value lx FADD (guarded by has_bias), acc += s*la +
      // b*lx - the values now come from the hoisted registers.
      const T* px = reinterpret_cast<const T*>(xv4[m]);
      float la = 0.0f, lx = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
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
