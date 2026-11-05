// Copyright © 2025 Apple Inc.

// qgd-gen2 probe: next-generation per-row (M = 1) QMM variants of the
// stock qmv_wide_kernel per-row wide path (the kernel that serves the
// qmv_gdn class - GDN 4 in-proj + out_proj QMMs, 24 layers, M = 1 -
// 4.59 ms at 36.3% of the 546 GB/s HBM floor, the #2 verify-wall class
// after mlp). The four variants are the gen2 memory-structure changes
// from REPORT-mlp-gen2.md ported onto the per-row kernel:
//
// V1 (MLX_ROCM_QGD_V1) HOISTED x (LDS x-staging): the per-tile x loads
// are staged ONCE per k-tile into LDS (1024 values, 256 threads x 4,
// 8B/16B-friendly scalar staging), shared across all 8 col warps of the
// block (the stock per-row kernel re-reads the same x values 8x - once
// per col warp). __syncthreads after each stage. LDS = 1024 values x
// sizeof(T): 2 KB (bf16/half) / 4 KB (f32).
//
// V2 (MLX_ROCM_QGD_V2) K-TILE DOUBLE BUFFER: software pipeline within the
// lane - the weight load of k-tile i+1 is issued at the top of iteration
// i (before the FMA of tile i) and dequantized into the alternating
// (A/B) weight register buffer AFTER the FMA of tile i.
//
// V3 (MLX_ROCM_QGD_V3) 32B WEIGHT LOADS: the 16B weight vector load is
// widened to a 32B logical access (two consecutive 16B aligned loads;
// the second half is accessed and elision-proofed but its value is not
// used - bit-identity), clamped to in-row windows only (k + 64 <= K):
// the 32B window overruns the row end (and the buffer) by up to 16B at
// the last k-tile where the stock 16B load fits exactly.
//
// V4 (MLX_ROCM_QGD_V4) BATCHED-X LOAD ORDER: the per-tile x loads are
// hoisted into registers and issued as ONE batched group of 16B vector
// loads (32 values: 4 x uint4 for 2B T, 8 x uint4 for f32) BEFORE the
// weight load, instead of the stock 32 scalar per-value loads inside the
// FMA loop.
//
// Every variant keeps the bit-identity contract of the stock per-row
// kernel: the per-(row, col) FMA traversal is unchanged (k tiles in
// ascending order, la q-major / j-minor per-value FMA order, per-value
// lx FADD guarded by has_bias in the same position, the same int->float
// conversions, the same acc += s*la + b*lx update, the same 32-lane
// shfl_down butterfly and lane-0 store), and the same early return
// (row >= M || col >= N) before any weight read. Only the
// memory-access structure changes.
//
// Grid: (ceil(N / wcols), M); block: (32, wcols) - the stock per-row
// wide geometry (wcols = 8 on the gfx1201 host).
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(256) qmv_wide_qgd_v1_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int WPER = 32;  // 4-bit only
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int row = blockIdx.y;
  // Same early return as the stock per-row kernel (BEFORE any weight read).
  if (row >= M || col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = K / 2;  // 4-bit
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;
  const T* xr = x + static_cast<size_t>(row) * K;
  float acc = 0.0f;
  // V1: LDS x-staging. 1024 values per k-tile, staged ONCE per block and
  // shared across all 8 col warps (the stock per-row kernel re-reads the
  // same x values once per col warp: 8x redundant x traffic). Launched
  // with 1024 * sizeof(T) bytes (2 KB bf16/half, 4 KB f32). Dynamic
  // shared memory as a byte buffer (extern __shared__ is a single symbol
  // per module - the stock kernels' sx/smem names are already taken).
  extern __shared__ uint8_t qgdv1_lsmem[];
  T* xs = reinterpret_cast<T*>(qgdv1_lsmem);
  // Uniform trip count (the stock kernel's per-lane k < K bound becomes a
  // per-iteration guard so the in-loop __syncthreads is block-uniform).
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
  const int nthreads = 32 * blockDim.y;
  const int tid = threadIdx.y * 32 + lane;
  for (int i = 0; i < niter; ++i) {
    int k0 = i * 32 * WPER;
    // Cooperative staging: all 1024 values per tile (256 threads x 4 on
    // the gfx1201 8-col block; stride-general for 4-col blocks); edge
    // values beyond K are staged as 0 (never used: the per-lane k < K
    // guard skips the whole last partial tile's out-of-range values).
    for (int idx = tid; idx < 32 * WPER; idx += nthreads) {
      int kk = k0 + idx;
      xs[idx] = (kk < K) ? xr[kk] : static_cast<T>(0.0f);
    }
    __syncthreads();
    int k = lane * WPER + k0;
    if (k >= K)
      continue;
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // Shared dequant: the weight tile is dequantized once per k-tile
    // (identical values, identical FMA order to the stock inline
    // per-value dequant).
    float wn[32];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
    // Per-row math (EXACT stock order): la FMA (q-major / j-minor) then
    // per-value lx FADD (guarded by has_bias), acc += s*la + b*lx - the
    // x values now come from LDS instead of the 8x per-warp re-read.
    float la = 0.0f, lx = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float xv = static_cast<float>(xs[lane * WPER + q * 8 + j]);
        la += xv * wn[q * 8 + j];
        if (has_bias)
          lx += xv;
      }
    acc += s * la + b * lx;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc);
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(256) qmv_wide_qgd_v2_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int WPER = 32;  // 4-bit only
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int row = blockIdx.y;
  // Same early return as the stock per-row kernel (BEFORE any weight read).
  if (row >= M || col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = K / 2;  // 4-bit
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;
  const T* xr = x + static_cast<size_t>(row) * K;
  float acc = 0.0f;
  // V2: K-tile double buffer (software pipeline within the lane). Two
  // alternating weight register buffers (A/B); the weight LOAD of tile
  // i+1 is issued at the top of iteration i (ahead of the FMA of tile i)
  // and its dequant lands in the buffer AFTER the FMA of tile i, so the
  // next tile's load latency is hidden under the current tile's x loads +
  // FMA + dequant. Per-(row, column) FMA order is the stock order
  // (ascending k tiles, la q-major / j-minor, lx per-value).
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
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
      // Per-row math (EXACT stock order): per-value scalar x loads (the
      // stock per-row access), la FMA (q-major / j-minor) then per-value
      // lx FADD (guarded by has_bias), acc += s*la + b*lx.
      float la = 0.0f, lx = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float xv = static_cast<float>(xr[k + q * 8 + j]);
          la += xv * wn[q * 8 + j];
          if (has_bias)
            lx += xv;
        }
      acc += s * la + b * lx;
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
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc);
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(256) qmv_wide_qgd_v3_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int WPER = 32;  // 4-bit only
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int row = blockIdx.y;
  // Same early return as the stock per-row kernel (BEFORE any weight read).
  if (row >= M || col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = K / 2;  // 4-bit
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;
  const T* xr = x + static_cast<size_t>(row) * K;
  float acc = 0.0f;
  // V3: 32B weight loads. The 16B weight vector load is widened to a 32B
  // logical access: two consecutive 16B aligned loads (the 4-bit layout
  // keeps both halves 16B-aligned per lane - no weight reordering, the
  // sidecar is read exactly as stored). Only the FIRST half (the stock
  // 32 values) is used in the math; the second half is accessed with an
  // elision-proofing asm so the 32B window is genuinely touched.
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
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
    // The 32B window overruns the row end (and the buffer) by up to 16B
    // when k + 64 > K (the stock 16B load fits exactly at the last tile:
    // (K-32)/2 + 16 = K/2). Clamp the second half to in-row windows only
    // (values k+32..k+63 are never used in the math - bit-identity - the
    // clamp changes access pattern only).
    bool full32 = (k + 64) <= K;
    uint32_t wv1x = 0, wv1y = 0, wv1z = 0, wv1w = 0;
    if (full32) {
      wv1x = pw[4], wv1y = pw[5], wv1z = pw[6], wv1w = pw[7];  // half 1 (accessed, elision-proofed, unused)
    }
    // Input-only (read-write "r" constraints are unlowerable in AMDGPU
    // codegen: "illegal VGPR to SGPR copy").
    asm volatile("" : : "r"(wv1x), "r"(wv1y), "r"(wv1z), "r"(wv1w));
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // Shared dequant: the weight tile is dequantized once per k-tile
    // (identical values, identical FMA order to the stock inline
    // per-value dequant).
    float wn[32];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
    // Per-row math (EXACT stock order): per-value scalar x loads (the
    // stock per-row access), la FMA (q-major / j-minor) then per-value lx
    // FADD (guarded by has_bias), acc += s*la + b*lx.
    float la = 0.0f, lx = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float xv = static_cast<float>(xr[k + q * 8 + j]);
        la += xv * wn[q * 8 + j];
        if (has_bias)
          lx += xv;
      }
    acc += s * la + b * lx;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc);
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(256) qmv_wide_qgd_v4_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int WPER = 32;  // 4-bit only
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int row = blockIdx.y;
  // Same early return as the stock per-row kernel (BEFORE any weight read).
  if (row >= M || col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = K / 2;  // 4-bit
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;
  const T* xr = x + static_cast<size_t>(row) * K;
  float acc = 0.0f;
  // V4: batched-x load order. The per-tile x loads (32 values) are
  // hoisted into registers and issued as ONE batched group of 16B vector
  // loads (4 x uint4 for 2B T, 8 x uint4 for f32) BEFORE the weight load,
  // instead of the stock 32 scalar per-value loads inside the FMA loop.
  // Same stored layout, same per-(row, col) FMA order - only the memory
  // issue order changes (x batch ahead of the weight stream, for HBM
  // row-buffer locality).
  int niter = (K + 32 * WPER - 1) / (32 * WPER);
  constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2B) / 4 (4B)
  constexpr int NU4 = WPER / VEC;  // 4 (2B) / 8 (4B)
  for (int i = 0; i < niter; ++i) {
    int k = lane * WPER + i * 32 * WPER;
    if (k >= K)
      continue;
    // Batched x hoist: 32 k values, issued before the weight load (values
    // identical to the stock scalar per-value loads).
    uint4 xv4[NU4];
#pragma unroll
    for (int r = 0; r < NU4; ++r)
      xv4[r] = *reinterpret_cast<const uint4*>(&xr[k + r * VEC]);
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    // Shared dequant: the weight tile is dequantized once per k-tile
    // (identical values, identical FMA order to the stock inline
    // per-value dequant).
    float wn[32];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
    // Per-row math (EXACT stock order): la FMA (q-major / j-minor) then
    // per-value lx FADD (guarded by has_bias), acc += s*la + b*lx - the
    // x values now come from the hoisted registers.
    const T* px = reinterpret_cast<const T*>(xv4);
    float la = 0.0f, lx = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float xv = static_cast<float>(px[q * 8 + j]);
        la += xv * wn[q * 8 + j];
        if (has_bias)
          lx += xv;
      }
    acc += s * la + b * lx;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc);
}

}  // namespace mlx::core::rocm
