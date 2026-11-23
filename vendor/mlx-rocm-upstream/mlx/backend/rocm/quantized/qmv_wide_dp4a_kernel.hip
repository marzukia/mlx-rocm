// Copyright © 2025 Apple Inc.

// T3388: M=4 decode dp4a-class port. The ggml mmq Q4_K pattern set
// (the T3381 SASS read of the 70.21 t/s reference, mul_mat_q<type39>)
// transcribed to the 4-bit affine g64 M=4 verify trunk:
//   - integer-exact int8 dot per (row, 32-tile): the mr3 32-per-value
//     f32 FMA chain becomes 32 v_mad_i32_i32 (dot) + 32 v_add (sumx)
//     on the integer ALU, x read as int8, w unpacked from staged
//     int8 (4 MACs per 4-value group, integer-exact: max |dot| =
//     32*15*127 fits i32), then the same two-term scale form
//     `acc += s*(sx*dot) + b*(sx*sumx)` (has_bias guard in place),
//   - LDS double-buffered Q8_1-class staging (weight int8 staged from
//     4B global loads, 8 nibbles per 4B; x int8 staged 8 dwords per
//     (row, lane); 4 scale words per stage),
//   - 4B global loads only for the weight stream (zero 128-bit loads,
//     volatile 4B loads keep the compiler off the uint4 path),
//   - in-order K lanes (stages ascending, 32-tiles ascending, acc
//     updated per 32-tile).
//
// Dot-unit portability read (T3388, measured): the reference's
// v_dot4c_i32_i8 SASS compiles for the gfx1036 (RDNA3) target and
// runs on this gfx1201 card via HSA target override (all 110,086
// v_dot4c occurrences in the a2f1e90e reference fatbin live in its
// 141 gfx1036 ELFs; its 141 native gfx1201 ELFs contain ZERO, and
// the ROCm 7.2 assembler rejects all dot4c variants for gfx1201).
// The native gfx1201 ISA has no int8 dot unit, so the dot is
// materialized on the integer ALU (v_mad_i32_i32 chain, the same
// 4-MAC-per-group structure, integer-exact); every other pattern
// element (int8 pre-quantized x, LDS double-buffered staging, 4B
// global loads, in-order K lanes, the two-term scale form) is
// preserved.
//
// x pre-quantization (the ggml "x pre-quantized to int8" step):
// qmv_wide_dp4a_xquant_kernel runs first, per (row, 256-tile):
// max/127 f32 scale, round to int8, scale in a separate f32 buffer.
//
// FMA-order contract (stated, pre-battery): K-lane order and the
// per-row per-tile accumulation structure are preserved (in-order K
// lanes, m-order rows, 3-stage 8-lane group reduction, the T cast).
// The within-tile math is NOT the per-value FMA chain: integer-exact
// int8 dot + two scale multiplies, x read as int8 (quantized). The
// port is the NEAR-TIE class, not bit-exact (the T3381 portability
// read: the int8 dot on quantized x changes the FP operation tree;
// the dot-unit substitution adds no FP ops - the tile sum is exact).
// The per-arm identity gate vs stream 311 labels the class per the
// T3378-B1/A2a precedent.
//
// Grid: (ceil(N/16), 1); block: 128 (4 waves); 8 lanes per column.
// K_vram = 256 (4 gs64 groups per stage). LDS: 2 x (4096 + 1024 + 16).
// Dispatch (qmm.hip): M = 4 only, 4-bit, bf16 x, K % 256 == 0, gs64;
// OPT-IN only (MLX_ROCM_M4DP4A, DEFAULT OFF — the default dispatch
// is byte-identical when the env is unset).
#include <cstdint>

namespace mlx::core::rocm {

template <typename T>
__global__ void __launch_bounds__(32) qmv_wide_dp4a_xquant_kernel(
    const T* __restrict__ x,
    int8_t* __restrict__ xq,
    float* __restrict__ xsc,
    int M,
    int K) {
  int row = blockIdx.y;
  int t = blockIdx.x;
  int lane = threadIdx.x;
  const T* px = x + static_cast<size_t>(row) * K + t * 256 + lane * 8;
  uint4 v4 = *reinterpret_cast<const uint4*>(px);
  const T* pv = reinterpret_cast<const T*>(&v4);
  float mx = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    mx = fmaxf(mx, fabsf(static_cast<float>(pv[j])));
  // full 32-lane butterfly (o=16,8,4,2,1): the block is one 32-lane
  // wave, each lane holds 8 of the 256 tile values, so the abs-max
  // must reduce across all 32 lanes. (The T3388 earlier in-group
  // o=4/2/1 form was a misread of the geometry - the 8-lane column
  // group belongs to the main kernel, not to xquant; the truncated
  // reduction made sx a group max, clipped the int8 values, and
  // garbled the output. The 5-stage form is the one the CPU model
  // validated.)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    mx = fmaxf(mx, __shfl_down(mx, o));
  float sx = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;
  float inv = 1.0f / sx;
  uint32_t qp[2] = {0u, 0u};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float qf = fminf(127.0f, fmaxf(-127.0f, static_cast<float>(pv[j]) * inv));
    int v = static_cast<int>(lrintf(qf));
    qp[j >> 2] |= static_cast<uint32_t>(v & 0xFF) << ((j & 3) * 8);
  }
  *reinterpret_cast<uint2*>(
      xq + static_cast<size_t>(row) * K + t * 256 + lane * 8) =
      make_uint2(qp[0], qp[1]);
  if (lane == 0)
    xsc[static_cast<size_t>(row) * (K / 256) + t] = sx;
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(128) qmv_wide_dp4a_kernel(
    const int8_t* __restrict__ xq,
    const float* __restrict__ xsc,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int MAXM = 4;
  constexpr int NCOLS = 16;
  int lane = threadIdx.x;
  int lane8 = lane & 7;
  int col = blockIdx.x * NCOLS + (lane >> 3);
  int nrows = (M < MAXM) ? M : MAXM;
  if (col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const uint8_t* wr = w + static_cast<size_t>(col) * (K / 2);  // 4-bit row
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr =
      has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;

  extern __shared__ uint8_t smem[];
  // per stage buffer: w 16 x 256 int8 (1024 dwords), x 4 x 256 int8
  // (256 dwords), scale 4 f32. Double-buffered: two of each.
  uint32_t* wL = reinterpret_cast<uint32_t*>(smem);
  uint32_t* xL = wL + 2048;
  float* scL = reinterpret_cast<float*>(xL + 512);

  int ns = K / 256;
  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    acc[m] = 0.0f;


  // stage prefetch: stage st -> buffer nb. 4B global loads only (the
  // volatile 4B loads keep the weight stream off the uint4/b128 path;
  // zero 128-bit loads in the weight stream by construction).
  auto prefetch = [&](int st, int nb) {
    // 32 nibbles = 16B per lane per column: 4 x 4B volatile loads.
    const volatile uint32_t* wp4 =
        reinterpret_cast<const volatile uint32_t*>(
            wr + (static_cast<size_t>(st) * 256) / 2 + lane8 * 16);
    uint32_t wreg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      wreg[i] = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t v = wp4[i];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // 4-bit value 0..15 (int8 range, no sign extension needed).
        uint32_t q = (v >> (4 * j)) & 0xF;
        // value 4i + j (4 nibbles per 4B word) -> dword 2i + (j/2),
        // bit (j%2)*8 (nibble-major, ascending within the 32-tile, the
        // base kernel's order).
        wreg[2 * i + (j >> 1)] |= q << ((j & 1) * 8);
      }
    }
    uint32_t* wbase = wL + nb * 1024 + col * 64 + lane8 * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      wbase[i] = wreg[i];
    for (int m = 0; m < MAXM; ++m) {
      if (m >= nrows)
        break;
      const volatile uint32_t* xg = reinterpret_cast<const volatile uint32_t*>(
          xq + static_cast<size_t>(m) * K + st * 256 + lane8 * 32);
      uint32_t* xb = xL + nb * 256 + m * 64 + lane8 * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        xb[i] = xg[i];
    }
    if (lane8 == 0) {
      int cb = lane >> 3;  // 0..3
      uint32_t* sb = reinterpret_cast<uint32_t*>(scL + nb * 4 + cb);
#pragma unroll
      for (int m = 0; m < 4; ++m)
        sb[m] = (m < nrows) ? __float_as_uint(xsc[static_cast<size_t>(m) * ns + st])
                           : 0u;
    }
  };

  prefetch(0, 0);
  __syncthreads();

  for (int st = 0; st < ns; ++st) {
    int bi = st & 1;
    if (st + 1 < ns)
      prefetch(st + 1, (st + 1) & 1);
    __syncthreads();
    float sxv[4];
    // Volatile LDS reads: the ROCm 7.2 (LLVM 20) O3 pass folds
    // addrspace(3) reads with the conditional same-thread store and its
    // phi default (0), which DCEs the whole dot chain (writes a constant
    // 0). A volatile read is a real ds_read and cannot be folded.
    const volatile float* scLv = reinterpret_cast<const volatile float*>(scL);
#pragma unroll
    for (int m = 0; m < 4; ++m)
      sxv[m] = scLv[bi * 4 + (lane >> 3)];
    const volatile uint32_t* wbase =
        reinterpret_cast<const volatile uint32_t*>(
            wL + bi * 1024 + col * 64 + lane8 * 8);
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m >= nrows)
        break;
      const volatile uint32_t* xb =
          reinterpret_cast<const volatile uint32_t*>(
              xL + bi * 256 + m * 64 + lane8 * 8);
      // This lane's 32 K values for the stage, ascending K order:
      // 8 x dwords x 4 int8; w value v = 4i + j is at byte (j >> 1) of
      // wbase[2i + (j & 1)] (nibble-major staging, 0..15 unsigned).
      // Integer-exact dot on the integer ALU (v_mad_i32_i32 chain,
      // 4 MACs per 4-value group - the reference's dot4c unit is
      // gfx1036-only, see the header portability read; the i32 chain
      // is the same structure, exact: max |dot| = 32*15*127, fits
      // i32). The x-dword loop is a real loop: the fully-unrolled
      // 1024-MAC form is miscompiled by the ROCm 7.2 (LLVM 20) SLP
      // pass (T3388 bisection: SLP-on-O2+GVN/EarlyCSE context DCEs
      // the dot; the trip-8 serial form cannot be vectorized).
      int dot = 0;
      int sumx = 0;
      for (int i = 0; i < 8; ++i) {
        uint32_t xv = xb[i];
        uint32_t wlo = wbase[2 * i];
        uint32_t whi = wbase[2 * i + 1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int x = static_cast<int>(static_cast<int8_t>(
              static_cast<uint8_t>(xv >> (8 * j))));
          int w = (((j & 1) ? whi : wlo) >> ((j >> 1) * 8)) & 0xFFu;
          dot += w * x;
          sumx += x;
        }
      }
      // gs64 group g = lane8/2 (2 x 32-lanes per 64 group), the base
      // kernel's two-term update (has_bias guard in place):
      // acc += s*la + b*lx with la = sx*dot, lx = sx*sumx (the
      // integer-exact tile sum, the within-tile tree change).
      int g = lane8 >> 1;
      float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[st * 4 + g]);
      float b = has_bias ? static_cast<float>(bcr[st * 4 + g]) : 0.0f;
      acc[m] += s * (sxv[m] * static_cast<float>(dot)) +
          b * (sxv[m] * static_cast<float>(sumx));
    }
  }

  for (int m = 0; m < MAXM; ++m) {
    if (m >= nrows)
      break;
    float a = acc[m];
    // 3-stage reduction across the 8-lane column group (shfl_up form:
    // the 8-lane subset cannot use the plain shfl_down ladder).
#pragma unroll
    for (int o = 4; o > 0; o >>= 1)
      if (lane8 >= o)
        a += __shfl_up(a, o);
    if (lane8 == 0)
      out[static_cast<size_t>(m) * N + col] = static_cast<T>(a);
  }
}

}  // namespace mlx::core::rocm
