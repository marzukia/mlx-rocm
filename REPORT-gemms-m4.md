# REPORT-gemms-m4 — Closing the M=4 GEMM excess in the K=3 MTP verify step

**Task:** reduce verify 22.13 ms → ≤ 16 ms and pair 41.35 ms → ≤ 35 ms by getting the
MLP + GDN projection GEMMs from 1.81x / 1.47x toward the 1.3x linear regime at M=4.
Hardware: guest-a, RX 9070 XT (gfx1201 / RDNA4), ROCm 10.1 userland, wheel built from
`vendor-rs` @ `75ca881` (= the `vendor.json` pinned ref).

**Verdict: NO improvement this session.** The B0 gate passed; the tested M≤4 lever
(decode-hoist) is a mild **regression**. The diagnosis is now precise: the M=1→M=4
GEMM excess is **scalar-FMA-throughput / load-latency bound, not a dispatch gap and not
weight-decode-bound**. The real lever is a **WMMA (bf16 matrix-unit) M≤4 kernel**, which
is the concrete next step (design below).

---

## 1. Diagnosis (step 1)

**Which kernel runs the M=4 GEMMs with `QMV_ROWS=1`?** With the task config
(`MLX_ROCM_QMV_ROWS=1`, `MLX_ROCM_QMV_ROWS_MIN_N=1`), **every M=4 GEMM shape runs
`qmv_rows_msg_kernel`** (the "rows-msg" morphology — Metal simdgroup→HIP wave32, 6 waves
/block, no LDS, no barriers). The `MIN_N=1` override is what forces it at all shapes:
the default `rows_min_n` floor is 16384, so by default only the large-N shapes
(gate_up N=24576, lm_head N=248320) would engage; `MIN_N=1` extends it to the GDN
projections (N=8192/4096), attention (N=4096) and MLP down (N=4096).

The rows-msg kernel **already does weight-reuse**: each (column, k-tile) 4-bit weight tile
is decoded once and MAC'd against **all M rows** (one weight stream pass regardless of M),
and it is **row-exact** (output row r == `quantized_matmul(x[r:r+1])` bit-for-bit — verified
in this session, maxdiff 0.0). So the rows path *is* dispatching and *is* reusing weights at
these shapes.

**Why the excess persists under weight-reuse.** The model is Qwen3-9B: hidden 4096,
intermediate 12288, 32 layers (24 GDN + 8 full-attn). At M=4:
- MLP gate_up (N=24576, K=4096), down (N=4096, K=12288), GDN in_proj (N=8192, K=4096),
  GDN out_proj (N=4096, K=4096), attention (N=4096, K=4096), lm_head (N=248320, K=4096)
  all run `qmv_rows_msg_kernel`.
- The rows-msg inner loop is **scalar FMA** (float32): per (row, col, k-tile) it does 32
  FMA + 32 decode (shift/mask/cvt) + an x-tile load. Weight-reuse removes the redundant
  *weight stream read*, but **the M=4 FMA compute is 4× the M=1 FMA** — that is the excess.
  Removing the weight read does not remove the 4× FMA.

**The decode-hoist experiment (the tested lever) proves the GEMMs are not decode-bound.**
I implemented `qmv_rows_msg_hoist_kernel` (opt-in `MLX_ROCM_QMV_ROWS_HOIST=1`, default
off, base kernel untouched): it hoists the per-column 4/8-bit weight **decode** to once
per column (column-outer / row-inner) instead of once per (row, column), cutting the
weight-decode ALU by ~37% (per-lane per-k-tile: 1024 → 640 ops). It is **bit-identical**
to the per-row reference and to the base rows-msg kernel (row-exact test, maxdiff 0.0,
finite — the ROCm 10.0 32-element-array codegen defect did **not** manifest on ROCm 10.1).

But it is **slower**, not faster:

| arm | toks/s | verify ms | pair ms |
|---|---|---|---|
| B0 (rows OFF, default) | 70.02 | 25.66 | 44.85 |
| control (rows ON, no hoist) | 75.36 / 74.81 | 22.04 / 22.24 | 41.16 / 41.46 |
| **R1 (rows ON + hoist)** | **74.12 / 74.46** | **22.28 / 22.51 / 22.60** | 41.49 / 41.66 / 41.84 |

The hoist reduces ALU ops yet gets ~0.4 ms slower on verify (a ~1.5–2% regression). The only
thing it added net was 4× more x-tile loads (x is re-read per column instead of shared
across columns) plus register pressure from the 32-element decode array. A kernel that gets
*slower* when its ALU/decode work is cut is **not ALU/decode-bound** — it is bound by
scalar-FMA throughput and/or memory-load latency. That is the 4× FMA physics: at M=4 the
kernel issues 4× the multiply-accumulate of M=1, and the ALU/FMA pipe + the L2 x-load
stream are the wall.

## 2. B0 gate (step 4, gate)

**PASS.** B0 (rows OFF, `MLX_ROCM_QMV_ROWS` unset) on the rebuilt wheel: **70.02 tok/s**,
verify 25.66 ms, pair 44.85 ms, n_refwd 100 — reproduces the required 69.48 tok/s within
+0.77% (well inside ±3%). The default path is byte-identical to the prior wheel (the change
is gated default-off), confirming no behavior change when rows is off. (n_refwd 100 ≈ the
expected 101.)

## 3. R1 result (step 4)

R1 (rows ON + hoist) = **74.12 tok/s, verify 22.60 ms, pair 41.84 ms** — a **regression**
vs the current best (rows ON, no hoist: 75.36 / 22.04 / 41.16) and below the 75.00
"current best" reference. The targets (verify ≤ 16 ms, pair ≤ 35 ms) are **not met**.
The rows-msg kernel's existing benefit (rows ON vs OFF: verify 25.66 → 22.04, −3.62 ms,
+7.6% tok/s) is real and already in the wheel; the decode-hoist on top of it does not add
and slightly subtracts.

## 4. The real lever: WMMA M≤4 kernel (recommendation)

The guest reports `native_wmma=1` (RDNA4 bf16 matrix units). The scalar FMA is the wall;
the fix is to move the M×K×N multiply-accumulate onto `wmma::mma_sync` (16×16×16 bf16).

Design (new opt-in kernel `qmv_wmma_m4_kernel`, `MLX_QMV_WMMA=1`, default off):
- 4-bit affine, per-group scale/bias. Within a 16-wide k-tile the group is constant
  (GROUP_SIZE ∈ {32,64,128} ≥ 16), so the dequant `w_dequant[n,k] = scale[g]·w4[n,k] +
  bias[g]` is applied once per 16×16 weight tile → bf16.
- GEMM `out[m,n] = Σ_k w_dequant[n,k]·x[m,k]`: A fragment = x tile (16×16, m×k), B fragment
  = `w_dequantᵀ` (16×16, k×n), accumulate 16×16 C across the k-tiles. M≤4 means the A
  fragment uses only 4 of 16 rows (some waste), but the matrix units still beat scalar FMA
  by ~4–8×.
- This changes FMA order/precision → **statistical, not bit-exact** (gate with a
  maxdiff ≤ 1e-2 tolerance, not the row-exact 0.0 contract).
- Expected: removes most of the 4× FMA excess; the 1.81x/1.47x → ~1.3x linear regime is
  plausible, i.e. verify 22 → ~16–18 ms. Hitting ≤ 16 ms is not guaranteed in one pass —
  the residual is the dequant (weight→bf16) + the B-fragment transpose cost, which is the
  follow-up to tune.

## 5. What's delivered / preserved

- **This report** (diagnosis + B0 gate + R1 negative result + WMMA design).
- The decode-hoist experiment source is preserved in **Appendix A** (kernel) and
  **Appendix B** (dispatch). It is gated `MLX_ROCM_QMV_ROWS_HOIST=1`, default off
  (zero behavior change when unset), so it is safe to keep in the wheel as a documented
  negative experiment. It is **not** re-pinned into the `vendor.json` base (it is a
  regression, not an improvement).

## 6. Numbers summary

| metric | B0 (rows OFF) | control (rows ON) | R1 (rows ON + hoist) | target |
|---|---|---|---|---|
| tok/s | 70.02 | 75.36 / 74.81 | 74.12 / 74.46 | — |
| verify ms | 25.66 | 22.04 / 22.24 | 22.28 / 22.51 / 22.60 | ≤ 16 |
| pair ms | 44.85 | 41.16 / 41.46 | 41.49 / 41.66 / 41.84 | ≤ 35 |
| n_refwd | 100 | 102 | 102 | — |

B0 gate: **PASS** (70.02 vs 69.48, +0.77%, inside ±3%). R1: **regression** vs control;
targets not met. Next step: WMMA M≤4 kernel (section 4).

---

## Appendix A — `qmv_rows_msg_hoist_kernel` (added to `qmv_rows_msg_kernel.hip`, `#include`d by `qmm.hip`)

```cpp
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
```

## Appendix B — `qmm.hip` dispatch (inside the `rows_ok` branch)

```cpp
static bool use_rows_qmv = (std::getenv("MLX_ROCM_QMV_ROWS") != nullptr);
static bool use_hoist =
    (std::getenv("MLX_ROCM_QMV_ROWS_HOIST") != nullptr);   // NEW, default off
...
#define LAUNCH_QMV_ROWS_MSG_H(TT, ST, GS, BV)                     \
  enc.add_kernel_node(                                          \
      &rocm::qmv_rows_msg_hoist_kernel<TT, ST, GS, BV>, rgrid, rblock, 0u,   \
      (const TT*)x_ptr, w_ptr, (const ST*)scales_ptr,           \
      (const ST*)biases_ptr, (TT*)out_ptr, M, N, K, has_bias)
#define LAUNCH_QMV_ROWS_MSG_GS(TT, ST, BV)                      \
  do {                                                          \
    if (use_hoist) {                                            \
      if (group_size_ == 32) { LAUNCH_QMV_ROWS_MSG_H(TT, ST, 32, BV); }   \
      else if (group_size_ == 64) { LAUNCH_QMV_ROWS_MSG_H(TT, ST, 64, BV); } \
      else { LAUNCH_QMV_ROWS_MSG_H(TT, ST, 128, BV); }           \
    } else {                                                    \
      if (group_size_ == 32) { LAUNCH_QMV_ROWS_MSG(TT, ST, 32, BV); }   \
      else if (group_size_ == 64) { LAUNCH_QMV_ROWS_MSG(TT, ST, 64, BV); } \
      else { LAUNCH_QMV_ROWS_MSG(TT, ST, 128, BV); }           \
    }                                                           \
  } while (0)
```

When `MLX_ROCM_QMV_ROWS_HOIST` is unset (default), the base `qmv_rows_msg_kernel` is
selected exactly as before — zero behavior change.
