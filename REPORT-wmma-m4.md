# REPORT-wmma-m4 — WMMA M≤4 QMV kernel on gfx1201 (negative result)

Task: t1339 (resumed from cancel80 at 3h19m). Fix the Rows≥1 garbage bug in the WMMA
M≤4 QMV kernel, pass the unit gate, A/B spot check, canonical benchmark (B0 /
ROWSONLY / R1).

**Verdict: the WMMA lever is a −31% regression in its current form.** Correctness
gates all pass; performance does not. The kernel genuinely uses RDNA4 WMMA
hardware (SASS-verified) and is at parity-or-better on isolated GEMMs, but in the
full model forward it costs +25–44µs per call vs the rows scalar kernel. That is
a data point, not a stop condition, per task brief — documented here.

## 1. The bug fix (what made the kernel correct)

Root cause of Rows≥1 garbage: the gfx1201 16×16×16 f32 WMMA accumulator fragment
layout is **lane L, element e → row 8·(L>>4)+e, col L&15** (probe-verified), not
the row-major-in-warp layout the first store loop assumed. Fixed in
`patches/qmv-wmma-m4/qmv_wmma_m4_kernel.hip` (store loop), plus fC zero-init and
debug-print removal. Commit `d55d511`.

## 2. Correctness gates — all PASS

- **Unit gate** (builds 18 and 19): 24/24 combos (6 shapes × M=1..4), worst
  maxdiff **0.007812 ≤ 1e-2**, zero NaN/Inf. `GATE1: PASS` on both builds.
- **A/B spot check** (K=3, 200 tokens): **200/200 identical, zero divergence**
  (bar: first divergence ≥ 100 or zero).
- **Model-level spot check**: maxdiff 0.0715 (bf16 noise), argmax identical
  (534190). The perf gap is a pure performance issue, not correctness.

## 3. Canonical benchmark (guest-a, RX 9070 XT, ROCm 10.1, canonical
`mtp-bench-k3-tape.py` + `gdn_tape_commit.py` md5-verified, sidecar pin
`model-mtp.safetensors` active)

| Arm | tok/s | verify ms | refwd ms | pair ms |
|---|---|---|---|---|
| B0 (rows OFF) | 70.26 / 69.84 | 25.54 / 25.79 | 9.09 / 9.21 | 44.70 / 44.97 |
| ROWSONLY (rows ON) | 74.75 / 74.77 | 22.25 | 9.32 | 41.49 / 41.48 |
| R1 (rows+wmma) | 48.55 / 48.52 / 48.85 | 33.69 / 33.90 | 23.18 / 23.03 | 66.94 / 66.98 / 66.53 |

- B0 control: 70.26/69.84 vs target 69.48 ±3% → **PASS** (+1.1% / −0.2%).
- R1 targets: verify ≤16 ms, pair ≤35 ms → **FAIL** (33.7–33.9 / 66.5–67.0).
- Regressions: **−30.7% vs B0, −34.7% vs ROWSONLY** (rows alone is a +6.4%
  win; WMMA on top costs −35%).

## 4. Root-cause work on the perf gap

**SASS**: `v_wmma_f32_16x16x16_f16` present in the compiled kernel → hardware
WMMA units are genuinely used, not a scalar-FMA fallback.

**Model GEMM inventory** (qwen3-9b-mlx, hybrid GDN, 4-bit group 64, bf16 scales
+ bf16 zero-point, ~249 quantized GEMMs per M=4 forward):

| N | K | count |
|---|---|---|
| 32 | 512 | 48 |
| 1024 | 512 | 16 |
| 4096 | 512 | 56 |
| 4096 | 1536 | 32 |
| 8192 | 512 | 32 |
| 12288 | 512 | 64 |
| 248320 | 512 | 1 |

**fwd4 (real model, M=4, 10-iter median)**: rows 31.63 ms vs wmma 43.43 ms
(+11.8 ms).

**N-gate bisection** (diagnostic `MLX_QMV_WMMA_MIN_N` build 19):

| min_n | which GEMMs use wmma | fwd4 median | Δ vs rows |
|---|---|---|---|
| 249000 | lm_head only | 31.59 | ≈ 0 |
| 29000 | + N=8192/12288 (96 calls) | 35.82 | +4.2 ms (+44µs/call) |
| 5000 | + N≤4096 (152 calls) | 39.70 | +8.1 ms (+25µs/call more) |
| 0 | all | 43.43 | +11.8 ms |

**Isolated GEMM parity** (bf16 x, **bf16 scales** — matching the model exactly,
group 64, 20-iter median):

| K | N | rows ms | wmma ms |
|---|---|---|---|
| 512 | 32 | 0.053 | 0.048 |
| 512 | 1024 | 0.046 | 0.044 |
| 512 | 4096 | 0.050 | 0.046 |
| 512 | 8192 | 0.062 | 0.048 |
| 512 | 12288 | 0.067 | 0.051 |
| 1536 | 4096 | 0.060 | 0.059 |

Wmma is **faster on 4 of 6 classes, at parity on 2** in isolation. (An earlier
"parity" data set was a null result — `nn.QuantizedLinear` microbench scales are
f32, which the dispatch gate rejects, so both arms ran the scalar kernel.)

**Ruled out**:
- CPU enqueue: build phase identical (1.34 ms both arms).
- Strided x: parity.
- Back-to-back GEMM load (248 calls): parity (5.67 ms both).
- Interleaved elementwise context (248 GEMMs + norm-like ops): wmma **faster**
  (5.35 vs 6.95 ms).
- f32 x / f32 scales: gate rejects → both arms scalar (no differential).

**Conclusion**: the +25–44µs/call penalty appears *only* in the full model stream
(GDN attention + SDPA + GEMM interleave with the exact QTensor weight views).
Isolated GEMM/elementwise emulations cannot reproduce it; the most likely
mechanism is scheduling/occupancy interaction of 32-thread (1-wave) WMMA
workgroups with the rest of the model's kernel stream on gfx1201, or the
loaded-model QTensor weight view layout, which the raw Python
`quantized_matmul` API on this mlx build could not be made to reproduce in
isolation (it rejects the loaded (N,K)-logical u32 QTensor with an
`expanded quantized matrix (4096, 248320)` shape error — an mlx API quirk,
orthogonal to the kernel).

## 5. Recommendation

Do not ship the WMMA lever as-is. To make it win, the per-k-tile dequant/staging
overhead and 1-wave workgroup granularity must go: larger K-tiles (fewer dequant
rounds per FMA wave), or in-register dequant, or a 2/4-wave block shape that
interleaves better with the model stream. The dispatch gate (MIN_N) already
exists and shows a safe per-shape fallback: min_n=249000 (lm_head-only wmma) is
free; min_n=29000 gives back part of the cost — a *partial* gate
(wmma only for N ≥ 29000) is net-neutral and buys nothing; a *skip* gate
(wmma only for N ≥ 5000 minus the 8192/12288 classes) would claw back ~4.2 ms
but the kernel as written has no N-class selectivity inside the model stream
beyond the MIN_N threshold, and no threshold setting beats rows.

## 6. Artifacts

- Wheel build 19: `mlx-0.32.3.dev20261102+59df5d9-cp314`, libmlx.so md5
  `4e5b423dd5583eb1e88c65b94268eca4` (with diagnostic MIN_N gate; the delivery
  patch set `patches/qmv-wmma-m4/` is clean of diagnostics).
- Build 18 (pre-bisect): libmlx.so md5 `7306af5dde58961a730b92784454c28f`.
- Vendored source pin: vendor.json (unchanged this task; only the guest scratch
  tree carries the diagnostic gate, reverted-not-applicable — scratch only).
- Guest scratch: `~/t1339-wmma/` (unit gate scripts, microbenches
  m4micro11/12, fwd4/bisect, unit19 gate PASS log).
