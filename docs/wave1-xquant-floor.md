# WAVE 1/3 (FAILURE-WAVE, ZERO-GPU, xhigh) — P1-d Q2_K mmq leg-3 terminal: the x-quantization floor claim

Date: 2026-11-27 (hydrogen, CPU-only; box queries read-only)
Auditor: fresh zero-GPU wave-1 run (independent of the leg-3 submitter)
Scope: confirm / refute / modify the claim that **2.29e-1 is the
correctly-measured architectural q8_1 d2s6 x-quantization floor (not floor +
residual defect)** on the G1 battery.

Source of record: `p1d-forkb-optionb-kernel` @ 42dd9c6 (mirror), findings card
4b0566b, /tmp/g1v2b.py + /tmp/xq2.py (read-only), wheel md5
23baab1985fc94042f06d3a95834d947 (re-verified on the box = the record).

## Verdicts

### 1. The 2.29e-1 floor number — CONFIRMED (independent numpy re-derivation)

Re-derived from source, independently of /tmp/xq2.py:

- **x-quantizer** (my own numpy implementation of
  `mmq_quant_in_q8_1_mmq`, d2s6 branch, mmq.hip @ 42dd9c6): per 128-block,
  two 64-value amax scales `d_inv = 127/amax`, `d = 1/d_inv` (the two-step
  f32 form), `t = clamp(rint(v * d_inv), [-127,127])`, f16-stored (RNE) d and
  `s_g = f16(d_f32 * qsum_g)` (integer-exact 16-value qsums, i<96 only);
  dequant per the option-B kernel (mmq-vec-dot.cuh `ggml_cuda_mmq_vec_dot_q2_K_file`):
  `x̂ = f16(d_g) * t + s_g/16` for k 0..95, `f16(d1) * t` for k 96..127.
- **W ground truth**: `deq_elem ktype==10` (kquant_file_gemm.hip), mul-mul-sub
  each rounded once, on the real `blk.3.attn_v.weight` (K=5120, N=1024,
  tensor bytes pulled read-only; max|W| = 0.226551).
- **Battery x** (identical spec to g1v2b.py): `RandomState(42).normal(0, 0.1,
  (2, 5120))`, row 0; ref = `x1 @ W` f32 (max|ref1| = 0.412988).

| variant | max|d| | rel |
|---|---|---|
| floor, kernel-faithful (s_g from f32 d) | 9.456496e-02 | 2.289772e-01 |
| floor, /tmp/xq2.py variant (s_g from f16 d) | 9.458412e-02 | 2.290236e-01 |
| **claimed (card)** | **9.458e-02** | **2.290e-01** |

Both independent variants round to **2.290e-01 at 4 digits** = the claimed
floor. The 1.9e-03 max|d| gap between the two variants is the xq2.py replica
deviation (see item 2), 0.2% of the floor — no effect on the 4-digit claim.

"No residual defect on top of the floor": the measured kernel total
(2.290e-01, on the record) sits inside the triangle bound
`total ≤ floor + own = 2.289772e-01 + 6.59e-05 = 2.290431e-01`, the
kernel-own error (2.722e-05, rel 6.592e-05 — re-checked against my
max|ref1|: 2.722e-5/0.412988 = 6.591e-05 ✓) is 15x below the 1e-3 gate, and
F2 is all-green (exact 3.0 within f32 noise, 2.99982). The total is
floor-dominated; no defect of order ≥ 2e-4 is consistent with the recorded
numbers. The kernel-own 2.722e-05 is of the same order as (a) f32
accumulation-order noise over K=5120 (~3e-5 class) and (b) the xq2.py
replica deviation below — the record's "pure f32 accumulation noise"
characterization is defensible at that magnitude.

### 2. /tmp/xq2.py vs /tmp/g1v2b.py setup consistency — CONFIRMED (one
documented deviation, quantified, immaterial)

Read both scripts in full. They share: the GGUF parser (byte-identical),
the tensor (`blk.3.attn_v.weight`), the W dequant (byte-identical formulas,
bit-exact vs `deq_elem ktype==10`), the x (same `RandomState(42)`,
`normal(0, 0.1, (2, K))`, `x1 = x2[0:1]`), the reference (f32 `x1 @ W`),
the rel definition (`max|d| / max|ref|`), and the kqfg call
(`kquant_file_gemm(x, wb, 10, 0, 0, 32, 1, 15)`). **No setup divergence
between the floor measurement and the G1 battery.**

One deviation inside the floor replica: the kernel stores
`s_g = f16(d_f32 * qsum_g)` (mmq.hip d2s6 branch: `d_g0 * (float)qsum[2*g]`
with the f32 `d0/d1`), while xq2.py computes `s_f16 = f16(f16(d) * qsum)`.
Impact, measured (item 1 table): 1.9e-03 max|d| / 4.6e-05 rel — 0.2% of the
floor, far below the 4-digit claim tolerance and far below the 2.722e-05
kernel-own figure's significance. Not a finding that changes any number on
the card; noted for the record.

### 3. "Option A included" — the floor is not option-B-specific — CONFIRMED

- Every in-tree mmq launcher pre-quantizes x with `mmq_quant_in_q8_1_mmq`:
  the Q2_K file path (option B, M ≤ 16 via the `x_file = (J_best == 16)`
  branch, `d2s6=true`), the in-tree Q2_K dp4a M>1 path (same d2s6 layout —
  `ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a` reads word 0 half2 (d0,d1) + the
  per-16 sums at words 1+), and the Q4_K M>1 class (`ds4=true`: per-32
  f16 d + per-32 raw-sum min correction). All quantize x to q8_1 int8 with
  per-sub-block `amax/127` scales and `rint` — the dominant floor term.
- Leg-2 option A (the f16/bf16 WMMA rewrite) by its own design (leg-2 card
  §3: "B = y-code (f16, exact ints)", per-k f16 pre-scales `dy' = d*dy,
  my' = d*my`) consumes the same q8_1 d2s6 x (the same pre-quantizer; only
  the mma dtype differs, adding a secondary W-side f16 double-rounding
  ~1e-3). Option A carries the same x-quantization floor.
- Corroboration from leg-2's independent analysis: repacking x onto a coarser
  grid gives ~10-30% per-element error / ~0.1 GEMV rel — 100x over the gate.
  The q8_1 x floor is the architectural cost of the mmq class, not an
  option-B artifact.

### 4. The "calibrated to kqfg-class" claim — CONFIRMED (measurement facts
for the wave-3 gate spec)

- The kqfg path takes x raw (kquant_file_gemm.h: "x (M, K) float32/
  bfloat16"), dequantizes W in-kernel (bit-exact `deq_elem`), f32
  k-order accumulation — **no x-quantization at all**. Measured KQFG M=2 =
  rel 2.454e-06 (on the record; consistent: 1.013e-6/2.454e-6 = 0.4128 ≈
  my independent max|ref| 0.4130).
- 1e-3 vs 2.454e-06 = ~408x margin: the literal tolerance is consistent
  with kqfg-class calibration. The mmq floor (2.290e-01) is 229x the gate:
  **no correct mmq implementation can pass the literal 1e-3 gate** — the
  failure is floor-bound (spec-incorrect for the mmq path), not
  kernel-defect. The kernel-own interpretation (6.59e-05 ≤ 1e-3) passes
  with 15x margin.
- Factual correction of one card annotation (does not change any number):
  the E0 identity is `0.0625 * max|W[0,:] + Σ_{j=1..15} W[j,:]` =
  0.0625 * 0.225531 = 1.40957e-02 (my replica on the real tensor, 4-digit
  match with the card's 1.409e-02); the card's "= 0.0625 * max|W[0,:]|"
  as written gives 2.88e-03 (max|W[0,:]| = 0.0461) — the 1/16
  min-correction spills over rows 1..15 of the one-hot's 16-group, and that
  spill is what the number actually is. The 1.0625x factor itself is exact
  (x̂[0] = (1/127)*127 + 1/16 = 1.0625).

## Overall verdict: CONFIRMED

2.29e-1 is the correctly-measured architectural q8_1 d2s6 x-quantization
floor, verified to 4 digits by an independent numpy re-derivation from the
in-tree source (9.456e-02 / 2.2898e-01 kernel-faithful; 9.458e-02 /
2.2902e-01 xq2 variant), the kernel total is floor-dominated (own 6.6e-5),
the floor is shared by option A, and the literal 1e-3 gate is
kqfg-class-calibrated and unattainable on any mmq path.

## NAMED RECOMMENDATION (to the lane / wave 3)

1. **Dispatch rule**: M ≤ 1 → kqfg (raise `mr_gate` ≥ 1, or route in
   `KQuantFileGemm::eval_gpu`); keep the option-B kernel as the env-gated
   fast path (MLX_KQFILE_MMQ_M1=1) pending the operator ruling on whether
   the 2.29e-1 x-quantization floor is acceptable for serve.
2. **Gate spec (for wave 3)**: split the G1 gate — the literal 1e-3 rel
   applies to the kqfg path (measured 2.45e-6, 408x margin); the mmq path
   is gated on **kernel-own error vs the exact x-quant reference** (measured
   6.59e-5, 15x margin), with the architectural floor (2.29e-1, x-quant
   only, W bit-exact) reported as a separate, non-gated architecture
   constant. A single literal 1e-3 on the mmq path is spec-incorrect and
   can never pass.
3. Record the two E0-annotation and xq2.py-replica notes (item 2, item 4)
   as documentation corrections in the findings card; no number changes.

## Method / provenance

- Independent re-derivation on hydrogen (CPU): fresh numpy implementation of
  the d2s6 quantizer + W dequant written from the 42dd9c6 source (not from
  xq2.py); tensor bytes (20480 x 84B, md5 92eb0db4bef078ac4c113d0c9831f9ab)
  and header info (dims [5120, 1024], t=10) read-only from the box
  (neon, RX 9070 XT / gfx1201); wheel md5 re-verified = the record.
- No GPU work, no builds, no kernel launches in this run.

## 6b CONFABULATION INTERROGATION (orchestrator addendum, 2026-11-27 09:3xZ)

Per the wave-16 ruling obligation (cloudy, 08:5xZ): the absurd-resource-number
frame - a 22.9% rel floor = interrogate the metric. The x~N(0,0.1)
near-cancellation battery inflates the relative metric; is the floor real for
SERVE-QUALITY = token flips under greedy near-ties, or a metric artifact of
the test battery? This section was appended by the orchestrator (the worker
card landed without it; the wave-1 brief addendum was posted after the
worker's brief-read phase).

1. METRIC MECHANICS (DERIVED, labeled): the battery's max|d|/max|ref| is a
   MAX-BASED relative metric on near-cancelling zero-mean inputs: outputs
   |y[n]| cluster small (sum of K=5120 cancelling terms), so the relative
   error's extreme tail is inflated where |y| ~ 0. The per-element q8_1 d2s6
   RMS quantization noise is step/12 with step = 2*amax/127 per 64-group: for
   x ~ N(0,0.1), per-64 amax ~ 0.1*2.65 ~ 0.265 -> RMS dx ~ 1.2e-3 ~ 1.2% of
   sigma(x); the output RMS relative error is the same ~1e-2 class (dy =
   sum(dx*W) scales with the same sqrt(K) as y). So: 2.29e-1 is REAL for the
   stated metric (item 1 confirms it to 4 digits) but is the worst-case
   relative number on adversarial near-cancellation inputs, not a typical
   error. The typical class is ~1e-2 relative.
2. SERVE-QUALITY READING: a greedy token flip requires |delta logit| > the
   top-2 logit gap at a near-tie; at the ~1e-2 RMS class, flips are confined
   to gaps below ~1e-2 relative - a small fraction of the softmax mass.
   Record evidence: (a) the 41.2 t/s llama.cpp control (the reference-quality
   lane control on the same file) runs the mmq q8_1 x-quantization class
   (GGML_HIP + MMQ_MFMA) - the same architectural floor, same file, same box;
   (b) the vehicle 2.5-bit run (mmq path active for M>1) produced coherent
   output (5gram-dup-rate 0.000 on the adversarial d500x prompt). No
   incoherence at this floor is on the record.
3. NAMED CPU-SIDE SETTLING MEASUREMENT (zero-GPU, runnable pre-P1-e):
   argmax-flip + near-tie-margin comparison from the record artifacts (numpy
   W ref + numpy d2s6 replica): exact kqfg logits (x@W f32) vs mmq-floor
   logits (d2s6-quantized x @ W) on (i) the g1v2 battery x and (ii) a
   non-cancelling control (constant-positive x of the same scale). Report:
   top-1 flip count, the top-2 gap distribution of the flipped items, and the
   fraction of outputs with |delta| > gap. Expected: flips only at sub-delta
   gaps; the max-relative number contracts on the non-cancelling control.
   The serve-quality ruling then rides on this CPU number + the P1-e flip
   A/B (token-agreement rate on real 27B decode), not on the literal metric.
