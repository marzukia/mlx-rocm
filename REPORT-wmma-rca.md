# REPORT-wmma-rca — RCA of the 48.5 tok/s WMMA collapse + ranked alternatives

Ticket: 1348 (RCA + alternatives, parallel to t1347). Wheel under analysis: the
t1347 continuation build (guest scratch `t1339-wmma/vend`, wheel `d0dabec`,
08:25Z) — the salvage t1339 kernel with the t1347 accumulator-fragment store fix.
All bench numbers below are t1347's canonical `mtp-bench-k3-tape.py` (md5
2c2d6c90), same guest (RX 9070 XT / gfx1201, ROCm 7.2), same session,
per-verify means over 128 verifies, two runs per arm.

## Verdict (one paragraph)

The 48.5 tok/s R1 collapse (vs 74.75 rows-only, vs 69.5 B0) is caused by the
t1339 dispatch gate sending **all of M=1..4** to the WMMA kernel. M=1 shapes
(every correction re-forward, `refwd`) blow up 12.6 -> 31.2 ms/forward because
the kernel's per-block work is M-invariant (a full 16-row 16x16x16 mma per
16-k tile, 256 scalar dequants per 32-k, 32-thread blocks, grid N/16), while
the scalar M=1 path it replaces wins by K-parallel blocks and weight-decode
reuse. M=4 verify (rows-msg 22.25 -> 33.69 ms, +11.4) is a real WMMA-kernel
loss vs the scalar rows-msg kernel **even though the kernel provably lowers to
native `v_wmma_f32_16x16x16_f16` matrix-unit instructions** (disassembly below).
The "M=4 isolated parity" micro data (1.601 vs 1.602 ms) is **INVALID**: the
micro used `nn.QuantizedLinear` defaults (f32 scales), the affine op promotes x
to f32, and the wmma gate declines f32 x — so the "wmma" micro arm ran the
rows/scalar kernel. The only genuine WMMA performance data is the full stack.
Recommended first fix: gate WMMA to M=4 only (rows/scalar for M<=3), which
recovers the refwd regression (~48.5 -> ~63-65 tok/s); it is a stop-the-bleed,
not a win over rows-only. See the ranked alternatives table below.

## 1. Arms and numbers (t1347 canonical bench, reproduced in-session)

| arm | env | tok/s | verify ms | single ms | refwd ms | pair ms |
|---|---|---|---|---|---|---|
| B0 | rows OFF | 70.26 / 69.84 | 25.54 / 25.79 | 9.84 / 9.74 | 9.09 / 9.21 | 44.70 / 44.97 |
| ROWSONLY | rows ON, min_n=1 | 74.75 / 74.77 | 22.25 | 9.70 | 9.32 | 41.49 / 41.48 |
| R1 (FAIL) | rows ON + MLX_QMV_WMMA=1 | 48.55 / 48.52 (48.85) | 33.69 / 33.90 | 9.83 / 9.80 | 23.18 / 23.03 | 66.94 / 66.98 (66.53) |

B0 reproduced 69.48 within +1.1% (gate PASS). R1 vs ROWSONLY deltas:
**verify +11.4 ms, refwd +13.9 ms, single +0.13 ms (unchanged)**. Targets
(verify <= 16 ms, pair <= 31.4 ms): FAIL by 2x. Streams diverge statistically
(D1PCT 75.8 vs 68.8, argmax-identical spot check maxdiff 0.07) — acceptable
under the statistical bar.

Bench accounting (read from the tape script): `verify` = the S=4 batched
forward (M=4, every iteration); `refwd` = the **S=1 correction forward**
(M=1 target-model forward), n_refwd = 95-102 per 128 verifies -> per-forward
12.6 ms (rows) -> 31.2 ms (R1); `single` = the 3 draft MTP-module calls +
3 M=1 lm_head calls (draft 5.54 ms + dlmh 4.22 ms per 3-call set).

## 2. Dispatch / shape routing (vend qmm.hip, verified in place)

Gate order: WMMA gate first, then rows, then mr2/mr/wide/scalar fallback.

- **WMMA gate** (when `MLX_QMV_WMMA` set): bits==4, **M = 1..4**, N%16==0,
  K%16==0, N >= `MLX_QMV_WMMA_MIN_N` (default 0; t1347 added this env for the
  in-flight N-bisect, source-only, not yet built), x NOT f32, gs in {32,64,128}.
  One kernel covers all M — M is a runtime argument, there are no per-M
  variants.
- **rows-msg gate**: M = 2..4 (NOT 1), N >= rows_min_n (1 in the bench),
  K%32==0.
- **M=1 otherwise** -> default scalar qmv kernel.

So in this bench (only M=1 and M=4 shapes occur — verify is always S=4,
draft/correction are S=1):

| shape class | B0 | ROWSONLY | R1 |
|---|---|---|---|
| M=1 (refwd, lm_head) | scalar qmv | scalar qmv | **WMMA** |
| M=4 (verify) | scalar qmv | rows-msg | **WMMA** |

## 3. Kernel structure and disassembly evidence

Kernel (`qmv_wmma_m4_kernel`): one block = one 32-thread wave = one 16-column
n-tile x all M rows. Per 32-wide k-pair: 8 LDS staging + 16 scalar B-decodes
per thread (nibble -> s*w+b -> fp16, 256 per block per k-pair), 2
`load_matrix_sync` (LDS->registers), 2 `mma_sync` 16x16x16, 2
`__syncthreads`; next-iteration global data prefetched to registers. Epilogue:
merge 2 accumulators, f32 -> LDS -> T-convert -> global (256 writes per block,
only M/16 useful).

**Work is M-invariant**: M=1 does the identical 16-row mma + 256-decode per
k-pair as M=4. A-fragment waste = 15/16 rows at M=1 (the mma shape is fixed
16x16x16; there is no partial mma on RDNA4).

Wheel disassembly (wheel `d0dabec` — the one t1347 benched; extracted
`.hip_fatbin` -> embedded GPU ELF): **all 18 template instantiations
(2 x dtype x 3 scale dtype x 3 group size) each contain 32 native
`v_wmma` instructions, 54 of them `v_wmma_f32_16x16x16_f16` per the
bf16/128-group variants** — i.e. every gate-eligible M=1..4 shape executes on
the matrix units. No scalar-FMA emulation of the mma. (t1347's own fatbin
check reached the same conclusion; the vendor tree also already ships
`qmm_wmma_dense_kernel` for M=4/6/8 and `gather_qmv_wmma_prefill_kernel`,
so native WMMA lowering for small-M is a proven property of this toolchain.)

The t1347 fix resolved the t1339 M>=2 garbage: the epilogue store mapping
assumed `row = (L + 32q) >> 4, col = L & 15`; the true gfx1201 16x16x16 f32
accumulator fragment layout (probe-verified) is `row = 8*(L>>4) + e,
col = L & 15` (a lane owns 8 consecutive rows of one column). That wrong
mapping happened to be exact for row 0 (why M=1 passed and M>=2 was garbage
in t1339). In-model correctness of the fixed wheel: maxdiff 0.07 vs rows,
argmax identical.

## 4. Causal chain (pinned to shapes)

1. R1 = rows ON + WMMA ON -> the leading gate reroutes **every** M=1 and M=4
   4-bit QMV (bf16 x/scales, the model layout) to `qmv_wmma_m4_kernel`.
   Verified in the benched wheel (section 3).
2. **refwd +13.9 ms/verify (9.32 -> 23.18)**: the S=1 correction forward is a
   full M=1 target-model forward — ~200+ quantized GEMMs per forward (28 GDN
   layers x in/out proj N=4096, k/v N=1024, GDN small-N projections, mlp
   N=14336, + lm_head N=248320). Per-forward 12.6 -> 31.2 ms (+18.6 ms)
   = ~+80-90 us per mid/small-N M=1 GEMM, consistent with the kernel's
   M=1 cost model: (a) 15/16 wasted mma rows — the matrix units do 16x the
   useful work; (b) 8x scalar-decode density per k-value (256 decodes per
   32-k per block vs the scalar M=1 kernel's ~1 decode per k-value with the
   weight decoded once and reused over K); (c) 32-thread blocks, grid N/16 —
   N=4096 -> only 256 blocks x 32 threads running 128 sequential k-iterations
   with 2 barriers each, vs K-parallel scalar blocks; (d) 16x16 LDS round-trip
   per k-pair. The replaced scalar M=1 path has none of these.
3. **verify +11.4 ms/verify (22.25 -> 33.69)**: M=4 GEMMs move from rows-msg
   to WMMA. The kernel is on the matrix units (section 3) yet still loses to
   the scalar rows-msg kernel, which decodes each 4-bit weight **once and MACs
   against all 4 rows** (one weight pass, 6-wave blocks). At M=4 the WMMA
   kernel pays the same per-block dequant (256/k-pair, independent of M) plus
   the LDS/barrier overhead, and the mma-unit gain does not cover it — the
   exact failure t1334 predicted ("scalar-FMA/load-latency bound" is the
   excess; the WMMA fix must beat rows-msg, not merely the scalar fallback).
4. **single unchanged (9.70 -> 9.83)** is the control that pins the collapse
   to specific shapes, not a global WMMA tax: its GEMMs are the 3 MTP-module
   calls (dense/non-quantized in this model layout — to be confirmed by the
   dispatch log, section 7) and the M=1 lm_head (N=248320, large-N class,
   15520 blocks — parallelism-saturated, where the WMMA kernel is at parity
   with scalar). If every M=1 shape were slow, `single` would have spiked.
5. Cross-check refwd 9.17 -> 23.2 (t1307 baseline -> R1): consistent with (2) —
   all of the delta is the M=1 target-model forwards going scalar -> WMMA.

## 5. The invalid-micro finding (load-bearing for the RCA)

t1347's "M=1..4 parity" micro data (M=4 1.601 vs 1.602 ms; M=1..3; small-N
0.056 vs 0.063 ms) came from `m4micro2/3/4.py`, which built
`nn.QuantizedLinear(K, N, bias=False, bits=4, group_size=32)` — **f32 scales
by default**. The affine op promotes x to the scales dtype before qmm dispatch
(the dispatch comment itself says so: "f32 x declines (the affine op promotes
x to the scales dtype; bf16 scales keep x bf16)"), and the wmma gate requires
`x.dtype() != float32`. Therefore with f32 scales the "wmma" micro arm never
reached the WMMA kernel — the "parity" is rows-vs-rows (or scalar-vs-scalar).
t1347 noticed the f32-**x** version of this flaw but then "fixed" it only at
Python level (bf16 x), which the op re-promotes to f32; the one correct script
(`m4micro5.py`, bf16 scales) failed on the Python API and was never re-run.
Consequences: (a) there is **no valid isolated WMMA-vs-rows number at all** —
the full-stack bench is the only genuine data; (b) the only valid numbers in
that micro session are rows-vs-scalar (M=4 N=248320 K=4096: rows 1.602 ms vs
scalar 4.874 ms — rows-msg is 3x the scalar fallback, as expected).
Note also the `micro1339.py` protocol from t1339 already does it right
(bf16 scales, gs=64, the model layout) and is ready to run.

## 6. Ranked alternatives (statistical bar for all; bit-exact NOT required)

Payoff estimates are on the 41.5 ms ROWSONLY pair / 74.75 tok/s baseline
(current machine state; t1307 reference pair 41.35 ms).

| # | alternative | what / why | expected payoff | how to test | risk |
|---|---|---|---|---|---|
| 1 | **(a) WMMA gated to M=4 only** (rows/scalar for M<=3) — the likely t1347 fix; the **baseline alternative** | One-line gate change (`M == 4`, or `MLX_QMV_WMMA_MIN_M` env). M=1 shapes return to the scalar M=1 kernel that `single` proves is fine. | R1 66.9 -> ~53 ms pair; 48.5 -> **~63-65 tok/s**; verify stays ~33.7. Parity with rows-only, not a win — but recovers the regression, confirms the M=1 attribution, and isolates the M=4 question. | 1-line dispatch patch + rebuild (note: t1347's build 19 failed on `venv-rs/bin/pip` (absent); use the build-18 command `python3 -m pip wheel . --no-deps --no-cache-dir -w wheelout/` in the vend dir) + canonical R1. | Low. Kernel untouched; statistical gate unchanged. |
| 2 | **(b) scalar rows-msg M=4 tuning** (workgroup size / register pressure / x-load layout; un-hoist per t1334) | t1334: the M=1->M=4 excess is scalar-FMA/load-latency bound and the t1253 decode-hoist is a regression; t1330: MLP GEMMs 65% + GDN projection GEMMs 24% of the excess. rows-msg M=4 = 22.25 ms verify vs the 16 ms target (6.25 ms gap). | ~2-5 ms of the verify gap: pair 41.5 -> ~38-39 ms, **~77-80 tok/s**. Probably closes part of the gap; the FMA-bound ceiling may leave the last 1-3 ms. | Kernel patch to `qmv_rows_msg_kernel` (or a new M=4 scalar variant) + unit gate (maxdiff <= 1e-2 vs rows ref) + canonical. | Medium: tuning whack-a-mole; the t1253 hoist-regression lesson. |
| 3 | **split-K + packed-dequant WMMA M=4 variant** (found in the code) | The kernel's measured per-k overheads are (i) 256 scalar dequants per 32-k per block (8x/k density vs rows' 1x), (ii) 2 `__syncthreads` + LDS round-trip per k-pair, (iii) N/16 blocks of 32 threads (underutilization at N <= 14336). Add grid.z K-split (x4) for N < 16384; dequant via `v_pk16` packed-fp16 ops (halves the scalar ALU); drop the redundant 2nd barrier. M=4 stays on the matrix units. | Unknown up front; if the dequant/barrier overhead is the whole 11.4 ms, verify 33.7 -> <22 and combined with #1 the pair could go below rows-only (up to +5-8 tok/s over 74.75). If dequant is not the bottleneck, ~0. | New kernel variant + `micro1339.py` (correct protocol: bf16 scales, gs=64) over the model N-classes + canonical. | Medium: one kernel build cycle; the unit gate + spot check are in place. |
| 4 | **(c) dequant-to-bf16 + rocBLASLt/rocBLAS gemm for M<=4** | **Availability checked on the guest (ROCm 7.2): rocBLASLt IS present and `libmlx.so` already links `librocblas.so.5.6` + `libhipblaslt.so.1.4`; but the rocblas prebuilt-kernel bundle ships ZERO gfx1201 kernels (JIT-only on this target), and rocBLAS small-M quality on RDNA4 is unverified (CDNA-centric history).** Per-call dequant doubles HBM weight traffic (4 -> 8 bit) in the weight-streaming-bound M<=4 regime = a direct loss; the pre-dequant-at-load variant doubles weight memory (~5 GB -> ~10 GB on the 16 GB GPU — fits, barely, with activations). | Unknown; per-call form expected negative; pre-dequant form only competitive if a rocBLASLt M=4 WMMA path is fast — **requires a GPU micro the t1347 lane currently blocks; do not claim a payoff before it.** | rocBLASLt bf16 gemm micro, M=1..4 x the model N-classes (N=4096/14336/28672/248320, K=4096) vs rows 1.6 ms; then a memory-fit check of the pre-dequant layout. | High: JIT first-call latency, unknown kernel quality, memory doubling. |
| 5 | **(d) 4x16 padded-batch / A-fragment layout** | Checked: the kernel is NOT per-row mma — it is ONE 16-row mma per k-tile with zero-padded rows, and its cost is already M-invariant (M=1 costs exactly what M=4 costs). Padding an M=1 call to 4 rows changes nothing; the 15/16-row waste at M<4 is intrinsic to the fixed 16x16x16 wmma shape (no partial mma exists on RDNA4). | 0 as a standalone lever. It is the *reason for* #1 (gate) and points the M=4 work at #2/#3. | n/a — structural fact, verified in kernel source + disassembly. | n/a. |

Ranking rationale: #1 first (cheap, low-risk, confirms the causal attribution
and unblocks the M=4 question); then #3 or #2 based on the per-shape table
from the corrected micro (below); #4 only if its micro shows a competitive
small-M path; #5 is evidence, not a lever.

## 7. Recommended next experiment (for t1347 / the orchestrator)

1. **Gate WMMA to M=4 only** (alt #1) + canonical R1: expected ~63-65 tok/s,
   verify ~33.7, refwd ~9.3 — confirms the M=1 attribution in the full stack.
2. **Run the correct-protocol micro** — t1339's `micro1339.py` (bf16 scales,
   gs=64, the model layout) — over M=1..4 x N in {64/128, 1024, 4096, 14336,
   28672, 248320} (K=4096), rows vs wmma. This produces the first VALID
   per-shape WMMA-vs-rows table in one GPU session and makes the in-flight
   `MLX_QMV_WMMA_MIN_N` source patch (5+ rebuilds) unnecessary. Discard the
   `m4micro2/3/4` numbers as invalid (section 5).
3. **Per-shape dispatch log** on the R1 wheel (the `MLX_T1339_DBG`-style
   printer t1347 removed — re-add temporarily): log M/N/K/gs/x-dtype/route for
   every QMV in one fwd. This (a) confirms exactly which M=4 verify shapes and
   M=1 refwd shapes take the WMMA path, and (b) settles why `single` is
   unchanged (hypothesis: the MTP module is dense/non-quantized in this model
   layout; the M=1 lm_head N=248320 is the large-N parity class).
4. Then pick #2 vs #3 (and optionally #4's micro) from the per-shape table.

## 8. Evidence inventory

- t1347 live session (read-only): B0/ROWSONLY/R1 canonical runs (2x each,
  interleaved re-check), the fwd4 cross-arm correctness (maxdiff 0.07, argmax
  identical), the invalid micro sequence (m4micro2-5), the fragment-layout fix
  discovery, the N-gate source patch + failed build 19.
- t1339 salvage session tail: the Rows>=1 garbage debug, the OOB tail
  bisect (K=1024), the acc-fragment mapping bug (row 0 exact / rows 1+
  garbage), build history (14 first clean, 17 orphaned at cancel).
- Guest (read-only): vend `qmm.hip` dispatch gates (WMMA M=1..4 leading;
  rows M=2..4); vend `qmv_wmma_m4_kernel.hip` (fixed store loop, no debug
  prints); `micro1339.py` (correct protocol); unit reference matrix (6 shapes
  x M=1..4 rows_*.npy + wmma_*.npy); wheelout/ disassembly (18 instantiations
  x 32 native v_wmma each, section 3); rocBLASLt/rocBLAS availability (section 6
  #4); GPU lane state: lock held by t1347 — no GPU micro run by this ticket
  (CPU-side evidence only, per policy).

## 9. Open questions / evidence gaps

- MTP module quantization status (dense vs quantized) — pending the dispatch
  log (#7.3); it determines whether `single` invariance is "dense module" or
  "all its shapes are parity classes".
- No valid isolated WMMA numbers exist yet (section 5) — the per-shape table
  (#7.2) is the gating datum for choosing between alternatives #2 and #3.
- rocBLASLt small-M performance on gfx1201 (JIT-only prebuilt bundle) —
  unmeasured; alt #4 stays unranked-on-payoff until the micro lands.
- Machine drift note: this session's B0 (70.2/69.8) and verify-B0 (25.5) run
  ~3 ms slower than the t1334-era numbers (22.04); all comparisons above are
  within-session, which is the valid protocol.
