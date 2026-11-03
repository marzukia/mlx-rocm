# NOTES-r5 — R5 design probe working notes

Purpose: externalize all analysis for REPORT-r5-design.md. One section per file
read; the report is assembled from this file at the end.

Base: branch mtp-r4b-fused-verify @ 09ddc98 (PR #14-era, 48.93 t/s stack).
Work branch: r5-correction-fusion.
GPU lock check at run start: fuser exit 1 (no holder) = lock FREE.

## 1. Measured situation (source: REPORT-mtp-traffic-audit.md, t1582/PR #14, commit 4cd7083 on branch mtp-traffic-audit)

Anchor mix: 161 verifies, 126 corrections = 78.3% of steps rejected.
Per-phase bytes (measured tensor bytes, median of 3):

| phase | target reads/step | target GiB/step | other GiB/step | total |
|---|---|---|---|---|
| verify (S=4, rows-fused M=4, one weight pass) | 1.000 | 4.16 | 0 | 4.16 |
| commit (tape rebind + KV trim, no forward) | 0 | 0 | 0 | 0.00 |
| correction (S=1 per rejected step) | 0.783 | 3.26 | 0 | 3.26 |
| draft (3x sidecar + 3x 4-bit head) | 0 | 0 | 1.97 | 1.97 |
| TOTAL | 1.783 | 7.42 | 1.97 | 9.39 |

- Target per forward = 4.16 GiB (trunk 3.63 GiB + 4-bit lm_head 0.533 GiB).
- Sidecar mtp module = 0.125 GiB; int2 draft head 0.296 GiB ABSENT in R4b wheel.
- KV reads ~0.07 GiB/step, negligible.
- Total 9.39 GiB/step <= ~11 GiB anchor budget => NOT weight-traffic-bound;
  remaining gap to 144.3 t/s is compute/schedule-side.
- R5 removes the correction read: 9.39 -> 6.13 GiB/step.

Per-phase wall (phase harness 64v, median of 3) — KEY for schedule analysis:
- draft 39.3 ms/step (also FORCES the lazily-evaluated correction forward)
- verify 0.5 ms (lazy graph build; S=4 GEMM forced by first argmax)
- commit 26.7 ms (S=4 GEMM + tape rebind)
- correction 0.4 ms (bookkeeping; GEMM billed to next draft)
- Anchor: 65.2 ms/step at 48.93 t/s (10.49 s / 161 steps).
- CRITICAL OBSERVATION: the correction forward is ALREADY LAZY — it executes
  inside the NEXT step's draft phase (the "draft 39.3 ms" includes it). So
  "one S=1 target re-read per rejected step" runs temporally in the draft
  window of the following step. Any fusion design must slot into that window
  or the verify window, not the (near-free) correction bookkeeping slot.
- Dispatch counters: GDN_TAPE_CALLS = 24 x (verifies + corrections) => the
  fused/tape GDN op path runs in BOTH verify and correction forwards.
- FIRST20 / accepts bit-stable across runs (lossless gate anchor).

## 2. R4b report (reports/mtp-r4b-fused-verify.md)

Facts that matter for the design:
- Anchor 48.93 t/s = +6.2% over OFF (46.07); lever = MLX_ROCM_QMV_ROWS=1 (env gate;
  the wheel change is belt-and-braces for bias-free calls).
- rows-fused M=4 = ONE weight read shared across the 4 verify rows; output vs
  per-row maxdiff 1.2e-4 (bf16, NOT bit-identical); acceptance drift is the
  expected FP-order effect (D3 -2.7 pts, D2 +3.4, tok/step 2.385 -> 2.410).
- FIRST20 identical across arms/runs: 271 1919 369 264 15352 ... (lossless anchor).
- mx.eval is a side effect: the S=4 verify GEMM is forced by the first argmax in
  the commit section and billed to commit. verify timer = lazy graph build only.
- Phase 64v ON totals (ms, /64 => per-step): draft 2515.2 (39.3), verify 35.3
  (0.55), commit 1709.2 (26.7), correction 25.5 (0.4). 512v: draft 14789.8
  (28.9), verify 278.1 (0.54), commit 25189.8 (49.2), correction 187.3 (0.37).
- Standing env: MTP6_TAPE=hybrid GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1. Wheels: venv-canon (md5 b1de59fd...) vs R4b wheel
  (md5 3d19e4f3...).
- Implied per-step schedule (64v): 39.3 + 0.55 + 26.7 + 0.4 = 66.9 ms/step, matching
  the audit's ~66.9. The S=4 GEMM weight read (4.16 GiB) happens inside the
  26.7 ms commit window; the correction S=1 read (3.26 GiB) happens inside the
  39.3 ms draft window of the NEXT step (audit: "GEMM billed to next draft").

## 3. Code map (files/functions)

[TO FILL: qwen3.py MTP path, FUSE/MTP_INJECT patches, gdn_fused_step_tape,
QMV rows-fused kernel — with file/function names and state-staging order]

## 4. Option (a) PIGGYBACK analysis

[TO FILL]

## 5. Option (b) ROLLBACK-REDRAFT analysis

[TO FILL]

## 6. Option (c) CONFIDENCE SKIP analysis

[TO FILL]

## 7. Option (d) / other findings

[TO FILL]

## 8. Recommendation + hook points

[TO FILL]

## 9. Measurement plan

[TO FILL]
