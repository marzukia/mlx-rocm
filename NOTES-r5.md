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

[TO FILL: rows-fused M=4 pass evidence, phase timings, 48.93 anchor details]

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
