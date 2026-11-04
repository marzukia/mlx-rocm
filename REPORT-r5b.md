# R5b report: does the commit-phase absorption (R5 +9.85 ms/step) come from the tape-row view or the deferred eval order?

Branch `r5b-commit-absorb`, base `r5-correction-fusion` @ 4b9285b. Box: neon (RX 9070 XT, gfx1201, ROCm 10.1 userland). Protocol: anchor MTP_MAX_TOKENS=322 + 64v phase, arms **B** (hybrid) / **R5** (control) / **R5b** (fix 1) / **R5e** (fix 2), 3 repeats interleaved, medians, 25 s idle, SCLK 2200 MHz.

## The two candidate fixes (A/B design)

The R5 report's "zero GEMMs" model missed something: the 64v commit phase absorbed
+9.85 ms/step (32.10 to 41.95) while the draft phase dropped 20.5 ms/step and the
net benched loss was -11.9%. Two unmodeled mechanisms were suspect:

- **R5b (the strided-view fix)** — at the commit rebind, materialize the selected tape
  row as a fresh contiguous array (`mx.array(row.reshape((1,)+row.shape))` on the
  GDN state and conv-state rows) instead of rebinding the row-slice view of the S=4
  tape batch. If the downstream fused GDN step / S=4 GEMM window is sensitive to the
  rebinded leaf being a strided view of the (round-reusable) tape buffer, the copy
  removes the cost. Bounded: ~1 MB/layer, sub-ms.
- **R5e (the deferred-eval-order fix)** — `mx.eval(logits, hidden)` right after the
  verify forward, so the S=4 correction GEMM lands in the *verify* window instead of
  being evaled by the first `argmax` in the *commit* window. If the commit-window
  eval-order (GEMM evaled between the rebind and the next step) is the cost, this
  re-attributes (and, if the mechanism is real, should also change wall time).

Both arms share the R5 defer path (no correction re-forward; REFWDS=0, CDEFERRED as
R5). R5 is the control so the fix delta is isolated from session drift. A second
regime pass (kfix venv) was run because the parallel kernel-fix ticket landed its
wheel mid-session.

## Setup

- Session 2026-11-04 06:28-07:00 UTC; GPU lock held for the whole session (lock acquired
  iter=1 06:28:36, ALL-DONE 07:00:14); 48/48 runs rc=0; 25 s idle between runs.
- SCLK: 96/96 samples 2200 MHz (pinned, no drift).
- Regime r4v: libmlx `8acf13fec6fc1636026d8e1dc69cd497` (wheel sha 75ca881c — the R5
  report's wheel, base 62651c7). Regime kfix: libmlx `69d73260670b59dfbc4ca626a0a6cd93`
  (parallel ticket's wheel, appeared mid-session; bonus consistency regime, not the pin).
- Sidecar `02a54359cda8b247f76b277dd9bc4d4c` identical pre/post both passes.
- Medians of 3 interleaved repeats; 64v per-step = cumulative phase window / 64 verifies
  (identical method to the R5 report — cross-check: this session's B 32.38 / R5 42.08
  reproduces the R5 report's 32.10 / 41.95).
- FIRST20 bit-stable across all 48 runs and vs the base (271 1919 369 264 15352 11
  70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279).

## Anchor 322 (medians of 3, r4v)

| arm | t/s | GEN | TOKS | tok/step | REFWDS | CDEFERRED |
|---|---|---|---|---|---|---|
| B (hybrid) | 43.51 | 11.72 | 510 | 2.385 | 126 | 0 |
| R5 | 38.31 | 9.74 | 373 | 2.311 | 0 | 124 |
| R5b | 38.33 | 9.73 | 373 | 2.311 | 0 | 124 |
| R5e | 38.31 | 9.74 | 373 | 2.311 | 0 | 124 |

R5b - R5 = +0.02 t/s (+0.06%, noise). R5e - R5 = 0.00 (-0.05%, noise). Net B to R5:
-11.95% — reproduces the R5 report's -11.9%.

## 64v phase (cumulative ms / 64 = per verify-step; medians of 3, r4v)

| arm | draft | verify | commit | correction | step total | CDEF/TCOM |
|---|---|---|---|---|---|---|
| B | 42.37 | 0.55 | 32.38 | 0.42 | 75.7 | 0/54 |
| R5 | 21.83 | 0.54 | 42.08 | 0.00 | 64.5 | 55/55 |
| R5b | 21.79 | 0.54 | 42.16 | 0.00 | 64.5 | 55/55 |
| R5e | 21.77 | 42.37 | 0.31 | 0.00 | 64.5 | 55/55 |

## 64v phase (kfix regime, for consistency)

| arm | draft | verify | commit | correction | step total |
|---|---|---|---|---|---|
| K-B | 39.45 | 0.55 | 101.05 | 0.40 | 141.4 |
| K-R5 | 20.59 | 0.54 | 109.81 | 0.00 | 130.9 |
| K-R5b | 20.51 | 0.54 | 109.74 | 0.00 | 130.8 |
| K-R5e | 20.60 | 110.09 | 0.30 | 0.00 | 131.0 |

(kfix wheel is ~2x slower per step than r4v — GEN 22.5 s vs 11.7 s at the anchor;
not a defect of this change, noted for the kernel-fix ticket.)

## Verdict

**(1) Commit absorption: NOT fixed — because it is not an artifact of either candidate
mechanism. The absorption reproduces (+9.70 ms/step this session, +9.85 in the R5
session) and both fixes are measured no-ops on it:**

- **Strided-view hypothesis: REFUTED.** R5b (materialized fresh contiguous row) vs R5:
  commit 42.16 vs 42.08 (+0.08, noise); anchors identical to 0.06%; step totals
  identical in both regimes. Corroborated at code level (vendor tree, wheel sha
  75ca881c): the tape is a fresh row-major `[T, Hv, 128, 128]` malloc — the selected
  row is already contiguous, so materializing it is a value-preserving near-no-op and
  the rebinded leaf was never a strided view in the costly sense.
- **Deferred-eval-order hypothesis: REFUTED as a fix.** R5e vs R5: step total
  identical (64.5 r4v, 131.0 kfix); the absorption merely relocates — commit
  42.08 to 0.31, verify 0.54 to 42.37. The commit phase is now pure bookkeeping
  (rebind + trim + argmax on pre-evaluated logits) and the wall time is unchanged.
- **Mechanistic attribution (established by R5e):** the +9.7-9.9 ms/step is the
  deferred S=4 correction GEMM, which R5's commit window must evaluate (the
  `argmax(l0)` sync); B's commit window instead evaluates the cheaper S=1 correction
  GEMM. It is real work, present in R5 and R5b alike, not a view or ordering
  artifact. R5's 64v per-step win over B is real and preserved: 64.5 vs 75.7
  ms/step (and -20.6 ms/step in draft), but the anchor net stays B 43.51 to
  R5 38.31 t/s (-11.95%), dominated by the tok/step drop (2.385 to 2.311, out of
  scope this round) which outweighs the per-step speed gain.

**(2) tok/step: unchanged (out of scope).** R5b and R5e have R5's exact acceptance
counts (r4v anchor D1/D2/D3 = 107/67/37; 64v D1/D2/D3 = 43/20/9) — the row
materialization and the eval-order move change no token.

**Bottom line:** the R5 draft win (-20.6 ms/step 64v, -20.5 ms/step anchor) is real;
the commit absorption is a real deferred-S=4-GEMM cost, not a measurement artifact,
and cannot be removed by fixing the view or the eval order — only by not evaluating
the S=4 batch in the first place (which is the S=1 re-forward R5 removed, or a
cheaper row-logits source). R5e is offered as an accounting option (clean phase
attribution: commit = 0.31 ms bookkeeping) with zero wall-time effect. R5b is a
refutation artifact; no further A/B on the view mechanism is warranted.

## Artifacts

- `bench/mtp-bench6.py`, `bench/mtp-bench6-phase.py` — arms r5b / r5e added (shared R5
  defer path; r5b materializes the selected tape row at rebind; r5e evals the verify
  GEMM in the verify window).
- `bench/r5b-runner.sh` — 4 arms x 2 protocols x 3 repeats interleaved, two regimes
  (r4v pin + kfix, kfix auto-skipped if its libmlx md5 equals r4v's), flock on
  `~/gpu.lock` (wait loop, no kill of foreign holders), SCLK sampler, md5 pre/post.
