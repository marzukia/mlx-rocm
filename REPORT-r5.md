# REPORT-r5 — R5 correction-fusion measurement (fold the MTP correction token into the next verify S=4)

Branch r5-correction-fusion, measured 2026-11-04 05:15-05:22 UTC on neon (bare metal, RX 9070 XT, gfx1201).

## TL;DR

- R5 eliminates the S=1 correction re-forward exactly as designed: anchor REFWDS 126 -> 0, GDN_TAPE_CALLS 24x287 -> 24x161 (verify-only), CDEFERRED 124 (anchor) / 55 (64v phase), zero fallback (no-tape) events.
- Same-session anchor (MTP_MAX_TOKENS=322, 161 verifies, medians of 3): B (MTP6_TAPE=hybrid) 43.85 t/s -> R5 38.61 t/s = -11.9%. Per-verify-step wall 72.2 -> 60.0 ms (-12.2 ms), but structural acceptance movement tok/step 2.385 -> 2.311 (-3.1%) outweighs the per-step gain. R5 is a net LOSS at the anchor in this measurement.
- Notes sec.8 estimate (65.2 -> ~45-50 ms/step; 48.93 -> 60-65 t/s) NOT MET: the draft phase dropped as predicted (-20.5 ms, predicted 22-26 ms target), but the commit phase absorbed +9.8 ms (32.1 -> 41.9 ms/step at 64v) that the estimate did not anticipate, and tok/step fell 3.1%.
- Lossless gate: PASS (FIRST20 bit-identical across all 12 runs and identical to the 390221f baseline; B acceptance bit-stable across its 3 runs). R5 acceptance drift (D1 -6.8 pts anchor / -1.6 pts at 64v) is the expected R4b-class structure effect and is recorded, not a gate failure.
- No wheel or model change: venv-r4v (libmlx md5 8acf13fec6fc1636026d8e1dc69cd497, mlx 0.32.3.dev20261104+75ca881) + SBCAR model md5 02a54359cda8b247f76b277dd9bc4d4c, verified pre and post; SCLK 2200 MHz in all 18 samples (clock-locked box).

## Setup

- Box: neon bare metal, GPU lock free at run start (fuser rc=1); the runner queued on the lock and acquired it at 05:15:12 UTC (iter=1) after a prior ticket released it; 25 s idle before the first run and between every run; SCLK sampled every 10 s during each run.
- Branch state: the task named base 62651c7; the branch head at start was 0a87dc3 = 62651c7 + one prose-only reflow of NOTES-r5.md (verified: 1 file, .md only, no code delta), so the measured code is exactly 62651c7.
- Arms (3 repeats each, interleaved B/R5 within each protocol): B = MTP6_TAPE=hybrid (the 48.93-era baseline config); R5 = MTP6_TAPE=r5 (the implemented fusion, bench-loop change only, NOTES-r5.md sec.8).
- Standing env both arms: GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1, MLX_MTP=4; protocols: anchor MTP_MAX_TOKENS=322 (161-verify regime, mtp-bench6.py) + phase 64v MTP_MAX_TOKENS=128 (mtp-bench6-phase.py, per-phase wall breakdown).
- Provenance: libmlx md5 8acf13fec6fc1636026d8e1dc69cd497 (pre and post), SBCAR model md5 02a54359cda8b247f76b277dd9bc4d4c pre and post (unchanged); SCLK 2200 MHz (locked) in all 18 samples, no drift across the 12 runs.
- Bench files were copied to the box fresh from the branch (mtp-bench6.py, mtp-bench6-phase.py) and run unmodified.

## Lossless gate (PASS)

- FIRST20, all 12 runs (B and R5, anchor and phase): 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 - bit-identical across runs and matching the 390221f baseline prefix. PASS.
- B acceptance bit-stable across its 3 anchor runs (D1ACC/D2ACC/D3ACC = 118/70/35, tok/step 2.385) and 3 phase runs (44/27/10, tok/step 2.27).
- R5 acceptance drift (expected structure effect, per the R4b precedent; the deferred correction re-anchors the draft chain, changing draft tokens): anchor D1 73.3 -> 66.5 (-6.8 pts), D2 59.3 -> 62.6 (+3.3), D3 50.0 -> 55.2 (+5.2), tok/step 2.385 -> 2.311 (-0.074, -3.1%); 64v: D1 68.8 -> 67.2 (-1.6), D2 42.2 -> 31.2 (-11.0), D3 15.6 -> 14.1 (-1.5), tok/step 2.27 -> 2.12 (-6.6%).

## Anchor (MTP_MAX_TOKENS=322, 161 verifies) - medians of 3

| metric | B rep1/2/3 | B median | R5 rep1/2/3 | R5 median | R5/B |
|---|---|---|---|---|---|
| TOKS_S | 44.07 / 43.85 / 43.74 | 43.85 | 38.78 / 38.61 / 38.56 | 38.61 | -11.9% |
| GEN s | 11.57 / 11.63 / 11.66 | 11.63 | 9.62 / 9.66 / 9.67 | 9.66 | -16.9% |
| ms/verify | 72.23 | 72.23 | 60.00 | 60.00 | -12.2 |
| TOKS | 510 x3 | 510 | 373 x3 | 373 | -26.9% |
| D1/D2/D3 pct | 73.3 / 59.3 / 50.0 | 73.3 / 59.3 / 50.0 | 66.5 / 62.6 / 55.2 | 66.5 / 62.6 / 55.2 | -6.8 / +3.3 / +5.2 pts |
| tok/step | 2.385 x3 | 2.385 | 2.311 x3 | 2.311 | -3.1% |
| REFWDS | 126 x3 | 126 | 0 x3 | 0 | eliminated |
| CDEFERRED | 0 x3 | 0 | 124 x3 | 124 | - |
| GDN_TAPE_CALLS | 6888 (24 x 287) x3 | 6888 | 3864 (24 x 161) x3 | 3864 | verify-only |

Context: the session B median 43.85 t/s sits below the task's 48.93 anchor figure - the absolute gate is regime/wheel-dependent (SCLK locked 2200 MHz here; the 48.93 band was a different regime), so the valid gate is the same-session B-vs-R5 relative, per the standing clock-regime rule.

## Phase 64v (MTP_MAX_TOKENS=128, 64 verifies) - per-phase wall, medians of 3

| phase (total ms / per-verify ms) | B | R5 | delta |
|---|---|---|---|
| prefill | 5.2 / 0.08 | 5.2 / 0.08 | 0 |
| draft | 2705.1 / 42.27 | 1391.8 / 21.75 | -20.52 (the S=1 correction read removed) |
| verify (lazy build) | 35.3 / 0.55 | 34.6 / 0.54 | -0.01 |
| commit (S=4 GEMM forced by argmax) | 2054.4 / 32.10 | 2684.9 / 41.95 | +9.85 (absorbed) |
| correction | 26.9 / 0.42 | 0.0 / 0.0 | -0.42 (REFFWD eliminated) |
| TOTAL | 4827.2 / 75.43 | 4115.3 / 64.30 | -11.13 (-14.8%) |

Per-run detail (total ms / per-verify ms): B draft 2705.1 / 2702.7 / 2712.9; R5 draft 1391.7 / 1391.8 / 1393.3; B commit 2054.4 / 2057.1 / 2054.8; R5 commit 2683.8 / 2684.9 / 2685.6; B correction 26.9 / 27.6 / 26.8; R5 correction 0.0 x3. tok/step: B 2.27 x3, R5 2.12 x3. TAPE_COMMITS 54 (B) / 55 (R5); GDN_TAPE_CALLS 2832 = 24 x 118 (B) / 1536 = 24 x 64 (R5, verify-only).

## Expected vs actual (NOTES-r5.md sec.8)

| item | estimate | actual | verdict |
|---|---|---|---|
| draft phase | 39.3 -> ~22-26 ms/step | 42.27 -> 21.75 ms/step (current-regime baseline) | shape MET (-20.5 ms, the 3.26 GiB correction read gone) |
| per-step total | 65.2 -> ~45-50 ms/step | 75.43 -> 64.30 ms/step | NOT MET (commit absorbed +9.85 ms) |
| anchor t/s | 48.93 -> 60-65 | 43.85 -> 38.61 same-session (-11.9%) | NOT MET |

The estimate assumed (a) the full draft saving lands and (b) acceptance is approximately preserved. (a) half-held: the draft saving is real but ~10 ms of it is re-absorbed in commit; (b) failed: tok/step dropped 3.1% at the anchor, which converts the per-step win into a net loss.

## Mechanism notes (measurement observation, no redesign attempted)

- The S=1 correction re-forward is structurally gone: REFWDS 0, CDEFERRED = rejects (124 anchor / 55 phase), GDN_TAPE_CALLS exactly 24 x verifies (no correction forwards touch the tape op), draft -20.5 ms.
- The unmodeled cost is in the commit window (+9.85 ms/step at 64v; the S=4 GEMM forced by argmax). Candidate mechanism (unverified in this measurement-only run, flagged for the next probe): in R5 the next step's GDN state input is the tape rebind - a row slice of the per-row state-tape batch (a strided view of the S=4 byproduct) - whereas in hybrid the state input is the contiguous post-S=1 correction-forward state; the fused GDN step / S=4 GEMM window on the strided state view may be slower. Alternative: the re-anchored draft chain (h_r from the previous S=4 graph) changes the deferred-evaluation order such that part of the next step's work is billed earlier.
- The D1 acceptance drop (-6.8 pts at the anchor) is the structural draft re-anchoring effect: the re-anchored first draft uses the rejected row's pre-norm h directly instead of the correction forward's h; D2/D3 rise, net tok/step -3.1%.

## Check lines

- Persona-leak grep (the run brief's pattern set: fleet-IP block, guest-identity prefixes, home/model mount paths, junkyard scripts, mrzk.io mail) over the branch diff vs 62651c7 and the new commits (git author lines exempt): 0 hits in this run's commits; the only raw-diff matches are two pre-existing NOTES-r5.md reflow lines (committed earlier at 0a87dc3, not added by this run). PASS.
- Line-wrap (scripts/line_wrap_check.py, one-paragraph-per-line) over REPORT-r5.md: 0 violations. PASS.
- Format (uv run ruff format --check .): clean. PASS.
- GPU lock protocol: fuser rc=1 at run start; runner held the lock for the whole sequence; no orphaned locks left (lock free at end).
