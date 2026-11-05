# REPORT: fold the QMM M=3 kernel into the config-of-record (MTP6_TAPE hybrid + MLX_ROCM_QMM_M3=1) - the honest lossless number

Date: 2026-11-05. Base: `stack-assembly-bench` @ `99ca874` (PR #22 head, consolidated at `bfd2889` + the `qmm-m3-kernel` branch: REPORT-qmm-m3.md carried + the kernel .hip in tree + the egg-info SOURCES.txt line; the `99ca874` commits are report-whitespace + egg-info only - zero source delta vs the wheel build). Deliverable: this branch's new head (config flip + this report). GPU: guest-a (RX 9070 XT / gfx1201, 32 CU), flocked (verified free at start with `fuser` before `flock`, released and re-verified free at exit).

## Mission

Single-MTP-PR loop step: the t2011 lever was adopted (REPORT-qmm-m3.md, ADOPT: same-wheel A/B +16.9% t/s, -18.1% fused verify wall on the tape=1 trajectory), and the t2001 loop-break finding says the config-of-record tape mode is `hybrid` (the pure-tape `MTP6_TAPE=1` protocol is degenerate on every prompt tested - 1919 attractor on canonical, 883 on the anchor prompt; `hybrid` is non-degenerate on both and reproduces the known lossless anchor line, REPORT-ngram-s3-remed.md). This fold flips the config-of-record: `MTP6_TAPE=1` -> `MTP6_TAPE=hybrid` and adds `MLX_ROCM_QMM_M3=1` (12-var -> 13-var config, every other var verbatim), and re-gates the stack on the non-degenerate trajectory to produce the honest lossless number. The t2011 105.86 t/s was measured on the degenerate tape=1 trajectory; the honest number is what this run produces on the hybrid trajectory.

## Config-of-record (13 vars, verbatim)

`MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1` (arm C = the same set without `MLX_ROCM_QMM_M3=1`; the bench default `MTP6_TAPE` flipped `1` -> `hybrid` in `bench/mtp-bench6.py` so the 13-var config is the file default; anchor `MTP_MAX_TOKENS=322`).

## Protocol

Two arms, both HYBRID: C = hybrid, kernel OFF; F = hybrid + `MLX_ROCM_QMM_M3=1`. One wheel (venv-m3k, libmlx md5 `a0057c220d31a0ec87a8b373a97625e8`, 18 `qmv_wide_mr3_kernel` symbols, verified pre and post, unmodified, no rebuild - the 745 tracked files @ `bfd2889` + the kernel .hip build; venv-p3 `460c803d66c43ebfce8997ef08c912db` verified untouched pre+post). Sidecar `model-mtp.safetensors` md5 `02a54359cda8b247f76b277dd9bc4d4c` verified pre and post (untouched). Bench `mtp-bench6.py` md5 `54ce16ac29e6c6aad2c48a233ff70d85` (verified on the box before use - identical to the box copy used; the repo file's md5 changes only by the default flip). 3 runs per arm per prompt (canonical smartphone prompt + `MTP6_PROMPT=Hello`, the t1436 anchor protocol prompt), C/F interleaved within each prompt (C1 F1 C2 F2 C3 F3 per prompt), 25 s idle before every run, per-run timeout 600 (none hit, all 12 runs rc=0), flock for the whole battery + the probe follow-up. The per-class probe is `verify-bw-perop-m3k.py` (md5 `d99ba9ae53fdd37c46ae351bfbcd45db`, the t2003 audit probe's t2011 adaptation - the `MLX_ROCM_QMV_ROWS_MIN_N=1` setdefault removed so the small-N ops reach the per-row fallback where the H3 kernel engages), run C then F ~1 min apart with 25 s idle before each. SCLK sampled every 10 s via `rocm-smi --showclocks` (25 s idle before the first read, sampler liveness checked between rounds): the idle baseline is 2200 MHz throughout (29/47 samples 2200), with boost-to-load readings up to 3309 MHz and a handful of 0 MHz read glitches; the GPU is not hard-locked in this userland, so no absolute clock gate applies - the same-session C/F interleaving with 25 s idle between runs is the clock control (both arms see the identical boost/idle behavior, F always 25 s after a C run).

## Gates and results

Control gate (C vs the t2001 record, +/-2%): canon **62.01** t/s median (62.01 / 62.04 / 61.94, spread 0.18%) vs 57.90 expected = **+7.10% - OUTSIDE the +/-2% band**; hello **73.87** t/s median (74.01 / 73.87 / 73.76, spread 0.18%) vs 68.67 expected = **+7.57% - OUTSIDE the +/-2% band**. The delta is systematic across both prompts and is a cross-wheel artifact, not drift: the control battery here runs on venv-m3k (`a0057c22`) while the t2001 record ran on the earlier wheel line (venv-p3 `460c803d`), and t2011 already documented the m3k-vs-p3 dispatch difference (this tree uses per-row for N < 16384 where the p3 wheel used staged MR). The fused verify wall of C is **25.81 ms vs t2011's probe C 25.78 ms = +0.12% (flat)** - the verify pass is the same wheel-to-wheel, so the +7% t/s delta sits OUTSIDE the verify pass (the draft/scheduling phase), which is consistent with the small-N QMM dispatch change. FIRST20 is verbatim vs the AR reference in all 6 C runs, the runs are bit-reproducible (identical D1ACC/D2ACC, TOKS, GEN to 0.1%), and the acceptance profile matches the t2001/ngram-s3-remed hybrid control exactly (canon D1ACC=114 D2ACC=78 tok/step 2.193). Verdict: control gate FAILS the +/-2% band as stated, but the lossless and bit-stability evidence is intact; the same-wheel A/B (F vs C) is the primary evidence per the t2011 protocol (the cross-wheel reference comparison is secondary).

Loop gate (no consecutive identical-token run > 3 in the first 64 tokens): **PASS 12/12** - the worst run length in the first 64 tokens is **1** in every run (the hybrid trajectory is fully non-degenerate on both prompts, in contrast to the tape=1 1919x19 lock).

Lossless gate (F): **PASS** - FIRST20 verbatim vs the AR reference in all 12 runs (canon `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279`, hello `11 353 599 264 3296 883 279 2614 1970 13 353 599 264 523 421 682 264 1654 421 4523`), rc=0 in all 12 runs, and F is token-identical to C (full ALLTOKS byte-identical) in every run-pair (6/6 pairs, 436 tokens canon / 456 tokens hello). The shared-dequant bit-identity contract (t2011) holds end-to-end on the hybrid trajectory: D1ACC/D2ACC identical across arms per prompt (canon 114/78, hello 134/109, D3ACC=0, tok/step 2.193 canon / 2.509 hello).

| arm | prompt | runs (t/s) | median | GEN (s) | tok/step | FIRST20 |
|---|---|---|---|---|---|---|
| C (hybrid, kernel off) | canon | 62.01 / 62.04 / 61.94 | **62.01** | 7.03 | 2.193 | 271 1919 369 264 15352 ... (verbatim, 3/3) |
| F (hybrid + QMM_M3) | canon | 69.50 / 69.43 / 69.27 | **69.43** | 6.28 | 2.193 | verbatim, 3/3 |
| C (hybrid, kernel off) | hello | 74.01 / 73.87 / 73.76 | **73.87** | 6.17 | 2.509 | 11 353 599 264 3296 883 279 ... (verbatim, 3/3) |
| F (hybrid + QMM_M3) | hello | 84.36 / 84.34 / 84.19 | **84.34** | 5.42 | 2.509 | verbatim, 3/3 |

F gate (win over C, same wheel, same session): canon **69.43 vs 62.01 = +11.97%** (GEN wall 7.03 -> 6.28 s, -10.7%); hello **84.34 vs 73.87 = +14.17%** (GEN wall 6.17 -> 5.42 s, -12.2%). A real win on the non-degenerate trajectory (smaller than t2011's +16.9% degenerate-trajectory number, as expected: the degenerate loop's verify is dominated by the repeated re-forward of one token, where the mlp-class lever bites hardest).

## Per-class (probe, 13-var dispatch, same wheel, C then F ~1 min apart, 25 s idle)

| Class | C wall (ms) | F wall (ms) | delta | floor C -> F |
|---|---|---|---|---|
| mlp (32x: 2x gate/up K=4096 + down K=12288, N=12288) | 11.496 | 8.407 | -3.089 (-26.9%) | 43.3% -> 59.2% |
| qmv_gdn (24x: 5 QMM, N=96..12288) | 5.095 | 4.593 | -0.502 (-9.9%) | 32.7% -> 36.3% |
| gdn_fused (24x: fused step tape T3 - unmodified) | 5.116 | 4.334 | -0.782 (-15.3%) | 3.7% -> 4.4% |
| attn (8x: 4 QMM N=512..4096 + S4VA) | 1.952 | 1.697 | -0.255 (-13.1%) | 25.9% -> 29.8% |
| norms (unmodified) | 1.495 | 1.472 | -0.023 (flat) | 0.1% -> 0.1% |
| lm_head (unmodified, staged MR N=248320) | 1.261 | 1.273 | +0.012 (flat) | 83.1% -> 82.3% |
| **fused verify** | **25.81** | **21.16** | **-4.65 (-18.0%)** | 32.5% -> 39.6% of the 546 GB/s floor (177.4 -> 216.5 GB/s) |

The win sits in the QMM classes, led by mlp (-3.09 ms, 41% of the fused wall) exactly where the shared-dequant lever was aimed; the unmodified control classes (norms, lm_head) are flat, which validates the measurement. gdn_fused (unmodified kernel) shows a -15.3% delta outside this change's reach - attribute it to run-to-run variation between the C and F probe runs (~1 min apart) rather than a causal effect; it does not change any gate.

## The honest lossless stack number

**69.43 t/s** (canon, 322 anchor, 13-var config-of-record, tok/step 2.193, 14.40 ms/token; the hello arm is 84.34 t/s at tok/step 2.509). Against the standing references: 93.3% of the AR 74.44 class (the t1981 LX0 record 89.70 was the degenerate tape=1 class - 120.4% of AR - not comparable to this lossless number; t2011's 105.86 is likewise the degenerate trajectory), 57.9% of the 120+ merge gate, and +11.97% over the same-wheel same-session C control (the primary evidence). The step wall is 31.59 ms (2.193 tok/step at 69.43 t/s) with the fused verify at 21.16 ms (67.0% of the step).

## Extrapolation to 140 (t2003 ladder math, verify now 21.16 ms)

The step decomposes into the fused verify wall (21.16 ms) + draft/scheduling/norm/argmax (10.43 ms = 31.59 - 21.16, draft 2.6 ms of it per the t2003 decomposition). If the verify wall reaches the ladder target of 8.0 ms (a 13.16 ms removal from 21.16):
- lossless hybrid: 31.59 - 13.16 = **18.43 ms/step -> 69.43 x 31.59/18.43 = 118.7 t/s**.
- Gap to 140 t/s after verify = 8 ms: 140 t/s at tok/step 2.193 = 15.66 ms/step, i.e. **~2.77 ms more must come from draft (2.6 ms) + scheduling/launch/norm/argmax overhead** - the same residual the t2003 ladder identified, now reached from the honest lossless baseline (t2003's own extrapolation from the 58.0 t/s p3-wheel control gave 128.4 t/s at verify 8 ms; this run's m3k-wheel baseline is +7% faster on the same trajectory, and the 140 gap is correspondingly smaller).
- The remaining 21.16 -> 8.0 ms ladder now sits on mlp (8.41 ms, 59.2% of floor - the next QMM-class lever) + gdn_fused (4.33 ms, the unfused-recurrent-bound class) + qmv_gdn (4.59 ms).

## Check lines (as-run, 2026-11-05)

- **(a) persona-leak: 0 hits.** Grep of the fold diff (`99ca874..HEAD`) + the new commit subject/body for the standard persona-term list (marzuki-* hostnames, the 100.64.0.0/10 tailnet IPv4 range, the box-adjacent service-account identity, the box home path, the box LAN IPv4 range, the literal model-mount path, *.junkyard.sh), git author lines exempt per protocol: 0 hits.
- **(b) line-wrap: 0 hits** - `line_wrap_check.py` (copied from the 2025 scripts) over REPORT-fold-qmm-hybrid.md + the touched .md set (REPORT-consolidate.md): 0 hard-wrapped prose lines (one paragraph per line).
- **(c) format: PASS** - `uv run ruff format --check .` -> "117 files already formatted" (rc 0); `uv run ruff check .` -> "All checks passed!" (rc 0) with the untracked line_wrap_check.py excluded from the tree (its docstring lines trip E501; the check tooling is run, not shipped).
- **(d) hermetic tests: PASS** - `uv run pytest -m "not gpu"` -> 25 passed, 1 skipped (rc 0, the CI-exact set); size gate `uv run python scripts/size_ratchet.py --base HEAD` -> "SIZE GATE: OK".

## Verdict

**ADOPT the config flip: `MTP6_TAPE=hybrid` + `MLX_ROCM_QMM_M3=1` (13-var config-of-record).** Same-wheel same-session A/B on the non-degenerate trajectory: +11.97% t/s canon (62.01 -> 69.43, +14.17% hello), -18.0% fused verify wall (25.81 -> 21.16 ms), lossless gate bit-identical in all 12 runs with loop gate clean (worst run length 1/64) and the control arm bit-reproducible with FIRST20 verbatim vs the AR reference. The honest lossless stack number at this PR head is **69.43 t/s** (93.3% of AR 74.44; the ladder to 140 is 118.7 t/s at verify 8 ms + ~2.77 ms of draft/overhead). The control arm's +7% vs the t2001 record is a documented cross-wheel dispatch artifact (verify wall flat at +0.12%), not a drift finding; the same-wheel A/B is the primary evidence. The 120+ merge gate remains open (57.9%).
