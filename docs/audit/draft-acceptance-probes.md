# Stream 3 — draft-acceptance probes + stack assembly (digest)

Consolidated narrative digest for the S=3/S=4 draft-acceptance campaign: stack assembly, the TMAX=3 acceptance probe, the ngram fill lever (adopt bar, decline, remediation), the draft roofline, the draft-gate probe, and the honest lossless hybrid fold. The root REPORT-*/NOTES* files of this stream (REPORT-stack-assembly.md, REPORT-acceptance-t3.md, REPORT-ngram-s3.md, NOTES-ngram-s3.md, REPORT-ngram-s3-remed.md, REPORT-draft-roofline.md, REPORT-fold-qmm-hybrid.md) are consolidated into this digest (deleted by the PR); the reports/ originals (REPORT-mtp-draft-gate-probe.md, k2-verify-r2.md, k2-tinvariance-probes.md, mtp-r4b-fused-verify.md, k2-non-gemm-fusion.md) are moved verbatim under `docs/audit/`. Box: neon / guest-a (RX 9070 XT / gfx1201, ROCm 10.1). Anchor: MTP_MAX_TOKENS=322, medians of 3, flock, 25 s idle, sclk 2200 locked.

**Critical reading note (degenerate trajectories).** The S=3 pure-tape protocol (`MTP6_TAPE=1`) locks into a single-token fixed point from position ~3 on every prompt tested: the 1919 attractor on the canonical prompt (`271 + 19x1919`), the 883 attractor on the t1436 anchor prompt. `MTP6_TAPE=hybrid` is non-degenerate on both. The 84.26 / 120.33 / 105.86 t/s numbers are all measured on the degenerate trajectory class and are **not comparable** to the honest lossless hybrid number (69.43).

## 1. Stack assembly — 49.07 t/s, R5 DROPPED

Merging the adopted arms (draft gate T0=16 + ngram n=4 + 4-bit MTP module) onto the base: arm B (hybrid) 49.07 t/s wins the anchor over R5 44.12 (-10.1%, acceptance also lower: tok/step 2.200 vs 2.282; REFWDS 90 -> 0, the same acceptance trade that made R5 a net loss), so **R5 is DROPPED from the stack**. THE STACK NUMBER = 49.07 t/s = 65.9% of the AR 74.44 class (does NOT meaningfully beat the ~80+ non-MTP AR bar) and 40.9% of the 120+ merge gate. The 4-bit MTP module + gate are both ADOPTED arms in isolation; the stack number flags the mission as an implementation gap, not a platform gap.

## 2. TMAX=3 + postnorm head — 84.26 t/s (degenerate class, ADOPTED into PR #22)

Adopted into PR #22: TMAX=3 (S=3 / K=2 spec-decode verify, `MLX_ROCM_GDN_T3=1`)
+ postnorm head input (`MTP6_HEAD_H=postnorm`, +2.1% free, no retraining).
NEW 84.26 t/s (84.26/84.26/84.02; tok/step 2.907; D1 98.8 / D2 93.1 / D3 0) = **113.2% of the AR 74.44 class (bar MET)** and 70.2% of the 120+ gate; 1.71x the OLD 49.28 arm (which reproduces the t1902 49.07 record). NEWSDPA >= NEW (84.32 >= 84.26). Note the trajectory: the S=3 arm locks into `271 + 19x1919` from position 3 (the S=4 arm escapes on the same prompt/box) — the 84.26 number is the degenerate-attractor class.

## 3. Ngram fill on S=3 — +34.2% measured, DECLINED; remediation REJECTED

- **t1986 (measured on the degenerate 1919 trajectory):** A 89.68 vs B 120.33
  t/s = **+34.2%** (tok/step 2.907 -> 2.969, D2 93.1 -> 99.4, D1 unchanged 98.8; FIRST20 bit-identical, 12/12 runs rc=0). DECLINED against the adopt bar (t/s >=+5% AND tok/step >= 3.0 AND FIRST20 identical): tok/step 2.969 < 3.0, and the 3.0 gate is **unreachable on that trajectory by construction** (ceiling 2.9876 at D1 98.8%). The t1959 no-op claim was confirmed: the NgramFill class existed only in the mode-4 gate harness; the candidate `MTP6_NGRAM_S3=1` wires it into the S=3 draft chain (default off = canonical S=3 verbatim; emitted tokens remain target-greedy, lossless by construction).
- **Remediation (re-measure on a non-degenerate trajectory):** lever (a)
  (t1436 anchor "Hello" prompt, config verbatim) FAILED — the S=3 pure-tape protocol locks into the 883 attractor on the anchor prompt (472/480 tokens 883): the collapse is **prompt-independent** (B vs A +27.2% there, a second degenerate datapoint). Minimal knob ablation: the single knob that breaks the loop is `MTP6_TAPE 1 -> hybrid` (P1); head-flavour/quant knobs (P3/P4) do not. On the hybrid ablation trajectory (canonical prompt): control 58.00 t/s vs B 57.22 = **-1.34% net loss** (tok/step 2.193 -> 2.168; D1 114 -> 117 accepts +1.9 pt — the expected position-1 fill effect — but D2 78 -> 71, +7 correction re-forwards). The leg that matters on real trajectories (D1) was unmeasurable on the loop. **REJECTED**: the +34.2% is a degenerate-attractor artifact, not a transferable lever.

## 4. Draft roofline — bandwidth-dominated

The draft step moves 8.06 GiB at 371 GB/s = **67.9% of the 546 GB/s floor** on r4v (406 GB/s = 74.3% on kfix); sum-of-parts 23.74 ms vs wall 23.36 ms (1.6% above) — **bandwidth-dominated, not launch-dominated** (launch/serialization ~= 0 within rep variance). int2 shared head: 3x 4-bit head argmax 1.51 GiB -> 0.94 GiB, saves ~2.2 ms; the old-regime NO-ADOPT verdict is explained by the old kernel floor — retest criterion: chain drops >= 2 ms (or tok/s +4%) at acceptance-neutral on the kfix wheel (stream 2's step-level A/B is that retest: 0% acceptance, NO ADOPT).

## 5. Draft-gate probe — +15.7% mechanism, lossless gate FAILS (NO-GO as lossless replacement)

The gate (T0=16, prev_k0 cooldown; fires on ~50% of steps; 8-step acceptance sum mean ~11.5, observed min >= 8 so T0 <= 8 never fires) cuts draft-phase cost by **+15.7%** (46.61 vs 40.30 t/s on the 512-verify protocol; ngram fill alone +3.1%, 41.55). **Lossless gate FAILS for every arm that changes the acceptance pattern**, root cause measured: the hybrid-tape speculative stream of the R4b config of record is **not bit-identical to the pure-AR greedy stream** (first divergence at generated token 32, reproduced three independent ways — the same token-32 divergence seen in t1796's greedy-equivalent stream). The tape-commit trajectory (S=4 tape-kernel row select + trims + S=1 correction) accumulates numerical state drift; any acceptance-pattern change moves the commit trajectory and flips argmaxes. "Greedy verification makes the output lossless either way" does not hold bit-exactly on this wheel. Verdict: **NO-GO as a lossless replacement; GO on the performance mechanism**, with two follow-ups (kernel-level state-equivalence audit of GDN ssm + attention KV at the first skip step; if not bit-exact, accept the lever's streams as a new config-of-record with its own baseline). The 2-draft (S=3) reduction was unavailable on that wheel (tape kernel TMAX gap, memory fault). (The gate itself — T0=16 skip + ngram n=4 — was later folded into the stack assembly as the op-ruling 11-04 config-of-record, per the stack-assembly merge.)

## 6. K2 verification + T-invariance probes

- **K2 verify r2 (post-r3 fix):** the r3 fix (cd1b6cd) changed the kv_append
  gate to the kernel's exact per-array stride requirements (the T stride is never used — T == 1, only t == 0 is read; the pre-fix gate mis-routed non-contiguous keys to fallback). Full suite: **69 PASS / 0 FAIL** (62 core unit checks + 7 MAJOR-3/4/5 + negative cases; the r1 failing case now PASS, keys mismatched 0/256).
- **K2 T-invariance probes:** the `k_proj` (4-bit QMM `qmv_wide_kernel`) is
  **bit-exact T-invariant** (n=0 over 8 FA layers x 20 random rows); the `mx.fast.rope` flow probe shows **NO LEAK** with the unmodified .so. The 1-ulp T-dependence the #20 audit attributed to k_proj was the **fused rope kernel** (T=1 1D kernel vs T>=2 batched kernel differ by 1 f32 ulp); the `single=false` rope fix + SDPA 1-pass patch are already in the wheel — the leak was rope, and it was already fixed.
- **MTP R4b fused verify:** S=4 verify **ADOPTED** (the R4b fused-verify
  wheel/patch line; the r4b config of record that streams 2 and 3 anchor against).

## 7. Fold: QMM_M3 on the honest lossless hybrid trajectory — 69.43 t/s

The t2011 lever (REPORT-qmm-m3.md) was ADOPTED (+16.9% t/s, -18.1% fused verify wall on the tape=1 trajectory); the t2001 loop-break finding set the config-of-record tape mode to `hybrid`. The fold flips `MTP6_TAPE=1 -> hybrid` and adds `MLX_ROCM_QMM_M3=1` (12-var -> **13-var config**, every other var verbatim) and re-gates the stack on the non-degenerate trajectory:

- **Control gate (C vs the t2001 record, +/-2%):** OUTSIDE the band (canon
  62.01 vs 57.90 expected, +7.10%; hello 73.87 vs 68.67, +7.57%) — a
  systematic cross-wheel artifact (venv-m3k `a0057c22` vs venv-p3 `460c803d`; the per-row vs staged-MR small-N QMM dispatch change sits outside the verify pass; C's fused verify wall 25.81 vs 25.78 = flat). Lossless + bit-stability evidence intact; the same-wheel A/B is the primary evidence.
- **Lossless gate (F): PASS** — FIRST20 verbatim in 12/12 runs; F is
  **ALLTOKS byte-identical to C in every run-pair (6/6 pairs, canon 436 / hello 456 tokens)**; the shared-dequant bit-identity contract holds end-to-end on the hybrid trajectory (D1/D2 identical across arms per prompt).
- **F vs C (same wheel, same session):** canon **69.43 vs 62.01 = +11.97%**
  (GEN wall 7.03 -> 6.28 s); hello 84.34 vs 73.87 = +14.17% (smaller than t2011's +16.9% degenerate number, as expected — the loop's verify is dominated by the repeated re-forward of one token, where the mlp-class lever bites hardest).

**The honest lossless stack number is 69.43 t/s** (canon, 322 anchor, 13-var config, tok/step 2.193; step wall 31.59 ms with fused verify 21.16 ms = 67.0% of the step) = 93.3% of the AR 74.44 class, 57.9% of the 120+ merge gate. Stack number evolution for the record: 49.07 (stack-assembly, degenerate S=3 tape=1 class) -> 84.26 (acceptance-t3, degenerate tape=1 + postnorm) -> 105.86 (t2011 qmm-m3, degenerate tape=1 + QMM_M3) -> **69.43 (fold-qmm-hybrid, HONEST lossless hybrid + QMM_M3)**.

## Verdicts in one line

Stack: 49.07 t/s, R5 DROPPED; TMAX=3 + postnorm: 84.26 (degenerate class, ADOPTED into PR #22); ngram S=3: +34.2% DECLINED (tok/step < 3.0, unreachable on that trajectory), remediation REJECTED (-1.34% on non-degenerate hybrid); roofline: 67.9% of floor, bandwidth-dominated; draft gate: +15.7% mechanism, lossless gate FAILS (tape vs AR token-32 drift) — NO-GO as lossless replacement; K2: 69/0 PASS, k_proj bit-exact T-invariant, the rope leak was already fixed; R4b S=4 verify ADOPTED; **honest lossless stack = 69.43 t/s** (13-var config, 6/6 ALLTOKS-identical).
