# MTP draft-phase cost probe: dynamic draft gating + n-gram fill (t1811, guest neon, 2026-11-04)

Worker: fleet-worker. Machine: neon (bare metal), RX 9070 XT / gfx1201, ROCm 10.1 guest userland. Model: <models-mount>/qwen3-9b-mlx (4bit, GDN-hybrid 24 GDN + 8 attention, sidecar model-mtp.safetensors md5 02a54359cda8b247f76b277dd9bc4d4c). Wheel: venv-int2w libmlx.so md5 0b8aa95df151a93404b53fee1e96d0de (78ba063, byte-identical to ~/wheelout-int2vgpr3c). Spec reference: llama.cpp @ 29c720e (draft-mtp; the spec'd dynamic gate exists but is disabled at p_min=0.0; the n-gram speculator is a separate path, not active in the t1436 anchor config).

## Result (one line)

The gate cuts draft-phase cost by **+15.7%** (46.61 vs 40.30 t/s, 49% of steps skipped, n-gram fill on the kept steps) and n-gram fill alone adds **+3.1%** (41.55) on the 512-verify protocol, but **the lossless gate FAILS for every arm that changes the acceptance pattern**, with the root cause measured: the hybrid-tape speculative stream of the R4b config of record is not bit-identical to the pure-AR greedy stream (first divergence at generated token 32, reproduced three independent ways), so any acceptance-pattern change (gate skip, n-gram fill) moves the commit trajectory and flips argmaxes; the 2-draft (S=3) reduction is **unavailable** on this wheel (tape kernel TMAX gap, memory fault). Verdict: **NO-GO** as a lossless replacement of the R4b config of record; the performance mechanism is GO subject to the two follow-ups below.

## 1. Setup and identity

- Arm A: `~/t1796/mtp-bench6.py` (md5 3c7d13b56509634aae1708d4a4df7b16, the K=3 R4b config-of-record bench). Arm B: `bench/mtp-bench6-dg.py` (this probe, commit ff86e39, as-run md5 96671a5f176346dab9091069bf80f55c) — an env-gated copy of arm A: default OFF = bit-identical A path (fw flags set, mtp_cache trim formula reduces to 3/2/1 with all-forwards), `MTP6_DRAFTGATE=1` enables the gate, `MTP6_NGRAM=1` enables the n-gram fill (n = MTP6_NGRAM_N, default 4).
- Gate policy: sliding window W=8 verify steps (MTP6_DGATE_W); sum < T0 (MTP6_DGATE_T0) -> skip the draft phase (plain AR forward, state advances, mtp_cache lags by 1); a skip forces the next step to full draft (prev_k0 cooldown, caps the skip rate at ~50%). T0=0 = never skip (n-gram-only config).
- N-gram fill: zero-cost continuation lookup over prompt + confirmed generated tokens (candidates never enter the context), mirroring `common_ngram_map_draft` (llama.cpp ngram-map.cpp:222-400): key = trailing (n-1) confirmed + candidate prefix, draft = most-frequent following token (min_hits=1); tie-break = most-frequent then smallest id (llama uses most-recent; noted deviation). On a hit no draft forward runs; the mtp_cache lags by the skipped rows (stale-attention approximation, the known gap-report #2 class).
- Protocol: 3-run medians (2 for the two control arms), 25 s idle between runs, flock ~/gpu.lock per run, sclk sampled every 2 s under load (median 2200 MHz, locked high - matches the r4b record), `LD_PRELOAD=libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201`, env `MLX_MTP=4 MTP6_TAPE=hybrid MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1`. 512v: MAX_TOKENS=1024 (step-bounded loop: 512 verify steps); 64v: MTP_MAX_TOKENS=64 (32 verify steps). Lossless gate: FIRST20 + ALLTOKS md5 per run pair, FAIL reported with reason.
- No wheel rebuild: arm B is Python-level only (bench change). All runs on the 78ba063 wheel.

## 2. Results (512-verify protocol)

| arm | config | t/s (runs) | median | x A | tokens | tok/step | D1/D2/D3 pct | gate counts | alltoks md5 |
|---|---|---|---|---|---|---|---|---|---|
| A | static K=3 (R4b record) | 40.46 / 40.31 / 40.30 | **40.30** | 1.000x | 1628 | 2.438 | 72.9 / 61.9 / 57.1 | - | 80281d6b |
| B | gate T0=4 + ngram n=4 | 41.57 / 41.55 / 41.53 | **41.55** | 1.031x | 1674 | 2.537 | 77.1 / 64.6 / 53.7 | NSKIP=0, NNG=113 | c8097603 |
| BA | gate T0=8 + ngram | 41.56 / 41.47 | **41.52** | 1.030x | 1674 | 2.537 | = B | NSKIP=0 | c8097603 |
| BN1 | ngram only (T0=0) | 41.52 | **41.52** | 1.030x | 1674 | 2.537 | = B | NNG=113 | c8097603 |
| Bm | gate T0=16 + ngram | 46.61 / 47.02 | **46.61** | **1.157x** | 1419 | 2.363 | 73.8 / 55.5 / 52.5 | NSKIP=336 (49%), NNG=91 | 421ec4a9 |
| Bc | gate T0=100 (degenerate always-skip), ngram off | 46.19 | **46.19** | 1.146x | 1419 | 2.378 | 72.4 / 58.6 / 54.1 | NSKIP=336 | 89307699 |

- FIRST20 identical across all arms (`271 1919 369 264 15352 11 70042 29006 314 1204 ...`, md5 bda3b13c); all arms deterministic across runs (identical alltoks md5 per arm).
- The gate threshold is set by the 8-step acceptance sum: mean ~11.5 (8 x 1.438), observed min >= 8 over 512 steps - so T0 <= 8 **never fires** (B, BA, BN1 are all the ngram-only behavior), and T0=16 fires on ~50% of steps (cooldown-capped). T0=100 is the degenerate always-skip; the prev_k0 cooldown makes it a 50% skip pattern, which is why Bc (46.19) and Bm (46.61) land together: the gate mechanism is worth +14.6-15.7% and ngram fill adds ~0.4 t/s on top of it.
- Per-phase ms (B run): draft 3.0-3.3 ms (3 MTP forwards), verify 0.2-0.3 ms, correction re-forward 0.1-0.2 ms, AR forward 0.2 ms - the skipped unit (draft + verify + re-forward ~ 3.5-3.7 ms) is ~17x the plain AR forward it replaces.

## 3. Results (64-token protocol, 32 verify steps)

| arm | t/s | tokens | tok/step | D1 pct | NNG |
|---|---|---|---|---|---|
| A64 | 43.38 | 106 | 2.500 | 81.2 | - |
| B64 | 42.09 | 102 | 2.375 | 75.0 | 7 |

- B64 = 0.970x A64 (-3.0%): ngram fill **hurts** in the short regime - a cold 4-gram context (7 fills in 32 steps) produces weaker drafts than the MTP module (D1 75.0 vs 81.2). Ngram fill is regime-dependent: +3.1% on 512v, -3.0% on 64v.

## 4. Lossless gate: FAIL (reported with reason)

- A vs B: first divergence at generated token 100 (A `... 4236, 13875, 318, 56194, 717, 594 ...` vs B `... 4236, 594, 33969, 717, 318, 760 ...`); not a position shift - a true stream break (the streams re-synchronize ~8 tokens later, a normal re-merge).
- A vs Bc (pure-AR skip steps, zero draft involvement): first divergence at generated token **32** (A `7752 329 2978 ...` vs Bc `12103 6599 1083 ...`).
- A vs Bm (gate + ngram): first divergence at token 32, same tokens as Bc - the gate (tape-drift) point dominates over the ngram point.
- Root cause, measured by isolation: (a) the 512v loop is step-bounded (1024 steps), so any acceptance-rate change changes the total token count (B 1674 > A 1628 > Bm/Bc 1419) - a pure bookkeeping contribution; (b) the load-bearing one: the **hybrid-tape speculative stream of the R4b config of record is not bit-identical to the pure-AR greedy stream** - Bc (k=0 skips are plain AR forwards, no drafts, no mtp involvement) diverges from A at token 32, and the same token-32 divergence was independently observed in t1796 (the int2w arm's 1.000-tok/step greedy-equivalent stream, REPORT-int2w-step.md lossless-gate section). The tape-commit trajectory (S=4 tape-kernel row select + trims + S=1 correction) accumulates numerical state drift relative to the sequential AR trajectory; any acceptance-pattern change (ngram fill -> different drafts/krow, gate skip -> AR steps interleaved) moves the commit trajectory and flips argmaxes (B at 100, Bm/Bc at 32). Gap-report #2's "greedy verification makes the output lossless either way" does not hold bit-exactly on this wheel; the R4b record's losslessness is within-tape-family, and arm A itself is a non-AR stream.
- Consequence: the specified lossless gate (ALLTOKS match vs A) is structurally unsatisfiable by any arm that changes the acceptance pattern on this wheel; the lever's streams are not the config-of-record stream.

## 5. The 2-draft (S=3) path is unavailable (kernel gap)

- The first arm B run (2-draft verify, T0=4/T1=12) **crashed**: HSA memory fault in `gdn_fused_step_tape_kernel<hip_bfloat16, 4>` (faulting addr 0x7fb9f30e2000, rptr=5883 wptr=6153).
- Root cause (vendor source, 75ca881 vendor tree): `gdn_fused_step_tape.hip:184` unrolls the row loop to the template constant (`for (int t = 0; t < TMAX; ++t)`), and the dispatcher instantiates TMAX only in {1, 2, 4} (:432, :513-534) - an S=3 verify dispatches to the TMAX=4 kernel and reads a 4th row out of bounds.
- The template is already generic in TMAX; a TMAX=3 instantiation is a one-branch dispatcher addition + wheel rebuild (follow-up, not done in this probe). The probe gate was therefore binary (skip / full 3-draft).

## 6. Verdict and follow-ups

- **NO-GO** as a lossless replacement of the R4b config of record (lossless gate FAIL, section 4).
- GO on the performance mechanism: the gate is the dominant lever (+15.7% at 49% firing, +14.6% degenerate) and ngram fill is a small regime-dependent add (+3.1% / -3.0%).
- Follow-up 1 (blocker for adoption): close the tape-drift losslessness gap - a kernel-level state-equivalence audit comparing GDN ssm state + attention KV between the tape-commit trajectory and the sequential AR trajectory at the first skip step (first divergence: token 32); if the tape commit cannot be made bit-exact, the lever's streams must be accepted as a new config-of-record with its own baseline, not as a lossless replacement.
- Follow-up 2: TMAX=3 instantiation in the tape dispatcher + wheel rebuild to make the 2-draft (S=3) reduction available (the probe's S=3 bench branch is disabled with an inline note at the `elif False` site in bench/mtp-bench6-dg.py).
- Note: an ngram-fill mtp_cache underflow hazard (verify-reject trims the nominal k rows while only n_fwd < k rows were pushed) was found by review and fixed in the probe bench (fw-flag accounting, arm A reduces to the original 3/2/1 trims); the same class of hazard applies to any production port of the ngram fill.

## 7. Checks

- Pins verified on-box before runs: wheel md5 0b8aa95d (78ba063), sidecar 02a54359, bench A md5 3c7d13b5, bench B as-run md5 96671a5f; GPU idle (0%) at session start; sclk 2200 MHz locked, sampled every 2 s under load.
- Local gates on the probe commit (ff86e39): py_compile OK, `uv run ruff check .` all checks passed (bench excluded from lint per house convention), `ruff format --check .` 99 files formatted, size gate OK, `pytest -m "not gpu"` 25 passed 1 skipped.
- PR #21 head: NOT resolvable from this repo - the worktree remote is the local bare (/srv/fleetwork/mlx-rocm.git) with no GitHub remote and no pull refs (checked `git ls-remote`); branch push verified directly: refs/heads/mtp-draft-gate-probe = ff86e39.
