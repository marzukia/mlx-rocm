# Stack-assembly bench: B+gate+4bit vs R5+gate+4bit - THE STACK NUMBER (t1902, neon bare metal, 2026-11-04)

## Result (one line)

Arm B (hybrid) WINS the anchor 49.07 vs R5 44.12 t/s (R5 -10.1%); per the decision rule R5 is DROPPED from the stack, and THE STACK NUMBER = **49.07 t/s** (B + gate T0=16 + ngram n=4 + 4-bit MTP module, anchor MTP_MAX_TOKENS=322, median of 3, r4v wheel) - 65.9% of the AR 74.44 class (-34.1%, does NOT meaningfully beat the ~80+ non-MTP AR bar) and 40.9% of the 120+ merge gate on PR #13 (-70.9 t/s).

## 1. Decision rule (stated)

R5+gate+4bit >= B+gate+4bit at the anchor (t/s at acceptance-neutral-or-better) -> R5 ADOPTED in the stack; else R5 DROPPED - the measurement gives R5 44.12 < B 49.07 with acceptance also lower (tok/step 2.200 vs 2.282), so R5 is DROPPED.

## 2. Setup and identity

Worker: fleet-worker. Machine: neon (bare metal), RX 9070 XT / gfx1201, ROCm 10.1 guest userland. Model: <models-mount>/qwen3-9b-mlx (4-bit trunk + bf16 MTP sidecar model-mtp.safetensors md5 02a54359cda8b247f76b277dd9bc4d4c, verified). 4-bit MTP sidecar: model-mtp-4bit.safetensors.disabled on the box, used BY PATH via MTP4_SIDECAR (NOT renamed; 152,130,753 B, md5 e34f85e43b8dcd4ab82367e1a0bee482, size+md5 verified pre/post). Wheel: venv-r4v libmlx.so md5 8acf13fec6fc1636026d8e1dc69cd497 - verified at start and end, not modified, not rebuilt (the parallel kernel ticket's round-2 wheel was abandoned per the 33% same-toolchain A/B regression and was not used).
Code: branch stack-assembly-bench from base r5-correction-fusion @ 4b9285b; merged mtp-draft-gate-probe @ c64096e (the ADOPTED draft gate, T0=16 skip + ngram n=4, the 11-04 op-ruling config-of-record, +15.7% on 512v) and cheap-head-chain @ 06ca81e (the ADOPTED 4-bit MTP module arm: CH_MTP_4BIT switch + MTP4_SIDECAR path + roofline harness, ADOPT t1890). Merge conflicts were shallow (the incoming branches predate the R5 arm): kept the R5 tape_mode/"r5" arm, the CDEFERRED INFO field, and dropped a duplicated mtp-bench6-dg.py E501 per-file-ignore already present - no semantic conflicts. The gate harness (mtp-bench6-dg.py) then needed the R5 tape arm + the 4-bit module block ported in (the battery runs BOTH arms through the gate file for identical method); the port's missing-"r5" tape_avail set was caught by smoke (fallback NameError on the first r5 reject), fixed (2ec9a22), and re-smoked clean.

## 3. Method (identical to the R5 report's anchor protocol)

Standing env both arms: MLX_MTP=4 GDN_FUSE=1 REFWD_M1=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1, HCC_AMDGPU_TARGET=gfx1201, LD_PRELOAD libhsa-runtime64; BOTH arms run gate (MTP6_DRAFTGATE=1, MTP6_DGATE_T0=16, ngram n=4) + 4-bit MTP (CH_MTP_4BIT=1, MTP4_SIDECAR by path); arm B = MTP6_TAPE=hybrid, arm R5 = MTP6_TAPE=r5. Protocols: anchor MTP_MAX_TOKENS=322 + 64v phase MTP_MAX_TOKENS=128 (phase harness = mtp-bench6-phase.py, which has no gate path, same as the R5 report's 64v method; the 64v dg-harness gate+4bit runs are recorded separately as the adopted-config 64v numbers). 3 repeats interleaved B/R5 per protocol, medians, 25 s idle before/between every run, SCLK sampled before/after (and every 10 s during) each run, all under flock on the box gpu.lock (LOCK-ACQ iter=1; sibling run t1910 held the lock in sequence, never killed; lock free at end, no orphans). Pre/post identity: libmlx 8acf13... / sidecar 02a54359... / 4-bit 152,130,753 B all unchanged.
Gate: SCLK 2200 MHz on all 65/65 samples (locked; not changed).

## 4. Anchor (MTP_MAX_TOKENS=322, 110 verifies, gate T0=16 + ngram n=4 + 4-bit MTP) - medians of 3

| arm | t/s (runs) | median | tok/step | D1/D2/D3 pct | REFWDS | CDEFERRED | NSKIP | NNG | tokens |
|---|---|---|---|---|---|---|---|---|---|
| B (hybrid) | 35.87 / 49.18 / 49.07 | **49.07** | 2.282 | 72.7 / 51.2 / 48.8 | 90 | 0 | 102 | 20 | 443 |
| R5 (r5) | 43.55 / 44.14 / 44.12 | **44.12** | 2.200 | 68.2 / 54.7 / 39.0 | 0 | 94 | 102 | 14 | 345 |

R5 at the anchor: -10.1% vs B (4.95 t/s), with the expected structure: REFWDS 90 -> 0 (CDEFERRED 94, zero fallback events), D1 -4.5 pts / D2 +3.5 / D3 -9.8, tok/step 2.282 -> 2.200 (-3.6%) - the same acceptance trade that made R5 a net loss in the no-gate R5 report (-11.9%); the gate (T0=16, NSKIP=102 = ~93% of the 110+102 step mix) + 4-bit module do not reverse it. B's run 1 (35.87) is the session cold-start outlier; runs 2-3 are 49.18/49.07 and R5 is bit-stable across all 3 runs.

## 5. THE STACK NUMBER and the bars

Winner's anchor t/s (THE STACK NUMBER) = **49.07** (B + gate + ngram n=4 + 4-bit MTP, median of 3). vs the AR 74.44 class: 49.07/74.44 = 65.9% (-34.1%, -25.37 t/s) - the bar to MEANINGFULLY BEAT non-MTP AR ~80+ is NOT met (49.07 < 80). vs the 120+ merge gate on PR #13: 49.07/120 = 40.9% (-70.9 t/s). Context: the stack's 4-bit MTP module + gate are both ADOPTED arms in isolation; the number says the adopted stack still sits at ~0.66x of the AR control - the same standing the MTP mission target (memory [214]) flagged as an implementation gap, not a platform gap (clock locked 2200 MHz, wheel md5-verified, SCLK 65/65 samples).

## 6. 64v phase breakdown (MTP_MAX_TOKENS=128, 64 verifies, phase harness, 4-bit MTP, no gate - the phase harness has no gate path, same as the R5 report's 64v) - per-verify-step wall (ms, medians of 3)

| phase | B (hybrid) | R5 (r5) | delta |
|---|---|---|---|
| draft | 37.87 (2423.6/64) | 18.49 (1183.4/64) | -19.38 |
| verify | 0.54 (34.6/64) | 0.54 (34.3/64) | -0.00 |
| commit | 32.23 (2062.8/64) | 41.62 (2663.4/64) | +9.39 |
| correction | 0.39 (24.9/64) | 0.00 (0/64) | -0.39 |
| per-step total | 71.03 | 60.64 | -10.39 |

The 64v per-step mechanism is unchanged from the R5 report: R5 eliminates the correction forward and halves the draft phase (-19.4 ms, the re-anchored chain runs 4-bit module calls instead of 3 + the hybrid correction S=1 trunk re-read) while the commit window absorbs +9.4 ms (the deferred S=4 correction GEMM evaluated in the commit window), net -10.4 ms/step; the anchor loss is the acceptance structure (tok/step -3.6%), identical mechanism.

## 7. 64v with the full adopted config (dg harness, gate T0=16 + ngram n=4 + 4-bit MTP, MTP_MAX_TOKENS=128, 45 verifies) - medians of 3

| arm | t/s (runs) | median | tok/step | D1/D2/D3 pct | REFWDS | CDEFERRED | NSKIP |
|---|---|---|---|---|---|---|---|
| B | 40.01 / 44.41 / 44.28 | **44.28** | 2.156 | 68.9 / 45.2 / 50.0 | 38 | 0 | 38 |
| R5 | 39.76 / 39.68 / 39.67 | **39.68** | 2.156 | 64.4 / 55.2 / 43.8 | 0 | 38 | 38 |

Same verdict at 64v with the gate: B +11.6%.

## 8. Bonus sanity: fused verify per-op on r4v

One batched S=4 GDN-fused verify forward (lm() + forced eval) at the 128-token regime, bf16 sidecar, 7 reps: median 34.43 ms (reps 34.39-34.63 after the 53.82 cold rep; GDN_TAPE_CALLS 3240 = 24 layers x 135 forwards, fused path active). EXPECTED ~45.9 ms (the r4v control of record): 34.43 is 25% BELOW the control. The wheel is exonerated by identity (md5 8acf13... pre/post, not rebuilt) and the 65/65 2200 MHz SCLK samples; the delta is a protocol/regime difference between this isolated-forward probe and the record's measurement (context length / in-loop state), and is flagged for reconciliation with the kernel ticket's 45.9 record - NOT a wheel defect.

## 9. Lossless + stability gates

PASS: FIRST20 bit-identical across all 19 runs (271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279, matching the known baseline prefix) and identical to the smoke runs; B and R5 each bit-stable across their 3 repeats (identical tokens/t/s structure per arm); all 19 runs rc=0; zero fallback (no-tape) events.

## 10. Check lines

Persona-leak grep (the run brief's pattern set: fleet-IP block, guest-identity prefixes, home/model mount paths, junkyard scripts, mrzk.io mail) over the PR diff (main..head, base 60ad8a7), covering the fix commits: 0 hits (two incoming-branch check lines that self-referenced the pattern set and one machine identity in an incoming report were sanitized in commit edb0453 before push; the NOTES-r5.md model-mount path was scrubbed in the same fix round). PASS.
Line-wrap (one-paragraph-per-line) over REPORT-stack-assembly.md: 0 violations (manual check; scripts/line_wrap_check.py absent at this snapshot, same as the R5 report). PASS.
Format (uv run ruff check . && uv run ruff format --check .): All checks passed + 105 files already formatted. PASS. Size gate (scripts/size_ratchet.py --base 4b9285b): OK.
GPU lock protocol: LOCK-ACQ iter=1 at 09:22:13; the runner held the lock for the whole sequence; the sibling run (t1910) held it in sequence before/after and was never killed; lock free at battery end (FUSER_RC=1), the VOP probe then acquired/released cleanly; no orphans left.
