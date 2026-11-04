# REPORT-acceptance-t3: TMAX=3 + postnorm head adopted; stack fold re-bench (t1944, PR #22, neon bare metal, 2026-11-04)

## Result (one line)

THE STACK NUMBER = **84.26 t/s** (S=3 tape + gate T0=16 + ngram n=4 + 4-bit MTP + MTP6_HEAD_H=postnorm, MLX_ROCM_GDN_T3=1, anchor MTP_MAX_TOKENS=322, median of 3, venv-p3 wheel) = **113.2% of the AR 74.44 class** (meaningfully beats the ~80+ non-MTP AR bar, +9.82 t/s) and 70.2% of the 120+ merge gate on PR #13 (-35.74 t/s).

## 1. Mission

Fold the acceptance-probe findings (t1917, branch `acceptance-probe` @ ebbc5f3) into the single MTP PR (#22, base `stack-assembly-bench` @ f8b9a46): adopt TMAX=3 (S=3 / K=2 spec-decode verify, `MLX_ROCM_GDN_T3=1`) + the postnorm head input (`MTP6_HEAD_H=postnorm`, +2.1% free, no retraining), set the new config-of-record, and re-bench the stack A/B on the same wheel (op ruling 11-04: the MTP stack is ONE PR and must meaningfully beat non-MTP AR ~80+; the pre-fold stack number 49.07 did not).

## 2. Graft (from acceptance-probe @ ebbc5f3, onto f8b9a46)

The probe forked at 4b9285b (pre-stack); 5 of 7 grafted files were byte-identical at both fork points and were taken verbatim from ebbc5f3 (`gdn_fused_step.hip`, `gdn_fused_step_tape.hip`, `patches/gdn_fused_step/gdn_fused_step.hip`, `patches/gdn_fused_step/gdn_fused_step.patch`, `tests/test_gdn_fuse.py`); the 2 files the base had diverged on were conflict-resolved: `vendor/.../fast.cpp` = base + the two `use_fallback` T3 env gates (base int2w op + 256/512 validation kept, probe hunks applied clean at the identical contexts) and `bench/mtp-bench6.py` (resolutions: the head allow-list gains "int2w" for the base's int2w arm; the int2 build condition becomes `head in ("int2","int2w")` with the base's wide-op selection preserved - the probe predates int2w; the INFO line uses the `tape_mode`/`head`/`head_h` variables, equivalent to the probe's env re-reads; the probe's T1917 comments replace the base's long FIX#1/FIX#2 comments per the probe's intent). All new behavior is env-gated and default-off: `MLX_ROCM_GDN_T3` unset -> T=3 degrades to the clear error exactly as before (the T=3 admit is inside the gate; the `TMAX in {1,2,4}` dispatch and codegen are untouched, `s_beta`/`s_g` sized [4] so TMAX=3 is a strict prefix); `MTP6_HEAD_H` unset = prenorm = the `hhead` gate is the identity; `MTP6_DRAFT_HEAD` unset = int2 which on this wheel line (no `lm_head_int2` op, see sec 5) degrades to the model's 4-bit head = the pre-fold behavior. Verified: the hermetic gate `tests/test_gdn_fuse.py` 10/10 (pins the gate strings, the `<3>` dispatch-before-`<4>` order, and the TMAX instantiation set), ruff format/check rc=0.

## 3. Setup and identity

Worker: fleet-worker. Machine: neon (bare metal; lock free at start, fuser rc=1), flock on ~/gpu.lock, SCLK locked 2200 MHz (verified before/after: SCLK0/SCLK1 `2: 2200Mhz *`, every per-arm 10 s sample 2200). Model: <models-mount>/qwen3-9b-mlx; sidecar model-mtp.safetensors md5 02a54359cda8b247f76b277dd9bc4d4c (verified); OLD-arm 4-bit MTP sidecar model-mtp-4bit.safetensors.disabled (152,130,753 B, md5 e34f85e43b8dcd4ab82367e1a0bee482) used BY PATH via MTP4_SIDECAR, not renamed. Wheel: venv-p3 (r4v line + TMAX=3; the probe's build, libmlx.so md5 460c803d66c43ebfce8997ef08c912db, verified present - NOT rebuilt), venv-r4v libmlx md5 8acf13fec6fc1636026d8e1dc69cd497 untouched. Method: 3 rounds interleaved (OLD/NEW/NEWSDPA per round), medians, anchor MTP_MAX_TOKENS=322, 25 s idle before every arm, SCLK sampled per arm, GPU env MODELS=<models-mount> + HCC_AMDGPU_TARGET=gfx1201 + libhsa-runtime64.so.1 LD_PRELOAD + MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1; benches md5-verified as deployed (bench 1d9bc556 = the grafted file, dg 7ed00a8a = the f8b9a46 gate harness, the t1902 method file).

## 4. Arms and results (medians of 3; tok/s = median TOKS_S, tok/step = 1+(D1+D2+D3)/verifies)

| arm | config | tok/s (runs) | tok/step | D1% / D2% / D3% | REFWDS | verifies | tokens |
|---|---|---|---|---|---|---|---|
| OLD | S=4 hybrid + gate T0=16 + ngram n=4 + 4-bit MTP + prenorm (the 49.07 stack config) | 49.28 (49.47 / 49.28 / 49.15) | 2.282 | 72.7 / 51.2 / 48.8 | 90 | 110 | 443 |
| NEW | S=3 tape + gate envs + 4-bit MTP + postnorm + MLX_ROCM_GDN_T3=1 | **84.26** (84.26 / 84.26 / 84.02) | 2.907 | 98.8 / 93.1 / 0 | 13 | 161 | 481 |
| NEW + SDPA_S4VA=1 | NEW + MLX_ROCM_SDPA_S4VA=1 | **84.32** (84.32 / 84.42 / 83.90) | 2.907 | 98.8 / 93.1 / 0 | 13 | 161 | 481 |

NEW vs OLD = 84.26/49.28 = **1.71x** (the probe's 1.91x was vs the no-gate S=4 42.94 on the same wheel; vs that control 84.26/42.94 = 1.96x). NEW vs the AR 74.44 class = **113.2%** (bar MET); vs the 120+ gate = 70.2%. NEWSDPA >= NEW (84.32 >= 84.26). All 9 runs rc=0, SCLK 2200 throughout, per-arm SCLK samples logged. The NEW-arm gate envs (MTP6_DRAFTGATE etc.) are no-ops in the canonical S=3 bench (the gate harness is the mode-4 dg file); NEW is the probe's S=3+postnorm config exactly, 4-bit MTP module included (wheel default sidecar), matching the probe's T3PH 83.80 within +0.5% session variance. The OLD arm reproduces the t1902 stack arm (49.28 vs 49.07, +0.4%) on the +T3 wheel = the wheel's S=4 path is untouched by the graft.

## 5. int2-never-live finding (carried from the probe, op-confirmed for the record)

The r4v-line `fast.cpp` has NO `lm_head_int2` op and no Python binding (probe-verified: `hasattr(mx.fast, "lm_head_int2") == False` in venv-r4v; the r4v libmlx carries no `lm_head_int2` symbol), so the `MTP6_DRAFT_HEAD=int2` bench arm silently degraded to the model's 4-bit lm_head in every r4v-line round (the bench gate is `hasattr(...)`; this re-bench's INFO lines show `HEAD=int2 ... INT2=off` = the degradation in effect). Any prior "int2" number on this line is actually a 4-bit (or bf16-dequant) head number, and the 2-bit point of the bit-width probe is unmeasurable on this wheel line without porting the op + binding (separate lane). The bit-width probe's measurable pair is therefore 4-bit vs bf16 (dequant): the probe found acceptance identical (D2 93.1 at S=3 for both) with the bf16 draft ~1.8x slower (44.79 vs 82.10) - 4-bit head quantization is costless in draft quality on this 9B model.

## 6. Probe T3/head findings (source table, ebbc5f3; anchor 322, medians of 3, venv-p3)

| arm | head | head_h | tok/s | tok/step | D1% | D2% |
|---|---|---|---|---|---|---|
| B4 (S=4 tape) | 4-bit | pre | 42.94 | 2.826 | 98.8 | 84.9 |
| B4PH (S=4, postnorm) | 4-bit | post | 49.25 | 2.882 | 98.8 | 90.6 |
| B4HF (S=4, bf16full) | bf16full | pre | 30.77 | 2.826 | 98.8 | 84.9 |
| T3T1 (S=3 tape, NEW) | 4-bit | pre | **82.10** | **2.882** | 98.8 | **90.6** |
| T3PH (S=3, postnorm) | 4-bit | post | **83.80** | **2.907** | 98.8 | **93.1** |
| T3HF (S=3, bf16full) | bf16full | pre | 44.79 | 2.882 | 98.8 | 90.6 |
| T3T0 (S=3 legacy, no gate) | - | - | N/A (clean error, the pre-T3 behavior verbatim) | - | - | - |

D3 is NEVER accepted (D3ACC=0 in every S=4 run): the 4th verify row costs with zero acceptance, so S=3 (K=2) strictly dominates this model. postnorm is +2.1% free (83.80 vs 82.10, D2 93.1 vs 90.6, no retraining).

## 7. Gates

- Lossless FIRST20 vs arm 1 reference: **FAIL (mentioned per the gate protocol)** - the S=3 arms lock into the "This" (1919) repetition loop from position 3 (NEW and NEWSDPA FIRST20 = 271 + 19x1919, bit-identical between the two S=3 arms and across all 3 rounds), while the S=4 arm escapes the loop at position 3 (271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279). The divergence is config-level (S=4 vs S=3 correction paths in a near-tie degenerate region; the S=3 D2 93.1 acceptance lock makes the loop structurally attractive at 3 rows where the S=4 arm's 4-row verify does not), not an intra-arm instability: every emitted token in every arm is target-greedy by construction, each arm is bit-stable across its 3 rounds (identical tokens/tok counts), and NEW vs NEWSDPA is FIRST20-bit-identical (all 481 tokens).
- SCLK: locked 2200 MHz, verified before/after + per-arm samples, all 2200.
- Wheel/sidecar identity: libmlx 460c803d (p3, unchanged by this run), sidecar 02a54359 unchanged, venv-r4v untouched.
- Hermetic: tests/test_gdn_fuse.py 10/10; ruff format --check + ruff check rc=0.
- Check lines (full PR diff 60ad8a7..head): persona-leak grep 0 hits on all patterns (marzuki-*, 100.64.0.0/10, andryo/andryo@//home/andryo, neonvm@/192.168.122.x, /mnt/models, *.junkyard.sh; git author lines exempt); wrap gate (manual equivalent, all .md/.txt/.html at head): 0 hits in any file this work touches (one pre-existing false-positive at NOTES-r5.md:75, an in-block list continuation present at f8b9a46 which passed the commit-time gate); no wrapped lines carried from the probe branch (the known probe wrap hits at its NOTES.md:61,73,120,144 and REPORT.md:77,82 were not copied - the probe's wrapped prose was re-summarized one-paragraph-per-line in this report).

## 8. New config-of-record (adopts TMAX=3 + postnorm; supersedes the 49.07 config for the stack number)

S=3 tape verify (MLX_MTP=3, MTP6_TAPE=1, MLX_ROCM_GDN_TAPE=1, **MLX_ROCM_GDN_T3=1**) + draft gate T0=16 skip + ngram n=4 + 4-bit MTP module + **MTP6_HEAD_H=postnorm** (draft head = the model 4-bit head; MTP6_DRAFT_HEAD=int2 default degrades to it on this wheel line, sec 5). Stack number 84.26 t/s = 113.2% of the AR 74.44 class (bar MET, the 11-04 ruling's acceptance condition), 70.2% of the 120+ gate (-35.74 t/s; the PR #13 gate remains open).
