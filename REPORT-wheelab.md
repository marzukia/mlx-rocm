# REPORT-wheelab — wheel-vs-guest A/B (the 17.4 ms question), t1432

**Task:** does the e6d3d182-era (11-02 01:52–04:10Z, the 69.48/75.00 band) wheel reproduce the ~17.4 ms rows-verify on TODAY'S guest? YES → the wheel change explains the loss; NO → guest/hardware state explains it.

**Headline: NO.** The e6d3d182-era wheel, rebuilt from its reconstructed source state, runs on today's guest at verify **29.9 ms** (rows ON) and **65.83 tok/s** — statistically the same as the t1426 battery wheel (30.58 ms, 64.95 tok/s) and ~12% below the 75.00 anchor. The wheels are performance-equivalent on this guest; the 11-01 → 11-02 loss is NOT the wheel. The old arm is a **REBUILD, not the original** (the e6d3d182 .so survives in no file on the guest; details in §5). Verdict per the brief: **next = hardware attention** (guest/hardware state; the DPM-sclk floor was already exonerated by the clock-locked t1431 round).

## 1. Setup (verified before the battery)

- **GPU lane:** lock free at dispatch (first poll, 19:57:41Z); held across D2D + battery; released and verified free at exit; no orphan GPU or sampler processes.
- **Clock lock:** `power_dpm_force_performance_level = high` at pre-check, pre-battery, and post-battery; sclk sampled at 2200 MHz (the max level) in 44/44 bracketing 2 s-cadence samples (§6).
- **Bench:** canonical `mtp-bench-k3-tape.py` md5 `2c2d6c90c2349ae087e53e5d6c60018e` (all six on-guest copies md5-identical) + guard `gdn_tape_commit.py` md5 `4a262a0d9239eea7326606cdfb177c60`, verbatim, env toggles only.
- **NEW arm:** the t1426 battery wheel, libmlx.so md5 `93810e871f7bc977e723d5288a3b73e2`, in the `venv-ra` venv under the graft build tree (unchanged from t1431; md5 re-verified in the battery log).
- **OLD arm:** rebuilt wheel, installed .so md5 `3e6f9d5e19114d2bbf7546625ab68b37`, in a fresh scratch venv (uv CPython 3.14.8, same layout as the other venvs: wheel + mlx_lm 0.32.2). §5.
- **Env:** same as the t1431 battery — models dir, hsa-runtime LD_PRELOAD, gfx1201 target; base `MLX_MTP=4 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1`; R0 nothing extra; R2 `+MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`.
- **Sidecar pin:** the MTP sidecar shard md5 `02a54359cda8b247f76b277dd9bc4d4c` active, the 4-bit sidecar `.disabled`, other shards parked — attested before and after the battery (three identical md5 readings; §7). The pin was not touched.
- Note: the brief's `pp_dpm_sclk` under the hwmon dir is stale on today's guest (the 12:31Z amdgpu rmmod renumbered sysfs); the active-level sclk was sampled from the device-level `pp_dpm_sclk` (format verified: `2: 2200Mhz *`).

## 2. Per-arm table (2 runs/arm, medians)

| arm | run 1 tok/s | run 2 tok/s | **median tok/s** | single ms (med) | verify ms (med) | refwd ms (med) | n_refwd | D-ACC | TAPE n_ok | TAPE_CALLS |
|---|---|---|---|---|---|---|---|---|---|---|
| OLD R0 | 60.84 | 60.43 | **60.635** | 9.745 (9.75/9.74) | 33.33 (33.15/33.51) | 9.27 (9.27/9.27) | 101 | 93/59/27 | 128/128 | 5496 |
| NEW R0 | 61.02 | 60.53 | **60.775** | 9.725 (9.73/9.72) | 33.315 (33.09/33.54) | 9.185 (9.19/9.18) | 101 | 93/59/27 | 128/128 | 5496 |
| OLD R2 (rows ON) | 66.00 | 65.66 | **65.83** | 9.72 | 29.925 (29.84/30.01) | 8.98 (8.98/8.98) | 99 | 95/61/29 | 128/128 | 5448 |
| NEW R2 (rows ON) | 65.77 | 64.12 | **64.945** | 9.725 (9.72/9.73) | 30.58 (29.92/31.24) | 9.035 (9.07/9.00) | 99 | 95/61/29 | 128/128 | 5448 |

In-run spread: OLD R0 0.68%, NEW R0 0.80%, OLD R2 0.52%, NEW R2 2.55% (the NEW-R2-2 verify 31.24 ms is one soft sample; the medians are the data points). FIRST20: all R0 runs `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` (identical, bit-identical to the t1431/t1426 R0 stream); all R2 runs' FIRST20 identical to the R0 stream (identical to t1431 R2). TAPE n_ok=128 n_fb=0 on all eight runs. The per-arm signatures (D-ACC, n_refwd, TAPE_CALLS) are exactly the t1426/t1431 battery's — wheel + model + script identity confirmed by signature, not just md5.

**Interleaving (exact order, bracketing drift):** OLD-R0-1 → NEW-R0-1 → OLD-R2-1 → NEW-R2-1 → OLD-R0-2 → NEW-R0-2 → OLD-R2-2 → NEW-R2-2, 20:04:04Z–20:05:28Z.

**OLD vs NEW:** R0 medians 60.635 vs 60.775 (OLD −0.23%); R2 medians 65.83 vs 64.945 (OLD +1.36%) — both inside the within-arm spread. The two wheels are performance-equivalent on today's guest, including the rows-ON verify (29.93 vs 30.58 ms, inside the R2 spread).

## 3. Verdict — the 17.4 ms question

| arm | this round (clock locked, today's guest) | 11-02 01:52–04:10Z anchor | delta |
|---|---|---|---|
| OLD R0 | 60.635 | 69.48 | **−12.7%** |
| OLD R2 (rows ON) | 65.83 / verify 29.925 ms | 75.00 / verify 17.4 ms | **−12.2% / +71.6%** |
| NEW R2 (rows ON) | 64.945 / verify 30.58 ms | (t1431: 65.465 / 30.22) | +1.2% / +1.2% vs t1431 |

The e6d3d182-era wheel does NOT reproduce the 17.4 ms rows-verify on today's guest (29.9 ms, +71.6%), and its rows-ON tok/s (65.83) sits on the t1431/t1426 band, 12.2% below the 75.00 anchor. **The wheel is exonerated; the degradation is guest/hardware state.** Per the brief's fork: next = hardware attention (the DPM-sclk floor is already exonerated by the clock-locked t1431 round; remaining suspects: a second power/thermal governor, or VM/host-layer clocking on the guest — a host-side probe of the physical box's clocks under the same battery is the logical next step).

## 4. Hardware reference (fresh, 2 s-cadence clock sampling concurrent)

D2D copy bandwidth (fresh `hipMemcpy` DtoD, 1 GiB device buffer, 20 timed copies each direction, per-copy device sync): **a→b median 3.790 ms = 263.9 GB/s; b→a median 3.780 ms = 264.5 GB/s** (copy rate; 1 GiB read + 1 GiB write per copy). A fresh reference number for the guest's current D2D path — not an anchor comparison. (The first measurement pass timed enqueue only — 0.001 ms — because DtoD was issued asynchronously; discarded and redone with per-copy synchronization.)

## 5. Old-binary identity

- **The e6d3d182 .so survives nowhere on the guest.** Full scan: every `libmlx.so` under the guest home (all venvs, all build trees, the uv cache archives) — no `e6d3d182`; every built wheel extracted and md5'd (39 wheels: all `wheelout-*` trees, the graft-tree wheelout, all pip-cache wheels) — no `e6d3d182`.
- The t1307 **wheel file** survives (`wheelout-rs`, 22,441,235 bytes, md5 `2442e0633606b8d0977fe91cebb4e6d1`, built 11-02 02:10) but its packaged .so is `db93ed29…` — a different (earlier packaging pass) from the e6d3d182 .so that was active during t1307; the t1307 session itself recorded and pinned this discrepancy.
- **The old arm is therefore a REBUILD, not the original.** Source reconstruction: the t1307 build tree = a copy of the t1297 V2 tree (base 75ca881 + vendor delta + tape conv-row-fix + S4VA + rows dispatch gate, "all four layers applied"), whose `qmv_rows_msg_kernel.hip` had been overwritten by a concurrent task at 02:05 — so the t1307-era kernel (the 00:35 ws[4] wsfix form) was reconstructed as the `.pre-wsfix` backup + `qmv-rows-msg-kernel-wsfix.diff` (applied cleanly, kernel md5 `3dc576ba…`); the tree's `qmm.hip` was untouched since 11-01 23:01 (the t1297 state). Stale `qmm*.o` objects deleted; built with the proven t1307 recipe (ROCm prefix, `CMAKE_HIP_ARCHITECTURES=gfx1201`, `BLA_VENDOR=OpenBLAS`, `pip wheel --no-deps --no-cache-dir`, incremental build dir, the CMakeCache path rewrite). Resulting wheel md5 `830f7d7785861d0a86bb3d0849b8641b` (22,441,344 bytes — 109 bytes larger than the t1307 original, inside build noise); installed .so md5 `3e6f9d5e19114d2bbf7546625ab68b37` == the wheel's extracted .so (consistent). The rebuilt wheel's R0/R2 signatures (D-ACC, n_refwd, TAPE_CALLS, FIRST20) are bit-identical to the t1426/t1431 battery and to the NEW arm — source identity confirmed by signature.

## 6. Clock samples (2 s cadence, battery window 20:04:04Z–20:05:28Z)

44/44 samples: level `high`, sclk `2200Mhz` (the max level), temp 31.0–35.0 °C. `power_dpm_force_performance_level` read `high` pre and post. The held level persisted across all eight runs.

## 7. Sidecar attestation

Before (pre-battery and battery-start) and after: the MTP sidecar shard md5 `02a54359cda8b247f76b277dd9bc4d4c`; the 4-bit sidecar `.disabled`; other shards parked. Unchanged; never touched.

## 8. Exit state

Lock released (verified free), no orphan GPU or sampler processes, sidecar unchanged, wheels/venvs left in place (the scratch rebuild tree + venv left on the guest for the follow-up round). Guest left clean.

## 9. Persona scrub

Report + commit message scrubbed against the mandated pattern set (foreign hostnames, tailnet/LAN IP ranges, foreign user + home paths, model-mount path, codenames, foreign email domains): 0 hits.
