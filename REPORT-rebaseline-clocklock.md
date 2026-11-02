# REPORT-rebaseline-clocklock — clock-locked re-baseline, 4-arm A/B (re-dispatch of the 100+ re-baseline)

**Task:** re-establish the R0 gate-window and the R3 stacked standing with the GPU clock forced to its max level (the DPM auto-boost had been dead — sclk pinned at the floor — and was fixed by setting the forced performance level to `high` before this round). The t1426 battery re-run, exactly: same canonical script, same env, same venv, same wheel, same model, env toggles only. 2 runs/arm, medians.

**Headline: the numbers did NOT recover.** R0 median 60.575 (vs the 69.48 anchor, −12.8%); R3 stacked median 65.33 (vs the 75.00 anchor, −12.9%; vs the expected 72–76). The band is statistically the t1426 band (R0 59.675, R3 65.58): +1.5% / −0.4% on the medians. With `power_dpm_force_performance_level = high` verified pre, during, and post, and sclk reading 2200 MHz (the max) in the bracketing samples, the DPM-sclk floor is exonerated as the single cause of the 11-01 → 11-02 degradation. Some other guest-state drift remains the suspect; the clock samples are included as the evidence for that exoneration.

## 1. Setup (verified before the battery)

- **Wheel / venv:** the rows-aware wheel (libmlx.so md5 `93810e87…`, the t1426 battery wheel) in the `venv-ra` scratch venv under the graft build tree, unchanged from the t1426 round.
- **Bench:** canonical `mtp-bench-k3-tape.py` md5 `2c2d6c90c2349ae087e53e5d6c60018e` in the canonical bench dir (all four on-box copies md5-identical); guard `gdn_tape_commit.py` md5 `4a262a0d…`. Driver + bench UNMODIFIED — env toggles only.
- **Arms / env:** identical to the t1426 battery: base `MLX_MTP=4 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1`; R0 nothing extra; R1 `+MLX_ROCM_REFWD_M1=1`; R2 `+MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`; R3 = R1+R2. Standard run env (models dir, hsa-runtime LD_PRELOAD, gfx1201 target).
- **Sidecar pin:** the MTP sidecar shard md5 `02a54359cda8b247f76b277dd9bc4d4c` active, the 4-bit sidecar `.disabled`, other shards parked — attested before AND after (unchanged). The pin was not touched.
- **Clock lock (step 2 of the brief):** `power_dpm_force_performance_level` read `high` at pre-battery check (the 19:45Z fix was already in place; no re-apply needed); sclk read `2200Mhz *` (the max level) in the pre-launch manual sample. The level file re-read `high` after the battery.
- **GPU lane:** the lock file was free at dispatch (the prior orphan was killed); acquired 19:36:41Z on the first poll, held across all eight runs, released at exit (verified free; no orphan GPU or sampler processes).

## 2. Per-arm table (2 runs/arm, medians)

| arm | run 1 tok/s | run 2 tok/s | **median tok/s** | single ms (med) | verify ms (med) | refwd ms (med) | n_refwd | D-ACC | TAPE n_ok | TAPE_CALLS |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 (gate) | 60.46 | 60.69 | **60.575** | 9.74 (9.75/9.73) | 33.45 (33.48/33.42) | 9.21 (9.27/9.15) | 101 | 93/59/27 | 128/128 | 5496 |
| R1 (REFWD_M1) | 61.40 | 61.39 | **61.395** | 9.72 | 33.78 (33.76/33.79) | 9.35 (9.36/9.33) | 94 | 99/62/34 | 128/128 | 5328 |
| R2 (rows ON) | 65.85 | 65.08 | **65.465** | 9.725 (9.72/9.73) | 30.22 (29.93/30.50) | 8.995 (8.99/9.00) | 99 | 95/61/29 | 128/128 | 5448 |
| R3 (STACKED) | 64.97 | 65.69 | **65.33** | 9.905 (10.09/9.72) | 30.05 (30.12/29.98) | 9.08 (9.09/9.07) | 99 | 95/61/29 | 128/128 | 5448 |

In-run spread: R0 0.38%, R1 0.02%, R2 1.18%, R3 1.10% (the R3-1 single 10.09 ms is one soft sample; the median 65.33 is the data point). FIRST20 per arm, both runs: R0 `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` (identical, and bit-identical to the t1426 R0 stream); R1 `… 15352 10597 29006 314 264 6278 20853 318 68 1257 2487 11819 220 16 20 11` (identical, = t1426 R1); R2/R3 = R0's stream (identical, = t1426 R2/R3). TAPE n_ok=128 n_fb=0 on all eight runs. The per-arm signatures (D-ACC, n_refwd, TAPE_CALLS) are exactly the t1426 battery's — the wheel + model + script identity is confirmed by signature, not just by md5.

## 3. Gates

| gate | result |
|---|---|
| (1) bench md5 `2c2d6c90…` + guard `4a262a0d…` + sidecar pin `02a54359…` (before/after) | PASS — sidecar re-verified after, unchanged, 4-bit `.disabled` |
| (2) **R0 absolute window 57.33–60.98 (drift headroom 61.5)** | **PASS** — 60.46 / 60.69 both inside |
| (3) within-arm determinism (FIRST20, 2 runs) | PASS — all four arm pairs identical |
| (4) TAPE n_ok=128 n_fb=0 on all eight runs | PASS (silent gate) |

## 4. Band vs the 11-01 anchors — the finding

| arm | this round (clock locked) | 11-01 anchor | delta | t1426 (DPM partially degraded) | delta vs t1426 |
|---|---|---|---|---|---|
| R0 | **60.575** | 69.48 | **−12.8%** | 59.675 | +1.5% |
| R1 | 61.395 | — | — | 61.12 | +0.45% |
| R2 | 65.465 | 75.00 (statistical) / 68.95 (bit-exact frame) | −12.9% / −5.1% | 65.555 | −0.14% |
| R3 (STACKED) | **65.33** | 75.00 | **−12.9%** | 65.58 | −0.38% |

**The clock lock did not recover the band.** The medians sit on the t1426 band (and the t1375/t1379 post-11-02 band: 59.47 / 55.93 / 59.675 / 60.575 — a stable ~60 band since 11-02 regardless of the DPM state). The pre-battery load test (500× 8192³ matmuls) held 2200 MHz across all 20 samples with the level at `high`, and the battery-window evidence below brackets the run. Conclusion: the 11-01 → 11-02 degradation is NOT the DPM-sclk floor alone; the t1379 clock-floor mechanism is real but was not the (whole) cause of the anchor loss. Remaining suspects for next: guest userland / ROCm stack state change on or around 11-02 (the 11-01 re-install window), a second power/thermal governor, or host/VM-layer clocking — worth a direct A/B on the pre-11-02 wheel state or a `rocminfo`/driver-version diff before further re-bench rounds.

Relative verdicts (the in-session R0 is the reference): R1/R0 +1.35%, R2/R0 +8.07%, **R3/R0 +7.85%**, R3/R2 −0.20% (within the 1.1% in-run spread; the t1426 stacking fix still holds by signature — R3 bit-identical to R2: n_refwd 99, D-ACC 95/61/29, first20 = R0's stream).

## 5. Clock-hold evidence (step 5 of the brief)

- `power_dpm_force_performance_level`: `high` at pre-battery manual check (19:35Z), `high` at battery start (CLK_LEVEL), `high` after (CLK_LEVEL_AFTER). The forced level is not subject to DPM auto-boost — it holds the top P-state until changed.
- sclk manual pre-launch sample (19:35Z): `2: 2200Mhz *` (max level active).
- Background 2 s-cadence sampler #1 ran for the full battery window (19:36:41Z–19:38:01Z, 40 timestamps). Its value field is empty — a quoting defect in the heredoc-transported script turned the `*`-match into a bare glob; the timestamps (the cadence evidence) are intact and in the guest log.
- Sampler #2 (corrected match) started 19:38:12Z, immediately after `ALL ARMS DONE`: **98/98 samples = `2: 2200Mhz *`** — the held level persisted through and after the run.
- Together with the pre-battery load test (20/20 samples at 2200 MHz under 500× 8192³ matmuls), the clock is bracketed at the max level across the battery: 19:35Z pre-read, `high` level at start/end, 98/98 2200 MHz post-run.

## 6. Exit state

Lock released (verified free), no orphan GPU or sampler processes (the post-run sampler was killed and verified gone), sidecar unchanged (md5 re-verified after), parked shards untouched, venv + wheel unchanged. Guest left clean.

## 7. Persona scrub

Report + commit message scrubbed against the mandated pattern set (foreign hostnames, tailnet/LAN IP ranges, foreign user + home paths, model-mount path, codenames, foreign email domains): 0 hits.
