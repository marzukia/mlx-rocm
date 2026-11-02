# REPORT-postreboot-rebaseline — post-reboot re-baseline, full 4-arm A/B (exact t1426/t1431 battery)

**Headline (numbers first): the full reboot did NOT recover the band.** R0 median 61.105, R3 stacked median 65.79 — still ~12% below the 11-02-morning anchor (69.48/75.00) and within +0.9% / +0.70% of the t1431 pre-reboot clock-locked band (60.575/65.33). **Verdict: NOT RECOVERED.** Whatever degraded the band on 11-02 was not stuck GPU device/driver state that a full reboot would clear (the 11-02 07:31Z page-fault burst trigger hypothesis is not supported: the state survives a full power cycle, or the burst was a symptom, not the cause). Combined with t1431 (clock exonerated) and t1432 (wheel exonerated), the suspect is platform-level: the 11-01 reinstall userland/ROCm-stack state, a second power/thermal governor, or host/driver-layer factors.

## 1. Per-arm table (2 runs/arm, medians)

| arm | run 1 tok/s | run 2 tok/s | **median tok/s** | single ms (med) | verify ms (med) | refwd ms (med) | n_refwd | D-ACC | TAPE n_ok | TAPE_CALLS |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 (gate) | 61.10 | 61.11 | **61.105** | 9.805 (9.87/9.74) | 32.955 (32.91/33.00) | 9.17 (9.17/9.17) | 101 | 93/59/27 | 128/128 | 5496 |
| R1 (REFWD_M1) | 62.16 | 62.09 | **62.125** | 9.735 (9.73/9.74) | 33.185 (33.16/33.21) | 9.285 (9.29/9.28) | 94 | 99/62/34 | 128/128 | 5328 |
| R2 (rows ON) | 65.40 | 65.86 | **65.63** | 9.78 (9.83/9.73) | 29.96 (30.01/29.91) | 9.06 (9.12/9.00) | 99 | 95/61/29 | 128/128 | 5448 |
| R3 (STACKED) | 65.74 | 65.84 | **65.79** | 9.73 (9.73/9.73) | 29.975 (30.01/29.94) | 8.99 (9.00/8.98) | 99 | 95/61/29 | 128/128 | 5448 |

In-run spread: R0 0.02%, R1 0.11%, R2 0.70%, R3 0.15%. FIRST20 per arm, both runs: R0 `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` (identical, bit-identical to the t1426/t1431 R0 stream); R1 `… 15352 10597 29006 314 264 6278 20853 318 68 1257 2487 11819 220 16 20 11` (identical, = t1426/t1431 R1); R2/R3 = R0's stream (identical, = t1426/t1431 R2/R3). TAPE n_ok=128 n_fb=0 on all eight runs. The per-arm signatures (D-ACC, n_refwd, TAPE_CALLS) are exactly the t1426/t1431 battery's — wheel + model + script identity confirmed by signature, not just by md5.

## 2. Band vs anchors

| arm | this round (post-reboot, clock locked) | 11-01 anchor | delta | t1431 (clock locked, pre-reboot) | delta vs t1431 |
|---|---|---|---|---|---|
| R0 | **61.105** | 69.48 | **−12.1%** | 60.575 | +0.87% |
| R1 | 62.125 | — | — | 61.395 | +1.2% |
| R2 | 65.63 | 75.00 (statistical) / 68.95 (bit-exact frame) | −12.5% / −4.8% | 65.465 | +0.25% |
| R3 (STACKED) | **65.79** | 75.00 | **−12.3%** | 65.33 | +0.70% |

The post-reboot band sits on the t1431 pre-reboot band (and on the whole stable ~60/65 band since 11-02: 59.47 / 55.93 / 59.675 / 60.575 / 61.105). Relative verdicts (in-session R0 reference): R1/R0 +1.67%, R2/R0 +7.41%, **R3/R0 +7.67%**, R3/R2 +0.24% (within spread; the stacking fix still holds by signature — R3 bit-identical to R2: n_refwd 99, D-ACC 95/61/29, first20 = R0's stream).

## 3. Gates

| gate | result |
|---|---|
| (1) bench md5 `2c2d6c90…` + guard `4a262a0d…` + wheel `93810e87…` + sidecar pin `02a54359…` (before/after) | PASS — sidecar re-verified after, unchanged |
| (2) R0 absolute window 57.33–60.98 (drift headroom 61.5) | PASS (note) — 61.10 / 61.11, both +0.12 above the strict window edge, inside the drift headroom 61.5; consistent with the +0.87% band shift vs t1431, not a new drift |
| (3) within-arm determinism (FIRST20, 2 runs) | PASS — all four arm pairs identical |
| (4) TAPE n_ok=128 n_fb=0 on all eight runs | PASS (silent gate) |

## 4. The reboot question — explicit verdict

**NOT RECOVERED.** A full power cycle (reboot 20:23Z, battery run 20:27:02–20:28:23Z with the guest up ~4 min) changed nothing material: the medians are within +0.9% of the t1431 pre-reboot clock-locked numbers and remain ~12% below the 11-01 anchor. The suspected stuck device state either was not the (whole) cause, or survives a full reboot (in which case it is not "device state" in the resettable sense). The 07:31Z page-fault burst as trigger is not supported. Three independent levers have now been cleared as the single cause of the ~12% band loss: the DPM sclk floor (t1431), the wheel (t1432), and stuck device state across a full reboot (this round). Remaining suspects: the 11-01 reinstall userland/ROCm stack state, a second power/thermal governor, or host/driver-layer factors — a `rocminfo`/driver-version diff and a pre-11-02 userland A/B are the next discriminating steps.

## 5. Clock-hold evidence (2 s-cadence background sampler)

- `power_dpm_force_performance_level`: `high` at pre-check (re-applied post-reboot by the orchestrator; no re-apply needed), `high` at lock acquisition (20:27:02Z), `high` after (CLK_LEVEL_AFTER).
- sclk manual pre-launch read at lock acquisition: `2: 2200Mhz *` (max level); post: `2: 2200Mhz *`.
- Background sampler (fixed 2 s-cadence format capturing level + sclk + temp in one line): **41/41 samples over 20:27:02Z–20:28:23Z = `lvl=high sclk=2: 2200Mhz *`**, temp 31–34 °C (min 31000 / max 34000 m°C; pre 29 °C). The full battery window was 81 s; the sampler bracketed it start to finish, no gaps.

## 6. Setup (verified before the battery)

- **Bench:** canonical `mtp-bench-k3-tape.py` md5 `2c2d6c90c2349ae087e53e5d6c60018e`; guard `gdn_tape_commit.py` md5 `4a262a0d…`. Driver + bench UNMODIFIED — env toggles only.
- **Wheel / venv:** the t1426 battery wheel (libmlx.so md5 `93810e871f7bc977e723d5288a3b73e2`) in the scratch venv under the graft build tree, unchanged.
- **Arms / env:** identical to the t1426/t1431 battery: base `MLX_MTP=4 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1`; R0 nothing extra; R1 `+MLX_ROCM_REFWD_M1=1`; R2 `+MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`; R3 = R1+R2. Standard run env (the model dir, hsa-runtime LD_PRELOAD, gfx1201 target).
- **Sidecar pin:** the MTP sidecar shard md5 `02a54359cda8b247f76b277dd9bc4d4c` active, the 4-bit sidecar `.disabled`, other shards parked — attested before AND after (unchanged). The pin was not touched.
- **GPU lane:** the lock was free at dispatch (no sibling holding it at that moment); acquired 20:27:02Z on the first poll, held across all eight runs, released at exit. The sibling anchor run started its sampler at 20:30Z, after my release — no overlap, no interference.

## 7. Exit state

Lock released (a sibling then acquired it at 20:30Z; my window is clean), no orphan GPU or sampler processes, sidecar unchanged (md5 re-verified after), parked shards untouched, venv + wheel unchanged. Guest left clean.

## 8. Persona scrub

Report + commit message scrubbed against the mandated pattern set: 0 hits.
