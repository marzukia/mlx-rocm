# S5-2 RUN 1 — STEP 1: the matched-clock M64 re-run (data of record)

**Date:** 2026-11-28 (06:40:50Z–07:26:29Z, ONE lock session, 45m39s) · **Box:** marzuki-neon (RX 9070 XT / gfx1201, dGPU HIP_VISIBLE_DEVICES=0) · **Script:** /home/andryo/s52/s52f.sh (s51-final.sh reference pattern + the S4(2) clock pattern RESTORED + order INVERTED)

## The protocol

FULL 6-leg depth-scaled ladder (M64 p50 + M128 warmup-excl per arm; same depth order 64/2/128 per arm as S5-1) in ONE flock-held detached session, **order INVERTED vs S5-1: TEST (venv-s51b1, C1 wheel e377b26f) FIRST, then CONTROL (venv-ar, leg-4 wheel cdb74a5794)** — the inversion removes S5-1's warm-clock confab in the opposite direction: now the TEST arm warms its own clocks from 0 and the control inherits warm clocks, biasing the test arm's delta UP (conservative for a positive C1 claim). Env = the exact C1 config both arms (`MLX_KQFILE_MMQ_M1=1`; `MLX_KQUANT_FILE_MR`/`MLX_MMQ_J`/`MLX_ROCM_RELEASE_SHADOW` unset — verified in every leg's DONE-STATS env line). Harness = c7run.py unchanged (WARMUP=100 > maxtok so all steps count).

**The S4(2) clock pattern is RESTORED and WORKING.** Named root cause of the S5-1 sclk gap on the record: `s51-final.sh`'s clock reader used `rocm-smi --showclockinf`, which is NOT a valid flag on this box (usage error, silent empty capture) — so S5-1 recorded start timestamps + the 15 s VRAM cadence but no clock class. This run uses the verified S4(2) read (`--showclocks` + `--showperf`, 15 s cadence; SCLK reads fine on this dGPU in-run — the spec's N/A hedge does not fire; perf-state recorded as-is).

**samval-first (the S4(1) lesson binds):** known-nonzero baseline BEFORE the session — 2 GiB fp32 hold on GPU0; sampler read 2.4177 GB ×3 vs baseline 59.9 MB → **SAMPLER-VAL-PASS**.

## The ladder (S5-2 re-run; S5-1 numbers alongside)

| leg | n | tps (all) | p50 ms/step (warmup-excl) | margin median | md5 full | S5-1 same leg (tps / p50) |
|---|---|---|---|---|---|---|
| **FTM64 (test, FIRST)** | 64 | **0.3653** | **2740.2** | 0.4375 | 387b252b… | 0.3818 / 2591.7 (test ran SECOND) |
| **FCM64 (control, SECOND)** | 64 | **0.3695** | **2728.4** | 0.4375 | 387b252b… | 0.3692 / 2692.9 |
| FTM2 | 2 | 0.6069 | 1647.7 | 0.4062 | ca803647… | 0.6122 / 1633.6 |
| FCM2 | 2 | 0.6068 | 1648.1 | 0.4062 | ca803647… | 0.6089 / 1642.4 |
| FTM128 | 128 | 0.2383 (all) / **0.1539** (wu-excl) | **6558.4** (p50-all 4188.5) | 0.5 | 23558e8f… | 0.2389 / 0.15336 / 6576.7 (p50-all 4170.8) |
| FCM128 | 128 | 0.2468 (all) / **0.1581** (wu-excl) | **6459.6** (p50-all 3923.1) | 0.5 | 23558e8f… | 0.2459 / 0.15689 / 6493.0 |

All 6 legs rc=0, clean; `1384 SharedSignalPool Signals leaked` at every clean exit (the 7th identical count — the named depth-independent exit-leak, unchanged, the S5-2 item (4)).

**MD5 invariance vs S5-1: BIT-IDENTICAL AT ALL THREE DEPTHS, BOTH ARMS** (64: `387b252b248fb4bbdc449ab0344204bb`, 2: `ca8036479956e01d6f0eaf322e756f57`, 128: `23558e8ff3bebe180903f8553af85494` — full stream, not just first100; margin distributions identical too: M64 count<0.1=6 / count<0.5=33 / maxabslogit 24.25; M128 count<0.1=15 / count<0.5=60 / maxabslogit 24.25). No divergence: the vehicle is lossless across S5-1 → S5-2 on both wheels.

## The clock-class lines (15 s cadence, decode windows; GPU[0])

| leg | sclk seq (MHz, decode window) | sclk bulk (min–max, median) | mclk | perf-state |
|---|---|---|---|---|
| FTM64 (test) | 3258, 3313, 3312, 3015, 2711, 2685, 2739, 2712, 2690, 2694, 2651, (215, 0 = post-run ramp-down) | 2651–3313, **median 2711** | 875→772 class | auto (all readings) |
| FCM64 (control) | 3293, 3329, 3325, 3208, 2947, 2840, 2737, 2695, 2705, 2711, 2710, 2696, 2741 | 2695–3329, **median 2737** | 875→772 class | auto (all readings) |
| FTM128 (test) | ramps 3353→3319 at decode start, eases 2600–2750 with inter-sample dips (the 79–218 dips = the 15 s sample landing in the inter-step host gap on a 6.5 s cadence) | median 2671 | 875→772 | auto |
| FCM128 (control) | same shape (warm start 3324–3313, 2640–2740 bulk) | median 2682 | 875→772 | auto |

**Clock-class assessment:** both M64 arms sit in the SAME class — decode opens in the 3250–3330 MHz class and eases to the 2650–2750 class (median 2711 vs 2737, Δ26 MHz = 1.0%); mclk 875→772 in both; perf-state `auto` as-is in both. The one asymmetry (the test arm's tail eased a touch earlier/deeper, 2651 floor; the control opened slightly warmer from inheritance) is the mirror image of S5-1's asymmetry and biases the test arm's delta UP, i.e. it is CONSERVATIVE for a positive C1 claim. FTM2's decode (3.3 s) is shorter than the 15 s cadence — no in-window reading; M2 is negligible either way (see the delta line).

## The VRAM census (direct rocm-smi, start/mid(decode start)/post + 3 s host cadence)

| leg | start | mid (decode start) | post | host-cadence max |
|---|---|---|---|---|
| FTM64 | 59.9 MB | 15.56 GB | 59.9 MB | 15.949 GB |
| FCM64 | 59.9 MB | 15.57 GB | 59.9 MB | 15.957 GB |
| FTM2 | 59.9 MB | 15.67 GB | 59.9 MB | 15.669 GB |
| FCM2 | 59.9 MB | 15.69 GB | 59.9 MB | 15.685 GB |
| FTM128 | 59.9 MB | 15.67 GB | 59.9 MB | 16.170 GB |
| FCM128 | 59.9 MB | 15.67 GB | 59.9 MB | 16.170 GB |

Same census class as S5-1/S4(2) (the 15.6–16.2 GB class; the 15.9–16.6 GB decode class, expected, not a gate — the 85% HBM cap 14.53 GB is exceeded by the decode, on record since S4(2)). The footprint is UNCHANGED by C1, as expected (C1 changes dispatches, not allocations).

## Provenance

- control lib md5 PRE/POST `cdb74a5794fed2bdfec5103e085969c6` (untouched); test lib md5 PRE/POST `e377b26f9740b3d368581c472f050858` (untouched).
- wheels: control `mlx-0.32.3.dev20261127+b9cb137` (leg-4), test `mlx-0.32.3.dev20261128+b9cb137` (C1-only, S5-1 B1 build).
- model sha256 `3b8d3c602f63414794d5f43ceb22bbb61eb586af2cddf07cbf10656860f5b80a` (re-verified at run time, every leg's gen log); prompt sha256 `be02c4f1a46bef08bc94d59024f9337c2406ad2dee848f807f831886424347d` (28,574 B / 4,096 tok, every leg).
- harness = c7run.py unchanged; env line verified per leg.

## GATE line (the citation-grade same-clock M64 delta)

**S5-1's +3.4% IS REVISED — the same-clock delta is zero within the noise class.** Same-clock M64: **test 0.3653 tps / 2740.2 ms p50 vs control 0.3695 tps / 2728.4 ms p50 = −1.1% tps / +0.43% p50 (the TEST arm is the slower one, under the order that biases it UP)**. With S5-1 (test second, warm) at +3.4% and this run (test first, self-warmed) at −1.1%, the sign FLIPS with the order while the magnitude stays ≤3.4% — the S5-1 number was the warm-clock upper bound exactly as flagged, and the matched-clock truth is |Δ| ≤ ~1.1% at M64 (≤2.6% at M128, ≤0.1% at M2). The control arm reproduces S5-1 to 3 digits (0.3695 vs 0.3692; 2728.4 vs 2692.9 ms = +1.3%), so the session is valid.

**M128 warmup-excl: the delta does NOT survive as a positive** — test 0.1539 vs control 0.1581 = −2.6% (test slower; p50 6558.4 vs 6459.6 = +1.5% test-slower), the same direction as S5-1 (−2.2%): a small, consistent, test-slower tilt in the noise class at 6.5 s/step where the depth-grown attention O(depth) cost dominates and a ~100 ms/step dispatch saving is ~1.5% of the step (the S5-1 class, named).

**GATE verdict: C1's unprofiled tps delta = ZERO within noise (lossless, md5 bit-identical at all 3 depths).** The +3.4% is NOT citable as a C1 effect. The mechanism (−404 dispatches/step, 3656 → ~3252 predicted) stands UNVERIFIED in profile — the STEP-2 dispatch cross-check decides whether the dispatch collapse actually landed (verified) or whether the whole C1 delta is reclassified as unexplained variance (cloudy note (2)).
