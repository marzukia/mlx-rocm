# S5-1 — the dispatch-count collapse (spec -> implement -> build -> same-session validation)

**Date:** 2026-11-28 (02:18Z–06:19Z worker; the detached final GPU session completed the last leg at 06:20Z and self-cleaned) · **Box:** the GPU box (RX 9070 XT / gfx1201, ROCm 10.1 guest userland) · **Terminal state: timed_out at the 4h wall (by design — the RUNTIME_MAX_BACKSTOP), mid final-validation session; all data on the record; the residue is the card, not the data.**

## Terminal-state note (on the record)

The worker died at the wall with the final lock session (s51-final.sh) still running. That session had been launched detached holding the flock, so it survived the kill: it completed the remaining ladder legs (FCM2, FCM128, FTM2, FTM128 — the last DONE-STATS at 06:20Z), wrote the provenance lines, released the lock, and cleared kfd. Landing verification (06:3xZ): LOCK-FREE, /dev/kfd empty, VRAM 59.9 MB, no orphan processes. Box tree state on the record: /home/guest/build-p2seq @ b9cb137 + leg-4 patch + C1 applied + the C2 hunk reverted (the B1 state that produced the validated wheel).

## Build ledger (two builds, one quarantined)

| build | venv | lib md5 | wheel | state |
|---|---|---|---|---|
| Build 1 = C1 + C2 | venv-s51 | cf6bc664 | mlx-0.32.3.dev20261128+b9cb137 | QUARANTINED — the test arm OOM-crashed during PREFILL (99.6–99.8% HBM, +1.1–1.4 GB over the control at the same phase; failure modes: malloc throw, null-pointer in the dequant kernel, `Event::is_signaled` assertion — one mode per retry). Prefill is the phase where C1 is INACTIVE (M=256, the mmq M=1 path does not run) → the excess is 100% C2. |
| Build B1 = C1 only | venv-s51b1 | e377b26f | mlx-0.32.3.dev20261128+b9cb137 (26.9 MB, sha256 3a9b26c3…) | VALIDATED — clean M2 smoke (B1-M2), clean full ladder. This is the test arm of record. |
| Control | venv-ar | cdb74a5794 | (the leg-4 wheel, unchanged) | re-verified at session start; untouched. |

**C2 root cause (named, not fully root-caused):** the uninitialized f32 scratch (`rocm::malloc_async` array, no `zeros()` fill) adds ~1.1–1.4 GB of HBM at prefill vs the control. The old code's `set_data` reallocation freed the fill buffer (it was dead weight anyway); the new path's array ownership/lifetime keeps something resident that the old path dropped (deleter/temporary bookkeeping — the exact line is the S5-2 RCA item). C2 is HBM-side: it touches the footprint axis as well as the dispatch axis. Revert was the only change between Build 1 and B1 (single-hunk revert, incremental build).

## The change set (full spec: docs/s5-1-dispatch-collapse-spec.md, same branch)

Per-call dispatch model (source-read): C arm = 4 dispatches/gemm call (dead zeros() fill + hipMemsetAsync pad + quant launch + mul_mat_q); B arm = 3. C-B = +404/step explains part of the measured 1142 gap; the residual ~738 is not attributable to the per-call source (S4(1) flags the C arm as profiled-cost-inflated: 34% inter-kernel gap, 118–270 µs/dispatch vs 9 µs).

- **C1 (VALIDATED, in the test wheel):** fold the pad-zero into the quant kernel — `mmq_quant_in_q8_1_mmq` gains a `real_total` param; pad blocks write 144 B of zeros in-kernel; the per-call `hipMemsetAsync` is deleted from `mmq_q2k_file`. −404 dispatches/step.
- **C2 (QUARANTINED):** replace the dead `zeros()` f32 temp with an uninitialized `malloc_async` buffer in `KQuantFileGemm::eval_gpu` (the kqfg kernels write ALL op elements; the fill was dead). −404 dispatches/step, but +1.1–1.4 GB HBM at prefill.
- Named structural sources disposition: (a) the per-step `flush_all_dirty_shadows` at device.cpp:2584 = verified ALREADY only-when-dirty (no dispatch when clean; the 404 kqfg calls use the flush-free Device member) — no change; (c) the per-chunk streams in `shadow_chunked_copy` = only >100 MB copies, not a per-step source — out of scope (S5-2, with the 1384-leak lane).
- Predicted dispatches for the validated (C1-only) arm: 3656 − 404 = **3252** (the spec's 2848 assumed C1+C2). The profiled dispatch cross-check (did it actually land at ~3252?) was NOT run — owed to S5-2 as the mechanism verification (secondary; the unprofiled tps is the gate).

## Validation (ONE lock session; CONTROL ladder first, TEST ladder second — order recorded)

64-token in-context AR (c7run class, CH=256 prefill, greedy, MTP-OFF, the d500x prompt), RELEASE_SHADOW=1 both arms (the standard arm per the ARM-CONFIG ruling), the exact C1 config env, same file, same load path, back-to-back. The depth-scaled protocol per arm (M64 p50 + M128 warmup-excl; the first-fire M128 lesson applied — the silence threshold scaled to the depth-grown step time). Sampler validated against a known-nonzero baseline before the session (the samval pattern; the S4(1) 14.47 GB artifact is the cost of skipping this).

| leg (arm) | n | decode wall (s) | tps (all) | tps (warmup-excl) | ms/step p50 (warmup-excl) | margin median | md5 full |
|---|---|---|---|---|---|---|---|
| FCM64 (control) | 64 | 173.36 | 0.3692 | 0.3692 | 2692.9 | 0.4375 | 387b252b… |
| FTM64 (test) | 64 | 167.63 | 0.3818 | 0.3818 | 2591.7 | 0.4375 | 387b252b… |
| FCM2 (control) | 2 | 3.28 | 0.6089 | 0.6089 | 1642.4 | 0.40625 | ca803647… |
| FTM2 (test) | 2 | 3.27 | 0.6122 | 0.6122 | 1633.6 | 0.40625 | ca803647… |
| FCM128 (control) | 128 | 520.52 | 0.2459 | 0.15689 | 6493.0 (p50-all 4008.4) | 0.5 | 23558e8f… |
| FTM128 (test) | 128 | 535.82 | 0.2389 | 0.15336 | 6576.7 (p50-all 4170.8) | 0.5 | 23558e8f… |

- **MD5 bit-identical at ALL THREE depths, both arms** (64: 387b252b…, 2: ca803647…, 128: 23558e8f…; margins identical) — the change is LOSSLESS on the vehicle.
- **The control reproduces the S4(2) baseline:** FCM64 2692.9 ms vs 2713.3 ms (−0.8%); FCM128 6493.0 ms vs 6509.8 ms (−0.3%) — the session comparison is valid, the box/wheel/harness are consistent.
- **VRAM per leg (rocm-smi 15 s host sampling / in-process hipMemGetInfo peak):** FCM64 15.957 / 15.842 GB; FTM64 16.047 / 15.951 GB; FCM2 15.669; FTM2 15.644 GB; FCM128 16.166 / 16.050 GB; FTM128 16.166 / 16.050 GB. All in the 15.9–16.6 GB census class (S4(2)); the C1 test arm is +0.09–0.11 GB at M64 vs the control (run-to-run variance class; C1 changes dispatches, not allocations — the pad region stays allocated via xq81 in both arms). The footprint is UNCHANGED by C1, as expected — the footprint axis is the S5-2 item.
- **Provenance:** control lib cdb74a5794 / test lib e377b26f (ledgered pre/post session); model sha256 3b8d3c60… (8,985,410,528 B, re-verified at run time); prompt sha be02c4f1… (28,574 B / 4,096 tok); harness = c7run.py (WARMUP=100 > maxtok so all steps count).
- **Named gap (on the record, binds S5-2):** the final session's clock reader recorded per-arm START timestamps + the 15 s VRAM cadence, but NOT the sclk/mclk class per arm (the S4(2) pattern is 15 s sclk/mclk + perf-state). The ORDER is recorded (control first, test second) and the test arm ran warmer → the M64 delta is an UPPER BOUND. This is the same confab class P2 caught; S5-2's validation restores the per-arm clock read (the established method, fully).

## GATE verdict

**SURVIVES-AS-MEASURED-MODERATE (C1 only).** At the 64-token in-context AR gate class (M64): **tps +3.4% (0.3818 vs 0.3692; p50 2591.7 vs 2692.9 ms = −3.8%) — an UPPER BOUND** (test ran second on warm clocks; the matched-clock truth is ≤ 3.4%). At M128 (warmup-excl): −2.2% (0.15336 vs 0.15689; 6576.7 vs 6493.0 ms) — the noise class at 6.5 s/step, where the depth-grown attention O(depth) cost dominates and a ~100 ms/step dispatch saving is ~1.5% of the step. At M2: +0.5% (negligible). The first measured positive of the S5 campaign is banked, lossless, with the mechanism (−404 dispatches/step, 3656 → ~3252 predicted) UNVERIFIED in profile — the profiled dispatch cross-check is the S5-2 debt. The warm-clock upper-bound note binds: a matched-clock re-run (order-swappable, per-arm sclk read) is owed before the +3.4% is cited as a number.

## S5-2 next-lane line (expanded scope per the 05:3xZ op callout)

In order: (1) the matched-clock re-run of the M64 gate (order-swappable, per-arm sclk/mclk + perf-state at 15 s — closes the named gap above; this is the citation-grade number for the tps axis) + the profiled dispatch cross-check of the C1 wheel (SDK-injected rocprof, the S4(1) method — confirms 3656 → ~3252; secondary evidence). (2) The C2 RCA (the ~1.1–1.4 GB prefill HBM excess — temporary lifetime/deleter in the f32 scratch path) + re-validate C2 if fixable (it is the second −404 AND an HBM-side item). (3) The mmq M=1 depth-growth measurement (the attention O(depth) decomposition via the S4(1) trace method — measure before fix; M128 p50 6.5 s is the number to beat). (4) The 1384-SharedSignalPool exit-leak fix (release the pool at interpreter exit; the per-chunk stream residue in shadow_chunked_copy rides this lane — depth-independent, 7×-reproduced defect). (5) THE HBM-FOOTPRINT FIX — the named WAVE-13 class (f32→bf16 tightening of the 360 f32 tensors + userland arena reclaim; expected 1.8–2.5 GB; the second root-cause axis per the op callout — spec anchors: docs/lever-finding-q2k-vehicle.md on this mirror + the WAVE-13 P2 spec dispatch record). S5-2 VALIDATION GATE LANGUAGE (binds, per the 05:3xZ correction): vehicle footprint vs the 14G rough gate WITH the fix + tps vs the 41.2 parity reference — no "axis closed / hard limitation" language in any S5 card until the fixed footprint is measured. Standing S5 validation gates unchanged (unprofiled same-clock order-swappable 64-tok AR; depth-scaled protocol; in-run census with samval-first; per-arm md5; provenance).

## Box hygiene (final)

Wall kill 06:19Z (by design); the detached s51-final.sh completed the ladder at 06:20Z and self-cleaned; landing-verified LOCK-FREE / kfd empty / VRAM 59.9 MB / no orphans. venv-ar untouched (cdb74a5794); venv-s51 (cf6bc664, quarantined) + venv-s51b1 (e377b26f, validated) on the box. Full session data banked to hydrogen ~/s51-data (677 MB incl. logits: s51f ladder, s51w M64 gate pair, s51/s51m2/s51b1 early gates + bisection, s51-final.sh, the b1 build log). No venv swap/restore; the leg-4 SoT library is byte-unchanged.
