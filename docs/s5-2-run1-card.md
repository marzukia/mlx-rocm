# S5-2 RUN 1 — the matched-clock M64 re-run + the profiled dispatch cross-check (card)

**Date:** 2026-11-28 (06:40:46Z–08:1xZ; step-1 session 06:40:50Z–07:26:29Z 45m39s, step-2 re-fired session 07:59:10Z–08:07:48Z 8m38s) · **Box:** the GPU box (RX 9070 XT / gfx1201, ROCm guest userland, dGPU HIP_VISIBLE_DEVICES=0) · **Two lock sessions** (both detached flock-held; the s51-final.sh pattern held) · **Scope:** the S5-2 ruling item (1) only — the matched-clock M64 re-run (TEST-FIRST order swap, full 6-leg ladder, per-arm sclk/mclk + perf-state at 15 s RESTORED = closes the S5-1 named sclk gap) + the profiled dispatch cross-check (S4(1) trace method). Items (2)–(5) are later runs. · **No build, no re-spec, no re-implement** (C1-only wheel venv-s51b1 e377b26f + control venv-ar cdb74a5794 reused byte-identical, ledgered pre/post both sessions).

## Step 1 — the matched-clock M64 re-run (GATE: the citation-grade same-clock delta)

Full data: docs/s5-2-run1-step1.md. Protocol: one flock session, order INVERTED vs S5-1 — TEST (C1) FIRST, CONTROL second (the mirror confab: test self-warms from 0, control inherits warm; biases a positive C1 claim UP = conservative). The S4(2) clock pattern is restored and working; **root cause of the S5-1 sclk gap named: `s51-final.sh` used `rocm-smi --showclockinf`, a non-existent flag (silent usage error) — this run uses the verified `--showclocks` + `--showperf` read**. samval-first PASS (59.9 MB → 2.4177 GB under the 2 GiB hold, ×3).

| leg | tps (all) | tps (wu-excl) | p50 ms/step (wu-excl) | margin median | md5 full | S5-1 same leg |
|---|---|---|---|---|---|---|
| **FTM64 (test, FIRST)** | **0.3653** | 0.3653 | **2740.2** | 0.4375 | 387b252b… | 0.3818 / 2591.7 (test 2nd) |
| **FCM64 (control, 2nd)** | **0.3695** | 0.3695 | **2728.4** | 0.4375 | 387b252b… | 0.3692 / 2692.9 |
| FTM2 | 0.6069 | 0.6069 | 1647.7 | 0.4062 | ca803647… | 0.6122 / 1633.6 |
| FCM2 | 0.6068 | 0.6068 | 1648.1 | 0.4062 | ca803647… | 0.6089 / 1642.4 |
| FTM128 | 0.2383 | **0.1539** | **6558.4** (p50-all 4188.5) | 0.5 | 23558e8f… | 0.2389 / 0.15336 / 6576.7 |
| FCM128 | 0.2468 | **0.1581** | **6459.6** (p50-all 3923.1) | 0.5 | 23558e8f… | 0.2459 / 0.15689 / 6493.0 |

**Clock-class lines (15 s, decode windows, GPU[0]):** FTM64 3258→2651 (bulk median 2711) vs FCM64 3293→2696 (bulk median 2737; Δ26 MHz = 1.0%); mclk 875→772 both; perf-state `auto` as-is both (the N/A hedge does not fire — SCLK was readable in-run). Same class; the one asymmetry is the mirror of S5-1's and is conservative for a C1 claim. **MD5 bit-identical at ALL 3 depths, both arms, vs S5-1** (`387b252b…` / `ca803647…` / `23558e8f…`, full stream) + identical margin distributions — lossless, no divergence. `1384 SharedSignalPool Signals leaked` at every clean exit (7th identical count). Census: decode-start 15.56–15.69 GB, host-cadence max 15.95–16.17 GB — the S4(2) census class, footprint unchanged by C1. Provenance: lib md5s pre/post both venvs, wheel dist-infos, model sha `3b8d3c60…`, prompt sha `be02c4f1…` (28,574 B / 4,096 tok), env line per leg.

**GATE line: S5-1's +3.4% IS REVISED — the same-clock M64 delta is zero within the noise class: test −1.1% tps / +0.43% p50 (the TEST arm is the slower one, under the order that biases it UP).** The sign flips with the order (S5-1 test-2nd +3.4% → S5-2 test-first −1.1%) while the control reproduces S5-1 to 3 digits (0.3695 vs 0.3692; +1.3% p50) — the S5-1 number was the warm-clock upper bound exactly as flagged. **M128 wu-excl: the delta does NOT survive as a positive** — −2.6% (test slower; 6558.4 vs 6459.6 = +1.5%), the same direction as S5-1 (−2.2%): a small consistent test-slower tilt in the noise class at 6.5 s/step where the depth-grown attention O(depth) cost dominates (the S5-1 class, named). The +3.4% is NOT citable as a C1 tps effect; the matched-clock truth is |Δ| ≤ ~2.6%.

## Step 2 — the profiled dispatch cross-check (GATE: mechanism verified / reclassified)

Full data: docs/s5-2-run1-step2.md. The S4(1) SDK-injected rocprof (1.1.0 new724 build; the 1.3.2 tool of record remains NO-TRACE), one measured in-decode step per arm (c7run glue byte-identical, 10 warmup steps, monotonic window — the SDK CSV `Start_Timestamp` is the CLOCK_MONOTONIC domain, verified by smoke; the CSV at CWD-relative `<hostname>/<pid>_kernel_trace.csv`, NOT `$ROCPROF_OUTPUT`), same C1 config env, 15 s clock reads kept, samval-first PASS. Two first-fires on the record (rc=127 missing `LD_LIBRARY_PATH` — the SDK needs `libhsa-amd-aqlprofile64.so.1` from `/home/guest/t2246/libs` + the 7.2.4 lib dir; the second fire ran clean but the SDK had defaulted to `results.db` (missing `ROCPROF_OUTPUT_FORMAT=csv`) AND the window was wall-clock against the monotonic domain → 0 count).

| arm | wheel | dispatches/step | the shape |
|---|---|---|---|
| **TRC3-TEST** | C1 (e377b26f) | **3337** | **`fillBufferAligned: 0`** + 6-arg (C1 `real_total`) `mmq_quant_in_q8_1_mmq:423` + `mul_mat_q:375` |
| **TRC3-CTRL** | leg-4 (cdb74a5794) | **3760** | **`fillBufferAligned:423`** (the pad memsets) + 5-arg `mmq_quant_in_q8_1_mmq:423` + `mul_mat_q:375` (rest of the mix byte-identical) |

gen11 (11 post-prefill greedy tokens) IDENTICAL both arms (`240691,27034,50074,157594,157594,113145,216474,128505,134961,203624,98714`), `next_argmax=9114` both, measured-step wall 1.930/1.904 s — the same step, the same clock class (3250–3369 in-run both). Control 3760 vs S4(1) 64-step-avg 3656 = +104 (+2.8%, single-step-at-depth-11 vs 64-step average, the run-to-run drift class).

**GATE line: MECHANISM VERIFIED.** Δ test−control = **423 ≈ predicted 404** (97.5% realized; the +85 gap over the 3252 prediction is fully carried by the +104 control-baseline drift, not a C1 failure) — and the 423 pad memsets vanished 1:1 into the C1-signed 6-arg quant kernel: dispatch-shaped, the C1 shape. NOT the reclassification branch (not still ~3656; the delta is the pad-memset shape exactly).

**Combined verdict:** the −404 dispatches/step collapse LANDED (3760 → 3337), is bit-identical lossless at all 3 depths, and produces **zero measurable tps delta within the noise class** (−1.1% / +0.43% p50, matched clock class). The C1 dispatch axis is CLOSED as measured; the tps axis stands at parity (the ~100 ms/step dispatch saving at S4(1)'s 118–270 µs/dispatch is sub-noise at the M64 gate class).

## S5-2 run-2 next-lane line (item (2): the C2 RCA)

Source-NAME the exact lifetime/deleter line for the 1.1–1.4 GB prefill HBM excess of the C2 change (the uninitialized f32 `malloc_async` scratch in `KQuantFileGemm::eval_gpu` replacing the dead `zeros()` temp): the old path's `set_data` reallocation freed the fill buffer (dead weight); the new path's array ownership/lifetime keeps something resident that the old path dropped (deleter/temporary bookkeeping — the exact line is the named RCA, no invented mechanism). The fix build (C2 fixed, on the C1 base); re-validate = prefill OOM repro RESOLVED (the C1+C2 OOM at prefill, 99.6–99.8% HBM) + prefill census shows the excess RELEASED (phase-aligned vs control) + md5 bit-identical at all 3 depths. If the scratch turns out to be load-bearing, C2 stays QUARANTINED with that finding (C1 stands as measured). Then (3) the mmq M=1 depth-growth measurement (measure before fix; M128 p50 6.5 s is the number to beat), (4) the 1384-SharedSignalPool exit-leak fix, (5) the HBM-footprint fix (the WAVE-13 class). S5-2 VALIDATION GATE LANGUAGE unchanged: vehicle footprint vs the 14G gate WITH the fix + tps vs the 41.2 parity reference.

## Box hygiene (final)

Both sessions detached flock-held, self-cleaned. Final state verified: LOCK-FREE, /dev/kfd empty, VRAM 59.9 MB (GPU0) / 16.6 MB (GPU1 idle), no orphans. venv-ar untouched (cdb74a5794 pre/post); venv-s51b1 untouched (e377b26f pre/post); venv-s51 (cf6bc664) remains quarantined, not used. Session data on the box: /home/guest/s52/ (s52f ladder, trace2/trace3, s52f.sh, s52trace.sh, trace1step.py), /home/guest/s52f.nohup, /home/guest/s52trace3.nohup.

## Budget record

wall 4h (RUNTIME_MAX_BACKSTOP); spent: setup+samval ~10 min, session A 45m39s, trace fires (1 failed 50 s + 1 clean-but-0-count ~9 min + smoke 2 × ~1 min + success 8m38s), card ~15 min — ~1h35m total, well inside the 212.5-min budget; exit before the wall by design (no in-run CI waits).
