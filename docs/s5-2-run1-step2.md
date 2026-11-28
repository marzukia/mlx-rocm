# S5-2 RUN 1 — STEP 2: the profiled dispatch cross-check (data of record)

**Date:** 2026-11-28 (re-fired session 07:59:10Z–08:07:48Z, ONE lock session, 8m38s; a first fire 07:36:18Z–07:37:15Z failed in 50 s on a missing `LD_LIBRARY_PATH` — rc=127, no GPU time consumed, clean post-state, on the record; the second fire ran both arms clean but counted 0 kernels — the SDK had defaulted to `results.db` (missing `ROCPROF_OUTPUT_FORMAT=csv`) and the window used wall-clock against the CSV's monotonic domain; both named here, the S4(1) smoke pattern's two env lines are load-bearing) · **Box:** the GPU box (dGPU gfx1201) · **Method:** the S4(1) SDK-injected rocprof (1.1.0 build `/home/guest/rcp/new724`; `LD_PRELOAD librocprofiler-sdk-tool.so` + `ROCPROFILER_LIBRARY_CTOR=1` + `ROCPROF_KERNEL_TRACE=1` + `ROCPROF_OUTPUT_FORMAT=csv` + `LD_LIBRARY_PATH` the 7.2.4 lib dir + /home/guest/t2246/libs — the first-fire failure was the missing LD_LIBRARY_PATH: the SDK needs `libhsa-amd-aqlprofile64.so.1` from there; the 1.3.2 tool of record remains NO-TRACE on this box)

## The method (the S4(1) trace method, per-step window)

One measured in-decode step per arm: `trace1step.py` (c7run model glue imported byte-identical — same GGUF/TOK/PROMPT, CH=256 prefill, greedy, seed 42, MTP-OFF; 10 warmup decode steps after prefill, then ONE step with a `time.monotonic()` window around `model()` + `mx.eval` — the SDK's CSV `Start_Timestamp` is the CLOCK_MONOTONIC domain, verified in a 2-call smoke this run, the S4(1) monotonic-domain lesson applied). Dispatches/step = kernel_trace.csv rows with `Start_Timestamp` in `[t0−0.1, t1+0.1]`. The CSV lands CWD-relative at `<hostname>/<pid>_kernel_trace.csv` (NOT `$ROCPROF_OUTPUT` — the S4(1) note holds). Both arms: the exact C1 config env (`MLX_KQFILE_MMQ_M1=1`, the other three MLX_* unset), same harness, same lock session, 15 s clock reads kept.

**Sanity (the measured step is the same step in both arms):** gen11 (the 11 post-prefill greedy tokens) = `240691,27034,50074,157594,157594,113145,216474,128505,134961,203624,98714` in BOTH arms; `next_argmax=9114` both; measured-step wall 1.930 s (test) / 1.904 s (control) — the depth-11 step, consistent with the S4(2) depth-growth class (1.65 s @M2 → 2.71 s @M64). Clocks: both arms in-run sclk 3250–3369 (the 41/600 dips = the 15 s sample in inter-kernel profiler idle), same class. samval-first: PASS (59.9 MB → 2.4177 GB ×3).

## The per-arm dispatch counts (the GATE numbers)

| arm | wheel | dispatches/step | kernel mix (the shape that decides the gate) |
|---|---|---|---|
| **TRC3-TEST** | venv-s51b1 (C1, e377b26f) | **3337** | copyBuffer 712, **mmq_quant_in_q8_1_mmq (6-arg, the C1 `real_total` signature) 423, mul_mat_q 375**, rms_norm 299, copy_g_byval 163, binary_vv Add 135, copy_gg_byval 134, binary_vs Mul 96, Cijk (rocblas small) 96, row_reduce 96, Sigmoid 67, fused-attention class 67 — **`fillBufferAligned: 0` (absent)** |
| **TRC3-CTRL** | venv-ar (leg-4, cdb74a5794) | **3760** | copyBuffer 712, **`fillBufferAligned` 423 (the per-gemm pad hipMemsetAsync), mmq_quant_in_q8_1_mmq (5-arg, pre-C1) 423, mul_mat_q 375**, rms_norm 299, … (the rest of the mix is byte-identical to the test arm's, incl. the 375 mul_mat_q and 712 copyBuffer) |

- Δ test−control = **423** (vs the predicted 404 = 104.7% of the prediction realized).
- The control's 3760 vs S4(1)'s 64-step-averaged 3656 = +104 (+2.8%): single-step-at-depth-11 vs the 64-step average (the 423 gemm-family calls this fire vs S4(1)'s 404 — the same run-to-run drift class as the tps numbers).
- **The 423 pad memsets vanished 1:1 into the C1-signed quant kernel** (6-arg `mmq_quant_in_q8_1_mmq` with the `real_total` param, the in-kernel 144 B pad-zero per the S5-1 change set) and the 5-arg pre-C1 signature is GONE from the test arm. The delta is DISPATCH-SHAPED, exactly the C1 shape.

## GATE line (the mechanism line)

**MECHANISM VERIFIED.** Test 3337 dispatches/step vs control 3760 = −423 ≈ the predicted −404 (3656→3252 prediction class; the +85 gap over 3252 is fully carried by the +104 control-baseline drift vs S4(1), NOT by a C1 failure — 423/404 of the predicted removals landed 1:1 with the kernel-signature swap as the direct proof). NOT the reclassification branch: the test arm is not still ~3656 and the delta is dispatch-shaped (the pad memsets are gone, not merely fewer).

**The combined S5-1+S5-2 verdict for the C1 change:** the −404 dispatches/step collapse LANDED in profile (3760 → 3337), is bit-identical lossless (md5 at all 3 depths), and produces **zero measurable tps delta within the noise class** (step-1 GATE: −1.1% / +0.43% p50, same clock class) — the dispatch saving at ~118–270 µs/dispatch (S4(1)'s inflated per-dispatch cost) is real but sub-noise at the M64 gate class. The C1 dispatch axis is CLOSED as measured; the tps axis stands at parity.

## Provenance

- control lib md5 PRE/POST `cdb74a5794fed2bdfec5103e085969c6`; test lib md5 PRE/POST `e377b26f9740b3d368581c472f050858` (both untouched).
- model sha256 `3b8d3c602f63414794d5f43ceb22bbb61eb586af2cddf07cbf10656860f5b80a` (both arms' gen logs); prompt = the d500x one-line (both arms, same sha class as step 1).
- CSVs: `trace3/<hostname>/241021_kernel_trace.csv` (test), `241261_kernel_trace.csv` (control) on the box.

## Box hygiene (post-step-2)

KFD empty, VRAM 59.9 MB (GPU0), LOCK-FREE, no orphans; the census POST lines (11.0/13.2 GB) are the pre-drain reads right after process exit — the FINAL read is 59.9 MB (the arena drain, the 1384-pool class). No venv modification (md5s ledgered pre/post).
