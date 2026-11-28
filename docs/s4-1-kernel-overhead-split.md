# S4(1) — kernel-vs-overhead split of the mmq M=1 path (27B Q2_K vehicle)

**Date:** 2026-11-27 (22:27Z–23:45Z) · **Box:** 100.78.68.72 (andryo, RX 9070 XT / gfx1201, ROCm 10.1 guest userland) · **One lock-holding GPU session** (23:12:17Z–23:31:05Z, 18m48s) · **No build**

## Question (GATE)

The leg-4 wheel (0.32.3.dev20261127+b9cb137) routes M=1 kquant gemm to mmq
(mr_gate=1). The 64-token AR step wall is 1566ms (SMOKE) / 2639ms (SC-C p50,
C7 leg-4 wheel) — tok/s ≈ 0.38. Is the M=1 path **kernel-bound** (GPU time ≈
step wall → the kernel campaign is the fix) or **overhead-bound** (GPU time <<
wall → host-side fix class)?

## Method

- **Timing:** this wheel has **no `mx.Event` API** (verified by namespace
  dump; `mx.rocm` exposes only arena/quant; `mx.metal` has no timing). Method
  = per-kernel durations from the memory-255 SDK-injected rocprof (1.1.0
  build at `/home/andryo/rcp/new724`, `librocprofiler-sdk-tool.so` +
  `ROCPROFILER_LIBRARY_CTOR=1` + `ROCPROF_KERNEL_TRACE=1`; the 1.3.2 tool of
  record is NO-TRACE on this box). Two-call trace smoke PASS (cfgq2k line
  present; 14 kernel rows with durations; CSV at
  `<cwd>/marzuki-neon/<pid>_kernel_trace.csv` — the SDK writes to a
  CWD-relative dir, not `$ROCPROF_OUTPUT`).
- **Windows:** scripts log per-call / per-step wall-ns windows (`t0` before
  submit, `t1` after `mx.eval` — eval is synchronous); the host-side
  attributor (`s41-attribute.py`, no GPU) joins kernel rows to windows by
  start timestamp (bisect; same monotonic clock domain, verified in the smoke).
- **Arms:** C = wheel default (mmq); B = `MLX_KQUANT_FILE_MR=0` (kqfg).
  Separate process runs; same file, same load path, same box state,
  back-to-back under one lock. No lib swap (venv SoT md5 cdb74a57… pre/post
  ledgered — the brief's 070122d8 is the venv-hbmfix / p2 save-lib.so, NOT
  the venv-ar leg-4 lib; recorded discrepancy, no action).
- **Vehicle:** Qwen3.8-27B-Q2_K-plain.gguf (8985410528 B; md5
  487b5c9d…c94a… ledgered), C7 harness class (CH=256 prefill, greedy,
  maxtok=64, FullLayer64 block-64 full-attention fix, MTP-OFF build),
  tokenizer + prompt from the C7 path; model glue = c7run.py class.
- **AR split cross-check (no profiler):** per step, host = `t_submit - t0`
  (CPU enqueue), sync = `t1 - t_submit` (GPU wait).

## The 410 set (header parse, runtime-verified)

| bucket | n | note |
|---|---|---|
| lin | 403 | in-model KQLinear gemm family (the per-step M=1 set) |
| lmhead | 1 | output.weight (in-model) |
| u64attn | 4 | blk.64 attn q/k/v/output (in file, UNUSED by MTP-OFF vehicle) |
| eh | 1 | blk.64.nextn.eh_proj (unused MTP) |
| embed | 1 | token_embd.weight (in-model via GATHER; gemm-shape call = bound figure) |
| small | 96 | N=48 class (ssm_alpha/ssm_beta; dequant-once + tiny bf16 matmul; excluded) |

410 = lin+lmhead+u64attn+eh+embed (N>=256 class); **in-model per-step gemm
family = 404** (lin+lmhead). All 410 pass the 84-remainder check (view =
(K, N/256, 84)); the 96 small tensors are not kquant-gemm-shaped at all
(N=48 is not 256-block-representable) — consistent with the dequant-once
class.

## Results

### Step 1 — pre-loaded per-tensor M=1 sweep (both arms, 2 passes)

GPU-attributed (rocprof kernel sum per call window), pass-2 steady:

| arm | bucket | n | GPU p50 (ms) | GPU mean (ms) | GPU sum (ms) |
|---|---|---|---|---|---|
| C (mmq) | lin | 403 | 15.708 | 15.281 | 6158.2 |
| C (mmq) | lmhead | 1 | 320.941 | 320.941 | 320.9 |
| C (mmq) | u64attn | 4 | 4.652 | 6.653 | 26.6 |
| C (mmq) | eh | 1 | 13.241 | 13.241 | 13.2 |
| C (mmq) | embed | 1 | 10.037 | 10.037 | 10.0 |
| B (kqfg) | lin | 403 | 281.567 | 272.963 | 110004.3 |
| B (kqfg) | lmhead | 1 | 5756.317 | 5756.317 | 5756.3 |
| B (kqfg) | u64attn | 4 | 63.914 | 91.262 | 365.0 |
| B (kqfg) | eh | 1 | 181.570 | 181.570 | 181.6 |
| B (kqfg) | embed | 1 | 26.218 | 26.218 | 26.2 |

Pass-1 (first-touch) sums: C 6522.2ms / B 115511.3ms — p1 ≈ p2 for both arms
(no autotune inflation: B/C gemm sums 109777.8 / 6200.5 → 110004.3 / 6158.2).

SPLIT (primary): pass-2 GPU sum of (lin+lmhead): **C 6479.1ms, B 115760.6ms**
(standalone ratio 17.8x) vs the in-context step wall 1566ms / 2639ms
(→ 4.14x / 2.46x for C; 73.9x / 43.9x for B).

**Sweep caveats (do NOT use these for the GATE):** (a) the sweep's x is
`mx.zeros` = **float32**, the vehicle's hidden is **bfloat16** — the
standalone kqfg arm may sit on a worse path than in-context; (b) some lin
tensors appear stored transposed in the GGUF (a cfgq2k line shows
M=1 N=256 K=348160 — not a vehicle shape), so a subset of sweep calls ran on
mis-read views; (c) sweep-C started at low clocks (sclk 1877MHz) vs the AR
runs (3213–3356MHz). The in-context AR below is the primary evidence.

### Step 2 — 64-token AR (both arms, vehicle-identical)

| arm | wall p50 (ms) | wall mean (ms) | GPU/step mean (ms, trace) | GPU share | span≈wall | host mean (ms) | sync mean (ms) | dispatches/step | margin median | md5 first100 |
|---|---|---|---|---|---|---|---|---|---|---|
| C (mmq) | 2969.5 | 2938.0 | 1944.0 | **0.662** | 2932.8 | 3.2 | 2934.8 | 3656 | 0.5000 | 5382c8384f7f… |
| B (kqfg) | 3320.2 | 3330.5 | 3301.2 | **0.991** | 3324.1 | 3.7 | 3326.8 | 2514 | 0.5938 | 847448d9203a… |

Per-step kernel mix (64-step sums, ms): C: mul_mat_q (mmq gemm) 99673.5
(=1557.7/step) + kqfg_mr_gemm (fallback shapes) 17438.5 (272.5/step) +
Cijk (rocblas small class) 6101.8 (95.3/step) + quant_in 179.0 + copy 296.5 +
fill 171.9. B: kqfg_gemm 205158.7 (3205.6/step) + Cijk 5535.0 (86.5/step) +
rms_norm 99.2 + sdpav 80.1 + row_reduce 75.3 + kqfg_cast 46.6.

→ In-context per-step gemm family: **C 1829.2ms vs B 3205.6ms = 1.75x mmq
faster** (directionally consistent with the C7 lever-find 2.03x).

**Cross-checks:** C wall 2194→4227ms grows across the 64 steps (profiler
backpressure; C7 unprofiled p50 2638.6ms — the profiled run is +12.5% p50).
B wall is flat (3307.0→3307.9ms, +2.6% vs C7 3234.9ms). C gap
(wall − attributed kernel) = 994ms/step (34%), B gap = 29ms/step (0.9%).
Both md5s are the word-salad class for the d500x adversarial prompt
(C7: C da0b502d6dc8 / B 8479176dbac7 — not comparable across runs; margins in
the same class: 0.5000 / 0.5938 vs C7 0.4375 / 0.3750, 10/64 and 9/64 <0.1).

## GATE verdict

**KERNEL-BOUND** — with a dispatch-gap caveat.

- **B (kqfg reference):** clean kernel-bound — GPU 99.1% of wall, 3.7ms host.
- **C (mmq, the path in question):** GPU-busy ≈ wall (sync wait = 2934.8ms
  vs wall 2938.0ms = 99.9% of the step is GPU-side); attributed kernel time
  = 66.2% of wall; host enqueue = 3.2ms (0.1% of wall — the host-side fix
  class is ruled out). The 34% remainder is **inter-kernel gap** (GPU-side
  dispatch-class): C issues 3656 dispatches/step vs B's 2514 (+45%), at
  118–270µs per dispatch vs B's ~9µs (the per-dispatch cost is amplified by
  profiler instrumentation; the true share is likely ~75–80% kernel, cf. the
  early-run gap of ~431ms/step = 20%).
- So: not purely kernel-bound (66% < 80%), but decisively NOT
  host-overhead-bound (3.2ms). The fix class is the GPU-side kernel campaign:
  (a) cut the per-step dispatch count (mmq issues 45% more micro-dispatches
  than kqfg — fuse the per-block loops), (b) the mul_mat_q M=1 kernel itself
  (1557.7ms/step = 53% of the C wall).

## Next-lane line

S4(2) as planned is **de-prioritized** (host-side ruled out as the lever by
3.2ms). Primary next lane = **S5 kernel campaign on the mmq M=1 path**:
(1) per-step dispatch-count reduction (3656 → ~2514 class), (2) mul_mat_q
M=1 kernel work (1557.7ms/step), (3) unprofiled re-verification of the gap
(2969.5ms profiled p50 vs 2638.6ms C7 unprofiled p50).

## Hygiene

- pre/post: kfd holders, VRAM, clocks (sclk class), lock holders —
  session.log. Session 23:12:17Z→23:31:05Z; pre-sweep VRAM 1.34GB → post
  60MB (clean release; the 7.2GB transient after session-1 kill released
  before session 2); clocks ramp 0→1877→3356MHz as expected; post: sclk
  3356 / no low-power.
- venv SoT: PRE/POST-LIB-MD5 = cdb74a5794fed2bdfec5103e085969c6 both;
  PRE/POST-WHEEL = 0.32.3.dev20261127+b9cb137.
- host sampler: the built-in 3s sampler had two bugs (pattern file path used
  as the pgrep pattern → self-match; `head -1` picked the `timeout` wrapper).
  Fixed (v3: `PAT=$(cat "$1")`, anchored pattern). The fixed live sampler
  (500 samples, both AR arms) reports: **vRSS max 10.1GB, /dev/otmpfile maps
  max 500, shmem max 880MB, VRAM max 14.47GB < 14.53GB cap (85%)**.
- SVAL gate: samval binary ran OK (SVAL-ALIVE pid-ok, SVAL-EXIT-OK rc=0;
  idle 1.34GB + 512MB copy = ~1.85GB ≪ 14.53GB cap); the *sampling* was the
  buggy one (see above) — recorded as PASS-with-note; the live fixed sampler
  stands as the host-data evidence.
- box end-state: kfd empty, gpu.lock released (fuser: no holders), VRAM 60MB
  (GPU0) / 16.6MB (GPU1 idle), no low-power, wheel/lib md5 intact.
