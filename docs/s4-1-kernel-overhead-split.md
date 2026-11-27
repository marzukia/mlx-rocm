# S4(1) — kernel-vs-overhead split of the mmq M=1 path (27B Q2_K vehicle)

**Date:** 2026-11-27 (22:27Z–) · **Box:** 100.78.68.72 (andryo, RX 9070 XT / gfx1201, ROCm 10.1 guest userland) · **One lock-holding GPU session** · **No build**

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
  `/home/andryo/marzuki-neon/<pid>_kernel_trace.csv`).
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

### Step 1 — pre-loaded per-tensor M=1 sweep (both arms)

| arm | pass | bucket | n | GPU p50 (ms) | GPU mean (ms) | GPU sum (ms) |
|---|---|---|---|---|---|---|
| C (mmq) | p1 first-touch | | | | | |
| C (mmq) | p2 steady | | | | | |
| B (kqfg) | p1 first-touch | | | | | |
| B (kqfg) | p2 steady | | | | | |

SPLIT (primary): pass-2 GPU sum of (lin+lmhead) vs the in-context step wall
1566ms / 2639ms → _pending_

### Step 2 — 64-token AR (both arms)

| arm | wall p50 (ms) | GPU/step (ms, trace) | GPU share | host mean (ms) | sync mean (ms) | margin median | md5 first100 |
|---|---|---|---|---|---|---|---|
| C (mmq) | | | | | | | |
| B (kqfg) | | | | | | | |

Cross-compare (banked C7, same wheel/model/prompt class): C 2639ms p50 /
0.3771 t/s; B 3234.9ms p50 / 0.3086 t/s; GEMM-only lever-find C 2684ms vs B
5452ms (2.03x); margin C 0.4375 (6/64 <0.1) vs B 0.3750 (6/64 <0.1).

## GATE verdict

_PENDING_ — criteria: KERNEL-BOUND if step GPU time ≈ step wall (share ≥ ~0.8,
and/or sweep pass-2 gemm-family sum ≥ ~0.8× wall); OVERHEAD-BOUND if GPU
share << 1 (host-side fix class: dispatch/launch/sync overhead is the target).

## Next-lane line

_PENDING_

## Hygiene

- pre/post: kfd holders, VRAM, clocks (sclk class), lock holders — see session.log
- venv SoT: PRE/POST-LIB-MD5 (expect cdb74a57… both), PRE/POST-WHEEL
- host sampler (3s cadence, 11-27 Used-Memory grep fix): vRSS / O_TMPFILE maps /
  shmem / VRAM per arm — see samp-*.log
- box end-state: _pending_
