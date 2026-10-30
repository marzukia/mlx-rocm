# OPTIMISATION GAUNTLET — Area B: Backend-Level Analysis

**Scope:** read-only analysis of the vendored ROCm backend (`vendor/mlx-rocm-upstream` @
the pinned base sha = upstream `the vendor ref`), the built guest wheel
(`mlx-0.32.3.dev20261026+the earlier vendor base`, built 04:11 from `/models/mlx-clone`), and
kernel-level timing on the guest (RX 9070 XT, gfx1201, ROCm 10.1, **32-CU vGPU
partition** — `cus=32` in the boot log; full card is 54 CU).

Headline: 9B-4bit decode is **63 tok/s** on the current wheel (the brief's 31 tok/s
baseline predates the an earlier run patches + ROCm 10.1 — the stack has already 2×'d).
Per token (15.9 ms): **~9.5–11.2 ms in the quantized GEMV (60–70%) at ~490–510 GB/s
(≈50% of 960 GB/s)**, ~4.9 ms in non-GEMM kernels (30%), ~1.5 ms launch/commit/python
gaps (10%). The decode path runs **eager** — 0 HIP graph launches in the trace even
though the backend has full build-once/replay support that nothing in the stack
enables.

---

## 1. GEMM path map

### Dense (`Matmul::eval_gpu` → `gemm_and_bias`, matmul.cpp:759)
- `can_use_gemv(M=1, …)` → **custom `gemv`** (gemms/gemv.hip: `gemv`, `gemv_batched`,
  `gemv_gather`, `gemv_gather_inline` — 128-bit wide loads, warp-per-column).
- else → `gemm_rocblas` → **hipBLASLt** (bf16/fp16), with rocBLAS and `naive_gemm`
  fallbacks. `MLX_ROCM_GEMM_BF16` is read in `preferred_gemm_precision()`
  (hipblaslt_gemm.cpp:259) but the result is `(void)precision` (line 371) —
  **a no-op in this tree**. Battery confirms: `bf16-unset` 63.9–64.3 tok/s ≈
  `bf16=1` 63.2–63.4 tok/s. The fp8 prefill guard is the capability probe only
  (see below).

### Quantized (`QuantizedMatmul::eval_gpu`, qmm.hip:3425) — dispatch order
1. **WMMA-QMM** `qmm_wmma_dense_kernel` (qmm.hip:3290): native bf16 matrix cores,
   64×128 tiles, graph-addable single node, no dequant materialization. Gated:
   transpose, Affine, bf16, batch=1, **M>1**, N%16==0, group_size==64, K%64==0,
   bits∈{4,6,8}, `has_native_wmma()`. Default ON (opt out `MLX_ROCM_WMMA_QMM=0`);
   skipped on low-CU iGPUs (gfx1152). **This is the prefill GEMM** (M≥2).
2. **Dequant GEMM** (`force_dequant_gemm || should_prefer_dequant`):
   a. **fp8 e4m3** (qmm.hip:3573): `M ≥ 64 && device_has_fp8_gemm()` →
      `dequant_fp8_gemm` → `hipblaslt_gemm_fp8_raw` (fp8 matrix cores, block-scaled
      via A/B scale pointers). **Gated only by the heuristic capability probe
      (`gemm_caps().fp8_e4m3`) — no env check.** mxfp8/mxfp4/nvfp4 modes are
      excluded unless `MLX_ROCM_MXFP8_GEMM=1` (opt-in dequant→bf16 route; the
      direct hipBLASLt VEC32_UE8M0 path SIGFPEs on gfx1200's rocRoller swizzle).
   b. **bf16 dequant** + LRU dequant cache → rocBLAS/hipBLASLt.
3. **QMV decode (M≤8, transpose)** — first match wins:
   - **`qmv_wide_kernel`** (qmm.hip:3223) — **DEFAULT ON** (`MLX_QMV_NO_WIDE` to
     opt out). 128-bit (uint4) weight loads, 8 cols/block (4 on low-CU iGPU),
     wave32. Gated: 4/8-bit, **Affine with bias**, transpose, non-batched,
     M≤8, K%32==0, GS∈{32,64,128}, bf16/fp16. Comment: "+14–20% on dense decode,
     ~175 vs ~108 GB/s" (gfx1200 numbers).
   - **`qmv_idot_kernel`** — opt-in (`MLX_QMV_IDOT`): int8 activation quant prepass
     + `v_dot4_i32_iu8` integer dot.
   - **`qmv_tiled_kernel`** — default ON (`MLX_ROCM_QMV_NO_TILED` to opt out),
     persistent CU-bounded grid, 4/8-bit affine (no bias requirement).
   - **`qmv_warp_shared/noshared/batched`** — fast warp path (env-tunable
     threads/cols).
   - **`qmv_fp8_gemv_kernel`** (qmm.hip:2217) — 8-bit E4M3 weights, ROWS=16
     register-chunked, 128-bit loads; used when `!use_fast_qmv`.
   - `qmv_t_kernel` / `qmv_kernel` — generic fallback.

**What a quantized-Q GEMM needs here:** the 4-bit decode path is the f32-FMA
`qmv_wide` loop (per 32 weights: 32× bf16→f32 cvt + 32× f32 FMA + 32× f32 add +
shift/and). RDNA4's 2× bf16-FMA and 4× fp8/`v_dot4` matrix-core rates are
**unused by the default 4-bit kernel** (the fp8 gemv kernel is also f32-FMA;
`qmv_idot` is the only true dot-product path and it is opt-in).

### Attention
`kernel_sdpav_1pass` (flash, 1-pass) — 400 calls/50 steps in the 9B trace,
1.8 ms total. Not a decode bottleneck.

### Graph decode (present but dormant)
`graph_decode_mode()` (device.cpp:1157) defaults OFF; only the C API
`gpu_set_graph_decode_mode` (eval.cpp:130) sets it — **nothing in this tree or in
mlx-lm calls it**. Backend support is complete: build-once capture,
`hipGraphExecUpdate` refresh, `decode_arena` (device.cpp:787–1157),
`MLX_GRAPH_DECODE=0` kill switch. Trace shows **0 graph launches, 0 kernels inside
graphs** → every decode step is eager: ~1,070 dispatches + ~20 buffer commits/token.

---

## 2. Upstream deltas (our tree vs `the vendor ref`)

`vendor.json` pin = the pinned base sha = current `the vendor ref` HEAD (verified byte-identical
clone). **GatedDeltaNet, mxfp8 prefill GEMM, and the gfx1200 fixes are already in
the pin** (the earlier vendor base "rocm: mxfp8 prefill GEMM, cooperative softmax, device_info
keys"). Deltas vs upstream = only the local patches:

| Delta | Files | Perf relevance |
|---|---|---|
| an earlier run chunked async D2H/H2D ≤300 MB, per-chunk `hipStreamSynchronize` | allocator.cpp/.h | Workaround for gfx1201/ROCm 7.7 KFD completion-event hang (300 MB+ sync copies). Small per-step cost (see §4). |
| an earlier run cand-A: lazy host-shadow staging (`has_gpu_contents`) | allocator.cpp/.h | Skips the D2H staging copy when the buffer is fresh and the CPU reader overwrites it. Saves one copy; zero cost when it skips. |
| an earlier run cand-B: per-chunk stream separation (create→async→sync→destroy per chunk) | allocator.cpp | Same hang, stronger workaround. Adds ~stream create/destroy per chunk. |
| ROCm 6.4 compat shims (16-bit `atomicCAS` in hip_atomic_compat, naive_gemm, kernel_utils, load, reduce_utils) | 5 files, ~16 lines | None. |

The guest wheel (`/models/mlx-clone`, 7 files / +133 lines vs the earlier vendor base) contains
exactly these — **both patches are active in the built wheel**. Guest runs ROCm
10.1 (not 7.7); the KFD hang is a 7.7-era bug, the patches are harmless
workarounds there.

---

## 3. Kernel-level timing (rocprofv3 1.4.1, 32-token decode + 16-token warmup)

9B-4bit (group_size 64, has_bias) — decode region after model load:

| Kernel | Time | Calls | Avg | Notes |
|---|---|---|---|---|
| `qmv_wide_kernel<bf16,bf16,GS=64,BV=4>` | **454.3 ms (76.8%)** | 12,450 (260/tok) | 36.5 µs, max 907.9 µs | the decode GEMV |
| `qmm_wmma_dense_kernel` (prefill) | 100.3 ms | 243 | 412.8 µs | M>1 path |
| `rms_norm_kernel` | 35.5 ms | 7,345 (153/tok) | 4.8 µs | |
| `row_reduce_simple` (Sum) | 14.3 ms | 2,544 | 5.6 µs | logit reduce/sampling |
| fused elementwise (swiglu/residual/exp/sigmoid chains) | ~30 ms | — | — | |
| copies (`copy_v`, `copy_gg_byval`, `copy_g_byval`) | 28.1 ms | 6,937 | 1.8–15 µs | KV append etc. |
| `logsumexp_kernel` (prefill attn) | 7.9 ms | 50 | 158.8 µs | |
| `rope_single_1d` | 2.3 ms | 800 | 2.8 µs | |
| `kernel_sdpav_1pass` (flash) | 1.8 ms | 400 | 4.4 µs | |

- Total kernel time 722 ms over a 1,730 ms span → **GPU busy 41.7%** (profiler
  run; profiler inflates wall time — the no-profiler bench is 15.9 ms/token).
- **0 HIP graph launches.**
- Per token: QMV 9.46 ms; non-QMV kernels 4.9 ms; the rest is gaps.
- QMV effective bandwidth: ~4.6 GB/token ÷ 9.46 ms ≈ **486 GB/s (51% of 960)**.
  Largest QMM (N=12288) would take 28 µs at full bandwidth vs 36.5 µs measured —
  the gap is per-QMM launch/tail (1,536 blocks, 6 waves of latency per CU),
  x re-reads from L2, and scale/bias loads.

0.6B-8bit (group_size 64) — same methodology:

| Kernel | Time | Calls | Avg |
|---|---|---|---|
| `qmv_wide_kernel<bf16,bf16,GS=64,BV=8>` | 102.3 ms (48%) | 9,850 (205/tok) | 10.4 µs |
| `qmm_wmma_dense_kernel` (prefill) | 18.3 ms | 191 | 96 µs |
| `copy_v` | 33.2 ms | 2,037 | 16.3 µs |
| `rms_norm_kernel` | 18.3 ms | 5,760 | 3.2 µs |
| `rope_single_1d` | 8.4 ms | 2,800 | 2.9 µs |
| `kernel_sdpav_1pass` | 4.6 ms | 1,400 | 3.3 µs |

- 213.5 ms over 703 ms span → 30.4% busy; GEN under profiler 114.8 tok/s
  (185 no-profiler). QMV effective ≈ 325 GB/s (34%) — smaller tiles reuse less.
- Scaling: time/token = 4.65 ms fixed + b·bytes; 9B = 4.65 + 11.2 ms = 15.9 ms
  (matches bench). The fixed ~4.65 ms/token is launch/commit/eval/python +
  non-QMV kernel time.

Supporting A/B (Area A battery, no profiler): `MLX_MAX_OPS_PER_BUFFER=1` →
17.2 tok/s (58.1 ms/token) vs default 63.2 tok/s → **~40 µs per extra buffer
commit** (~1,050 extra commits/token). KV is pre-allocated and flat
(9B ~411 MB, 0.6B 77.6 MB) — no per-step allocation churn.

---

## 4. Patch status + per-step cost (cand-A + an earlier run in the wheel)

Both patches are **active** in the built wheel (verified in `/models/mlx-clone`:
`has_gpu_contents` in allocator.h, `chunked_copy_stream_separated` in allocator.cpp).

Per decode step the only touched path is the token-id readout:
- cand-A: fresh id buffer → `has_gpu_contents==false` → **D2H skipped** (zero cost).
- an earlier run/cand-B: when a D2H does happen (a few bytes), it is one ≤300 MB chunk on a
  fresh stream: 1× `hipStreamCreate` + `hipMemcpyAsync` + `hipStreamSynchronize` +
  `hipStreamDestroy` ≈ **30–80 µs/token ≈ 0.2–0.5% of the 15.9 ms step**.
- The 300 MB chunking only bites for large copies (model save, whole-buffer D2H) —
  not on the per-token path. H2D staging (prompt embeddings) is the same small path.

Verdict: the patches are active and cheap; they are not a decode bottleneck. The
an earlier run `clear_cache()`→`hipFree` 20 s hang is on the model unload/reload path
(`buffer_cache_.clear()` → free callback → hipFree/hipFreeAsync), not per-step;
ROCm 10.1 may not exhibit it, but the serve stack should avoid `clear_cache()`
mid-session until an earlier run lands.

---

## 5. Top-3 optimization targets

### 1. QMV bandwidth: 490 → 800+ GB/s — expected +35–50% (63 → ~90–95 tok/s), effort: medium
The 4-bit decode GEMV runs at ~51% of peak. Levers (in order of confidence):
- **Larger tiles / fewer blocks**: 8 cols/block → 16–32; amortize launch+tail
  (36.5 µs avg vs 28 µs floor on the biggest QMMs; smaller QMMs are worse).
- **Stage x in LDS** (as `qmv_fp8_gemv` does with ROWS=16) to kill the per-column
  x re-reads from L2.
- **Fuse gate+up** (two 4096×12288 QMMs → one kernel, one x pass).
- **Use the dot-product hardware**: `v_dot4` (fp8 or int8, 4× rate) instead of the
  f32 FMA loop — the opt-in `qmv_idot` already exists as a template; a 4-bit
  fp8-dot variant (x→e4m3 cvt once per 32, 4-bit→fp8 via 16-entry LUT) is the
  cleanest route to ~800 GB/s.
Weight stream is 4.6 GB/token: 490 → 800 GB/s cuts 9.5 → 5.75 ms/token.

### 2. Non-GEMM kernel fusion: 4.9 → 1–2 ms/token — expected +20–30%, effort: medium-high
1,070 dispatches/token; per layer: rms_norm (153/token), KV-append copies
(~140/token), rope, swiglu/residual elementwise chains. Fuse rms_norm+residual-add,
in-place KV append (kill `copy_v`/`copy_gg_byval`), batch the small copies, merge
the swiglu chains. This is the gap between "fast GEMM" and the 177 tok/s ceiling,
which assumes ~0.5 ms of everything-but-weights.

### 3. Enable graph decode in the serve loop — expected +5–10%, effort: small
`gpu_set_graph_decode_mode(1)` (one line in qmlx-serve's generation loop; the
backend's build-once/`hipGraphExecUpdate`/`decode_arena` machinery is already
there). Collapses ~1,070 dispatches + ~20 commits/token into one graph launch
(~40 µs/commit × 20 ≈ 0.8 ms + per-dispatch gaps). Verify interaction with the
an earlier run/cand-B per-chunk streams (id readout stays outside the graph — no conflict).
A/B it with `MLX_GRAPH_DECODE=0` as the kill switch.

**Combined expectation: 63 → ~120–130 tok/s (68–73% of the 177 ceiling).**

---

## 6. Environment notes (affect interpretation)

- **32-CU vGPU partition** (`cus=32` in the boot log; full 9070 XT = 54 CU).
  Memory bandwidth is unpartitioned (~960 GB/s), so the bandwidth ceiling holds,
  but kernel occupancy is CU-limited (48 blocks/CU on the 1,536-block QMV grid).
- **ROCm 10.1** on the guest (the KFD-hang workarounds target 7.7).
- `MLX_ROCM_GEMM_BF16=1` is a **no-op** in this tree (dead `preferred_gemm_precision`
  result); the fp8-prefill guard is the capability probe alone. Battery confirms no
  bf16-unset delta. The "safe config" is inert for decode (M=1 never touches the
  fp8 prefill path; the 9B model is 4-bit so `qmv_fp8_gemv` doesn't apply either).
- 0.6B reference model is **8-bit** quantized (not 4-bit) — keep that in mind when
  comparing its 185 tok/s against 9B's 63 tok/s.

## Evidence index
- Battery (Area A): 9B 63.2–63.4 / bf16-unset 63.9–64.3 / ops-per-buf1 17.2 /
  no-fp8-gemv 61.6–61.9 tok/s; 0.6B 183–189 tok/s; t_step 15.96 ms (9B), 4.86 ms
  (0.6B); KV 411 MB / 77.6 MB flat.
- rocprofv3 traces: `/tmp/rocpd9b`, `/tmp/rocpd06` on the guest (32-token decode
  region, kernel + memory-copy + graph traces).
- Source: `vendor/mlx-rocm-upstream/mlx/backend/rocm/{matmul.cpp, gemms/hipblaslt_gemm.cpp,
  gemms/gemv.hip, quantized/qmm.hip, device.cpp, eval.cpp, allocator.cpp, host_stage.cpp}`;
  patches on branch `an earlier run-salvage-candA` (bd061d6, 37e2834, d82ce84); guest build tree
  `/models/mlx-clone` (the earlier vendor base + 7 files / +133 lines).
