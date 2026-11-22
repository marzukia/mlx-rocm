# Env flags in the tree

Every env flag ACTUALLY read by code in this repo, as of 0.3.0-alpha (released 2026-11-20; main `3e84320e`): the ROCm backend (`vendor/mlx-rocm-upstream/`, the partial byte-verified reference snapshot that the pinned build tree carries), the in-tree model (`models/qwen3_5.py`), and the `bench/` / `scripts/` harnesses. A "Default" is the value when the variable is unset (for presence-gates, `unset = off`). Only what is in the tree is listed — no invented flags. For a flag that is set by the harness (e.g. the record driver exporting `MLX_MTP=3`), the default is the code-side one.

## The 0.3.0 record protocol (MTP K=3 decode)

The measured record protocol is the `MLX_MTP=3` + `MTP6_PREFILL_FILL=1` class (MTP K=3; 111.01 t/s decode banked on main; in-tree record `docs/w1/REPORT-W1-RECORD-11184-20261114.md`). `docs/mtp-enablement.md` is the recipe. The in-tree repro driver (`bench/gdn_seq1024_repro.sh`) exports the arm env verbatim.

| Flag | Default | Effect |
|---|---|---|
| `MLX_MTP` | `0` | the bench spec-decode mode: `0` plain greedy; `1` K=1 (two T=1 target forwards); `2` K=1 (one S=2 batch); `3` K=2 (one S=3 batch — the record mode); `4` K=3 (one S=4 batch). Read by the bench harness, not the model. |
| `MTP6_PREFILL_FILL` | `0` | `1` = the MTP prefill fill (the H4 decisive test in `bench/k2arm-pinned.py`). |
| `MTP6_STREAM_FILL` | `0` | `1` = the streaming catch-up fill. |
| `MTP6_TAPE` | `0` (asserts `0/1/hybrid/r5/slot/slot2/seq`) | the S=3/S=4 GDN tape-arm selector; routes the `MLX_ROCM_GDN_*` arm env (see below) and sets `MLX_ROCM_GDN_T3=1` + `MLX_ROCM_GDN_FUSE=1`. |
| `MTP6_DRAFT_HEAD` | `int2` (the mtp-bench6 family: `bf16`) | the drafter head width: `int2` / `int2w` / `bf16` (model 4-bit `QuantizedLinear` direct) / `bf16full` (bf16 dequant probe). |
| `MTP6_HEAD_H` | `prenorm` | the drafter hidden input feature: `prenorm` / `postnorm` (trunk final RMSNorm in-bench). |
| `MTP6_MOD_BF16` | `0` | `1` = the half-precision (bf16) draft module (post-load-eval lazy cast; `models/qwen3_5.py`). |
| `MTP6_NGRAM` / `MTP6_NGRAM_S3` | `0` | the zero-cost n-gram continuation fill (mode-3 S=3 arm, t1986). |
| `MTP6_NGRAM_N` | `4` | the n-gram length. |
| `MTP6_D1POS` / `MTP6_D3POS` | `0` | the draft-position probes. |
| `MTP6_MARGIN` | `0` | the margin probe (with `MTP6_D1POS`). |
| `MTP6_DRAFTGATE` / `MTP6_DGATE_T0` / `MTP6_DGATE_T1` / `MTP6_DGATE_W` | `0` / `4` / `12` / `8` | the draft-gate threshold class. |
| `MTP6_ACCEPT` | `savestate` | the acceptance mode. |
| `MTP6_MTPCACHE_TRIM` | `1` | the MTP cache trim (default on in the pinned bench). |
| `MTP6_MTPTRIM` | `0` | the MTP trim variant. |
| `MTP6_IDLE_S` | `25` | the idle window (s). |
| `MTP6_REPS` | `5` | the bench reps. |
| `MTP6_SLOT_COPY` / `MTP6_SLOT_IMM` | `0` | the slot copy/immediate probes. |
| `MTP6_DTYPE_PROBE` | `0` | the dtype probe. |
| `STEPDEC_OUT` | unset | set (with mode `3`) = the step-decode output. |
| `MTP4_SIDECAR` / `MTP4_SIDECAR_F32` / `MTP_SIDECAR` | unset | the sidecar path overrides; `MTP4_SIDECAR` → `model-mtp-4bit.safetensors`, `MTP4_SIDECAR_F32` → `model-mtp.safetensors` (next to the 9B weights, `bench/k2arm-pinned.py`). |
| `CH_MTP_4BIT` / `MTP_4BIT` | `0` | `1` = the 4-bit MTP module (the `model-mtp-4bit.safetensors` sidecar; `models/qwen3_5.py` `MTP_4BIT`). |
| `MTP_Q4K` / `MTP_Q4K_GGUF` | `0` / `perclass-probe/Qwen3.5-9B-Q4_K_M.gguf` | the Q4_K sidecar class (the pinned bench). |
| `MTP_QBITS` | `4` | the quant bits for the in-tree model rebuild (`models/qwen3_5.py`). |
| `MTP_MAX_TOKENS` | `1024` | the bench token budget. |
| `MTP_MODEL` / `MTP_AR_N` | `$MODELS/qwen3-9b-mlx` / `512` | the model ref / the AR control length. |
| `MTP6_PROMPT` | the canonical W1 prompt (verbatim) | the record prompt override (t1996). |

## GDN / fused-decode kernels (backend + in-tree model)

| Flag | Default | Effect |
|---|---|---|
| `MLX_ROCM_GDN_FUSE` | off | `1` = the fused GDN step (the 4-launch GDN decode chain → 1 launch; `gdn_fused_step.hip`; admitted when `B==1`, `T<=4`, `Dk/Dv==128`). |
| `MLX_ROCM_GDN_FALLBACK` | off | `1` = force the GDN fallback even with `GDN_FUSE=1`. |
| `MLX_ROCM_GDN_T3` | off | `1` = admit the `T=3` (S=3 verify) `TMAX=3` instantiation (default off; unset = the clear error / fallback, as pre-`TMAX=3`). |
| `MLX_ROCM_GDN_TAPE` | off | `1` = the GDN tape mode (the op-snapshot tape; the `MTP6_TAPE` 1/hybrid/r5 arm). |
| `MLX_ROCM_GDN_SLOT` | off | `1` = the slot op (snapshot commit rows; `gdn_fused_step_slot`). |
| `MLX_ROCM_GDN_SLOT2` | off | `1` = the slot2 fused sequential slot (one per-row phase for all M rows; `gdn_fused_step_slot2`). |
| `MLX_ROCM_GDN_SEQ` | off | `1` = the W1 sequential GDN kernel (the llama.cpp GDN design port, bit-exact; `gdn_fused_step_seq`). Engages under `MTP6_TAPE=seq` + `MLX_ROCM_GDN_SEQ=1`. |
| `MLX_ROCM_W1_INBA_FUSE` | off | `1` = the W1 in_ba fusion (in-tree model, `models/qwen3_5.py`). |
| `MLX_W1_INBA_DUMP` | off | `1` = the layer-0 in_ba npz dump. |
| `MLX_W1_INBA_ALLOC_PROBE` | unset | set = the in_ba allocation probe (`allocator.cpp` + `qmm.hip`). |
| `W1_DUMP_TAG` / `W1_PROBE_TAG` | `x` / unset | the dump/probe file tag. |
| `MLX_GDN_TAIL_FUSION` | unset | the wheel-gates arm selector (`scripts/wheel-gates.sh`: `1` fused / `0` unfused). |
| `P2_GDN_DEBUG` | unset | set = the fused-step GDN debug (`gdn_fused_step.hip`). |
| `MLX_GDN_DEBUG` | unset | set = the GDN debug prints (the `gated_delta_update` + copy + device paths). |

## SDPA / attention

| Flag | Default | Effect |
|---|---|---|
| `MLX_SDPA_FATTN` | off (unset or `0` = bit-identical pre-port dispatch) | non-empty non-`0` (e.g. `1`) = the attention port: RDNA WMMA `mma-f16` flash attention (`flash_attention_fattn.hip`). ON in the merged record state per op letter (a) = ADOPT (the #36 lane record); the 12% prefill-attention kernel win. |
| `MLX_SDPA_NO_WMMA` | off | set = disable the WMMA flash dispatch (the S1 fix axis: `MLX_SDPA_NO_WMMA=1` = f32 prefill; see the bf16 WMMA Q-staging 1-ULP tie class, `docs/w1` record caveat). |
| `MLX_SDPA_DECODE_FLASH` | off | set = enable the decode-flash SDPA path. |
| `MLX_SDPA_FLASH_VJP` / `MLX_SDPA_NO_FLASH_VJP` | off | set = enable / disable the flash VJP (the LSE forward the VJP needs). |
| `MLX_ROCM_SDPA_S4VA` | unset | set = the S=4 (K=3 spec-decode verify batch) 1-pass vector-kernel dispatch; default off, bit-identical with the env unset. |
| `MLX_ROCM_KV_F16` | unset | set = the f16 KV gate (`fast.cpp` KV path). |

## GEMM / allocator / MoE

| Flag | Default | Effect |
|---|---|---|
| `MLX_ROCM_GEMM_BF16` | off (fp8 e4m3 when the card reports `fp8_e4m3`) | `1` = force the bf16 GEMM precision — the safe default on gfx1201 until the fp8 long-prefill is proven clean (the fp8 probe passes; long-context prefill garbles on some RDNA4 drivers). |
| `MLX_ROCM_GEMM_DEBUG` | off | set = the GEMM debug output. |
| `MLX_NO_HIPBLASLT` | off | set = bypass hipBLASLt (the naive GEMM path). |
| `MLX_ROCM_NO_HIPBLASLT_EPILOGUE` | off | set = disable the hipBLASLt epilogue. |
| `MLX_HIPBLASLT_NO_PIPE_CACHE` | off | set = disable the hipBLASLt pipeline algo cache. |
| `MLX_HIPBLASLT_EXACT_CACHE` | off | set = the exact algo-cache mode. |
| `MLX_ROCM_HIPBLASLT_TUNE` | off | set = the algo tune (auto-tune on first use). |
| `MLX_ROCM_STAGING_RETAIN_CAP` | unset | the plan-D staging-retention cap (the `allocator.cpp` staging path). |
| `MLX_ROCM_FINEGRAINED` | off | set = the fine-grained allocation mode. |
| `MLX_ROCM_USE_ASYNC_POOL` / `MLX_ROCM_FORCE_ASYNC_POOL` / `MLX_ROCM_NO_ASYNC_POOL` | off | the HSA async-pool on/off gates (async-pool off by default in the fork; `hbmfix` = the sync `hipMalloc` fallback when the pool fails to extend, the HBM-full-after-reclaim edge). |
| `MLX_ROCM_USE_VRAM` | off | set = the VRAM allocation path. |
| `MLX_ROCM_ALLOW_MANAGED_FALLBACK` / `MLX_ROCM_NO_MANAGED_FALLBACK` | off | set = allow / disable the managed-memory fallback. |
| `MLX_ROCM_D2H_CHUNK_MB` | unset | the D2H copy chunk size (MB). |
| `MLX_GRAPH_POISON_FREE` / `MLX_GRAPH_NODEFER` | off | set = poison freed memory / never defer graph frees (debug). |
| `MLX_WORKER_DEBUG` | unset | set = the verbose worker debug prints. |
| `MLX_COUNT_COPIES` | off | set = count copy ops (diagnostic). |
| `MLX_EVENT_BLOCKING` | off | set = blocking events (debug). |
| `MLX_ROCM_MOE_*` (the `ZERO_SYNC`, `PACK`, `PACK_EXACT`, `PAD_SYNC`, `DEVICE_SEG`, `SEG_MIN`, `ASYNC`, `NO_PIN_CACHE`, `PIN_STATS`, `VJP_DEVICE_SEG` set) | off | the MoE gate class (27B / MoE — not validated on gfx1201). |
| `MLX_ROCM_SORTED_GATHER` | unset | set = the sorted-gather gate (MoE `matmul.cpp` path). |

## QMM / QMV / mmq kernel tuning

| Flag | Default | Effect |
|---|---|---|
| `MLX_ROCM_QMM_DEQUANT_GEMM` | unset | set = the opt-in dequant→bf16 GEMM path (the mxfp4 / 4-bit prefill path; no native mxfp4 kernel yet). |
| `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE` | unset | the dequant-cache size. |
| `MLX_ROCM_QMM_M3` | unset | the M3 gate (the `qmm.hip` W1-fix path). |
| `MLX_ROCM_FORCE_LOW_CU` | off | `1` = force the low-CU config. |
| `MLX_Q8_0_MMQ` | off | `1` = the Q8_0 mmq path. |
| `MLX_ROCM_WMMA_QMM` | unset | the WMMA dense-qmm allowlist config (the RDNA4 tier crossover). |
| `MLX_ROCM_MXFP8_GEMM` | unset | set = the mxfp8 GEMM gate. |
| `MLX_ROCM_MMQ_Q4K` | unset | set = the Q4_K mmq path (the #36 port). |
| `MLX_ROCM_MMQ_Q4K_ALLOC` | pool-routed scratch | `0` = the raw fallback (the P1B4 scratch fix's off switch; default = pool-routed scratch in `mmq_q4k_g64`). |
| `MLX_ROCM_MMQ_Q4K_SCRATCH` | unset | the Q4_K scratch mode. |
| `MLX_ROCM_MMQ_Q4K_REPACK` | unset | the Q4_K repack gate. |
| `MLX_ROCM_MMQ_Q4K_KQFILE` | unset | the Q4_K kquant-file path. |
| `MLX_ROCM_MMQ_REPACK_MAX_SHAPES` / `MLX_ROCM_MMQ_REPACK_MAX_BYTES` | unset | the per-shape repack caps (plan-D); `MAX_BYTES` accepts case-insensitive SI (1000-base) `k`/`m`/`g` suffixes (the t3171 F1-medium fix). |
| `MLX_KQFG_MR_TILE` / `MLX_KQUANT_FILE_MR` | unset | the kquant-file MR gate / tile. |
| `MLX_MMQ_J` | the heuristic | the J tile-size override (8..128, multiple of 8). |
| `MLX_MMQ_CFG` | unset | the mmq config string override. |
| `MLX_MMQ_DBG` / `MLX_MMQ_NONFATAL` / `MLX_4KDBG` | off | the mmq debug / non-fatal errors / 4K mmq debug. |
| `MMQ_AB_DUMP` / `KQFG_AB_DUMP` | unset | the A/B dump directory. |
| `MLX_GLU_MMQ` / `MLX_GLU_Q4K` | unset | the GLU (swiglu) mmq / Q4_K gate. |
| `MLX_ROCM_QMV_ROWS` / `MLX_ROCM_QMV_ROWS_HOIST` | unset | the W1 QMV rows cap / the rows-hoist. |
| `MLX_ROCM_QMV_TILE_N` | the heuristic | the QMV N tile. |
| `MLX_ROCM_QMV_NO_TILED` / `MLX_QMV_NO_WIDE` / `MLX_QMV_NO_MR` | on (tiled / wide / MR) | set = disable the tiled / wide / MR QMV (the default is the fast path). |
| `MLX_ROCM_QMV_6BIT_SLOW` | off | set = the 6-bit slow path. |
| `MLX_QMV_LX` | unset | the lx (x-chunk-sum) mode. |
| `MLX_QMV_MR2_F32` | off | set = the MR2 f32 path. |
| `MLX_QMV_IDOT` / `MLX_QMV_IDOT_GATHER` | off | set = the idot QMV / gather idot. |
| `MLX_ROCM_QMV_NO_FP8_GEMV` | off | set = disable the fp8 gemv. |
| `MLX_ROCM_GATHER_QMV_USE_TILED` | off | set = the tiled gather QMV. |
| `MLX_ROCM_M4ROWEXACT` / `MLX_ROCM_M4PREF` | off | the M4 row-exact / prefix. |
| `MLX_T1251_DBG` | off | set = the T1251 debug. |
| `MLX_ROCM_GROUPED_PREFILL` / `MLX_ROCM_GROUPED_PREFILL_MIN_B` | unset | the grouped-prefill gate / min batch. |
| `MLX_GRIDX_MULT` | the heuristic | the grid-x multiplier. |
| `MLX_ROCM_FORCE_WARP_SIZE` | unset | force the warp size. |

## Plan-C / Plan-D (2026-11-18, the 3.7K lane)

| Flag | Default | Effect |
|---|---|---|
| `MLX_ROCM_ACT_BF16` | off | exactly `1` = hold the matmul output in bf16 when it would otherwise be f32 (the trunk-activation GEMM class; unset / ≠1 = bit-identical to legacy). |
| `MLX_ROCM_LOGITS_BF16` | off | exactly `1` = hold the lm-head matmul output in bf16 (the plain-Linear / tied-embedding path; the f32 default is preserved otherwise). |
| `MLX_ROCM_INPROJ_PAD` | off | `1` = the GDN `in_proj_b/a` via one fused GEMM padded to N=128 (existing kernels only; the #37 pad; e2e unmeasured at merge, no win claimed at the default). |

## Graph decode (the G1 lane — retired, NO-GO verdict)

| Flag | Default | Effect |
|---|---|---|
| `MLX_GRAPH_DECODE` / `MLX_ROCM_GRAPH_DECODE` | off | set = the graph-decode mode (the G1 lane; `docs/audit/slot-path-mtp-verify.md` NO-GO; the engine mode is not enabled). |
| `MLX_ROCM_GRAPH_DECODE_ARENA_MB` / `MLX_ROCM_GRAPH_DECODE_BIG_OPS` / `MLX_ROCM_GRAPH_DECODE_VERIFY` | unset | the arena size / big-ops gate / the verify gate. |
| `MLX_GRAPH_PREFILL_REPLAY` / `MLX_GRAPH_REPLAY_SLOTS` | off / unset | the prefill graph replay / the replay slot count. |
| `MLX_GRAPH_FREE_LAG` / `MLX_GRAPH_DEFER_MAX_MB` / `MLX_GRAPH_SPLIT_LOG` | unset | the free lag / the deferred-free cap / the split log. |
| the `MLX_G1_*` class (`BISECT`, `BISECT_MS`, `DEBUG`, `EXECDUMP`, `EXECDUMP_N`, `FIRST_ERR`, `H2DLOG`, `HIPLOG`, `IDXDBG`, `KVAPP_DBG`, `MEMDUMP`, `NODUMP`, `NODUMP_N`, `NO_COMMIT`, `NO_LAUNCH`, `NO_PREH2D`, `OPDUMP`, `SENTINEL`, `SDPA_DBG`, `TRACE`, `TRACE_OPS`, `KSL`) | off | the G1 debug/diagnostic class (each: set = enable the named diagnostic; unset = off). The G1 lane is retired — these ride the retired engine. |
| `MLX_G1X_OFF` / `MLX_G1Y_DEBUG` | off | set = the G1 x-off / y debug. |
| `MLX_SAVE_CUDA_GRAPHS_DOT_FILE` | unset | the graph DOT-file dump (the inherited CUDA-era name). |

## Platform / distributed (inherited)

| Flag | Default | Effect |
|---|---|---|
| `ROCM_HOME` / `ROCM_PATH` | unset | the ROCm prefix for the JIT (`jit_module.cpp`). |
| `CUDA_HOME` / `CUDA_PATH` | unset | the CUDA prefix (the inherited CUDA JIT path). |
| `MLX_HSACO_CACHE_DIR` / `MLX_PTX_CACHE_DIR` | unset | the HSACO / PTX cache directory. |
| `MLX_DISABLE_COMPILE` | off | set = disable the compile step. |
| `MLX_RANK` / `MLX_HOSTFILE` / `MLX_RING_VERBOSE` | unset | the distributed ring (rank / hostfile / verbose). |
| `MLX_MPI_LIBNAME` | unset | the MPI library name. |
| `JACCL_IBV_DEVICES` / `JACCL_COORDINATOR` / `JACCL_RANK` / `JACCL_RING` | unset | the JACCL (the distributed IB transport; the `MLX_*` fallbacks are read as second names). |
| `NCCL_DEBUG` | unset | the NCCL debug level. |
| `DEVICE` / `LOW_MEMORY` | unset | the python test-harness device / low-memory gates. |
| `GGML_CUDA_DISABLE_GRAPHS` / `GGML_CUDA_PDL` / `GGML_CUDA_MMQ_PREC` | unset | the inherited llama.cpp-era mmq names (graphs off / PDL / the mmq precision). |

## Smoke / serve / harness

| Flag | Default | Effect |
|---|---|---|
| `MLX_ROCM_SMOKE_LLM` | off | `1` = the Phase 0 smoke runs the 0.6B generate leg (`scripts/smoke.py`). |
| `MLX_ROCM_SMOKE_MODEL` | `Qwen/Qwen3-0.6B` | the smoke model ref. |
| `QMLX_SERVE_BIN` / `QMLX_SERVE_MODEL` | required | the qmlx-serve entrypoint / the 9B-class model ref (`scripts/smoke-serve.sh`). |
| `QMLX_SERVE_PORT` | `8080` | the serve port. |
| `MODELS` | `/models` | the model root (every bench / script). |
| `M8_RESTORE_DTYPE` | `f32` | the state-restore dtype (the M8 audit probes). |
| `DIAG_MAX` | `160` | the diagnostic max. |
| the `P2_*` class (`P2_MODE`, `P2_N`, `P2_WARM`, `P2_SKIP`, `P2_PROMPT_TOKENS`, `P2_MAX_TOKENS`, `P2_CHUNK`, `P2_STATE_OUT`, `P2_STATE_STEPS`, `P2_STREAM_OUT`, `P2B_DEBUG`, `P2B_STEPS`) | per-harness | the P2 (state-diff / stream) probe harness controls. |
| the `T14_*` / `T29K_*` class (`T14_MAX_TOKENS`, `T14_WARM`, `T14_SKIP`, `T29K_TARGET`, `T29K_CHUNK`) | per-harness | the 14-token / 29K warm-harness controls (`bench/warm-mtp-bench.py`). |
| the `QMV_*` class (`QMV_M`, `QMV_N`, `QMV_K`, `QMV_REPS`, `QMV_WARMUP`) | `1` / `12288` / `4096` / `50` / `10` | the QMV micro-bench shape / reps / warmup. |
| `ROCPROF` / `ROCPROF_V3` / `ROCTX` / `ROCTX_SO` / `SPAN` / `SPAN_SO` | unset / per-harness | the profiler plumbing (the rocprofv3 / SDK-injection harness, `scripts/gpu-util.sh`). |
| `RUN_789_EXPECTED_WHEEL_SHA` | `75ca881` | the expected wheel sha (the run-789 gate). |
| `SPEC` / `VENV_ID` | unset | the harness spec / venv id. |
