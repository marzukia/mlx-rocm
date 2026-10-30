# mlx-rocm: minimum work to build an ROCm backend for MLX

**Operator order 2026-10-25.** Investigate `mlx-cuda` (Apple's CUDA backend for MLX) and
determine the minimum work to create an equivalent ROCm backend (`mlx-rocm`). No AMD GPU on
the host, so this is source-reading + web.

**End use:** run the fleet's qMLX inference stack (qmlx-serve, Qwen3.x serving, MTP spec
decode, prefix/disk KV cache, mlx-lm) on **the build VM** — AMD RX 9070 XT + Ryzen 9700X,
ROCm 10.1 (verified guest toolchain; 7.7 was the original target).

**ROCm version note (2026-10-26, supersedes the 6.4/7.7 references below):** the verified
guest toolchain on the build VM is **ROCm 10.1** (core at `/opt/rocm/core-10.1`,
`CMAKE_PREFIX_PATH=/opt/rocm/core-10.1`, `amdrocm-*-dev10.1` packages; see
docs/quickstart.md). Nothing else is verified: 6.4 was the early probe validation and
7.7 the original target. Where this report says "on ROCm 7.7" for the box, read 10.1.

**Method:** read `ml-explore/mlx` (main @ `56c2bf9`, v0.32.5), the two named prior-art repos,
plus two additional MLX-ROCm forks discovered during the investigation (the single most
important finding — see §3). Every architecture claim is cited to `file:line`.

---

## TL;DR

The single most important finding: **the work is ~90% done and already in an open upstream
PR.** `ml-explore/mlx` PR **a community MLX ROCm backend PR ROCm backend** (head
`the vendor ref`, open since 2025-06-16, last updated 2026-08-14, closes issue
#2556 "Add ROCm Support for AMD GPUs") contains a complete `mlx/backend/rocm/` that mirrors
the CUDA backend file-for-file. Two independent downstream projects already build and run on
top of it (a C++ LLM engine on Strix Halo, and an Ollama fork on an RX 9060 XT).

So "minimum work to create mlx-rocm" is **not** "port the CUDA backend from scratch." It is:

1. **Adopt** the `the vendor ref` tree (or the newer `the vendor ref`
   forward-merge, which adds GatedDeltaNet + mxfp8 + gfx1200 fixes) as the base.
2. **Build** the `mlx` wheel with `-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`.
3. **Validate on gfx1201** (RX 9070 XT) — the fork is tested mostly on Strix Halo APUs
   (gfx1150/1151) and RX 9060 (gfx1200); gfx1201 has one known fp8 GEMM driver bug.
4. **Smoke-test qmlx-serve** with a 9B-class dense Qwen3 (the "minimum" target).

That is a **~2.5–5 person-week** effort for the minimum (9B dense on gfx1201). The full
qMLX 122B hybrid stack (GatedDeltaNet, MoE, MTP, disk KV) is an additional ~2–4 person-weeks
and is mostly "validate + fix known issues," not "write new kernels."

A critical naming correction up front: **the RX 9070 XT is `gfx1201`, not `gfx110x`** (see §4).

---

## 1. mlx-cuda architecture (what we are mirroring)

The CUDA backend lives at `mlx/backend/cuda/` in `ml-explore/mlx` @ `56c2bf9`. Its structure
is the template the ROCm backend must match. Key abstractions:

### 1.1 Device / Worker / CommandEncoder (the CUDA-graph-capture model)

- `cuda/device.h` — `cu::Device` (one per physical GPU), owns a `Worker` per stream and a
  `CommandEncoder` per stream. `DeviceType` in `mlx/device.h:14` is `{cpu, gpu}`; **both
  Metal and CUDA map to `Device::gpu`**. This is why a ROCm backend can drop in without
  touching the public API: it just becomes *the* `gpu` device at build time.
- `cuda/worker.h` — a `Worker` owns the CUDA stream + event pool for one MLX `Stream`.
- `cuda/device.h` `CommandEncoder` — the heart of the design. It batches a sequence of
  kernel launches into a **CUDA graph** (`cudaGraph_t` → `cudaGraphExec_t`), with an LRU
  cache of execs keyed by the op topology. On `commit()` it either replays a cached exec or
  instantiates a new one. This is how MLX gets low per-op launch overhead on the GPU.
- `cuda/cuda_utils.h` — RAII wrappers: `CudaHandle<T>` (create/destroy with error checking),
  `CudaGraph`, `CudaGraphExec`, `CudaStream`.

The ROCm port mirrors this exactly: `rocm/device.h` has a `CommandEncoder` with
`hipGraph_t build_graph_`, an `LRUCache<hipGraphExec_t> graph_cache_{400}`, a per-topology
`ExecSlot` pool for `hipGraphExecUpdate` reuse, and a "build-once decode-graph replay" path
(`decode_capture_begin/end_record/replay`). The comments in `rocm/device.h` explicitly note
they are "mirroring CUDA CommandEncoder" and call out a **ROCm CLR bug **: the
runtime stores a kernel node's `kernelParams` **by pointer, not a deep copy** (CUDA
deep-copies), so the arg storage must outlive the exec. That is the single biggest
ROCm-specific correctness hazard in the whole backend.

### 1.2 The backend-agnostic shim (the integration point)

`mlx/backend/gpu/` is the **backend-agnostic** layer. It exposes `init`, `new_stream`,
`eval`, `finalize`, `synchronize`, `clear_streams`, `is_available`, `device_count`,
`device_info`. A new GPU backend plugs in here. `gpu/eval.cpp` does
`make_current()` → `primitive().eval_gpu(...)` → add temporaries → `commit()` the encoder.
`gpu/primitives.cpp` is where `MLX_USE_ROCM` / `MLX_USE_CUDA` selects the backend at compile
time. **This is the only "seam" a ROCm backend needs** — everything above it (scheduler,
streams, the Python `mx.gpu`) is backend-agnostic.

### 1.3 Op dispatch and the NO_GPU pattern

Every primitive has three entry points: `eval_gpu`, `use_fallback(Stream)`, and `eval_cpu`.
Ops that a backend has not implemented are marked with a `NO_GPU(Name)` macro that makes
`eval_gpu` throw and `use_fallback` return true, so the op runs on the CPU fallback. The CUDA
backend's `NO_GPU` set is: `LUF, QRF, SVD, Inverse, Eig, Eigh, GatedDeltaUpdate, Send, Recv`.

**Important for the qMLX workload:** `GatedDeltaUpdate` (the DeltaNet / linear-attention
primitive, ~75% of the layers in Qwen3.5-122B-A10B) is `NO_GPU` on CUDA. It works via a
**backend-agnostic per-timestep fallback** in `mlx/fast.cpp:1078-1204` — a `for t in range(T)`
loop that decomposes the update into standard ops (multiply/sum/subtract/add/where/astype).
The ROCm fork takes the same path (`rocm/primitives.cpp:59-72`, `use_fallback` always true).
So GatedDeltaNet **runs on ROCm out of the box**, but as a per-timestep loop: fine for
decode (T=1), slow for long prefill (T=4096 → ~32k small kernel launches). This is the main
performance gap for the 122B hybrid model, not a correctness gap.

### 1.4 Build system

`mlx/backend/cuda/CMakeLists.txt` — gated on `MLX_BUILD_CUDA`, finds `CUDAToolkit` (with an
explicit `CUDA 13.1 is not supported` guard), `cuDNN`, `CUTLASS`. Uses `--extended-lambda`
and **NVRTC JIT**: 17 device headers are embedded via `bin2h` and JIT-compiled at runtime
with a disk cache at `/tmp/mlx/<version>/ptx`. Deps: CCCL v3.1.3, NVTX, cuBLASLt, cuFFT,
cuSOLVER, cuDNN frontend v1.16.0, CUTLASS v4.4.2. Windows WDDM + delay-load handling.

### 1.5 The heavy kernels

- **Matmul** (`cuda/matmul.cpp`): GEMV or `CublasGemm` (cuBLASLt) for dense;
  `BlockMaskedMM` / `AddMM` / `GatherMM` (CUTLASS) for the sparse/MoE paths; `SegmentedMM`.
- **QMM** (`cuda/quantized/quantized.cpp`): a cascade of `supports_*` gates →
  `qmm_sm90` (Hopper TMA/CUTLASS) → `qmm_sm80` (Ampere CUTE) → `qmm_naive` → `qmv` → `fp_qmv`.
- **SDPA** (`cuda/scaled_dot_product_attention.cpp`): cuDNN for prefill + a custom
  `sdpa_vector` 1/2-pass kernel for decode.
- **Allocator** (`cuda/allocator.cpp`): managed/unified memory primary, a small pool, a buffer
  cache, a memory pool.
- **Custom kernels** (`cuda/custom_kernel.cpp`): `mlx.core.fast` fused kernels compiled via
  NVRTC using `__grid_constant__` + `cooperative_groups`.

### 1.6 The CUDA API surface (what must be translated)

Enumerated from the CUDA backend: `cudaMalloc/Free/Memcpy/Memset`, `cudaStream*`,
`cudaEvent*`, `cudaGraph*` (the full graph API), `cudaLaunchKernel`, `cuBLASLt*`,
`cuFFT*`, `cuSOLVER*`, `nvrtc*` (JIT), `cuda::std::` / `thrust` (CCCL), `cooperative_groups`,
`__grid_constant__` (×130 uses), `__shared__`, PTX instructions (mov, mul.f32x2, cvt incl.
e2m1x4 FP4, cp.async, cp.async.bulk.tensor TMA, mbarrier, mma.sync, ldmatrix, bfe), and a
custom `steel/` GEMM with raw `mma.sync` PTX.

---

## 2. CUDA → ROCm API map (for exactly what mlx-cuda uses)

The `the vendor ref` tree is the de-facto reference translation. It maps the CUDA
surface in §1.6 as follows. `rocm/CMakeLists.txt` finds: `hip`, `rocblas`, `rocthrust`,
`rocprim`, `hiprand`, and (conditionally) `rocwmma`.

| CUDA (mlx-cuda) | ROCm (mlx-rocm fork) | Notes / fidelity |
|---|---|---|
| `cudaMalloc/Free/Memcpy/Memset` | `hipMalloc/Free/Memcpy/Memset` | 1:1. No CUDA-style *managed* memory on discrete GPUs — see memory row. |
| `cudaStream*` / `cudaEvent*` | `hipStream*` / `hipEvent*` | 1:1. |
| `cudaGraph*` (capture + exec + update) | `hipGraph*` (capture + exec + update) | 1:1 API, **but** ROCm CLR stores kernel-node `kernelParams` by pointer (not deep-copied) — the fork's `ExecSlot` pool exists specifically to keep arg storage alive (`rocm/device.h`). |
| `cudaLaunchKernel` | `hipLaunchKernel(GGL)` | 1:1. Wavefront size differs (32 RDNA vs 64 CDNA) — the fork queries `hipDeviceProp::warpSize` at runtime and never pins a compile-time constant (`rocm/device.h` `warp_size()`, `rocm/CMakeLists.txt` warp policy). |
| `cuBLASLt` (dense GEMM) | `hipBLASLt` | Near 1:1. The fork adds a `GemmPrecision` capability table (`Bf16, Fp8E4M3, Fp8E5M2, Int8`, `gemms/hipblaslt_gemm.cpp:249`) and a persistent GEMM pipeline (cached layouts + heuristics). |
| (general GEMM fallback) | `rocBLAS` | The fork keeps a `rocblas_gemm.cpp` path for shapes/precisions hipBLASLt doesn't cover. |
| `NVRTC` (JIT fused kernels) | `hipRTC` | 1:1. `rocm/jit_module.cpp` wraps `hipRTC`; it even silences `comgr`'s unconditional stderr flood. `__grid_constant__` has **no ROCm equivalent** — the fork drops it and passes params normally. |
| `cuFFT` | *(not ported — `NO_GPU(FFT)`)* | `rocFFT`/`hipFFT` exist in ROCm but the fork does not use them. FFT is a gap. |
| `cuSOLVER` (LAPACK) | *(not ported — `NO_GPU(LUF/QRF/SVD/Inverse/Cholesky/Eig/Eigh)`)* | `rocSOLVER` exists but is unused. All LAPACK ops are a gap (CPU fallback). |
| `cuDNN` (SDPA prefill) | **custom `rocWMMA` flash attention** | The fork does *not* use a cuDNN equivalent (no MIOne/AOTIT). It writes its own `flash_attention_wmma.hip` (forward) + `flash_attention_bwd_wmma.hip` (backward) using `rocwmma` matrix-core tiles, plus the ported `sdpa_vector` for decode. Cleaner dependency-wise than CUDA, more kernel work. |
| `CUTLASS` (BlockMaskedMM/AddMM/GatherMM) | **custom WMMA + hipBLASLt** | No CUTLASS-on-ROCm. `BlockMaskedMM` is `NO_GPU`; `GatherMM`/`GatherQMM` are implemented via custom kernels + hipBLASLt in `matmul.cpp`/`quantized/qmm.hip`. |
| `CCCL` (`cuda::std::`, `thrust`) | `rocThrust` | 1:1. |
| `CUB` | `rocPRIM` | 1:1. |
| `curand` | `hipRAND` | 1:1. |
| `NCCL` (multi-GPU) | **custom P2P all-reduce** | No RCCL. `rocm/distributed.hip` implements AllReduce/AllGather/ReduceScatter/Send/Recv with P2P + custom reduce kernels. |
| `cooperative_groups` | *(not used)* | Dropped. |
| `__shared__` (LDS) | `__shared__` (LDS) | Same keyword; LDS size differs per arch (queried at runtime). |
| `mma.sync` PTX (Ampere/Hopper) | `rocwmma` fragment API | The fork's `qmm_wmma_dense_kernel` (`quantized/qmm.hip:3147`) uses `rocwmma::fragment<...>` + `load_matrix_sync`/`mma` — the matrix-core path for RDNA3+/CDNA. |

### 2.1 The two ROCm-specific hazards the fork had to engineer around

1. **Wavefront 32 vs 64.** RDNA is wave32, CDNA is wave64. A multi-arch fatbin that pins the
   host `WARP_SIZE` to 64 mis-launches on RDNA (the fork hit "garbage decode" from this —
   `rocm/CMakeLists.txt` warp policy, `rocm/device.cpp` runtime `warp_size()`). RDNA4
   (gfx1200/1201) supports *both* 32 and 64; the fork defaults RDNA to 32.
2. **No clean CUDA-managed-memory equivalent.** On a discrete GPU (RX 9070 XT), `hipMalloc`
   returns real VRAM that is *mapped into the host address space through a large PCIe BAR*.
   Reading it from the CPU runs at uncached-MMIO speed. The fork measures this on gfx942:
   **CPU direct BAR read 54 MB/s vs hipMemcpy 50.4 GB/s** (`rocm/host_stage.cpp`). It adds a
   `host_stage.cpp` pinned-bounce path (64 MB, ~36.9 GB/s pinned) for host reads of device
   memory (model save/load) and uses `host_shadow` instead of CUDA's
   `move_to_unified_memory` for host-accessed buffers. The async `hipMallocAsync` pool is
   **off by default** on ROCm (it caused gfx page faults / IMA 700 on MI300X —
   `rocm/allocator.cpp`).

These two are the reasons a naive `cuda*→hip*` sed is not enough, and they are the parts to
test hardest on the discrete RX 9070 XT.

---

## 3. Prior art

### 3.1 THE one: `ml-explore/mlx` a community MLX ROCm backend PR (head `the vendor ref`)

This is the real, ongoing, upstream MLX→ROCm port. Facts:

- **Open PR** `an open community MLX ROCm backend PR` "[Experiment] ROCm backend," created **2025-06-16**,
  last updated **2026-08-14**, head `the vendor ref`, **closes #2556** ("Add ROCm
  Support for AMD GPUs"). ~16 months of work.
- **Complete `mlx/backend/rocm/`** mirroring the CUDA backend: `allocator`, `device`/
  `worker`/`CommandEncoder` (with HIP-graph capture + build-once decode replay), every
  primitive `.hip` port (`arange`, `arg_reduce`, `binary`, `copy`, `reduce`, `random`,
  `softmax`, `sort`, `scan`, `rope`, `layer_norm`, `rms_norm`, `indexing`, `distributed`,
  `event`, …), `quantized/` (qmm 7386 lines + qmv + affine/fp quantize + convert_fp8),
  `gemms/` (hipblaslt + rocblas + naive + gemv), `conv/`, `flash_attention` (vector +
  WMMA fwd + WMMA bwd), `moe_swiglu`, `jit_module` (hipRTC), `custom_kernel`, `host_stage`,
  `load`, `device_info`, `no_rocm` stub.
- **Per-arch tier tuning** in `quantized/qmm.hip:578-615`: detects `Rdna4` (gfx1200/1201),
  `Rdna35` (gfx1150/1151/1152), `Rdna3` (gfx11xx), `Rdna2` (gfx10xx), `Cdna` (gfx9xx) from
  `hipDeviceProp::gcnArchName`, and tunes QMV/QMM crossover thresholds + a WMMA allowlist
  (`qmm.hip:620-631` includes gfx1200/gfx1201).
- **Multiple contributors** (Geramy, goniz, lemonade-sdk, antmikinka, soloish90, a contributor to the ROCm backend).
  PR history: #1 device info + gfx1151 (2026-02-03) → #5 "WMMA prefill, hipBLASLt, 4-bit MoE
  dispatch, QMV tuning, iGPU allocator" (2026-03-27) → #11 JIT INFINITY/NAN fix (2026-07-14)
  → #13 dual-load SEGV investigation (2026-07-19).
- **CI** (`.github/workflows/build_rocm.yml`): builds the `mlx` wheel on a **self-hosted
  `strix-halo` runner** with `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DMLX_ROCM_ARCHITECTURES=gfx1151"`,
  then runs basic GPU tests + `mlx_lm.generate` (Qwen3-0.6B bf16/8bit) under lldb.
- **Known open issue** (`docs/rocm-investigation-2026-07-19/RESULTS.md`): **dual large-model
  process load → SIGSEGV** (exit 139) on a unified-memory APU (gfx1150) — a resource
  overcommit / ungraceful-OOM in the allocator/HIP, *not* a single-stream kernel bug.
  Exclusive single-process loads pass. **For the build VM (a single discrete GPU, single
  serving process) this is not the blocking case**, but it is the one to watch if you ever
  run two models.
- **Packaging:** `setup.py` has no `[rocm]` extra yet — the wheel is just the standard `mlx`
  wheel built with `-DMLX_BUILD_ROCM=ON`. No separate `mlx-rocm` package, no bundled
  ROCm deps (the system ROCm install provides them). Simpler than the CUDA wheel model.

**Verdict:** this is the base to adopt. It is not "done" (it's an open experiment, tested
mainly on Strix Halo APUs + gfx1200), but it is a complete, actively-maintained,
file-for-file CUDA mirror with per-arch tuning already in place.

### 3.2 `lemonade-sdk/lemon-mlx-engine` (a.k.a. `bogdan-d/bleeding-edge`)

- A **pure-C++ LLM inference engine** (62 stars) that builds *on top of* the fork via
  `FetchContent(the ROCm backend fork @ the vendor ref)` — `lemon-mlx-engine/CMakeLists.txt`.
- Provides per-model definitions (Qwen3, Qwen3.5, Qwen3-MoE, Qwen3-Next/GDN, MTP, VLMs) and
  the `chat`/`server`/`diagnose` binaries. The engine itself is **closed-source** (binary
  releases); the model defs + tests are in the repo.
- **Benchmarks (Strix Halo gfx1151, 128 GB unified):** Qwen3-0.6B 4bit **151.2 tok/s**,
  8B 20.8 tok/s; 29–85% faster than vLLM-ROCm. Built with **TheRock 7.13** from source,
  native Tensile (55 native Tensile kernels), FP16 GEMM 32.97 TFLOPS, wave32 fused ternary
  kernel for BitNet.
- **Relevance:** proves the fork runs real LLM serving end-to-end (including GatedDeltaNet +
  MoE + MTP model defs) on ROCm, and gives a working reference for the model-side code.

### 3.3 `the Ollama ROCm-MLX integration`

- An **Ollama fork**: GGUF (llama.cpp ROCm `rocm_v7_2`) + MLX (`*-mlx`, `rocm_v10`) in one
  `ollama serve`. **Tested on RX 9060 XT (gfx1200, RDNA4), ROCm 10.**
- Uses a **forward-merged** copy of the fork: `a maintainer of the ROCm backend/mlx` branch `the vendor ref`
  (forward-merge of a community MLX ROCm backend PR onto the MLX commit Ollama pins, `59d600b5`). **Actively
  maintained — last commit 2026-09-28** ("mxfp8 prefill GEMM, cooperative softmax,
  device_info keys"; "ROCm 10 gfx1200 fixes"; "add ROCm GatedDeltaUpdate fallback").
- **hipRTC custom kernels:** routes `fast::cuda_kernel` to the ROCm backend so MLX's fused
  kernels (GatedDeltaNet, depthwise conv, mamba2) run on ROCm. GatedDelta kernels are
  **bit-exact vs the graph reference** (`go test ./mlx/`).
- **Perf:** fp8 GEMV with 16-row weight reuse (~5× prefill at M>1, ~150 GB/s decode);
  `qwen3.5:0.8b-mlx (fp8)` ~104 tok/s.
- **Relevance:** this is the closest thing to "mlx-rocm already shipping on a discrete RDNA4
  dGPU." It auto-detects the GPU's gfx target via `rocm_agent_enumerator` and is the
  template for a turnkey build/install.

**Bottom line on prior art:** three independent groups have converged on the same base
(`the vendor ref`) and all three run it on RDNA. The risk is not "can MLX run on
ROCm" (demonstrated) — it is "does it run *well* on gfx1201 specifically, under the qMLX
workload, on ROCm 10.1 (the verified guest toolchain)."

---

## 4. GFX targets (RDNA4 vs CDNA3, ROCm 7.x coverage)

### 4.1 Naming correction (important)

The task context says "RX 9070 XT (RDNA4, gfx110x)." **That is not what ROCm calls it.**
Per the official ROCm GPU spec table (ROCm 10.1.0) and the compatibility matrix:

| GPU | Arch | LLVM target | VRAM | CUs | Wavefront | LDS |
|---|---|---|---|---|---|---|
| **Radeon RX 9070 XT** | RDNA4 | **gfx1201** | 16 GB | 64 | 32 or 64 | 128 KiB |
| Radeon RX 9070 | RDNA4 | gfx1201 | 16 GB | 56 | 32 or 64 | 128 KiB |
| Radeon RX 9060 XT | RDNA4 | gfx1200 | 16 GB | 32 | 32 or 64 | 128 KiB |
| Radeon RX 7900 XTX | RDNA3 | gfx1100 | 24 GB | 96 | 32 or 64 | 128 KiB |
| Radeon RX 7800 XT | RDNA3 | gfx1101 | 16 GB | 60 | 32 or 64 | 128 KiB |
| Radeon RX 7600 | RDNA3 | gfx1102 | 8 GB | 32 | 32 or 64 | 128 KiB |
| Ryzen AI Max (Strix Halo) | RDNA3.5 | gfx1151 | dynamic | 40 | 32 or 64 | 128 KiB |
| **Instinct MI300X** | CDNA3 | **gfx942** | 192 GB | 304 | 64 | 64 KiB |
| Instinct MI300A | CDNA3 | gfx942 | 128 GB | 228 | 64 | 64 KiB |

So: **`gfx110x` is the RDNA3 / RX 7000 series. The RX 9070 XT (RDNA4) is `gfx1201`.**
Both prior-art repos agree (bleeding-edge: "gfx120X = rdna4"; the fork's CMake: "RDNA4:
gfx1200, gfx1201"). **the build VM's target arch is `gfx1201`.** (If the box's
`rocm_agent_enumerator`/`rocminfo` reports something else, trust the tool — but expect
gfx1201.)

### 4.2 ROCm coverage for gfx1201 (verified: 10.1)

- **RDNA4 (gfx1200/gfx1201) support landed in ROCm 6.4.0** (release notes: "Support for
  gfx1201"). It is carried through 7.x and 10.x. **ROCm 10.1 (the verified guest
  toolchain on the build VM) supports gfx1201.**
- The fork's `CMakeLists.txt` arch list explicitly targets "ROCm 6.4.0 – 7.2.0 compatibility
  matrix" and includes gfx1200/gfx1201; the fork's CI runs TheRock 7.13 and the Ollama fork
  runs ROCm 10 — so gfx1201 is a first-class target across the 7.x line.
- **One known gfx1201-specific issue:** the fork's `gemms/hipblaslt_gemm.cpp:253-270` notes
  that **fp8 e4m3 activation GEMM "may be kernel-buggy on some RDNA4 drivers — long-context
  prefill on gfx1201 garbles,"** with a `MLX_ROCM_GEMM_BF16=1` env to force bf16. Plan to
  test fp8 prefill on the real box and keep the bf16 fallback.

### 4.3 What the fork already does for gfx1201 (RDNA4)

- `quantized/qmm.hip:590-597` — `Rdna4` arch tier (2 SIMDs/CU) with its own QMV/QMM tuning.
- `quantized/qmm.hip:620-631` — WMMA allowlist **includes gfx1200/gfx1201** → the dense
  fused WMMA quantized GEMM (`qmm_wmma_dense_kernel`, q4/q6/q8 affine) is available on the
  RX 9070 XT.
- `rocm/CMakeLists.txt:22-23` — default arch list includes gfx1200/gfx1201.
- `rocm/CMakeLists.txt` — RDNA4 stays wave32 (with `MLX_ROCM_FORCE_WARP_SIZE` override).
- `gemms/hipblaslt_gemm.cpp` — fp8 e4m3 GEMM capability probe + bf16 fallback.

**CDNA3 (gfx942 / MI300)** is also a first-class target (wave64, MFMA via the same `rocwmma`
fragment API, clean bf16 MFMA + hipBLASLt pointer-offset GEMM per `device.h`
`supports_cdna_mfma_gemm()`). So a single build can target both `gfx1201;gfx942` if you ever
want the same wheel on a MI300 box.

---

## 5. Name check (GitHub / PyPI collisions)

- **PyPI `mlx-rocm`:** already **claimed at `0.0.0`** (1 release, 1 file — a placeholder
  wheel, no summary/project URLs). That is *our* slot. No other owner. `mlx_rocm`
  normalizes to the same project.
- **GitHub `mlx-rocm`:** **no repository, organization, or user by that exact name** (all
  404). The name is free.
- **Closest existing names** (not collisions, but do not confuse them):
  `the Ollama ROCm-MLX integration` (the Ollama fork), `bogdan-d/bleeding-edge` (halo-ai CORE),
  `the ROCm backend fork` + `a maintainer of the ROCm backend/mlx` (the forks). None is named `mlx-rocm`.

**Verdict:** the name `mlx-rocm` is clean on both GitHub and PyPI (our own placeholder
holds the PyPI slot). No renaming needed.

---

## 6. Minimum `mlx-rocm` definition (phased plan + effort)

"Minimum" = **qmlx-serve runs a 9B-class dense Qwen3 on the build VM (RX 9070 XT / gfx1201,
ROCm 10.1), correctly, at usable decode speed.** MTP, disk KV, and 27B+/122B-hybrid are
deferred to Phase 2.

### Phase 0 — Adopt + build + smoke (0.5–1 person-week)

1. Base on `the vendor ref` (or `the vendor ref` — newer, adds
   GatedDeltaNet + mxfp8 + gfx1200 fixes; prefer it if it builds clean on 10.1).
2. Build the `mlx` wheel:
   `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS"`.
   (No `setup.py` change needed — the fork builds the standard `mlx` wheel; the ROCm system
   install provides hip/rocblas/rocprim/hiprand/hiprtc/hipblaslt/rocwmma.)
3. On the build VM: `python -c "import mlx.core as mx; print(mx.rocm.is_available())"`,
   `mx.set_default_device(mx.gpu)`, then the fork's basic tests (matmul, softmax) and
   `mlx_lm.generate` on Qwen3-0.6B (bf16 + 8bit).
   **Exit criteria:** `is_available()==True`, matmul/softmax correct, 0.6B generates sane
   text.

### Phase 1 — Validate gfx1201 + qmlx-serve minimum (1–2 person-weeks)

1. **QMM (4bit) on gfx1201** — the `Rdna4` tier + WMMA dense qmm. Verify 4bit Qwen3
   (e.g. Qwen3-8B/9B-4bit) numerics vs the CPU fallback.
2. **SDPA decode (vector) + prefill (WMMA flash) on gfx1201** — verify attention numerics
   and that prefill actually uses the WMMA path (not the slow vector path).
3. **fp8 GEMM** — test the known gfx1201 fp8 prefill bug; keep `MLX_ROCM_GEMM_BF16=1` as
   the safe default until proven clean.
4. **Allocator on a discrete GPU** — confirm the `host_shadow` / pinned `host_stage` path
   (not the APU fine-grained path) is what runs on the RX 9070 XT; confirm async-pool-off
   default doesn't page-fault.
5. **Run qmlx-serve** with a 9B-class dense Qwen3. qmlx uses the generic `mx.gpu` /
   `mx.default_device()` / `mx.new_thread_local_stream()` API (not a `mlx.core.cuda`-specific
   API), so it should work unmodified once the backend is built in — but verify the
   `is_metal` error-classification path in `vllm_mlx/engine_core.py:675` doesn't mis-handle
   ROCm runtime errors (add a `rocm`/`hip` needle).
   **Exit criteria:** qmlx-serve serves a 9B Qwen3 on gfx1201; decode tok/s is in the same
   ballpark as the Ollama fork's ~104 tok/s on the 0.8B (scaled); no NaNs, no OOM→SEGV on
   a single process.

### Phase 2 — Full qMLX stack (deferred; ~2–4 person-weeks)

1. **GatedDeltaNet (122B hybrid):** works via the per-timestep fallback (Phase-0 code), but
   long prefill is slow. Option A: accept it (qMLX's disk-KV makes prefill rare). Option B:
   port the Ollama fork's **bit-exact GatedDelta hipRTC kernel** (it exists and is tested).
2. **MoE (A10B / 27B+):** `GatherQMM` + `moe_swiglu_sorted` are already in the fork
   (`moe_swiglu.cpp`, `quantized/qmm.hip`). Validate on a MoE Qwen3.
3. **MTP spec decode:** known open issue **M3** — `There is no Stream(cpu, N)` in
   `mlx/backend/cpu/encoder.cpp` (stream TLS / MTP thread hygiene). Must be fixed before MTP
   is productized (the fork deferred it).
4. **Disk KV cache:** validate `Load`/`Save` through `host_stage` (pinned bounce, ~36.9
   GB/s) — this is qMLX's core value prop and the BAR-memory path is the part most likely
   to be slow on a discrete GPU.
5. **Dual-model / OOM→SEGV** (the fork's RESULTS.md issue): only relevant if you run two
   large models concurrently; not a blocker for single-process serving.

### Effort summary

| Scope | Effort | Risk |
|---|---|---|
| **Minimum: 9B dense Qwen3 on gfx1201 (Phase 0+1)** | **2.5–5 person-weeks** | Low-medium. The backend exists; the risk is gfx1201-specific validation (fp8 GEMM, allocator, SDPA path) on ROCm 10.1. |
| Full qMLX 122B hybrid + MTP + disk KV (Phase 2) | +2–4 person-weeks | Medium. GatedDelta prefill perf + the M3 stream-TLS fix are the real work. |
| (For contrast) Porting the CUDA backend from scratch | ~15–25 person-weeks | What we are *not* doing. |

### Open questions (for the operator)

1. **Base choice:** `the vendor ref` (the the upstream base head, TheRock 7.13-tested)
   vs `the vendor ref` (newer, ROCm 10 + gfx1200 fixes, GatedDeltaNet). Need to
   confirm which builds clean on **ROCm 10.1** specifically (the fork's CI is 7.13, Ollama is
   10).
2. **Wheel vs source:** ship a real `mlx-rocm` PyPI wheel (needs a `setup.py` `[rocm]` extra
   + a CI matrix), or build-from-source on the box (simpler, what the fork does today)?
3. **fp8 default:** is gfx1201 fp8 prefill clean on ROCm 10.1's driver, or do we ship
   `MLX_ROCM_GEMM_BF16=1` by default?
4. **MTP:** is MTP in the minimum, or strictly Phase 2? (It is blocked on the M3 stream-TLS
   fix either way.)

---

## Appendix A — Key file:line index

**Upstream `ml-explore/mlx` @ `56c2bf9` (v0.32.5):**
- `mlx/device.h:14` — `DeviceType {cpu, gpu}` (Metal & CUDA both `gpu`).
- `mlx/backend/cuda/` — the CUDA backend (the template).
- `mlx/backend/gpu/` — backend-agnostic shim (the integration seam).
- `mlx/fast.cpp:1078-1204` — `gated_delta_update` + per-timestep fallback.

**`the vendor ref` (a community MLX ROCm backend PR head):**
- `mlx/backend/rocm/CMakeLists.txt:22-23` — arch list incl. gfx1200/gfx1201; warp policy.
- `mlx/backend/rocm/device.h` — `CommandEncoder` (HIP-graph capture, `ExecSlot` pool,
  build-once decode replay); CLR#138 kernelParams-by-pointer note.
- `mlx/backend/rocm/allocator.cpp` — BufferCache + hipMalloc + 8-byte SmallSizePool;
  async-pool-off default; `host_shadow` vs fine-grained.
- `mlx/backend/rocm/host_stage.cpp` — pinned-bounce host reads (BAR 54 MB/s vs 50.4 GB/s).
- `mlx/backend/rocm/jit_module.cpp` — hipRTC JIT + comgr stderr silencing.
- `mlx/backend/rocm/quantized/qmm.hip:578-615` — per-arch tier (Rdna4/35/3/2/Cdna).
- `mlx/backend/rocm/quantized/qmm.hip:620-631` — WMMA allowlist (incl. gfx1200/1201).
- `mlx/backend/rocm/quantized/qmm.hip:3147-3187` — `qmm_wmma_dense_kernel`.
- `mlx/backend/rocm/gemms/hipblaslt_gemm.cpp:249-270` — `GemmPrecision` + fp8 gfx1201 bug.
- `mlx/backend/rocm/scaled_dot_product_attention.cpp` — vector (decode) + flash + WMMA
  (prefill) + WMMA bwd.
- `mlx/backend/rocm/primitives.cpp:31-42` — NO_GPU set (FFT, LAPACK, BlockMaskedMM, …).
- `mlx/backend/rocm/moe_swiglu.cpp` — fused sorted-MoE SwiGLU.
- `mlx/backend/rocm/distributed.hip` — custom P2P all-reduce (no RCCL).
- `python/src/rocm.cpp` — `mlx.core.rocm` module (`is_available`, train arena, moe_swiglu).
- `.github/workflows/build_rocm.yml` — CI on `strix-halo`, wheel build + mlx_lm tests.
- `docs/rocm-investigation-2026-07-19/RESULTS.md` — dual-load SEGV (M1).

**`the vendor ref` (Ollama's forward-merge):**
- `mlx/backend/rocm/primitives.cpp:59-72` — `GatedDeltaUpdate::use_fallback` always true.
- Last commit 2026-09-28 (mxfp8 prefill GEMM, cooperative softmax, gfx1200 fixes).

**Prior-art repos:**
- `lemonade-sdk/lemon-mlx-engine/CMakeLists.txt` — `FetchContent(the ROCm backend fork @ the vendor ref)`.
- `bogdan-d/bleeding-edge` — Strix Halo gfx1151, 151 tok/s @ 0.6B, TheRock 7.13.
- `the Ollama ROCm-MLX integration` — RX 9060 XT gfx1200, ROCm 10, fp8 GEMV 104 tok/s.

**ROCm (web):**
- ROCm 10.1.0 GPU spec table — RX 9070 XT = gfx1201 (RDNA4), 16 GB, 64 CU, wavefront 32/64.
- ROCm 6.4.0 release notes — "Support for gfx1201" (RDNA4 first supported).
- ROCm 10.1.0 compatibility matrix — gfx1200/gfx1201 supported.
