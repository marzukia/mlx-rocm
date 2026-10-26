# Limitations — mlx-rocm on gfx1201

Plain list, 2026-10-26. All GPU results are from the marzuki-neon guest VM
(RX 9070 XT / gfx1201, ROCm 10.1 userland; the guest runs a 32-CU partition
of the card, so timings are partition-scaled, not whole-card). Evidence
pointers name the branch/report.

## What works (validated on the guest)

- **9B-class dense Qwen3 (4-bit qmm)** — served and benched: 55.9–57.8 tok/s
  A/B (`k2-non-gemm-fusion`), 63.8 tok/s mean with a 16-token warmup
  (`opt-gauntlet-areaA`, `low-hanging-fruit-battery`). No NaNs in the bench
  runs.
- **8-bit quantized (affine 8-bit QMM)** — 0.6B-8bit benched at 207.6–214.3
  tok/s (`k2-non-gemm-fusion`).
- **GatedDeltaUpdate port (GDN)** — the t506 `copy_general_general` segfault is
  fixed (chunked host-shadow copy, no per-chunk `hipStreamDestroy`) and GDN
  numerics pass (recurrent + chunk paths vs the op-loop fallback and an f32
  reference) on the `k3-gdn-port` re-run.
- **bf16 GEMM** — matmul GPU vs CPU max_err 0.0000 (`t480-report` Phase 0;
  the earlier 6.4 probe: bf16 GEMM PASS 0.0078).
- **fp8 e4m3 probe** — the 256³ e4m3 GEMM vs host reference passes
  (max_rel_err 0.0057, 0/65536 bad). That is a probe, not a prefill
  validation (see caveats).

## In flight (not validated yet)

- **MXFP4 (K4)** — mxfp4 prefill exists only as the opt-in dequant→bf16 GEMM
  path (`qmm.hip:3475-3499`); no native mxfp4 kernel. No dedicated branch yet.
- **Graph decode (G1)** — pilot wiring + the ROCm 10.1 `hipStreamIsCapturing`
  fix are on `g1-graph-decode`; the symbol is compiled in but hidden
  (`-fvisibility=hidden`, no `MLX_API` export) and nothing in the engine
  enables the mode, so no A/B is possible until a rebuild.
- **QMV/GEMV bandwidth (K1)** — the int8-activation QMV bug at N=12288 is
  fixed on `k1-qmv-gemv-bandwidth`; no bench yet.

## Not supported

- **27B+ / 122B-hybrid / MoE** — Phase 2. A 27B load was in flight on the
  guest (t479/t480) but not validated.
- **MTP spec decode** — blocked on the M3 stream-TLS fix (`There is no
  Stream(cpu, N)` in `mlx/backend/cpu/encoder.cpp`); Phase 2 at the earliest.
- **Non-gfx1201 archs** — only gfx1201 is validated here. The fork's arch
  list includes gfx1200 and RDNA3 gfx110x, but nothing in this repo has run
  or validated them.

## Caveats (known, shipped around)

- **fp8 e4m3 prefill on gfx1201** — upstream notes it "may be kernel-buggy on
  some RDNA4 drivers — long-context prefill garbles". The probe passes; real
  prefill is unvalidated, so `MLX_ROCM_GEMM_BF16=1` is the safe default
  (`scripts/smoke-serve.sh`).
- **NO_GPU op set** — FFT, the LAPACK family (LUF/QRF/SVD/Inverse/Eig/Eigh),
  and BlockMaskedMM fall back to CPU.
- **GDN long prefill** — the per-timestep fallback is fine for decode (T=1)
  but slow for long prefill (T=4096 ≈ 32k small kernel launches) until the
  fused GDN port (K3) lands.
- **Dual large-model load on unified-memory APUs** — SIGSEGV (exit 139) from
  resource overcommit; not the blocking case for a single discrete dGPU, but
  the one to watch if two models run.
- **Allocator hazards on discrete GPUs** — BAR read ~54 MB/s vs hipMemcpy
  50.4 GB/s (pinned `host_stage` bounce engineered around it); async
  `hipMallocAsync` pool off by default. Large D2H shadow copies are chunked
  (≤100 MB, `MLX_ROCM_D2H_CHUNK_MB`) to avoid the intermittent KFD
  completion hang.
- **Cold start** — the first generation in a fresh process is ~1.7× slower
  (hipBLASLt solution cache, first QMV launch, ~600 MB allocator pool
  expansion). A 16-token warmup before serve readiness removes it
  (`low-hanging-fruit-battery`).
- **No install path yet** — the wheel is built on the guest; nothing is
  published (PyPI slot `mlx-rocm` held, unreleased).
