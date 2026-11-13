# Limitations — mlx-rocm on gfx1201

Plain list, re-assessed 2026-11-12 against `main` @ `aed6a25f` (the
2026-10-26 list is the historical snapshot; since then the ROCm-version
story settled to 10.1, the license fixed to MIT, MTP K=2 (#26) and the W1
line (#30) merged, and the install path + published numbers landed). All
GPU results are from the guest-a guest VM (RX 9070 XT / gfx1201, ROCm
10.1 userland; the guest runs a 32-CU partition of the card, so timings
are partition-scaled, not whole-card). Evidence pointers name the
branch/report.

## What works (validated on the guest)

- **9B-class dense Qwen3 (4-bit qmm)** — served and benched: 55.9–57.8 tok/s
  A/B (`k2-non-gemm-fusion`), 63.8 tok/s mean with a 16-token warmup
  (`opt-gauntlet-areaA`, `low-hanging-fruit-battery`). No NaNs in the bench
  runs.
- **8-bit quantized (affine 8-bit QMM)** — 0.6B-8bit benched at 207.6–214.3
  tok/s (`k2-non-gemm-fusion`).
- **GatedDeltaUpdate port (GDN)** — the run-506 `copy_general_general` segfault is
  fixed (chunked host-shadow copy, no per-chunk `hipStreamDestroy`) and GDN
  numerics pass (recurrent + chunk paths vs the op-loop fallback and an f32
  reference) on the `k3-gdn-port` re-run (merged, #3).
- **9B hybrid (GDN + FA, 4-bit)** — the main workload, supported: AR 97.52
  t/s @1024 and MTP K=2 107.56 t/s @1024 certified (W1, #30; 84.8% of the
  same-GPU 126.9 t/s reference; lossless 3/3; in-tree repro
  `bench/gdn_seq1024_repro.sh`).
- **MTP spec decode (K=2)** — adopted and merged (#26: K=1/K=2 at 92.23 t/s
  @1024, token-identical to plain decode); the W1 sequential GDN kernel (#30)
  carries the 107.56 t/s @1024 certified record. Same-session 12-arm battery
  (2026-11-12): K2 prefill-fill +2.87% (TPK 2.297), K2 streaming +1.64%
  (TPK 2.332), K3 streaming TPK 2.670 (best tokens/step on the card, +6.01%
  within-session); the absolute re-stamp is pending the next card session —
  107.56 t/s stands as the certified record.
- **bf16 GEMM** — matmul GPU vs CPU max_err 0.0000 (`run-480-report` Phase 0;
  the earlier 6.4 probe: bf16 GEMM PASS 0.0078).
- **fp8 e4m3 probe** — the 256³ e4m3 GEMM vs host reference passes
  (max_rel_err 0.0057, 0/65536 bad). That is a probe, not a prefill
  validation (see caveats).

## In flight (not validated yet)

- **MXFP4 (K4)** — mxfp4 prefill exists only as the opt-in dequant→bf16 GEMM
  path (`qmm.hip:3475-3499`); no native mxfp4 kernel. No dedicated branch yet.
- **MTP K=3 streaming** — measured in-session (TPK 2.670, +6.01%
  within-session; best tokens/step on the card); the absolute re-stamp is
  pending the next card session, so it is not yet a validated absolute number.

## Retired

- **Graph decode (G1)** — the graph-capture NO-GO verdict
  (`docs/audit/slot-path-mtp-verify.md`; the run-1059 digest in
  `docs/audit/README.md`); the H2D staging re-delivery + the ROCm 10.1
  `hipStreamIsCapturing` fix are merged (#2/#3), but the engine mode is not
  enabled, so no A/B is possible.
- **QMV/GEMV bandwidth (K1)** — closed as shipped: the int8-activation QMV
  fix (N=12288) + the K1 kernel are in the committed vendor tree
  (`qmv_lx_kernel.hip`; `docs/vendor-overlay-k1.md`).

## Not supported

- **27B+ / MoE** — Phase 2. A 27B load was in flight on the guest
  (run-479/run-480) but not validated. (9B-hybrid: supported — the main
  workload, see "What works".)
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
  but slow for long prefill (T=4096 ≈ 32k small kernel launches); the fused
  GDN port (K3, merged #3) covers the decode path and the W1 sequential GDN
  kernel (#30) is in the certified record.
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
- **Publish state** — the PyPI `mlx-rocm` releases `0.0.0` and `0.1.2`
  (uploaded 2026-10-30) are published; the full `mlx` wheel is built on the
  guest (`scripts/build-wheel.sh`, the `vendor.json` pin verified at build
  time, fail-loud gate); 0.2.0-alpha is in flight (PR #31).
