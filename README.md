# mlx-rocm

ROCm backend for [MLX](https://github.com/ml-explore/mlx) — **MVP skeleton** (v0.0.1).

End use: run the fleet's qMLX inference stack (qmlx-serve, Qwen3.x, MTP spec
decode, prefix/disk KV) on **marzuki-neon** — AMD RX 9070 XT (RDNA4,
**gfx1201** — *not* gfx110x), ROCm 10.1 userland (the verified guest toolchain;
core at `/opt/rocm/core-10.1`. 6.4 was the early probe validation and 7.7 the
original target — neither is the verified version).

## What this is

Three pieces:

1. **`vendor/mlx-rocm-upstream/`** — the vendored ROCm backend tree. The
   work is ~complete in the vendored base: `ml-explore/mlx` a community MLX ROCm backend PR (head
   `the vendor ref`) carries a complete `mlx/backend/rocm/`
   mirroring the CUDA backend file-for-file; we vendor the newer
   forward-merge `the vendor ref` @ `the pinned base sha` (adds
   GatedDeltaUpdate fallback, mxfp8 prefill GEMM, gfx1200/RDNA4 fixes).
   Provenance: `vendor/mlx-rocm-upstream/README.md`.
2. **`src/cpp/mlx_rocm/gemm/`** — a minimal standalone hipBLASLt GEMM
   (bf16 + fp8 e4m3, persistent handle + workspace, heuristic algo,
   row-major layouts) plus a dependency-free CPU reference GEMM.
   Modeled on the fork's `gemms/hipblaslt_gemm.cpp`.
3. **`scripts/smoke.sh`** — builds the GEMM, runs a small GEMM on the
   build host (hydrogen, CPU reference), then rsyncs the tree to
   marzuki-neon (`ssh andryo@100.91.94.56`), builds it against local
   ROCm, and runs the GPU GEMM + fp8 probe there (neon phase budgeted
   ~120 s; failures are recorded, not fatal).

## gfx1201 notes

- **RX 9070 XT = `gfx1201`** (RDNA4). `gfx110x` is RDNA3 (RX 7000). The
  fork's default arch list includes gfx1200/gfx1201; the RDNA4 tier gets
  its own QMV/QMM crossover tuning and the WMMA dense-qmm allowlist.
- **fp8 GEMM status on this box**: the smoke's
  `mlx_rocm_gemm_fp8_probe()` (256³ e4m3 GEMM vs. host reference)
  **passes** on neon gfx1201 (max_rel_err 0.0057, 0/65536 bad) — the
  probe shows no driver bug. Upstream notes fp8 e4m3 activation GEMM
  "may be kernel-buggy on some RDNA4 drivers — long-context prefill on
  gfx1201 garbles" (t386 evidence, a different workload than this
  probe). Keep `MLX_ROCM_GEMM_BF16=1` as the safe default until real
  prefill validation proves the driver clean for that workload. The
  probe is part of the smoke run.
- Discrete-GPU allocator hazards the fork engineered around (BAR read
  ~54 MB/s vs hipMemcpy 50.4 GB/s → pinned `host_stage` bounce;
  async-pool off by default) are inherited from the vendored tree.

## Build

Host-only (no ROCm needed — CPU reference path):

```
cmake -S . -B build && cmake --build build
./build/gemm_smoke host
```

With ROCm (gfx1201 box, e.g. the neon guest — see `docs/quickstart.md`):

```
cmake -S . -B build -DMLX_ROCM_WITH_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 \
  && cmake --build build
./build/gemm_smoke hip        # bf16 512^3 + fp8 e4m3 probe
```

Full `mlx` wheel from the vendored tree (next phase):

```
CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1" \
  python setup.py bdist_wheel   # from vendor/mlx-rocm-upstream layout
```

## Smoke

```
scripts/smoke.sh
```

Local: build + small host GEMM. Neon: `ssh` reachability + `rocm-smi`
check, tar transfer, ROCm build, GPU GEMM + fp8 probe. Results land in
`scripts/smoke-results/` (gitignored).

## Supported models (gfx1201, ROCm 10.1 guest)

| Model | Quant | Status on gfx1201 |
|---|---|---|
| Qwen3-0.6B | 8-bit (affine) | validated — benched 207.6–214.3 tok/s (`k2-non-gemm-fusion`) |
| Qwen3-0.6B | bf16 | Phase 0 smoke target (`scripts/smoke.py` ladder) |
| 9B-class dense Qwen3 | 4-bit (affine qmm) | validated — benched 55.9–57.8 tok/s A/B (`k2-non-gemm-fusion`), 63.8 tok/s with a 16-token warmup (`opt-gauntlet-areaA`) |
| 27B+ / 122B-hybrid / MoE | — | not supported (Phase 2; a 27B load was in flight on the guest, unvalidated) |
| MTP spec decode | — | not supported (blocked on the M3 stream-TLS fix) |
| non-gfx1201 archs | — | not validated (only gfx1201 is validated in this repo) |

Details, in-flight items, and caveats: `docs/limitations.md`.

## Converting a model

`mlx_lm` (≥ 0.32) converts a HuggingFace ref to the safetensors layout the
loader expects:

```
# bf16
mlx_lm.convert --model Qwen/Qwen3-0.6B
# 4-bit affine qmm (validated on gfx1201)
mlx_lm.convert --model Qwen/Qwen3-8B --quantize --bits 4 --group-size 64
# 8-bit (AWQ-style)
mlx_lm.convert --model Qwen/Qwen3-8B --quantize --bits 8 --q-mode awq
```

Or use a pre-converted `mlx-community/*` ref (e.g.
`mlx-community/Qwen3-0.6B-8bit`). **mxfp4 is the in-flight target (K4)**:
prefill runs via the opt-in dequant→bf16 GEMM path today; there is no native
mxfp4 kernel yet.

## Status — in-flight lanes

| Lane | What | Branch |
|---|---|---|
| K1 (QMV/GEMV bandwidth) | int8-activation QMV fix (N=12288); no bench yet | `k1-qmv-gemv-bandwidth` |
| K2 (non-GEMM fusion) | bench complete, verdict ADOPT; PR #2 open | `k2-non-gemm-fusion` |
| K3 (GDN port) | segfault fix + numerics PASS; PR #3 open | `k3-gdn-port` |
| K4 (MXFP4) | opt-in dequant→bf16 GEMM only; no native kernel, no dedicated branch yet | — |
| G1 (graph decode) | pilot wiring + ROCm 10.1 `hipStreamIsCapturing` fix; symbol export / engine wiring pending | `g1-graph-decode` |
| alpha (docs/housekeeping) | alpha-state checklist + this docs wave | `alpha-checklist` / `alpha-docs` |

## Python package

`pip install`-able skeleton (`pyproject.toml`, `src/python/mlx_rocm/`):
ctypes bindings to `libmlx_rocm.so` (host GEMM always; GPU GEMM + fp8
probe in ROCm builds) with a pure-numpy bf16 fallback. The wheel does not
yet bundle the full `mlx` C++ core — that is the next milestone (full
wheel + qmlx-serve 9B validation).

## License

MIT (MLX upstream license; see `LICENSE`).

## Repo layout (scaffold)

- `docs/plan.md` — the research report + phased plan (architecture, CUDA→ROCm map,
  prior art, gfx1201 findings, effort).
- `docs/limitations.md` — what works / in flight / not supported / caveats.
- `docs/quickstart.md` — the verified build + smoke path on a gfx1201 box.
- `PLAN.md` — living plan (phases, current state, open questions).
- `vendor.json` — the pinned vendor fork + ref (primary/fallback) and the build flags.
- `scripts/build-wheel.sh` — build the `mlx` wheel with `-DMLX_BUILD_ROCM=ON` for gfx1201 (runs on neon).
- `scripts/smoke.py` — Phase 0 smoke: is_available, matmul/softmax numerics, 0.6B generate.
- `scripts/smoke-serve.sh` — Phase 1 smoke: qmlx-serve with a 9B-class dense Qwen3.
- `scripts/size_ratchet.py` + `size-ratchet.json` — deterministic size gate (CI).
- `tests/` — hermetic tests (CI) + `@pytest.mark.gpu` tests (neon only).

## Commands

```bash
uv sync
uv run ruff format . && uv run ruff check .
uv run pytest -m "not gpu"          # hermetic (CI)
uv run python scripts/size_ratchet.py --base HEAD
```

GPU commands run on marzuki-neon (see AGENTS.md).

## Rules

Branch + PR to main (protected: no direct pushes), CI green, adversarial review by a
different agent than the submitter, peer merges `--squash`. See AGENTS.md.
