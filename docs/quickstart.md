# Quickstart — verified build + smoke path on a gfx1201 box

The verified toolchain is the **guest-a guest VM** (RX 9070 XT / gfx1201,
ROCm 10.1 userland). This is the path that has actually been run (run-479/run-480, k2/k3, gauntlet, battery). Nothing else is verified.

## Guest environment (what must be true)

- **Driver:** amdgpu 7.1.3 (31.50) loaded, `/dev/kfd` present. `rocm-smi` does
  not exist in ROCm 10.1 — the equivalent is `/opt/rocm/bin/amd-smi`
  (expect `TARGET_GRAPHICS_VERSION: gfx1201`).
- **ROCm 10.1 userland:** `amdrocm-*10.1` packages installed (base/llvm/runtime/
  blas/solver/amdsmi/profiler, all 10.1.0-3, incl. the gfx1201 packages). The
  core lives at **`/opt/rocm/core-10.1`** (not a default CMake search prefix).
- **Dev packages:** `amdrocm-core-dev10.1` (+ `-gfx1201`) — the ROCm 10.1 core
  packages ship no cmake configs or dev headers without it. Restores
  `/opt/rocm/core-10.1/include/{hip,rocblas,…}` and
  `/opt/rocm/core-10.1/lib/cmake/{hip,rocblas,rocprim,hiprand,rocthrust,…}`.
- **LAPACK/BLAS headers:** `libopenblas-dev` + `liblapack-dev` do not ship
  `lapack.h`; the 8 OpenBLAS headers (lapack.h, lapacke.h,
  lapacke_config.h, lapacke_mangling.h, lapacke_utils.h, openblas_config.h,
  …) must be copied to `/opt/rocm/core-10.1/lib/host-math/include/openblas/`
  (a single-file copy fails on the sibling `lapacke_mangling.h`).
- **venv:** Python 3.12 — the canonical venv Python for the wheel path —
  (`python3.12 -m venv /models/venv`; reinstall `python3.12-venv` if
  ensurepip files are missing) with cmake 4.4.4, ninja, setuptools.
  (`BUILD-RECIPE.md`'s CPython 3.14 venv is the 2026-11-01 historical
  provenance snapshot, not the current path.)

## 1. Build the canonical wheel (the `mlx_rocm` shim)

The wheel build is **guest-a-only**: a ROCm gfx1201 box is required.
`scripts/build-wheel.sh` clones the pinned vendor tree (the `vendor.json`-
pinned, sha-verified checkout of this repo — it fails loud if the pin is
unresolvable or the checkout mismatches; the committed
`vendor/mlx-rocm-upstream/` is a partial, byte-verified *reference*
snapshot without a build manifest, NOT the build input), then runs
`pip wheel .` on the repo root (setuptools, `packages = mlx_rocm`, no
`setup.py`); the deliverable is the **`mlx_rocm-0.2.0` shim wheel ONLY**
(`dist/mlx_rocm-0.2.0-*.whl`) and the canonical path does not build the
`mlx` framework.

```
/models/venv/bin/pip wheel . --no-deps -w /models/dist/   # what scripts/build-wheel.sh runs
```

### The full `mlx` framework wheel — historical-recipe path (NOT the canonical deliverable)

The `mlx-*.whl` shape (`import mlx.core`, `site-packages/mlx/lib`, the
CMake `-DMLX_BUILD_ROCM=ON` build) is the 2026-11-01 historical recipe
(`BUILD-RECIPE.md`); its CMake build manifest is no longer in-tree
(`CMakeLists` preserved as `CMakeLists.upstream.txt`), so the full
framework wheel is **not produced by the canonical path** — rebuilding it
is the follow-up box session.

(`-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1` is the addition over the
`vendor.json` CMAKE_ARGS for that historical CMake path — the 10.1 core
is not on CMake's default search path. `scripts/build-wheel.sh` does the
same thing for the pinned fork.)

## 2. Install into the venv (stale-lib trap)

```
/models/venv/bin/pip install --force-reinstall --no-deps /models/dist/mlx_rocm-*.whl
```

The stale-lib trap applies to the full `mlx` framework wheel of the
historical-recipe path, not the shim: overwrite **both**
`site-packages/mlx/lib` and `site-packages/mlx/lib64` and sha-verify — a
stale `libmlx.so` in either directory silently shadows the new one
(run-461/run-462).

## 3. Host-only GEMM build (no full wheel needed)

The scaffold's standalone hipBLASLt GEMM builds against the same core:

```
cmake -S . -B build -DMLX_ROCM_WITH_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 \
  && cmake --build build
./build/gemm_smoke hip        # bf16 512^3 + fp8 e4m3 probe
```

## 4. Model files needed (placeholders; no personal paths required)

The smoke ladder and the serve smoke need, under `$MODELS` (the model
directory mount, e.g. `/models`):

- `qwen3-06b-mlx` — Qwen3-0.6B 8-bit (bf16 for the Phase 0 bf16 leg),
  from the conversion recipe in the README ("Converting a model",
  `mlx_lm.convert --model Qwen/Qwen3-0.6B --quantize --bits 8 --q-mode awq`).
- `qwen3-9b-mlx` — 9B-class dense Qwen3, 4-bit affine
  (`mlx_lm.convert --model Qwen/Qwen3-8B --quantize --bits 4 --group-size 64`).
- `model-mtp-fixed.safetensors` — the MTP sidecar next to `qwen3-9b-mlx`
  (the fixed-W+1 RMSNorm build; see `docs/audit/mtp-numpy-rca.md`).

Pre-converted `mlx-community/*` refs work as well (README, "Converting a
model").

## 5. Smoke ladder

```
# Phase 0: is_available + matmul/softmax numerics (+ 0.6B generate with MLX_ROCM_SMOKE_LLM=1)
PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm MLX_ROCM_GEMM_BF16=1 \
  /models/venv/bin/python scripts/smoke.py

# Phase 1: qmlx-serve with a 9B-class dense Qwen3 (scripts/smoke-serve.sh)
# (QMLX_SERVE_PORT defaults to 8080; set it to move the port)
QMLX_SERVE_BIN=... QMLX_SERVE_MODEL=... QMLX_SERVE_PORT=8080 bash scripts/smoke-serve.sh
```

`MLX_ROCM_GEMM_BF16=1` stays the safe default until the gfx1201 fp8 prefill is
proven clean (docs/plan.md §4.2). GPU benches take `flock /models/gpu-bench.lock`
and run with `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`.

## 6. Verify the GPU

```
sudo /opt/rocm/bin/amd-smi        # gfx1201, 64 CU (guest partition: 32 CU)
```
