# Quickstart — verified build + smoke path on a gfx1201 box

The verified toolchain is the **marzuki-neon guest VM** (RX 9070 XT / gfx1201,
ROCm 10.1 userland). This is the path that has actually been run (t479/t480,
k2/k3, gauntlet, battery). Nothing else is verified.

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
- **venv:** Python 3.12 (`python3.12 -m venv /mnt/models/venv`; reinstall
  `python3.12-venv` if ensurepip files are missing) with cmake 4.4.4, ninja,
  setuptools.

## 1. Build the `mlx` wheel

Clone the pinned vendor (or use the committed, byte-verified
`vendor/mlx-rocm-upstream/`), then build in the venv:

```
CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1" \
PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm \
  /mnt/models/venv/bin/pip wheel . --no-deps -w /mnt/models/dist/
```

(`-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1` is the addition over the
`vendor.json` CMAKE_ARGS — the 10.1 core is not on CMake's default search
path. `scripts/build-wheel.sh` does the same thing for the pinned fork.)

## 2. Install into the venv (stale-lib trap)

```
/mnt/models/venv/bin/pip install --force-reinstall --no-deps /mnt/models/dist/mlx-*.whl
```

Overwrite **both** `site-packages/mlx/lib` and `site-packages/mlx/lib64`
and sha-verify — a stale `libmlx.so` in either directory silently shadows the
new one (t461/t462).

## 3. Host-only GEMM build (no full wheel needed)

The scaffold's standalone hipBLASLt GEMM builds against the same core:

```
cmake -S . -B build -DMLX_ROCM_WITH_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 \
  && cmake --build build
./build/gemm_smoke hip        # bf16 512^3 + fp8 e4m3 probe
```

## 4. Smoke ladder

```
# Phase 0: is_available + matmul/softmax numerics (+ 0.6B generate with MLX_ROCM_SMOKE_LLM=1)
PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm MLX_ROCM_GEMM_BF16=1 \
  /mnt/models/venv/bin/python scripts/smoke.py

# Phase 1: qmlx-serve with a 9B-class dense Qwen3 (scripts/smoke-serve.sh)
QMLX_SERVE_BIN=... QMLX_SERVE_MODEL=... bash scripts/smoke-serve.sh
```

`MLX_ROCM_GEMM_BF16=1` stays the safe default until the gfx1201 fp8 prefill is
proven clean (docs/plan.md §4.2). GPU benches take `flock /mnt/models/gpu-bench.lock`
and run with `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`.

## 5. Verify the GPU

```
sudo /opt/rocm/bin/amd-smi        # gfx1201, 64 CU (guest partition: 32 CU)
```
