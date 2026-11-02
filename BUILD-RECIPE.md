# BUILD-RECIPE.md — PR3-HYBRID fused wheel (gdn_fused_step, gfx1201)

Provenance: this recipe reproduces, from git, the fused-AR mlx wheel that measured **80.93 tok/s** on greedy (AR) decode of Qwen3-9B (GDN) vs **72.57 tok/s** no-fuse on the same host (**+11.5%**). All host-specific paths are placeholders: `<neon-host>` (the bare-metal ROCm box), `<guest-user>`, `<models-mount>` (the model directory mount), `<repo>` (this checkout), `<vendor-dir>`, `<wheelout>`, `<venv>`.

## What the wheel is

- **Vendor base**: the mlx source snapshot at commit `75ca881` (full sha `75ca881b2e82f7b4069c8564a203792aa16d8a29`) — the single-commit snapshot of the ROCm-port fork (GDN tail fusion + the `gdn_prep` launcher pointer fix already in base).
- **Graft**: `patches/gdn_fused_step/gdn_fused_step.patch` (already in main) — adds `mx.fast.gdn_fused_step` (4-launch GDN decode chain → 1 fused launch, AR S=1): new kernel `mlx/backend/rocm/gdn_fused_step.hip`, API in `mlx/fast.h`/`mlx/fast.cpp`, primitive in `mlx/fast_primitives.h`, python binding in `python/src/fast.cpp`, +1 line in `mlx/backend/rocm/CMakeLists.txt`.
- **Proven artifact**: `mlx-0.32.3.dev20261101+75ca881-cp314-cp314-linux_x86_64.whl` (21,965,815 bytes; built 2026-11-01 04:42 UTC on `<neon-host>`; installed in a uv-managed CPython 3.14.8 venv).
- **Graft verification on the proven wheel**: `libmlx.so` carries the `GdnFusedStep` symbols; the installed venv reports `mx.fast.gdn_fused_step = True` and `rms_norm_residual = False` — i.e. the proven build tree was base `75ca881` + the GDN patch only (the K2 `rms_norm_residual` graft present in the pre-build working-tree snapshot was NOT in the built wheel).

## Prerequisites

- Bare-metal ROCm 10.1 host with an RX 9070 XT (`gfx1201`) — `<neon-host>` — ROCm userland at `/opt/rocm` (the 10.1 line; `LD_PRELOAD` target `/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1` present).
- CPython 3.14 (3.14.8 for the proven venv), `uv`, `patch`.
- Model at `<models-mount>/qwen3-9b-mlx`, with the `model-mtp-fixed.safetensors` sidecar in the same directory.
- Hold the host's GPU lock for the whole build+bench session (host convention).

## Step 1 — vendor base

```
git clone --depth 1 --branch tail-fusion-gdn <vendor-fork-url> <vendor-dir>
git -C <vendor-dir> rev-parse HEAD
# must print 75ca881b2e82f7b4069c8564a203792aa16d8a29
```

## Step 2 — apply the GDN graft (idempotent)

```
cd <vendor-dir>
grep -q "gdn_fused_step" mlx/backend/rocm/CMakeLists.txt && echo already-patched \
  || patch -p1 < <repo>/patches/gdn_fused_step/gdn_fused_step.patch
# expect 6 files checked, 0 rejects, no .orig/.rej left behind
git -C <vendor-dir> status --short
# expect: 5 modified (CMakeLists.txt, fast.cpp, fast.h, fast_primitives.h, python/src/fast.cpp) + 1 untracked (mlx/backend/rocm/gdn_fused_step.hip)
```

Note on the r4d GDN kernels: NO functional r4d change is part of the graft. The r4d `r4d_gdn_*.hip`/`gated_delta_update.cpp` files differ from base `75ca881` in the proven tree only by a cosmetic comment scrub (provenance attribution line) — functionally irrelevant to the wheel.

## Step 3 — build the wheel (proven command, run ON `<neon-host>`)

```
cd <vendor-dir>
CMAKE_PREFIX_PATH=/opt/rocm \
CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS" \
python -m pip wheel . --no-deps -w <wheelout>/
ls -l <wheelout>/   # proven: mlx-0.32.3.dev20261101+75ca881-cp314-cp314-linux_x86_64.whl (21,965,815 B)
```

Note: `vendor.json`'s `core-10.1` prefix reference (`/opt/rocm/core-10.1`) is stale for this build — the proven wheel was built with `CMAKE_PREFIX_PATH=/opt/rocm`.

The `.so` md5 varies with build path/timestamp (documented build nondeterminism); source identity is what the graft evidence is about, not byte-identical artifacts.

## Step 4 — venv + install (proven venv = uv-managed CPython 3.14.8, no pip module)

```
uv venv --python 3.14 <venv>
uv pip install --python <venv> <wheelout>/mlx-0.32.3.dev20261101+75ca881-cp314-cp314-linux_x86_64.whl
uv pip install --python <venv> mlx-lm==0.32.2
# proven venv package list (for an exact-matching venv):
#   mlx 0.32.3.dev20261101+75ca881 (the wheel), mlx-lm 0.32.2, numpy 2.5.4, transformers 5.22.1, huggingface_hub 2.4.0, hf-xet 1.8.0, tokenizers 0.23.3, safetensors 0.8.0, regex 2026.9.29, sentencepiece 0.2.2, httpx2 2.16.0, anyio 4.15.1, filelock 4.2.6, fsspec 2026.9.0, h11 0.16.0, jinja2 3.1.6, packaging 26.3, protobuf 7.36.2, pyyaml 6.0.3, shellingham 1.5.4, tqdm 4.70.1, truststore 0.10.5, typer 0.27.3, typing_extensions 4.16.0, annotated_doc 0.0.5, click 8.5.0, markdown-it-py 4.2.0, pygments 2.21.0, rich 15.0.0
```

Sanity gate (no GPU needed):

```
<venv>/bin/python -c "import mlx.core as mx; assert hasattr(mx.fast, 'gdn_fused_step'); print('graft live')"
```

## Step 5 — model-side patch (idempotent; the FUSE_BRANCH_PATCH marker)

The 80.93 state also requires the patched `mlx_lm/models/qwen3_5.py` (diffed against stock mlx-lm 0.32.2; this patch round-trips to a byte-identical copy of the proven installed file, 25,912 bytes). Markers in the patch: `FUSE_BRANCH_PATCH` (the one that matters — the S==1 && B==1 branch that dispatches to `mx.fast.gdn_fused_step`, gated on `MLX_ROCM_GDN_FUSE=1`, with the `GDN_FUSE_CALLS` dispatch counter), plus `MTP_INJECT_PATCH` and `PRENORM_PATCH` (inert under `MLX_MTP=0`).

```
SITEPKG=<venv>/lib/python3.14/site-packages
grep -q "FUSE_BRANCH_PATCH" $SITEPKG/mlx_lm/models/qwen3_5.py \
  && echo already-patched \
  || patch -p1 -d $SITEPKG < <repo>/patches/qwen35_fuse_branch/qwen35_fuse_branch.diff
# expect: 1 file patched, 0 rejects
<venv>/bin/python -c "
import mlx.core as mx
assert hasattr(mx.fast, 'gdn_fused_step')
import mlx_lm.models.qwen3_5 as m
assert 'FUSE_BRANCH_PATCH' in open(m.__file__).read()
print('model patch live')"
```

## Step 5b — model-side tape patch (optional; the t1265 tape-verify state)

The GDN state-tape verify state (RUN card `docs/RUN-gdn-state-tape.md`) additionally requires the `qwen35_tape_branch` model patch: a superset of Step 5's patch, diffed against the same stock mlx-lm 0.32.2 `qwen3_5.py` (round-trips byte-clean on a fresh stock install: 20,559 -> 27,721 bytes; the t1265 installed tape model, 27,593 bytes, is the pre-gate-fix variant and is regenerated from this patch for the tape run — the gate changed S<=4 to S in {1, 2, 4}). It adds, on top of `FUSE_BRANCH_PATCH`/`MTP_INJECT_PATCH`/`PRENORM_PATCH`, the `TAPE_BRANCH_PATCH` dispatch (`mx.fast.gdn_fused_step_tape` for S in {1, 2, 4} — S=3 falls through to the F-branch — gated on `MLX_ROCM_GDN_TAPE=1` + `MLX_ROCM_GDN_FUSE=1`, per-layer `last_tape` capture, `GDN_TAPE_CALLS` counter) and `import os`. The wheel it targets is built from the repo's self-contained graft chain (the `vendor/mlx-rocm-upstream/` tree + `patches/gdn_fused_step/` + `patches/gdn_fused_step_tape/`; see the RUN card's chain-verification note).

```
SITEPKG=<venv>/lib/python3.14/site-packages
grep -q "TAPE_BRANCH_PATCH" $SITEPKG/mlx_lm/models/qwen3_5.py \
  && echo already-patched \
  || patch -p1 -d $SITEPKG < <repo>/patches/qwen35_tape_branch/qwen35_tape_branch.diff
# expect: 1 file patched, 0 rejects
# (supersedes Step 5: apply this instead of qwen35_fuse_branch.diff for the tape state)
```

## Step 6 — bench (proven environment; ON `<neon-host>`, GPU locked)

```
export MODELS=<models-mount>
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
export HCC_AMDGPU_TARGET=gfx1201
export MLX_MTP=0

# no-fuse arm (proven: 72.57 tok/s)
<venv>/bin/python <repo>/bench/mtp-bench3.py

# fused arm (proven: 80.93 tok/s, +11.5%)
MLX_ROCM_GDN_FUSE=1 <venv>/bin/python <repo>/bench/mtp-bench3-side.py
```

- Both arms: same prompt, `MAX_TOKENS=1024` (stop at EOS 248044), greedy, `MTP_MODEL` defaults to `$MODELS/qwen3-9b-mlx`; the `-side` variant additionally wires the `model-mtp-fixed.safetensors` sidecar when the config asks for MTP (skipped under `MLX_MTP=0`).
- Lossless gate: `FIRST16`/`ALLTOKS` byte-identical between the two arms.
- The fused arm should show the `GDN_FUSE_CALLS` counter advancing (fused dispatch is live).

## Provenance cross-checks (how this recipe was established)

- The graft patch applies cleanly (dry-run) to base `75ca881`; its `gdn_fused_step.hip` is byte-identical (md5 `d8d4a30f…`) to the working-tree copy in the pre-build vendor snapshot, whose `*.orig` files are the pre-patch backups.
- The built wheel's `libmlx.so` contains `GdnFusedStep` (34 symbol hits) and lacks `RMSNormResidual` (0) — the wheel was base + GDN patch, nothing else.
- The model patch was extracted by diffing the installed venv `qwen3_5.py` against stock `mlx-lm==0.32.2` (`pip download mlx-lm==0.32.2 --no-deps`); it round-trips to a byte-identical install (25,912 bytes).
- See the appended `REPORT.md` section for the vendor-tree ↔ repo relationship and the SHA comparison (`75ca881` / `9320ee9` / main `da9eeca`).
