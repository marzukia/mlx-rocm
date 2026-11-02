# vendored upstream: mlx ROCm backend (provenance)

## Source

| | |
|---|---|
| **Repository** | the vendor ref (an mlx fork) |
| **Branch** | `rocm-current` |
| **Commit** | the vendor-ref pin (2026-09-28) |
| **Last commit subject** | "rocm: mxfp8 prefill GEMM, cooperative softmax, device_info keys" |
| **Lineage** | Forward-merge of the open ROCm-base experiment "[Experiment] ROCm backend" (commit `75915908dfe5028335d318b10340313744fd3a8d`, 2026-07-19) onto the MLX base the Ollama fork pins. Adds GatedDeltaUpdate fallback, mxfp8 prefill GEMM, gfx1200/RDNA4 fixes, hipRTC custom-kernel routing. |
| **License** | MIT (see `LICENSE.upstream`, also `../LICENSE`) |

## What was vendored

- `CMakeLists.upstream.txt` — the fork's top-level `CMakeLists.txt`
  (carries the `MLX_BUILD_ROCM` option and the ROCm build wiring:
  `CMAKE_HIP_ARCHITECTURES` handling, hip compiler discovery).
- `mlx/` — the full `mlx/` source tree from the fork, most importantly:
  - `mlx/backend/rocm/` — the complete ROCm backend mirroring
    `mlx/backend/cuda/` file-for-file: allocator, device/worker/
    CommandEncoder (HIP-graph capture + build-once decode replay + the
    `ExecSlot` pool for the ROCm CLR kernelParams-by-pointer bug, clr#138),
    all primitive ports, `quantized/` (WMMA QMM incl. the `Rdna4`
    gfx1200/1201 tier + WMMA allowlist), `gemms/` (hipBLASLt + rocBLAS +
    naive + GEMV, with the `GemmPrecision` capability table and the
    gfx1201 fp8 GEMM bug note + `MLX_ROCM_GEMM_BF16` escape hatch),
    flash attention (vector decode + WMMA fwd/bwd prefill), hipRTC JIT,
    `host_stage` pinned-bounce host reads, custom P2P all-reduce
    (no RCCL), `moe_swiglu`, per-arch tuning.
  - `mlx/backend/gpu/` — the backend-agnostic GPU shim (the integration
    seam: `MLX_USE_ROCM` selection in `gpu/primitives.cpp`).
- `ci/build_rocm.yml` — the fork's CI (self-hosted Strix Halo runner,
  `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DMLX_ROCM_ARCHITECTURES=gfx1151"`).

Not vendored: `python/` (Pybind11 bindings incl. the `mlx.core.rocm`
module), `tests/`, `benchmarks/`, `.github` (except the ROCm workflow).
They are in the upstream repo at the pinned commit; pull them in when the
full-wheel build phase starts.

## Graft-base reconstruction (2026-11-02, PR #8 fix round)

The `75ca881` single-commit snapshot is NOT present in the public fork's refs (checked 2026-11-02: 9 refs, no `tail-fusion-gdn`). The committed tree therefore carries reconstructed graft-base regions so that `patches/gdn_fused_step/` + `patches/gdn_fused_step_tape/` form a self-contained, checkable chain (the wheel build itself still uses the guest's `75ca881` clone, which carries the same regions):

- `mlx/fast.cpp` — the `gdn_prep` + `gdn_norm_gated` function region (verbatim from the gfs patch's deletion hunk) is present in this tree (the gfs patch removes it; the tape patch then adds the `use_fallback` gate on top). Marked `// [GRAFT-BASE RECONSTRUCTION...]`.
- `mlx/fast_primitives.h` — the `GdnPrep` class stub (the gfs patch deletes it; the tape patch adds `GdnFusedStepTape` after the `GdnFusedStep` class it leaves). Marked `// [GRAFT-BASE RECONSTRUCTION...]`; the `GdnPrep` declaration is synthesized from its usage (definition lives in `gdn_prep.hip`) — it is NOT a compile target of this repo, it is a chain reference.
- `mlx/fast.h` — the `gdn_norm_gated` declaration (verbatim from the gfs patch context).
- `python/src/fast.cpp` — a reconstructed slice carrying the `init_fast` region the grafts reference (marked; the full upstream file is not vendored).

Chain verification (RUN card, `docs/RUN-gdn-state-tape.md`, "Graft chain verification"): fresh worktree of this directory → `patch -p1` for the gfs graft (host git 2.53 cannot `git apply --check` the gfs patch; see the git-version note) → `git apply --check` for the tape graft passes. The tape patch's `.hip` section is byte-identical to `../gdn_fused_step_tape/gdn_fused_step_tape.hip`.

## Build (upstream layout, for the full wheel)

```
CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS" \
  python setup.py bdist_wheel
```

`gfx1201` = RX 9070 XT (RDNA4). **Not** `gfx110x` (that is RDNA3 / RX 7000).
ROCm 6.4+ supports gfx1200/1201; guest-a runs ROCm 7.x (TheRock/Fedora stack).
