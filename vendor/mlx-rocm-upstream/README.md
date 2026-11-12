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
full-wheel build phase starts. Exception: `python/src/fast.cpp` is vendored as an inert source snapshot (the W1 additions below).

## W1 additions (PR #30, 2026-11-12; default-off, env-gated)

The half-precision draft module + the sequential GDN kernel (the llama.cpp GDN design port, ref 29c720ea) + slot2 (R1b) land in the vendor tree: `mlx/backend/rocm/gdn_fused_step_tape.hip` +1030 (slot2 kernel/launcher/eval_gpu + the seq kernel pair/launchers/eval_gpu, incl. the unique-tid 2D-block build fix), `mlx/fast.cpp` +240 (the slot2 op + the seq op; the MTP6_TAPE=seq dispatch side only), `mlx/fast.h` +51 (both decls), `mlx/fast_primitives.h` +121 (the GdnFusedStepSlot2 + GdnFusedStepSeq classes), all pure insertions with zero deletions, plus `python/src/fast.cpp` NEW (the box-patched pybind state verbatim; inert in this repo, the root CMake builds only src/cpp/mlx_rocm/gemm/hipblaslt.cpp).

Engagement gates (all default OFF; the stock 97.52 path is the unchanged default): the seq op under MLX_ROCM_GDN_SEQ=1 + MTP6_TAPE=seq; the slot2 op under MLX_ROCM_GDN_SLOT2=1 + MLX_ROCM_GDN_FUSE=1 + the shape gate; the model-side half-precision cast under MTP6_MOD_BF16=1 (guarded, default "0"). Named pre-existing delta: the repo vendor snapshot predates the gdn_fused_step_slot (B2) + gdn_fused_step_tape_t1 ops; the insertion blocks are self-contained and reference them in comments only.

## Build (upstream layout, for the full wheel)

```
CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS" \
  python setup.py bdist_wheel
```

`gfx1201` = RX 9070 XT (RDNA4). **Not** `gfx110x` (that is RDNA3 / RX 7000).
ROCm 6.4+ supports gfx1200/1201; guest-a runs ROCm 7.x (TheRock/Fedora stack).
