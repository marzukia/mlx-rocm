# rocm-vendor-delta — the 81-build vendor delta (v2 wheel tree)

This directory reproduces, from stock 75ca881, the exact vendor source tree
that builds the v2 wheel (and the v3/v4 wheels of this branch). It is the
single source of truth for "what is in the wheel".

Contents:

- **`delta.patch`** — the cumulative `git diff` (34 files, 3114 insertions /
  523 deletions) of the 81-build vendor tree against stock 75ca881
  (`lifeofpiboy/mlx` fork, commit
  `75ca881b2e82f7b4069c8564a203792aa16d8a29`). Covers the GDN fused/tape op
  C++ side (`mlx/fast.cpp`, `mlx/fast.h`, `mlx/fast_primitives.h`,
  `python/src/fast.cpp`, `mlx/transforms.cpp`, `CMakeLists.txt`), the QMV rows
  kernels (`qmm.hip`), the 81-build backend work (`device.cpp`, `allocator`,
  `copy.hip`, `rms_norm.hip`, `rope.hip`, `flash_attention.hip`,
  `indexing.hip`, `eval.cpp`, ...), and the CUDA/other cleanup hunks.
- **`new_files/`** — the untracked new files of the same tree:
  `gdn_fused_step.hip` (F-branch kernel), `gdn_fused_step_tape.hip`
  (**pre-conv-fix** — the fix is chain step 3, `patches/gdn-fused-step-tape/
  conv-row-fix.diff`), `graph_decode_pilot_logic.h`, and
  `quantized/{qmv_lx,qmv_rows_msg,qmv_wide_mr}_kernel.hip` (rows kernels;
  runtime-gated OFF by default).

## Apply (chain steps 1-2 — see docs/RUN-verify-attention.md)

```
git clone --filter=blob:none https://github.com/lifeofpiboy/mlx v
cd v && git checkout 75ca881b2e82f7b4069c8564a203792aa16d8a29
git apply --check <repo>/patches/rocm-vendor-delta/delta.patch   # silent
git apply <repo>/patches/rocm-vendor-delta/delta.patch
cp -r <repo>/patches/rocm-vendor-delta/new_files/. mlx/backend/rocm/
```

Verified (this branch): full chain
`delta → new_files → conv-row-fix → sdpa_s4_env` applies
`git apply --check` + apply on a fresh 75ca881 worktree, and the resulting
`gdn_fused_step_tape.hip` and `scaled_dot_product_attention.hip` are
byte-identical to the guest-tree files that build the wheels.

Note: the historical `patches/gdn_fused_step/` in this repo was generated
against a pre-patched base (PR3-HYBRID) and does **not** apply on stock
75ca881 (its fast.cpp/fast.h hunks need the pre-patch context). This delta
supersedes it for wheel reproducibility; `gdn_fused_step/` is kept for
lineage only.
