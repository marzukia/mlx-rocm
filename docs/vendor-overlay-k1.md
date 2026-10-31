# Vendor overlay note — K1 N=12288 correctness fix (run-543, MAJOR-1)

Status: **overlay documented — fork push failed (no push creds), re-pin pending.**

`vendor.json` primary pins the vendor ref (branch `rocm-current`).
The N=12288 stale-output correctness fix (`99cddb5`) + the K1 v_dot4
int8-activation QMV kernel (`b42efb6`) live in the vendored tree
(`vendor/mlx-rocm-upstream/`, byte-verified vs the guest clone working tree)
as a delta over that pin:

- `mlx/backend/rocm/quantized/qmm.hip` — K1 gate (admits f32 x + float32
  instantiation) + tiled-QMV gate restricted to bf16/f16 x. The N=12288
  fix: f32 x (affine `quantized_matmul` promotes x to scales dtype) matched
  the old gate at N=12288 (gfx1201 tile_n=24 divides 12288) but the inner
  dispatch instantiates nothing and the block returns — no kernel, stale
  `out`.
- `mlx/backend/rocm/quantized/qmv_lx_kernel.hip` — new file: the K1 kernel
  (sxq staged as [even x | odd x] per 8-value group for 4-bit; warp-reduce
  acc before the out write).

## Why the overlay

The delta was committed for the fork as `1051504`
("rocm/qmv: K1 v_dot4 int8-activation QMV + N=12288 stale-output fix") in
the guest worktree `/tmp/mlx-k1-push` (guest-a, a `git worktree` of
`/models/mlx-clone-k1` at the vendor ref). The push to the vendor ref
failed 2026-10-26: the guest clone has no credentials (bare https remote,
no netrc/gh), and `<op-fork-user>` — the only token available — has no push
access to the vendor ref (403). The commit is ready to push once creds
exist:

```sh
# on guest-a
cd /tmp/mlx-k1-push && git push origin HEAD:rocm-current
```

## Consequence for builds

Until the fork re-pin lands, a standard pinned build
(`scripts/build-wheel.sh`: clone the vendor ref per
`vendor.json`) does NOT contain the N=12288 fix or the K1 kernel. It needs
the 2-file overlay above applied to the clone before building
(`MLX_VENDOR_DIR` pointing at a pre-overlay clone, or the guest
`/models/venv-k1` build which already carries it). The vendored tree in
this repo IS the overlay: `vendor/mlx-rocm-upstream/` = pinned tree + delta
(curated subset of the fork; `CMakeLists.upstream.txt` / `LICENSE.upstream`
are the fork's `CMakeLists.txt` / `LICENSE` renamed, `README.md` is the
provenance note). Re-pin `vendor.json` primary.sha to the pushed sha in a PR
when the fork push lands.
