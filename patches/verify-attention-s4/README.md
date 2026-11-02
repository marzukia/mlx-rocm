# verify-attention S=4 → vector SDPA graft (env-gated port)

Ported from the t1274 branch (`worker/verify-attention-s4`), **re-worked to be
default-OFF**. Single file: `mlx/backend/rocm/scaled_dot_product_attention.hip`.
The `sdpa_s4_env.patch` here applies cleanly on the stock 75ca881 file (verified
`git apply --check` on a fresh 75ca881 checkout; the cumulative vendor delta in
`patches/rocm-vendor-delta/` does not touch this file).

Two hunks:

1. **m8 single-kernel runtime causal mask** (== t1274's hunk 1, already
   GPU-verified in the m8 wheel: S=2 causal row 1 vs S=1 no-mask n=0, G4
   lossless MATCH n=1024, first_div fixed). Removes the compile-time `do_causal`
   template split so all rows of a causal batch and the AR S=1 no-mask row use
   the same compiled code (bit-identical per row). This hunk is active in both
   env states — it is bit-identical to the stock kernels on every row the stock
   kernels serve (m8 wheel G4: M1-M0-CONTROL MATCH), so with the env OFF the
   graft's only *new* code path is dormant.
2. **Vector gate extension, env-gated**: stock gate
   `query_sequence_length < 4` becomes
   `query_sequence_length < 4 || (query_sequence_length == 4 && s4va_enabled())`
   where `s4va_enabled()` reads `MLX_ROCM_SDPA_S4VA` **once, process-lifetime
   (`static` init), `=1` engages**. Env unset/other → the gate is exactly the
   stock `< 4` → qL=4 rows go to the scalar flash (prefill) kernel exactly as on
   the no-graft wheel (bit-identical behavior, no new code path executes).
   `MLX_ROCM_SDPA_S4VA=1` → the K=3 verify batch (qL=4) is admitted to the
   1-pass vector decode kernel (grid (H, qL, B), row-independent, 32-tile
   parallel key scan) instead of the prefill kernel (~4.7 ms/call at ~1200 keys
   vs "tens of us" vector decode).

No `.cpp` change: `use_fallback` and the `eval_gpu` dispatch both follow
`supports_sdpa_vector`. WMMA gate (qL>4) and the LSE gate are untouched.
`scaled_dot_product_attention.hip` (post-patch file, committed here for
verification) = 339 non-blank lines.

## Apply (chain step 4 — see docs/RUN-verify-attention.md)

```
cd <worktree at 75ca881 + vendor delta + conv-row-fix>
git apply --check patches/verify-attention-s4/sdpa_s4_env.patch   # must be silent
git apply patches/verify-attention-s4/sdpa_s4_env.patch
grep -c "MLX_ROCM_SDPA_S4VA" mlx/backend/rocm/scaled_dot_product_attention.hip   # expect 3
diff -q mlx/backend/rocm/scaled_dot_product_attention.hip \
  patches/verify-attention-s4/scaled_dot_product_attention.hip                  # must be identical
```

## Env semantics

| `MLX_ROCM_SDPA_S4VA` | qL=4 rows | qL<4 rows | qL>4 rows |
|---|---|---|---|
| unset (default) | scalar flash (prefill) kernel — **stock behavior, bit-identical to no-graft wheel** | vector (stock) | prefill (stock, WMMA gate untouched) |
| `=1` | **vector (the lever)** | vector (stock) | prefill (stock) |

The read is `static`-cached (one `getenv` per process lifetime); there is no
per-call env cost in either state.

## Verification done on this branch (GPU, guest-a / RX 9070 XT / gfx1201)

- **G1 compile bar**: `hipcc --offload-arch=gfx1201 -fsyntax-only` on the exact
  post-patch file: 0 errors.
- **Chain gate**: full 4-step vendor chain (delta → new files → conv-row-fix →
  this graft) applies `git apply --check`+apply on a fresh 75ca881 checkout;
  post-patch SDPA file and post-fix tape kernel are byte-identical to the
  guest-tree files that build the wheel.
- **G2 kernel gate (env ON)**: per-row bit-exact vs AR S=1 (5 kL × 16 trials
  × 4 rows), qL=1/2 regression rows unchanged, fp32-reference sanity within
  2e-2 (bf16) — results in REPORT.md.
- **G4 lossless (env ON)**: K=3 stream bit-matches AR on the 400-tok battery
  (REPORT.md, arms A1/A3).
- **Default-OFF gate**: env-unset (A0/A2) vs the no-graft baseline wheel
  (REPORT.md).
