# GDN fused T=2 (MTP verify) — final report

**Verdict: the fused GDN kernel now handles T=2 (S=2 MTP verify) bit-exactly and losslessly; Gate 1 (byte-identical tokens) and Gate 2 (fused-AR unchanged) PASS. Gate 3 (MTP-on exceeding the 81.37 tok/s fused-AR baseline) is NOT met: MTP-on measures 72.7 tok/s, 11% below AR, at a 73.4% draft-acceptance rate — the fused path is the correct one and delivers the full fusion gain (+6.6% vs unfused MTP), but the acceptance rate at this verify cost cannot beat AR (physical ceiling: 88% of AR even at 100% acceptance).**

Guest: guest-a, RX 9070 XT / gfx1201, ROCm 10.1. Model: Qwen3-9B (MLX), MTP sidecar `model-mtp.safetensors` (sha256 `edef9ec6…`). Wheel built from the vendor clone (`$MLX_VENDOR_DIR`, 75ca881 + PR3-HYBRID dirty state + this T=2 graft), installed in `venv-fused2`.

## 1. What changed

- `patches/gdn_fused_step/gdn_fused_step.hip` — the fused GDN kernel (single `for (t = 0; t < TMAX; ++t)` structure over all 5 phases; h state loaded once and updated row-by-row in registers, state written after the loop — verbatim r4d recurrent geometry). `TMAX` is a **template constant** (1 or 2, selected at launch): for T=1 the loop unrolls and the conv-window ternaries fold, so the decode path compiles to exactly the original straight-line T=1 kernel (a runtime loop bound left a ~0.3% regression in the latency-bound AR loop; unrolling recovered it).
- The op's host side force-copies `q3/k3/v3` to packed layout **only for T>1** (see §3, the bug this fixed); T=1 takes the input references with zero extra host work.
- Fallback scope: T=1 falls back to the 4-op chain (safe at T=1 — the chain is 1-D-only in `gdn_prep`); T>1 with the fused path disabled (FUSE=0, a CPU stream, or Dk/Dv != 128) throws a clean `std::invalid_argument` (the chain would GPU-fault at T>1); T=0 (empty input) is rejected. The old `MLX_ROCM_GDN_FUSE_T2=0` kill-switch (which routed T=2 back into that broken chain) was removed from `use_fallback`; lossless A/B is `FUSE=0` vs `FUSE=1`.
- `patches/qwen35_fuse_branch/qwen35_fuse_branch.diff` — model F-branch gate `S == 1` → `S <= 2`, so MTP verify hits the fused op.
- `bench/` — `mtp-bench3-side.py` (sidecar support, `GDN_FUSE_CALLS` counter), `gdn_fused_t2_gates.sh` (the 3 gates), `gdn_fused_t2_probe.py` (op-level bit-exactness), plus the development diagnostic scripts (excluded from lint in `pyproject.toml`, kept as as-run evidence).

## 2. Gate results (final wheel)

| gate | config | result |
|---|---|---|
| 1 lossless A/B | MTP=2, FUSE=0 vs FUSE=1 | **PASS** — all 1024 tokens byte-identical (md5 `b1b35eca…` both arms), 376/512 accepted in both arms, 68.21 vs 72.69 tok/s, 18816 fused calls in the fused arm |
| 2 fused-AR unchanged | MTP=0, FUSE=1 | **PASS** — 80.9–81.1 tok/s, equal to the pre-T2 T=1 kernel measured in the same environment (81.05–81.17); 24576 fused calls; the original 81.37 figure was measured in a slightly faster environment (the old kernel now also sits at ~81.1) |
| 3 MTP-on speed | MTP=2, FUSE=1 | **NOT MET** — 72.7 tok/s < 81.37 |

Op-level probe (`gdn_fused_t2_probe.py`, 8 checks, all PASS): fused T=1 and T=2 out+state bit-exact vs the model default chain; T=2 ≡ two T=1 steps for both the chain and the fused op (invariance).

## 3. The T=2 bug found and fixed

The first T=2 build was bit-exact at T=1 but diverged at T=2 (1–2 ulp on ~90% of row-1 intermediates across all 32 heads, 100% of SSM state elements). Every per-phase computation verified bit-exact for row 0, and window-shift probes confirmed correct row reading — until the input layout was checked: the model F-branch passes `q3 = qkv[:, :, :key_dim]` etc., **row-strided slices** of the `[1, T, 8192]` qkv buffer (row stride 8192 ≠ packed 2048). The kernel addresses them packed (`t * key_dim + ch`):

- T=1 reads only row 0, at offset 0 — indistinguishable from packed, so the bug is invisible (the pre-T2 wheel was correct by this accident);
- T=2 row 1 reads buffer[2048:4096] — row 0's k-channel region.

Fix: the op force-copies the sliced inputs to packed layout for T>1 (T=1 keeps zero extra launches — an earlier unconditional-copy version measurably regressed the AR loop, confirming the host-encode latency sensitivity).

## 4. Why Gate 3 is not met (and what it would take)

MTP-on (fused) = 72.7 tok/s vs fused-AR = 81.1. The fused verify step costs 2.27× an AR step (14.1 s / 512 verifies vs 12.5 s / 1024 AR tokens). At the measured 73.4% acceptance, expected throughput is (2·0.734 + 0.266) / 2.27 = 0.77× AR. The ceiling even at 100% acceptance is 2 / 2.27 = 0.88× AR: **with this verify cost, MTP-on cannot beat AR at any acceptance rate.** Beating 81.37 requires either a verify step under ~2× an AR step (deeper verify fusion — the MTP draft module's full-attention forward over 2 tokens is currently unfused) or a materially higher acceptance rate (better drafts). The fusion work delivered here is the prerequisite for both: the S=2 verify now takes the fused GDN path with the full GDN-chain gain (68.2 → 72.7 tok/s, +6.6%).

## 5. Verification trail

- Op-level bit-exactness: `gdn_fused_t2_probe.py` 8/8 PASS on the final wheel (T=1/T=2 vs default chain, invariance chain and fused).
- Development bisection (kept in `bench/`): T=1-vs-T=2 state discrimination (`t1t2_op_disc.py`), per-row intermediate captures via the `P2_GDN_DEBUG=1` channel (`dbg_cmp*.py`), window-shift probes (`window_bisect.py`), real-value op comparison (`op_t2_realvals.py`), in-process A/B cache-snapshot bisect (`gdn_t2_bisect.py`) — the snapshot had to be copy-based with dtype tracking (in-place `cache.advance()` mutation corrupts raw-array snapshots).
- A/B kernel swap (old T=1 kernel vs new T=2 kernel, same tree) localized the 0.3% T=1 gap to the runtime t-loop; the TMAX template fix closed it.
- `tests/test_gdn_fuse.py` 11/11 (updated for the T=2 structure; includes the hermetic patch well-formedness test), full hermetic suite 23 passed / 1 skipped (GPU), ruff clean, size gate OK.

## 6. Reproduce (on guest-a)

```bash
# wheel (the vendor clone is the $MLX_VENDOR_DIR target of scripts/build-wheel.sh)
cd "$MLX_VENDOR_DIR" && python3 setup.py bdist_wheel
"$VENV/bin/pip" install --force-reinstall --no-deps \
  dist/mlx-0.32.3.dev20261101+75ca881-cp314-cp314-linux_x86_64.whl
# gates (see bench/gdn_fused_t2_gates.sh for the full script)
export MODELS=<models dir> MTP_SIDECAR=model-mtp.safetensors
# Gate 1: MLX_MTP=2 MLX_ROCM_GDN_FUSE=0  vs  MLX_MTP=2 MLX_ROCM_GDN_FUSE=1
# Gate 2: MLX_MTP=0 MLX_ROCM_GDN_FUSE=1
# Gate 3: MLX_MTP=2 MLX_ROCM_GDN_FUSE=1   (compare vs Gate 2)
```

Note: `mtp-bench3-side.py`'s sidecar guard compares `MTP_SIDECAR` against basenames — pass `model-mtp.safetensors`, not a path; a prior run left the sidecar renamed `*.disabled` (restore with `mv` before Gate 1).

## 7. Follow-ups (out of scope here)

1. Gate 3 unmet (§4): fuse the MTP draft module's 2-token forward, or improve draft acceptance; re-measure Gate 3 after either.
2. `mx.eval()` returns None in this build (sync flush only) — numpy snapshots must go through `np.asarray(x.astype(mx.float32))`.
3. The 4-op fallback chain remains in the tree as the T=1 `FUSE=0` path; it is safe at T=1, T=0 is rejected before it, and it must not be reached at T>1 (the op throws; the model F-branch only routes T≤2 to the fused op).
4. `WORKSTATE.md` / `WALL-NOTE.md` updates for the next session; the vendor re-pin PR when the base moves.
