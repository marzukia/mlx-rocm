# Stream 4 — QMM M=3 shared-dequant kernel (digest)

Consolidated narrative digest for the L2/H3 QMM lever. The root audit record (REPORT-qmm-m3.md) is consolidated into this digest (deleted by the PR). Box: guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest userland). One wheel (venv-m3k) for both arms; 322 anchor; 3 rounds interleaved (C F / F C / C F); 25 s idle; flock for the whole battery; SCLK sampling degenerated (the amdrocm-10 guest userland exposes no `pp_dpm_sclk` — the 2200-MHz readings predate the userland move, so no clock gate applies this run).

## The lever

At M=3 the small-N M>=2 QMM fallback is the per-row `qmv_wide` kernel, which re-reads and re-dequantizes the weight tile once per output row (FMA:dequant 1:1 per row; 3x weight stream at M=3). The change adds `qmv_wide_mr3_kernel`: a dense wide-load multi-row QMV **WITHOUT x-staging and WITHOUT LDS** — the weight tile is dequantized ONCE per k-tile and MAC'd against all block rows (FMA:dequant M:1, 3:1 at M=3). 4-bit only, non-hoist (inline la/lx, no `__syncthreads`), no co-residency cap; 16B vector x-loads (8x uint4 for 2B T, 4x uint4 for f32) — values and the q-major/j-minor order identical to the per-row scalar traversal. Env-gated (`MLX_ROCM_QMM_M3`, **DEFAULT OFF**): when unset the per-row path runs verbatim (zero behavior change; the C arm is the stock dispatch).

Build: the repo's committed vendor tree is the 745-file subset (authoritative for the tracked files; `vendor.json` pins fork + ref, sha 75ca881); the build tree on guest-a = `vendor-p3` (the box's complete tree that built the record wheel 460c803d) + overwrite of all 745 tracked files from the repo tree @ `bfd2889` (carries the t1251/t1253/t1558 dispatch evolution + the H3 kernel). Wheel `mlx-0.32.3.dev20261104+75ca881`, libmlx md5 `a0057c220d31a0ec87a8b373a97625e8` (18 `qmv_wide_mr3_kernel` symbols) in venv-m3k; venv-p3 (460c803d) verified untouched pre+post. The 12-var config-of-record of the run (tape=1 class — see stream 3 for the hybrid flip): `MLX_MTP=3 MTP6_TAPE=1 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0`.

## Results (same wheel, one build)

- **F vs C: +16.9% t/s (90.58 -> 105.86), GEN wall 5.31 -> 4.54 s (-14.3%).**
- **Fused verify wall: 25.78 -> 21.11 ms (-18.1%)**; effective bandwidth
  177.6 -> 216.9 GB/s (32.5% -> 39.7% of the 546 GB/s floor).
- The win sits in the QMM classes, led by **mlp (-2.99 ms, 45% of the wall)**
  exactly where the shared-dequant lever was aimed (the K=4096 N=12288 gate/up class plus the K=12288 down class, which the t1964 nostage result had left at 45.9% of floor); the mlp class moves from 43.3% to 58.5% of the HBM floor (-26.0% wall). Unmodified control classes (norms, lm_head) are flat, validating the measurement; gdn_fused shows a -15.8% delta outside this change's reach — attributed to run-to-run variation between the C and F probe runs (~1 min apart), not a causal effect; it changes no gate.
- Control gate within tolerance (t/s +0.98%, wall +2.3%); lossless gate
  bit-identical in all runs.

The L2/H3 hypothesis is confirmed at scale: sharing the 4-bit weight-tile dequant across the M rows (FMA:dequant 1:1 -> 3:1) with a staging-free dense wide-load layout is what pays at the K=4096 N=12288 class — the t1964 nostage result (which kept the per-row x reads) had captured only part of this (down -0.73 ms at K=12288). The full M-row sharing (x vectorized too, one weight stream per tile) is the delta.

## Verdict

**ADOPT `MLX_ROCM_QMM_M3=1`** as the config-of-record env change: same-wheel A/B +16.9% t/s and -18.1% fused verify wall, both well past the bar (+5% / -15%); the 105.86 t/s number is the degenerate tape=1 trajectory (see stream 3 for the honest lossless hybrid re-gate at 69.43 t/s with this lever folded in). Default stays OFF in code until the env is folded into the config of record (done by the fold run, stream 3 §7). Remaining ladder from 21.11 ms: gdn_fused (4.33 ms) + mlp (8.50 ms, now 58.5% of floor) + qmv_gdn (4.50 ms).
