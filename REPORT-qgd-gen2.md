# REPORT-qgd-gen2 — the four gen-2 QMM variants on the qmv_gdn class (M=1 per-row QMM)

Date: 2026-11-05 · Branch: `qgd-gen2-probe` (off `mlp-gen2-probe` a03b5e3) · Box: guest-a (RX 9070 XT, gfx1201, ROCm 10.1)

## Charter

Port the four gen-2 QMM memory-structure variants (V1 LDS x-staging, V2 k-tile double buffer, V3 32B weight loads, V4 batched-x load order) onto the kernel serving the
qmv_gdn class — GDN 4 in-proj + out_proj QMMs, 24 layers, M=1, 4-bit group-64
affine, no bias — 4.59 ms / 36.3% of the 546 GB/s floor, via `MLX_ROCM_QGD_V1..V4`
(default OFF), build `venv-qgd` on the neon box, 5-arm per-class probe (C + V1..V4);
BAR: a variant passes iff its qmv_gdn class reaches ≥ C's floor AND the bench shows
≥ +1% same-session; else FAIL-as-chartered with a ceiling analysis.

## Headline

**FAIL-as-chartered.** No variant beats C's floor at the +1% bar. All four arms are
within the 1–2% same-session noise band (the no-op control arm Q1 itself moves
+0.4%). The class is **latency/occupancy-bound, not HBM-bound** — the gen-2
memory-structure changes (load width/order) cannot move it; the ceiling fix is a
*parallelism* change (K-split), not a memory-structure change.

## The misdiagnosis (session A) — and how it was caught

Session A ported the variants onto the `qmv_wide` per-row kernel (the documented
M=1 path). The session-A probe showed **zero signal** — all four arms bit-for-time
identical to C (qmv_gdn 4.473–4.497 ms vs C 4.483), with *no* regression on V1/V3
(the mlp-gen2 analogs of those were 2.3×/1.85×). That is not "neutral variants";
it is "variants never ran."

Root cause: `wide_ok` requires `has_bias || (M>=2 && M<=4)` — at M=1 the per-row
wide kernel only serves **with-bias** ops. The GDN projections are bias-free
(`in_proj_qkv bias: None`, scales (8192, 64), group 64), so the class-under-test
actually runs the **tiled QMV kernel** (`qmv_tiled_kernel`: block (32, tile_n), one
warp per output column, BSK=512 k-tiles, cooperative LDS x staging, per-lane 2×uint32
streaming weight loads, fused dequant + 4 partial dot products, scale/bias once per
group). Verified empirically: `MLX_ROCM_QMV_NO_TILED=1` slows the class (4.479→
4.509 ms) — the per-row fallback is slightly *slower*, i.e. tiled is the live path.

Session B re-ported the four variants onto the tiled structure (bit-identical FMA
order preserved in all four):

| arm | tiled port |
|---|---|
| Q1 | **no-op control** — V1's x-staging is already the stock tiled design (x_shared staged once per k-tile, shared by all tile_n column warps) |
| Q2 | k-tile double buffer: next tile's 2×uint32 slice issued before the current tile's dequant; 16 float weights pre-dequantized into A/B alternating buffers; FMA order exactly `dequant_and_dot4` (pack p, value i) |
| Q3 | 16B weight window (the m3-V3 32B pattern at the tiled 8B slice width): two 8B-aligned volatile 32-bit pairs; second half read only when the window is in-row (`k_val + 32 <= K`, values never used in the math — access pattern only) and elision-proofed with asm |
| Q4 | batched-x load order: x staging issued as 16B vector batches (8×bf16) wherever the chunk start is 16B-aligned, scalar for the rest; before the weight loads as in stock |

Dispatch: inside the tiled gate (which already enforces use_tiled, 4/8-bit, bf16/f16 x,
non-batched, tile_n≥4, affine), `qgdok = bits==4 && M==1 && any QGD env set`.
Default OFF ⇒ stock tiled kernel runs verbatim. 8-bit and unset fall through to stock.

## Verification

- Sidecar md5-of-record `02a54359cda8b247f76b277dd9bc4d4c` pre/post ✓
- venv-gen2 `55b164e8…` (the mlp-gen2 A3 wheel — all four `MLX_ROCM_QMM_M3_V*` strings
  present, used as the venv-qgd base), venv-m3k `a0057c22…`, venv-p3 `460c803d…`
  — all pre==post ✓ (never touched)
- venv-qgd = copy of venv-gen2 (copy md5 verified `55b164e8…`) + force-reinstall of
  wheel `mlx-0.32.3.dev20261105+2ce5d54`; post md5 `36037cae6d551ec107d550ce3bd1e9ad`
  (differs ✓); all four `MLX_ROCM_QGD_V*` env strings present; 66× `qmv_tiled_qgd_v4_kernel`
  symbol references in libmlx ✓
- Bit-exact outputs: 5 GDN projections (in_proj_qkv/z/b/a, out_proj) with fixed seed,
  per-arm processes — **identical** hash + first values across DEFAULT, Q1, Q2, Q3, Q4 ✓
- Probe FUSED_VERIFY passes in every arm (rc=0) ✓

## Probe results (per-class, per-run, 25 s idle between arms, 13 config-of-record envs on in every arm)

Session B (the tiled-target build), qmv_gdn class (0.9095 GB):

| arm | ms | GB/s | % floor | Δ vs C |
|---|---|---|---|---|
| C (stock tiled) | 4.507 | 201.8 | 37.0% | — |
| Q1 (no-op control) | 4.523 | 201.1 | 36.8% | +0.4% (noise floor) |
| Q2 (double buffer) | 4.466 | 203.7 | 37.3% | −0.9% |
| Q3 (16B window) | 4.499 | 202.2 | 37.0% | −0.2% |
| Q4 (batched-x) | 4.444 | 204.7 | 37.5% | −1.4% |

lm_head clock control: 451.8–454.0 GB/s (0.5% spread) across all arms ✓. mlp class
(327.8/330.8/324.7/324.3/325.6 GB/s) unchanged within 1% — the QGD dispatch does not
touch the mlp class ✓.

Same-session A/B (3×C vs 3×Q4, the best arm): C = 4.520 / 4.447 / 4.452 (mean 4.473);
Q4 = 4.458 / 4.484 / 4.484 (mean 4.475). **Q4 wins 1 of 3; mean is a dead tie.**
The single-run −1.4% does not reproduce. BAR not met → no end-to-end A/B.

Session A (the wide-target build, for the record): all four arms 4.473–4.497 ms vs C
4.483 — zero signal, which is what exposed the misdiagnosis.

## Ceiling analysis — why the qmv_gdn class sits at 37.2%

The class is **latency/occupancy-bound**, not HBM-bound — which is exactly why the
four *memory-structure* variants (load width and ordering) produce sub-noise effects:

1. **Parallelism per wave.** Stock tiled launch: grid (M=1, min(n_tiles,
   CUs×blocks/CU)); for N=8192, tile_n=8: n_tiles=1024, persistent grid_y≈512 → 512
   blocks, each **grid-striding 2 column tiles serially**, each tile a 2-`__syncthreads`
   LDS staging of 512 floats + a short 16-FMA per-lane chain + one 2×uint32
   nontemporal weight load **with no software pipelining** (load→wait→dequant→FMA→next).
   Per k-tile in-flight weight traffic: 512 blocks × 256 B = 128 KB; saturating
   546 GB/s at ~500 ns HBM latency needs ~273 KB in flight. The kernel runs at ~47%
   of the outstanding-load requirement, and the per-tile sync/staging overhead sits in
   the serial chain. (Q2's prefetch doubles per-block in-flight traffic but the block
   count — not the per-block chain — is what caps the wave; measured: −0.9%, a wash.)
2. **The small-N projections** (in_proj_b/a, N=32 → 4 column tiles = 4 blocks) run on
   4 blocks of 256 threads — ~0.8% of the 512-block wave — but they carry only
   ~128 KB of the 910 MB class bytes, so they are not the class's ceiling; they are
   a separate (small) tax.
3. **x re-reads.** Every block re-stages the 8 KB x row per k-tile: 512 blocks × 8
   tiles × 1 KB ≈ 4 MB of x traffic per big projection (L2-resident, not HBM) — LDS
   bandwidth + 2 syncs per tile, inside the serial chain. Q4's 16B vector staging
   (−1.4% single-run, non-reproducing) is the only arm that touched this, and it
   touched it within noise.
4. **Contrast with the classes above it:** lm_head (GEMM, 452 GB/s, 82.8% floor) has
   full 2D tiling; the mlp class (mr3, M=2..4, 326–328 GB/s, 60% floor) amortizes the
   per-row chain over M rows; the tiled M=1 kernel has neither.

**The ceiling fix is a parallelism change, not a memory-structure change:** K-split
the M=1 path — e.g. 4× K-splits turns 512 blocks×(2 tiles×8 k-iter) into 4096
blocks×1 tile: 4× the blocks, ~512 KB in flight (1.9× the saturation requirement),
no grid-stride serialization, one sync pair per block. (Or an mr3-style M-tiling
is unavailable at M=1 by definition; K-split is the M=1 analog of M-tiling.) That
is the next chartered probe for this class. If K-split cannot reach the floor, the
residual is the 2-sync-per-tile staging pattern itself (a K-split removes it too:
one staging per block).

## Deliverables

- `vendor/.../quantized/qmv_tiled_qgd_gen2_kernel.hip` — the four tiled variants
  (4-bit; Q1 no-op control / Q2 double buffer / Q3 16B window / Q4 batched-x).
- `vendor/.../quantized/qmm.hip` — QGD dispatch in the tiled gate (bits==4, M==1,
  env set; all-default-OFF ⇒ stock verbatim). (Session A's wide-section QGD dispatch
  + per-row wide QGD kernel were removed as dead for this model — kept in git
  history at e94b0b4.)
- `scripts/qgd-session-a.sh`, `scripts/qgd-session-b.sh` (session B also fixed
  session A's finding; session-B probe output landed in `session-a.log` via an
  un-sed'd `L=` line — the raw numbers above are from that log, timestamps 04:24–04:28Z).
- Env vars: `MLX_ROCM_QGD_V1..V4`, all default OFF.

## Verdict

**FAIL-as-chartered.** Q1 (control) +0.4%, Q2 −0.9%, Q3 −0.2%, Q4 −1.4% (non-reproducing;
A/B mean 4.475 vs 4.473). No variant reaches ≥ C floor at the +1% bar. The qmv_gdn
class ceiling (37.2% floor / 203 GB/s) is set by per-block K-chain serialization and insufficient in-flight loads (128 KB vs ~273 KB to saturate), not by load width or x ordering — the gen-2 memory-structure family is exhausted for this class. Next
chartered probe: K-split M=1 (4× K-splits → 4096 blocks, one tile per block).
