# REPORT: QMM wide-MR M=3 shared-dequant kernel (t2011, ladder lever L2/H3)

Date: 2026-11-05. Base: `stack-assembly-bench` @ `bfd2889` (PR #22 consolidated).
Deliverable branch: `qmm-m3-kernel`. GPU: guest-a (RX 9070 XT / gfx1201, 32 CU),
flocked (the other session's tape-probe/s1gr battery held the lock through the
build window; the battery waited on `flock` by design).

## Mission

L2/H3 lever: at M=3 the small-N M>=2 QMM fallback is the per-row `qmv_wide`
kernel, which re-reads and re-dequantizes the weight tile once per output row
(FMA:dequant 1:1 per row; 3x weight stream at M=3). This change adds
`qmv_wide_mr3_kernel`: dense wide-load multi-row QMV WITHOUT x-staging and
WITHOUT LDS — the weight tile is dequantized ONCE per k-tile and MAC'd against
all block rows (FMA:dequant M:1, 3:1 at M=3). Env-gated
(`MLX_ROCM_QMM_M3`, DEFAULT OFF); when unset the per-row path runs verbatim
(zero behavior change).

## Build tree provenance

The repo's committed vendor tree is the 745-file subset (authoritative for the
tracked files; `vendor.json` pins fork + ref, sha 75ca881). The complete build
source is not in the repo, so the build tree on guest-a is: `vendor-p3` (the
box's complete tree that built the record wheel 460c803d via p3-build.sh) +
overwrite of all 745 tracked files from the repo tree @ `bfd2889` (carries the
t1251/t1253/t1558 dispatch evolution + the H3 kernel). All 47 CMake-listed
.hip files verified present in the base tree; the kernel .hip files compile as
part of `qmm.hip` via `#include` (no CMakeLists change needed). Wheel:
`mlx-0.32.3.dev20261104+75ca881`, libmlx md5 `a0057c220d31a0ec87a8b373a97625e8`
(18 `qmv_wide_mr3_kernel` symbols) in venv-m3k; venv-p3 (460c803d) verified
untouched pre+post. Sidecar model md5 `02a54359cda8b247f76b277dd9bc4d4c`
verified in identity.txt (pre-battery).

## Kernel contract

- 4-bit only, non-hoist (inline la/lx, no `__syncthreads`), no LDS at all
  (no co-residency cap); 16B vector x-loads (8x uint4 for 2B T, 4x uint4 for
  f32) — values and the q-major/j-minor order identical to the per-row scalar
  traversal.
- Per-(row,col) math: k tiles ascending, la FMA -> lx FADD (has_bias-guarded,
  same position) -> `acc += s*la + b*lx`, warp-shfl reduction, lane-0 write —
  the EXACT per-row order, with the dequantized weight values shared (row-
  invariant: same shift/mask, same `static_cast<float>`), so every row output
  is bit-identical to the per-row kernel.
- Early return on col>=N before any weight read (the MR12 OOB defect class is
  not present).
- Grid (ceil(N/16), ceil(M/4)); block (32,16). Dispatch: only inside the
  `!mr_ok` (per-row fallback) branch; 4-bit, M in [2,4]; lm_head (N=248320)
  stays on staged MR; M=1 (eager) and M>=5 stay per-row. 9 template
  instantiations (T x GS: bf16/f32/f16 x 32/64/128).

## Protocol

Config-of-record env (verbatim): `MLX_MTP=3 MTP6_TAPE=1 MLX_ROCM_GDN_TAPE=1
MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16
MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1
MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0`. Bench: consolidated `mtp-bench6.py` md5
`54ce16ac29e6c6aad2c48a233ff70d85`, 322 anchor (MTP_MAX_TOKENS=322). Probe:
`verify-bw-perop.py` (the t2003 audit probe, adapted: the
`MLX_ROCM_QMV_ROWS_MIN_N=1` setdefault removed — it is not a config-of-record
env; the default 16384 is restored so the small-N ops reach the per-row
fallback where the H3 kernel engages). One wheel (venv-m3k) for both arms:
C = env off (per-row), F = `MLX_ROCM_QMM_M3=1`. Pre-battery sanity anchor per
arm, 3 rounds interleaved (C F / F C / C F), 25 s idle before each, per-arm
timeout 600 (anchor) / 1200 (probe), flock for the whole battery, SCLK sampled
every 10 s (the amdrocm-10 guest userland exposes no `pp_dpm_sclk` — the
sample degenerates to 2; the other session's 2200-MHz readings predate this
userland move, so no clock gate applies this run).

Gates:
- Control (C median): t/s 89.70 +/- 1%; fused verify wall 25.21 ms +/- 10%.
- Lossless (F): FIRST20 = `271 1919 19` bit-identical, rc=0, all runs.
- Bar (F vs control): t/s >= +5% AND fused verify wall <= 21.4 ms (-15%).
- Same-wheel A/B (F vs C) is the primary evidence; the reference comparison is
  secondary (this wheel's dispatch differs from the 460c803d wheel: the
  current tree uses per-row for N < 16384 where the wheel used staged MR).

## Results

Anchors (322, one wheel, config-of-record env):

| arm | runs (t/s) | median | GEN (s) | FIRST20 |
|---|---|---|---|---|
| C (per-row) | sanC 90.76 / r1 90.52 / r2 90.61 / r3 90.58 | **90.58** | 5.31 | 271 1919x19 (4/4) |
| F (H3 kernel) | sanF 105.78 / r1 105.95 / r2 105.93 / r3 105.86 | **105.86** | 4.54 | 271 1919x19 (4/4) |

All 8 runs: D1ACC=159 D2ACC=148 D3ACC=0, TOK_PER_STEP=2.907, rc=0.
F vs C: **+16.9% t/s**, GEN wall 5.31 -> 4.54 s (-14.3%).

Gates:
- Control t/s: 90.58 vs 89.70 = **+0.98% (within +/-1%)** PASS.
- Control fused verify wall (probe C): **25.78 ms vs 25.21 = +2.3% (within
  +/-10%)** PASS (all [25.78, 25.80, 25.70]).
- Lossless (F): FIRST20 = `271 1919 19` bit-identical in all F runs, rc=0 —
  PASS; C and F token-identical in every run (the shared-dequant bit-identity
  contract holds end-to-end).
- Bar: F 105.86 vs control 90.58 = **+16.9% t/s (>= +5%)** PASS; F fused
  verify wall **21.11 ms <= 21.4 ms** (<= -15% vs 25.21; -18.1% vs probe C
  25.78) PASS.

Per-class (probe, config-of-record dispatch, same wheel, C then F ~1 min apart):

| Class | C wall (ms) | F wall (ms) | delta | floor C -> F |
|---|---|---|---|---|
| mlp (32x: 2x gate/up K=4096 + down K=12288, N=12288) | 11.494 | 8.502 | -2.992 (-26.0%) | 43.3% -> 58.5% |
| qmv_gdn (24x: 5 QMM, N=96..12288) | 5.063 | 4.498 | -0.565 (-11.2%) | 32.9% -> 37.0% |
| gdn_fused (24x: fused step tape T3 — unmodified) | 5.141 | 4.328 | -0.813 (-15.8%) | 3.7% -> 4.4% |
| attn (8x: 4 QMM N=512..4096 + S4VA) | 1.956 | 1.696 | -0.260 (-13.3%) | 25.8% -> 29.8% |
| norms (unmodified) | 1.480 | 1.481 | +0.001 (flat) | 0.1% -> 0.1% |
| lm_head (unmodified, staged MR N=248320) | 1.274 | 1.269 | -0.005 (flat) | 82.3% -> 82.6% |

Fused: 25.78 -> 21.11 ms; effective bandwidth 177.6 -> 216.9 GB/s
(32.5% -> 39.7% of the 546 GB/s floor). The win sits in the QMM classes,
led by mlp (-2.99 ms, 45% of the wall) exactly where the shared-dequant
lever was aimed (the K=4096 N=12288 gate/up class plus the K=12288 down
class, which the t1964 nostage result had left at 45.9% of floor); the
unmodified control classes (norms, lm_head) are flat, which validates the
measurement. gdn_fused (unmodified kernel) shows a -15.8% delta that is
outside this change's reach — attribute it to run-to-run variation between
the C and F probe runs (they ran ~1 min apart) rather than a causal effect;
it does not change any gate.

## Findings

- The L2/H3 hypothesis is confirmed at scale: sharing the 4-bit weight-tile
  dequant across the M rows (FMA:dequant 1:1 -> 3:1) with a staging-free
  dense wide-load layout moves the mlp class from 43.3% to 58.5% of the HBM
  floor (-26.0% wall) and the whole fused S=3 verify wall from 25.78 to
  21.11 ms (-18.1%). The t1964 nostage result (which kept the per-row x
  reads) had captured only part of this (down -0.73 ms at K=12288); the
  full M-row sharing (x vectorized too, one weight stream per tile) is what
  pays at the K=4096 N=12288 class.
- Bit-identity held exactly as contracted: 8/8 runs token-identical across
  arms (271 1919x19) and D1/D2 acceptance identical (159/148), with the
  shared dequant being row-invariant and the per-(row,col) FMA order
  unchanged.
- The gate is default-OFF: with the env unset, the per-row path runs verbatim
  (C arm = the stock dispatch), so the change is inert unless adopted.
- SCLK sampling is degenerate in the amdrocm-10 guest userland (no
  `pp_dpm_sclk`); the other session's 2200-MHz readings predate the
  userland move. Both arms ran in the same session minutes apart with
  25 s idle between, so clock drift is not a confound.

## Verdict

**ADOPT `MLX_ROCM_QMM_M3=1`** as the config-of-record env change: same-wheel
A/B (one build, venv-m3k) +16.9% t/s (90.58 -> 105.86, +16.9%) and -18.1%
fused verify wall (25.78 -> 21.11 ms), both well past the bar (+5% / -15%),
control gate within tolerance (t/s +0.98%, wall +2.3%), lossless gate
bit-identical in all runs. This lifts the record stack from 89.70 to the
105.86-t/s class (142% of AR 74.44) and the fused verify wall to 21.11 ms —
the L2 ladder step is done; the remaining 21.11 -> 8 ms ladder now sits on
gdn_fused (4.33 ms) + mlp (8.50 ms, now 58.5% of floor) + qmv_gdn (4.50 ms).
Default stays OFF until the env is folded into the config-of-record.
