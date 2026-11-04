# REPORT: t1334 rows-decode hoist — 7.2.0 A/B (t1910)

**Verdict: NO-ADOPT.** Hoist S=4 fused verify median **40.92 ms vs fresh ctl 39.64 ms
(+3.2% SLOWER)**, 3/3 rep pairs consistent; step arm 42.31 vs 43.05 tok/s (-1.7%).
Bit-exact (all md5s identical HOIST off/on), tokens identical (all 612, not just FIRST20).
Gate (adopt on >= 5% fused-verify improvement) not met.

## 1. The t1858 "hoist" number was a no-op A/B (confound found 2026-11-04)

t1858 recorded `hoist2=41.20 vs ctl=39.89 ms` in the 7.2.0 tree and concluded "not
positive". That A/B was invalid: in the 7.2.0 box tree, `qmm.hip`'s `wide_ok` gate
required `has_bias`, and the Qwen3-9B mlx model is fully bias-free (no bias arrays).
The R4b bias-free open — `has_bias` -> `(has_bias || (M >= 2 && M <= 4))`, already
positive-adopted in the repo lineage (run-952, r4v, 2026-11-02) — was missing from
the box tree. With the gate closed, both the base rows-msg kernel AND the t1334
hoist kernel were dead code for S=4 verify in both arms: `HOIST=1` changed nothing.
`41.20 vs 39.89` was pure run-to-run noise (rep spread in that battery was 39.78-
39.89; 41.20 was a single unreplicated run). The same no-op structure applied to the
round-2 (kfix) measurement only in the sense that kfix's 33% regression was real
kernel code; the "hoist" arm of that battery was inert.

"Completing t1334" therefore meant: graft the one-line R4b open into the 7.2.0 box
tree (backup: `qmm.hip.bak-t1910`), then build ONE wheel whose arms differ only by
`MLX_ROCM_QMV_ROWS_HOIST` (unset = base rows-msg kernel; set = t1334 hoist kernel).
The hoist kernel itself (`qmv_rows_msg_hoist_kernel`) was already present in 75ca881
(unmodified); the t1334 WIP that existed in the 7.2 tree was only the dispatch hook
(`use_hoist` + `LAUNCH_QMV_ROWS_MSG_H`) plus the missing R4b open.

## 2. Protocol (box neon, 2026-11-04 09:23-09:42 UTC)

- ROCm 7.2.0 (guest userland at /opt/rocm-7.2.0), gfx1201 RX 9070 XT (PCI 1002:7550,
  32 CU, wave32), bare-metal tree (DPM, not the guest 2200 MHz lock — see SCLK note).
- One wheel from the grafted tree: `mlx-0.32.3.dev20261104+75ca881`,
  `libmlx.so` md5 `d58aca91c4de2fd9deb461ac36fb9053` (t1858 ctl wheel md5
  `d0b892759c4f9c92c468582d37448334` for reference; same 7.2 tree, pre-graft state).
- GPU locked (`~/gpu.lock`, flock, acquired iter=1); 25 s idle before every run;
  SCLK sampled at 1 Hz throughout, phase-tagged.
- Arms, 3 reps, interleaved per rep (ctl-r, hoist-r):
  - fused verify S=4: `bench/roofline-verify-perop.py` (512 AR warmup, median of 5
    internal reps), env `MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1
    MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1` (t1858 recipe; R4b open makes the
    rows kernel active for bias-free M=2..4; HOIST toggles base vs hoist kernel).
  - step: `bench/mtp-bench6-phase.py` MTP4 r5 arm (MTP_MAX_TOKENS=512).
  - bit-exact: `bench/qmm_bitexact_ab.py` HOIST off vs on (all QMM classes).
- Bench files: repo 9e4e289 state (md5-verified on box).

## 3. Results (median of 3 rep medians)

| metric | ctl (HOIST off) | hoist (HOIST=1) | delta |
|---|---|---|---|
| fused verify S=4 (ms) | **39.64** (39.95/39.63/39.64) | **40.92** (40.69/40.92/41.34) | **+3.2% (slower)** |
| step r5 (tok/s) | 43.05 (43.11/43.01/43.05) | 42.31 (42.35/42.31/42.28) | -1.7% |
| per-class mlp (ms) | 11.75/11.75/11.77 | 12.44/12.68/12.53 | +5-8% |
| per-class qmv_gdn (ms) | 6.17/6.14/6.06 | 6.48/6.46/6.53 | +5% |
| gdn_fused / attn / norms | neutral (within 1%) | neutral | - |
| bit-exact (all md5s) | match | match | row-exact holds |
| generated tokens | 612, identical | 612, identical | neutral (FIRST20 + full) |

Fresh ctl 39.64 vs stale 4h-backstop anchor 39.78: -0.4% (tree reproduces the anchor
within noise). Hoist 40.92 vs the invalid t1858 "hoist2" 41.20: same loss signal —
the stale number was not an artifact after all, but it was never a hoist
measurement.

SCLK: 1 Hz sampling throughout; both arms at max DPM level 5 while busy (matmul
probe on the same box: steady state 2633-2648 MHz; idle level 1 = 2627 MHz). The
2200 MHz lock expected from the guest-a recipe does not exist in this bare-metal
tree (DPM); arms are symmetric (same box, same run, interleaved), so the gate is
clock-neutral. (rocm-smi reports a second device, PCI 1002:13C0 at 2200 MHz — not
the bound device; mlx binds device 0 = the 9070 XT.)

## 4. Mechanism read

The t1334 hoist caches the dequantized weight tile (`wdec[32]`, 4-column unroll
-> up to 128 VGPR live) so each 4/8-bit weight word is decoded once instead of
once per row, and re-reads the staged `xn` per row. Measured outcome: the base
rows-msg kernel is NOT decode-ALU-bound at S=4 (K=4096/12288, N>=16384); the
extra register pressure (fewer resident wavefronts) plus the 4x `xn` re-reads cost
more than the ~3x decode-ALU saving returns. Hoist loses 5-8% in the QMM-bearing
classes (mlp, qmv_gdn) and 3.2% on fused verify. Row-exactness holds (bit-exact
match, identical tokens) — this is a pure performance loss, not a correctness
issue.

## 5. Disposition

- t1334 hoist kernel stays **opt-in** (`MLX_ROCM_QMV_ROWS_HOIST`, default off); no
  default behavior change. No further t1334 work is warranted on this evidence.
- The R4b bias-free open is the one-line dispatch completion that makes the rows
  (base) kernel active for bias-free S=2..4; the fresh ctl arm shows it is neutral-
  positive (39.64 vs 39.78 per-row anchor, -0.4%). It belongs in the base tree in
  its own small PR (it is the same open as lineage run-952).
- The base rows-msg kernel (t1253) remains the S=4 verify path.

## 6. Artifacts

- `patches/t1334-rows-decode-hoist-72tree.diff`: 7.2.0 box tree state (75ca881 +
  t1858 grafts, R4b open, t1334 dispatch hook) — the exact tree the one wheel was
  built from. qmm.hip only; the rows/hoist kernel files are unmodified in 75ca881.
- Box: `~/t1910/` (log + out/ with all 16 run logs + sclk.log), `/tmp/hoist-wheel/`,
  venv-hoist; pre-graft qmm.hip at `qmm.hip.bak-t1910` (md5 1ca31eeb...).
- Wheel: md5 `66a6ada9...` (pip sha256), libmlx.so md5 `d58aca91...`.
