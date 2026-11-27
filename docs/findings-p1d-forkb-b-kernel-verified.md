# FINDINGS - P1-d fork-(b): Q2_K file-layout vec-dot kernel — BUILT, VERIFIED, GATE MIS-SPECIFIED

Date: 2026-11-27 08:05Z (neon clock)
Agent: option-B build session (leg-3)
Revised: 2026-11-27 (WAVE-16 docs fix per cloudy's consolidated ruling at the wave close: E0 line, xq2 replica-deviation note, min-term decomposition section, the irreducible-floor wording, the routing line)
Vehicle: neon (guest-a tailnet peer), RX 9070 XT / gfx1201, ROCm 10.1 guest userland
Box state at measurement: GPU idle (load 0.01), GPU temp 37.9C (hwmon, pre-run), uptime 13h

## Verdict (bottom line)

**The option-B kernel is correct.** It reads the Q2_K file layout directly and matches
an exact numpy replication of the in-tree d2s6 x-quantizer to **2.7e-5 max|d|
(rel 6.6e-5)** — pure f32 accumulation noise. The G1 gate as literally specified
(M=1 total rel <= 1e-3) **cannot be met by ANY mmq path** (option A included),
because the mmq architecture quantizes x to q8_1 d2s6, whose floor on the G1
random-x battery is **2.29e-1** (verified to 4 digits). Gate: **FAIL (literal) /
PASS (kernel-own-error interpretation, 6.6e-5 << 1e-3).** Named next step below.

## G1 gate numbers (DELIVERABLE: build #11 clean probes-off wheel, 0 probe lines in output,
2026-11-27 08:24Z; identical numbers on probe-on build #10)

| check | result | status |
|---|---|---|
| KQFG-CHECK M=2 (regression) | max|d| = 1.013e-06, rel = 2.454e-06 | PASS (no regression) |
| G1v2-MMQ M=1 random-x total rel | 2.290e-01 (tol 1e-3 literal) | FAIL literal / floor-bound |
| G1v2-MMQ M=1 kernel-own error (vs exact x-quant ref) | 2.722e-05, rel = 6.592e-05 | PASS (<< 1e-3) |
| G1v2-MMQ E0 unit probe | max|d| = 1.409e-02 measured; the exact AS-BUILT structural attribution = 0.0625*(W[0,:] + SUM of the 15 sibling rows) = 0.0625*0.225531 = 1.40957e-02 (to 4 digits; the min-term is the signal at E0, so the error is x-quant structure, not kernel-own, and there is no contradiction with the 6.6e-5 kernel-own class). The original line's "= 0.0625*max|W[0,:]|" is the NO-SPILL form (2.88e-03 for this tensor), not the as-built expectation. The E0 gate constant is settled by the leg-4 named re-verify (print max|W[0,:]|, max|row0+SUM-siblings|, max|W| alongside max|d|; as-built AND fixed variants; fixed-variant constant = the leg-4 measurement, grid-class expected) per cloudy's consolidated ruling, WAVE-16 close 2026-11-27 | PASS (small, exact) |

Legacy vehicle on the same gate: 1.65 / 1.47 (garbage). Option-B: 2.29e-1 total = the as-built d2s6 min-term design, NOT the irreducible q8_1 floor (see the min-term decomposition below; the host-side fix to the 5.36e-3 class is named there and leg-4 t3591 is verifying); the kernel contributes 2.7e-5 on top.

## F2 discrimination matrix (build #9, pre-clean) — ALL GREEN

F2a window0 rows 0-15, F2b window1 16-31, F2c 32-boundary 28-43, F2d 128-boundary
128-143, F2e shift4 192-207, F2f super2 256-271, F2g k128, F2h k5119, F2i k255:
every 16-row window / boundary / shift group / superblock / K-row carries data
(exact 3.0 within f32 noise: measured 2.99982). The leg-2 fragment hole
(rows 8-15 of every 16-window dead) is closed.

## Root causes fixed (4 real bugs, 3 fix build iterations + 2 compile-fail iterations)

1. **Warp-window base**: `b = n0 % 256` -> `b = (n0 + i0) % 256` (warp 16-row
   window base; n0 tile base + i0 warp offset).
2. **Code-byte base**: `16 + 32*(b/128) + (b % 32)` -> `16 + 32*(b/128) +
   16*((b/16) & 1)` (the 16-group offset pattern is 0,16,0,16,... not 0,16,0,0).
3. **dmin read**: `&dd + 1` on `const uint32_t dd` reads +4 bytes (bytes 84-85);
   fixed to `hp[1]` with `hp = reinterpret_cast<const half *>(&dd)` (+2 bytes,
   dmin f16 @82).
4. **8-byte unrolled load (the sneaky one)**: the `#pragma unroll` sequence of 8
   per-element byte reads `blk[cbb + 8h + l]` read ZEROS on gfx1201 while scalar
   probes of the same bytes in the same thread were correct (GPU file confirmed
   intact: x[16]=3 etc.). Replaced with two explicit aligned 4B loads
   (`b4a`/`b4b` from `blk + cbb + 8h [+4]`) with byte extraction from the 4B
   values. After this single change, F2 went from all-zeros to all-green.
   (Recorded as an RDNA4 byte-load vectorization quirk; 4B loads are the safe
   form for this code path.)

## Architecture fact (the gate mis-specification)

mmq (mul_mat_quantized) accepts x only as q8_1 (d2s6 header: two 64-value scales d0/d1, six per-16 partial sums, f16-stored). Any mmq path — option A (in-tree dp4a) or option B (this file-layout vec-dot) — carries the q8_1 x-quantization floor. On the G1 battery (x ~ N(0, 0.1), real GGUF attn_v weight K=5120 N=1024) the as-built floor is max|d| = 9.458e-02, rel = 2.290e-01 (exact numpy replication of `mmq_quant_in_q8_1_mmq`, d2s6 branch, mmq.hip:328-356: two 64-scales, t = rint(v * 127/amax) clamped, f16-stored d and d*qsum, min for i<96 only). WAVE-16 correction (wave 3 t3590 MODIFIED, cloudy's consolidated ruling ACCEPTED 2026-11-27): the 2.29e-1 is the current d2s6 min-term design, not an irreducible q8_1 floor — a host-side quantizer fix (the mma unchanged) brings it to the 5.36e-3 class, 43x; see the min-term decomposition below. The KQFG path (M=2, direct f32 dot, no x-quantization) is 2.45e-6 — the 1e-3 tolerance was evidently calibrated to kqfg-class accuracy and does not apply to mmq.

## min-term decomposition (WAVE-16, 2026-11-27)

The as-built 2.29e-1 decomposes (same G1 battery, same W; wave 3 t3590, independently re-derived by the wave-1 and wave-3 auditors):

| x-quant variant | max|d| | rel |
|---|---|---|
| grid-only (no min-term, s_g = 0) | 2.701e-03 | 6.539e-03 |
| as-built d2s6 (t on raw values; s_g = f16(d_g*SUM(t))) | 9.458e-02 | 2.290e-01 |
| d2s6 layout, proper mean-correction (host-side fix) | 2.215e-03 | 5.363e-03 |

Mechanism (the add-the-mean artifact): the quantizer computes t = rint(v*127/amax) on the RAW values and stores s_g = f16(d_g*SUM(t)); the mma reconstructs x' = d_g*t + s_g/16 = d_g*(t + mean(t)) — the group mean is added on top of the raw-value quantization instead of the quantization being done on the mean-subtracted residual. The min-term is the entire 35x amplification (grid-only 6.539e-03 -> as-built 2.290e-01); f16 scale rounding alone is negligible (2.290e-01 -> 2.288e-01). The E0 "1.0625x" factor is exactly this artifact (x = e0: x'[0] = 1.0625, the 15 sibling elements = +0.0625 each, exact).

Fix spec (C5.3, host-side quantizer only — the mma reads s_g/16 unchanged): per-16-group mean subtract (f32); the two per-64-half scales recomputed on the residual; s_g = f16(raw group sum of the values, i<96 only; last-32 plain); t = rint(residual*127/amax_residual) clamped. By construction x' = (v - m) + m = v (on e0: 0.9375 + 0.0625 = 1.0 exact). The 5.363e-03 is a CPU numpy MEASURED PREDICTION of the fixed design (label: DERIVED, not a kernel-measured number); the 41.2 t/s mma cost is untouched. Even the minimum-achievable q8_1 floor (5.36e-3) is 5.4x the literal 1e-3 gate, so the corrected-gate structure stands — with the in-band floor check replacing the frozen constant: the measured total floor within 0.9x..1.1x of the numpy-replica floor; a jump outside the band = quantizer regression FAIL; self-updating (as-built 2.29e-1 -> fixed 5.36e-3 class). GPU verification = leg-4 (t3591, running at this revision).

Note — xq2 replica deviation (wave 1, quantified, immaterial): the kernel stores s_g = f16(d_f32 * qsum) with the f32 d0/d1, while /tmp/xq2.py computes f16(f16(d) * qsum); the measured gap is 1.9e-03 max|d| / 4.6e-05 rel = 0.2% of the floor, far below the 4-digit claim tolerance and below the kernel-own 2.722e-05 figure's significance.

## Provenance

- Box: neon (RX 9070 XT / gfx1201, ROCm 10.1 guest userland; guest-a tailnet peer)
- GPU state: idle (load avg 0.01), temp 37.9C pre-run, uptime 13h, no clock lock
  (default clocks; no pstate intervention)
- Build: wheel from /home/andryo/build-p2seq via /home/andryo/p1dforkb-build.sh
  (CMAKE: MLX_BUILD_ROCM=ON, gfx1201, OpenBLAS), DELIVERABLE = build #11 (clean,
  probes off, 0 probe lines), log /home/andryo/p1dforkb-build.log (appended),
  nohup log p1dforkb11-nohup.log
- Wheel (deliverable): /home/andryo/wheelout-p1dforkb/mlx-0.32.3.dev20261127+b9cb137-cp314-cp314-linux_x86_64.whl
  md5 = 23baab1985fc94042f06d3a95834d947 (build #11; probe-on build #10 wheel md5 was
  70fbd147fbb380267f0caa90aee90d69 - identical gate numbers)
- G1 venv: /home/andryo/venv-ar (wheel installed via python zipfile extract)
- G1 scripts: /tmp/g1v2b.py (matmul W, not W.T; ktype 10, mode 0, axis 0, group 32,
  keep_ends 1, pairs 15; M=2 -> kqfg, M=1 -> mmq per mr_gate default 1),
  /tmp/xq2.py (exact d2s6 x-quant replication + floor measurement)
- F2 probe: /tmp/f2p.py (synthetic 84B file, one 2-bit code = 3 in target rows)
- Source SoT: local /srv/fleetwork/work/3587/vendor/mlx-rocm-upstream (3 modified
  files: mmq.cuh, mmq-vec-dot.cuh, qmm.hip; base commit 4e43c8b)

## NAMED NEXT STEP (option A remains the standing fallback)

**Dispatch M=1 to kqfg** (routing per the WAVE-16 consolidated ruling, 2026-11-27: the dedicated `MLX_KQFILE_MMQ_M1` env gate — default off, M=1 stays on kqfg; on, option-B mmq M=1. No mr_gate change: the default is already mr_gate=1, and the original "raise mr_gate to >= 1" wording was mis-specified; `MLX_KQUANT_FILE_MR=0` is the alternative with the use_mr side effect; the existing kqfg fallback when `mmq_q2k_file` returns false is the second safety): the kqfg path is G1-verified at 2.45e-6 (<= 1e-3 gate passes), and single-token decode is exactly the M=1 case. Keep the option-B mmq kernel as an opt-in fast path (env-gated, `MLX_KQFILE_MMQ_M1=1`). Op-stance data on the record (cloudy consolidated ruling, WAVE-16 close): (a) kqfg 2.45e-6 class @ 0.28 t/s (a serve latency dead-end) / (b) as-built mmq 2.29e-1 @ ~41 t/s class / (c) fixed mmq 5.36e-3 class @ ~41 t/s class — the as-built floor is the current d2s6 min-term design (option A shares it), the host-side fix to (c) is named above and leg-4 (t3591) is verifying; the choice is accuracy-vs-latency, not kernel-correctness. Option A (in-tree dp4a + the same 4 root-cause fixes' analogues) is the standing fallback if the file-layout path is ever reverted.
