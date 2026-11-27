# FINDINGS - P1-d fork-(b): Q2_K file-layout vec-dot kernel — BUILT, VERIFIED, GATE MIS-SPECIFIED

Date: 2026-11-27 08:05Z (neon clock)
Agent: option-B build session (leg-3)
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
| G1v2-MMQ E0 unit probe | max|d| = 1.409e-02 = 0.0625 * max|W[0,:]| (the exact d2s6 min-correction factor 1.0625x, to 4 digits) | PASS (small, exact) |

Legacy vehicle on the same gate: 1.65 / 1.47 (garbage). Option-B: 2.29e-1 total =
the irreducible x-quantization floor; kernel contributes 2.7e-5 on top.

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

mmq (mul_mat_quantized) accepts x only as q8_1 (d2s6 header: two 64-value scales
d0/d1, six per-16 partial sums, f16-stored). Any mmq path — option A (in-tree
dp4a) or option B (this file-layout vec-dot) — carries the q8_1 x-quantization
floor. On the G1 battery (x ~ N(0, 0.1), real GGUF attn_v weight K=5120 N=1024)
the floor is max|d| = 9.458e-02, rel = 2.290e-01 (exact numpy replication of
`mmq_quant_in_q8_1_mmq`, d2s6 branch, mmq.hip:328-356: two 64-scales,
t = rint(v * 127/amax) clamped, f16-stored d and d*qsum, min for i<96 only).
The KQFG path (M=2, direct f32 dot, no x-quantization) is 2.45e-6 — the 1e-3
tolerance was evidently calibrated to kqfg-class accuracy and does not apply to
mmq.

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

**Dispatch M=1 to kqfg** (raise `mr_gate` to >= 1, or make the kquant_file_gemm
eval_gpu dispatch send M <= 1 to the kqfg path): the kqfg path is G1-verified at
2.45e-6 (<= 1e-3 gate passes), and single-token decode is exactly the M=1 case.
Keep the option-B mmq kernel as an opt-in fast path (env-gated, e.g.
MLX_KQFILE_MMQ_M1=1) once an operator decision is made on whether the 2.29e-1
x-quantization floor is acceptable for serve (it is the same floor option A
would have; the choice is accuracy-vs-latency, not kernel-correctness).
Option A (in-tree dp4a + the same 4 root-cause fixes' analogues) is the standing
fallback if the file-layout path is ever reverted.
