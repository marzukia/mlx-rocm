# INT2-WMMA-VGPR — register-resident x for the vectorized int2 draft head

**Per-row xcheck (warm 10-iter loop, N=248320, K=4096, M=1, gfx1201, wheel a5b79f84 in venv-int2w):**

| head | per-row (2 runs) | vs 4-bit |
|---|---|---|
| 4-bit `lm_head` (anchor, re-measured this session) | **0.954 / 0.950 ms** | 1.00x |
| int2 scalar port (R4c) | 19.06 / 18.90 ms | ~20x |
| int2 wide, per-element LDS x (baseline lane, 9.73 ms) | — | 10.2x |
| **int2 wide, register-resident x (this lane)** | **1.354 / 1.351 ms** | **1.42x — GATE FAIL (>= 0.953)** |

**Status: NEGATIVE (measured), 7.2x faster than the baseline but 1.42x the 4-bit anchor. The FMA-loop fix worked (per-column int2 is now 5.4% FASTER than the 4-bit per-column); the 2-blocks/CU occupancy loss is what costs the row. Per the gate, the full A/B is NOT run; this file carries the counter evidence (occupancy/registers/LDS breakdown) + the identified next lane.**

## What was done

Per the negative verdict of the baseline lane (REPORT-int2-wmma.md): the FMA loop read x from LDS per element (128 LDS x-reads per lane per column), while the 4-bit `qmv_tiled_kernel` preloads x into VGPRs. This lane preloads the lane's 128 x-values (32 j per quarter x 4 quarters, 32 contiguous per quarter) into VGPRs — `float xv[4][32]` — mirroring the 4-bit `x_local` staging, and hoists the x staging itself out of the grid-stride loop (the x row is fixed per call, so the baseline's per-tile re-staging — 122 rounds x 4 LDS stores + 122 x 128 LDS reads per lane — is pure waste and is gone). The FMA loop now does **0 LDS x-reads**. Per-lane summation order is unchanged (same `d[q]` accumulation sequence, same 4-lane group reduction, same final 5-step warp reduction), so bit-identity with the scalar port is preserved by construction and re-verified by the probe (below). Candidate selection, group algebra (`scale = s`, `zs = z*s`), guards, binding, and the `MTP6_DRAFT_HEAD=int2w` env gate are byte-unchanged.

An alternative (per-quarter streaming `x_local`, the closest 4-bit BSK6 mirror) was also built and measured: 10.54 ms/row standalone with the occupancy API reporting 0 blocks/CU — the per-quarter structure did not cut register pressure (176 VGPR), so the 128-value preload was shipped as the fix.

## Correctness (PASS)

- 16-random-row probe on the exact quantized head: wide-vs-scalar per-column logits **max abs diff = 0.0, max rel diff = 0.0** (bit-identical, as the baseline lane).
- `sel_match = 0/256` on the top-32 candidate set: the expected per-32-vs-per-64 coverage difference (the wide set strictly refines the scalar's), not a bug — same as the baseline lane.
- The 4-bit default path is untouched (the only code change is `lm_head_int2w.hip`); the anchor re-measured in the same session reproduces 0.953 (0.954 / 0.950).

## Build

- Wheel a5b79f84 (sha256 e96f6c74, 22.5 MB) built in the t1697 build tree (incremental: one `.hip` recompile + relink) and installed into the venv-int2w lineage (canonical venv untouched). Baseline wheel ff56e7aa -> a5b79f84. `mx.fast.lm_head_int2_wide` present; the scalar op and the 4-bit default are byte-unchanged.

## Counter evidence (why 1.35 ms, not ~0.5)

1. **The inner loop won.** Per-column time: int2 = 1.354 ms / (248320/32 waves) = **174.5 ns/column** vs the 4-bit anchor = 0.954 ms / (248320/48 waves) = **184.4 ns/column** — the register-resident int2 column is 5.4% faster per wave (halved weight bytes; the 8 extra 4-lane `__shfl_xor` group reductions largely cancel). 254 MB of 2-bit weights at 1.354 ms = 188 GB/s, vs the 4-bit kernel's 533 GB/s. The per-element LDS x-read is gone (0 `ds_load` in the FMA loop; 106 `v_fma` present; the 21 block-prologue LDS loads are the one-time staging + preload).
2. **The occupancy lost the row.** The 128-value preload costs **192 VGPR/lane** (max v191 in the disassembly, plus 33 scratch loads — the kernel spills), and the occupancy API (block (32,32), 16 KB dynamic LDS) reports **1 block/CU** where the baseline's 51-VGPR kernel got 2 (64 active waves; the 4-bit anchor runs 2 x 24 = 48). LDS was never the constraint (32.4 KB of the 64 KB budget, unchanged). 32 active waves vs 48 = 0.67x waves; the row cost = (48/32) x (174.5/184.4) = 1.42x — exactly the measured gap. The baseline's 100% occupancy at 9.73 ms and this run's 50% occupancy at 1.35 ms bracket the same kernel: the per-column speedup (9.73 -> 1.35 = 7.2x at equal occupancy would be ~0.68 ms) is real; the row is lost to scheduling, not compute or bandwidth.
3. **Anomaly noted (not fully resolved):** the per-quarter variant (176 VGPR, 0 spills) got 0 blocks from the API while the 192-VGPR variant with spills got 1 — the API's scratch/LDS interaction makes the exact per-lane budget ambiguous; the measured facts (51 VGPR -> 2 blocks; 192 VGPR -> 1 block) bound the 2-block requirement to roughly <= 96 VGPR/lane.

## A/B

NOT run — the per-row gate failed (1.354 ms/row >= 0.953 ms/row), per the gate. FIRST20/ALLTOKS losslessness vs the 48.93 stack is therefore not claimed for this head; the bit-identical per-column logit check above is the correctness signal.

## The next lane (identified, not done)

Restore 2 blocks/CU (64 waves) by getting the kernel to ~<= 96 VGPR/lane. At the measured 174.5 ns/column, 64 waves would give ~0.68 ms/row — beats the 0.953 anchor and engages the 2-bit draft path (the ~0.71 GiB/step weight-traffic saving class from R4c, REPORT-mtp-r4c-int2-head.md). Candidate shapes: per-quarter streaming with a genuinely smaller live window (the attempted per-quarter cut kept 32 x + 8 weights + ~40 other live = 176 VGPR; the 4-bit kernel achieves 51 because its per-chunk live window is 16 x + 12 weights + ~20 other), i.e. 16-j x 2-sub-iteration chunks per quarter, or a 2-bit dequant restructure that fuses the 4 plane decodes into fewer temporaries. The `int2w` op, binding, guards, env gate, xcheck/probe/runner instrumentation, and this 1.35 ms baseline all ship for it.
