# INT2-WMMA — vectorized int2 g128-asymmetric draft head (R4c follow-up)

**Status: IN PROGRESS (this file is updated per verified step; the per-row xcheck number lands in the TL;DR once measured).**

## TL;DR

(FILLED AFTER XCHECK — the per-row number first: vectorized int2 per-row ms vs 18.7 scalar int2 / 0.95 4-bit, and the verdict.)

## The lever under test

R4c (REPORT-mtp-r4c-int2-head.md) proved the int2 draft head lossless and wheel-clean but NO-ADOPT: the ported scalar kernel is 18.7 ms/row (19.75x the 4-bit head's 0.95 ms/row), so the 2-bit weight-traffic saving (~0.71 GiB/step at the anchor) never engaged. This lane ships the vectorized int2 kernel that should make per-row time match or beat the 4-bit path, at which point both the per-row head compute and the 2-bit traffic are in int2's favor.

## Study — how the 4-bit head hits 0.95 ms/row

The shared 4-bit `lm_head` at M=1 (N=248320, K=4096, no bias, bf16) dispatches `qmv_tiled_kernel` in the ROCm `QuantizedMatmul::eval_gpu` (qmm.hip): no bias + M=1 misses the `wide_ok` gate (`has_bias || (M>=2 && M<=4)`), the idot/LX paths are opt-in (unset), so the default lands on the tiled QMV with `tile_n = 24` (RDNA4, 32 CUs, 8 MB L2 via `get_arch_tuning`). Block = (32, 24) = 768 threads, one 32-lane wave per output column, grid (1, 10347) with grid-stride over a CU-bounded resident grid (2 blocks/CU at 2048 max threads). Per column: each lane owns 16 K-values (2x u32 = 16 weight bytes, `__builtin_nontemporal_load` streaming so the once-read weights bypass L2 and leave it for the reused x/scales); x is staged f32 in LDS (512 floats per 512-K chunk, two `__syncthreads` per chunk) and shared by the 24 column-warps of the block; per lane-iteration: 16 per-element `shift+and` decodes fused straight into 4 independent qdot accumulators + xsum (one FMA per weight — the dequant is a bit-pattern decode in the MAC loop, no pre-pass); the per-group (GS=128) `scale*qdot + bias*xsum` is applied once per group; final 5-step `__shfl_xor` warp reduction, lane 0 writes. 127 MB of 4-bit weights at 0.95 ms ≈ 134 GB/s — the kernel is in the latency/bandwidth mixed regime, not HBM-bound, which is exactly why halving the weight bytes (2-bit: 64 MB) should win.

## Design — the vectorized int2 kernel (`lm_head_int2_wide`, new op, byte-unchanged default path)

Same structure as the tiled 4-bit kernel, adapted to the quarter-split 2-bit packing (packed byte p holds plane-q value for element `q*(K/4) + p`; group G = 8q + gl owns 128 consecutive elements of quarter q = 128 consecutive packed bytes):

- Block (32, 64) = 2048 threads (`__launch_bounds__(2048)`), one wave per column, 64 columns per block = exactly one scalar-port 64-column sub-block; grid-stride over n_tiles = N/64, resident grid CU-bounded (1 block/CU at 2048 max threads/CU).
- x staged f32 in dynamic LDS (K floats = 16 KB at K=4096) once per tile, two `__syncthreads`.
- Per lane: 128 K-values per column = 32 j per quarter = 4 iterations x 8 j (2 u32 = 8 weight bytes per iteration, 8-byte aligned, streaming load). 128-group G = 8q + (lane>>2) is exactly the 4 lanes with lane>>2 == gl (4 lanes x 32 j = 128 j).
- Per iteration: 8 packed bytes; byte b contributes plane q value `(b >> (2q)) & 3` to `d[q]`, FMA against `x[q*(K/4) + b]` — the dequant is the per-element decode fused into the MAC loop (one FMA + one decode per weight, same instruction count per weight as the 4-bit kernel), no scalar pre-pass.
- Per quarter: 4-lane `__shfl_xor` reduction of `d[q]` (offsets 2, 1; the 4-lane group is 4-aligned so the offsets stay in-group), then the fused group algebra `acc += d[q] * scale[G] - sx[G] * zs[G]` (scale = s, zs = z*s, per the R4c semantics), one per quarter (8 group ops per lane per column — the same count as the 4-bit kernel's 8).
- Final 5-step `__shfl_xor` warp reduction (identical to the tiled kernel), lane 0 writes the column logit.
- Candidate emission: the EXACT scalar-port 8-round per-64-column top-8 selection transcribed onto the 64 shared entries (shared `sh[64]`/`shi[64]`, `wmax[2]` merge, offsets 16..1, strict `>` so lower index wins ties on ties, 8 rounds, `sres[8]`), executed by the block's first 64 threads; identical candidate (value, col) contract so the bench re-rank (exact 4-bit re-dequant of the top-32 candidates) is untouched.

Per-lane instruction budget vs the 4-bit tiled kernel: 128 FMA = 128, 128 decodes = 128, 128 LDS loads = 128, 8 group ops = 8, +16 4-lane shfl (the one extra), weight bytes 32 vs 64 (half). Expected: per-row at or below the 4-bit 0.95 ms, plausibly ~0.5 ms if bandwidth-bound.

Correctness contract: the coarse per-column logit math is the same g128-asymmetric dequant algebra as the scalar port (per-group `dot_g*s - sx_g*(z*s)`), differing only in float summation order (lane partials + 4-lane/warp reductions vs the scalar's sequential 128-group loop) — ulp-level at most; the lossless gate (FIRST20/ALLTOKS bit-identical to the scalar int2 port AND the 4-bit head at the bench protocol) re-verifies end-to-end because the final token is the argmax of the exact re-rank and both candidate sets cover the top-32 (per-64 top-8 in the scalar, the identical structure here).

Integration: new op `lm_head_int2_wide` add-on — `LMHeadInt2W` primitive, `lm_head_int2_wide()` fast op, python binding `mx.fast.lm_head_int2_wide`, host guards cloned from the scalar (M==1, wq [N, K/4], scale/zs [N, K/128], sx [K/128], plus N%64==0 and K%512==0). `MTP6_DRAFT_HEAD=int2w` selects it in the bench harnesses; the scalar op and the 4-bit default are byte-unchanged when the env is unset.

## Build + xcheck

(FILLED AFTER BUILD: venv lineage, md5s, per-row xcheck table vs 18.7 / 0.95, occupancy/counter evidence if it fails the gate.)

## Protocol re-measure + anchor A/B

(FILLED AFTER MEASURE, or the negative verdict with counter evidence if the kernel is >= the 4-bit per-row.)
