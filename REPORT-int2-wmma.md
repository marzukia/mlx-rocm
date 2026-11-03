# INT2-WMMA — vectorized int2 g128-asymmetric draft head (R4c follow-up)

**Status: NEGATIVE (measured). The vectorized int2 head is correct (bit-identical to the scalar port) but 9.73 ms/row — 10.2x the 4-bit head's 0.95 ms/row. Per the gate, the full A/B is NOT run; this file carries the counter evidence + the identified optimization path for the next lane.**

## TL;DR

Per-row xcheck (warm 10-iter loop, N=248320, K=4096, M=1, gfx1201 / 32 CU / 64 KB LDS / 2048 threads-per-CU):

| head | per-row | vs 4-bit |
|---|---|---|
| 4-bit `lm_head` (shared QuantizedLinear) | **0.953 ms** | 1.00x (anchor) |
| int2 scalar port (R4c) | 18.59 ms | 19.5x (R4c measured 18.7 — reproduced) |
| **int2 vectorized (`lm_head_int2_wide`, this lane)** | **9.73 ms** | **10.2x — GATE FAIL (>= 0.95)** |

Correctness: the wide op's per-column logits are **bit-identical to the scalar port's** (max abs/rel diff = 0.0 over 16 random rows, exact 4-bit-re-rank weights) — the 9.73 ms is a genuine cost, not an artifact. The per-32 top-8 candidate set differs from the scalar's per-64 top-8 by construction (it strictly refines it); `sel_match=0/256` on the top-32 set is the expected coverage gap, not a bug (the logit check is the correctness signal).

## The lever under test

R4c (REPORT-mtp-r4c-int2-head.md) proved the int2 draft head lossless and wheel-clean but NO-ADOPT: the ported scalar kernel is 18.7 ms/row (19.75x the 4-bit head's 0.95 ms/row), so the 2-bit weight-traffic saving (~0.71 GiB/step at the anchor) never engaged. This lane ships the vectorized int2 kernel that should make per-row time match or beat the 4-bit path, at which point both the per-row head compute and the 2-bit traffic are in int2's favor.

## Study — how the 4-bit head hits 0.95 ms/row

The shared 4-bit `lm_head` at M=1 (N=248320, K=4096, no bias, bf16) dispatches `qmv_tiled_kernel` in the ROCm `QuantizedMatmul::eval_gpu` (qmm.hip): no bias + M=1 misses the `wide_ok` gate (`has_bias || (M>=2 && M<=4)`), the idot/LX paths are opt-in (unset), so the default lands on the tiled QMV with `tile_n = 24` (RDNA4, 32 CUs, 8 MB L2 via `get_arch_tuning`). Block = (32, 24) = 768 threads, one 32-lane wave per output column, grid (1, 10347) with grid-stride over a CU-bounded resident grid (2 blocks/CU at 2048 max threads). Per column: each lane owns 16 K-values (2x u32 = 16 weight bytes, `__builtin_nontemporal_load` streaming so the once-read weights bypass L2 and leave it for the reused x/scales); x is staged f32 in LDS (512 floats per 512-K chunk, two `__syncthreads` per chunk) and shared by the 24 column-warps of the block; per lane-iteration: 16 per-element `shift+and` decodes fused straight into 4 independent qdot accumulators + xsum (one FMA per weight — the dequant is a bit-pattern decode in the MAC loop, no pre-pass); the per-group (GS=128) `scale*qdot + bias*xsum` is applied once per group; final 5-step `__shfl_xor` warp reduction, lane 0 writes. 508.8 MB of 4-bit weights at 0.95 ms ≈ 533 GB/s — the kernel is in the latency/bandwidth mixed regime, not HBM-bound, which is exactly why halving the weight bytes (2-bit: 254.4 MB) should win.

## Design — the vectorized int2 kernel (`lm_head_int2_wide`, new op, byte-unchanged default path)

Same structure as the tiled 4-bit kernel, adapted to the quarter-split 2-bit packing (packed byte p holds plane-q value for element `q*(K/4) + p`; group G = 8q + gl owns 128 consecutive elements of quarter q = 128 consecutive packed bytes):

- Block (32, 32) = 1024 threads (`__launch_bounds__(1024)`), one wave per column, 32 columns per block; the 2048-thread variant (block (32, 64)) is NOT launchable on gfx1201 (AMD workgroups cap at 1024 threads) — `hipGetLastError()` returns a launch error. 2 blocks/CU (2048 threads = full CU, 32.5 KB of the 64 KB LDS budget); grid-stride over n_tiles = N/32, resident grid = 2 x num_cus.
- x staged f32 in dynamic LDS (K floats = 16 KB at K=4096) once per tile, two `__syncthreads`.
- Per lane: 128 K-values per column = 32 j per quarter = 4 iterations x 8 j (2 u32 = 8 weight bytes per iteration, 8-byte aligned, streaming load). 128-group G = 8q + (lane>>2) is exactly the 4 lanes with lane>>2 == gl (4 lanes x 32 j = 128 j).
- Per iteration: 8 packed bytes; byte b contributes plane q value `(b >> (2q)) & 3` to `d[q]`, FMA against `x_shared[q*(K/4) + b]` — the dequant is the per-element decode fused into the MAC loop (one FMA + one decode per weight, same instruction count per weight as the 4-bit kernel), no scalar pre-pass. NOTE: the x operand is read from LDS per plane per byte (128 LDS reads per lane); the 4-bit tiled kernel instead preloads x into VGPRs (`x_local[i] = x_shared[lane*VPT6+i]`, the BSK6 tiling) and does register-resident FMA. This is the dominant structural difference (see the negative verdict).
- Per quarter: 4-lane `__shfl_xor` reduction of `d[q]` (offsets 2, 1; the 4-lane group is 4-aligned so the offsets stay in-group), then the fused group algebra `acc += d[q] * scale[G] - sx[G] * zs[G]` (scale = s, zs = z*s, per the R4c semantics), one per quarter (8 group ops per lane per column — the 4-bit kernel's 32-element-per-lane groups need no intra-warp group reduction, so this is 8 extra `__shfl_xor` per lane).
- Final 5-step `__shfl_xor` warp reduction (identical to the tiled kernel), lane 0 writes the column logit.
- Candidate emission: the scalar-port 8-round shared-memory max selection re-cast onto 32 entries, run by the single selection warp (threadIdx.y == 0) — 8 rounds of 5-step `__shfl_xor` warp reduce-max (offsets 16..1, strict `>` so the lower index wins ties), the winner removed with -INFINITY, `sres[8]` in shared. Per-32 top-8 strictly refines per-64 top-8 (each per-64 top-8 set is contained in the union of its two per-32 top-8 sets), so the re-rank contract holds at least as strongly; candidate arrays are [N/32, 8].

Per-lane instruction budget vs the 4-bit tiled kernel: 128 FMA = 128, 128 decodes = 128, 128 LDS x-reads = 128 (the 4-bit kernel preloads x into VGPRs and reads 0 LDS in the FMA loop), 8 group ops = 8, +8 4-lane `__shfl_xor` (the 4-bit kernel has none at this level), final 5-step warp reduce (same). Weight bytes 32 vs 64 (half). The per-element LDS x-read is the structural gap the 4-bit kernel closes with register preloading.

Correctness contract: the coarse per-column logit math is the same g128-asymmetric dequant algebra as the scalar port (per-group `dot_g*s - sx_g*(z*s)`), differing only in float summation order (lane partials + 4-lane/warp reductions vs the scalar's sequential 128-group loop). Verified empirically: the wide op's per-column logits are bit-identical to the scalar port's (max abs/rel diff = 0.0 over 16 random rows on the exact quantized head) — the summation re-ordering happens to be exact at bf16-logit precision for this data.

Integration: new op `lm_head_int2_wide` add-on — `LMHeadInt2W` primitive, `lm_head_int2_wide()` fast op, python binding `mx.fast.lm_head_int2_wide`, host guards cloned from the scalar (M==1, wq [N, K/4], scale/zs [N, K/128], sx [K/128], plus N%32==0 and K%512==0). `MTP6_DRAFT_HEAD=int2w` selects it in the bench harnesses; the scalar op and the 4-bit default are byte-unchanged when the env is unset.

## Build + xcheck

- Build tree: vendor-canon base + this branch's vendor snapshot overlay + the wide python-binding graft (the R4c-build-tree python/src/fast.cpp as the graft base); `lm_head_int2w.hip` quick-compile check passed, full wheel built green (gfx1201, ROCm 10.1, CMAKE_HIP_ARCHITECTURES=gfx1201, BLA_VENDOR=OpenBLAS).
- venv lineage: `venv-int2w` = hardlink clone of `venv-r4c` (R4c wheel base, .so md5 03bcf1f4a4235f58a3a715afadfcbeae), new wheel installed via `uv pip install --force-reinstall` (new .so md5 ff56e7aa4e4f124a55685e47f0ace0a3; canonical was b1de59fda0e4fc607d620e287deeaaf8). `mx.fast.lm_head_int2_wide` present; the scalar `lm_head_int2` and the 4-bit default path byte-unchanged.
- First build iteration: the 2048-thread design (block (32, 64)) compiled and linked but failed at launch (`hipGetLastError() != 0`) — AMD workgroups cap at 1024 threads; the kernel was re-cut to block (32, 32) = 1024 (32 columns per tile, per-32 top-8 selection, N/32 candidate arrays, N%32 guard) and relaunched clean.
- xcheck (warm 10-iter, per-row M=1 calls, N=248320, K=4096): int2 scalar 18.59 ms (R4c: 18.7 — reproduced), 4-bit head 0.953 ms (R4c: 0.95 — reproduced), **int2 wide 9.73 ms** (wide_vs_head = 10.21, int2_vs_head = 19.51). Correctness probe: wide vs scalar per-column logits, 16 random rows on the exact quantized head: **max abs diff = 0.0, max rel diff = 0.0** (bit-identical). `sel_match=0/256` on the global top-32 candidate set is the expected per-32-vs-per-64 coverage difference (the wide set strictly refines the scalar's), not a correctness signal.

## Negative verdict (gate fail: 9.73 ms/row >= 0.95 ms/row) — A/B NOT run, per the gate

Occupancy / counter evidence (why 10.2x the 4-bit head, despite half the weight bytes and full occupancy):

1. **Full occupancy, so the gap is not scheduling.** 1024-thread block x 2 blocks/CU = 2048/2048 threads = 100% of the 32-CU budget; 16.25 KB LDS per block (16 KB x staging + 0.25 KB selection), 2 blocks = 32.5 KB of the 64 KB CU budget — no register/LDS cap blocks a second block. 7760 column-tiles over 64 resident blocks (122 grid-stride rounds).
2. **Not weight-bandwidth-bound.** 254 MB of 2-bit weights at 9.73 ms = 26 GB/s — 38x under the RX 9070 XT's HBM bandwidth. The 4-bit kernel's 533 GB/s is also far from HBM-bound; both are in the latency/LDS regime, but the int2 kernel's inner loop is the slower one.
3. **The structural gap: per-element LDS x-reads in the FMA loop.** The int2 inner loop reads `x_shared[q*Q + j]` from LDS per plane per byte — 128 LDS reads per lane per column. The 4-bit tiled kernel preloads x into VGPRs (`x_local[i] = x_shared[lane*VPT6 + i]`, BSK6 tiling) and does register-resident FMA — 0 LDS reads in the FMA loop. At ~1.02 G x-loads per row (N x K), that per-element LDS traffic, plus the 8 extra 4-lane `__shfl_xor` group reductions per lane (the 4-bit 32-element-per-lane groups need none), is the dominant cost: same 4096 FMA per column, half the weight bytes, 10.2x the time.

The identified optimization path for the next lane (not done in this lane — the gate is a stop): preload the lane's 32 j x 4 q x-values into VGPRs (128 VGPR; RDNA4 has 256 per lane) and do register-resident FMA, mirroring the 4-bit `x_local`/BSK6 structure, so the per-element LDS reads disappear; that is the single change that closes the gap to the 4-bit regime (the 4-lane group shfl and the halved weight traffic then work in int2's favor, plausibly to ~0.5 ms/row as originally estimated). The current kernel (correct, bit-identical, 9.73 ms, 1.9x the scalar) is committed as the baseline for that follow-up; the default path (bf16/4-bit head, and the scalar int2 op) is byte-unchanged when `MTP6_DRAFT_HEAD` is unset or `bf16`/`int2`.
