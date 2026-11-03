# NOTES-int2vgpr — register-resident x fix for the int2 wide draft head

Branch: int2-wmma-vgpr from int2-wmma @ e6a06da (measured negative baseline: 9.73 ms/row, bit-identical).

## Task
1. Preload lane's 128 x-values (32 j x 4 q) into VGPRs; register-resident FMA, 0 LDS x-reads in the loop.
2. Everything else as-is (candidate selection, group algebra, final warp reduction, guards, binding, env gate) unless the restructure forces a layout change (note it).
3. Correctness: per-column logits bit-identical to scalar port (16-row probe, max abs/rel diff 0.0; A/B lossless gate vs 48.93 either way).
4. Gate: per-row < 0.953 ms (4-bit anchor re-measured same session, N=248320 K=4096 M=1, warm 10-iter). PASS -> full A/B; FAIL -> counter-evidence, stop.

## Facts (from REPORT-int2-wmma.md + source)
- Kernel: vendor/mlx-rocm-upstream/mlx/backend/rocm/lm_head_int2w.hip (232 lines at base).
- Block (32,32)=1024, 2 blocks/CU (2048 threads, 32.5 KB LDS of 64). Grid-stride over n_tiles=N/32=7760; 122 rounds.
- Per lane per column: 4 iters x 8 bytes (one uint2 streaming load each), 128 FMA + 128 decode; inner loop reads x_shared[q*Q + j] per plane per byte = 128 LDS x-reads per lane per column. That is the structural gap vs the 4-bit qmv_tiled_kernel (x_local/BSK6 register staging).
- 4-bit reference structure (qmv_tiled_kernel.hip): per 512-elem chunk: `float x_local[VPT6]; for i: x_local[i] = x_shared[lane*VPT6+i];` then register FMA, 0 LDS in loop.
- int2 lane x layout: quarter q, j in [0,32): LDS index q*Q + 128*gl + 32*l3 + j. Consecutive j = 32 contiguous LDS per quarter per lane; 4 q's = 128. x is the SAME across tiles (M=1 single row) -> hoistable to block prologue.

## Design (planned)
- Hoist the x_shared staging + a one-time `float xv[4][32]` preload into the prologue (before the tile loop). x is fixed per call, so the per-tile re-staging is pure waste (122 rounds x 4 LDS stores/lane + 122 x 128 LDS reads/lane at base).
- Tile loop keeps: 4 streaming uint2 weight loads (hoist them before the x FMA so global latency overlaps the decode), 128 FMA with xv[q][b], 4x 4-lane shfl group reduce, fused group algebra, 5-step final warp reduce, lane0 logit write, per-32 top-8 selection (sh[ty] staging + 2 __syncthreads per tile, unchanged).
- Barrier check: with x_shared written once in the prologue (one __syncthreads after staging), the in-loop barriers only guard sh[32] reuse (write sh[ty]=acc -> syncthreads -> selection reads sh / writes sh -INFINITY -> syncthreads). Iter N+1's sh write waits for iter N's last syncthreads. No barrier lost.
- FMA order unchanged (same d[q] accumulation sequence, same shfl order) -> bit-identity with the scalar port is preserved BY CONSTRUCTION (xv values = the same f32 bits from the same staging). Verify with the 16-row probe anyway.
- __launch_bounds__(1024) unchanged (max threads only; no min-blocks hint). LDS unchanged (16 KB). Registers: 128 (xv) + 8 (wv) + ~12 (acc/d/selection) ~ 150 VGPR/lane - the occupancy check is the open question (need per-lane VGPR limit on gfx1201; base report claims RDNA4 has 256/lane; verify empirically with an occupancy query).

## Open questions
- [x] per-lane VGPR limit on gfx1201: empirically, 51-VGPR base kernel -> 2 blocks/CU; the 128-x-value preload variant uses 192 VGPR (max v191 in disasm) -> hipOccupancyMaxActiveBlocksPerMultiprocessor = 1 block/CU. gfx1201: 32 CU, 2048 threads/CU, 64 KB LDS/CU. So the VGPR budget for 2 blocks/CU is between 104448 (base) and 208896 (192/lane x 2048); 2 blocks fit iff R <= ~64-96/lane.
- [x] Decision: SHIP Option B (128-x hoisted preload, 192 VGPR, 1 block/CU, 33 scratch loads = spills, 1.41 ms/row standalone). Option A (per-quarter streaming): occupancy API = 0 blocks/CU, 176 VGPR, 10.54 ms/row standalone (4.4x slower than B) - NOT shippable; the per-quarter structure did not cut register pressure as hoped (the compiler keeps the 32 x + 8 weight + ~40 other live). The 2-blocks/CU loss is measured and reported (occupancy trade, not silently accepted): 32 active waves (B) vs 64 (base) vs 48 (4-bit anchor).
- [ ] wheel build + xcheck + 16-row probe (the real gate; standalone absolute values differ from the wheel loop by ~1.67x on the base - do NOT extrapolate, measure on the wheel).
- [ ] A/B (only if gate PASS).

## Option A design (per-quarter streaming, the 4-bit x_local mirror)
- Prologue: x_shared staging once per block (x fixed per call) + __syncthreads.
- Per column per quarter q (4 iters over quarters): preload float xv[32] = x_shared[q*Q + 128*gl + 32*l3 + [0..32)] (32 contiguous, 8x b128 vectorizable); load the 4 uint2 (32 weight bytes); 32 FMA into d_q (sequential j order = same per-lane summation order as base -> bit-identity preserved by construction); 4-lane shfl group reduce (offsets 2,1); fused group algebra acc += d_q*sr[G] - sx[G]*zr[G].
- d[q] replaced by a scalar d_q per quarter (the 4 d values are never simultaneously needed: base accumulates all 4 then reduces each; A reduces each immediately - the per-quarter d_q values are independent, so the final acc sum order is identical: acc += q0 term, q1 term, q2 term, q3 term, same sequence).
- Register estimate: 32 x + 8 w + ~20 = ~60-70 VGPR -> 2 blocks/CU (64 waves).
- Option B (current edit): xv[4][32] hoisted before the tile loop, 0 LDS x-reads per column; 192 VGPR -> 1 block/CU (32 waves).
- Standalone bench plan: hip harness, real shapes (N=248320, K=4096, M=1), synthetic random wq/scale/zs/sx/x (timing-representative; correctness verified separately by the 16-row xcheck probe on the exact head), warm 10-iter, per-row ms via hip events.

## Standalone occupancy probe (done)
- occ probe (hipOccupancyMaxActiveBlocksPerMultiprocessor, block (32,32), 16 KB dynamic LDS): base kernel = 2 blocks/CU; VGPR (Option B) kernel = 1 block/CU. Disasm max v-reg: base 50, Option B 191. Option B inner loop: 0 ds_load (106 v_fma present; the 16+5 ds_load = prologue staging + xv preload, once per block).
