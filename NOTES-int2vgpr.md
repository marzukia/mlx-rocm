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
- [x] wheel build + xcheck + 16-row probe: wheel a5b79f84 (sha256 e96f6c74, 22.5 MB) in venv-int2w (pip --target --upgrade; the venv has no pip/uv on the guest - system pip 26.1.2 used). 16-row probe: max abs/rel = 0.0. xcheck (2 runs): wide 1.354 / 1.351 ms, head 0.954 / 0.950 ms, int2 scalar 19.06 / 18.90 ms, sel_match=0/256. GATE FAIL (1.35 >= 0.953) -> A/B NOT run, per the gate.
- [x] Counter-evidence: 1 block/CU (32 waves) vs base 2 blocks (64) vs 4-bit 48 waves; 192 VGPR + 33 scratch loads vs base 51 VGPR; LDS unchanged (not the constraint); 0 LDS x-reads in the FMA loop; per-column int2 174.5 ns vs 4-bit 184.4 ns (5.4% faster per wave - the inner loop won, the occupancy lost the row).
- [x] A/B: NOT run (gate FAIL).

## Option A design (per-quarter streaming, the 4-bit x_local mirror)
- Prologue: x_shared staging once per block (x fixed per call) + __syncthreads.
- Per column per quarter q (4 iters over quarters): preload float xv[32] = x_shared[q*Q + 128*gl + 32*l3 + [0..32)] (32 contiguous, 8x b128 vectorizable); load the 4 uint2 (32 weight bytes); 32 FMA into d_q (sequential j order = same per-lane summation order as base -> bit-identity preserved by construction); 4-lane shfl group reduce (offsets 2,1); fused group algebra acc += d_q*sr[G] - sx[G]*zr[G].
- d[q] replaced by a scalar d_q per quarter (the 4 d values are never simultaneously needed: base accumulates all 4 then reduces each; A reduces each immediately - the per-quarter d_q values are independent, so the final acc sum order is identical: acc += q0 term, q1 term, q2 term, q3 term, same sequence).
- Register estimate: 32 x + 8 w + ~20 = ~60-70 VGPR -> 2 blocks/CU (64 waves).
- Option B (current edit): xv[4][32] hoisted before the tile loop, 0 LDS x-reads per column; 192 VGPR -> 1 block/CU (32 waves).
- Standalone bench plan: hip harness, real shapes (N=248320, K=4096, M=1), synthetic random wq/scale/zs/sx/x (timing-representative; correctness verified separately by the 16-row xcheck probe on the exact head), warm 10-iter, per-row ms via hip events.

## Build (in flight)
- Graft: modified lm_head_int2w.hip (md5 6e58970a49f457911c481d88bfee726b) copied into ~/vendor-int2wbuild/mlx/backend/rocm/ (the t1697 build tree; cmake cache: Release, ROCm ON, gfx1201, OpenBLAS - verified).
- Incremental: `cmake --build . -j$(nproc)` in build/temp.linux-x86_64-cpython-314/mlx.core, log ~/build-int2vgpr.log, started PID 504223 (env HCC_AMDGPU_TARGET=gfx1201 + libhsa LD_PRELOAD).
- Next: pip wheel . -> ~/wheelout-int2vgpr/, md5-check, uv pip install --force-reinstall into venv-int2w (lineage of venv-r4c; canonical venv untouched), then xcheck (r4c_xcheck.py) + 16-row probe (int2w_probe.py, both in ~/bench-int2w), 4-bit anchor re-measured in the same session (r4c_xcheck prints head=...us).
- Standalone bench (synthetic data, hip events, warm 3 + timed 10): base 16.19 ms/row (2 blocks/CU), Option A 10.54 (0 blocks - not shippable), Option B 1.41 (1 block/CU). NOTE: standalone base is ~1.67x the wheel xcheck's 9.73 - do not extrapolate; the wheel xcheck is the gate.
- occ probe (hipOccupancyMaxActiveBlocksPerMultiprocessor, block (32,32), 16 KB dynamic LDS): base kernel = 2 blocks/CU; VGPR (Option B) kernel = 1 block/CU. Disasm max v-reg: base 50, Option B 191. Option B inner loop: 0 ds_load (106 v_fma present; the 16+5 ds_load = prologue staging + xv preload, once per block).
