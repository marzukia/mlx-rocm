# NOTES: int2 draft head v3 — restore 2 blocks/CU (<= 96 VGPR/lane)

t1755 re-dispatch, instance 2. Base head 6af56b9 (verified on mirror + org, ghnew).

## Task (one kernel task)

The t1735 v2 head (`lm_head_int2_wide`, env gate MTP6_DRAFT_HEAD=int2w) preloaded the lane's 128 x-values into VGPRs -> 192 VGPR/lane + 33 scratch spills -> occupancy API reports 1 block/CU (32 active waves) where the 4-bit kernel (51 VGPR) gets 2 blocks/CU (48 waves). Per-row 1.354 ms = 1.42x the 4-bit anchor 0.953 ms/row, exactly 0.67x x 1.054. Per-column int2 is 5.4% faster (174.5 vs 184.4 ns).

Goal: restructure the inner loop to <= 96 VGPR/lane (occupancy API must report 2 blocks/CU for block (32,32), 16 KB dynamic LDS on gfx1201 / 32 CU), keeping register-resident FMA (no per-element LDS x-reads in the loop) and the exact per-column logit arithmetic (bit-identity with the scalar port).

Gate: per-row xcheck, warm 10-iter loop, N=248320, K=4096, M=1, anchor re-measured in the SAME session. <= 0.953 ms/row -> GATE PASS -> full A/B (512-token greedy anchor protocol, FIRST20 lossless vs the 48.93 stack). > 0.953 -> GATE FAIL, no A/B, report counter evidence (VGPR from disassembly, occupancy API, spills) + next lane.

## 5-step plan

1. Read the 4-bit qmv_tiled_kernel (<= 100-line chunks, one section per turn, notes per section). Note its live window: 16 x + 12 weights + ~20 other = 51 VGPR.
2. Read the int2w wide kernel (same chunking). Note the 128-VGPR preload shape and where the FMA loop reads x from registers.
3. Design the <= 96 VGPR inner loop. Candidate (a) first: per-quarter streaming with 16-j x 2-sub-iteration chunks (16-wide j window per sub-iteration, matching the 4-bit window). Fallback (b): 2-bit dequant restructure fusing the 4 plane decodes into fewer temporaries. Invariant: per-lane summation order UNCHANGED (bit-identity).
4. Implement, rebuild wheel in the t1697 build tree (one .hip recompile + relink, venv-int2w lineage), get disassembly VGPR count + occupancy API for block (32,32) with 16 KB dynamic LDS. Must show <= 96 VGPR and 2 blocks/CU.
5. Per-row xcheck (protocol above). THEN gate branch: PASS -> full A/B; FAIL -> report.

## Hardened thinking discipline (post t1755 death: 47887-char thinking block, empty final)

- Read anything in <= 100-200 line chunks; one file section per turn.
- If a read triggers > ~2000 chars of thinking: stop, write the reasoning here, END TURN.
- Notes per section committed as they are written.
- FIRST measurement after design = per-row xcheck of the UNCHANGED v2 baseline (expect ~1.35 ms/row) to validate env before trusting post-change numbers.
- Salvage: push worktree HEAD to a study branch at the first chance after any real work.

## Env

- GPU box: bare metal, ssh <host-user>@<tailnet-ip> (sshpass). nohup needs </dev/null. Multi-line python3 -c in ssh double-quotes loses lines -> use heredoc or a file.
- GPU flock protocol: fuser the lock file BEFORE GPU work; kill own orphan flocks after.
- env: MODELS=<models-mount> + libhsa-runtime64.so.1 LD_PRELOAD + HCC_AMDGPU_TARGET=gfx1201; sidecar pin 02a54359 ONLY (verify pre/post); clock locked high.
- Build: t1697 build tree, incremental (one .hip recompile + relink) into venv-int2w lineage. Report new wheel sha256.

## Section notes

### 4-bit kernel (qmv_tiled_kernel.hip, 505 lines, read in 3 chunks)

- Grid (M, ceil(N/tile_n)), block (32, tile_n=32) = 1024 threads; one wave per column.
- BSK = VPT x 32 = 512 (VPT=16 for 4-bit). Per K-tile per lane: x_local[16] float from LDS (one LDS staging per K-tile, 16 values), w_local[2] u32 streaming, dequant_and_dot4 x 2, per-group acc += scale*reduce_qdot4 + bias*xsum.
- Then 5-step __shfl_xor warp reduce, lane 0 stores.
- Live window per K-tile: 16 x + 2 w u32 + dequant temporaries + d[4] + xsum + pointers/indices = the reported 51 VGPR. 2 blocks/CU.
- The x read pattern: 16 contiguous floats from LDS per lane, per K-tile, per lane. For K=4096 that is 8 K-tiles x 16 = 128 LDS x-reads per lane per column, and the kernel is 184.4 ns/column with them in the loop. => LDS x-reads in the FMA loop are PROVEN TOLERABLE at this shape (the v1 baseline's 10.2x was the 8-round per-element scalar decode + selection, not the LDS reads alone).

### int2w kernel (lm_head_int2w.hip, 245 lines, read in 3 chunks)

- Block (32,32), 1024 threads; x staged f32 once per block in dynamic LDS (16 KB at K=4096); xv[4][32] = 128 floats preloaded before the tile loop.
- FMA loop: per column, 4 iterations (8 j each), uint2 (8 packed bytes), per byte 4 planes (2-bit, shift+and fused into the FMA), d[4] accumulators.
- After: per-quarter 4-lane group reduce (shfl_xor 2,1) + group algebra acc += d[q]*sr[G] - sx[G]*zr[G] (G = 8*q + gl), 5-step warp reduce, lane 0 stores the logit, per-32-col top-8 selection by the ty==0 warp.
- v2 facts: 192 VGPR + 33 spills -> 1 block/CU; per-column 174.5 ns (5.4% faster than 4-bit). The per-quarter (32-j window) variant: 176 VGPR, 0 blocks/CU.
- Tail (201-245) = host eval_gpu, max_blocks = num_cus * 2, lds = K*4.

### Design v3 (candidate a): per-quarter streaming, 16 j x 2 sub-iteration chunks

- Drop the 128-value preload. Per column, per quarter q (4), two sub-iterations (16 j each): stage 16 x-values from x_shared LDS (4 x float4 vector LDS reads, 16 contiguous floats, same pattern as the 4-bit x_local), a 4 x u32 weight load (bytes j0..j0+15 = the 16 packed bytes for the 16 elements, 4-byte aligned: j0 = 128*gl + 32*l3 + 16*sub is a multiple of 8 so each u32 is 4-byte aligned; the 32-byte read for the quarter covers j=0..31 and stays inside the quarter).
- FMA order per lane per quarter is byte-identical to v2 (bytes in ascending j order, plane q innermost) -> per-lane d[q] accumulation sequence unchanged -> bit-identity with the scalar port by construction (probe re-verifies).
- Live window per sub-iteration: 16 x + 4 d + 1 acc + 4 w u32 + 4 byte temps + ~20 pointers/indices ~= 16 x + ~16 others, est. ~55-70 VGPR (target <= 96).
- No per-tile re-staging of x_shared (hoisted, as in v2); the 16 KB LDS staging stays once per block. Per lane per column: 4 quarters x 2 subs x 4 float4 = 32 LDS float4 reads (128 float reads) + 8 u32 weight reads + 128 FMA + 128 decodes. The 4-bit kernel runs the same shape of 128 LDS x-reads/column at 184.4 ns/column, so the added LDS traffic is proven tolerable.
- Everything after the FMA loop (group reduce, algebra, warp reduce, selection, host launcher, binding) is byte-unchanged.

### Log

- [x] start: head verified 6af56b9 on mirror + org (local branch ref e841eed is an unpushed report-only delta; working on 6af56b9 per the brief).
- [x] design settled (candidate a); notes committed.
- [x] kernel implemented (5370be7): 16 j x 2 sub streaming; FMA order byte-identical to v2.
- [x] env validated: venv-int2w on v2 wheel (sha256 e96f6c74, .so md5 matches the wheel); sidecar pin 02a54359cda8b247f76b277dd9bc4d4c VERIFIED PRE (the canonical model-mtp.safetensors; the -fixed and -4bit sidecars are .disabled, as since Nov 1 21:21); flock was free; GPU clock locked high per t1431.
- [x] BASELINE v2 per-row xcheck (env validation, same-session anchor): wide = 1365.4 us (1.365 ms/row, expect ~1.35 OK), head = 953 us, int2 scalar = 19248 us. Env good.
- [x] v3 kernel grafted into the build tree (md5 52a932652bb6a400 verified on the guest); incremental wheel build + standalone occ probe build started.
- [x] v3..v3e: 192 -> 118 VGPR; `__noinline__` is a no-op in ROCm.
- [x] v3f: 4-j chunk streaming (8 c x 4 j x 4 planes = 128 FMA/col; per-plane x windows x_shared[p*1024 + j0]; per-accumulator order j ascending) - the FMA math bug (single x window for all 4 planes) is fixed in this restructure.
- [x] v3i (per-c u32 weight load): 89 VGPR, 26 SGPR, 2 blocks/CU; bit-identity confirmed (16-row probe diff 0.0); xcheck wide = 9664 us (8x redundant 32B/c weight traffic: the compiler re-issues the whole window twice per chunk as 2 x b128).
- [x] v3j (32B preload + if/else select chain): 87 VGPR but 26 SGPR (scalar ALU branch scratch) - 1 block.
- [x] v3k/v3l/v3m/v3n/v3o: output-loop variants - 26-30 SGPR, all 1 block. The candidate b128-vectorized stores allocate a fresh s-pair (s[24:25]); volatile 64-bit locals are WORSE (30 SGPR); #pragma unroll is ignored on the runtime-r loop; explicit unrolling does not remove the s-pair.
- [x] v3q: removing N/K args does not reduce fixed SGPR (32-bit args are s_load'd on demand, not fixed slots).
- [x] v3s/v3t (merged cand allocation / 32-bit col index on the val base / NG hardcoded): 86 VGPR, 24 SGPR - STILL 1 block/CU (48-wave cap at 512 threads, LDS = 0).
- [x] OCCUPANCY INVERSION: v3i (89 VGPR, 26 SGPR) fits 64 waves; v3t (86 VGPR, 24 SGPR, identical instruction multiset) caps at 48. No static resource model (VGPR/SGPR caps, rounding to 8/16/32, per-CU budgets) is consistent with both. The 24-SGPR/wave budget is NOT the binding constraint (v3i exceeds it and fits; v3t under it does not).
- [x] v3x (v3i shape + volatile 4B weight: kills the b128 vectorization, 0 b128 in the .s): 97 VGPR, 26 SGPR, 48-wave cap. Volatile raises VGPR.
- [x] v3y (v3i shape + volatile 4B weight + permuted x layout, k2 = p*1024 + 128gl + 4j + l3, scalar loads): 62 VGPR, 22 SGPR, 2 blocks/CU. wheel + xcheck: 2675.9 us/row (1.96x v2) - the permutation is STILL a conflict: the 8 groups' 128gl base all map to the same 4 banks (8-way). sel_match = 0/256 is a STRUCTURAL artifact, not a bug: the scalar op's pool is 16 candidates per 128 cols (top-8 per 64) vs the wide op's 32 (top-8 per 32) - the top-32 sets differ by construction.
- [x] v3z = v3y + the cand_col pointer restored as an explicit kernel arg (the v3s merged-allocation op section does not compile: malloc_async returns a Buffer, not void*; op fixed to standard separate set_data). 62 VGPR, 22 SGPR, 2 blocks/CU.
- [x] v3b = v3z + bank-spread x layout k2 = p*1024 + 32j + 4gl + l3 (a 32-lane load at fixed (c,b) reads words 4gl + l3: 32 distinct banks, one LDS cycle; the staging store is a one-time 32-way conflict, amortized to ~8.5 cycles/tile over 121 tiles). 62 VGPR, 22 SGPR, 2 blocks/CU. wheel + xcheck: 1149.7 us/row (1.21x the anchor) - GATE FAIL; bit-identity re-confirmed (probe diff 0.0).
- [x] v3c = v3b + the 32-byte weight preload (2 x 16B at the tile head, non- volatile, L1-hot) + 7-ternary select per chunk (replacing the volatile 4B per-chunk load; the volatile flat load is the bottleneck - SCOPE_SYS path). 74 VGPR, 22 SGPR, 0 scratch, 2 blocks/CU (no occupancy anomaly in this structure, unlike v3t). wheel + xcheck x3: 570.8 / 570.4 / 571.8 us/row (0.60x the 953 us 4-bit anchor) - GATE PASS. int2 scalar = 18922-19015 us (v2 baseline 19248), head = 950-954 us. 16-row probe: max_abs_diff = 0.0 - bit-identity holds.
