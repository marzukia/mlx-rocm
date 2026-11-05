# REPORT-ceiling.md

Date: 2026-11-05. Base: `stack-assembly-bench` @ `7ea5bca` (the fold; 13-var
hybrid config-of-record, 69.43 t/s canon / 84.34 hello, verify wall 21.16 ms).
Branch: `ceiling-report`. GPU: guest-a (RX 9070 XT / gfx1201, 32 CU, 2048
threads/CU, 64 KB LDS/CU - the NOTES-int2vgpr box record), flocked, 25 s idle
before every measurement, SCLK sampled every 10 s, venv-m3k libmlx md5
`a0057c22...` (the COR wheel; re-verified pre and post), sidecar md5
`02a54359...` (verified pre and post, untouched).

Purpose: the per-class ceiling table for the 4-bit MTP-3 9B decode lane - what
each class is bound by (verified, not asserted), what each class can reach,
what variants remain open, and the concrete path to the 15.7 ms/step budget.

## 1. HBM floor (re-derived, this session)

Microbench: one H2D init of a 6 GiB f32 buffer, then `mx.sum` over the whole
buffer per iteration (pure read), median of 20 iters (5 warm); a read+write
copy arm for reference. Idle 2200 MHz at 25 s pre-sample; boost-to-load
readings up to 3309 MHz (the GPU is not hard-locked in this userland - the
same-session 25 s idle protocol is the clock control, as in the fold run).

| Arm | GB/s |
|---|---|
| READ (6 GiB `mx.sum`, median 20) | _FILL: floor_read_ |
| COPY (read+write, 2x bytes) | _FILL: floor_copy_ |

The floor used below is 546 GB/s (the t2003/fold record). If this session's
microbench re-derives a materially different number, the table below is
re-stated at the measured floor and the discrepancy is noted.

## 2. MLP: is it dequant/latency-bound? (3-arm same-session verification)

One process, one model load, AR 64 (probe uses 295; the per-class GEMM walls
are steady-state and AR-insensitive past warm), 24 layers x {gate/up:
N=12288 K=4096; down: K=12288 N=4096}, M=3, venv-m3k, `MLX_ROCM_QMM_M3=1`
(stock mr3 dispatch; class 2.718 GB, t2031 8.41 ms / 59.2%; the gen-2 probe
reproduced 8.175 ms / 60.9% with lm_head (the no-dequant 16-bit reference)
at 82.7% / 451.6 GB/s in the same session):

- **A** stock 4-bit mr3 (the COR baseline; t2031 class wall 8.41 ms)
- **B** same GEMMs with weights pre-dequantized to bf16 in HBM (4x weight
  traffic, zero in-loop dequant, zero scale/bias gather)
- **C** stock 4-bit mr3 pointed at one shared 80 MB weight set (layer 0's
  buffers, warm pass first) - isolates weight HBM traffic: if 80 MB fits the
  cache, C strips HBM weight latency from the measurement; if it does not
  (arch-tuning L2 = 8 MB), C ~ A and the isolation is inconclusive (noted)
- **C2** stock weights, constant scale/bias (isolates the per-group
  scale-gather ALU+traffic from the per-weight dequant)

| Arm | class wall (ms) | reading |
|---|---|---|
| A (stock 4-bit mr3) | _FILL_ | baseline (expect ~8.2-8.4; 332 GB/s) |
| B (bf16, 4x bytes) | _FILL_ | B/A ~ 4 => HBM-byte-bound; B/A < 4 (expect ~2.5-3, i.e. ~20-25 ms) => the 4-bit pipeline costs the gap to the ~83% no-dequant reference |
| C (shared 80 MB) | _FILL_ | C << A => weight HBM traffic is the bound; C ~ A (expected if L2 < 80 MB) => inconclusive, noted |
| C2 (const scale) | _FILL_ | C2 < A => the per-group scale/bias gather costs (2 loads + branch per group boundary); C2 ~ A => dequant-only cost |

Per-GEMM medians (A, one layer): gate _FILL_ / up _FILL_ / down _FILL_ ms.

Context classes (same session, probe definitions): qmv_gdn _FILL_ ms (COR
4.59), gdn_fused full layer _FILL_ ms (COR 4.33 - this includes the four
in_proj GEMMs; see 4.3), pure gdn_fused tape _FILL_ ms.

**Verdict:** _FILL after measurement_ (A/B/C/C2 discriminate cleanly: HBM-bound
=> C << A and B ~ 4x A; dequant-ALU-bound => C ~ A and B < 4x A).

## 3. GDN_FUSED occupancy (kernel-dump verification)

Kernel resources, dumped from the COR wheel's embedded code object
(`llvm-readobj --notes`, re-dumped this session):

| kernel | VGPR | SGPR | LDS | block |
|---|---|---|---|---|
| gdn_fused_step_tape TMAX=3 | 212 | 77 | 2464 B | 256 threads (8x wave32) |
| gdn_fused_step_tape TMAX=4 | 215 | 77 | 2464 B | 256 threads |
| qmv_tiled_kernel 4-bit GS64 | _FILL: 89-94 range_ | 41-44 | 2048 B | (32, 8) = 256 |
| qmv_wide_mr3_kernel (T=3, 4-bit) | _FILL_ | _FILL_ | 0 | (32, 16) = 512 |

Occupancy arithmetic (gfx1201: 32 CU, 2048 threads/CU, 64 KB LDS/CU; the
NOTES-int2vgpr empirical VGPR boundary - 2 blocks of 1024 threads fit at
<= 64-96 VGPR/lane, i.e. ~131072 VGPR/CU for 64-lane-worth budget - applied
here to wave32: 131072 / (R x 32) waves):

- TMAX=3 (212 VGPR/lane): 131072/(212x32) = 19.5 -> 19 waves = 608 threads ->
  **2 blocks (512 threads) fit** the resource files (16 waves x 212 x 32 =
  108,544 <= 131,072; 3 blocks = 24 waves = 162,816 > 131,072, no).
  LDS 65536/2464 = 26 blocks - not binding. SGPR not binding.
- The grid is **32 blocks** (num_v_heads) on **32 CUs** -> exactly 1 block per
  CU. Achieved occupancy = 256/2048 = **12.5% - the claim is true, but the
  binding constraint is the grid (32 blocks / 32 CUs), not the resource files
  (which admit 2 blocks/CU)**.
- "2 blocks per CU" is therefore only reachable by doubling the grid: 64
  blocks with half the per-block row work (the t2042 architecture: 64 blocks
  x 128 threads, h[32] per thread instead of h[64]). Per-block VGPR would
  drop ~212 -> ~116 (h/qq/kk 64->32 elements), 131072/(116x32) = 35 waves >=
  16 needed for 512 threads/CU - it fits.
- ms at 2 blocks/CU: the per-block serial chain (T rows x {3 rms_norm passes
  with 3 syncthreads each, 64-float h update, 2 shfl_xor reductions}) halves
  with the row split, so the tape wall -> ~half of the measured tape-only
  class. Tape-only measured _FILL_ ms (24 layers); ceiling at 2 blocks/CU
  (64 x 128) ~ _FILL/2_ ms, assuming latency-bound scaling holds (falsifier:
  a t2042 build > 0.75x the tape-only wall means the serial chain does not
  halve and the state work is not the bound).

## 4. QMV_GDN ceiling arithmetic (recomputed from the code, then the data)

### 4.1 The kernel structure (qmv_tiled_kernel, 4-bit, the COR dispatch)

Per k-tile (BSK = 512 K-values), per warp (32 lanes): PPT = 32/4 = 8
weights/uint32, 2 x uint32 = 16 weights = 8 B/lane (16 nibbles packed) -> 32
lanes x 8 B = **256 B per warp per k-tile**. Per block (tile_n = 8 warps):
8 x 256 B = **2048 B per k-tile**. Per k-tile the block: `__syncthreads`,
stages 512 floats of x into LDS (2048 B, 2 iterations of 256 threads),
`__syncthreads`, then per-warp LDS x-read + 2 x uint32 streaming weight load
(nt) + dequant-FMA + per-group (64) scale-dot + bias-xsum.

### 4.2 The launch (host side, qmm.hip 4-bit branch)

`blocks_per_cu = max_threads_per_cu / (tile_n x warp)` = 2048/256 = **8**
(host model); `grid_y = min(n_tiles, num_cus x blocks_per_cu)` =
min(N/8, 256). For N = 8192: **grid_y = 256, not 512** - the in-lane report's
"512 blocks" assumes 16 blocks/CU (4096 threads), which the 2048-thread
hardware limit forbids. The actual resident count is further VGPR-capped:
94 VGPR/lane (4-bit GS64 variant, 89-94 across group sizes) ->
131072/(94x32) = 43.5 -> 43 waves = 1376 threads -> **5 resident blocks/CU**
(8 blocks = 64 waves = 191,488 VGPR > 131,072). So at most **160 blocks
resident** of the 256 launched; the rest wait in the queue.

### 4.3 In-flight, corrected

- The report's "512 blocks x 256 B = 128 KB" double-miscounts: (a) the grid
  is 256, not 512; (b) it counts one warp's 256 B per block when a block has
  8 warps (2048 B per k-tile); (c) only resident blocks contribute to
  in-flight (160, not the full grid).
- Corrected in-flight potential: 160 resident blocks x 2048 B = **327 KB**
  (512 KB if all 256 launched blocks were resident). Both are >= the 273 KB
  requirement (546 GB/s x 500 ns HBM latency).
- Measured 203 GB/s (0.9095 GB / 4.59 ms) => effective in-flight =
  203 GB/s x 500 ns = **101 KB = 31% of the 327 KB potential, 37% of the
  273 KB requirement**.
- Q2 (double-buffer, 2x per-block in-flight, same block count) = -0.9% wash -
  consistent with per-k-tile in-flight already exceeding the requirement and
  the per-k-tile **serial chain** (LDS staging + 2 x `__syncthreads` per 512
  K-values, while the 8 B weight loads wait ~500 ns) being the binding
  constraint. The warps of a block are in-phase: all 8 wait on their weight
  loads together, then all 8 stage the next k-tile together - the 2048 B is
  in flight ~500 ns out of a ~900-1200 ns per-k-tile cycle.

### 4.4 K-split M=1 (4 K-splits -> 4096 blocks for N=8192, 6144 for the
N=12288 qkv GEMM): predicted ms

K-split changes the per-block chain from K/512 k-tiles to K/4/512, but the
**per-k-tile bubble per weight is invariant** (staging covers 512 K-values in
both; the grid-stride tile boundary adds no extra sync - the k loop just
continues). Resident block count is invariant (same block shape: 5 blocks/CU
by VGPR, 8 by the host model - unchanged). By the Q2 wash, in-flight is not
the binding variable, so K-split 4 is predicted to deliver **no meaningful
gain**: 4.4-4.7 ms (the small residual: 4x block turnover overlaps new blocks'
first-k-tile staging with retiring blocks' last k-tile; minus the 2-pass
partial-sum traffic (4 x N x 4 B per GEMM, ~0.2 MB for N=12288) and ~120
extra reduce launches). **Falsifier: a K-split build < 4.0 ms refutes this
model** (and the in-lane report's ~1.7-2.0 ms expectation, which rests on the
512-block/128-KB arithmetic corrected above).

### 4.5 The open variant that the corrected model points at

The binding constraint is the per-k-tile bubble (staging + 2 sync per 512
K-values). Two shapes remove it without changing the per-weight ALU:

- **Chunked x-staging (4x)**: stage 2048 floats (8 KB LDS) per 4 k-tiles, one
  `__syncthreads` pair per 4. Bubble per weight / 4. LDS: 5 resident
  blocks/CU x 8 KB = 40 KB <= 64 KB - fits. Predicted duty ~500/(500+~100) =
  83% -> ~450 GB/s -> class 0.9095 GB / 450 GB/s ~ **2.0 ms** (bracket
  1.67-2.1; falsifier: > 2.6 ms).
- **Direct x from L2**: drop the staging; each lane reads its 16 bf16 x-values
  (32 B, x row = 8 KB, L2-hot - read by 1024 column warps) with no sync at
  all. Same predicted bracket; more load instructions, zero sync.

## 5. Exhaustion audit (per 4-bit class)

### mlp (t2031: V1-V4 all FAIL; class 8.41 ms, 59.2% of floor; 2.718 GB per
the probe; the same-session no-dequant reference is lm_head at 82.7% / 451.6 GB/s)

Tested and dead (mlp-gen2 probe, same-session, lm_head clock control 449.7-454.1 GB/s):
- V1 (LDS x-staging + per-iteration `__syncthreads`, 512-thread block): 18.61 ms (2.27x)
- V2 (k-tile double buffer): 9.44 ms (1.16x - the dequant moved onto the FMA's dependent path)
- V3 (32B weight loads): 15.15 ms (1.85x - 8 serial scalar volatile loads where the stock issues one 16B vector load; the unclamped window was OOB at the last k-tile - a genuine correctness find)
- V4 (batched-x issue order): 8.69 ms (+6.3% - least-bad; bench A/B -0.95%)

Open (the axis the variants did not touch, per the gen-2 ceiling analysis):
- the 3-arm verdict of section 2 (dequant ALU vs scale gather vs weight HBM
  latency - the arms discriminate; C2 isolates the scale gather directly);
- pre-scaled / packed weights removing the in-loop dequant (the gen-2
  recommendation; changes the storage format - a new class, not a variant);
- launch geometry / occupancy of the 768 col-blocks x 512-thread CTAs;
- split-K (the mr3 kernel has none; the tiled kernel's 5-resident-block cap
  would apply equally - see 4.2).

### qmv_gdn (t2040: Q1-Q4 all FAIL; class 4.59 ms, 36.3%; 0.9095 GB; C 201.8 GB/s)

Tested and dead (qgd-gen2 probe, tiled-target build, lm_head clock control
451.8-454.0 GB/s, Q1 no-op control +0.4%):
- Q1 (no-op control - V1's x-staging is already the stock tiled design): +0.4%
- Q2 (k-tile double buffer, 2x per-block in-flight, same block count): -0.9% (wash)
- Q3 (16B weight window, the 32B pattern at 8B slice width): -0.2%
- Q4 (batched-x load order): -1.4% single-run, not reproducible in 3x3 A/B (dead tie)

The session-A misdiagnosis (variants ported onto the wide per-row kernel,
which does not serve bias-free M=1 ops - `wide_ok` requires
`has_bias || (M>=2 && M<=4)`; `MLX_ROCM_QMV_NO_TILED=1` slows the class,
i.e. tiled is the live path) is recorded in the in-lane report.

Open: K-split M=1 (predicted wash by section 4.4 - falsifier 4.0 ms),
**chunked 4x x-staging** and **direct-L2 x** (section 4.5, predicted ~2.0 ms).

### gdn_fused (unmodified kernel; class 4.33 ms, 4.4%)

Tested: nothing this lane (t2042 occupancy-target build not yet run). Open:
the t2042 architecture (64 blocks x 128 threads, h[32]) - section 3 gives the
occupancy arithmetic (2 blocks/CU is resource-feasible, grid-bound today) and
the ~halved-tape ceiling.

### 4.3 audit finding (double count)

The gdn_fused class wall **includes the four in_proj GEMMs** (qkv 25.2 MB,
z 8.4 MB, b/a small - 50 MB/layer x 24 = 1.2 GB at 323 GB/s ~ 3.7 ms) while
its byte accounting (state/z/b/a/conv/norm ~102 MB) excludes them; qmv_gdn
counts the same GEMMs once. The per-class table therefore double-counts
~3.7-4.3 ms across the two classes; the 21.16 ms fused verify total is the
ground truth, and the tape-only measurement (section 2 context) splits the
4.33 into GEMM (~3.7) vs tape (~0.6) components.

## 6. Ceiling table (per 4-bit class, COR session 21.16 ms verify)

| Class | current ms | bound (verified) | ceiling ms | lever | open variant |
|---|---|---|---|---|---|
| mlp (24x3 QMM) | 8.41 | _FILL: arm verdict_ | _FILL_ | _FILL_ | _FILL_ |
| qmv_gdn (24x5 QMM) | 4.59 | per-k-tile bubble chain (4.3: 101 KB effective vs 273 KB requirement; Q2 wash) | ~2.0 (4.5) | chunked 4x x-staging / direct-L2 x | K-split (predicted wash, falsifier 4.0 ms) |
| gdn_fused tape | 4.33 (full layer; ~_FILL_ tape-only) | 32-block grid / 32 CU (resource files admit 2 blocks/CU; 12.5% achieved) | ~_FILL/2_ (t2042 64x128) | grid 64 x 128 threads, h[32] | t2042 build (falsifier: > 0.75x tape wall) |
| attn (8x) | 1.70 | S4VA + 4 QMM (unmodified) | ~1.4 (QMM part) | shares the qmv_gdn lever | inherits qmv_gdn open variants |
| norms | 1.47 | elementwise (0.1% of floor - not BW) | ~1.4 | launch/overhead | out of this lane's scope |
| lm_head | 1.27 | 82.3% of floor - near BW-bound | ~1.25 | (staged MR, N=248320) | none open |
| **verify total** | **21.16** | | **~11.5-12** | | |
| draft+schedule | 10.43 | draft 2.6 (3.6 GiB trunk re-read per draft token - ESTIMATE) + scheduling/launch overhead 7.8 | ~6.5-7 (ESTIMATE: overhead cut to ~4) | graph-capture/launch batching (mtp-graph-capture-probe line) | _ESTIMATE_ |

## 7. The 15.7 ms/step budget path

Target: 140 t/s at tok/step 2.193 = 15.66 ms/step. Current 31.59 ms/step =
21.16 (verify) + 10.43 (draft+schedule). Must remove **15.93 ms**:

1. **verify 21.16 -> ~12 (remove ~9.2)**: mlp 8.41 -> _FILL_ (section 2
   lever), qmv_gdn 4.59 -> ~2.0 (section 4.5), gdn_fused tape 4.33 ->
   _FILL/2_ (t2042), attn/norms/head ~flat at ~4.4.
2. **draft+schedule 10.43 -> ~6.5-7 (remove ~3.5-4)**: ESTIMATE - the draft
   2.6 ms (3.6 GiB trunk re-read) is near-fixed; the cut is
   scheduling/launch overhead (7.8 -> ~4) via the graph-capture line.
3. Total: ~18.5-19 ms/step -> ~115-119 t/s at the 12 ms verify; the last
   ~2.5 ms to 15.66 needs the mlp lever to hit its floor (section 2) or the
   draft re-read to shrink - both tracked, both beyond this report's
   verification (marked ESTIMATE in the table).

## Check lines

- persona-leak grep: 0
- hard-wrap violations: 0
- ruff: 0
