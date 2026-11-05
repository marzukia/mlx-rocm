# REPORT-mlp-gen2: next-generation mlp-class GEMM variants vs the adopted mr3 kernel

**Probe.** The mlp class of the qMLX MTP-3 stack was 59.2% of the 546 GB/s HBM floor
(8.407 ms, 2.718 GB) at the fold re-gate (REPORT-fold-qmm-hybrid.md), below the 65%
adoption floor. This probe implements four next-generation variants of the adopted
`qmv_wide_mr3_kernel` (shared-dequant, M in [2,4], 4-bit), dispatches them in the same
`!mr_ok` per-row fallback branch, and A/Bs them per-class (bandwidth probe) and, for the
best variant, end-to-end (mtp-bench6, 13-var config-of-record, 322-token anchor, 3 rounds
interleaved C vs F). All variants are DEFAULT OFF (`MLX_ROCM_QMM_M3_V1..V4` unset = the
stock mr3 kernel runs verbatim; `MLX_ROCM_QMM_M3` is the existing mr3 gate, ON in every
arm).

**Protocol.** Guest-a (marzuki-neon, RX 9070 XT / gfx1201), flock, 25 s idle before each
run, `MODELS=/mnt/models LD_PRELOAD=libhsa-runtime HCC_AMDGPU_TARGET=gfx1201`. venv-gen2
= copy of venv-p3 + force-reinstall of the gen2 wheel (vendor tree @ 7ea5bca, byte-identical
to the tree that built venv-m3k, + the four gen2 kernels). Bit-identity contract per variant: identical per-(row, col) FMA traversal (ascending k-tiles, la q-major / j-minor, per-value lx FADD guarded by has_bias, same int→float conversions, same
`acc += s*la + b*lx`, same 32-lane `shfl_down` butterfly and lane-0 store, same early
return `col >= N` before any weight read, same shared-dequant FMA:dequant = M:1).
Only the memory-access structure changes.

**md5-of-record (this session).**

| artifact | md5 |
|---|---|
| sidecar `model-mtp.safetensors` (pre = post, all sessions) | `02a54359cda8b247f76b277dd9bc4d4c` |
| venv-m3k libmlx (pre = post, never touched) | `a0057c220d31a0ec87a8b373a97625e8` |
| venv-p3 libmlx (pre = post, never touched) | `460c803d66c43ebfce8997ef08c912db` |
| venv-gen2 libmlx (distinct from p3 and m3k; 4 `MLX_ROCM_QMM_M3_V*` env strings compiled in) | `199f9996a8f555b885aeea0839a425a8` |
| probe script `verify-bw-perop-m3k.py` | `d99ba9ae53fdd37c46ae351bfbcd45db` |
| bench `mtp-bench6.py` (repo copy, hybrid default) | `2e6d71b7b54a2614e055c7db313b7bc0` |
| vendor-gen2 qmm.hip / gen2 kernel (session A) | `993c383e6f68d6c74097136855bf2ee0` / `f4c3dfd13a13424b615bb3ee810aa923` |
| gen2 kernel V3 OOB-clamp revision (session A3) | `f40f6baefcbfdd87c63a10bd1a38e81a` |
| wheel | `mlx-0.32.3.dev20261105+32a2900-cp314` (git sha of the build tree) |

## The four variants

- **V1 — hoisted la/lx (LDS x-staging).** The per-row x loads are staged ONCE per k-tile
  into LDS per warp (32 lanes × 32 values), shared across all 16 cols of the warp, instead
  of re-reading the same x values from global/L2 once per col (16× redundant x traffic).
  One row at a time (1024 values/warp slab), `__syncthreads` after each stage; uniform
  trip count with a per-iteration `k < K` guard so the in-loop barrier is block-uniform.
  LDS = 16 warps × 1024 values × sizeof(T): 32 KB (bf16/fp16) / 64 KB (f32); the dispatch
  gate caps it (`m3block.x × 128 ≤ 65536` for all dtypes; LDS passed as the dynamic-shared
  bytes of the launch).
- **V2 — k-tile double buffer.** Software pipeline within the warp: the weight load of
  k-tile i+1 is issued at the top of iteration i (before the FMA of tile i) and
  dequantized into the alternating (A/B) weight register buffers AFTER the FMA of tile i, hiding the next tile's load latency under the current tile's x loads + FMA + dequant.
- **V3 — 32B weight loads.** The 16B weight vector load is widened to a 32B logical access
  (two consecutive 16B aligned loads; the 4-bit layout is 16B-aligned per lane for the
  stored sidecar, the window is 32B-aligned for even lanes; no weight reordering, the
  sidecar is read exactly as stored). Only the first half (the stock 32 values) is used in
  the math; the second half is accessed with an input-only `asm volatile`
  elision-proofing (read-write `"+r"` constraints are unlowerable in AMDGPU codegen:
  "illegal VGPR to SGPR copy") and is **clamped to in-row windows only** (`k + 64 ≤ K`):
  the 32B window overruns the row end (and the buffer) by up to 16B at the last k-tile
  where the stock 16B load fits exactly ((K−32)/2 + 16 = K/2) — the unclamped version
  faulted (HSA `MEMORY_FAULT`, `qmv_wide_mr3_v3_kernel<hip_bfloat16, hip_bfloat16, 64>`),
  was rebuilt and re-probed (session A3, arm V3b).
- **V4 — load-pattern variant.** The per-row x loads of the k-tile are hoisted into
  registers and issued as ONE batched group (all rows, 16 × 16B) BEFORE the weight load,
  instead of the stock per-row interleave (x loads after the weight dequant, one row at a
  time). Same stored layout, same per-(row, col) FMA order — only the memory issue order changes (x batch ahead of the weight stream, for HBM row-buffer locality).

Note on class coverage: the m3 dispatch branch is shared by ALL 4-bit M∈[2,4] QMM ops of the stack, so each variant kernel also serves the qmv_gdn / gdn_fused / attn qkv projections (that is why the per-class table below shows cross-class deltas for the
variants; the mlp row is the 3 large MLP GEMMs, 2.718 GB).

## Per-class bandwidth probe (one session, arms C / V1 / V2 / V3b / V4b)

lm_head (the clock-drift control) held 449.7–454.1 GB/s (82.4–83.2% floor) across all
five arms — no clock drift. `FUSED_VERIFY_MS` = the 322-token verify pass.

| class | C (stock mr3) | V1 | V2 | V3b | V4b |
|---|---|---|---|---|---|
| **mlp** | **8.175 ms — 332.5 GB/s — 60.9%** | 18.609 ms — 146.1 GB/s — 26.8% | 9.440 ms — 287.9 GB/s — 52.7% | 15.152 ms — 179.4 GB/s — 32.9% | 8.686 ms — 312.9 GB/s — 57.3% |
| qmv_gdn | 4.516 ms — 201.4 GB/s — 36.9% | 8.376 ms — 108.6 — 19.9% | 4.821 ms — 188.7 — 34.6% | 7.738 ms — 117.5 — 21.5% | 4.529 ms — 200.8 — 36.8% |
| gdn_fused | 4.318 ms — 24.0 GB/s — 4.4% | 7.218 ms — 14.3 — 2.6% | 4.661 ms — 22.2 — 4.1% | 6.684 ms — 15.5 — 2.8% | 4.418 ms — 23.4 — 4.3% |
| attn | 1.674 ms — 164.7 GB/s — 30.2% | 2.715 ms — 101.6 — 18.6% | 1.792 ms — 153.9 — 28.2% | 2.583 ms — 106.7 — 19.5% | 1.732 ms — 159.2 — 29.1% |
| norms | 1.461 ms | 1.488 ms | 1.479 ms | 1.474 ms | 1.494 ms |
| lm_head | 1.267 ms — 451.6 GB/s — 82.7% | 1.264 ms — 452.5 — 82.9% | 1.260 ms — 454.1 — 83.2% | 1.268 ms — 451.2 — 82.6% | 1.272 ms — 449.7 — 82.4% |
| fused_verify (total) | **20.96 ms** (eff 218.5 GB/s, 40.0%) | 35.74 ms (128.1, 23.5%) | 22.78 ms (201.0, 36.8%) | 31.66 ms (144.6, 26.5%) | 21.57 ms (212.3, 38.9%) |

**Result: no variant beats the stock mr3 kernel.** C = 60.9% floor (the 59.2% fold number
reproduced within session noise). Best variant = **V4** (least-bad: mlp 8.686 ms, +6.3%
slower than C; every other class within +0.3…+3.5%). V1 (LDS staging + per-iteration
`__syncthreads` over a 512-thread block, ~49 barrier rounds for K = 49344), V2 (the
dequant moved after the FMA serializes the pipeline it was meant to hide) and V3 (8
serial scalar volatile loads per tile where the stock issues one 16B vector load) all
regress hard (mlp 1.85× / 1.16× / 1.85×). V4's batched-x issue order costs a small
constant (the x batch is issued before the weight stream instead of interleaved).

## Bench A/B (best variant V4 vs C), 13-var config-of-record

Config (both arms; F additionally sets `MLX_ROCM_QMM_M3_V4=1`):
`MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1
MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4
MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0
MLX_ROCM_QMM_M3=1`. 322-token canonical anchor + "Hello", 3 rounds interleaved C/F/C,
25 s idle before each run, per-run timeout 600 s.

| round | arm | gen_ms | toks | toks/s | tok/step | D1/D2/D3 acc |
|---|---|---|---|---|---|---|
| r1 | C | 6.27 | 436 | 69.55 | 2.193 | 70.8/68.4/0.0 |
| r1 | V4 | 6.35 | 436 | 68.70 | 2.193 | 70.8/68.4/0.0 |
| r2 | C | 6.30 | 436 | 69.22 | 2.193 | 70.8/68.4/0.0 |
| r2 | V4 | 6.35 | 436 | 68.71 | 2.193 | 70.8/68.4/0.0 |
| r3 | C | 6.30 | 436 | 69.17 | 2.193 | 70.8/68.4/0.0 |
| r3 | V4 | 6.36 | 436 | 68.54 | 2.193 | 70.8/68.4/0.0 |

Means: **C 69.31 tok/s (6.29 ms)**, **V4 68.65 tok/s (6.35 ms)** → **V4 = −0.95%** (a
regression, consistent with the probe's mlp +6.3% once the mlp class is folded into
the full verify). hello arm: C 84.44 vs V4 83.35 tok/s (−1.29%). VERIFIES=161 and
acc identical across all 12 runs.

**FIRST20/ALLTOKS drift check: PASS (bit-identity held).** ALLTOKS md5 identical
across all 6 canonical runs (md5/8 `14262530`, C and V4, r1–r3) and all 6 hello runs
(`3d3d58dc`); FIRST20 byte-identical on every run. The variants are numerically
inert — the −1% is pure access-pattern overhead, not a math change.

## Verdict

**FAIL (as expected under the probe charter).** Bar: mlp floor ≥ 65% AND bench ≥ +3%
same-session. Actual: best mlp floor = **C itself at 60.9%** (no variant ≥ C; best
variant V4 at 57.3%, −1.2 pts below C); bench A/B of the best variant (V4) =
**−0.95%** (a regression, bit-identical output). No variant reaches the 65% floor, so
per the probe charter the deliverable switches to the ceiling analysis below.

## Ceiling analysis: why the mlp class stalls at ~60% floor

The five data points of this probe, read together, say the same thing from five angles:
the stock mr3 kernel is already at the structural limit of this access pattern, and the obvious "next-generation" memory reorganizations all trade one bound for a worse one.

1. **The mlp class is not bandwidth-bound in the way the floor suggests — it is
   latency/occupancy-bound under a 4-bit dequant pipeline.** C achieves 332.5 GB/s
   (60.9%) on the 2.718 GB mlp traffic; lm_head (pure 16-bit GEMM traffic, no
   dequant) hits 451.6 GB/s (82.7%) in the same session. The ~22-point gap is the
   cost of the 4-bit dequant + per-group scale/bias gather interleaved with the
   weight stream. Every variant that touched that interleave (V1, V2, V3) lost;
   the one that only reordered the x side (V4) lost a small constant. The kernel
   is issue/latency-bound, not HBM-bound, at this shape — which is exactly why
   "more clever memory access" does not move it.
2. **Synchronization is the wrong axis (V1).** Staging x in LDS removes the 16×
   redundant x traffic, but the mlp class's x is already L2-resident (the 3 M-rows
   × K bf16 stream is ~6 MB, re-read per col-block); what V1 bought with that was
   ~49 block-wide `__syncthreads` barriers per CTA (K/1024 rounds) over 512
   threads — a 2.27× mlp regression. The redundant x reads were cheap; the
   barriers were not.
3. **Software pipelining the dequant is a losing trade at this tile size (V2).**
   Prefetching the next tile's weight and dequantizing it after the current FMA
   moved the dequant (the ALU-heavy part) into the dependent path of the FMA
   instead of overlapping it: mlp 1.16×. With 32-value tiles the dequant is small
   enough that the stock issue order (dequant → FMA, tile serial) already keeps
   the pipeline fed by the *next tile's* outstanding loads; explicit double
   buffering only added register pressure (two 32-float weight buffers per lane).
4. **Wider weight loads don't help when the layout is 4-bit (V3).** The 16B vector
   load is already the layout's natural quantum (32 values / lane). Widening to a
   32B logical window means 8 scalar volatile loads (the backend cannot merge
   volatile scalar loads into a vector) where the stock issues one 16B vector load — 1.85× slower. (And the unclamped window was OOB at the last k-tile: a
   genuine correctness find, fixed with the `k + 64 ≤ K` clamp.)
5. **The remaining headroom to the 65% floor (60.9% → 65% ≈ 5.1% of 8.175 ms ≈
   0.42 ms) is not in the weight/x access structure.** It is in the things the
   variants did NOT touch: (a) the per-group scale/bias gather (one scalar load
   per tile per group — `load_scale_value` is 2 loads + a branch at group
   boundaries); (b) the 4-bit→float dequant ALU (32 shifts + masks + conversions
   per tile); (c) launch geometry (16 col-blocks of 16 cols; N = 12288 for the
   gate_up GEMM → 768 col-blocks × M-rows of CTAs, each 512 threads) and its
   occupancy/queueing; (d) the bf16→f32 x conversion inside the FMA loop. Those
   are kernel-internal ALU/scheduling changes, not memory-access reorganizations —
   a different (and riskier, for bit-identity) axis than the four variants probed.

**Recommendation.** Do not adopt any of the four variants (all ≤ C; V4 is the least
bad and is the only one close enough to consider a future A/B against a dequant-
pipeline rework). Keep the stock `qmv_wide_mr3_kernel` as the mlp-class kernel of
record. If the 65% floor is a hard requirement, the next probe should target the
dequant/scale path (fused scale gather, pre-scaled weights at load time, or a
higher-bit storage for the mlp class), not the access structure — and the mlp
class should be re-measured against the 82.7% lm_head ceiling to size the real
gap before another variant battery.
