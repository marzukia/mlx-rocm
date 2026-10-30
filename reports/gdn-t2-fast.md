# GDN T=2 fast: fix the non-linear S=2 verify cost (K=1 MTP spec-decode)

Date: 2026-10-27 (guest UTC). Worker: kernel-perf (an earlier run). Machine: the build VM,
RX 9070 XT / gfx1201, ROCm 10.1 guest userland. Model: /models/qwen3-9b-mlx
(4bit, 9B GDN-hybrid = 24 GDN + 8 attention layers).

## Result (one line)

The batched S=2 verify was 4.6x a T=1 forward because the **WMMA dense GEMM
(prefill-shaped 64x128 tile) is gated on `M > 1`**, so M=2 (and M=2..8) ran the
prefill tile GEMM instead of the QMV decode path. Gating it on `M > 8` routes
small-M decode/spec-decode shapes to the QMV path: **S=2 is now 1.32x T=1**
(18.45 ms vs 13.94 ms; target 2.0-2.3x), **MTP batched 15.66 -> 63.8 tok/s
(4.1x)**, MTP=0 unchanged (64.49 vs 64.94), GDN prim bit-exact (maxdiff 0.0),
MTP=2 stream byte-identical to MTP=0 (greedy).

## 1. Per-component timing (one S=2 vs two T=1 forwards)

Per-layer wall timing (sync-sandwiched, /tmp/g1-t2-layers.py), pre-fix wheel
mlx-0.32.3.dev20261026+fa18fbf (sha f6c17db5...):

| component | T=1 (ms) | S=2 (ms) | ratio | verdict |
|-----------|----------|----------|-------|---------|
| **QuantizedLinear (all quant matmuls)** | **10.7** | **124.8** | **5.84x** | **culprit** |
| lm_head (M=1 / M=2 QMV vs WMMA) | 0.49 | 9.43 | 9.57x | same cause |
| RMSNorm (all 33 norms) | 2.2 | 6.4 | 1.45x | near-linear, OK |
| embed + final norm | ~1.0 | ~2.0 | ~2x | OK |
| GDN layers (24) | - | - | 2.0x | bit-exact (prim check) |
| attention (8, vector decode kernel, qL<=4 both T=1 and T=2) | - | - | ~2x | OK |
| **total (unsandwiched)** | **13.9** | **118.8** | **4.26x** | (4.98x sandwiched) |

The excess is entirely in the quantized matmuls: +103 ms (QuantizedLinear) + 9 ms
(lm_head) at S=2. GDN/attention/norms are ~2x as expected.

## 2. Root cause

`QuantizedMatmul::eval_gpu` (mlx/backend/rocm/quantized/qmm.hip): the fused WMMA
quantized GEMM — 64x128 output tiles, grid ((M+63)/64, (N+127)/128) — is a
**prefill-shaped** kernel, but its dispatch gate was `M > 1`. So:

- M=1 (decode): gate false -> QMV decode path (tiled QMV, grid (M, grid_y),
  per-row independent, weight streamed once per row). Fast.
- M=2..8 (decode + spec-decode verify): gate true -> WMMA tile GEMM. At M=2 each
  64-row tile does 32x the useful work; measured **5.84x** the QMV path.

The QMV decode path already supports small M: the wide QMV kernel is spec'd for
M<=8 (grid (N/wcols, M)); the tiled QMV kernel takes any M (grid (M, grid_y),
`x_row = x + m*K`, per-row accumulators); the dequant/rocBLAS crossover
(select_qmv_qmv_crossover thresholds 16/12/8, arch config) takes over at M>=8 on
large shapes by design. M=2..7 therefore lands on QMV for every shape in this
model once the WMMA gate is raised.

A/B proof without a rebuild: `MLX_ROCM_WMMA_QMM=0` (opt out of WMMA QMM) drops
S=2 from 118.8 ms to 38.9 ms (QuantizedLinear 124.8 -> 28.5 ms, lm_head 9.43 ->
2.07 ms).

The earlier mtp-bench.md finding ("no fast small-T GDN path") was real but is NOT
the cause: the T=2 GDN forward runs the recurrent kernel via the T<=8 dispatch
change (below) and is bit-exact vs 2x T=1 (state/y maxdiff 0.0) at ~2x cost.

## 3. Fix (smallest correct)

Two changes, both in the guest build tree /models/mlx-clone-a (HEAD fa18fbf),
both now ported to vendor/mlx-rocm-upstream/:

1. **qmm.hip**: WMMA dense GEMM gate `M > 1` -> `M > 8` (+ comment). M=1 path
   unchanged (it never qualified for WMMA); M=2..8 -> QMV decode; M>=9 -> WMMA
   (prefill behavior unchanged).
2. **gated_delta_update.cpp**: GDN dispatch `T == 1` -> `T <= 8` (pre-existing
   uncommitted change, now included so the repo matches what runs). T=2..8 run
   the recurrent kernel; verified bit-exact vs 2x T=1.

T=1 byte-identical by construction: M=1 was already excluded from WMMA, and T=1
satisfies T<=8 and runs the same recurrent kernel with the same args. fp32 GDN
state untouched.

## 4. Verification (new wheel mlx-0.32.3.dev20261027+fa18fbf, sha f2290a7a...)

- **GDN prim** (/tmp/mtp-bench-gdn3.py): S=2 vs 2x T=1 — state maxdiff 0.0,
  y maxdiff 0.0 (bit-exact). PASS.
- **S=2 forward** (unsandwiched, /tmp/g1-t2-parts.py): 118.8 ms -> **18.45 ms =
  1.32x T=1 (13.94 ms)**. Target 2.0-2.3x — beaten (sub-2x because T=1 is
  launch-bound and T=2 amortizes per-op fixed cost).
- **bench4** (1024 tokens, greedy, fresh process per run, flock):

| mode | pre-fix (tok/s) | post-fix (tok/s) | target |
|------|-----------------|------------------|--------|
| 0 (AR) | 64.94 | **64.49** | ~64.9 unchanged: PASS |
| 1 (2x T=1 verify) | 54.32 | 54.71 | - |
| 2 (S=2 verify) | **15.66** | **63.81 / 63.68 / 63.46** | >=70: 9% short (see 5) |

  ACCEPTED=389/512 (76%) on m1 and m2.
- **LOSSLESS**: m2 token stream byte-identical to m0 (ALLTOKS match), m1==m2a,
  all three m2 runs identical. PASS.

## 5. Gap analysis: 63.8 vs 70 tok/s

Per-verify breakdown of the m2 run (512 verifies, 389 accepts, 123 rejects,
16.05 s total):

| component | cost | share |
|-----------|------|-------|
| S=2 verify forward (512 x 18.45 ms) | 9.45 s | 59% |
| reject re-forward (123 x 2 x 13.94 ms) | 3.43 s | 21% |
| draft step (512 x 3.73 ms: embed 0.09 + MTP module 1.79 + lm_head 1.85 + sync 0.02) | 1.91 s | 12% |
| Python loop overhead (save/restore GDN, KV trim, mx.array, argmax syncs) | ~1.26 s | 8% |

S=2 is already 1.32x T=1 (sublinear — the QMV M=2 path does 2x the M=1 work and
the rest of the forward is launch-bound). The remaining gap to 70 tok/s is
structure/harness overhead (reject re-forwards + draft + Python), not S=2.

## 6. Next candidates (not done here — beyond the smallest fix)

1. **2-rows-per-block QMV for M=2**: the tiled QMV streams the weight tile once
   per m-row block (grid (M, grid_y)); a 2-row block halves weight streaming
   (M=2 ~= 1.1-1.3x M=1 instead of 2x). Per-row results stay bit-exact
   (independent accumulators, same grid-stride tile order), T=1 untouched.
   Projected m2 ~= 72-78 tok/s.
2. **Mid-verify GDN state**: expose the intermediate (after token 0) state from
   the batched recurrent kernel; the reject path drops from 2x T=1 re-forwards
   to 1x T=1 (saves ~1.7 s -> +~8 tok/s).
3. **Graph-capture the decode loop**: the port is launch-bound (~838
   launches/token); the G1 diagnostics (now in the tree, env-gated) are the
   groundwork.

## 7. Vendor sync + rollback

- Ported the full rocm-backend diff from the guest build tree to
  vendor/mlx-rocm-upstream/: 19 files + new g1_hiplog.h (the G1 diagnostics are
  env-gated and harmless off; included so the repo matches what runs). Includes
  the T<=8 GDN dispatch (previously uncommitted) and the WMMA M>8 gate.
- Rollback: previous live wheel mlx-0.32.3.dev20261026+fa18fbf
  (sha f6c17db563d1b7fe83cbaec0b9e82a0bebc88b4cf2f7ae143f48f9ef7a2eb8b2);
  new wheel mlx-0.32.3.dev20261027+fa18fbf
  (sha f2290a7a935bc266c416525dae8e93d79a405e0c6da90c4ee30de7bf9ad099be).

## Assets

- Guest /tmp: g1-t2-layers.py (per-layer sandwich timing), g1-t2-parts.py
  (unsandwiched T=1/S=2/draft timing), g1-draft-parts.py (draft components),
  mtp-bench-gdn3.py (GDN prim), mtp-bench3.py + bench4.sh (bench),
  g1-wmma-gate-fix.py (the edit script).
- Bench outputs: /models/tmp/bench4/ (m0, m1, m2a/b/c, full streams).
- Pre-fix report: /srv/<fleet>/work/608/reports/mtp-bench.md.
