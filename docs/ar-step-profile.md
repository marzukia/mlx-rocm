# AR T=1 decode: per-layer kernel-selection profile (static)

Task: close the AR gap 73.45 → ~87 t/s (llama.cpp anchor 86.9) on the MTP=0
fused-AR path. ~2.1 ms/token = ~66 µs/layer across 32 layers; GEMM is 77% of
total (10.7 ms of 13.9 ms sandwich, 10-27 per-layer numbers).

This is the STATIC profile (no GPU): the exact kernel selection for every op in
one AR decode step, canonical wheel (0.32.3.dev20261102+75ca881, venv-canon),
standing config `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1`,
model `/mnt/models/qwen3-9b-mlx` (4-bit affine, group 64, bf16, 4096 hidden,
32 layers = 24 GDN + 8 full-attention, D=256, GQA 32:4, vocab 248320).

> Note on the standing config: `MLX_ROCM_REFWD_M1` is a no-op in venv-canon
> (the env does not exist in that wheel; the t1434 wheel has it, but its gate
> is rows-aware — with QMV_ROWS=1, M=1 stays on the wide path). So the
> standing config's M=1 QMMs run the per-row **wide** kernel in both wheels.

## Per-GDN layer (×24)

| # | op | shape | kernel (canonical dispatch) | weight traffic |
|---|----|-------|-----------------------------|----------------|
| 1 | input_layernorm | 1×4096 | `rms_norm_kernel` (256t block, 2-pass) | 8 KB |
| 2 | in_proj_qkv | 1×4096→8192 | `qmv_wide_kernel` (M=1; rows gate is M≥2) | 16.8 MB |
| 3 | in_proj_z | 1×4096→4096 | `qmv_wide_kernel` | 8.4 MB |
| 4 | in_proj_b | 1×4096→32 | `qmv_wide_kernel` (4 blocks) | 65 KB |
| 5 | in_proj_a | 1×4096→32 | `qmv_wide_kernel` | 65 KB |
| 6 | conv_state update | concat/clip/arange/take_along_axis | small indexing + copy kernels | ~0 |
| 7 | gdn_fused_step | conv1d+silu+qk-norm+recurrent+gated-norm, 1 launch (256t) | `gdn_fused_step_kernel` (GDN_FUSE=1) | 32 KB conv + ~2 MB state r/w |
| 8 | out_proj | 1×4096→4096 | `qmv_wide_kernel` | 8.4 MB |
| 9 | post_attention_layernorm | 1×4096 | `rms_norm_kernel` | 8 KB |
| 10 | gate_proj | 1×4096→12288 | `qmv_wide_kernel` | 25.2 MB |
| 11 | up_proj | 1×4096→12288 | `qmv_wide_kernel` | 25.2 MB |
| 12 | swiglu | 3 small binary ops | binary kernels | 0 |
| 13 | down_proj | 1×12288→4096 | `qmv_wide_kernel` | 25.2 MB |

GDN-layer GEMM: 103.8 MB; ~14 launches.

## Per-attention layer (×8)

| # | op | shape | kernel | weight traffic |
|---|----|-------|--------|----------------|
| 1 | q_proj | 1×4096→8192 | `qmv_wide_kernel` | 16.8 MB |
| 2 | k_proj | 1×4096→1024 | `qmv_wide_kernel` | 2.1 MB |
| 3 | v_proj | 1×4096→1024 | `qmv_wide_kernel` | 2.1 MB |
| 4 | q_norm / k_norm | strided 1×256 | `rms_norm_strided_kernel` | ~1 KB |
| 5 | rope | 1×32×256 | rope kernel | 0 |
| 6 | KV append | few indexing/copy ops | small | 0 |
| 7 | sdpa | 1×(32,256)×KV | `sdpa_vector` (D=256, one pass) | KV read ~2×kvlen×1024×2 B |
| 8 | o_proj | 1×4096→4096 | `qmv_wide_kernel` | 8.4 MB |
| 9–13 | norms + MLP | as GDN layer | `rms_norm_kernel` / `qmv_wide_kernel` | 75.6 MB |

Attention-layer GEMM: 105 MB; ~15 launches.

## Tail (per step)

embed gather (1 row, 8 KB) → final `rms_norm_kernel` → **lm_head**
1×4096→248320 (508 MB) on `qmv_wide_kernel` (MR2 is M≥2-only, M=1 per-row
wide) → argmax → D2H scalar (forces step sync).

## Totals and bandwidth arithmetic

- GEMM weight traffic/token ≈ 24×103.8 + 8×105 + 508 = **3.84 GB**
  (+ scales/biases ≈ 6% ≈ 4.0 GB; + GDN state 96 MB r/w; + KV + x ≈ 15 MB).
- At 13.62 ms/token (canonical, 400-window class): **≈ 294 GB/s = 48%** of
  608 GB/s peak.
- llama.cpp anchor 11.5 ms/token: **≈ 348 GB/s = 57%**.
- 87 t/s target = 11.49 ms/token: **≈ 349 GB/s**.

## Why the ~66 µs/layer gap lives in the QMM

1. **ALU per weight byte.** Canonical M=1 wide: 32 scalar FMAs per 16 B of
   weights (dequant-in-register, f32 FMA). llama.cpp `DOTQ4K` uses the int8
   dot-product path (`__builtin_amdgcn_sudot4` / i8dp4): 1/8 the ALU work per
   weight, x staged once in LDS. Same HBM traffic, 8× less ALU — higher
   bandwidth utilization. Our own K1/LX kernel (int8 x-staging + sudot4)
   measured **1.05–1.33×** per-op faster than the wide baseline across
   N = 1024–12288 at M=1 (reports/k1-bench.md, f32 x); the standing
   REFWD_M1 gate exists to A/B exactly this, but is rows-gated off when
   QMV_ROWS=1 (t1391) — so the standing config never engages it for M=1.
2. **Block shape.** Wide: 256-thread block (8 waves) owns 8 columns. The
   rows-msg kernel (M=2..4 under rows-ON) is a 192-thread (6-wave) block
   owning 24 columns, weight-decoded once per k-tile, no LDS/barriers,
   per-row math transcribed verbatim from qmv_wide (row-exact contract).
   For M=1 the kernel is M-natural (MAXM=4, `if (m >= M) break`); only the
   dispatch gate excludes it.
3. **Launch structure (secondary).** ~460 ops/token eager-Python vs
   llama.cpp graph-captured (~100 kernels/step, C++ host). Host build time
   overlaps GPU execution; the per-step argmax D2H sync drains the queue and
   costs tail latency. RMSNorm: 33× 256-thread 2-pass kernels, ~2–5 µs GPU
   each (sandwich-inflated to 66 µs/op in the 10-27 sync profile; ≈ 0.3–0.5
   ms un-sandwiched).

## Fix candidates (all opt-in, default-inert)

- **B1 — `MLX_ROCM_M1_LX=1`:** M=1 QMMs → K1/LX int8-dot4 path (the
  REFWD_M1 re-route, but rows-aware: engages even with QMV_ROWS=1, since
  rows only serves M=2..4). cols default 32; `MLX_QMV_LX=16` selects 16.
  Expected 1.05–1.33× GEMM speedup (k1-bench.md). **Not bit-identical**:
  int8 x-staging → ~1% maxrel (k1-bench observed 5.5e-3–1.05e-2). Gate on
  token-stream identity (ALLTOKS) vs the baseline wheel.
- **B2 — `MLX_ROCM_M1_ROWSMSG=1`:** M=1 QMMs → `qmv_rows_msg_kernel`
  (192-thread / 24-col morphology). **Bit-identical** (row-exact contract:
  row r == per-row wide for every r). Pure scheduling/geometry A/B.
- **B3 — `MLX_QMV_WCOLS` (wide cols 4/8/16):** wide block-height tuning,
  bit-identical, secondary.

Predicted landing: GEMM ×1.2 → ~85 t/s; ×1.3 → ~93 t/s (13.62 × (1 − 0.77·(1−1/s))).
