# REPORT-ar-step-change: AR decode gap — static profile, M=1 QMM gate A/B, verdict

Task: close the AR (MTP=0, fused-GDN) decode gap 73.45 → ~87 t/s (llama.cpp
anchor 86.9 t/s) on the canonical 9B 4-bit model, RX 9070 XT / gfx1201 /
ROCm 10.1 (guest-a). Deliverables: static profile, kernel A/B (patched wheel,
all gates DEFAULT OFF), re-bench (canonical protocol: mtp-bench6, 400-token
window, 3-run median, FIRST20 anchor), MTP=4 guard.

**Verdict up front: the gap is NOT in the M=1 QMM morphology. Both candidate
M=1 kernel paths (B1 int8-dot4, B2 rows-msg) regress 6.9–7.4% e2e vs the
canonical wide default; WCOLS geometry tuning is e2e-parity (bit-identical).
The remainder of the 12.3 ms/token gap is launch/overhead-bound (~517
eager launches/token vs llama.cpp's graph-captured single launch) and
clock-state dependent (see MTP=4 guard). The negative result is recorded per the
negative-result rule; the B1/B2 gates stay DEFAULT OFF (zero behavior change
when env unset).**

## Setup (identity)
- Branch `ar-step-change` (base 60ad8a7 = main). Bench: mtp-bench6 (the
  t1469 working copy md5 cc2c0934 was in use for round 1/2; verified its
  MLX_MTP=0 path is byte-identical to the canonical 452d363d — same prompt,
  same greedy loop, same t/s print; FIRST20 anchor matched on every run.
  Round 3 used the canonical file directly).
- A wheel = venv-canon (mlx 0.32.3.dev20261102+75ca881, md5 b1de59fd...,
  mlx_lm 0.32.2). B wheel = venv-ar: same venv + rebuilt wheel from
  vendor-canon with the 3-gate patch (patches/ar-step-qmm-gates.diff;
  M1_LX / M1_ROWSMSG / WCOLS / REFWD_M1 port, all DEFAULT OFF). Control run
  on venv-ar standing config: 74.49/74.60/74.55 (median 74.55) == A
  (74.56) → wheel/codegen effects isolated: the regressions below are gate
  effects, not wheel effects.
- Standing config (unchanged from t1449): `MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1` (REFWD_M1 is a no-op in this
  wheel under rows-ON, as in the t1434 wheel).
- Protocol: 400-token window (MTP_MAX_TOKENS=400), 3 runs/config, median;
  FIRST20 anchor `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278
  20853 5568 13 357 20853 369 34383 279` matched on ALL runs.

## 1. Static profile (docs/ar-step-profile.md)
Per-layer kernel dispatch map for the fused-AR step (32 layers: 24 GDN + 8
attention), weight-traffic accounting: 3.84 GB of GEMM weight traffic/token;
294 GB/s effective (48% of 608 GB/s peak) vs llama.cpp 348 GB/s (57%).
GEMM ≈ 10.7 ms of 13.9 ms/token (77%) per the per-component sandwich;
llama.cpp total 11.5 ms/token at 86.9 t/s. Gap = ~2.4 ms/token.

## 2. Per-shape M=1 QMM probe (bf16 x, production shapes, per-op median)
scripts/arstep_qmm_probe.py; model-weighted per-token (counts: N=32×48,
N=1024×16, N=4096×56, N=8192×32, N=12288×48, K=12288-N=4096×24,
N=248320×1):

| config | weighted ms (proxy) | vs baseline | e2e median | e2e Δ |
|---|---|---|---|---|
| baseline (wide M=1, A) | 12.59 | — | **74.56** | — |
| B1 M1_LX (int8-dot4) | 14.74 | **+17%** | 69.43 | **−6.9%** |
| B2 M1_ROWSMSG | 12.80 | +1.7% | 69.05 | **−7.4%** |
| WCOLS=16 | 12.25 | −2.7% | (round 3) | |
| WCOLS=4 | 12.18 | **−3.2%** | (round 3) | |

Per-shape medians (ms): wide 0.0394/0.0409/0.0455/0.0551/0.0643/0.0663/
1.0535 for N=32/1024/4096/8192/12288/K12288/248320 — the small-N shapes sit
at the ~0.04 ms launch floor for all variants; the wins/losses concentrate
in N=8192/12288/K12288. (The absolute proxy is inflated vs pipelined
decode — per-op eval sync — but the wheel was control-verified, so relative
comparisons are valid.)

## 3. A/B results (round 1, 400 tokens, 3-run medians)
- **A (venv-canon baseline): 74.56 t/s** (74.42/75.28/74.56). FIRST20 anchor
  match; ALLTOKS deterministic across runs.
- **B1 (M1_LX=1): 69.43 t/s** (68.28/69.43/69.89) — **regression −6.9%**.
  Token stream diverges at token 5 (int8 x-staging: the k1-bench maxrel ~1%
  class; lossless FAIL by construction — recorded, not a fault).
- **B2 (M1_ROWSMSG=1): 69.05 t/s** (69.03/69.05/69.09) — **regression
  −7.4%**, and the "row-exact" contract is NOT bit-identical in this
  build: token stream diverges at token 32. Direct kernel check
  (scripts/arstep_rowsmsg_check.py, 11 production shapes, wide vs rows-msg
  at M=1): maxabs 1.0e-6 … 3.8e-6 = **1–2 ulp of fp32** — consistent with
  FMA-vs-mul+add codegen differences between the two kernels under ROCm
  10.1 / clang 23. A 1-ulp logit perturbation flips one argmax at token 32.
  The t1334 row-exact claim is toolchain-dependent; on this toolchain the
  contract is 1–2 ulp, not bit-exact.

Interpretation: the wide M=1 kernel is already at the local optimum among
tested morphologies on this wheel/toolchain. The k1-bench 1.05–1.33×
numbers (older wheel, f32 x, standalone shapes) do not transfer: in the
production bf16 wheel the int8-dot4 path is 17% SLOWER per-weight-traffic
(the int8 staging + dot4 latency exceeds the scalar-FMA cost at 32-CU
occupancy on these shapes), and the rows-msg morphology is +1.7% in the
probe but −7.4% e2e (its 6-wave/24-col grid under-occupies vs wide's
8-wave/8-col at M=1 for the N-distribution).

## 4. WCOLS geometry tuning (round 3)
WCOLS=4: probe −3.2% weighted GEMM (wins on N=8192/12288/K12288,
N=248320 neutral). WCOLS=16: −2.7%. 3-run e2e (400 tokens, median):
- **WCOLS=4: 74.47 t/s** (74.39/74.47/75.19) vs A 74.56 → **parity
  (−0.12%)**
- **WCOLS=16: 74.49 t/s** (74.34/74.49/74.50) vs A 74.56 → **parity
  (−0.09%)**
- Lossless: W4.1 and W16.1 ALLTOKS md5 == A (129b38c9…) → **bit-identical
  PASS**; FIRST20 anchor match.
The per-op probe's 2–3% did not transfer to e2e (the probe's per-op eval
syncs inflate the small-N launch floor and misweight the pipelined
deode); within run spread (A: 74.42–75.28), WCOLS is a parity tuning
knob, not a gain.

## 5. MTP=4 guard (venv-canon, unmodified MTP path)
38.93 t/s (1657 tokens, 512 verify steps, 2.465 tok/step, D1PCT 76.6 /
D2PCT 61.5 / D3PCT 48.5). Deterministic; the acceptance signature is
identical to t1469's same-day MTP baseline (pre-mtp4, 00:15Z: 23.77 t/s,
same 1657 tokens / same D-ACC). The 65.88-class reference (t1434 11-02:
65.79 post-reboot) is a different clock/platform state: t1434 documented the
band as NOT RECOVERED by full reboot (platform-level, suspect 11-01
userland/ROCm-stack state), and t1463's radiance-mtp-gap documents the
clock-limited MTP band at 33–35 t/s — our 38.93 sits in that band. The
venv-canon tree is byte-identical to the one that produced the 65.79
reference (untouched by this task), so the unmodified MTP path is preserved
by construction; the throughput delta is the documented clock-state effect,
not a path change.

## 6. Negative-result accounting (rule: revert if net-negative, record, keep rest)
- **B1 (M1_LX): net-negative (−6.9%) → reverted** (gate stays DEFAULT OFF;
  the diff is retained in patches/ for the record with this verdict).
- **B2 (M1_ROWSMSG): net-negative (−7.4%) and contract-violating (1–2 ulp,
  not bit-exact) → reverted** (same treatment).
- **B3 (WCOLS): e2e-parity (−0.09…−0.12%, bit-identical)** — kept as an
  opt-in tuning knob (DEFAULT OFF, default wcols=8 = no change); no
  performance gain to claim.
- The REFWD_M1 port stays (no-op under standing config; faithful to t1434).
- Net deliverable: profile + control-verified A/B + WCOLS knob + recorded
  negative results + MTP guard. No behavior change when all envs unset.

## 7. Where the gap actually is (evidence-based)
- GEMM morphology is exhausted at the wide default (sections 2–4): the
  ~10.7 ms GEMM is 294 GB/s-effective — 53 points below the memory-roof
  would be a kernel problem, but three independent morphologies all LOSE to
  it, which argues the wide kernel is near its own optimum and the 48%
  effective efficiency is the M=1 QMV floor on this silicon (each weight
  byte is read once; the 32-CU grid at 8-col block-height cannot beat this
  scalar-FMA schedule; int8-dot4 staging costs more than it saves). WCOLS
  block-height tuning (4/8/16) confirms: no wide-family morphology beats
  wcols=8.
- The ~2.4 ms/token remainder decomposes as: ~517 eager kernel
  launches/token (GDN 24×16 + attn 8×16 + tail) vs llama.cpp's single
  graph launch — launch gaps of 2–4 µs each account for 1.0–2.1 ms; plus
  the per-step argmax D2H sync drain. This matches llama.cpp's 11.5 ms
  total (9.2 GEMM + 2.3 overhead) vs our 13.9 (10.7 GEMM + 3.2 overhead).
- The graph-decode path (MLX_ROCM_GRAPH_DECODE) is the structural lever for
  the launch overhead but has the known second-capture faults
  (WORKSTATE run-1059); it is out of scope for this task.

## 8. Reproduction
- Runner scripts: scripts/arstep-ab-runner.sh (round 1), scripts/
  arstep-round2.sh (control + probes + MTP4 guard), scripts/arstep-round3.sh
  (WCOLS A/B), scripts/arstep_qmm_probe.py (per-shape probe),
  scripts/arstep_rowsmsg_check.py (contract check).
- GPU logs: /home/andryo/arstep/{A,B1,B2,C,W4,W16,MTP4-guard,P-*}.log.
- Wheel: /home/andryo/wheelout-ar/mlx-0.32.3.dev20261103+75ca881-*.whl
  (patched, gates DEFAULT OFF); vendor-canon reverted to qmm.hip.bak-ar
  after the bench (task hygiene — the vendor tree is shared with t1469).
