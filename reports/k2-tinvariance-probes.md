# this round — K=2 k_proj T-invariance: probes, gates, bench (2026-10-30)

Lane: guest-a (<tailnet-ip> / <guest-ip>, RX 9070 XT gfx1201, ROCm 10.1),
`flock /models/gpu-bench.lock` for every GPU run.
Wheel: `the prior 29k wheel/venv-m8`, `libmlx.so` sha256[:12]=`de6de5f154c0` (m8rebase,
prior-wheel-verified; **no rebuild — the patch is python-side, .so untouched**).
Model: `/models/qwen3-9b-mlx` (5.95GB 4bit, 32 layers, 8 full-attention
layers [3,7,11,15,19,23,27,31], 24 GDN layers).
Repo: `worker/k2-kproj-tinvariance` off base `2027d51` (m8 audit #20).

## TL;DR

- **Gate (a) PASS** — the audit's direct k_proj T2-vs-T1 probe: model k_proj
  (4-bit QMM) is **bit-exact T-invariant** (n=0 over 8 FA layers × 20 random
  bf16 trials, row1(T=2) vs row0(T=1)). The m8 in-flow probe now reports
  `NO LEAK` (state exact) at the fork. The 1-ulp the #20 audit attributed to
  k_proj was the **fused rope kernel** (T=1 1D kernel vs T>=2 batched kernel)
  — the `single=false` rope fix + SDPA 1-pass patch are already in this wheel
  (m8 rebase); no .so change was needed this round.
- **Gate (b) PASS** — G4 5-arm: `M2-M0-LOSSLESS MATCH n=1024`, no first_div;
  all 4 control comparisons MATCH n=1024.
- **K=2 byte-identical (greedy) PASS at both prompt lengths**: 14-token
  prompt: K=2 stream (1399 toks) is an **exact prefix** of the 2048-token AR
  run; 29K prompt: K=2 stream (513 toks) is an **exact prefix** of the 512-token
  AR run (512-token prefix byte-identical; the 513th K=2 token is a
  target-verified argmax, i.e. the AR continuation).
- **Bench (warmup-first, steady-state window skip=56)**: K=2 does **not** beat
  AR. 14-token prompt: AR 76.75, K=1 68.91, K=2 59.00 tok/s (window). 29K:
  AR 61.04, K=1 45.77, K=2 37.35 tok/s (window; 40.95 in the un-warmed 512-tok
  run). Honest acceptance below.

## Root cause (why the wheel already fixed it)

The #20 audit's flow probe found a 1-ulp in `k_proj` at the first verify
divergence. The m8 round-2 (the earlier round) correction re-located it: the model's
`k_proj` is a **4-bit QMM** (`qmv_wide_kernel`) — T-invariant per row by
construction (verified here: D1 n=0). MTP's plain-matmul `k_proj` runs only at
T=1 (gemv, deterministic). The actual T-dependence was `mx.fast.rope`: the
T=1 1D rope kernel vs the T>=2 batched rope kernel differ by 1 f32 ulp on
`rx2`. That fix (`single=false`, single-kernel) plus the SDPA 1-pass patch are
both in the m8 rebase wheel this round runs on, which is why the flow probe
now shows `NO LEAK` and all gates pass with the **unmodified .so**.

Residual (non-gating): plain bf16 matmul still shows value-specific 1-ulp
flips between same-shape T=2 and T=1 calls (D2: 5/20 trials, 1 elem, on
k_proj-shaped `x(2,4096)@W(1024,4096).T`; 11/20 trials, 2 elems, on
t-shaped `x(2,1024)@W(4096,1024).T`) — the the earlier round "follow-up candidate". It
cannot affect the greedy lossless path: in the model k_proj is QMM, and in
the MTP draft path the plain matmul is T=1-only (same kernel, same result,
deterministic: D3 n=0).

## Changes (python-side only; no .so rebuild)

- `scripts/k2-kproj-tinvariance-probe.py` — gate (a): D1 model k_proj QMM
  T=2 vs T=1 (8 FA layers × 20 random bf16 trials, expect n=0), D1b q/v_proj
  control, D2 plain-matmul direct P3 test (both orientations, informational),
  D3 determinism re-run.
- `bench/warm-mtp-bench.py` — warmup-first bench harness: mtp-bench5
  MTP flow (mode 0/2/3, draft chain + reject rollbacks verbatim) + the prior probe
  chunked 29K prefill + discarded warm pass + iteration-based window
  (skip=56), 14-token and 29K prompt, acceptance stats, ALLTOKS.
- `scripts/warm-mtp-run.sh` — lane driver (all runs under flock, venv-m8).

## Gate results (lane, 2026-10-30 01:16-01:30Z)

| gate | result |
|---|---|
| (a) k_proj probe T2-row1 == T1 | **PASS** — D1 n=0 (8 layers × 20 trials); D1b q/v n=0; D3 n=0; flow probe `NO LEAK` |
| (b) G4 5-arm M2-M0-LOSSLESS, n>=512, no first_div | **PASS** — MATCH n=1024; controls M1-M0 / M1-M2A / M2-A-B / M2-B-C all MATCH n=1024 |
| (c) bench AR + K=1 + K=2 @ 14-token and 29K (warmup-first) + K=2 acceptance | **DONE** — tables below |
| K=2 byte-identical greedy vs AR | **PASS** — 14-token: prefix 1399/1399 exact; 29K: prefix 512/512 exact |

G4 arm speeds (14-token prompt, 1024 toks): m0 AR 69.72; m1 K=1 (2×T=1) 55.49
(372/512); m2a/b/c K=1 batched 60.85 (376/512, bit-identical across arms).

## Bench (window = iterations 57+, skip=56; warm pass discarded and logged)

14-token prompt (27-token encode), 1024 toks, warmup-first:

| mode | decode tok/s (all) | tok/s (window) | it median | verifies | D1% | D2% | tok/step |
|---|---|---|---|---|---|---|---|
| AR (m0) | 76.10 | **76.75** | 13.0 ms | 0 | — | — | — |
| K=1 (m2) | 68.31 | **68.91** | 18.8 ms | 512 | 72.7 | — | 1.727 |
| K=2 (m3) | 58.19 | **59.00** | 25.3 ms | 373 | 74.8 | 63.1 | 1.748 |

29K prompt (29000 toks, chunked prefill 1024), 256 toks, warmup-first:

| mode | decode tok/s (all) | tok/s (window) | it median | verifies | D1% | D2% | tok/step |
|---|---|---|---|---|---|---|---|
| AR (m0, the prior probe unmodified) | 61.30 | **61.04** | 16.3 ms | — | — | — | — |
| K=1 (m2) | 4.56 | **45.77** | 23.7 ms | 128 | 40.6 | — | 1.406 |
| K=2 (m3) | 4.45 | **37.35** | 32.1 ms | 98 | 60.2 | 25.4 | 1.602 |

K=2 at 29K, un-warmed 512-tok run (window still skips 56 iters): 40.95 tok/s,
D1 68.6%, D2 39.7%, 1.686 tok/step. (The low "all" numbers are the first
iteration's ~50s outlier — KV allocation / first-forward at a fresh 29K
context; the window excludes it.)

## K=2 vs AR — honest verdict

**K=2 < AR at both prompt lengths** (14-token: 59.00 vs 76.75 = 0.77×;
29K: 37.35 vs 61.04 = 0.61×; 40.95 vs 61.06 = 0.67× un-warmed 512-tok).
Acceptance: 14-token D1 74.8% / D2 63.1% (1.748 tok/step); 29K D1 60.2% /
D2 25.4% (1.602; 68.6%/39.7% → 1.686 in the 512-tok run). The verify step
costs 32.1ms median vs AR's 16.3ms (≈1.97× a T=1 step), so even a perfect
2.0 tok/step would only reach parity; 1.748/1.97 ≈ 0.89 theoretical, and
reject re-forwards (2-3 extra T=1 forwards) drag the measured 14-token ratio
to 0.77×. Not chasing per task — next step if wanted: drop the reject
re-forward by trimming KV in-place (the GDN save/restore + 2-3 T=1 re-forwards
are the main overhead), or extend K to 3-4 to raise tok/step above 2.

## Reproduction

```
# on guest-a, under flock /models/gpu-bench.lock:
PY=/models/tmp/the prior probe-29k/venv-m8/bin/python
D=/models/tmp/k2-tinvar   # this round's uploaded scripts + outputs
$PY $D/k2-kproj-tinvariance-probe.py        # gate (a)
bash $D/k2-tinvariance-run.sh                        # full: gates + bench
```

Evidence: `$D/{kproj-probe,flowprobe,gate,k2-14tok-gate,k2-29k-gate}.log|.out`,
`$D/{m0,m1,m2a,m2b,m2c,ar2048,m3-14tok}.out`,
`$D/{bench14-m0,bench14-m2,bench14-m3,ar29k-the prior probe,bench29k-m2,bench29k-m3,
ar29k512,m3-29k512}.out`, `$D/run.log`, `$D/import.log`
(wheel sha `de6de5f154c0edec6aaa6ec5f1f9f970b0a4b955841eea0921200f5b300b0caf`).

worker: <operator>; reviewer pass to follow.
