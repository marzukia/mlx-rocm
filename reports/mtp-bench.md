# MTP spec-decode benchmark (K=1) — Qwen3.5-9B mlx-rocm, neon gfx1201

Date: 2026-10-27 (guest UTC). Salvage + re-bench by piboy (t608 hit the wall at 20:02Z in the
final phase; its bench script passed the mode as a positional arg while the probe reads env
MLX_MTP — all four of its final runs silently ran mode 0; re-run here with the correct interface).

## Protocol
- Model /mnt/models/qwen3-9b-mlx (4bit, model-mtp.safetensors = fixed norms file), venv-g1,
  LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib, greedy, MAX_TOKENS=1024, fresh process per run,
  flock gpu-bench.lock. Prompt: "Explain in great detail how a modern smartphone works..."
  (1024 tokens generated; the fruits prompt EOSes at 149 and undersells MTP).
- Modes: 0 = plain AR. 1 = K=1 spec decode, verify = two 1-token (T=1) target forwards
  (existing probe structure). 2 = K=1 spec decode, verify = ONE batched 2-token (S=2) forward.

## Results (3 runs per MTP mode; deterministic — runs identical)

| mode | verify structure | tok/s (median) | acceptance | notes |
|------|------------------|----------------|------------|-------|
| 0 | AR (1 T=1 fwd/token) | **64.94** (64.94/65.37/65.26 across prompts+days) | - | baseline = op's ~60 / repo 55.9-57.8-63.8 class |
| 1 | 2x T=1 fwd per 2-token step | **54.32** | 389/512 = 76% | 16% SLOWER than AR |
| 2 | 1x S=2 fwd per 2-token step | **15.66** | 378/512 = 74% | 4.1x SLOWER than AR |

Correctness: STREAMS-MATCH — mode 2 stream is byte-identical to mode 1 (batched math correct,
incl. reject-path GDN restore + KV trim + re-forward).

## Findings

1. **Batched verify is not fast on this port.** One S=2 forward costs ~128 ms vs ~15.4 ms/token
   for T=1 (4.6 token-equivalents). The T=2 GDN forward is correct (bit-exact vs 2x T=1 after
   the wheel fix the t608 worker validated: state/y maxdiff 0.0) but runs the chunk-scan /
   general path — there is NO fast small-T (T=2) recurrent decode path in the K3 GDN kernels.
   The fast path is T=1-only; anything wider pays chunk-scan prefill-shaped overhead.
2. **At 76% acceptance the K=1 verify structure itself is the bottleneck, not acceptance.**
   m1 does 2 target forwards per 1.76 tokens (~1.14 fwd/token) + the 1-layer draft forward —
   more target compute than AR's 1.0 fwd/token. That is why m1 loses 16% despite 389 accepted.
3. **Acceptance is healthy**: 74-76% on both streams (fruits: 59/75 = 79% on the gate). The
   draft (fixed-norms file) is good; the speedup is locked behind kernel work.

## Implications / next action

- Do NOT enable MTP in the serve yet: as implemented it is net-negative (54.3 best vs 64.9 AR).
- The win case: a fast T=2 GDN recurrent kernel (cost ~2x T=1) makes mode 2 ~0.85 fwd/token
  -> projected ~75-80 tok/s (+15-20%). That is a K3 kernel task (small-T decode path), not a
  probe task. Until it exists, K=1 on this GDN hybrid cannot beat AR (verify inherently costs
  2 tokens of compute; at best it is ~break-even + draft overhead).
- Fallback for the 100-tok/s goal without the kernel: lever-2 tail fusion (memory: ->100-105)
  stays the higher-EV path; MTP becomes additive once T=2 GDN is fast.
- Kernel diagnostic for the follow-up: per-layer timing of one S=2 vs 2x T=1 forward to confirm
  the cost is the GDN layers (24x) vs the 8 FA layers / MLP / conv.

## Assets
- Guest /tmp: mtp-bench3.py (probe, smartphone prompt, MAX_TOKENS=1024, env MLX_MTP),
  bench4.sh (clean runner), outputs /mnt/models/tmp/bench4/ (full streams).
- Inline AR/MTP reference runs (fruits 149-tok + smartphone 1024-tok) in piboy channel 2026-10-27.
