# WORKSTATE — t1045 (ent4 final: port t1044 fixes + per-tensor E<3.9 gating)

Branch `phase3-entropy-compress`, base 7f9d1bd (t1042 final). Succeeds t1042
(compressor + lossless gates + A/B, REPORT-entropy.md) and t1044 (3 compressor
fixes on its own v1-format script; net-zero finding — see below).

## Timeline
- 04:00Z: worktree up; t1044's 3 commits (764fc08 timing/RSS/traceback, 4f4a9e1
  int32 cumsum, 8ec0580 stream-end OOB) confirmed; manual port required (t1044
  targets a different v1 format).
- 04:40Z: E discrepancy resolved — t1044's 3.9707 "dataset" was measured on
  /models/qwen3-9b-mlx-clean/model.safetensors (reproduced its per-tensor
  values bit-for-bit by running its exact script on that file). t1042's 3.713
  was measured on the final served model /models/qwen3-9b-mlx (independent
  re-measurement: aggregate E_huff 3.7368; per-class el 3.747 / attn 3.728 /
  mlp 3.736; per-tensor max 3.754; 16 distinct symbols everywhere).
- 05:40Z: step 2 done — per-tensor E<3.9 gating (default --gate 3.9; --gate 0 =
  compress-all legacy). Gated .ent4 on the final 9B model (the build VM):
  250/250 tensors compressed (per-tensor E ≤ 3.754, all < 3.9; file size
  4,192,642,800 B = t1042's 4,192,642,550 B + 250 flag bytes, streams
  otherwise identical) — the gate is a no-op on this dataset. Per-class
  re-measure (compressor's own path): el 3.7472 / attn 3.7331 / mlp 3.7392
  bits/sym; TOTAL raw 4.4764 GB → comp 4.1849 GB (x1.0696, = t1042).
- 06:20Z: step 3 done — lossless gates on the build VM: rebuilt wheel (mlx-m8r2
  tree: updated ent4.h/ent4_json.h + load.cpp +3 alloc/memcpy, pip wheel OK),
  CPU verifier 250/250 bad=0 on the gated .ent4 (flag byte + +3 pad), GPU
  kernel gate 3/3 bad=0 (ktest updated: flag byte + 3-byte pad; tensors 0/
  1/249), full-stack A/B (fresh wheel, gated file): baseline vs ent4-gated
  logs byte-identical (forward logits + 32-token generation).
- 07:30Z: step 4 done — A/B bench (flock): 64-token arm baseline
  66.20/65.91/65.83 (median 65.91) vs ent4-GATED 65.85/65.90/65.84 (median
  65.85) tok/s → ~0.1 % overhead, texts identical. 29k-ctx arm (A/B/B2,
  both modes, all rc=0): A 72.29/73.16, B (warmed) 61.86/61.60 decode win
  (parity, LOSSLESS-GATE=PASS both), B2 fresh 60.85/57.25 (first-touch
  decode inside the measured window — methodology artifact, warmed arm is
  the steady-state number).
- 08:10Z: step 5 — REPORT-entropy-final.md + final push.

## Key findings
- The two "datasets" never disagreed about the code: they are two different
  model builds — see timeline 04:40Z. Decision C's E<3.9 gate is a no-op on
  the final model (all 250 tensors E ≤ 3.754, all compressed; the 6.5 %
  memory saving from t1042 stands) but is the safety mechanism for models
  where a class runs above 3.9 (e.g. the -clean build's attn 4.21 / mlp 4.08
  would go plain).

## Gate
- hermetic: ruff format/check (no new findings vs 7f9d1bd), pytest -m "not gpu"
- GPU: green (step 3 — CPU verifier 250/250, kernel gate 3/3, full-stack A/B
  byte-identical; the build VM, flock)
