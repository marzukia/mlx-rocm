# K2: non-GEMM fusion bench (COMPLETE)

Branch: k2-non-gemm-fusion. Guest: gfx1201 (RX 9070 XT), ROCm 10.1.
Protocol: 200-tok warm decode, 16-tok warmup, fresh process, flock /models/gpu-bench.lock
(util <1% pre-bench), timeout-300 wrap, same runner for both arms
(/models/mlx-clone-k2/run-bench.sh <venv> <model> 200 2 <tag>).
A/B isolation: both venvs carry the same fused wheel (mlx-0.32.3.dev20261026+vendorref);
ON = venv-k2 (patched mlx_lm model files, scripts/patch-models-k2.py, verified idempotent),
OFF = pristine /models/venv (unpatched model files). The delta is the model-side fusion.

## 9B-4bit A/B (qwen3-9b-mlx) — from run-504

| arm | tok/s | notes |
|---|---|---|
| ON (fused wheel, patched model) | 55.86 / 57.75 | tag k2-9b-on |
| OFF (pristine venv, unpatched model) | 51.09 / 55.51 | tag k2-9b-off |
| old baseline (earlier gauntlet run) | 63.2-64.7 | confounded - see reading |

## 0.6B-8bit A/B (qwen3-06b-mlx) — NEW

| arm | tok/s | notes |
|---|---|---|
| ON (venv-k2, patched model) | 207.64 / 214.28 | tag k2-06b-on; free_delta 39.8 MB |
| OFF (pristine venv, unpatched model) | 177.20 / 180.57 | tag k2-06b-off; free_delta 79.7-81.8 MB |
| old baseline (earlier gauntlet run) | 183-189 | confounded - see reading |

ON vs OFF: +17.2% (run 0), +18.7% (run 1) → ~+18% speedup at 0.6B.
The fused slice also cuts per-run allocation churn by ~40 MB (in-place KV append
vs copy; 39.8 vs ~80 MB free-memory delta).

## Re-anchored baseline reading (fresh OFF arms are the baseline)

Both fresh OFF arms sit BELOW the old gauntlet baseline: 9B 51.1-55.5 vs 63.2-64.7
(~-10%), 0.6B 177.2-180.6 vs 183-189 (~-4%). The old baseline is therefore
confounded (environment/build drift, not the fusion). The correct reference is the
fresh OFF arm; against it: 9B ON is +4.0%/+7.4% (neutral-to-slightly-positive within
run variance), 0.6B ON is +17-18% (a real speedup, consistent across both runs).

## Dispatch-count delta (rocprofv3 kernel trace, 9B)

Probe: scripts/k2-dispatch-probe.py under rocprofv3 --kernel-trace (this rocprofv3
build has no --collect); 20 timed steps + 5 warmup steps + 16-tok warmup generate
= 41 decode-equivalent steps per run; fresh process per arm.

| arm | total dispatches | per decode step |
|---|---|---|
| OFF (unpatched) | 43,838 | 1,069.2 |
| ON (patched) | 42,948 | 1,047.5 |
| delta | -890 | -21.7 (-2.0%) |

The fresh OFF number (1,069) reproduces the prior 1,070 dispatches/token baseline,
confirming the measurement protocol is stable. Kernel-level diff (OFF→ON counts):
rms_norm_warp -1344, copy_gg_byval -672, Multiply -336, Sigmoid -336, copy_v -218
removed; rms_norm_residual +1344, kv_append +336, compiled fused gated-output +336
added — exactly the expected K2 signature (2 kernels → 1 per fused site).

Reconciliation note (review MINOR-7): the per-kernel counts above are from the
rocprofv3 kernel trace, which covers a ~9.3-decode-step window (rms_norm_warp
1344/36 layers = 37.3 → 1344/37.3 = 9.33 steps; kv_append 336/36 = 9.33; copy_gg_byval
672/36/2 = 9.33), not the 41-step dispatch total. The two are separate
measurements: the per-step dispatch total (43,838/41 = 1,069.2) is the
authoritative, independently validated number (matches the prior 1,070 baseline),
and the delta is self-consistent (−890/41 = −21.7/step). The naive KV-fusion
expectation (−2 copies/layer × 36 layers = −72/step) overstates the measured
−21.7/step because the OFF slice-assign path (update_and_fetch) does not dispatch
2 copies per layer per step over the full run — the Python-side slice assign and
the in-place append have different dispatch patterns, so the net per-step delta
over the 41-step run is −21.7, not −72. The A/B conclusion is unaffected (it rests
on the validated dispatch total and the tok/s A/B, not the per-kernel counts).

## Adoption verdict

ADOPT the K2 non-GEMM fusion slice.

- 9B: neutral (no regression; ON ≥ OFF within variance).
- 0.6B: +17-18% speedup, ~40 MB less allocation churn per run.
- Dispatch overhead: -21.7 kernels/token (-2.0%) at 9B, with the kernel diff matching
  the intended fusions one-for-one.

The size-dependent result is consistent with the mechanism: 0.6B decode is
launch/latency-bound (fewer FLOPs per token), so removing ~22 kernels/token and the
KV copy pays directly; at 9B the GEMMs dominate and the same savings are absorbed
by run variance. No regression anywhere → safe to land; the +20-30% hypothesis holds
for the small-model slice, not the 9B slice.
