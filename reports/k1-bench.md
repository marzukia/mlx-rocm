# K1 bench — first real speed number for the K1 QMV lane

## Env + settings

- **GPU**: guest-a, dedicated guest `guest-a@<guest-ip>` (via
  `ssh -J operator@<tailnet-ip>`), AMD Radeon RX 9070 XT / gfx1201, ROCm 7.7
  (guest runtime `/opt/rocm/core-10.1`).
- **Wheel**: venv-k1 `mlx 0.32.3.dev20261026+vendorref` — libmlx.so already carries
  submitter's verified fix (worktree head `99cddb5`, branch base); no rebuild.
- **Env for every guest python**: `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`,
  `/models/venv-k1/bin/python`.
- **GPU serialization**: `flock /models/gpu-bench.lock` (short-wait poll,
  re-poll if held >300s). One holder at a time; each config ran alone.
- **Shapes** (decode-relevant, as used by the QMV probes): T=1, batch=1, M=1,
  K=4096, group_size=64, bits=4, f32 x (the affine `quantized_matmul` op runs
  with f32 x / f32 scales), `mx.quantize` weights, `N in {1024, 2048, 4096,
  8192, 12288}`.
- **Paths**: (a) K1 engaged — `MLX_QMV_LX=16` (16 cols/block) or `MLX_QMV_LX=1`
  (32 cols/block); (b) baseline — no `MLX_QMV_LX`, wide/warp path
  (`MLX_QMV_NO_WIDE` unset). Separate process per config (gate env is
  static-initialized at first use).
- **Timing**: per-op latency = host op creation + `mx.eval`, 10 warmup reps,
  150 timed reps per (config, N); median + p95 reported.
- **Accuracy sanity** (not a gate): K1 stages x as int8 (per-group scale) →
  expected ~1% relative error vs the ~3e-6 baseline; observed K1 maxrel
  5.5e-3–1.05e-2, baseline maxrel ~3.5e-7. Consistent with the fix report.

## Results — per-op latency (ms)

### cols=16

| N    | K1 med | K1 p95 | baseline med | baseline p95 | speedup (med) |
|------|--------|--------|--------------|--------------|---------------|
| 1024 | 0.0594 | 0.0797 | 0.0789       | 0.0966       | 1.33x         |
| 2048 | 0.0708 | 0.0932 | 0.0746       | 0.1062       | 1.05x         |
| 4096 | 0.0747 | 0.0897 | 0.0854       | 0.1089       | 1.14x         |
| 8192 | 0.0899 | 0.0993 | 0.1091       | 0.1246       | 1.21x         |
| 12288| 0.1038 | 0.1134 | 0.1324       | 0.1618       | 1.27x         |

### cols=32

| N    | K1 med | K1 p95 | baseline med | baseline p95 | speedup (med) |
|------|--------|--------|--------------|--------------|---------------|
| 1024 | 0.0604 | 0.0715 | 0.0789       | 0.0966       | 1.31x         |
| 2048 | 0.0769 | 0.1171 | 0.0746       | 0.1062       | 0.97x         |
| 4096 | 0.0809 | 0.1032 | 0.0854       | 0.1089       | 1.06x         |
| 8192 | 0.0952 | 0.1217 | 0.1091       | 0.1246       | 1.15x         |
| 12288| 0.1124 | 0.1469 | 0.1324       | 0.1618       | 1.18x         |

## Verdict

At decode batch=1, K1 is a win across the entire tested N range (1024–12288)
with no crossover to find: 1.05-1.33x at every N (cols=16 medians:
1.33/1.05/1.14/1.21/1.27 for N=1024..12288), and the advantage does not
monotonically grow with N — it dips to 1.05x at N=2048, then recovers
(1.14x at N=4096, 1.21x at N=8192, 1.27x at N=12288). The LDS x-staging +
sudot4 path beats the wide/warp baseline everywhere, and the only cost is the
expected ~1% relative error from int8 x-staging (observed 5.5e-3–1.05e-2
maxrel vs ~3.5e-7 baseline), which the
spec says to accept rather than chase. Between the two block shapes,
cols=16 is the better default: it wins or ties cols=32 at every N on median
(and on p95 except N=1024, where cols=32 is marginally better), while
cols=32 actually regresses to 0.97x at N=2048. Recommendation: keep K1 live
for f32 x at all N with cols=16.

## Exact commands

Harness: `scripts/k1_bench.py` (k1_thresh.py pattern; sets `MLX_QMV_LX` before
first op; f32 x; K=4096, gs=64; 10 warmup / 150 timed reps; prints
`CFG=<cfg> N=<N>: med_ms=… p95_ms=… maxrel=…` per N). Runner:
`scripts/k1bench_run.sh` (flock short-wait poll, `timeout -s KILL 900` wrap,
kills the process group once all 5 N lines land — results complete before the
known exit-time HIP teardown hang; never waits beyond the timeout).

```sh
# upload (from worktree)
scp -J operator@<tailnet-ip> scripts/k1_bench.py scripts/k1bench_run.sh \
    guest-a@<guest-ip>:/models/tmp/

# one per config (baseline, k1-16, k1-32), run sequentially
ssh -J operator@<tailnet-ip> guest-a@<guest-ip> \
    'timeout 570 sh /models/tmp/k1bench_run.sh baseline'
ssh -J operator@<tailnet-ip> guest-a@<guest-ip> \
    'timeout 570 sh /models/tmp/k1bench_run.sh k1-16'
ssh -J operator@<tailnet-ip> guest-a@<guest-ip> \
    'timeout 570 sh /models/tmp/k1bench_run.sh k1-32'

# results land in /models/tmp/k1bench_{baseline,k1-16,k1-32}.out
```
