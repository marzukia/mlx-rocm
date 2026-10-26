# K2: non-GEMM fusion bench (partial - t504 burned the wall at 1207s, F59 #19)

Branch: k2-non-gemm-fusion @ c49538c (salvage commit adds this report + probe scripts).
Guest: gfx1201 (RX 9070 XT), 192.168.122.234. Protocol: 200-tok warm decode, 16-tok warmup,
fresh process, flock /mnt/models/gpu-bench.lock, timeout-300 wrap. Status: PARTIAL - 0.6B arm
and dispatch-count delta NOT RUN (wall burn at 06:56Z, mid A/B attribution).

## What landed in t504

- uint4 vector-copy fix VERIFIED (the prior run found f32 VEC=4 wrote 4 of 8 elements for D=8;
  the D=8 array probe now matches - see scripts/k2-d8-probe.py).
- Unit suite green at the 80% wall mark.

## 9B-4bit A/B (qwen3-9b-mlx, warm 200-tok decode)

| arm | tok/s | notes |
|---|---|---|
| ON (fused wheel /mnt/models/mlx-clone-k2, patched model) | 55.86 / 57.75 | tag k2-9b-on |
| OFF (pristine /mnt/models/venv, unpatched model, same runner) | 51.09 / 55.51 | tag k2-9b-off |
| old baseline (earlier gauntlet run) | 63.2-64.7 | confounded - see reading |

## Reading (t504 session, last analysis before the burn)

The OFF arm is ALSO ~10% below the old baseline (51-55.5 vs 63-64.7). The ~10% regression is
therefore NOT attributable to the K2 fusion - it is an environment/baseline drift (wheel build,
guest state, or the old baseline was measured under different conditions). ON >= OFF, so the
fused slice is performance-neutral to slightly positive within run variance. The K2 +20-30%
speedup hypothesis is NOT confirmed.

## Outstanding (re-filed as the K2 completion pass)

1. 0.6B-8bit A/B (qwen3-06b-mlx; old baseline 183-189 tok/s) - ON + OFF arms.
2. Re-anchor the 9B baseline discussion to the fresh OFF number.
3. Kernel dispatch-count delta via rocprofv3 if budget allows (baseline 1,070 dispatches/token).
4. Adoption verdict: K2 slice = neutral (no regression, no speedup at 9B) unless 0.6B shows otherwise.
