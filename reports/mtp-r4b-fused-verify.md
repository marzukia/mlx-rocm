# R4b — fused S=4 verify (multi-row QMV) A/B report

Anchor protocol: `bench/mtp-bench6.py` (md5 25c1790bc323587a9435e61de9cffa70), `bench/mtp-bench6-phase.py` (md5 2bce8190412bf2c00312cec1d6842caf), 3 repeats per arm, 25 s idle before the first run and between runs, all under `flock -x` on the GPU lock (lock was free at start; no kill, only wait). Model SBCAR md5 02a54359cda8b247f76b277dd9bc4d4c pre and post (unchanged). Both arms: `MTP6_TAPE=hybrid GDN_FUSE=1 REFWD_M1=1` (the canonical t1541 env); GPU: RX 9070 XT / gfx1201, 32 CUs.

## Lever and wheels

- OFF: the standing canonical wheel (venv-canon, libmlx md5 b1de59fda0e4fc607d620e287deeaaf8) + canonical env (multi-row QMV **off**).
- ON: the R4b wheel (built on neon from the vendor pin + the one-lever change in `mlx/backend/rocm/quantized/qmm.hip`, `wide_ok` no longer requires `has_bias` for 2 <= M <= 4; wheel `mlx-0.32.3.dev20261103+6f410ed`, libmlx md5 3d19e4f3420b3b067abaeec6ae971dac) + canonical env + `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`.
- CROSS (attribution control): the canonical wheel + canonical env + `MLX_ROCM_QMV_ROWS=1`.

## Probe (bias-free nn.QuantizedLinear, task env vs rows env)

| shape (K x N) | canon rows off M4/M1 | r4b rows off M4/M1 | canon rows on M4/M1 | r4b rows on M4/M1 | rows-on maxdiff vs per-row |
|---|---|---|---|---|---|
| outproj/q/o 4096 x 4096 | 1.71x | 1.69x | 1.78x | 1.77x | 9.2e-05 |
| gate/up 4096 x 12288 | 2.30x | 2.37x | 2.24x | 2.20x | 9.2e-05 |
| down 12288 x 4096 | 2.18x | 2.25x | 2.35x | 2.35x | 1.4e-04 |
| lm_head 4096 x 248320 | 3.88x | 3.87x | 1.45x | 1.46x | 1.2e-04 |

Wheels behave identically in all four probe configurations: with the rows env off, bias-free M=4 is per-row (2.2-3.9x M1); with the rows env on, it takes the shared-weight multi-row path (probe 1.4-2.4x M1, i.e. below the per-position cost). rows-on output differs from per-row by <=1.4e-4 (bf16).

## Bench matrix (median of 3)

| protocol | OFF t/s | ON t/s | CROSS t/s | delta |
|---|---|---|---|---|
| anchor (322 iters, 161 verifies) | 46.07 | 48.93 | 48.92 | **+6.2%** |
| phase 64v (128 iters, 64 verifies) | 43.35 | 50.02 | — | **+15.4%** |
| phase 512v (1024 iters, 512 verifies) | 38.55 | 40.30 | — | **+4.5%** |

Per-run values: anchor OFF 46.44/46.07/46.03 (t1541 OFF arm 46.49 — reproduced within 0.9%); ON 48.34/48.93/48.94; CROSS 48.98/48.78/48.92. FIRST20 is identical across all three repeats of every arm and equal across arms: `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279`.

Acceptance (the measured FP-order tradeoff — the rows kernel output differs from per-row by <=1.4e-4 (bf16) per probe, and the reduced FMA tree shifts the 4-bit rounding decisions, so acceptance moves; reported per task instruction, not as a quality signal):

| protocol | arm | D1PCT | D2PCT | D3PCT | tok/step |
|---|---|---|---|---|---|
| anchor | OFF | 73.3 | 59.3 | 50.0 | 2.385 |
| anchor | ON | 73.3 | 62.7 | 47.3 | 2.410 |
| 64v | OFF | 68.8 | 42.2 | 15.6 | 2.27 |
| 64v | ON | 79.7 | 56.2 | 20.3 | 2.56 |
| 512v | OFF | 76.6 | 47.1 | 22.9 | 2.46 |
| 512v | ON | 72.9 | 45.1 | 25.8 | 2.44 |

At the anchor D3PCT drops 50.0 -> 47.3 (-2.7 pts) while D2PCT rises +3.4 pts and total t/s rises +6.2% (tok/step 2.385 -> 2.410): the measured tradeoff is net positive. At 64v the ON arm accepts strictly more (D1/D2/D3 all up).

## Phase attribution (instrumented copy)

| 64v | prefill | draft | verify | commit | correction |
|---|---|---|---|---|---|
| OFF | 5.3 | 2597.1 | 34.6 | 1942.9 | 26.4 |
| ON | 5.2 | 2515.2 | 35.3 | 1709.2 | 25.5 |

| 512v | prefill | draft | verify | commit | correction |
|---|---|---|---|---|---|
| OFF | 5.1 | 15157.0 | 273.4 | 27439.4 | 191.1 |
| ON | 5.3 | 14789.8 | 278.1 | 25189.8 | 187.3 |

The lever lands in **commit** (30.2 -> 26.7 ms/step at 64v, -11.6%; 53.5 -> 49.1 ms/step at 512v, -8.2%). The `verify` timer only covers the lazy graph build: in this wheel `mx.eval` is a side effect and the batched S=4 forward is forced by the first `argmax` in the commit section, so the S=4 GEMM work is billed to commit (plus the acceptance change above, which reduces re-forwards: REFWDS 54 -> 51 at 64v, 395 -> 380 at 512v).

## Findings (probe-first, per protocol)

1. **The committed gate fix is empirically inert for `nn.QuantizedLinear` layers.** The python layer always passes the quantize bias array as a fourth op input (`biases=self.get("biases")`, a zero array even when `bias=False`), so the C++ op sees `has_bias=1` (input-count based) on every call. The `wide_ok` `has_bias` gate therefore never blocks these layers; with rows env on, the standing wheel already fuses bias-free M=4 (probe 1.45x). The committed change remains correct and is belt-and-braces for genuinely bias-free op calls (e.g. direct `mx.quantized_matmul` with 3 inputs).
2. **The effective lever is `MLX_ROCM_QMV_ROWS`** (the canonical env name; the task config line's `QMV_ROWS=1` is a misspelling the wheel ignores — this is also why R4a's `QMV_ROWS=1` was inert). CROSS == ON (48.92 vs 48.93) attributes the entire gain to the env gate, not the wheel delta.
3. The rows kernel shares one weight read across all 4 verify rows (llama.cpp-class weight traffic per the probe): probe M4/M1 3.88x -> 1.45x on lm_head, maxdiff vs per-row 1.2e-4 (bf16, not bit-identical).
4. FP-order/health: FIRST20 identical per arm and across arms; per-run t/s spread <=1.2% (anchor), 0.5% (64v), 0.5% (512v); acceptance drift is the expected FP-order effect (section above).

## Verdict

The fused S=4 verify lever (multi-row QMV for bias-free M=4) is measured net positive at the anchor protocol: 46.07 -> 48.93 t/s (+6.2%), largest at short KV (64v: +15.4%), positive at long KV (512v: +4.5%), with a -2.7 pt D3 acceptance tradeoff offset by +3.4 pt D2 and higher tok/step. The standing wheel already contained the functional path; the enabling gate is `MLX_ROCM_QMV_ROWS=1`, and the committed `wide_ok` change guarantees it for truly bias-free calls.
