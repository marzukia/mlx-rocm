# REPORT-wmma-m4-v2: multi-wave WMMA M=4 QMV — result: LOSES, not shipped

**Verdict: do not ship.** The v2 (2/4-wave blocks) and v3 (barrier-free) WMMA
M=4 QMV kernels are correct (unit 24/24 PASS) but lose to the scalar `rows`
kernel on every model-relevant benchmark, monotonically worse as waves per
block grow. Per task: report the mechanism, ship nothing.

## What was built

On top of the v1 kernel (REPORT-wmma-m4), two changes, each gated off by
default (`MLX_QMV_WMMA=1`, M=4 only, `MLX_QMV_WMMA_WAVES` 1/2/4):

- **v2**: template `<T, ST, GROUP_SIZE, WAVES>` — WAVES waves grouped in ONE
  block tiling 16*WAVES consecutive n-columns (wave `g` owns n-tile
  `blockIdx.x*16*WAVES + g*16`), LDS indexed per wave, so the workgroup
  interleave matches the rows kernel's multi-wave blocks.
- **v3**: removed the two `__syncthreads` per 32-k-chunk (the A/B LDS is
  per-wave and a wave is a SIMD lockstep unit, so the barriers order nothing
  between threads) and replaced the Cs LDS store round-trip with a direct
  accumulator->global store (the f32 acc fragment layout puts all M<=4 output
  rows on lanes 0..15, elements 0..M-1: independent per-thread stores).
  Per-tile math is bit-identical to v1 (same unit maxdiff).

## Gates

**Unit (24 combos, vs rows reference, maxdiff bar 1e-2)**: PASS for 2-wave and
4-wave — worst maxdiff 0.007812, identical to v1.

**fwd4 (M=4 forward through the loaded model, median of 10)**:

| arm | v1 (32-th blocks) | v2 | v3 (barrier-free) |
|---|---|---|---|
| rows | 31.60 | 31.60 | 31.54 |
| wmma 1-wave | 43.43 | 44.73 | 46.29 |
| wmma 2-wave | — | 47.41 | 48.46 |
| wmma 4-wave | — | 54.78 | 52.11 |

Penalty scales monotonically with waves per block in every build.

**Canonical MTP bench** (mtp-bench-k3-tape, tape engaged: n_ok=128 n_fb=0,
TAPE_CALLS~5496; same wheel/env for all arms):

| arm | verify ms | pair wall ms |
|---|---|---|
| B0 (stock) | 34.81 | 54.47 |
| ROWSONLY | 30.94 | 50.53 |
| R1v2 1-wave | 50.89 | 81.72 |
| R1v2 2-wave | 47.82 | 67.61 |
| R1v2 4-wave | 51.77 | 71.46 |

R1v2 loses +55..65% on verify and +34..62% on pair vs ROWSONLY at every wave
count. (B0 baseline note: this session's B0 pair is 54.47, not the 69.48 gate
record — the wheel base differs from the gate session; the within-run A/B is
the decision basis. ROWSONLY itself also misses the 22.25/41.5 targets on this
base, consistent with that mismatch.)

## Mechanism (controlled micro; rocprof unavailable on this userland)

`rocprofv2` refuses navi4x ("Unsupported hardware. Use rocprofv3"); `rocprof`
v1 aborts (the 7.2 aqlprofile lib is ABI-incompatible with the 10.1 HSA
runtime — `hsa_amd_external_semaphore_handle_close` undefined). No kernel
timeline was capturable; mechanism was instead established by bisection over
controlled streams (same v3 wheel, 10-iter medians):

- **Kernel resources are not the limit.** Compiled-object metadata (probe
  build): 74 VGPR (1-wave) / 78 (2,4-wave), 2/4/8 KB LDS, no spills — full
  8-wave-per-CU occupancy available.
- **v3 barrier removal changed nothing** (1-wave 44.73 -> 46.29, same class):
  the per-tile barriers were not the mechanism.
- **Short-K GEMMs: wmma wins.** K=512 isolated (4/6 shape classes) and
  K=512 chained/parallel streams with GDN-tape + attention co-scheduled:
  wmma 5.48 ms vs rows 6.48 ms (chained 13.21 vs 14.56).
- **Model-shape GEMMs (M=4, K=4096/12288): wmma loses everywhere.**
  - isolated, per call: K=4096 N=12288: rows 142 us vs w1 237 us (1.67x);
    K=4096 N=4096: 77 vs 88; K=12288 N=4096: 155 vs 183.
  - 249-GEMM stream, real model shape mix, independent: rows 19.82 vs
    w1 33.65 vs w2 37.27 ms.
  - same stream chained (dependency-serial, as in the model): rows 30.74
    vs w1 37.80 vs w2 40.55 ms (without GDN-tape; adding 24 tape calls
    does not change the outcome: 33.12 / 38.97 / 41.34).
- **The penalty scales with waves per block in every context** (fwd4:
  +14.75/+16.92/+20.57 ms for 1/2/4-wave; bench pair: +31/+17/+21 ms over
  ROWSONLY; micro: +7.1/+9.8 ms) — consistent with multi-wave workgroup
  co-scheduling + per-block LDS reservation cost under full-CU occupancy,
  compounded by the wmma kernel's per-32-k-chunk scalar dequant + LDS
  staging pipeline being slower than the rows scalar FMA pipeline at
  K=4096 (256 chunks/GEMM) on this GPU.

The per-GEMM steady-state cost at model shapes, dominated by the N=12288
in_proj class (64 calls; isolated +95 us/call x 64 ~ 6.1 ms of the 7.1 ms
micro delta), fully accounts for the stream loss: there is no
"inter-kernel interaction" mechanism beyond the wmma kernel's own
steady-state cost at K=4096 plus block-size scheduling.

## Disposition

- v2/v3 kernel + dispatch kept in `patches/qmv-wmma-m4/` (gated off by
  default; zero behavior change when `MLX_QMV_WMMA` unset), for the record
  and for a future base where the K=4096 steady-state cost may differ.
- No wheel shipped; the guest scratch wheel is a test artifact only.
- Next lever, if WMMA M=4 is still wanted: an M=4-native path (4-row
  fragments / m16n16k16 with only 4 live rows is 75% waste) — e.g. two
  n8 tiles per wave, or a WMMA kernel that skips the 12 dead accumulator
  rows — or accept `rows` for M=4.
