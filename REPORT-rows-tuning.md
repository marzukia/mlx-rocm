# REPORT-rows-tuning — M=4 verify GEMM: 2–3 variant knobs for `qmv_rows_msg` (CPU round)

**Task:** cut the M=4 verify step below **22.25 ms** (the current best,
REPORT-gemms-m4.md: 22.04/22.24 ms, rows ON base) by adding 2–3 opt-in
variant knobs to the rows-msg kernel: `MLX_QMV_ROWS_WS` (4 base / 32
pre-decoded), `MLX_QMV_ROWS_KBLOCK` (1 base / 2 k-tiles per lane step),
`MLX_QMV_ROWS_WARPS` (6 base / 4 waves per block). This round is
**CPU-side** (the GPU is held by t1353): implement, build, symbol-verify,
validate math on the CPU, and stage the GPU battery
(`bench/run-rows-tuning.sh`) for the next round.

**Verdict (CPU round): the three knobs are implemented, built into the
guest wheel (symbols verified), and validated on the CPU (bit-identical
to the base kernel, maxdiff 0.0; base vs scalar reference ≤ 1e-2). The
GPU speed data is the next round (the GPU was held by t1353 this
round). First-principles analysis: only `KBLOCK=2` has a physical path
to a win (load-latency bound); `WS=32` and `WARPS=4` are expected
neutral — the battery is what tells us which wall we are on.

---

## 1. Context (from REPORT-gemms-m4.md)

The M=1→M=4 GEMM excess is **scalar-FMA-throughput / load-latency bound,
not a dispatch gap and not weight-decode-bound** — proven by the
t1334 decode-hoist negative experiment (decode ALU −37% → verify
**slower** by 0.4 ms, because it added 4× x-tile loads and a
32-element decode array). The rows-msg kernel already does
weight-reuse (one weight stream pass regardless of M) and is row-exact;
the M=4 FMA compute is 4× the M=1 FMA, and the ALU/FMA pipe + the
L2 x-load stream are the wall. The real lever (a WMMA M≤4 kernel,
REPORT-gemms-m4.md section 4) is out of scope this round; this round
tests the three cheap scalar-kernel variants.

## 2. The three knobs (design)

All three are **opt-in, default OFF** (all envs unset → the shipped base
path exactly: warps 6, WS 4, KBLOCK 1 — zero behavior change), and all
three are **bit-identical to the base kernel by construction** (the
unit gate, section 3, pins the argument at maxdiff 0.0; the GPU round
re-verifies with the 1e-2 gate + the token-stream md5 gate). Dispatch:
WS wins over KBLOCK if both are set; WARPS applies to every variant
(including the t1334 hoist). The base kernel gained one trailing
runtime arg `int warps = 6` (`col0 = blockIdx.x * (warps * BN) + wid *
BN`, `block = 32 * warps`, `grid = ceil(N / (4 * warps))`) — with
`warps = 6` the base path is unchanged.

**R1 — `MLX_QMV_ROWS_WS=32` (pre-decoded weight tiles).**
`qmv_rows_msg_ws32_kernel`: per (lane, k-tile) the **4 columns'** 32
4-bit weight values are decoded ONCE into a `wdec[BN][WPER]` array
(128 floats at 4-bit) before the per-row MAC loop, instead of decoded
per (row, column) inline (the base decodes 16 (m,c) pairs × 32 values).
The per-tile decoded values, the FMA order (q-major, j-minor `la`), the
`lx` chain and the acc update order are all identical to the base
kernel. Per-(lane, k-tile) decode ALU at M=4: **1536 → 384 (−75% of the
decode ALU; 512 → 128 decoded values)**; FMA count identical (512); the
per-row x-tile is shared across columns exactly as in the base (no
added loads — the t1334 hoist's failure mode). Register cost: `wdec`
= 128 VGPRs (6 waves × 128 = 768 VGPRs per block — within the RDNA4
budget; a spill would show up as a regression, which the battery
measures).

**R2 — `MLX_QMV_ROWS_KBLOCK=2` (two k-tiles per lane step).**
`qmv_rows_msg_kb2_kernel`: each lane step processes the lane's tiles
**2i and 2i+1 of the base k sequence** (k = lane*WPER + (2i+t)*32*WPER,
t = 0 then 1) in one unrolled step, so both tiles' 128-bit weight loads
(and the 4 rows' x loads) are issued before the FMA of the first tile:
2× the independent load chains in flight to hide weight/x load latency
behind the FMA pipe. The k sequence and the per-output accumulation
order are IDENTICAL to the base kernel → bit-identical. (An earlier
draft used a `t*WPER` stride — adjacent tiles — which is a *different*
per-lane k sequence and therefore NOT bit-identical; the CPU unit gate
caught it and the kernel was corrected to the base-sequence form before
any GPU run. The corrected form is what is in the wheel.)

**R3 — `MLX_QMV_ROWS_WARPS=4` (4 waves per block).** Packing only:
`block = 32 * warps`, `grid = ceil(N / (4 * warps))`, `col0` scaled by
`warps` — the total wave count is unchanged (N/4 either way: 6144
waves for N=24576), so the per-(wave, lane, col) math is identical and
the output is packing-invariant. It changes block-level scheduling
granularity (128-thread blocks instead of 192) — a probe of whether the
6-wave block shape is optimal.

## 3. CPU validation (this round, no GPU)

**(a) Unit gate — `tests/test_qmv_rows_tuning_math.py` (4 tests, all
PASS, 9.7 s, zero-dep pure-Python float32 mirror of the kernel op
order).** The mirror replicates each kernel's float32 arithmetic
(struct-packed f32 op per kernel op, in kernel order: the per-lane k-tile
sequences, the per-row `lx` chain, the q-major/j-minor `la` FMA chain,
the `s*la + b*lx` acc order, the 5-stage `shfl_down` butterfly) for the
model shape (M=4, N=24, K=512, 4-bit, group 64, has_bias — the tape
verify QMM shape) plus the band edges (M=2/3, N=25, 8-bit, group 32/128,
K=1024/4096):
- **`test_kb2_tile_sequence_is_the_base_sequence`: PASS** — the kb2 k
  sequence equals the base k sequence per lane, in order (the bit-
  exactness argument; this is the test that caught the `t*WPER` draft
  bug).
- **`test_variants_bit_identical_to_base`: PASS** — ws32 / kb2 /
  warps=4 packing vs base: **maxdiff 0.0** on all 7 cases (the knob
  contract; gate maxdiff ≤ 1e-2).
- **`test_base_within_1e2_of_scalar_reference`: PASS** — base vs the
  scalar reference (per-output fp32 sequential dequant-MAC, one f32 op
  at a time): maxdiff ≤ 1e-2 (accumulation-order noise only).
- **`test_source_shape_decodes_and_warps_arg`: PASS** — pins the
  source shape the FMA analysis rests on: the base decode chain inside
  the per-(m,c) MAC loop (512 decoded values per (lane, k-tile) at
  M=4), the ws32 pre-decode into `wdec[c][...]` (128), the kb2
  two-tile guard, and all four launch forms passing `rows_warps` with
  the knob defaults (WS 4 / KBLOCK 1 / WARPS 6 → the base path).

**(b) Guest build + symbol verification (PASS).** The wheel
(`mlx-0.32.3.dev20261102+75ca881-cp314`, 22,516,552 bytes, sha256
`86b7598f1499cbf0…`, built from the guest `vendor-rs` @ `75ca881` +
`patches/qmv-rows-tuning.diff` — the dispatch hunk + the two new kernel
bodies + the `warps` runtime arg on the base/hoist kernels) **compiles
clean** and `nm` on `mlx/lib/libmlx.so` shows all four kernel families
instantiated at 36 template combos each (3 `T` × 2 `ScaleT` × 3
`GROUP_SIZE` × 2 `BITS`): `qmv_rows_msg_kernel`,
`qmv_rows_msg_hoist_kernel`, `qmv_rows_msg_ws32_kernel`,
`qmv_rows_msg_kb2_kernel` — and the base kernel's signature ends in the
trailing `int` (the `warps` runtime arg; `… bool, int)`). The first
build attempt (the new symbols absent) was a stale-tree artifact (the
patch had only been `git apply --check`'d into the build tree after the
kb2 correction); the patched tree was re-verified by md5 before the
successful build. (The B0-gate of the GPU battery re-confirms the
default path is unchanged.)

## 4. First-principles FMA analysis — expected ms per variant

**The per-(lane, k-tile) op budget (4-bit, M=4, BN=4, K=4096 → 4
k-tiles per lane):**

| | FMA | decode ALU | decoded values | extra loads |
|---|---|---|---|---|
| base | 512 | 1536 | 512 | x 4×(2×16B + 32 cvt) |
| ws32 (R1) | 512 (identical) | 384 (−75%) | 128 | none (x shared as base) |
| kb2 (R2) | 512 (identical) | 1536 (identical) | 512 | none (2× in-flight) |
| warps4 (R3) | 512 (identical) | 1536 (identical) | 512 | packing only |

The t1334 experiment fixed the wall: cutting the decode ALU 37% (and
*adding* loads) made verify **slower** by ~1.5–2% — the kernel is not
decode/ALU-bound; it is bound by **scalar-FMA throughput** (4× the M=1
FMA count is the excess) and/or **memory-load latency**. Against that:

- **R1 ws32: expected neutral (22.2 → 22.3 ms, ±0.3).** The FMA count is
  the wall and it is unchanged; the −1152 decode ALU per (lane, k-tile)
  only helps if the ALU and FMA pipes were both saturated (RDNA4 has
  distinct ALU and FMA units, so there is some overlap room — at best a
  ~1% gain). Risk: the 128-float `wdec` array → register pressure → a
  spill would show as a regression. If R1 is neutral-to-negative while
  its ALU is cut, that is further confirmation the FMA pipe is the wall.
- **R2 kb2: expected −0.3 to −0.8 ms IF load-latency bound (21.6–21.9);
  neutral if pure FMA-throughput bound.** This is the only knob with a
  physical path to beating 22.25: 2× the weight/x loads in flight per
  lane step hides load latency behind the FMA pipe without changing any
  FMA. If R2 is neutral, the FMA-throughput wall is confirmed and the
  WMMA M≤4 kernel (REPORT-gemms-m4.md section 4, ~4–8× the FMA
  throughput of the matrix units) is the required next step — which the
  knobs infrastructure (dispatch + bit-exact contract + this battery)
  already stages.
- **R3 warps4: expected neutral (±0.2 ms).** Same total wave count
  (N/4); it probes block-scheduling granularity only.

**Expected battery (base 22.25 ms):** R0 22.0–22.3 (repro), R1 22.0–22.3,
R2 21.6–22.0, R3 22.0–22.3. The honest prior: none of the three is
expected to beat 22.25 by a wide margin; **R2 is the best candidate and
the battery's real output is the wall identification** (R2 wins →
load-latency bound; all neutral → FMA-throughput bound → WMMA is the
decisive lever).

## 5. The GPU battery (next round) — `bench/run-rows-tuning.sh`

`bash bench/run-rows-tuning.sh <venv-py> [model] [tape-bench]` on
guest-a, one fresh process per env, the canonical tape bench
(`mtp-bench-k3-tape.py`, md5 `2c2d6c90c2349ae087e53e5d6c60018e`,
md5-verified at start) with `MLX_ROCM_GDN_TAPE=1`, the model
`/mnt/models/qwen3-9b-mlx` (sidecar `model-mtp.safetensors` md5-pinned
at start and re-checked after the battery), the `gpu-bench.lock` as the
run gate (FLOCK-FAIL if the GPU is held). Arms: **B0** (rows OFF —
regression gate: must reproduce the m4 B0 reference 70.02 tok/s /
25.66 ms / 44.85 ms within ±3% — the tuning code is default-off, so B0
is behaviorally the m4 wheel), **R0** (rows ON, base), **R1** (WS=32),
**R2** (KBLOCK=2), **R3** (WARPS=4), all with `MLX_ROCM_QMV_ROWS=1
MLX_ROCM_QMV_ROWS_MIN_N=1` (the m4 config — every M=4 shape on the
rows path). Gates: (3) **bit-exactness** — R1/R2/R3 `FIRST20` + `ALLTOKS`
md5 == R0 (any divergence = FAIL, the arm is not a speed data point —
the bit-identical contract from section 3); (4) `n_refwd` in 95..107.
Target: a rows arm (R0–R3) with **verify ms < 22.25**.

## 6. What's delivered / preserved

- The two variant kernels + the `warps` runtime arg
  (`vendor/mlx-rocm-upstream/.../qmv_rows_msg_kernel.hip`, +2 variants,
  +trailing `warps` arg on base/hoist) and the guest dispatch
  (`patches/qmv-rows-tuning.diff` — qmm.hip hunk: the three env knobs,
  WS > KBLOCK > hoist > base selection, all four launch forms passing
  `rows_warps`; kernel hunk: the two bodies + the base/hoist arg).
  Default-off (all envs unset → the shipped base path exactly).
- The CPU unit gate (`tests/test_qmv_rows_tuning_math.py`, 4 PASS) and
  the source-shape pin for the FMA analysis.
- The guest wheel (symbols verified) + the GPU battery
  (`bench/run-rows-tuning.sh`, B0 gate + 4 arms + bit-exactness /
  n_refwd gates + target check).
- The t1334 hoist negative experiment remains in the tree, documented,
  default-off (REPORT-gemms-m4.md appendices).

**Next:** the GPU round (run `bench/run-rows-tuning.sh` when the GPU is
free) → the wall-identification verdict (section 4) → the WMMA M≤4
kernel (REPORT-gemms-m4.md section 4) if R2 is neutral.
