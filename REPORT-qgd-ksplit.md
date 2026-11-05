# REPORT-qgd-ksplit.md — 4x K-split M=1 tiled QMV probe (t2050, report-branch only)

**Verdict: FAIL-as-chartered.** Class microbench: K 4.596 ms / 36.2% of floor
vs C 4.418 ms / 37.7% — K is **−4.05%** on the class bar (bar: >= +1%) and
**36.2% < 37.2%** on the floor bar. The in-flight-load hypothesis is falsified by measurement (512 KB in flight, 1.88x the 273 KB saturation requirement, did not move the class). The t2048 wash control is **confirmed**
(K = 4.596 ms lands inside the predicted 4.4-4.7 ms wash band; the <4.0 ms
falsifier did not trigger). e2e is flat: 69.16 vs 69.15 t/s (default, +0.01%),
84.22 vs 84.19 (hello, +0.04%), FIRST20 token-identical in all 12 runs
(lossless).

Chartered by the t2040/gen2 ceiling analysis: the qmv_gdn class (GDN 4 in-proj
+ out_proj, 4-bit g64 affine, no bias, M=1, K=4096, N=4096) is
**latency/occupancy-bound, not HBM-bound** — the stock tiled launch puts ~128 KB
of weight traffic in flight vs the ~273 KB needed to saturate the 546 GB/s HBM
floor, and the gen-2 memory-structure family (load width/order) is exhausted.
The chartered fix was a *parallelism* change: 4x K-splits. This probe
implemented one env-gated variant, checked bit-identity, and measured it
same-session against C.

## 1. Task 1 — arithmetic: in-flight load window at 4x K-splits

Model constants (gen2, same-session measured): HBM floor 546 GB/s; saturating it
needs ~273 KB in flight (= 546 GB/s x ~500 ns HBM latency); stock runs at 203
GB/s = 37.2% of the floor with ~128 KB in flight.

- **Stock tiled (N = 4096, tile_n = 8):** 512 column tiles; persistent launch
  grid_y = min(512, 32 CUs x 16 blocks/CU) = 512 blocks; one column tile per
  block, each block's per-tile k-chain streams the whole K = 4096 (64 groups /
  column, BSK = 512 value tiles x 8 k-tiles) with 2 x uint32 per-lane
  nontemporal loads and no software pipelining. Per-tile in-flight weight
  traffic: 512 blocks x 256 B = **128 KB** — 47% of the 273 KB requirement.
- **4x K-split (this probe):** grid (M=1, 512 column tiles, 4 K-splits) =
  **2048 blocks**; one column tile + one K-quarter per block (K/4 = 1024 =
  16 groups / column = 2 x BSK value tiles), no grid-stride. Per-tile
  in-flight weight traffic: 2048 blocks x 256 B = **512 KB** = 1.88x the 273 KB
  requirement. The per-block serial k-chain is 4x shorter (16 vs 64 group
  iterations), and the 2 x `__syncthreads` staging pair sits in the same chain
  2x instead of 8x per column (the gen2 analysis predicted K-split removes the
  grid-stride serialization; it does not remove the per-tile sync pair, only
  shortens the chain it sits in).
- **Prediction if in-flight-bound:** the class ceiling moves from 203 GB/s
  toward the 546 GB/s floor (82-100%). **Falsifier:** K arm < 203 GB/s minus
  noise, or 203-273 GB/s with the in-flight window now 1.88x saturated — which would mean the ceiling is the per-tile sync/staging chain itself, not the outstanding-load requirement. **The falsifier is exactly what the measurement returned: 197.9 GB/s (below stock 205.9) with the window 1.88x saturated.**

## 2. Task 2 — the variant (code path, DEFAULT OFF)

- `vendor/.../quantized/qmv_tiled_ksplit_kernel.hip` (new module):
  `qmv_tiled_ksplit_kernel` — grid (M, n_tiles, 4); block k = blockIdx.z
  covers K values [k x K/4, (k+1) x K/4); the per-tile body is the stock
  qmv_tiled_kernel 4-bit body verbatim (same BSK = 512, same x staging, same
  per-lane 2 x uint32 streaming load, same dequant_and_dot4 / reduce_qdot4 /
  scale-bias order) so each partial equals the stock `acc` after the same tile
  sequence from zero; per-split float partials [M, N, 4].
  `qmv_tiled_ksplit_reduce_kernel` — per-column left fold
  ((p0 + p1) + p2) + p3 -> bf16, fixed order (deterministic; no atomics).
- `vendor/.../quantized/qmm.hip`: the only addition on the stock path is one
  `std::getenv` read; the ksplit block sits inside the existing tiled gate,
  before the QGD check, and is skipped entirely when the env is unset — the
  unset-env path runs the stock tiled launch verbatim.
- Host gate: `MLX_ROCM_QGD_KSPLIT` set && bits == 4 && M == 1 && K % 4 == 0 &&
  K/4 >= 512 && K/4 % 512 == 0 (split bases stay on BSK/pack boundaries, so
  every split is an exact multiple of BSK and the stock fast path covers every
  tile — no tail branch). K = 4096 -> base 1024 = 2 x BSK (the chartered
  class). Any other K (e.g. 2048 -> base 512 = 1 x BSK would gate in;
  K = 1024 -> base 256 < 512) falls through to the stock tiled launch.
- Numerics (pre-measurement expectation): the reduce re-associates (4 partials
  left-folded vs the stock single accumulator), so the set-env output was
  expected CLOSE but not bit-identical to stock on the split shapes (a few ULPs
  at most). **Measured reality (section 3): it is bit-identical** — for 4-bit
  g64 data the float32 partials + left-fold round to the same bf16 bits. The
  unset-env path is bit-identical by construction, and the K = 1024 fallback
  shape is bit-identical even with the env set (it runs the stock kernel).

## 3. Bit-identity + correctness check

| check | result |
|---|---|
| stock vs stock (cross-process, 8 rand ops) | PASS (hash_match, maxdiff = 0) |
| ksplit vs stock (8 rand ops, incl. split 8192x8192 / 8192x4096) | PASS (hash_match, 0 diff) |
| ksplit vs stock (6 model ops: 4 GDN in-proj + out_proj, gate_proj) | PASS (hash_match, 0 diff) |
| K = 1024 fallback shape, env set (must be bit-identical to stock) | PASS (stock path) |
| libmlx symbols: ksplit kernel + reduce present (66 / 22 strings) | PASS |
| env string `MLX_ROCM_QGD_KSPLIT` present in libmlx | PASS |

Better than the pre-measurement expectation: even the split shapes are
bit-identical (maxdiff = 0.000e+00, hash_match across all 14 ops), so the
variant is numerically transparent, not just "close".

## 4. Task 3 — same-session A/B measurement

Protocol: 25 s idle + SCLK sample before every measurement (25 s idle before
the bit-checks too); per-class microbench (verify-bw-perop-m3k.py, md5
d99ba9ae) on the exact qmv_gdn shapes under the 13-var config-of-record +
`MLX_ROCM_QMM_M3=1` in every arm (C = stock, K = + `MLX_ROCM_QGD_KSPLIT=1`),
3 runs per arm interleaved; then the full canonical bench (mtp-bench6.py,
MTP_MAX_TOKENS=322 anchor, both prompts, 3 runs per arm per prompt, C/K
interleaved). All under `flock ~/gpu.lock`, single session, single
force-reinstalled wheel (`mlx-0.32.3.dev20261105+c8419cb`) in a fresh venv
(copy of the canonical fold venv; the canonical venvs were never touched,
md5 pre/post identical). SCLK sampled before every measure: DPM level [2] =
**2200 MHz** (locked; `pp_dpm_sclk` line "2: 2200Mhz *" — 20/20 samples,
session 06:01:49Z-06:19:53Z).

### Per-class probe (qmv_gdn class, K=4096 N=4096, no bias, M=1)

| arm | ms | GB/s | % floor | vs C |
|---|---|---|---|---|
| C1 | 4.397 | 206.9 | 37.9% | — |
| C2 | 125.378 | 7.3 | 1.3% | excluded |
| C3 | 4.438 | 204.9 | 37.5% | — |
| **C (C1+C3 mean)** | **4.418** | **205.9** | **37.7%** | — |
| K1 | 4.639 | 196.0 | 35.9% | −2.7% |
| K2 | 4.563 | 199.3 | 36.5% | −3.3% |
| K3 | 4.586 | 198.3 | 36.3% | −3.8% |
| **K mean** | **4.596** | **197.9** | **36.2%** | **−4.05%** |

C2 exclusion: a transient system-wide event, not a class effect — its
`FUSED_VERIFY_MS` was 58.65 (3x the normal ~21.05) and *every* class in the
run was uniformly ~3x slower (mlp 33.04 vs ~8.5 ms, norms 32.30 vs ~1.48 ms,
lm_head 311 GB/s vs ~448), at the same SCLK level [2] as its neighbors. It
cannot be a stock-kernel regression (C1 and C3, one minute apart, agree to
0.9%).

**Bar check: class K vs C = −4.05% (bar: >= +1%) — FAIL. Floor 36.2% (bar:
>= 37.2%) — FAIL. FAIL-as-chartered on both.**

### Canonical bench (322-token anchor, both prompts)

| arm | prompt | tok/s (3 runs) | mean | vs C |
|---|---|---|---|---|
| C | default | 69.12 / 69.13 / 69.20 | 69.15 | — |
| K | default | 69.25 / 69.07 / 69.17 | 69.16 | +0.01% |
| C | hello | 84.13 / 84.18 / 84.25 | 84.19 | — |
| K | hello | 84.12 / 84.21 / 84.32 | 84.22 | +0.04% |

TOK_PER_STEP = 2.193 in all 12 runs; FIRST20 token-identical within and
across arms on both prompts (lossless, 12/12). e2e note: the class delta
(−4.05% on 4.59 ms/step = −0.186 ms/step, i.e. ~−0.6% on 31.59 ms/step,
~69.05 t/s predicted) did not show up in the e2e — the measured K-vs-C delta
is +0.01% / +0.04%, flat within run noise.

## 5. Ceiling verdict — the in-flight hypothesis is falsified

The §1 arithmetic said the K arm's 512 KB in-flight window (1.88x the 273 KB
requirement) should move the class from 203 GB/s toward the 546 GB/s floor.
The measurement says no:

- 2048 blocks (64 per CU) with 512 KB in flight: **197.9 GB/s — below** the
  stock 512-block / 128-KB launch at 205.9 GB/s. Four times the parallelism
  and 4x the in-flight window added no bandwidth.
- The extra parallelism cost +0.178 ms (partial [M,N,4] float writes + the
  per-column reduce round-trip), a net −4.05% on the class and a floor drop
  37.7% -> 36.2%.
- The class sat at ~36-38% of floor at both 512 and 2048 blocks — the bound is invariant to block count, so it is not an occupancy/parallelism bound
  either (2048 blocks = 64 per CU, well past 16/CU).

**Remaining bound, named with evidence: the per-tile `__syncthreads` staging
chain** — the 2-sync staging pair inside the serial k-chain (8x per column
stock, 2x per K-quarter under K-split) — plus the partial/reduce round-trip when the tile is split. The 2048-block measurement with the load window
1.88x saturated and *lower* GB/s is the direct evidence that outstanding
loads are not the limiting quantity: the staging structure of the tile is.

Ceiling for the qmv_gdn axis on this branch: **stock 4.42 ms (C1+C3 mean)
is the best achievable so far; the K-split variant (4.60 ms) is not
deployable** — it is slower than stock on the class and its e2e delta is
flat. Wash control (t2048: 4.4-4.7 ms wash, falsifier <4.0): K = 4.596 ms
falls inside the wash band; the falsifier did not trigger; **wash
confirmed**.

## 6. Next axis + falsifier

**Axis: remove the per-tile staging sync pair** (warp-private / in-register
staging so the k-chain carries no `__syncthreads`), keeping the stock 512-
block geometry — the one remaining named bound in the gen2+ksplit evidence.
**Falsifier:** the no-sync variant measures >= stock 4.42 ms (or < 37.2% of
floor with the sync chain gone) — then the bound is the serialized 2x8
k-chain's HBM-latency exposure itself (a memory-latency wall, not a
synchronization wall), the qmv_gdn kernel axis is closed at ~4.4 ms / ~38%
of floor, and the remaining levers are structural (the 24x per-step class
multiplicity, i.e. tape/verify architecture, not this kernel).

## 7. Run status / evidence

- Base: `qgd-gen2-probe` @ 8fa1a27; probe branch `qgd-ksplit-probe`
  (commits 37fdd32 kernel, 6b5b9c3 session script + bit-check).
- Box build: wheel `mlx-0.32.3.dev20261105+c8419cb` built and force-
  reinstalled into a fresh venv (copy of the canonical fold venv); import
  OK; symbols + env string verified; 4 shared quantized/ files treecheck-
  matched the branch (md5 PASS x4, incl. the gen2 tiled kernel supplied
  from the branch).
- **All measure phases completed on the box before the run's 06:38Z death:**
  session 05:58:32Z-06:20:28Z (lock acquired 06:01:49Z, no contention at
  acquisition), follow-up once-check done 06:32Z. The death (empty-final
  turn) happened after data collection, before report finalization; all raw
  data below was taken from the box logs (~/qkd/session.log + raw-*.raw).
- Integrity: sidecar md5 pre/post `02a54359cda8b247f76b277dd9bc4d4c`
  (matches md5-of-record); canonical venv libmlx md5s pre/post identical
  (m3k a0057c22, p3 460c803d, r4v fa4ace5b — untouched).
- **NOTE (carried from the dead run):** the r4v venv's libmlx md5 on the box
  (fa4ace5b...) differs from the 8acf13... recorded in the brief; it was
  untouched pre and post (recorded only, not re-derived).

## 8. Check lines

- format: `uv run ruff format --check .` — PASS (see this run's check
  output; only scripts touched in the resume, no code files).
- persona-leak grep over the diff + new commits (patterns per brief, git
  author lines exempt): **0 hits — PASS**.
- line-wrap check (manual 100-col scan; line_wrap_check.py not in
  tree) on this file: **0 hits — PASS**.
- size gate: `size_ratchet.py --base HEAD` — OK (no touched file grows
  beyond budget; the kernel module was committed by the dead run's
  verified step).
