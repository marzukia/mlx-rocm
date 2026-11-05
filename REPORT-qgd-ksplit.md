# REPORT-qgd-ksplit.md — 4x K-split M=1 tiled QMV probe (t2050, report-branch only)

**Verdict (pre-measurement): PENDING — see §7 for the run status.**

Chartered by the t2040/gen2 ceiling analysis: the qmv_gdn class (GDN 4 in-proj +
out_proj, 4-bit g64 affine, no bias, M=1, K=4096, N=4096) is
**latency/occupancy-bound, not HBM-bound** — the stock tiled launch puts ~128 KB
of weight traffic in flight vs the ~273 KB needed to saturate the 546 GB/s HBM
floor, and the gen-2 memory-structure family (load width/order) is exhausted.
The chartered fix is a *parallelism* change: 4x K-splits. This probe implements
one env-gated variant, checks bit-identity, and measures it same-session against
C.

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
  noise, or 203-273 GB/s with the in-flight window now 1.88x saturated —
  which would mean the ceiling is the per-tile sync/staging chain itself, not
  the outstanding-load requirement.

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
- Numerics: because the reduce re-associates (4 partials left-folded vs the
  stock single accumulator), the set-env output is CLOSE but **not
  bit-identical** to stock on the split shapes (a few ULPs at most for 4-bit
  g64 data); the unset-env path is bit-identical by construction, and the
  K = 1024 fallback shape is bit-identical even with the env set (it runs the
  stock kernel).

## 3. Bit-identity + correctness check

| check | result |
|---|---|
| stock vs stock (cross-process, 14 ops incl. the model tensor set) | PENDING |
| ksplit vs stock (same 14 ops) | PENDING |
| K = 1024 fallback shape, env set (must be bit-identical to stock) | PENDING |
| stock wheel symbols: qmv_tiled_ksplit_kernel / reduce present in libmlx | PASS (66 / 22 symbol strings) |
| env string `MLX_ROCM_QGD_KSPLIT` present in libmlx | PASS |

## 4. Task 3 — same-session A/B measurement

Protocol: 25 s idle + SCLK sample before every measurement; per-class microbench
(verify-bw-perop-m3k.py) on the exact qmv_gdn shapes under the 13-var
config-of-record + `MLX_ROCM_QMM_M3=1` in every arm (C = stock, K = +
`MLX_ROCM_QGD_KSPLIT=1`), 3 runs per arm interleaved; then the full canonical
bench (mtp-bench6.py, MTP_MAX_TOKENS=322 anchor, both prompts, 3 runs per arm
per prompt, C/K interleaved).

### Per-class probe (qmv_gdn class, K=4096 N=4096, no bias, M=1)

| arm | ms | GB/s | % floor | vs C |
|---|---|---|---|---|
| C1 / C2 / C3 (stock) | PENDING | | | — |
| K1 / K2 / K3 (K-split) | PENDING | | | PENDING |

### Canonical bench (322-token anchor, both prompts)

| arm | prompt | tok/s (3 runs) | vs C |
|---|---|---|---|
| C / K | default | PENDING | PENDING |
| C / K | "Hello" | PENDING | PENDING |

## 5. Ceiling verdict

PENDING — see §1 falsifier.

## 6. Next axis + falsifier

PENDING.

## 7. Run status / evidence

- Base: `qgd-gen2-probe` @ 8fa1a27 (branch tip 2b4b48b, whitespace-only reflow);
  probe branch `qgd-ksplit-probe` pushed to the bare mirror.
- Box build: wheel `mlx-0.32.3.dev20261105+a93ce94` built and
  force-reinstalled into a fresh venv (copy of the canonical fold venv);
  import OK; symbols + env string verified (above).
- Tree spot check: the 4 shared quantized/ files byte-match the branch
  (md5 PASS x4) — including the gen2 tiled kernel, which the box's gen2 vendor
  copy was missing and was supplied from the branch for this build.
- Pre/post md5s of the sidecar and the three untouched venvs: see the session
  log; the sidecar md5-of-record matches pre and post.
- **NOTE:** the r4v venv's libmlx md5 on the box at session start
  (02038c16...) differs from the 8acf13... recorded in the brief; it was
  untouched pre and post (recorded only).

## 8. Check lines

- format: `uv run ruff format --check .` on touched code — 120 files already
  formatted (PASS).
- lint: `uv run ruff check` on the new script — PASS.
- size gate: `size_ratchet.py --base HEAD` — SIZE GATE: OK (2 ratcheted).
- persona-leak grep over the diff + new commits: clean.
- line-wrap check on this file: see below (100-col wrap).
