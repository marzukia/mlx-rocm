# K2 non-GEMM fusion — round-4 fix report (monky)

Branch: `k2-non-gemm-fusion` (base `33b13dd`). Fixes the round-3 adversarial
review (t535, `reviews/k2-r3-review.md`): 0 BLOCKER / 3 MAJOR / 2 MINOR / 3 NIT.
Scope of this round: the 3 MAJORs (one-line guards in `fast.cpp`) + MINOR-1
(report reconciliation). MINOR-2 (fast.h wording) and NITs are left open
(see §7).

## 1. The fixes

All in `vendor/mlx-rocm-upstream/mlx/fast.cpp` (the kv_append gate). The
verified wheel (t531, source `ec48eb4c`, `/mnt/models/mlx-clone-k2`) carries
MAJOR-1; MAJOR-2/3 are new one-line checks per the review's suggested fix
list. No other file changed (fast.h / fast_primitives.h / copy.hip /
rms_norm.hip were already byte-identical to the wheel source — md5-verified).

### MAJOR-1 — `kernel_layout_ok`: unscheduled guard (ported from the wheel)

An unscheduled graph node carries the C-default strides from
`ArrayDesc::init()` (and all-true default flags), which always pass the
stride gate, so a non-standard lazy view was routed to the fast kernel and
misaddressed at eval time — the r1 failure mode, open on the branch.

```diff
   // fallback (arbitrary strides).
   //
   // Layout metadata is only trustworthy once the array is materialized:
   // an unscheduled graph node carries the C-default strides from
   // ArrayDesc::init() (and all-true default flags), not the true view
   // strides, while the kernel reads the input's strides at eval time
   // (after materialization). Checking the stale metadata would dispatch
   // the fast kernel for a transposed view (e.g. [2,2,1,8] strides
   // (8,16,1,1) of a transposed [1,2,8] buffer) and misaddress the B
   // rows. This matches the verify harness's _row_contiguous numpy-bridge
   // check, which materializes before reading strides: both reject the
   // view.
   auto kernel_layout_ok = [](const array& a) {
+    if (a.status() == array::Status::unscheduled) {
+      return false;
+    }
     const auto& sh = a.shape();
```

**Exact-port proof** — `diff` of the branch's pre-fix fast.cpp (git `33b13dd`)
against the wheel source (`/mnt/models/mlx-clone-k2/mlx/fast.cpp`,
`ec48eb4c`): the wheel's only delta is the comment paragraph above + the 3-line
guard. After the port, `diff` of the branch's post-fix fast.cpp against the
wheel source shows **only** the two new checks (MAJOR-2/3 below) — the
guard and comment are byte-identical to the wheel:

```diff
# diff: wheel fast.cpp -> branch post-fix fast.cpp (only delta)
209a210,217
>   // K2 review MAJOR-3 (round 4): the fast kernel writes cv at
>   // offset * cache_values strides[2] with no bounds check, so a smaller-T
>   // cache_values would be written out of bounds. Validate up front.
>   if (cache_keys.shape(2) != cache_values.shape(2)) {
>     throw std::invalid_argument(
>         "[kv_append] cache_keys and cache_values must have the same "
>         "T (shape(2)).");
>   }
228a237,245
>   // K2 review MAJOR-2 (round 4): the fast kernel is single-dtype (one T
>   // for all four pointers, dispatch on keys dtype only), so a mixed
>   // keys/values dtype pair (each matching its own cache) would be read
>   // with the wrong element size — silent corruption. Validate up front.
>   if (keys.dtype() != values.dtype()) {
>     throw std::invalid_argument(
>         "[kv_append] keys and values must have the same dtype (the fast "
>         "kernel is single-dtype).");
>   }
```

### MAJOR-2 — mixed k/v dtype check (new)

The fast kernel is single-dtype (one `T` for all four pointers, dispatch on
`keys.dtype()` only); a mixed keys/values pair (each matching its own cache)
passed the pre-existing k==ck / v==cv checks and was misread with the wrong
element size. Added before the fast path, after the existing dtype checks:

```cpp
  if (keys.dtype() != values.dtype()) {
    throw std::invalid_argument(
        "[kv_append] keys and values must have the same dtype (the fast "
        "kernel is single-dtype).");
  }
```

### MAJOR-3 — cache T-shape check (new)

The fast kernel writes `cv` at `offset * cache_values strides[2]` with no
bounds check; `cache_values.shape(2) <= offset < cache_keys.shape(2)` was an
OOB global write (the offset range check is against `cache_keys` only).
Added after the B/H/D cache-shape check:

```cpp
  if (cache_keys.shape(2) != cache_values.shape(2)) {
    throw std::invalid_argument(
        "[kv_append] cache_keys and cache_values must have the same "
        "T (shape(2)).");
  }
```

## 2. Rebuild (guest, /mnt/models/mlx-clone-k2)

fast.cpp synced (md5-verified), then in `/mnt/models/venv-k2` PATH:
`cmake -S . -B build/temp.../mlx.core -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1
-DBLA_VENDOR=OpenBLAS -DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
-DCMAKE_BUILD_TYPE=Release` (reconfigure, 9.3s) → `cmake --build ... -j8`
(fast.cpp recompiled, libmlx.so relinked, python extension rebuilt) →
`pip wheel . --no-deps -w /mnt/models/dist/ --no-build-isolation` →
force-reinstall into venv-k2. Log: `/mnt/models/tmp/k2-r4-build.log`.

Wheel: `mlx-0.32.3.dev20261026+the earlier vendor base-cp312-cp312-linux_x86_64.whl`,
`sha256=d617a1b841c324e426367c0299154c23c566d562e805046dcc130a4711e881ee`
(supersedes `ec48eb4c`).

Build log tail:

```
[ 31%] Building CXX object CMakeFiles/mlx.dir/mlx/fast.cpp.o
[ 35%] Linking CXX shared library libmlx.so
[ 86%] Built target mlx
[ 92%] Building CXX object python/src/CMakeFiles/core.dir/fast.cpp.o
[100%] Linking CXX shared module .../mlx/core.cpython-312-x86_64-linux-gnu.so
[100%] Built target core
BUILD_EXIT=0
  Created wheel for mlx: filename=mlx-0.32.3.dev20261026+the earlier vendor base-cp312-cp312-linux_x86_64.whl size=21540077 sha256=d617a1b841c324e426367c0299154c23c566d562e805046dcc130a4711e881ee
WHEEL_EXIT=0
Successfully installed mlx-0.32.3.dev20261026+the earlier vendor base
INSTALL_EXIT=0
=== BUILD END Mon Oct 26 12:41:04 UTC 2026 ===
```

Install verification: installed `core.so` / `libmlx.so` md5-identical to the
wheel; `strings libmlx.so` contains both new guard messages;
`import mlx.core` OK in venv-k2. (Note: venv-k2's `bin/pip` shebang points at
`/mnt/models/venv`, so the reinstall used `venv-k2/bin/python -m pip` — the
first `pip` invocation installed into the base venv; corrected before any
GPU run.)

## 3. Suite result (this wheel, log preserved on disk)

`scripts/verify-k2.py` (69 checks), venv-k2, `flock /mnt/models/gpu-bench.lock`,
`LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`. Full log:
**`reports/k2-r4-suite-full.log`** (79 lines, preserved in-repo).

**Verdict: 69 PASS / 0 FAIL** (`grep -c '^\[PASS\]'` = 69,
`grep -c '^\[FAIL\]'` = 0, "all unit checks passed"). No regression vs the
t531 run (69/0 on `ec48eb4c`): the new checks only fire on invalid input,
and the unscheduled guard routes the same lazy suite inputs to the same
slice_update fallback the t531 wheel used.

This also resolves MINOR-1: the 69/0 evidence is now substantiated on disk —
`reports/k2-r4-suite-full.log` is the full raw log of the suite run on the
branch-as-committed wheel (`d617a1b8`), and `reports/k2-r2-suite-evidence.txt`
retains the t531 session extraction for the `ec48eb4c` run.
`reports/k2-verify-r2.md`'s sourcing note is updated accordingly.

## 4. Guard-firing evidence (probe, log: `reports/k2-r4-guards.log`)

`k2-r4-guards-probe.py` (5 cases, same env/lock protocol):

1. **Materialized standard layout, ck buffer shared via view** (so the
   fallback's buffer-donation cannot fake in-place):
   `ck buffer modified at offset row: True | cv: True | rest untouched: True`
   → **VERDICT: FAST KERNEL (in-place, correct)**. The fast path is intact
   for scheduled inputs; the unscheduled guard is a no-op there (no
   happy-path change — (c)).
2. **Materialized non-contig view** (strides (8,16,1,1)): `0/256` mismatched
   → **FALLBACK (correct)** — the stride gate rejects it (case would have
   been misaddressed by the fast kernel).
3. **Unscheduled non-contig view** (the suite's case, lazy at gate time):
   `0/256` mismatched → **FALLBACK (correct)** — **MAJOR-1 guard fires**
   (without it, C-default strides pass the gate and the fast kernel
   misaddresses: r1's 16/256).
4. **Mixed k/v dtype** (keys bf16 + values f32, each matching its own
   cache): `RAISED ValueError: [kv_append] keys and values must have the
   same dtype (the fast kernel is single-dtype).` → **MAJOR-2 guard fires**.
5. **Cache T mismatch** (ck T=8, cv T=4, offset=5): `RAISED ValueError:
   [kv_append] cache_keys and cache_values must have the same T (shape(2)).`
   → **MAJOR-3 guard fires**.

## 5. Host-side checks

`uv run ruff check .` / `ruff format --check .`: clean.
`uv run python scripts/size_ratchet.py --base HEAD`: SIZE GATE OK.
`uv run pytest -m "not gpu"`: 6 passed, 1 skipped.

## 6. Known open items (not fixed this round)

- MINOR-2 (review): `fast.h` kv_append doc wording ("non-row-contiguous
  inputs" → "inputs not satisfying the kernel's stride layout"). Left as-is
  to keep the branch's fast.h byte-identical to the verified wheel; one-line
  doc fix for the next round.
- NIT-1: uint4 row-base alignment not gated (pre-existing; misaligned vector
  loads are legal on AMD). NIT-2: harness H==1 corner (unreachable in the
  current suite). NIT-3: AGENTS.md ROCm 7.7 vs guest ROCm 10.1 (profile
  drift).
- HIP teardown exit-hang after the suite (results complete before the hang;
  runs use the timeout wrapper + early kill on the completion marker).

# t789 — vendor-pin fix (dangling pin + silent build-wheel) — worker report

Branch: `worker/t789-vendor-pin-fix` (base `origin/main` @ 64b98b8).
Fixes the t781 review-FAIL finding on PR #10 (96a1677 "fix(rocm): GDN fused
decode T=1 launcher pointer write"): `vendor.json` pinned
`75ca881b2e82f7b4069c8564a203792aa16d8a29`, but the fork branch
`lifeofpiboy/mlx:tail-fusion-gdn` was at `96a1677` (the fix's parent) —
`75ca881` existed only in a guest build clone and was never pushed, and
`scripts/build-wheel.sh` never verified the checkout against the pin.

## 1. Push the pin (part 1)

- `75ca881` recovered from the t761 build clone (`/mnt/models/mlx-clone-g4`;
  the task text said g1, but g4 is the clone that carries it — g1's HEAD is
  `1641333`). Fetched into a local repo, then:

  ```
  git push --force-with-lease origin 75ca881:refs/heads/tail-fusion-gdn
  #   To github.com:lifeofpiboy/mlx.git
  #    + 96a1677...75ca881 75ca881 -> tail-fusion-gdn (forced)
  ```

  Fast-forward on the branch (96a1677 is 75ca881's parent; the push is a
  2-commit advance, no history rewrite of the pinned commit).

- Verification (all three agree):
  - `git ls-remote https://github.com/lifeofpiboy/mlx.git refs/heads/tail-fusion-gdn`
    → `75ca881b2e82f7b4069c8564a203792aa16d8a29`
  - `gh api repos/lifeofpiboy/mlx/commits/75ca881b2e82f7b4069c8564a203792aa16d8a29`
    → 200, `sha=75ca881b...`, `commit.message="fix(rocm): GDN fused decode T=1 launcher pointer write"`,
    parent `96a16773a1133795406a85545b654b6972d93721`
  - `gh api repos/lifeofpiboy/mlx/branches/tail-fusion-gdn` →
    `sha=75ca881b...`
  - `75ca881` is also in the fork's default branch (`main`) history
    (`git merge-base --is-ancestor` → yes), so the pin is resolvable from
    any clone of the fork.

## 2. build-wheel.sh: verify the checkout against the pin (part 2)

`scripts/build-wheel.sh` now reads the pin from `vendor.json`
(`PIN_SHA=$(python3 -c 'import json; print(json.load(open("vendor.json"))["ref"])')`)
and, after the clone/checkout, verifies it:

```bash
# Verify the checkout matches the pinned sha — a dangling pin must fail
# loudly, not build a different commit silently.
CHECKED_OUT_SHA=$(git rev-parse HEAD)
if ! git cat-file -e "${PIN_SHA}^{commit}" 2>/dev/null; then
  echo "ERROR: pinned sha ${PIN_SHA} not resolvable in the vendor clone" >&2
  echo "       (pin=$PIN_SHA checked-out=$CHECKED_OUT_SHA)" >&2
  exit 1
fi
if [ "$(git rev-parse "${PIN_SHA}^{commit}")" != "$CHECKED_OUT_SHA" ]; then
  echo "ERROR: checkout does not match vendor.json pin" >&2
  echo "       (pin=$PIN_SHA checked-out=$CHECKED_OUT_SHA)" >&2
  exit 1
fi
echo "vendor checkout verified: ${CHECKED_OUT_SHA} == vendor.json pin ${PIN_SHA}"
```

Behavior (exercised with a stubbed clone, three branches):
- pin not resolvable in the clone → `ERROR ... not resolvable` + exit 1
- pin resolvable but checkout differs → `ERROR ... does not match vendor.json
  pin (pin=... checked-out=...)` + exit 1
- pin matches → `vendor checkout verified: ... == vendor.json pin ...` + continue

No other change to the script (clone/fetch/install flow untouched).

## 3. G1/G2 on a wheel built from the landed ref (part 3)

Built on neon (guest, ROCm 10.1, gfx1201) under `flock /mnt/models/gpu-bench.lock`:

- Fresh clone of `lifeofpiboy/mlx` @ `tail-fusion-gdn` →
  `/mnt/models/mlx-clone-t789`; `git rev-parse HEAD` =
  `75ca881b2e82f7b4069c8564a203792aa16d8a29` (the pinned commit, from the
  fork — not the guest build clone).
- `MLX_VENDOR_DIR=/mnt/models/mlx-clone-t789 bash scripts/build-wheel.sh`
  (with the new verification — it printed `vendor checkout verified` and
  proceeded) → `/mnt/models/dist-t789/mlx-0.32.3.dev20261028+75ca881-cp312-...whl`.
- Installed into a scratch venv (`/mnt/models/venv-t789`, python 3.12.3;
  `venv-g4` untouched). The scratch venv mirrors venv-g4's mlx_lm model
  files: `qwen3_5.py` (S==1-guarded fused branch) **and** `gated_delta.py`
  (K3 patch: non-fused GDN update routes through
  `mx.fast.gated_delta_update` on ROCm). The first gate round ran with the
  stock PyPI `gated_delta.py` (Python op loop) and G2 diverged at token 33 —
  an environment artifact, not a wheel difference: the m0 reference and the
  t771 AR baseline were both produced with the K3-patched wrapper. With the
  K3 file in place, all gates below pass.

### Rule-6 effective config (printed + asserted by `scripts/t789_config.py`)

```
t789-config: OK
  env: MLX_GDN_TAIL_FUSION=1  MLX_ROCM_GDN_FALLBACK=0  MLX_GDN_MLXFAST=1
  wheel: 0.32.3.dev20261028+75ca881  (source sha 75ca881b2e82f7b4069c8564a203792aa16d8a29)
  device: gpu  (rocminfo: gfx1201, AMD Radeon RX 9070 XT)
```

Kernel names (rocprofv3 `--kernel-trace` over a dispatch of all three fused
ops): `gdn_prep`, `gdn_qk_norm`, `gdn_norm_gated` all dispatched.

### G1 — no-fault serve/generation: PASS

`TAIL_FUSION=1`, greedy, 1024 tokens, qwen3-9b-mlx 4bit:
`FUSION=1 LOAD=2.82 GEN=14.71 TOKS=1024 TOKS_S=69.59`, rc=0, no HSA-fault
signature in the log (grep `HSA|fault|Aborted|core dumped|Segmentation|HIP`
→ no match).

### G2 — maxdiff 0.0 vs the 4bit-MLX reference + fused==unfused: PASS

Same model/prompt/seed as the t612 reference (bench4 `m0.out`, 1024 greedy
tokens), vacuity-guarded stream compare:

```
G2-FUSED-VS-M0-REF:    STATUS=MATCH (maxdiff 0.0) first_div=- match=1024/1024
G2-FUSED-VS-UNFUSED:  STATUS=MATCH (maxdiff 0.0) first_div=- match=1024/1024
GATE-LIVE (all compared streams non-empty)
```

The fused stream is bit-identical to both the 4bit-MLX (m0) reference and
the same-wheel unfused stream. Cross-check: the t771 AR baseline
(`bench-g4d/m0a.out`, 75ca881 wheel, venv-g4) is byte-identical to m0
(diff 0/1024), so the reference itself is stable across the two 75ca881
builds.

### t761 primitive gates (re-run on this wheel): PASS

```
PRIM-GATES-PASS
FWD-LAYER maxdiff=0.000000e+00            (fused T=1 layer fwd, 5-layer chain)
[zero]/[rand] norm_gated maxdiff=0.0      (gdn_norm_gated vs (x/||x||)*z)
[zero]/[rand] qn,kn,vc,beta,g maxdiff=0.0 (gdn_prep vs conv1d+silu+rms_norm+sigmoid+compute_g; g float32)
```

### Sibling state (GPU lock)

Lock was free for every t789 run. Sibling t784 (QMV lane, dist-g4e):
`build_g4e3.log` shows the `ced8c58` wheel build completed 12:23 UTC
(sha256 `d85065b3...` recorded) — no GPU process of theirs was active at
check time; no wall-clock remainder visible.

## 4. Files

- `scripts/build-wheel.sh` — pin verification (this PR)
- `scripts/t789-run.sh`, `scripts/t789_config.py`, `scripts/t789_gen.py`,
  `scripts/t789_kernels.py` — gate harness (kept in-repo for re-run; all
  skip-safe off-GPU)
- `REPORT.md` — this report

# t771: Speed bench on the fixed gdn wheel (dist-g4d @75ca881)

MTP-ON 100 tok/s lane. Measured the fixed wheel (t761: gdn_prep launcher fix,
14 kernel-arg host `Data*` ptrs passed as device ptrs; fix commit `75ca881`).
G1/G2 had already PASSed on dist-g4d (no-fault, maxdiff 0.0 vs reference,
fused==unfused). This run is the first speed measurement on the fixed wheel.

## Verdict (gates, operational order 2026-10-28)

| Gate | Result |
|---|---|
| 1. Lossless (G4, m2 vs m0, #8 harness) | **FAIL** — first divergent step **149** (1-based), 865/1024 tokens differ |
| 2. Beat AR (64.94 pre-fix baseline) | MTP 64.34 vs AR 69.65 → **FAIL** (-7.6% vs AR) |
| 3. Enable MTP default | **OFF** (gate 1 failed; gate 2 failed) |

Lane goal 100 tok/s decode: **not reached**. AR moved to 69.65 tok/s; MTP K=1
M=2 is 64.34 tok/s @ 76.0% accept — the gdn fix did **not** close the MTP gap.

## Phase 1 — pre-stage + verify (all PASS)

- SSH chain: `ssh -J andryo@100.91.94.56 neonvm@192.168.122.234` (guest neonvm).
- Build clone: `git -C /mnt/models/mlx-clone-g4 log --oneline -1`
  → `75ca881 t761: fix gdn_prep launcher - data_shared_ptr().get() is a host Data*, not a device pointer`
- Wheel identity (venv `/mnt/models/venv-g4`, python3.12):
  `mlx-0.32.3.dev20261028+75ca881` (+ `mlx_lm-0.32.2`) — matches the fixed
  clone ref. `import mlx.core` OK under `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`.
- Lock: `flock -w 10 /mnt/models/gpu-bench.lock -c true` → free (no GPU siblings).
- Model-load smoke (flock, 10 tokens, Qwen3.5 GDN-hybrid 9B 4bit
  `/mnt/models/qwen3-9b-mlx`): `SMOKE-OK n=10 dt=1.53`, no fault,
  gfx1201 bound.

## Phase 2 — bench (flock, fresh process per run, 3 repeats, median)

Harness: repo `bench/mtp-bench3.py` (AR = `MLX_MTP=0`, MTP K=1 M=2 =
`MLX_MTP=2`), 1024 tokens greedy, smartphone prompt, run via new runner
`bench/bench-g4d.sh` (copied to guest `/mnt/models/tmp/bench-g4d/`):

```
flock -w 300 /mnt/models/gpu-bench.lock env MLX_MTP=<0|2> \
  /mnt/models/venv-g4/bin/python /mnt/models/tmp/bench-g4d/mtp-bench3.py
```

| Config | n | min | **median** | max | accept |
|---|---|---|---|---|---|
| AR (`MLX_MTP=0`) | 3 | 69.31 | **69.65** | 69.75 | — |
| MTP K=1 M=2 (`MLX_MTP=2`) | 3 | 63.92 | **64.34** | 64.46 | 389/512 = 76.0% |

Raw lines (identical accept across all 3 m2 repeats):

```
MTP=0 LOAD=2.76 GEN=14.68 TOKS=1024 TOKS_S=69.75 ACCEPTED=0/0
MTP=0 LOAD=2.78 GEN=14.70 TOKS=1024 TOKS_S=69.65 ACCEPTED=0/0
MTP=0 LOAD=2.78 GEN=14.78 TOKS=1024 TOKS_S=69.31 ACCEPTED=0/0
MTP=2 LOAD=2.77 GEN=16.02 TOKS=1024 TOKS_S=63.92 ACCEPTED=389/512
MTP=2 LOAD=2.79 GEN=15.89 TOKS=1024 TOKS_S=64.46 ACCEPTED=389/512
MTP=2 LOAD=2.79 GEN=15.92 TOKS=1024 TOKS_S=64.34 ACCEPTED=389/512
```

**AR delta vs pre-fix (dist-g4c @96a1677): 69.65 vs 64.94 → +4.71 tok/s (+7.2%).**
The gdn fix changed the base path (gdn_prep runs on every target forward,
including AR). Pre-fix numbers are delta-only, not current.

MTP vs AR on the fixed wheel: 64.34 vs 69.65 → **-5.31 tok/s (-7.6%)**.
Pre-fix the gap was -1.8% (63.8 vs 64.94); the fix widened it. Accept rate
unchanged (76.0% vs 76% pre-fix).

MTP K=2 check: **not run** — conditional on (b) >= AR; 64.34 < 69.65.

## Phase 3 — G4 lossless gate (#8 harness, vacuity-guarded)

`GATE-LIVE` (all compared streams non-empty, 1024 tokens each).

| Compare | Status |
|---|---|
| G4-M2-M0 (m2a vs m0a, exact) | **FAIL** first_div=149 match=159 diff=865 of 1024 |
| M2-A-B, M2-B-C (m2 determinism) | MATCH 1024/1024 |
| M0-A-B, M0-B-C (m0 determinism) | MATCH 1024/1024 |

First divergent step: **149** (1-based). Tokens 145-160:

```
m0 (AR): 4779,11,321,5090,24296,13,271,794,357,13,561,717,383,29900,318,4272
m2 (MTP): 4779,11,321,5090,33443,51010,82,13,271,794,357,13,561,717,383,29900
```

m2 inserts 2 extra tokens (`51010, 82`) at 150-151 before re-syncing at 271 —
a state-advance/rollback mismatch in the MTP verify path (reject path:
GDN save/restore + KV trim + re-forward). Both paths are run-to-run
deterministic, so this is systematic, not noise.

**G4 verdict: FAIL (lossless-only; drift at step 149).**

Note for follow-up (not diagnosed here): the harness `save_gdn` zero-copy
save (`conv, ssm, lengths` per GDN layer) predates the fix; if the fixed
gdn_prep uses an additional state buffer, the MTP rollback would silently
miss it. The pre-fix t612 run reported m2 byte-identical to m0, so the
divergence is specific to the fixed wheel's state layout/path.

## Phase 4 — harness change

`bench/bench-g4d.sh` (new): parameterized runner (venv/harness/out overridable,
defaults to venv-g4), 3 repeats per config with median, conditional K=2, and
the PR #8 vacuity-guarded lossless gate (same `cmp_streams` helper as
`bench4.sh`/`bench5.sh`). No changes to `mtp-bench3.py`/`mtp-bench5.py`.

## Exact commands (repro)

```
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234
git -C /mnt/models/mlx-clone-g4 log --oneline -1        # 75ca881
ls /mnt/models/venv-g4/lib/python3.12/site-packages/ | grep mlx
flock -w 10 /mnt/models/gpu-bench.lock -c true          # LOCK-FREE
# smoke (10 tokens, flock -w 240): venv-g4 python, load + 10x argmax step
scp bench/{mtp-bench3.py,mtp-bench5.py,bench-g4d.sh} neonvm@...:/mnt/models/tmp/bench-g4d/
bash /mnt/models/tmp/bench-g4d/bench-g4d.sh            # 6 runs + gate (K2 not run)
```

---

# m6: rope_q row-invariant (MTP lossless gate G4 fix)

**Status (2026-10-28 23:20Z): GPU complete. Rope fix VERIFIED (focused
A/B + layer-3 chain row-invariant); G4 gate still FAILs at 148 - the
remaining T-dependence is in the GDN (gated delta net) path (first
diverging layer 4), a separate root-cause class (m3/m4 territory).
No bench regression (AR -0.32%, MTP K=1 -0.05% vs baselines).

## Problem (from m5)

MTP lossless gate G4 (2-row causal verify vs 1-row AR) FAILs at token
149. m5's A/B localized the first diverging component to `rope_q`
(1/4096 elements, maxabs 9.766e-04 = 1-2 ULPs bf16; numeric, not
logic). All upstream MATCH; downstream DIFFs are propagation only.

Root cause class (same as m4's SDPA): `rope_q` is the fused
`mx.fast.rope` kernel, whose ROCm implementation dispatches T-dependent
variants — a 1D single-token kernel for T=1 (AR) and a batched kernel
for T>1 (verify) — that can differ by 1 ULP on the same logical row
(same input, same position, different kernel schedule).

## Fix

`patches/m6-ropeq-rowinvariant.diff` (applied by
`scripts/apply-m6-patch.py`, idempotent, `.bak-t866` backup per the
t718c convention):

- New `RowInvariantRoPE` module in `qwen3_next.py`: the same rotation
  computed with plain elementwise ops — inv_freqs as an fp32 constant
  (`base ** (-i/32)`, mirroring the kernel's `exp2f(-(i/32) *
  log2(base))` as closely as the API allows), theta per row
  (`(offset + row) * inv_freqs`), cos/sin precomputed in fp32, then
  split/rotate/multiply/add in the model dtype (bf16). Row r depends
  only on (row r, offset + r), so T=1 and T=2 are bit-identical per
  row by construction.
- Preserved exactly: partial_rotary_factor (dims=64 of 256 rotated,
  192 passthrough bit-exact), traditional=False (pair
  (x[i], x[i+32]) rotated by theta_i), per-row offset semantics
  (pos = offset + row_idx), the causal string (no mask change), no
  padding.
- `Qwen3NextAttention.__init__`: the default (no rope_scaling) path
  uses `RowInvariantRoPE`; non-default scaling configs keep the
  original `initialize_rope` path (behavior unchanged).

## Verification

- CPU logic (guest, no GPU): row-invariance TRUE (f32 + bf16),
  passthrough bit-exact, matches the fp64 LLaMA convention (maxdiff
  ~7e-6 at fp32 rounding level), inv_freqs = base^(-i/32).
- GPU focused A/B (`scripts/diag-s62-m6.py`, m5 captured tensors,
  patched venv-g4): **M6-AB-OK** (exit 0).
  - rope_q vs-capture T=1 (the AR path): MATCH 0/4096 - the explicit
    rope is bit-exact with the fused 1D single-token kernel.
  - rope_q vs-capture T=2: row 0 MATCH 0/4096; row 1 DIFF 1/4096
    (h=10, d=38, maxabs 9.766e-04) - the documented m5 artifact: the
    old fused T=2 kernel's row 1 is 1 ULP off from its own T=1
    sibling, so no single explicit formula can match both kernels on
    that element; the patched rope matches T=1 and is row-invariant.
  - rope_q row-invariance (A row1 vs B): MATCH 0/4096.
  - rope_k A/B/row-invariance: all MATCH (0/2048, 0/1024, 0/1024).
  - full chain re-run (m5 Part 2, patched rope, captured inputs):
    all 19 components row-invariant - **first diverging component:
    NONE** (pre-patch m5: rope_q was the first; downstream DIFFs were
    propagation only). m3 pkl cross-check validates the capture
    (layer-3 k 0/91136; layer-4 gdn ssm 515560/524288).
- G4 gate (`scripts/run-m6-g4.sh`, bench4 logic on venv-g4, 5 arms):
  - vacuity-guarded (guard + ALLTOKS stream-format check: `scripts/run-m6-g4.sh` header, `bench/README.md` t638 audit).
  - pre-patch control (fresh baselines): m0 AR 69.31 tok/s, m2a MTP
    K=1 64.87 tok/s (ACCEPTED=389/512); M2-M0 FAIL first_div=148
    (= @149 1-indexed, the documented divergence) - gate confirmed
    live.
  - post-patch: **M2-M0-LOSSLESS FAIL first_div=148** (a=33443 m2a,
    b=24296 m0) - same position as pre-patch. Controls: M1-M0 MATCH,
    M2-A-B MATCH, M2-B-C MATCH (M1-M2A FAIL first_div=148 - the T=2
    verify vs the T=1 verify, same GDN cause).
  - **Root-cause re-localization (post-patch m3 layer diff, full
    model, S62 arm): first diverging layer = 4 (GDN conv 2059/24576,
    ssm 515560/524288) - bit-identical to pre-patch m3** (the GDN
    kernel is untouched by the m6 patch, as expected). Layers 0-3
    all clean (layer-3 attn k/v 0/91136 - the rope fix verified at
    the model level). l0 logits (after prev=279): MATCH (also
    pre-patch - the rope T-dependence was row-1 only); l1 logits
    (after draft=3545): DIFF 192967/248320 (maxabs 1.25e-01) -
    bit-identical pre/post patch, i.e. the GDN T-dependence
    dominates and the rope contribution was negligible in the
    logits.
  - The layer-4 conv state is a pure copy of the layer-3 input
    vectors (shift register, no arithmetic); its 2059/24576 diff
    proves the GDN-2 output (the layer-3 input) is T-dependent.
    The m6 A/B control verified h3(3545) (row 1) is row-invariant
    and the patched layer-3 block is row-invariant, so the
    remaining diff is in h3(279) (row 0) - the GDN kernel's
    row-0 per-token output (and intermediate state) is T-dependent
    (the T=2 batched kernel computes row 0's output differently
    from the T=1 single-token step, even though the final ssm
    state matches for layers 0-2). This is the m3/m4 root-cause
    class (GDN kernel T-dependence) - the next leg (m7) must make
    the GDN conv/ssm path row-invariant (e.g. run the GDN layers
    as 2xT=1 in the verify forward, or fix the T=2 kernel's row-0
    output computation).
- Bench (from the post-patch gate runs): AR (m0) 69.43 tok/s vs
  baseline 69.65 (**-0.32%, within the <1% budget**; fresh pre-patch
  control 69.31); MTP K=1 (m2a) 64.31 tok/s vs baseline 64.34
  (**-0.05%**; fresh control 64.87, ACCEPTED=389/512). No
  regression.

## Verdict

The m6 rope fix is verified: the attention path (rope) is now
row-invariant (focused A/B M6-AB-OK; layer-3 chain first diverging
component NONE; layer-3 attn k/v clean in the full model). The G4
lossless gate still FAILs at token 149 (first_div=148) - the
remaining T-dependence is in the GDN (gated delta net) layers
(first diverging layer 4; the GDN kernel's row-0 per-token output
is T-dependent). G4 will need the GDN path made row-invariant (m7)
to pass bit-exact. No bench regression (AR -0.32%, MTP K=1
-0.05% vs baselines).

## Files

- `patches/m6-ropeq-rowinvariant.diff` — the model patch (3 hunks:
  import math; RowInvariantRoPE class; conditional rope init).
- `scripts/apply-m6-patch.py` — idempotent applier + .bak-t866.
- `scripts/diag-s62-m6.py` + `scripts/run-m6-ab.sh` — focused A/B.
- `scripts/run-m6-g4.sh` — G4 gate + bench extraction.
# Step 0 — in-guest streaming-read ceiling + runtime CU count

Worker: piboy (2026-10-28). Source of truth: method-correction doc
(https://drop.cain.rocks/planky/3150ddc860c16a6d3bf769f3.html) — section 4 step 0,
section 2 rules 1/2/6/7. No model load. All GPU time under
`flock /mnt/models/gpu-bench.lock` (lock free at dispatch; t771 terminal).

## Headline

**Measured in-guest streaming-read ceiling: 636 GB/s (99.4% of the 640 GB/s spec
peak). No partition flag: 636 ≥ ~550, so the doc's section 4 step 0 exit
condition does NOT trigger — the guest CAN saturate the 9070 XT, and the doc's
640 GB/s-based targets stand (rescaled table in §4 below, numerically ~unchanged
because 636 ≈ 640).**

The runtime CU count is split (doc section 1 row 4): **HIP userspace reports 32
CUs (this is what mlx uses, `cus=32`); the KFD driver and HSA runtime report the
full 64 CUs.** The 32-CU number is a userspace view, not a hardware partition —
it does not throttle streaming bandwidth (636 GB/s measured).

## 1. CU count — what the runtime actually reports

| Layer | Command | Reports |
|---|---|---|
| HIP userspace (`hipGetDeviceProperties().multiProcessorCount`; the value mlx logs as `cus=`) | direct HIP query (`bench/step0_hip_cu.cu`, compiled with `/opt/rocm/core-10.1/bin/hipcc`) | **32** |
| HSA runtime | `/opt/rocm/core-10.1/bin/rocminfo` | **64** CUs (node 1 = gfx1201 "AMD Radeon RX 9070 XT", 2 SIMDs/CU, 4 shader engines; node 0 = Ryzen 7 9700X CPU, 8 cores) |
| KFD driver topology | `cat /sys/class/kfd/kfd/topology/nodes/1/properties` | **64** CUs (`simd_count 128`, `simd_per_cu 2`) |
| amd-smi | `/opt/rocm/core-10.1/bin/amd-smi` | full passthrough RX 9070 XT, 16304 MB, Partition-Mode N/A, ROCm 10.1.0 |

`amdgpu` module parameters: `disable_cu=(null)`, no partition mode. So the
"32-CU vGPU partition" claim (reports/opt-gauntlet-areaB-backend.md §6) matches
only the HIP/mx userspace view; driver + HSA see the full 64 CUs, and the
bandwidth measurement in §2 shows no partition throttling.

Note for kernel work: mlx's `cus=32` feeds kernel-tuning heuristics
(e.g. `qmm.hip` block-count sizing). Any heuristic that assumes 32 CUs is
under-provisioning by 2x relative to the hardware — flag for step 1.

## 2. Streaming-read ceiling (doc rule 1/2: same guest, same ROCm, same flock; GB/s from rocprofv3 kernel time + bytes moved)

Method: plain read-only kernel — `mx.sum` reduction (one heavy
`all_reduce_kernel` pass that reads the whole buffer once + one tiny final
scalar reduce) over an f32 device buffer, 3 warmup + 10 measured sums, timed by
`rocprofv3 --kernel-trace` kernel time only. No per-op latency, no host op
creation / `mx.eval` in the timing path (doc rule 2). rocm-bandwidth-test is not
installed in the guest.

| Pass | Buffer | Kernel (rocprofv3) | Total kernel time | GB/s min / median / max | Headline GB/s |
|---|---|---|---|---|---|
| 4 GiB (1,073,741,824 f32) | 4,294,967,296 B | `all_reduce_kernel` (20 dispatches) | 67.559 ms | 634.2 / 635.9 / 637.1 | 635.7 |
| 4 GiB (repeat) | 4,294,967,296 B | `all_reduce_kernel` (20 dispatches) | 67.516 ms | 634.8 / 636.1 / 637.1 | 636.1 |
| 2 GiB (536,870,912 f32) | 2,147,483,648 B | `all_reduce_kernel` (20 dispatches) | 33.859 ms | 631.6 / 634.7 / 636.4 | 634.2 |

**Ceiling = 636 GB/s** (median of the 4 GiB passes; flat across 2–4 GiB,
<1% spread across 3 passes). Bytes moved = buffer bytes (read) + 8 B per
iteration (negligible partial-read/scalar-write).

Effective config (doc rule 6, printed + asserted by the bench):
`venv=venv-g4 rocm=10.1 mlx_wheel=0.32.3.dev20261028+75ca881
buffer_bytes=4294967296 n_elements=1073741824 warmup=3 iters=10` — kernel
names actually dispatched: `all_reduce_kernel` only (from the rocprof pass).

## 3. Cross-checks

| Quantity | Value | % of measured ceiling (636 GB/s) |
|---|---|---|
| Fleet QMV (reports/opt-gauntlet-areaB-backend.md §3: 4.6 GB/token ÷ 9.46 ms) | 486 GB/s | **76.4%** (doc §1 row 1: ~76% — confirmed; the "51% of 960" was the wrong-denominator artifact) |
| Spec peak (256-bit GDDR6 @ 20 Gbps) | 640 GB/s | ceiling is 99.4% of spec |
| Measured ceiling | 636 GB/s | 100% (this number is "100%" per doc rule 1) |

## 4. Doc section 3 budget table, rescaled to the measured ceiling (C = 636 GB/s)

GEMV floor = 4.6 GB ÷ C; realistic target = 4.6 GB ÷ (0.88·C); non-GEMM and
launch/gap rows are not bandwidth-scaled (unchanged from the doc).

| Bucket | Now | Floor (100% of 636 GB/s) | Realistic target (~88%) | Headroom |
|---|---|---|---|---|
| Quantized GEMV (4.6 GB weight stream) | 9.46 ms | **7.23 ms** | **8.22 ms** | **1.24 ms** |
| Non-GEMM kernels (norms, copies, elementwise, GDN, attn) | 4.9 ms | <1 ms | ~1.5 ms | ~3.4 ms |
| Launch / commit / Python gaps (eager, ~1,070 dispatches) | ~1.5 ms | ~0 | ~0.3 ms (graph decode) | ~1.2 ms |
| **Total** | 15.9 ms = 63 tok/s | **7.23 ms = 138.3 tok/s** | **~10.0 ms = ~99.8 tok/s** | **~5.9 ms** |

vs the doc's 640 GB/s estimates (7.2 ms / 139 tok/s, 8.2 ms, ~10 ms / ~100
tok/s): differences <0.5% — the doc's targets stand as written. The "177 tok/s
ceiling" (960 GB/s-derived) remains unreachable; the zero-other-cost floor is
~138 tok/s.

## 5. Repro (exact)

```bash
# lock check first
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234 'flock -w 5 /mnt/models/gpu-bench.lock -c true && echo LOCK_FREE'

# deploy bench
scp -J andryo@100.91.94.56 bench/step0_streaming_read.py bench/step0_parse.py bench/step0_run.sh neonvm@192.168.122.234:/mnt/models/tmp/

# 4 GiB pass (default) — CU count + ceiling, under flock, rocprofv3 kernel time
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234 'bash /mnt/models/tmp/step0_run.sh'

# 2 GiB variant
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234 'STEP0_NBYTES=$((2*1024*1024*1024)) STEP0_OUT=/mnt/models/tmp/step0_2g bash /mnt/models/tmp/step0_run.sh'

# CU count (independent of the bench)
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234 '/opt/rocm/core-10.1/bin/rocminfo | grep -B1 -A4 "Marketing Name"; /opt/rocm/core-10.1/bin/amd-smi | head -14; cat /sys/class/kfd/kfd/topology/nodes/1/properties'

# direct HIP CU query (the value mlx uses)
ssh -J andryo@100.91.94.56 neonvm@192.168.122.234 'export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib; /opt/rocm/core-10.1/bin/hipcc -o /mnt/models/tmp/hipcu bench/step0_hip_cu.cu && flock -w 60 /mnt/models/gpu-bench.lock -c /mnt/models/tmp/hipcu'
```

Files: `bench/step0_run.sh` (guest-side runner: CU count + flock + rocprofv3 +
parse), `bench/step0_streaming_read.py` (kernel script; prints/asserts effective
config, doc rule 6), `bench/step0_parse.py` (trace parser; GB/s from kernel time
+ bytes moved, doc rule 2; prints kernel names actually dispatched),
`bench/step0_hip_cu.cu` (direct `hipGetDeviceProperties` CU query).

## 6. Exit condition (doc section 4 step 0)

Ceiling 636 GB/s ≥ ~550 GB/s → **no rescale flag**; operator may proceed to
step 1 (decode QMV kernel) with the doc's 640 GB/s-based targets. The 32-CU
HIP view is recorded for kernel-tuning heuristics (§1 note).
