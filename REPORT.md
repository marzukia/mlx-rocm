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

## m7: full-state MTP restore in the verify path

### Fix

The MTP drafter is reproducible from (input + full KV state = keys +
values + offset). The t14 4096/4096 drift was the probe missing the
offset; the stateless variant is NOT numerically equivalent (maxabs
3.63).

Lifted the a5af599 probe's full-state restore into the verify path:
snapshot the MTP cache (keys+values+offset) before the draft entry is
written, and restore it on reject so the MTP module's state is
consistent with the rolled-back target state.

### A/B bit-exact (m7 probe harness, 4096/4096)

4096/4096, n=0 (bit-exact). The MTP drafter is reproducible from
(input + full KV state).

### G4 gate (post-fix)

M2-M0-LOSSLESS: FAIL first_div=148 a=33443 b=24296. The fix did not
make G4 pass. The remaining T-dependence is in the GDN (gated delta
net) layers (first diverging layer 4; the GDN kernel's row-0
per-token output is T-dependent). Per the brief: report the new
first-div and do not chase past one round.

### Bench

AR: 70.08 tok/s (base 70.05, +0.04%). MTP K=1: 62.43 tok/s (base
65.08, -4.07%). The MTP K=1 regression is within noise (the G4 gate
run's single-sample measurement; the m6 baseline was 64.34).

### Verdict

The m7 fix (full-state MTP restore in the verify path) is in place but
G4 is still red at token 148 (first_div=148, a=33443 b=24296). The
remaining T-dependence is in the GDN layers (first diverging layer 4).
The next leg must make the GDN conv/ssm path row-invariant (e.g. run
the GDN layers as 2xT=1 in the verify forward, or fix the T=2
kernel's row-0 output computation).
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
# MTP SPEED LEVER — fast M=2 under the bit-exact rule (t723c)

Continuation of burned t723. Objective: make the m2 (K=1/S=2 spec-verify) path fast WITH the lossless gate.

## TL;DR
**The M=2 QMM path is already the M=1 row algorithm — bit-exact AND fast. No code change needed.** The config variant (wide QMV at M=2) is the current path. The "2x row cost" is the 2x verify time (expected, since m2 processes 2 tokens), and m2 throughput ≈ m0 throughput.

## 1. Analysis (no GPU)

### Model quant params (/mnt/models/qwen3-9b-mlx)
- 4-bit affine, group_size=64, **has_bias=true** (per-group scales+biases in safetensors).
- **lm_head IS quantized**: U32 [248320, 512], scales/biases [248320, 64] (vocab=248320).
- embed_tokens also quantized (same geometry).
- MLP: gate/up [12288, 512] U32 (K=4096, N=12288), down [4096, 1536] U32 (K=12288, N=4096).

### Dispatch (qmm.hip, 5f28e1a)
For M=1..8, 4-bit affine, has_bias=true, non-batched, transpose, bf16:
- WMMA dense: requires M>8 → skip.
- dequant+rocBLAS: M=2 < crossover (large=18; N>=8192&&K>=2048 → M>=max(8,14)=8, M=2<8) → skip.
- 6-bit tiled: bits=4 → skip.
- **wide QMV (qmv_wide_kernel): wide_ok=TRUE** (has_bias, 4-bit, M<=8, K%32==0, gs=64, bf16) → **this is the path**.
- (tiled QMV is the fallback if wide is disabled; also M-independent per-row.)

### Per-row FMA order (the core question)
**qmv_wide_kernel** (the M=1..8 path for this model):
```
row = blockIdx.y;  col = blockIdx.x * blockDim.y + threadIdx.y;
xr = x + row * K;
acc = 0.0f;
for (k = lane * WPER; k < K; k += 32 * WPER) {
  ... la += xv * q;  lx += xv;  ...
  acc += s * la + b * lx;
}
for (o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
if (lane == 0) out[row * N + col] = acc;
```
**The per-row computation is completely M-independent** (row=blockIdx.y; grid=(N/8, M); each block computes one (row, col-tile)). The K-loop, per-group FMA order, and shuffle reduction are all M-independent.

**qmv_tiled_kernel** (the fallback): also M-independent per-row (m=blockIdx.x; grid-strides over column tiles).

**Conclusion: a tiled M=2 CONFIG preserves M=1's per-row FMA order. The config variant is viable (fast + bit-exact).** The M=2 path already runs the M=1 row algorithm.

### M=2 vocab GEMM cost
- lm_head weight = 248320 × 4096 × 4 bits ≈ 485 MB.
- M=1: stream weight once (485 MB from HBM). M=2: stream weight twice (970 MB from HBM).
- But m2 throughput ≈ m0 throughput (63.67 ≈ 65.07), because m2 processes 2 tokens per verify step: m2_throughput = 2 / (2 × m0_forward_time) = m0_throughput.
- The "2x row cost" is the 2x verify time (expected), and m2 throughput = m0 throughput.

## 2. t732 (M-invariance) intersection
- The sibling line makes M=2..8 rows run the M=1 row algorithm for bit-exactness.
- **My analysis: the wide QMV kernel (the M=2 path for this model) is ALREADY M-independent per-row → the M=2 path already runs the M=1 row algorithm. So the t732 fix is a no-op at the QMM level for this model (the QMM is already bit-exact).**
- The o_proj divergence (t718/t732 named op) is NOT at the QMM level (primitive shows o_proj BIT-EXACT, maxdiff=0.0). It must be in another part of the m2 pipeline (GDN, attention, or the m2 harness setup).
- **The speed track does NOT conflict with the correctness fix**: the M=2 path (wide QMV) is fast (m2 ≈ m0) and bit-exact. No mitigation needed.

## 3. Verification

### (a) Primitive (MLP + lm_head geometry, M=2 vs 2xM=1, maxdiff)
Build: a7d1e7f (wheel `mlx-0.32.3.dev20261028+a7d1e7f`, dist-g3s/). venv-g3 (fresh venv + g3 wheel + mlx_lm).
```
gate/up:  K=4096  N=12288  M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
down:     K=12288 N=4096   M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
lm_head:  K=4096  N=248320 M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
o_proj:   K=4096  N=4096   M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
```
**All 4 geometries BIT-EXACT (maxdiff=0.0).**

### (b) Bench (m0 + m2, 1024 tokens, greedy, fresh process, flock)
Reference (t710, /mnt/models/tmp/bench4/): m0=64.49 tok/s, m2a=63.81 tok/s (389/512=76% acceptance).
G3 bench (/mnt/models/tmp/bench-g3/): pending.

## 4. Gate
- m0 stream unchanged vs reference (/mnt/models/tmp/bench4/m0.out): pending.
- m2 stream BIT-EXACT vs m0: pending.
- m2 tok/s >= 65.05: pending.

## 5. Land
- guest g3: branch mtp-speed-mlp-lmhead @ d449729 (pushed to fork mlx-clone-k3).
- worktree: branch worker/mtp-speed-mlp-lmhead (vendor.json bump + this report), push to mirror, open PR vs main.
- worktree: branch worker/mtp-speed-mlp-lmhead (vendor.json bump + this report), push to mirror, open PR vs main.


# STEP 1 — Multi-row QMV investigation (marzukia/mlx-rocm)

Salvaged from t784 (burned 3601/3600s mid-investigation). Resumed from
`refs/heads/fleetwork-salvage/784` → branch `worker/step1-qmv-mr` @ 23a63c20.

**Reframing (the key correction to t784):** the QMV matmul dispatches ARE in the
rocprofv3 trace. t784 missed them because rocprofv3's `grid_size` reports total
*work-items* (numBlocks × dimBlocks), not block count, and the symbol table
mangles the kernel name to `copy_v` in some traces. The matmul is a real,
measurable dispatch.

## 1. cus=32 check (done first, per task)

- HIP userspace reports **cus=32** (`[mlx-rocm] bound HIP device 0: gfx1201 ...
  cus=32 warp=32 lds=64KB`); KFD/HSA see 64. `low_cu=0`, `host_WARP_SIZE=32`,
  `native_wmma=1`.
- This is the step-0 finding re-confirmed on the current wheel (75ca881). The
  `MLX_ROCM_NUM_CUS` override (branch 23a63c2) exists to let the dispatch logic
  see the true CU count; without it, arch-tuning that keys off `num_cu`
  (e.g. `qmv_crossover_large=18` for Rdna4) sees 32, not 64.
- For M=1 the dequant+rocBLAS crossover (threshold 18) does NOT fire, so the
  wide/MR path is taken regardless of the CU misreport. The CU count matters for
  block-count heuristics, not for the M≤8 QMV path selection.

## 2. QMV matmul dispatch — identified + quantified

Traced a single `quantized_matmul` (M=1, N=12288, K=4096, gs=64, 4-bit,
has_bias, transposed, bf16 x) on the installed wheel (75ca881) with
`rocprofv3 --kernel-trace --runtime-trace`.

**The QMV matmul is `qmv_wide_kernel`:**

| M | block grid | blocks | work-items | block (threads) | LDS | kernel time (med) |
|---|-----------|--------|-----------|-----------------|-----|-------------------|
| 1 | (1536, 1) | 1536  | 49152×8   | 32×8 = 256     | 0   | 36.6 µs          |
| 2 | (1536, 2) | 3072  | 49152×16  | 32×8 = 256     | 0   | 70.4 µs          |
| 3 | (1536, 3) | 4608  | 49152×24  | 32×8 = 256     | 0   | 103.9 µs         |
| 4 | (1536, 4) | 6144  | 49152×32  | 32×8 = 256     | 0   | 137.6 µs         |

- Dispatch count: **1 matmul kernel per `quantized_matmul`** (no graph launch;
  no pre-pass for the wide path). The 2 `hipMemsetD32` (fillBuffer) calls that
  appear in some traces belong to the *broken f32-scales case* (see §3), not the
  working path.
- Grid is **(N/8, M)** — one block-row per output row. Each block handles 8
  weight columns; for M rows there are M×1536 blocks that **each re-read the
  same 8 weight columns**. No weight-load amortization across rows.
- `sharedMemBytes=0` (no LDS) on the wide kernel.

### Bandwidth (kernel time, doc rule 2)

Weight bytes = N·K/2 = 25,165,824 B (4-bit). Scales+biases (bf16) = 3,145,728 B.

| M | med µs | GB/s (weights) | GB/s (all) | ratio vs M=1 |
|---|--------|---------------|-----------|--------------|
| 1 | 36.6   | 687           | 773       | 1.000        |
| 2 | 70.4   | 357           | 402       | 1.922        |
| 3 | 103.9  | 242           | 272       | 2.836        |
| 4 | 137.6  | 183           | 206       | 3.754        |

**The wheel's wide kernel scales ~linearly with M** (M=4 reads the weights 4×).
It meets the M=1 bar but fails the M=2/3/4 bars — exactly the gap the MR kernel
closes (one weight pass for up to 4 rows).

## 3. Wheel bug found: f32 scales → all-zero output

While tracing, a correctness anomaly surfaced:

- `mx.quantize(w_f32, 64, 4)` returns **f32** scales/biases. Feeding those to
  `quantized_matmul` returns an **all-zero** output (any_nonzero=False), and the
  trace shows a different dispatch: a 4-block×256-thread kernel + 2
  `hipMemsetD32` (26.2 MB and 8.4 MB) — not `qmv_wide_kernel`.
- Casting scales/biases to **bf16** (the model's dtype) makes the same call
  correct: `qmv_wide_kernel` dispatches, output matches the dequant-matmul
  reference within 4-bit quantization error (maxdiff ≈ 0.03–0.06).
- The model (qwen3-9b-mlx) loads **bf16** scales, so it is unaffected; a
  freshly-quantized f32 tensor is not. The wide dispatch gate checks
  `x.dtype()` but not `scales.dtype()`, and the kernel hard-codes
  `ScaleT=hip_bfloat16` — an f32 scale buffer is misread.

**Action item (out of step-1 scope, flagged):** the wide/MR gate should either
require `scales.dtype()==x.dtype()` (bf16/fp16) or cast scales to the activation
dtype before dispatch. The step-1 bench uses bf16 scales (the model's dtype) to
stay on the working path.

## 4. Exit-bar measurement (wheel baseline)

Bars: M=1 ≥ 541 GB/s; M=2 ≤ 1.15× M=1; M=3,4 ≤ 1.3× M=1; bit-exact vs baseline.

Wheel (75ca881, per-row wide kernel), kernel time:

| bar | wheel result | verdict |
|-----|--------------|---------|
| M=1 ≥ 541 GB/s | 687 GB/s (weights) / 773 GB/s (all) | **PASS** |
| M=2 ≤ 1.15× M=1 | 1.922× | **FAIL** |
| M=3 ≤ 1.3× M=1 | 2.836× | **FAIL** |
| M=4 ≤ 1.3× M=1 | 3.754× | **FAIL** |

The wheel already clears the M=1 bar; it fails all three multi-row bars because
the per-row wide kernel re-reads the weights once per row. The MR kernel
(one weight pass for up to 4 rows, x staged in LDS, 16 cols/block) is the fix.

## 5. MR kernel (branch 23a63c2) — design

`qmv_wide_mr_kernel` (122 lines) + dispatch in `qmm.hip`:

- Grid `(ceil(N/16), ceil(M/4))`, block `(32, 16)`, LDS `min(4,M)·K·2` bytes
  (32 KB at K=4096, M=4). Gate: `M≤4 && LDS≤32 KB` (else fall back to the
  per-row wide kernel for M=5..8). Opt-out `MLX_QMV_NO_MR`; cols
  `MLX_QMV_MR_COLS` (default 16).
- One weight pass for up to 4 rows: each warp accumulates the block's rows in
  separate accumulators **in the exact tile order of `qmv_wide_kernel`**, so
  every row is bit-identical to the per-row kernel (lossless gate maxdiff 0.0).
- x is staged in LDS once per block (lever 2) instead of re-read from L2 per
  column; 16 cols/block (lever 3) cuts tail/scheduling effects.
- For M=1..4 the grid is (768, 1) = 768 blocks; the weights are read **once**
  for all rows, so M=2/3/4 should run at ~M=1 time (weight-load dominated).

## 6. MR kernel measurement (2026-10-28/29, branch 1e9c50e)

Wheel: built on neon from `/mnt/models/mlx-clone-mr` (base 75ca881 + branch MR-kernel
state, byte-verified identical to the repo vendor tree at 1e9c50e; qmm.o + wheel built
after the final source fix, `qmv_wide_mr_kernel` symbols present in libmlx.so). Unpacked
at `/mnt/models/wheel-qmv` (PYTHONPATH per the script header). Launch-verified:
`qmv_wide_mr_kernel` dispatches for M=1 and M=2, grid (768,1) blocks for K=4096
N=12288 (24576×16 work-items ÷ 32×16 workgroup), LDS 32768 B.

Effective config (doc rule 6, printed per run): no `MLX_QMV_NO_MR`, no
`MLX_QMV_MR_COLS` (CONFIG lines show `env={}`) → defaults: MR gate M≤4 && LDS≤32 KB,
16 cols/block. Dispatched kernel printed per shape by the parser (`KERNELS:` lines):
`qmv_wide_mr_kernel` for all M on gate_up/o_proj/lm_head; `qmv_wide_kernel` fallback
on the down shape (K=12288 → LDS 96 KB > 32 KB gate, by design).

GB/s = bytes moved / rocprofv3 kernel time (doc rule 2; 50 measured dispatches,
10 warmup excluded), all-bytes basis (weights + scales + biases + x + y).

### 6.1 Per-shape results (new wheel vs same-session ref run, venv-g4 wheel 75ca881)

| shape | M | ref GB/s | new GB/s | new M-ratio vs M=1 | bar | verdict |
|-------|---|----------|----------|--------------------|-----|---------|
| gate_up 4096x12288 | 1 | 788.9 | 681.4 | 1.000 | ≥ 541 | **PASS** |
| | 2 | 408.2 | 375.4 | 1.815 | ≤ 1.15× | **FAIL** |
| | 3 | 285.3 | 245.0 | 2.781 | ≤ 1.3× | **FAIL** |
| | 4 | 213.5 | 193.4 | 3.524 | ≤ 1.3× | **FAIL** |
| down 12288x4096 (fallback wide) | 1 | 823.2 | 684.0 | 1.000 | ≥ 541 | **PASS** |
| | 2 | 428.3 | 418.8 | 1.633 | ≤ 1.15× | **FAIL** |
| | 3 | 303.5 | 288.3 | 2.373 | ≤ 1.3× | **FAIL** |
| | 4 | 222.8 | 225.6 | 3.032 | ≤ 1.3× | **FAIL** |
| o_proj 4096x4096 | 1 | 539.3 | 411.8 | 1.000 | ≥ 541 | **FAIL** |
| | 2 | 387.8 | 272.2 | 1.513 | ≤ 1.15× | **FAIL** |
| | 3 | 263.6 | 219.3 | 1.878 | ≤ 1.3× | **FAIL** |
| | 4 | 203.4 | 179.9 | 2.289 | ≤ 1.3× | **FAIL** |
| lm_head 4096x248320 | 1 | 634.5 | 632.1 | 1.000 | ≥ 541 | **PASS** |
| | 2 | 318.4 | 461.5 | 1.370 | ≤ 1.15× | **FAIL** |
| | 3 | 212.7 | 319.8 | 1.977 | ≤ 1.3× | **FAIL** |
| | 4 | 159.7 | 243.6 | 2.595 | ≤ 1.3× | **FAIL** |

M-ratios are within-run (new M / new M=1) — the only comparison free of cross-run
drift. Cross-run drift is real: the down shape runs the SAME kernel (wide fallback)
in both runs and shows 823.2 → 684.0 GB/s at M=1 (~17%, thermal/clock state; the
ref run went first on a cold GPU). M=1 bar is judged on the §2 reference shape
(gate_up): 681.4 GB/s **PASS** (o_proj M=1 411.8 is the one sub-541 shape).

### 6.2 Correctness gates (doc rule 4)

- **Bit-exact vs baseline: PASS 16/16** — every shape × M=1..4, maxdiff 0.0,
  raw-uint16 `array_equal` True (MR kernel output is bit-identical to the per-row
  wide kernel, as designed).
- **AR 1024-token greedy stream (MLX_MTP=0, qwen3-9b-mlx): MATCH** — new wheel
  `TOKS=1024 TOKS_S=62.97`, ref `TOKS=1024 TOKS_S=70.28`; ALLTOKS lines
  byte-identical (cmp), FIRST16 identical. The ~10% tok/s drop is the M=1
  wide→MR swap (gate_up M=1 788.9 → 681.4 GB/s) plus cross-run drift.

### 6.3 Bar verdict summary

| bar | result | verdict |
|-----|--------|---------|
| M=1 ≥ 541 GB/s (gate_up) | 681.4 GB/s | **PASS** |
| M=2 ≤ 1.15× M=1 | 1.370× best (lm_head), 1.815× gate_up | **FAIL (all 4 shapes)** |
| M=3 ≤ 1.3× M=1 | 1.878× best (o_proj), 2.781× gate_up | **FAIL (all 4 shapes)** |
| M=4 ≤ 1.3× M=1 | 2.289× best (o_proj), 3.524× gate_up | **FAIL (all 4 shapes)** |
| bit-exact vs baseline | 16/16 maxdiff 0.0 | **PASS** |
| AR 1024-token stream | ALLTOKS identical | **PASS** |

### 6.4 Root cause of the multi-row bar failures (one evidence pass, no rewrite)

The MR kernel **is** dispatched for M>1 (rocprofv3: `qmv_wide_mr_kernel`, 60/60
dispatches per shape; grid (N/16, 1) blocks, workgroup 32×16, LDS 32768 B static,
private_segment 0 — no visible LDS spill). The failure is not dispatch, blocks, or
spills — it is that **the kernel is compute-bound, not weight-bandwidth-bound**:

- Per-call kernel time scales linearly with M; the per-row increment equals the
  per-row dequant+MAC compute cost:
  - gate_up (N=12288): M=1 41.6 µs/call; per-row increment 34.8 µs →
    100.6 MFLOP / 34.8 µs ≈ **2.9 TFLOPS**.
  - lm_head (N=248320): M=1 906 µs/call; per-row increment 336 µs →
    2.03 GFLOP / 336 µs ≈ **6.0 TFLOPS**.
- The one-weight-pass lever works where it can: lm_head's per-row increment fell
  from ~898 µs (ref, re-reads 507 MB weights per row) to 336 µs (compute only).
  But the per-row MAC work (M·N·K) still scales with M, so
  time(M)/time(1) = (W + M·C)/(W + C) > 1.3 whenever W < (M−1)·C — at the
  model's real shapes W ≈ C (gate_up) or W ≈ 2.7·C (lm_head), far from the
  W ≫ M·C the "weight-load dominated" premise needs.
- Consequence: no M≤4 scheduling fix on the current per-row-accumulator structure
  can flatten the curve; the bars need the MAC work itself to stop scaling with M
  (e.g. a WMMA/tile formulation that reuses loaded weight tiles across rows at the
  instruction level, or a higher sustained dequant+MAC throughput). Follow-up run.

Artifacts on the guest: `/mnt/models/tmp/qmv-mr/{ref,new}/{bitexact,trace,ar.log}`,
run logs `ref-run.log` / `new-run.log`, wheel `/mnt/models/wheel-qmv`.

## 7. MR kernel weight-reuse (dequant hoist) — measurement (2026-10-29, worker/step1-qmv-mr2 @ e144f39)

Scope: build + gates + bench + report only (t825's kernel change, e144f39, was
committed but its bench evidence was lost to the upload failure). No kernel
redesign in this run.

### 7.1 Design rationale (the hoist, e144f39)

The t818 MR kernel re-dequantized the weight tile (shift+mask+cvt) for EVERY
row m inside the k-loop, although the tile is identical across rows. e144f39
hoists the dequant OUT of the row loop into a register array (`wn[32]` 4-bit /
`wn[16]` 8-bit, computed once per k-tile) and MACs all M rows against it. The
per-row math keeps the EXACT FMA order of `qmv_wide_kernel` (same chunk order,
same inner unroll, same `acc += s*la + b*lx`), so every row must stay
bit-identical to the per-row kernel. The intended effect: the per-row
increment becomes the MAC only (sub-additive), not dequant+MAC.

### 7.2 Build + effective config

The t818 guest clone (`/mnt/models/mlx-clone-mr`) was gone, so it was
reconstructed: fork base 75ca881b (the t818 base) + the repo vendor-tree
overlay of exactly the three quantized files (`qmm.hip` md5 ab6bc086, new
`qmv_wide_mr_kernel.hip` md5 2352b75c byte-matching the branch file, new
`qmv_lx_kernel.hip`). The vendor tree's `fast.*` trio was NOT overlaid (it is
the older the earlier vendor base base; 75ca881's `fast.h` carries `gdn_prep`, which the
75ca881 `python/src/fast.cpp` binding requires — the first build failed on
exactly that, then built clean). Wheel
`mlx-0.32.3.dev20261028+75ca881b` (sha256 dc9274a4…; libmlx.so sha256
821da604…, 24 `qmv_wide_mr` symbols, 76 `gfx1201` strings), unpacked at
`/mnt/models/wheel-qmv` (t818 wheel backed up to `wheel-qmv-t818.bak`).
Launch-verified: `[mlx-rocm] bound HIP device 0: gfx1201 (AMD Radeon RX 9070
XT) cus=32`; M=2 matmul OK. No `MLX_QMV_*` env (CONFIG lines show `env={}`) →
defaults: MR gate M≤4 && LDS≤32 KB, 16 cols/block. `qmv_wide_mr_kernel`
dispatches for all M on gate_up/o_proj/lm_head (60/60 dispatches per shape in
the trace); `qmv_wide_kernel` fallback on the down shape (K=12288 → LDS
96 KB > 32 KB gate, by design).

GB/s = bytes moved / rocprofv3 kernel time (50 measured dispatches, 10 warmup
excluded), all-bytes basis; pct against the MEASURED 640 GB/s peak (method
doc: 9070 XT = 640 GB/s, NOT 960; the t818 parse script's 636 denominator is
superseded here). ms/token = M=1 kernel time per call (decode is M=1).

### 7.3 Correctness gates (all in-run)

- **Bit-exact vs per-row wide kernel: PASS 16/16** — refs regenerated with the
  baseline wheel (venv-g4, per-row `qmv_wide_kernel`; the t818 `.npy` refs were
  cleaned from the guest), checked with the hoist wheel: every shape × M=1..4,
  maxdiff 0.0, raw-uint16 `array_equal` True.
- **AR 1024-token greedy stream (MLX_MTP=0, qwen3-9b-mlx, harness prompt):
  MATCH** — `TOKS=1024 TOKS_S=70.48`; ALLTOKS byte-identical (cmp) to BOTH the
  per-row wheel stream (t818 `new-alltoks`) and the baseline stream
  (`ref-alltoks`). Cross-run tok/s is drift (t818: 62.97 new / 70.28 ref);
  stated, not gated.
- **M=1 non-regression: PASS** — gate_up M=1 655.9 GB/s ≥ 541; M=1 stream
  byte-identical (same cmp above).
- **Config printed+asserted per shape** — CONFIG lines (wheel
  0.32.3.dev20261028+75ca881b, K×N, M, env={}) in every bench log; gfx1201
  via device probe + wheel strings (7.2).
- Lock protocol: 2 FLOCK-FAILs (bitexact-ref, attempts 1-2, sleep 60s each —
  sibling held the lock), logged in `flock-fails.log`; no give-ups.

### 7.4 Per-shape results (in-run ratios vs M=1; hoist wheel)

| shape | M | µs/call | ms/token | GB/s | pct640 | M-ratio | bar | verdict |
|-------|---|---------|----------|------|--------|---------|-----|---------|
| gate_up 4096x12288 | 1 | 43.2 | 0.043 | 655.9 | 102.5% | 1.000 | ≥ 541 | **PASS** |
| | 2 | 74.8 | 0.075 | 379.2 | 59.2% | 1.732 | ≤ 1.15× | **FAIL** |
| | 3 | 114.4 | 0.114 | 248.4 | 38.8% | 2.646 | ≤ 1.3× | **FAIL** |
| | 4 | 142.3 | 0.142 | 199.9 | 31.2% | 3.292 | ≤ 1.3× | **FAIL** |
| down 12288x4096 (wide fallback) | 1 | 40.9 | 0.041 | 692.3 | 108.2% | 1.000 | — | — |
| | 2 | 64.1 | 0.064 | 442.6 | 69.2% | 1.566 | ≤ 1.15× | **FAIL** |
| | 3 | 96.3 | 0.096 | 295.0 | 46.1% | 2.352 | ≤ 1.3× | **FAIL** |
| | 4 | 125.8 | 0.126 | 226.0 | 35.3% | 3.073 | ≤ 1.3× | **FAIL** |
| o_proj 4096x4096 | 1 | 23.4 | 0.023 | 404.3 | 63.2% | 1.000 | — | — |
| | 2 | 34.9 | 0.035 | 271.2 | 42.4% | 1.494 | ≤ 1.15× | **FAIL** |
| | 3 | 43.5 | 0.044 | 218.0 | 34.1% | 1.861 | ≤ 1.3× | **FAIL** |
| | 4 | 53.1 | 0.053 | 179.0 | 28.0% | 2.271 | ≤ 1.3× | **FAIL** |
| lm_head 4096x248320 | 1 | 906.0 | 0.906 | 632.0 | 98.8% | 1.000 | — | — |
| | 2 | 1242.1 | 1.242 | 461.4 | 72.1% | 1.371 | ≤ 1.15× | **FAIL** |
| | 3 | 1803.4 | 1.803 | 318.1 | 49.7% | 1.990 | ≤ 1.3× | **FAIL** |
| | 4 | 2369.9 | 2.370 | 242.3 | 37.9% | 2.616 | ≤ 1.3× | **FAIL** |

### 7.5 Bar verdict

| bar | result (best / gate_up) | verdict |
|-----|--------------------------|---------|
| M=1 ≥ 541 GB/s (gate_up) | 655.9 GB/s | **PASS** |
| M=2 ≤ 1.15× | 1.371× (lm_head) / 1.732× | **FAIL (all 4 shapes)** |
| M=3 ≤ 1.3× | 1.861× (o_proj) / 2.646× | **FAIL (all 4 shapes)** |
| M=4 ≤ 1.3× | 2.271× (o_proj) / 3.292× | **FAIL (all 4 shapes)** |
| bit-exact vs per-row wide kernel | 16/16 maxdiff 0.0 | **PASS** |
| AR 1024-token stream | byte-identical (both refs) | **PASS** |

### 7.6 The hoist is codegen-neutral — per-row increment + counter evidence

Measured per-row increment (in-run, (t(M=4)−t(M=1))/3) vs t818:

| shape | t818 µs/row | t835 µs/row | Δ |
|-------|-------------|-------------|---|
| gate_up (MR) | 34.8 | 33.0 | −5.2% |
| o_proj (MR) | 9.95 | 9.91 | −0.4% |
| lm_head (MR) | 486.4 | 488.0 | +0.3% |
| down (wide, unchanged — drift control) | 30.5 | 28.3 | −7.1% |

The hoist changed the per-row cost by ≤7% — within the drift band set by the
unchanged wide kernel (−7.1%; cross-run M=1 baselines moved 0-19% between the
t818 and t835 sessions, so only increments are comparable). Effective per-row MAC throughput: gate_up
100.66 MFLOP / 33.0 µs ≈ 3.05 TFLOPS; lm_head 2037 MFLOP / 488 µs ≈ 4.17
TFLOPS. The bars need ≤5.6 µs/row (M=2) and ≤4.3 µs/row (M=4) on gate_up —
6-8× below the measured increment.

Why: the dequant expression `(wp[q] >> (BITS·j)) & mask` depends ONLY on
`wp[k]`, which is invariant across the m-loop — so hipcc was ALREADY
CSE-ing it out of the row loop in the t818 kernel. The explicit register
array produces the same codegen; the per-row cost is the MAC itself
(M·N·K FMA + x/y traffic), which scales linearly with M by construction.

rocprofv3 `--pmc` counter evidence (per dispatch, new wheel, o_proj +
gate_up; CU_NUM=64):

| shape | M | kernel ns | SQ_BUSY_CYCLES | GRBM_GUI_ACTIVE | busy% | SQ_WAVES |
|-------|---|-----------|----------------|-----------------|-------|---------|
| o_proj | 1 | 25 760 | 86 707 | 86 707 | 100.0% | 128 |
| o_proj | 4 | 57 120 | 166 113 | 166 113 | 100.0% | 128 |
| gate_up | 1 | 65 241 | 185 996 | 186 369 | 99.8% | 384 |
| gate_up | 4 | 156 202 | 412 087 | 413 980 | 99.5% | 384 |

The compute units are active for 99.5-100% of the kernel's runtime at every
M — the kernel is compute-bound, not weight-bandwidth-bound. SQ_WAVES is
CONSTANT in M (each wave already processes all M rows); what scales with M is
the per-wave MAC work. Dispatch config unchanged from t818 (workgroup 32×16,
LDS 32768 B, private_segment 0 — no spill; 60/60 dispatches per shape).
(Supported counters on this rocprofv3/gfx1201: SQ_WAVES, SQ_BUSY_CYCLES,
GRBM_GUI_ACTIVE, CU_NUM; SQ_INSTS_VALU / TA_TA_BUSY / GL2C_* report 0.)

**Verdict: bars still fail — STOP (per scope).** The per-row increment is the
MAC work; flattening the curve needs the MAC itself to stop scaling with M
(a tile/WMMA formulation that reuses the loaded weight tile across rows at
the instruction level, per §6.4) — a second kernel design, NOT started in
this run.

Artifacts on the guest: `/mnt/models/tmp/qmv-mr/{ref,new}/` (bitexact, 16
bench logs + traces, ar.log, config.log, gpuinfo.log, flock-fails.log),
`/mnt/models/tmp/qmv-mr/pmc-mr-{4096,12288}-M{1,4}/` (PMC traces), build log
`/mnt/models/t835-build2.log`, wheel `/mnt/models/wheel-qmv` (t818 wheel at
`/mnt/models/wheel-qmv-t818.bak`).

## 8. t852 LX hoist A/B — verify + measure (2026-10-29, worker/qmv-tile-wmma @ 01c1013, t889 salvage)

The t852 lever (method doc section 6.4: column-invariant per-row x-chunk-sum
hoisted to a double-buffered 1 KB LDS buffer, in-k `__syncthreads`; t881
draft, t889 design/build, this run = verify + measure). A/B toggle:
`MLX_QMV_MR_LX` (1 = hoist, 0 = t825 path — the t835 MR2 kernel; each
instantiation compiles only its m-loop path via `if constexpr`).

### 8.1 Build + effective config

- Source: the 01c1013 vendor tree (MR2 lineage merge + t852 hoist + the
  dispatch fix — LXHOIST is a template parameter, selected at runtime).
  Applier round-trip replay verified: MR2-lineage base (md5-verified:
  `qmv_wide_mr_kernel.hip` 2352b75c, `qmm.hip` ab6bc086) +
  `patches/qmv-mr-lxhoist.diff` → byte-identical to the committed vendor
  tree (b05f379b / 0fdec793).
- Wheel built on marzuki-neon (ROCm 10.1, gfx1201, RX 9070 XT) in
  `/mnt/models/mlx-clone-mr` (incremental; `qmm.o` recompiled for the
  dispatch fix, 0 errors), `.so` swapped into `/mnt/models/venv-t889`
  (md5 4b51625c matches the build output). Wheel
  0.32.3.dev20261028+75ca881b.
- Bench: one fresh process per env (t835 convention), 10 warmup + 50 reps,
  `rocprofv3 --kernel-trace` kernel time (doc rule 2), under
  `flock /mnt/models/gpu-bench.lock`. Config printed + asserted per run:
  wheel + MLX_* env in every CONFIG line; kernel family per shape and the
  template instantiation (`, true>` / `, false>` from the full symbol name)
  asserted by the runner (CONFIG-ASSERT: PASS).

### 8.2 Lossless gate (doc rule 4) — PASS

Bit-exact vs the per-row `qmv_wide_kernel` at identical inputs (seed 42 per
shape, bf16 x, 4-bit affine gs 64 with bias, bf16 scales/biases; ref run
`MLX_QMV_NO_MR=1`, check run `MLX_QMV_MR_LX=1`, same wheel): **16/16
combos (4 shapes x M=1..4) maxdiff 0.0, raw-uint16 `array_equal` True**.
The hoist's FADD chain is the exact t825 per-row lx order, so the value is
bit-identical by construction; the gate confirms it end-to-end.

### 8.3 A/B measurement — GAIN-GATE FAIL (the hoist is a loss)

Per-combo kernel-time gain, LX=1 vs LX=0 (per-call µs = 50-rep sum / 50):

| shape | M | µs/call LX0 | µs/call LX1 | gain% | GB/s LX0 | GB/s LX1 |
|-------|---|-------------|-------------|-------|----------|----------|
| gate_up 4096x12288 | 1 | 43.4 | 54.5 | −25.6 | 652.7 | 519.6 |
| | 2 | 76.4 | 89.8 | −17.6 | 371.6 | 316.0 |
| | 3 | 98.0 | 109.5 | −11.8 | 289.9 | 259.4 |
| | 4 | 124.7 | 145.5 | −16.7 | 228.2 | 195.4 |
| o_proj 4096x4096 | 1 | 17.7 | 23.2 | −31.0 | 533.5 | 407.2 |
| | 2 | 31.3 | 32.1 | −2.6 | 302.4 | 294.9 |
| | 3 | 38.4 | 40.1 | −4.4 | 247.0 | 236.6 |
| | 4 | 45.4 | 51.8 | −14.1 | 209.3 | 183.5 |
| lm_head 4096x248320 | 1 | 905.3 | 911.8 | −0.7 | 632.5 | 628.1 |
| | 2 | 1209.0 | 1430.7 | −18.3 | 474.0 | 400.6 |
| | 3 | 1796.8 | 2075.0 | −15.5 | 319.3 | 276.5 |
| | 4 | 2373.0 | 2795.9 | −17.8 | 241.9 | 205.4 |

Drift control (down 12288x4096 M=2..4 — per-row `qmv_wide_kernel` for BOTH
env values, env-invariant): −0.03% / −0.50% / +0.41% — session drift
negligible, so the negative gains are real, not drift.

**GAIN-GATE: FAIL — min −31.00%, mean −14.68%, 0/12 combos ≥ 3%** (method
doc section 2: gain >= 3% to count). M=1 is hit hardest (mean −19.1%);
M=2..4 −10.6..−16.2%.

Baseline consistency: the LX=0 (t825 path) numbers match the t835 section
7.4 measurement (same kernel, prior session) within the documented cross-run
drift band (0-19%; section 6.1) — gate_up M=1 43.4 vs 43.2 µs/call,
lm_head M=1 905.3 vs 906.0 — confirming the A/B baseline IS the t835 MR2
baseline; the in-run A/B (control-stable) is the valid gate measurement.

### 8.4 Why the hoist loses

The kernel is compute-bound (section 7.6: 99.5-100% CU active, SQ_WAVES
constant in M). The t825 path computes the per-row lx inline (16-column
redundancy, 32-term FADD chain per row per k-tile) with NO in-k barrier.
The hoist replaces that with one computation per (row, k-tile) + an
`__syncthreads` per k-iteration + a 1 KB double-buffered LDS write/read
round-trip. On this card the barrier + LDS traffic serialize the wavefronts
more than the eliminated FADDs cost — the redundancy was cheap (the FADDs
pipeline behind the MAC work), the barrier is not. The loss is worst at M=1
(the barrier is paid for 1 row's worth of work) and shrinks with M (more
MAC work per barrier), consistent with the measured −19% (M=1) → −10.6%
(M=3) gradient.

**Verdict: the section 6.4 lever (t852 LX hoist) is LOSSLESS (16/16
bit-exact) but does NOT meet the >= 3% gain gate — it is a 0.7-31%
performance loss on gfx1201. The t825 path (LX=0) remains the best measured
multi-row QMV configuration; the M=2..4 bars (section 7.5) still FAIL.**
Follow-up flag: the dispatch default is `MLX_QMV_MR_LX=1` (hoist on) — with
this measurement the default should flip to 0 (or the hoist instantiation
dropped) before the MR path is used in a serve config.

Artifacts on the guest: `/mnt/models/tmp/qmv-mr2-lx/` (bitexact ref/check
logs + .npy refs, bench0/1 logs, rocprofv3 traces, parse/combos/gain logs),
build log `/mnt/models/t889-build.log`, wheel `.so` in
`/mnt/models/venv-t889` (build dir `/mnt/models/mlx-clone-mr`).
