# K2 non-GEMM fusion — round-4 fix report (submitter)

Branch: `k2-non-gemm-fusion` (base `33b13dd`). Fixes the round-3 adversarial
review (an earlier run, `reviews/k2-r3-review.md`): 0 BLOCKER / 3 MAJOR / 2 MINOR / 3 NIT.
Scope of this round: the 3 MAJORs (one-line guards in `fast.cpp`) + MINOR-1
(report reconciliation). MINOR-2 (fast.h wording) and NITs are left open
(see §7).

## 1. The fixes

All in `vendor/mlx-rocm-upstream/mlx/fast.cpp` (the kv_append gate). The
verified wheel (an earlier run, source `ec48eb4c`, `/models/mlx-clone-k2`) carries
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
against the wheel source (`/models/mlx-clone-k2/mlx/fast.cpp`,
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

## 2. Rebuild (guest, /models/mlx-clone-k2)

fast.cpp synced (md5-verified), then in `/models/venv-k2` PATH:
`cmake -S . -B build/temp.../mlx.core -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1
-DBLA_VENDOR=OpenBLAS -DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
-DCMAKE_BUILD_TYPE=Release` (reconfigure, 9.3s) → `cmake --build ... -j8`
(fast.cpp recompiled, libmlx.so relinked, python extension rebuilt) →
`pip wheel . --no-deps -w /models/dist/ --no-build-isolation` →
force-reinstall into venv-k2. Log: `/models/tmp/k2-r4-build.log`.

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
`/models/venv`, so the reinstall used `venv-k2/bin/python -m pip` — the
first `pip` invocation installed into the base venv; corrected before any
GPU run.)

## 3. Suite result (this wheel, log preserved on disk)

`scripts/verify-k2.py` (69 checks), venv-k2, `flock /models/gpu-bench.lock`,
`LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`. Full log:
**`reports/k2-r4-suite-full.log`** (79 lines, preserved in-repo).

**Verdict: 69 PASS / 0 FAIL** (`grep -c '^\[PASS\]'` = 69,
`grep -c '^\[FAIL\]'` = 0, "all unit checks passed"). No regression vs the
an earlier run run (69/0 on `ec48eb4c`): the new checks only fire on invalid input,
and the unscheduled guard routes the same lazy suite inputs to the same
slice_update fallback the an earlier run wheel used.

This also resolves MINOR-1: the 69/0 evidence is now substantiated on disk —
`reports/k2-r4-suite-full.log` is the full raw log of the suite run on the
branch-as-committed wheel (`d617a1b8`), and `reports/k2-r2-suite-evidence.txt`
retains the an earlier run session extraction for the `ec48eb4c` run.
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

# an earlier run — vendor-pin fix (dangling pin + silent build-wheel) — worker report

Branch: `worker/an earlier run-vendor-pin-fix` (base `origin/main` @ 64b98b8).
Fixes the an earlier run review-FAIL finding on PR #10 (96a1677 "fix(rocm): GDN fused
decode T=1 launcher pointer write"): `vendor.json` pinned
`75ca881b2e82f7b4069c8564a203792aa16d8a29`, but the fork branch
`the vendor fork:tail-fusion-gdn` was at `96a1677` (the fix's parent) —
`75ca881` existed only in a guest build clone and was never pushed, and
`scripts/build-wheel.sh` never verified the checkout against the pin.

## 1. Push the pin (part 1)

- `75ca881` recovered from the an earlier run build clone (`/models/mlx-clone-g4`;
  the task text said g1, but g4 is the clone that carries it — g1's HEAD is
  `1641333`). Fetched into a local repo, then:

  ```
  git push --force-with-lease origin 75ca881:refs/heads/tail-fusion-gdn
  #   To github.com:the vendor fork.git
  #    + 96a1677...75ca881 75ca881 -> tail-fusion-gdn (forced)
  ```

  Fast-forward on the branch (96a1677 is 75ca881's parent; the push is a
  2-commit advance, no history rewrite of the pinned commit).

- Verification (all three agree):
  - `git ls-remote https://github.com/the vendor fork.git refs/heads/tail-fusion-gdn`
    → `75ca881b2e82f7b4069c8564a203792aa16d8a29`
  - `gh api repos/the vendor fork/commits/75ca881b2e82f7b4069c8564a203792aa16d8a29`
    → 200, `sha=75ca881b...`, `commit.message="fix(rocm): GDN fused decode T=1 launcher pointer write"`,
    parent `96a16773a1133795406a85545b654b6972d93721`
  - `gh api repos/the vendor fork/branches/tail-fusion-gdn` →
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

Built on the build VM (guest, ROCm 10.1, gfx1201) under `flock /models/gpu-bench.lock`:

- Fresh clone of `the vendor fork` @ `tail-fusion-gdn` →
  `/models/mlx-clone-an earlier run`; `git rev-parse HEAD` =
  `75ca881b2e82f7b4069c8564a203792aa16d8a29` (the pinned commit, from the
  fork — not the guest build clone).
- `MLX_VENDOR_DIR=/models/mlx-clone-an earlier run bash scripts/build-wheel.sh`
  (with the new verification — it printed `vendor checkout verified` and
  proceeded) → `/models/dist-an earlier run/mlx-0.32.3.dev20261028+75ca881-cp312-...whl`.
- Installed into a scratch venv (`/models/venv-an earlier run`, python 3.12.3;
  `venv-d` untouched). The scratch venv mirrors venv-d's mlx_lm model
  files: `qwen3_5.py` (S==1-guarded fused branch) **and** `gated_delta.py`
  (K3 patch: non-fused GDN update routes through
  `mx.fast.gated_delta_update` on ROCm). The first gate round ran with the
  stock PyPI `gated_delta.py` (Python op loop) and G2 diverged at token 33 —
  an environment artifact, not a wheel difference: the m0 reference and the
  an earlier run AR baseline were both produced with the K3-patched wrapper. With the
  K3 file in place, all gates below pass.

### Rule-6 effective config (printed + asserted by `scripts/wheel-gates_config.py`)

```
an earlier run-config: OK
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

Same model/prompt/seed as the an earlier run reference (bench4 `m0.out`, 1024 greedy
tokens), vacuity-guarded stream compare:

```
G2-FUSED-VS-M0-REF:    STATUS=MATCH (maxdiff 0.0) first_div=- match=1024/1024
G2-FUSED-VS-UNFUSED:  STATUS=MATCH (maxdiff 0.0) first_div=- match=1024/1024
GATE-LIVE (all compared streams non-empty)
```

The fused stream is bit-identical to both the 4bit-MLX (m0) reference and
the same-wheel unfused stream. Cross-check: the an earlier run AR baseline
(`bench-decode/m0a.out`, 75ca881 wheel, venv-d) is byte-identical to m0
(diff 0/1024), so the reference itself is stable across the two 75ca881
builds.

### an earlier run primitive gates (re-run on this wheel): PASS

```
PRIM-GATES-PASS
FWD-LAYER maxdiff=0.000000e+00            (fused T=1 layer fwd, 5-layer chain)
[zero]/[rand] norm_gated maxdiff=0.0      (gdn_norm_gated vs (x/||x||)*z)
[zero]/[rand] qn,kn,vc,beta,g maxdiff=0.0 (gdn_prep vs conv1d+silu+rms_norm+sigmoid+compute_g; g float32)
```

### Sibling state (GPU lock)

Lock was free for every an earlier run run. Sibling an earlier run (QMV lane, dist-g4e):
`build_g4e3.log` shows the `ced8c58` wheel build completed 12:23 UTC
(sha256 `d85065b3...` recorded) — no GPU process of theirs was active at
check time; no wall-clock remainder visible.

## 4. Files

- `scripts/build-wheel.sh` — pin verification (this PR)
- `scripts/wheel-gates.sh`, `scripts/wheel-gates_config.py`, `scripts/wheel-gates_gen.py`,
  `scripts/wheel-gates_kernels.py` — gate harness (kept in-repo for re-run; all
  skip-safe off-GPU)
- `REPORT.md` — this report

# an earlier run: Speed bench on the fixed gdn wheel (dist-decode @75ca881)

MTP-ON 100 tok/s lane. Measured the fixed wheel (an earlier run: gdn_prep launcher fix,
14 kernel-arg host `Data*` ptrs passed as device ptrs; fix commit `75ca881`).
G1/G2 had already PASSed on dist-decode (no-fault, maxdiff 0.0 vs reference,
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

- SSH chain: `ssh -J operator@<tailnet-ip> the build VM@<guest-ip>` (guest the build VM).
- Build clone: `git -C /models/mlx-clone-g4 log --oneline -1`
  → `75ca881 an earlier run: fix gdn_prep launcher - data_shared_ptr().get() is a host Data*, not a device pointer`
- Wheel identity (venv `/models/venv-d`, python3.12):
  `mlx-0.32.3.dev20261028+75ca881` (+ `mlx_lm-0.32.2`) — matches the fixed
  clone ref. `import mlx.core` OK under `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`.
- Lock: `flock -w 10 /models/gpu-bench.lock -c true` → free (no GPU siblings).
- Model-load smoke (flock, 10 tokens, Qwen3.5 GDN-hybrid 9B 4bit
  `/models/qwen3-9b-mlx`): `SMOKE-OK n=10 dt=1.53`, no fault,
  gfx1201 bound.

## Phase 2 — bench (flock, fresh process per run, 3 repeats, median)

Harness: repo `bench/mtp-bench3.py` (AR = `MLX_MTP=0`, MTP K=1 M=2 =
`MLX_MTP=2`), 1024 tokens greedy, smartphone prompt, run via new runner
`bench/bench-decode.sh` (copied to guest `/models/tmp/bench-decode/`):

```
flock -w 300 /models/gpu-bench.lock env MLX_MTP=<0|2> \
  /models/venv-d/bin/python /models/tmp/bench-decode/mtp-bench3.py
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
miss it. The pre-fix an earlier run run reported m2 byte-identical to m0, so the
divergence is specific to the fixed wheel's state layout/path.

## Phase 4 — harness change

`bench/bench-decode.sh` (new): parameterized runner (venv/harness/out overridable,
defaults to venv-d), 3 repeats per config with median, conditional K=2, and
the PR #8 vacuity-guarded lossless gate (same `cmp_streams` helper as
`bench4.sh`/`bench5.sh`). No changes to `mtp-bench3.py`/`mtp-bench5.py`.

## Exact commands (repro)

```
ssh -J operator@<tailnet-ip> the build VM@<guest-ip>
git -C /models/mlx-clone-g4 log --oneline -1        # 75ca881
ls /models/venv-d/lib/python3.12/site-packages/ | grep mlx
flock -w 10 /models/gpu-bench.lock -c true          # LOCK-FREE
# smoke (10 tokens, flock -w 240): venv-d python, load + 10x argmax step
scp bench/{mtp-bench3.py,mtp-bench5.py,bench-decode.sh} the build VM@...:/models/tmp/bench-decode/
bash /models/tmp/bench-decode/bench-decode.sh            # 6 runs + gate (K2 not run)
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
`scripts/apply-m6-patch.py`, idempotent, `.bak-an earlier run` backup per the
an earlier runc convention):

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
  patched venv-d): **M6-AB-OK** (exit 0).
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
- G4 gate (`scripts/run-m6-g4.sh`, bench4 logic on venv-d, 5 arms):
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
- `scripts/apply-m6-patch.py` — idempotent applier + .bak-an earlier run.
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

Worker: <operator> (2026-10-28). Source of truth: method-correction doc
(https://drop.<host>/planky/3150ddc860c16a6d3bf769f3.html) — section 4 step 0,
section 2 rules 1/2/6/7. No model load. All GPU time under
`flock /models/gpu-bench.lock` (lock free at dispatch; an earlier run terminal).

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
`venv=venv-d rocm=10.1 mlx_wheel=0.32.3.dev20261028+75ca881
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
ssh -J operator@<tailnet-ip> the build VM@<guest-ip> 'flock -w 5 /models/gpu-bench.lock -c true && echo LOCK_FREE'

# deploy bench
scp -J operator@<tailnet-ip> bench/step0_streaming_read.py bench/step0_parse.py bench/step0_run.sh the build VM@<guest-ip>:/models/tmp/

# 4 GiB pass (default) — CU count + ceiling, under flock, rocprofv3 kernel time
ssh -J operator@<tailnet-ip> the build VM@<guest-ip> 'bash /models/tmp/step0_run.sh'

# 2 GiB variant
ssh -J operator@<tailnet-ip> the build VM@<guest-ip> 'STEP0_NBYTES=$((2*1024*1024*1024)) STEP0_OUT=/models/tmp/step0_2g bash /models/tmp/step0_run.sh'

# CU count (independent of the bench)
ssh -J operator@<tailnet-ip> the build VM@<guest-ip> '/opt/rocm/core-10.1/bin/rocminfo | grep -B1 -A4 "Marketing Name"; /opt/rocm/core-10.1/bin/amd-smi | head -14; cat /sys/class/kfd/kfd/topology/nodes/1/properties'

# direct HIP CU query (the value mlx uses)
ssh -J operator@<tailnet-ip> the build VM@<guest-ip> 'export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib; /opt/rocm/core-10.1/bin/hipcc -o /models/tmp/hipcu bench/step0_hip_cu.cu && flock -w 60 /models/gpu-bench.lock -c /models/tmp/hipcu'
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
# MTP SPEED LEVER — fast M=2 under the bit-exact rule (an earlier runc)

Continuation of burned an earlier run. Objective: make the m2 (K=1/S=2 spec-verify) path fast WITH the lossless gate.

## TL;DR
**The M=2 QMM path is already the M=1 row algorithm — bit-exact AND fast. No code change needed.** The config variant (wide QMV at M=2) is the current path. The "2x row cost" is the 2x verify time (expected, since m2 processes 2 tokens), and m2 throughput ≈ m0 throughput.

## 1. Analysis (no GPU)

### Model quant params (/models/qwen3-9b-mlx)
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

## 2. an earlier run (M-invariance) intersection
- The sibling line makes M=2..8 rows run the M=1 row algorithm for bit-exactness.
- **My analysis: the wide QMV kernel (the M=2 path for this model) is ALREADY M-independent per-row → the M=2 path already runs the M=1 row algorithm. So the an earlier run fix is a no-op at the QMM level for this model (the QMM is already bit-exact).**
- The o_proj divergence (an earlier run/an earlier run named op) is NOT at the QMM level (primitive shows o_proj BIT-EXACT, maxdiff=0.0). It must be in another part of the m2 pipeline (GDN, attention, or the m2 harness setup).
- **The speed track does NOT conflict with the correctness fix**: the M=2 path (wide QMV) is fast (m2 ≈ m0) and bit-exact. No mitigation needed.

## 3. Verification

### (a) Primitive (MLP + lm_head geometry, M=2 vs 2xM=1, maxdiff)
Build: a7d1e7f (wheel `mlx-0.32.3.dev20261028+a7d1e7f`, dist-g3s/). venv-c (fresh venv + g3 wheel + mlx_lm).
```
gate/up:  K=4096  N=12288  M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
down:     K=12288 N=4096   M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
lm_head:  K=4096  N=248320 M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
o_proj:   K=4096  N=4096   M=2 vs 2xM=1 maxdiff=0.000e+00  BIT-EXACT
```
**All 4 geometries BIT-EXACT (maxdiff=0.0).**

### (b) Bench (m0 + m2, 1024 tokens, greedy, fresh process, flock)
Reference (an earlier run, /models/tmp/bench4/): m0=64.49 tok/s, m2a=63.81 tok/s (389/512=76% acceptance).
G3 bench (/models/tmp/bench-g3/): pending.

## 4. Gate
- m0 stream unchanged vs reference (/models/tmp/bench4/m0.out): pending.
- m2 stream BIT-EXACT vs m0: pending.
- m2 tok/s >= 65.05: pending.

## 5. Land
- guest g3: branch mtp-speed-mlp-lmhead @ d449729 (pushed to fork mlx-clone-k3).
- worktree: branch worker/mtp-speed-mlp-lmhead (vendor.json bump + this report), push to mirror, open PR vs main.
- worktree: branch worker/mtp-speed-mlp-lmhead (vendor.json bump + this report), push to mirror, open PR vs main.


# STEP 1 — Multi-row QMV investigation (marzukia/mlx-rocm)

Salvaged from an earlier run (burned 3601/3600s mid-investigation). Resumed from
`refs/heads/fleetwork-salvage/784` → branch `worker/step1-qmv-mr` @ 23a63c20.

**Reframing (the key correction to an earlier run):** the QMV matmul dispatches ARE in the
rocprofv3 trace. an earlier run missed them because rocprofv3's `grid_size` reports total
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

Wheel: built on the build VM from `/models/mlx-clone-mr` (base 75ca881 + branch MR-kernel
state, byte-verified identical to the repo vendor tree at 1e9c50e; qmm.o + wheel built
after the final source fix, `qmv_wide_mr_kernel` symbols present in libmlx.so). Unpacked
at `/models/wheel-qmv` (PYTHONPATH per the script header). Launch-verified:
`qmv_wide_mr_kernel` dispatches for M=1 and M=2, grid (768,1) blocks for K=4096
N=12288 (24576×16 work-items ÷ 32×16 workgroup), LDS 32768 B.

Effective config (doc rule 6, printed per run): no `MLX_QMV_NO_MR`, no
`MLX_QMV_MR_COLS` (CONFIG lines show `env={}`) → defaults: MR gate M≤4 && LDS≤32 KB,
16 cols/block. Dispatched kernel printed per shape by the parser (`KERNELS:` lines):
`qmv_wide_mr_kernel` for all M on gate_up/o_proj/lm_head; `qmv_wide_kernel` fallback
on the down shape (K=12288 → LDS 96 KB > 32 KB gate, by design).

GB/s = bytes moved / rocprofv3 kernel time (doc rule 2; 50 measured dispatches,
10 warmup excluded), all-bytes basis (weights + scales + biases + x + y).

### 6.1 Per-shape results (new wheel vs same-session ref run, venv-d wheel 75ca881)

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

Artifacts on the guest: `/models/tmp/qmv-mr/{ref,new}/{bitexact,trace,ar.log}`,
run logs `ref-run.log` / `new-run.log`, wheel `/models/wheel-qmv`.

## 7. MR kernel weight-reuse (dequant hoist) — measurement (2026-10-29, worker/step1-qmv-mr2 @ e144f39)

Scope: build + gates + bench + report only (an earlier run's kernel change, e144f39, was
committed but its bench evidence was lost to the upload failure). No kernel
redesign in this run.

### 7.1 Design rationale (the hoist, e144f39)

The an earlier run MR kernel re-dequantized the weight tile (shift+mask+cvt) for EVERY
row m inside the k-loop, although the tile is identical across rows. e144f39
hoists the dequant OUT of the row loop into a register array (`wn[32]` 4-bit /
`wn[16]` 8-bit, computed once per k-tile) and MACs all M rows against it. The
per-row math keeps the EXACT FMA order of `qmv_wide_kernel` (same chunk order,
same inner unroll, same `acc += s*la + b*lx`), so every row must stay
bit-identical to the per-row kernel. The intended effect: the per-row
increment becomes the MAC only (sub-additive), not dequant+MAC.

### 7.2 Build + effective config

The an earlier run guest clone (`/models/mlx-clone-mr`) was gone, so it was
reconstructed: fork base 75ca881b (the an earlier run base) + the repo vendor-tree
overlay of exactly the three quantized files (`qmm.hip` md5 ab6bc086, new
`qmv_wide_mr_kernel.hip` md5 2352b75c byte-matching the branch file, new
`qmv_lx_kernel.hip`). The vendor tree's `fast.*` trio was NOT overlaid (it is
the older the earlier vendor base base; 75ca881's `fast.h` carries `gdn_prep`, which the
75ca881 `python/src/fast.cpp` binding requires — the first build failed on
exactly that, then built clean). Wheel
`mlx-0.32.3.dev20261028+75ca881b` (sha256 dc9274a4…; libmlx.so sha256
821da604…, 24 `qmv_wide_mr` symbols, 76 `gfx1201` strings), unpacked at
`/models/wheel-qmv` (an earlier run wheel backed up to `wheel-qmv-an earlier run.bak`).
Launch-verified: `[mlx-rocm] bound HIP device 0: gfx1201 (AMD Radeon RX 9070
XT) cus=32`; M=2 matmul OK. No `MLX_QMV_*` env (CONFIG lines show `env={}`) →
defaults: MR gate M≤4 && LDS≤32 KB, 16 cols/block. `qmv_wide_mr_kernel`
dispatches for all M on gate_up/o_proj/lm_head (60/60 dispatches per shape in
the trace); `qmv_wide_kernel` fallback on the down shape (K=12288 → LDS
96 KB > 32 KB gate, by design).

GB/s = bytes moved / rocprofv3 kernel time (50 measured dispatches, 10 warmup
excluded), all-bytes basis; pct against the MEASURED 640 GB/s peak (method
doc: 9070 XT = 640 GB/s, NOT 960; the an earlier run parse script's 636 denominator is
superseded here). ms/token = M=1 kernel time per call (decode is M=1).

### 7.3 Correctness gates (all in-run)

- **Bit-exact vs per-row wide kernel: PASS 16/16** — refs regenerated with the
  baseline wheel (venv-d, per-row `qmv_wide_kernel`; the an earlier run `.npy` refs were
  cleaned from the guest), checked with the hoist wheel: every shape × M=1..4,
  maxdiff 0.0, raw-uint16 `array_equal` True.
- **AR 1024-token greedy stream (MLX_MTP=0, qwen3-9b-mlx, harness prompt):
  MATCH** — `TOKS=1024 TOKS_S=70.48`; ALLTOKS byte-identical (cmp) to BOTH the
  per-row wheel stream (an earlier run `new-alltoks`) and the baseline stream
  (`ref-alltoks`). Cross-run tok/s is drift (an earlier run: 62.97 new / 70.28 ref);
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

Measured per-row increment (in-run, (t(M=4)−t(M=1))/3) vs an earlier run:

| shape | an earlier run µs/row | an earlier run µs/row | Δ |
|-------|-------------|-------------|---|
| gate_up (MR) | 34.8 | 33.0 | −5.2% |
| o_proj (MR) | 9.95 | 9.91 | −0.4% |
| lm_head (MR) | 486.4 | 488.0 | +0.3% |
| down (wide, unchanged — drift control) | 30.5 | 28.3 | −7.1% |

The hoist changed the per-row cost by ≤7% — within the drift band set by the
unchanged wide kernel (−7.1%; cross-run M=1 baselines moved 0-19% between the
an earlier run and an earlier run sessions, so only increments are comparable). Effective per-row MAC throughput: gate_up
100.66 MFLOP / 33.0 µs ≈ 3.05 TFLOPS; lm_head 2037 MFLOP / 488 µs ≈ 4.17
TFLOPS. The bars need ≤5.6 µs/row (M=2) and ≤4.3 µs/row (M=4) on gate_up —
6-8× below the measured increment.

Why: the dequant expression `(wp[q] >> (BITS·j)) & mask` depends ONLY on
`wp[k]`, which is invariant across the m-loop — so hipcc was ALREADY
CSE-ing it out of the row loop in the an earlier run kernel. The explicit register
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
the per-wave MAC work. Dispatch config unchanged from an earlier run (workgroup 32×16,
LDS 32768 B, private_segment 0 — no spill; 60/60 dispatches per shape).
(Supported counters on this rocprofv3/gfx1201: SQ_WAVES, SQ_BUSY_CYCLES,
GRBM_GUI_ACTIVE, CU_NUM; SQ_INSTS_VALU / TA_TA_BUSY / GL2C_* report 0.)

**Verdict: bars still fail — STOP (per scope).** The per-row increment is the
MAC work; flattening the curve needs the MAC itself to stop scaling with M
(a tile/WMMA formulation that reuses the loaded weight tile across rows at
the instruction level, per §6.4) — a second kernel design, NOT started in
this run.

Artifacts on the guest: `/models/tmp/qmv-mr/{ref,new}/` (bitexact, 16
bench logs + traces, ar.log, config.log, gpuinfo.log, flock-fails.log),
`/models/tmp/qmv-mr/pmc-mr-{4096,12288}-M{1,4}/` (PMC traces), build log
`/models/an earlier run-build2.log`, wheel `/models/wheel-qmv` (an earlier run wheel at
`/models/wheel-qmv-an earlier run.bak`).

## 8. an earlier run LX hoist A/B — verify + measure (2026-10-29, worker/qmv-tile-wmma @ 01c1013, an earlier run salvage)

The an earlier run lever (method doc section 6.4: column-invariant per-row x-chunk-sum
hoisted to a double-buffered 1 KB LDS buffer, in-k `__syncthreads`; an earlier run
draft, an earlier run design/build, this run = verify + measure). A/B toggle:
`MLX_QMV_MR_LX` (1 = hoist, 0 = an earlier run path — the an earlier run MR2 kernel; each
instantiation compiles only its m-loop path via `if constexpr`).

### 8.1 Build + effective config

- Source: the 01c1013 vendor tree (MR2 lineage merge + an earlier run hoist + the
  dispatch fix — LXHOIST is a template parameter, selected at runtime).
  Applier round-trip replay verified: MR2-lineage base (md5-verified:
  `qmv_wide_mr_kernel.hip` 2352b75c, `qmm.hip` ab6bc086) +
  `patches/qmv-mr-lxhoist.diff` → byte-identical to the committed vendor
  tree (b05f379b / 0fdec793).
- Wheel built on the build VM (ROCm 10.1, gfx1201, RX 9070 XT) in
  `/models/mlx-clone-mr` (incremental; `qmm.o` recompiled for the
  dispatch fix, 0 errors), `.so` swapped into `/models/venv-an earlier run`
  (md5 4b51625c matches the build output). Wheel
  0.32.3.dev20261028+75ca881b.
- Bench: one fresh process per env (an earlier run convention), 10 warmup + 50 reps,
  `rocprofv3 --kernel-trace` kernel time (doc rule 2), under
  `flock /models/gpu-bench.lock`. Config printed + asserted per run:
  wheel + MLX_* env in every CONFIG line; kernel family per shape and the
  template instantiation (`, true>` / `, false>` from the full symbol name)
  asserted by the runner (CONFIG-ASSERT: PASS).

### 8.2 Lossless gate (doc rule 4) — PASS

Bit-exact vs the per-row `qmv_wide_kernel` at identical inputs (seed 42 per
shape, bf16 x, 4-bit affine gs 64 with bias, bf16 scales/biases; ref run
`MLX_QMV_NO_MR=1`, check run `MLX_QMV_MR_LX=1`, same wheel): **16/16
combos (4 shapes x M=1..4) maxdiff 0.0, raw-uint16 `array_equal` True**.
The hoist's FADD chain is the exact an earlier run per-row lx order, so the value is
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

Baseline consistency: the LX=0 (an earlier run path) numbers match the an earlier run section
7.4 measurement (same kernel, prior session) within the documented cross-run
drift band (0-19%; section 6.1) — gate_up M=1 43.4 vs 43.2 µs/call,
lm_head M=1 905.3 vs 906.0 — confirming the A/B baseline IS the an earlier run MR2
baseline; the in-run A/B (control-stable) is the valid gate measurement.

### 8.4 Why the hoist loses

The kernel is compute-bound (section 7.6: 99.5-100% CU active, SQ_WAVES
constant in M). The an earlier run path computes the per-row lx inline (16-column
redundancy, 32-term FADD chain per row per k-tile) with NO in-k barrier.
The hoist replaces that with one computation per (row, k-tile) + an
`__syncthreads` per k-iteration + a 1 KB double-buffered LDS write/read
round-trip. On this card the barrier + LDS traffic serialize the wavefronts
more than the eliminated FADDs cost — the redundancy was cheap (the FADDs
pipeline behind the MAC work), the barrier is not. The loss is worst at M=1
(the barrier is paid for 1 row's worth of work) and shrinks with M (more
MAC work per barrier), consistent with the measured −19% (M=1) → −10.6%
(M=3) gradient.

**Verdict: the section 6.4 lever (an earlier run LX hoist) is LOSSLESS (16/16
bit-exact) but does NOT meet the >= 3% gain gate — it is a 0.7-31%
performance loss on gfx1201. The an earlier run path (LX=0) remains the best measured
multi-row QMV configuration; the M=2..4 bars (section 7.5) still FAIL.**
Follow-up flag: the dispatch default is `MLX_QMV_MR_LX=1` (hoist on) — with
this measurement the default should flip to 0 (or the hoist instantiation
dropped) before the MR path is used in a serve config.

Artifacts on the guest: `/models/tmp/qmv-mr2-lx/` (bitexact ref/check
logs + .npy refs, bench0/1 logs, rocprofv3 traces, parse/combos/gain logs),
build log `/models/an earlier run-build.log`, wheel `.so` in
`/models/venv-an earlier run` (build dir `/models/mlx-clone-mr`).

## m8: full-state audit + T-dependence patch (an earlier run)

### P1: state inventory (static audit, no GPU)

Guest stack audited: venv-d wheel 75ca881 (mlx-clone-mr; GDN tail
fusion fbb7dc95 + an earlier runc + an earlier run), model qwen3_5 (32 layers,
full_attention_interval=4 → 24 GDN + 8 FA; 32 v-heads, 16 k-heads,
head 128, conv kernel 4, conv_dim 8192), MTP = single full-attention
drafter with its own KVCache. Fused path (MLX_GDN_TAIL_FUSION=1,
mask=None): gdn_prep (conv+silu, beta, g, qk_norm) → gated_delta
(T<=8 → recurrent) → gdn_norm_gated.

**State written by a draft/verify forward, per owner:**

| # | state | owner | shape | write path | verify-path rollback (bench/mtp-bench3.py) | verdict |
|---|-------|-------|-------|-----------|-------------------------------------------|---------|
| 1 | conv_state | GDN ArraysCache[0] | (1,3,8192) bf16 | `cache[0] = mx.contiguous(conv_input[:, -3:, :])` (qwen3_next.py fused) | save_gdn/restore_gdn (ref restore) | covered (an earlier run n=0) |
| 2 | ssm_state | GDN ArraysCache[1] | (1,32,128,128) f32 | `cache[1] = state` (recurrent out) | save_gdn/restore_gdn | covered (an earlier run n=0) |
| 3 | lengths | GDN ArraysCache | None | never prepared (probe/bench) | carried in save_gdn | n/a (None) |
| 4 | left_padding | GDN ArraysCache | None | never set | carried in save_gdn | n/a (None) |
| 5 | keys | FA KVCache | (1,n_kv,T,hd) | update_and_fetch append | c.trim(2) | covered (an earlier run n=0) |
| 6 | values | FA KVCache | (1,n_kv,T,hd) | update_and_fetch append | c.trim(2) | covered (an earlier run n=0) |
| 7 | offset | FA KVCache | int | +1/forward; feeds RowInvariantRoPE position + k/v write slot | trim(2) restores | covered (m7: load-bearing, t14 4096/4096 drift when omitted) |
| 8 | mtp keys | MTP KVCache | (1,16,T,256) | MTP draft forward | save_mtp/restore_mtp (m7, numpy forced-eval) | covered (m7 4096/4096 bit-exact) |
| 9 | mtp values | MTP KVCache | (1,16,T,256) | MTP draft forward | save_mtp/restore_mtp | covered |
| 10 | mtp offset | MTP KVCache | int | +1/draft; RoPE position | save_mtp/restore_mtp | covered |
| 11 | position | global | — | no global counter; per-cache offset (FA #7, MTP #10); GDN position-free | per-cache | covered |
| 12 | mask metadata | ArraysCache/KVCache | — | create_ssm_mask→lengths=None,left_padding=None→mask=None; create_attention_mask→causal/None | n/a | n/a (mask=None → fused kernel, no fallback) |
| 13 | draft/sampler | MTP module | — | drafter is a stateless single forward (state = mtp_cache #8-10); argmax, no RNG | via #8-10 | covered |
| 14 | allocator pool | ROCm hipMallocAsync | — | not model state; OOB reads in gdn_prep (below) land in pool memory | not restorable | suspect (deterministic within a run; see compute items) |

**Compute-level T-dependence (fused path, S=2 verify vs S=1 decode):**

| # | kernel (guest wheel 75ca881) | T-dependent quantity | row-0 impact | verdict |
|---|------------------------------|--------------------|--------------|---------|
| 15 | gdn_prep_kernel conv (gdn_prep.hip) | `conv_state[i * conv_dim + c]` with conv_dim = S*8192 (total input size), but conv_state row stride is 8192 (S=1 layout). S=2 row-0 reads [row0, row2, OOB] not [row0, row1, row2]; token-1 reads [row1, OOB, OOB] not [row2, in0, in1] | **row-0 T-dependent** (the m6 class) | NOT covered by rollback (compute, not state) — the leak |
| 16 | gdn_prep_kernel conv_weight | `conv_weight[c*4+i]`, c up to S*8191, weight buffer 8192*4 → OOB for c>=8192 | token-1 only (row-0 in range) | NOT covered |
| 17 | gdn_prep_kernel beta/g | `A_log[t]`, `dt_bias[t]` for t < S*32; A_log/dt_bias are (32,) → OOB for t>=32 | token-1 only (row-0 in range) | NOT covered |
| 18 | gdn_qk_norm_kernel | one block per (token,head) row | row-invariant | OK |
| 19 | r4d_gdn_recurrent_update | sequential t loop; row t reads only row-t inputs + state; grid (B,H,vs) independent of T | row-invariant | OK |
| 20 | gdn_norm_gated_kernel | one block per (token,head) row | row-invariant | OK |
| 21 | FA rope/SDPA | RowInvariantRoPE (m6 fix): per-row (offset+row) theta | row-invariant (m6) | OK |

**P1 verdict:** all 14 state items are covered by the verify-path
rollback (1-10, 11, 13; 3/4/12 are no-op Nones). The G4
first_div=148 is a **compute-level** T-dependence in the GDN fused
path, not a state leak. The row-0 T-dependence is item 15 (the conv
in gdn_prep_kernel); items 16-17 are token-1-only OOB reads that the
an earlier run state-identity (n=0) shows are benign in the current pool layout
but are not row-invariant by construction. The patch (P3) makes the
gdn_prep conv row-invariant (per-token causal sliding window) and the
beta/g + conv_weight reads in-range for all S, preserving S=1
bit-exactness (sequential k-order sum) and matching the unfused
nn.Conv1d reference for all S.

**P2 (single-fork probe, one GPU run): DONE (an earlier run, venv-d 75ca881, step 100,
prev=11 draft=6278).** scripts/m8-state-audit.py (hook fixed: class-level
__call__ + id(self) lookup; the salvaged instance-attr __call__ hook
does not fire — dunders are looked up on the type).

**Result DIFFERS from dbb6644 — exact delta reported before patching:**

DYNAMIC PROOF (S=2 [prev,draft] vs two S=1 [prev],[draft], numpy
forced-eval state capture):
- GDN layers 0,1,2: conv n=0/24576, ssm n=0/524288 — **CLEAN** (bit-
  identical; the GDN conv + recurrent kernel are row-0 T-invariant
  given the same inputs — refutes the static-analysis hypothesis that
  the gdn_prep conv_state indexing (conv_dim = S*8192 as the row
  stride) is the row-0 T-dependence source; the OOB reads for row 0
  (conv_state[1*16384+c]=row2, conv_state[2*16384+c]=OOB) evidently
  return values that keep the row-0 conv + ssm bit-exact in this
  pool layout, or the conv is not the source).
- GDN layers 4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,
  30: conv n=16384/24576, ssm n=524288/524288 — **LEAK** (all ssm
  elements differ; the GDN state is NOT bit-identical post-rollback
  for layers 4+ — a consequence of the layer-3 input T-dependence,
  not a rollback bug: the rollback restores the pre-fork state, but
  the S=2 forward's layer-4+ inputs differ from the two-S=1 inputs
  because layer 3's row-0 output is T-dependent).
- FA layer 3: KV keys n=0/262144, values n=0/262144 — **CLEAN** (the
  KV state is bit-identical; the m6 rope fix holds at the state
  level).
- FA layers 7,11,15,19,23,27,31: KV keys n=2048/262144, values
  n=2048/262144 — **LEAK** (2048 = 16 heads * 128 = one token's k/v;
  the draft token's k/v differs — a consequence of the upstream
  T-dependence).
- DYNAMIC PROOF verdict: **LEAK FOUND** (the state is NOT fully
  bit-identical post-rollback — the GDN layers 4+ and FA layers 7+
  leak; the GDN layers 0,1,2 and FA layer 3 are clean).

COMPUTE-LEVEL (per-layer row-0 output S=2 vs S=1):
- Layers 0,1,2 (GDN): n=0/4096 — **CLEAN** (row-0 T-invariant).
- Layer 3 (FA): n=4096/4096 maxabs=2.287e+00 — **T-DEPENDENT** (the
  FIRST diverging layer).
- Layers 4-31: n=4096/4096 — T-DEPENDENT (propagation).
- COMPUTE-LEVEL verdict: **first diverging layer = 3 (FA, not GDN)**.

Row-0 logits (A[0,0] vs B1[0,0]): n=248320/248320 maxabs=2.443e+01
(T-DEPENDENT, consistent with dbb6644's "row-0 logits differ on all
248320 entries").

**Root-cause re-localization (an earlier run):** the G4 first_div=148 is a
**compute-level FA (SDPA) T-dependence at layer 3**, NOT a GDN conv
T-dependence and NOT a state-rollback leak. The GDN layers 0,1,2 are
row-0 T-invariant (clean state + clean row-0 output). The FA layer 3
has a clean KV state (n=0) but a T-dependent row-0 output (n=4096) —
so the T-dependence is in the SDPA compute (the m6 rope fix holds at
the state level; the SDPA kernel computes row 0's attention
differently for S=2 vs S=1). The GDN state leak (layers 4+) and the
FA state leak (layers 7+) are CONSEQUENCES of the layer-3 row-0
T-dependence (the upstream input differs), not independent leaks.

**P3 patch target (revised):** the SDPA kernel's row-0 T-dependence
(the FA path), NOT the gdn_prep conv. The patch must make the SDPA
row-0 attention T-invariant (row 0's attention over [cache keys, row 0
key] must be bit-identical for S=2 and S=1). The gdn_prep conv
T-dependence (the static-analysis hypothesis) is refuted by the probe
(the GDN layers 0,1,2 are clean); it may still be a latent issue for
other S values but is not the G4 first-div source.

---

## P3 (an earlier run): root cause found - SDPA 1-pass kernel compile-time causal split (row-1 T-dependence)

**Probe-discipline fix (this session):** all state snapshots are now
numpy copies restored DTYPE-PRESERVING (`mx.array(numpy, dtype=orig)`).
The P2/an earlier run "layer-3 row-0 T-dependence" was a probe artifact of the
f32 GDN-restore (see P2 addendum above); with dtype-preserving restore
the 148-fork re-run shows **row 0 fully clean and row 1 clean at the
147 state** (PREDICTED token@148: mtp(l1A)=33443,
ar(B2 S=1 [d])=33443 - the flow's token@148=33443 is reproduced, so
the MTP-vs-AR divergence at 148 is NOT from any T-dependence AT the
147 state; the state trajectories had already diverged earlier).

**venv-d GDN source read:** the fused path (gdn_prep +
mx.fast.gated_delta_update + gdn_norm_gated) is ALWAYS active in
inference mode (MLX_GDN_TAIL_FUSION=1 default, no S gate) - both S=1
and S=2 use the same fused kernel. The only dtype change in the
mtp-bench3 flow is restore_mtp (MTP module KV, f32 on reject) - the
MTP module is separate from the target; its f32 affects drafts only.

**Per-verify-step state trace (m8-mtp-state-trace.py):** S=2 post-state
vs 2xS=1 post-state at every one of 85 verify steps:
- **first state leak at verify step 47** (reject step, t2=9249 !=
  d=24051): GDN layer 16 first (conv 3.1e-2, ssm 96% of entries
  ~2.8e-4 f32), layers 17-30 conv+ssm, then FA KV 15/19/23/27/31
  keys+values (downstream). 4 leaking steps total (47, 61, x, 169) -
  value-dependent.
- **81 clean steps: the fused GDN kernel (gdn_prep +
  gated_delta_update) is itself S-invariant (bit-exact S=2 vs 2xS=1
  from identical inputs)** - every leak is downstream of an upstream
  input change, not an independent GDN leak.

**Step-47 fork bisection (m8-mtp148-bisect.py, per-layer FA captures,
rowx key-matching fixed):**
- ROW-0: all 32 layers + FA components clean (row-0 T-invariant
  holds).
- ROW-1: first T-dependent component = **FA15_sdpa_out n=1/4096
  maxabs=9.766e-04 (one bf16 ulp)** - FA15 inputs (input_ln,
  q/k/v proj+norm, rope, sdpa_q/k/v) all bit-identical.
- FA3_sdpa_k "hit" in the first pass was a rowx artifact (kv-head-1
  of A compared with kv-head-0 of B2); direct per-row comparison:
  A vs B2 keys bit-identical (prefix n_diff_rows=0, fullslice
  n=0/19200).
- DYNAMIC PROOF: A post-state vs B1+B2 post-state = the trace leak
  (GDN 16+ conv/ssm, KV 15+ keys/values) - consistent with the FA15
  row-1 1-ulp input.

**Mechanism (source read, venv wheel 75ca881b + vendor the earlier vendor base
identical file):** `mlx/backend/rocm/scaled_dot_product_attention.hip`
- the decode path (qL < 4) uses `kernel_sdpav_1pass<T, do_causal, D>`
  with **`do_causal` a compile-time template parameter**: S=2
  (mask="causal", N>1) dispatches the `std::true_type`
  instantiation; S=1 (mask=None, N==1) dispatches
  `std::false_type`. Two different compiled kernels with the same
  per-row math; for row 1 (all keys visible in both - `use_key = i
  <= kL-1` always true) the only difference is the codegen (f32
  contraction/scheduling of the online-softmax accumulation), which
  yields a value-dependent 1-ulp flip in bf16. Row 0 is
  T-invariant (causal row 0 masks the last key - the earlier
  isolation probe showed O1 (S=1) vs O2 (S=2 causal) row 0
  bit-exact; the kernel is deterministic, O2 re-run bit-exact).
- The 1-ulp FA15 row-1 flip contaminates the S=2 state (GDN 16+
  conv/ssm via the fused path's f32 state accumulation), which
  compounds over the remaining ~100 steps (4 injection points) until
  the top-2 logits at 147/148 separate enough to flip argmax
  (33443 vs 24296) - the G4 first_div=148. AR (m0, S=1, mask=None)
  never runs the causal instantiation, so its state stays clean -
  the MTP-vs-AR divergence.

**P3 patch (patches/m8-sdpa1pass-single-kernel.diff):** remove the
compile-time `do_causal` template parameter; pass it at runtime
(`params.do_causal`); single instantiation per (dtype, D). The S=2
causal row 1 and the S=1 no-mask row then use the SAME compiled
code and are bit-identical for identical inputs (determinism
proven); causal row 0 still masks the last key via the runtime
branch (a no-op iteration - no accumulation change). No other kernel
or the 2-pass prefill path is touched.

**Verification (pending, post-rebuild):** (1) state trace:
n_leak_steps must be 0 (was 4); (2) G4 5-arm: M2-M0-LOSSLESS MATCH
(bit-exact; pre-patch FAIL first_div=148 a=33443 b=24296); (3) A/B
MTP-vs-MTP 4096/4096 bit-exact regression (must stay); (4) bench
TOKS_S AR / MTP K=1 / K=2 (baselines AR 69.65, MTP K=1 64.34).

### P3 final results (one-round-max: first_div MOVED - report + end)

**Patch + rebuild:** `patches/m8-sdpa1pass-single-kernel.diff` applied to a
clean clone at `75ca881b` (`/models/tmp/mlx-m8`); wheel built
(`/models/tmp/mlx-m8/dist-m8/mlx-0.32.3.dev20261029+75ca881b-...whl`)
and installed (`.so` sha256 `b651be0a...`; pre-m8 backup
`/tmp/libmlx.so.pre-m8` = `90cf8214...`, restored at the end).

**SDPA patch verified (direct kernel tests, new .so):**
- D=64 kL=80 f32, D=256 kL=76 f32, D=256 kL=75 (f32/f16/bf16):
  S=2 causal row 1 vs S=1 no-mask row 0: **n=0** (bit-exact; pre-patch
  1-ulp). Deterministic (re-runs n=0). Causal row 0 still masked
  (row0 != row1). In-flow fresh-recompute: `freshA-r1 == flowA-r1`
  (n=0), `freshB == flowB` (n=0), `S1(A-r1vals) == S2causal(A-r1vals)`
  n=0 for ALL FA layers, not allocation-sensitive, not strided-layout
  sensitive (layout changes absolute values, not T-invariance).
  **The SDPA kernel is fixed: same q values, causal-row-1 (S=2) is now
  bit-identical to no-mask (S=1).**

**But the in-flow leak persists - first_div MOVED upstream:**
- State trace (new .so, 3x deterministic): n_leak_steps=4 (steps 47,
  61, ..., 169), first at verify 47 (REJECT, d=24051), token@148=33443
  (unchanged).
- Step-47 fork bisection (new .so): row 0 fully clean; row 1 first
  divergence = FA15_sdpa_out n=1/4096 (1-ulp) - same as pre-patch.
- Probe direct value check (new .so): **A's cached k vs B2's cached k
  differ by exactly 1 element: (batch 0, kv-head 2, row 74 = draft row,
  dim 32), a=-0.012207 vs b=-0.012268 (bf16 1-ulp; persists after a
  bf16 round-trip - not a cast artifact).** All other FA layers' k:
  n=0.
- Root cause of that 1 element: **k_proj matmul T-dependence.**
  `x @ W.T` with T=2 (verify row) vs T=1 (re-forward row) dispatches to
  different ROCm kernels; direct test on the guest (random bf16,
  1024x4096): T2-row1 vs T1 **n=1/4096 (1-ulp, 1.221e-04)**; T1
  deterministic (rerun n=0). The bisection's earlier "k clean" evidence
  was a probe bug: `rowx` collapses (B,H,S,D)->(S,D) and compared
  kv-head 0 only (the diff is at head 2); and all instance-level
  `__call__` hooks (k_norm/rope) were silently dead (instance-dict
  `__call__` is bypassed by C-level tp_call), so "k_norm/rope n=0" was
  never actually measured.
- Also observed: the T=1 k_proj call on the real flow values raises a
  divide-by-zero trap in libmlx.so (FPU trap, exit 136) - value-
  specific, in the unpatched matmul code (not the SDPA patch).

**GDN fused kernel: S-invariant.** 81/85 verify steps clean (identical
inputs -> bit-exact S=2 vs 2xS=1); all 4 leaking steps are downstream
of the FA15 k 1-ulp flip, not the GDN kernel itself.

**Mechanism (final):** k_proj's T=2/T=1 kernel split puts 1 bf16-ulp of
T-dependent noise into FA15's cached draft-row key (row 74) at 4
reject-steps; the SDPA (patched or not) faithfully propagates it to the
row-1 output; the S=2 GDN state then accumulates it at 4 injection
points; ~100 steps of GDN recurrent amplification push the drift past
1-ulp in the hidden state, flipping argmax at 148 (33443 MTP vs 24296
AR). The SDPA patch removes one real T-dependence (proven bit-exact in
isolation) but the matmul T-dependence is the remaining (and now
primary) source; fixing it is a second-round item (M=1 vs M=2 qmm/qmv
kernel unification in mlx core), out of scope for this round per the
one-round-max rule (first_div moved: FA15_sdpa_out -> FA15_sdpa_k).

**Gates (baseline .so, this session):**
- G4 5-arm: M2-M0-LOSSLESS **FAIL first_div=148 a=33443 b=24296**
  (deterministic, matches pre-session an earlier run); M1-M0-CONTROL MATCH;
  M1-M2A FAIL first_div=148 (a=24296 b=33443); M2-A-B MATCH;
  M2-B-C MATCH. Bench: AR 69.57 tok/s (baseline 69.65), MTP K=1
  65.44 tok/s (baseline 64.34), ACCEPTED=389/512.
- A/B: **M6-AB-OK** - full-chain row-invariance all MATCH, first
  diverging component NONE (m3 cross-checks match recorded values).
- G4 red -> no K=2 bench (gate-conditional; K=1/AR numbers above).

**End of round.** Next round (new ticket): patch the matmul M=1/M=2
split (or force T=2 decode path for the re-forward row), rebuild,
re-run state trace (expect n_leak_steps=0) + G4 + A/B.

## m8 round 2 (an earlier run): rope 1D/batched dispatch unification - first_div fix

### Correction of the P3 attribution

P3 attributed the FA15 1-ulp to "k_proj matmul T-dependence".
**That attribution is wrong.** With wrapper-based (non-dead) probes on
the SDPA-patched .so, the real-flow per-component audit shows:

- `FA15_k_proj_0_out: n=0/4096` (4bit QMM - `qmv_wide_kernel` for both
  M=1 and M=2, T-invariant by construction)
- `FA15_k_norm_0_out: n=0/1024`, `FA15_q_norm_0_out: n=0/4096`
- `FA15_rope_0_in/out: n=0` (q path)
- **`FA15_rope_1_out: n=1/1024 (1 f32 ulp) - the FIRST row-1
  divergence**, `FA15_rope_1_in: n=0`
- everything downstream (SDPA, KV row 74 dim 32, GDN) is propagation of
  that single rope element.

So the 1-ulp enters at the **fused `mx.fast.rope`** (the model's
`nn.RoPE` - qwen3_5's `rope_parameters` is non-None, so
`rope_scaling` is non-None and the m6 RowInvariantRoPE path is
bypassed), not at the matmul. (P3's direct random-bf16 matmul test did
show a value-specific 1/4096 on one .so, but on the real flow values
k_proj is n=0 and it is not the first divergence.)

### Root cause: `RoPE::eval_gpu` 1D vs batched kernel split

`mlx/backend/rocm/rope.hip:463`:

```cpp
bool single = in.flags().row_contiguous && B == 1 && T == 1;
```

T=1 row-contiguous input dispatches to the 1D single-token kernels
(`rope_single_1d` / `rope_single_freqs_1d`); T>=2 dispatches to the
batched `rope` / `rope_freqs` kernels. **The two kernels compile the
same source expression (`rx1 = x1*cos - x2*sin`, `rx2 = x1*sin +
x2*cos`, via `sincosf`) to different AMDGPU FMA-contraction patterns**
(same .so, same launch config, identical inputs + position):

- Standalone decisive test (wheel flags, gfx1201): T=2 row1 (batched)
  vs T=1 (1D), same input, same pos: **25/256 elements differ by 1 f32
  ulp, ALL in the upper-half `rx2` formula**; maxabs 1.192e-07.
- Control: 1D vs 1D n=0; **batched T=2 vs batched T=1 n=0/256 (the
  batched kernel is bit-identical across T)**.
- Per-element instrumentation (x1, x2, sin, cos, theta, rx1, rx2, L,
  batch_offset, in/out indices) is bit-identical for T=1 vs T=2 under
  both -O2 and the wheel flags - the difference is pure codegen of the
  contraction, not a value or index difference.
- In the flow, the k-path input is row-contiguous for both arms (C
  (1,4,1,256) / C (1,4,2,256)), so T=1 (the accepted re-forward) uses
  the 1D kernel and T=2 (verify) the batched kernel -> the FA15 rope
  1-ulp -> GDN accumulation at 4 reject-steps -> ~100 steps of GDN
  recurrent amplification -> argmax flip at 148 (MTP 33443 vs AR
  24296). Same mechanism P3 described, one component upstream.

### Fix: force the batched kernel for all T

`patches/m8-rope-single.diff` - `rope.hip:463`:

```cpp
- bool single = in.flags().row_contiguous && B == 1 && T == 1;
+ bool single = false;
```

Always the batched kernels (bit-identical across T, verified). The 1D
path was a single-token decode fast-path with the same launch size
(256 threads / 128 active) as the batched kernel at T=1 - no
measurable performance cost. The 1D kernels remain in the tree for the
CPU/Metal paths; only the ROCm GPU dispatch changes.

### Build

Fresh guest clone `/models/tmp/mlx-m8r2` from `75ca881b` (the m8
wheel's base - not the vendor.json primary the earlier vendor base; `rope.hip` is
byte-identical between the two, so the patch applies to both). Both
patches in the working tree: `patches/m8-sdpa1pass-single-kernel.diff`
(P3) + `patches/m8-rope-single.diff` (this round). Wheel:
`/models/tmp/mlx-m8r2/dist-m8r2/mlx-0.32.3.dev20261029+75ca881b-...whl`
(.so md5 `06044ea9b888...`; pre-m8r2 venv backup
`/models/tmp/libmlx.so.pre-m8r2` = the baseline `90cf8214`-era
.so, `af07d9069267...`).

(Note: `/models/venv-d/bin/pip` is a script with a shebang to
**venv-a** - installing "with venv-d pip" silently lands in venv-a.
Use `/models/venv-d/bin/python -m pip`.)

### Verification (all on the m8r2 .so, /models/qwen3-9b-mlx 4bit)

**1. Direct kernel T-invariance (20 random trials, f32, D=256 partial
rope dims=64, base 1e7):** T=2 row1 (pos 74) vs T=1 row0 (pos 74),
same input: **0/20480, bit-exact.** (T=2 r0 vs T=1 r0 same pos also
0/1024 per trial; passthrough dims 64+ exact.)

**2. Flow probe (wrapper hooks, fork at verify token idx 47, A/B1/B2):**
- `FA15_rope_1_out: n=0/1024` (was n=1 - the root-cause element)
- **ROW-1 FIRST DIVERGING COMPONENT: None** (was FA15_rope_1_out);
  ROW-0 first diverging: None
- all FA layers' k_proj/k_norm/q_norm/rope: n=0
- **DYNAMIC PROOF: NO LEAK (state exact)** (A post-state ==
  B1+B2 post-state, GDN conv/ssm + KV keys/values all n=0)
- k-path FUSED rope direct A-r1 vs B2-r0: n=0/1024; q-path n=0/4096
- deterministic (re-run n=0)

**3. State trace (all verify steps, 3x deterministic):**
**n_leak_steps=0, first_leak=None** (pre-fix: 4, first at verify 47).

**4. G4 5-arm gate: ALL MATCH.**
- M2-M0-LOSSLESS (the m6 gate): **MATCH n=1024** (pre-fix:
  FAIL first_div=148 a=33443 b=24296)
- M1-M0-CONTROL: MATCH n=1024
- M2-A-B: MATCH n=1024
- M2-B-C: MATCH n=1024
- (M1-M2A implied by the first two.) Deterministic (re-runs
  identical).

**5. A/B (diag-s62-m6 full chain): M6-AB-OK.** Full-chain
row-invariance all MATCH, first diverging component NONE; m3
cross-checks match recorded values (layer-3 k 0/91136; layer-4
gdn ssm 515560/524288). The documented 1/4096 re-run-vs-capture
DIFF (maxabs 9.766e-04 = 1 bf16 ulp at ~0.28) is the m5-artifact
row-1 element vs the OLD T=2 kernel capture - expected per the
harness docstring; row-invariance itself is MATCH n=4096.

**Bench (G4 gate-conditional; 1024-token arms unless noted):**
- AR (mtp-bench3, MLX_MTP=0): **69.75 tok/s** (baseline 69.65,
  +0.10 - parity)
- MTP K=1 (mtp-bench3, MLX_MTP=2): **60.79 / 60.98 tok/s**
  (two runs; baseline 64.34, P3 m8 65.44; ACCEPTED 376/512 vs
  P3's 389/512). The drop tracks the acceptance rate, NOT kernel
  cost: isolated T=1 rope wall-time is identical between the m8r2
  .so (batched) and the baseline .so (1D): 136.75 vs 136.18
  us/call (T=2: 137.55 vs 137.00). The 1D->batched rounding
  change at T=1 decode shifts which drafts match (376 vs 389
  accepted), and MTP tok/s is acceptance-rate-limited.
- MTP K=2 (mtp-bench5, MLX_MTP=3): **56.13 tok/s** (1399 tokens in
  1024 verify steps, D1 73.2% / D2 65.3%, tok/step 1.732).
  **K=2 lossless:** the K=2 stream is an exact prefix of a
  2048-token AR run (prefix 1399, AR continues to 2048; the
  1024-vs-1399 "len" note from the K=1-style cmp is a step-budget
  artifact, not a divergence).

### Hermetic gates (host)

- `uv run ruff check .` / `uv run ruff format --check .`: this round's
  files (scripts/m8-kproj-flowprobe.py, patches/m8-rope-single.diff)
  are clean. Pre-existing at c04c34c (unchanged this round): 168 ruff
  errors + 17 reformat diffs, all in committed as-run diagnostic
  scripts (kept byte-identical to as-run evidence per repo convention;
  no check-runs are configured on the GitHub repo, CI workflow
  aspirational).
- `uv run pytest -m "not gpu"`: **6 passed, 1 skipped** (PASS).
- `uv run python scripts/size_ratchet.py --base HEAD`: 1 pre-existing
  bloat (scripts/diag-s62-m7.py, 885 non-blank lines, identical at
  HEAD; not in size-ratchet.json) - not touched this round;
  m8-kproj-flowprobe.py is 658 non-blank (< 800).

### Notes / open items

- The plain-bf16-matmul M=1 gemv vs M=2 GEMM split (the P3 suspect)
  remains as a value-specific 1-ulp possibility on some .so builds,
  but (a) the QMM path (the model's k_proj) is T-invariant, (b) MTP's
  plain-matmul k_proj only ever runs at T=1, and (c) on real flow
  values k_proj is n=0 - so it does not gate the lossless path. Left
  as a follow-up candidate (unify M=1/M=2 GEMM dispatch, or split
  T=2 into 2xT=1) if a future value-specific case surfaces.
- venv-d was restored to the pre-m8r2 baseline .so at the end
  (backup convention per P3); the m8r2 wheel stays at
  /models/tmp/mlx-m8r2/dist-m8r2/ for re-install
  (`/models/venv-d/bin/python -m pip install --force-reinstall
  --no-deps`), and the pre-m8r2 baseline backup is at
  /models/tmp/libmlx.so.pre-m8r2. Evidence logs on the guest:
  /tmp/flowprobe-m8r2.txt, /tmp/state-trace-m8r2.txt,
  /models/tmp/g4m8r2/ (g4 arms + m3 K=2 + m0-2k + ab.log).

---

# an earlier run — REBASE of PR #20 (worker/m8-full-state-audit) onto main @287aeb9

## Result

- **New head:** `d4d433c27d15a306a5dbc58ea8519b96816431a3` (branch `worker/m8-full-state-audit`,
  force-with-lease pushed to gh; replaces the gate-passed head `11f38ca`).
- **GitHub:** `gh pr view 20` → `mergeable=MERGEABLE`, `headRefOid=d4d433c`, state=OPEN
  (mergeStateStatus=BLOCKED = branch-protection/review gate only, no conflicts).
- All 20 branch commits survived the rebase onto 287aeb9 (merge-base now = 287aeb9).

## Conflict resolution (11 stops, all mechanical)

main gained #15/#16 (squashes of the branch's own m6/m7 work), #17/#18 (qmv), #9-#13
since the fork (64b98b8). Conflicts and resolutions:

| Commit | File(s) | Resolution |
|---|---|---|
| 543f085, accc9e2 | — | auto-dropped ("patch contents already upstream") |
| 447c2d9 | `scripts/diag-s62-sdpa.py` (add/add) | branch version |
| 7d1f5ad | `scripts/diag-s62-m5.py` (add/add) | branch version (an earlier run note: slimmer, 800 ≤ main's ratchet 949) |
| 9847f5b | `scripts/run-m6-g4.sh` (add/add); `scripts/diag-s62-m6.py`; `REPORT.md` (4 hunks) | run-m6-g4.sh: branch version (266d407 updates it to main-identical later). diag-s62-m6.py: branch version. REPORT.md: main-only sections (an earlier run/an earlier run/Step-0/qmv) kept from upstream; the m6 section kept the branch's intermediate text so 266d407's updates apply cleanly (final m6 section text = main's, verified) |
| 3b6c0bc | `scripts/diag-s62-m7.py` (add/add) | branch version (885 ≤ main's ratchet 904) |
| 0bd8203 | `scripts/m7-{mtp,state}-probe.py`, `scripts/m7-probe1..5.py` (7× add/add) | branch versions (main's #16 versions are ruff-formatted copies of the same code) |
| a5af599 | `scripts/m7-mtp2-probe.py` (add/add) | branch version |
| 503c789 | `scripts/m7-g4-probe.py` (add/add) | branch version |
| e8cb010 | `REPORT.md` (m7 section) | m7 section auto-merged (main's #16 m7 section is text-identical to the branch's modulo one trailing blank line); remaining hunk = main-only Step-0+qmv content kept |
| c904b85 | `REPORT.md` (m8 section vs main's Step-0/qmv tail) | union: main's sections + branch's m8 section (m8 at file end) |
| e4b8d88, c04c34c, 11f38ca | — | clean |

No content was hand-edited. Main-only files (bench/qmv_mr*, bench/step0*, bench/an earlier run*,
bench/bench-decode.sh, bench/README.md, patches/qmv-mr-lxhoist.diff, scripts/an earlier run-*,
scripts/apply-qmv-mr-lxhoist.py, reports/tail-fusion-gdn.md, WORKSTATE.md,
qmm.hip, qmv_wide_mr_kernel.hip, tests/test_qmv_mr2_lx_instantiation.py) and main's
`build-wheel.sh` / `vendor.json` / `size-ratchet.json` (2-entry ratchet) are all
present at main's versions.

## Classification: CONTENT-NEUTRAL

Post-rebase tree = pre-rebase branch tree (`11f38ca`) + exactly main's base changes.
Every branch-owned m7/m8 file is **byte-identical** to `11f38ca`, including all audited
paths: `patches/m8-rope-single.diff`, `patches/m8-sdpa1pass-single-kernel.diff`,
`patches/m6-ropeq-rowinvariant.diff`, `bench/mtp-bench3.py` (full-state MTP restore),
`scripts/m8-{state-audit,sdpa-isolated,mtp148-bisect,mtp-state-trace,fa-bisect,kproj-flowprobe}.py`,
`scripts/diag-s62-{m5,m6,m7,sdpa}.py`, all m7 probes, run-m6/m5 scripts. No edits to the
sdpa patch / k_proj / rope / mtp-restore paths → only the 4 merge-blocking gates
re-run (not the full an earlier run check-line set).

Hermetic gates on the rebased tree: `uv run pytest -m "not gpu"` 12 passed + 1 skipped
(6 new = main's test_qmv_mr2_lx_instantiation); size gate OK (2 ratcheted; branch m5/m7
800/885 ≤ budgets 949/904); `scripts/m8-kproj-flowprobe.py` ruff clean (other m7/m8
script reds = the pre-existing set, unchanged).

## Guest re-verify (the build VM, fresh build /models/tmp/mlx-m8rebase)

- Fresh vendor clone @ pin `75ca881b2e82f7b4069c8564a203792aa16d8a29` (full sha verified
  against the m8r2 clone); the two post-rebase patch files applied; patched
  `rope.hip`/`scaled_dot_product_attention.hip` **md5-identical to the audited m8r2
  clone** (ef1b0896…/b0f19b48…); full source `diff -r` = identical (only cosmetic
  version-string + leftover build dirs).
- Wheel built in 6 min (venv-d python, same CMAKE_ARGS as m8r2); .so md5 `a889fba4…`
  (build-path/timestamp nondeterminism — source identical). Installed into venv-d
  (never venv-a). venv-d's `qwen3_next.py` already carries the RowInvariantRoPE patch
  (required by the A/B gate). Lock: FREE at start; per-job `flock` hold; FREE at end.
  No HSA faults in any log. venv-d .so restored to baseline (`af07d906…`) afterwards.
- Evidence: `/models/tmp/m8rebase-ev/` (direct-kernels.out, flowprobe.out,
  state-trace-run1/2.out, g4/, ab/).

| Gate | Expect (an earlier run) | Measured (fresh rebuild) |
|---|---|---|
| (a) direct k_proj T=1 vs T=2 | 0/160 + 0/160 | **0/160 (draft) + 0/160 (prev)**, all 8 FA layers × 20 trials; FA15 q/v 0/40; direct rope 0/20480; SDPA 6-config n=0 + determinism n=0 (ALL-N0) |
| (a) in-flow | n=0/1024, FA15_rope_1_out n=0 | **k_proj/k_norm/rope n=0/1024 on all 8 FA layers; FA15_rope_1_out n=0/1024**; fork@idx47 clean |
| (b) state trace | n_leak_steps=0, 2 runs identical, token@148=33443 | **n_leak_steps=0, first_leak=None, accepted 68/85 (both runs), token@148=33443 (both); run2 byte-identical to run1** |
| (c) G4 5-arm | M2-M0-LOSSLESS MATCH n=1024 + 4 controls MATCH | **M2-M0-LOSSLESS MATCH n=1024; M1-M0-CONTROL, M1-M2A, M2-A-B, M2-B-C all MATCH n=1024**; bench AR 69.24 tok/s (baseline 69.65; an earlier run 69.66 — parity), MTP K=1 61.06 tok/s, ACCEPTED 376/512 (same acceptance as an earlier run → consistent stream) |
| (d) A/B | M6-AB-OK | **M6-AB-OK; first diverging component NONE; all 16 components MATCH; m3 cross-checks match (layer-3 k 0/91136; layer-4 gdn ssm 515560/524288); the documented m5 artifact 1/4096 (9.766e-04) present as expected**; ab.log byte-identical to an earlier run's re-run log |

## Verdict

Rebase complete and clean: all 4 merge-blocking gates PASS on the fresh build of the
rebased content. PR #20 is `MERGEABLE` at `d4d433c`. Ready for the peer merge
(`--squash`; the an earlier run merge-mechanics notes still hold — m5/m7 resolve to the branch's
versions, which the rebase has already made the branch's committed state, and main's
size-ratchet.json is in the tree).

## CI fix (post-mergeable, <agent> forensics run <run-id>)
`lint-and-check` was red at d4d433c: 168 ruff errors (E501-dominated) in the 17 m7/m8
as-run audit probe scripts (branch-introduced; main's CI is green). Fix = the repo's
own as-run-evidence convention (np-rca-* precedent): all 17 flagged scripts added to
`[tool.ruff] extend-exclude`, kept byte-identical to as-run evidence. `ruff check .`
and `ruff format --check .` both clean at the new head. No kernel/patch/build input
touched - the 4-gate evidence stands.
