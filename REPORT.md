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
