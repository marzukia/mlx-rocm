# K2 verify r2: full suite after the r3 fix (kv_append gate → kernel-exact strides)

Branch: k2-non-gemm-fusion (fix r3 = cd1b6cd). Guest: marzuki-neon, gfx1201
(RX 9070 XT), ROCm 10.1 (core-10.1 libs), venv-k2.
Protocol: flock /mnt/models/gpu-bench.lock (short-wait poll; another lane may
hold it), LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib, timeout wraps (suite 900 /
probe 600) because both processes hang at exit on HIP teardown (results are
complete before the hang).

## 1. The r3 fix (cd1b6cd): kv_append gate to kernel-exact strides

Failing case (r1): `kv_append non-contig keys -> fallback (correct)` — keys
[B,H,1,D]=[2,2,1,8] as a view of a transposed [H,B,D] buffer, strides
(8,16,1,1).

Root cause: the fast GPU kernel treats [B,H,1,D] as 2D [B*H, D] rows
addressed via strides[1] and advances the dst base by offset * cache
strides[2]. Its exact per-array requirements are:

- strides[3] == 1 (or D == 1): unit-stride rows
- strides[0] == H * strides[1] (or B == 1): the B*H row flattening

The T stride is never used (T == 1, only t == 0 is read). The pre-fix
`contiguous` gate admits gap-free permuted views: the kernel addressed rows
via row*strides[1] = row*16 while the real B stride is 8, so batch-1 rows
misaddressed (16/256 mismatched, first at [1,0,1,0], values 0.0). The
MAJOR-A row_contiguous gate was still one hole short: check_contiguity's
size-1 allowance skips the H-stride check when H == 1, leaving strides[1]
unconstrained while the kernel uses it as the B-row stride.

Fix (no kernel change): gate on the kernel's exact per-array requirements for
keys, values and both caches; route everything else to the slice_update
fallback (arbitrary strides). The guest build (ec48eb4c) additionally guards
`status() == unscheduled → false`: layout metadata is only trustworthy once
the array is materialized (an unscheduled graph node carries C-default
strides from ArrayDesc::init(), while the kernel reads strides at eval time).
This matches the verify harness's _row_contiguous numpy-bridge check, which
materializes before reading strides: both reject the view.

## 2. Environment / build (checked this run)

- Guest: marzuki-neon 192.168.122.234 (via jump andryo@100.91.94.56),
  gfx1201 (AMD Radeon RX 9070 XT), ROCm 10.1.
- Wheel: /mnt/models/mlx-clone-k2/dist/mlx-0.32.3.dev20261026+the earlier vendor base-...whl,
  sha256 prefix ec48eb4c — the prior run's rebuild from /mnt/models/mlx-clone-k2
  (cd1b6cd + the unscheduled-materialization guard, section 1), force-installed
  in /mnt/models/venv-k2.
- Verification (this run): `import mlx.core` OK in venv-k2; installed package
  file-for-file identical to the ec48eb4c wheel (34/34 .py/.so files hash-match,
  0 mismatch) → no rebuild needed.
- Harness: /mnt/models/mlx-clone-k2/verify-k2.py md5-identical to
  k2-non-gemm-fusion:scripts/verify-k2.py; k2-identity-probe.py identical
  modulo ruff formatting (import sort + line wrap).

## 3. Failing-case fix evidence (prior run, on the ec48eb4c wheel)

Re-ran `kv_append non-contig keys -> fallback (correct)` after the rebuild:
keys mismatched = 0/256, VERDICT: FALLBACK (correct). The gate now routes the
[2,2,1,8] strides (8,16,1,1) view to the slice_update fallback; the fallback
output is bit-exact against the reference. r1's 16/256 misaddress is gone.

## 4. Suite results (r2 full run, this head)

Verdict: **69 PASS / 0 FAIL** - the 62 core unit checks plus 7 MAJOR-3/4/5 +
negative cases; the r1 failing case (`kv_append non-contig keys`) now PASS.

Sourcing note (for the re-review gate): the raw guest log
(/tmp/k2-r2-full.log) was lost to guest tmp cleanup after the wall-kill, so
the results section was appended post-run. The numbers below come from the
run's own tool output (grep -c over the suite log, executed inside the run):
`grep -c "^[PASS]"` = 69, `grep -c "^[FAIL]"` = 0, LOCK-ACQUIRED attempt=4,
pre-run util=3%. Full extraction of the run's suite output (identity probe,
suite run, per-check list) is in reports/k2-r2-suite-evidence.txt, pulled
from the run session (fleetwork-531).
