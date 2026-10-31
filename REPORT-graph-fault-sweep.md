# REPORT — graph-decode replay fault: fix-candidate sweep

Verdict: **NO-CANDIDATE-CLEARED** — exit per protocol (no PR). All three
candidate cycles (plus the un-gated pin of the same candidate wheel)
reproduce the deterministic second-capture launch fault.

Base = the new public main merge of the graph-decode work (main @ 588e78b,
vendor ref 75ca881, self-vendored tree). The fix-hunter build tree on the
build VM was verified byte-identical to this tree's
`vendor/mlx-rocm-upstream` rocm backend files, so every probe below is
directly on this tree's design.

## Design under test (landed in main)

Per-step recapture -> end_record -> ExecUpdate/fresh-instantiate ->
immediate launch; two-slot double buffer; deterministic decode arena (2GB
single hipMalloc, bump allocator with floor freeze/reset-to-floor) for
capture-path allocations; G1X persistent copy of kernel arg slots
(slot-owned keepalive); [pd] fault-catch + faultVA/buffer-map
instrumentation retained (faults non-fatal, probe survives).

## Candidate sweep (fresh process per run; flock + 15min timeout; graph
armed via the pilot env gate; N=6 decode steps, 14-token prompt)

CLEARS criterion: zero `HSA_STATUS_ERROR_MEMORY_FAULT` lines across all
capture+launch steps, 2 consecutive runs, then the 29K-context probe once.

| # | Candidate | Result | Fault signature |
|---|-----------|--------|-----------------|
| 1 | as-built arena wheel (tree + [pd] diagnostics, no pin code) | FAULT | step 4 (2nd capture launch): `HSA_STATUS_ERROR_MEMORY_FAULT`, gather_rows_kernel<bf16,int> grid [256,1,1], addr 0x7d80e0de7000, then queue hang |
| 2 | REFRESH: re-point every kernel node's arg params via ExecNodeSetParams to a fresh slot-owned copy of the source graph's current values, before each launch | FAULT | `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (access beyond largest legal address) on copy_g_byval, 1050th dispatch; refresh applied (SetParams 1024 kernel nodes); queue hang |
| 3 | SINGLE-SLOT: two-slot double buffer disabled (one capture slot) | FAULT | step 4: same as #1 (gather_rows, addr 0x773c0b5e7000, 630ms stall) |
| + | PIN (same wheel as #2/#3, untested in the as-built wheel): hold every graph-referenced device VA (kernel arg values + memcpy/memset ptrs) at end_record; frees of pinned VAs HELD until next capture begin | FAULT | step 4: same as #1 (gather_rows, addr 0x71996cde7000); 86/463 VAs pinned on captures 1/2 |

Notes:
- #2's failure mode is distinct from the stale-VA class (beyond-largest-
  legal-address = garbage pointer, not an unmapped-but-close VA). Caveat:
  the refresh block keeps its values+pointers array in a DEVICE block;
  HIP kernelParams expects a HOST array of pointers to values, so part of
  #2's signature may be the param-array location, not just staleness.
  #2 is not exculpatory for the root cause either way.
- #3 shows cross-slot reuse is not sufficient cause (single slot still
  faults on the second capture's launch).
- #+ shows the stale value is not explained by a FREE (held buffers,
  no free/realloc between capture and its launch, still faults) — the
  buffer CONTENT is what goes stale, not its mapping.

## Narrowed suspect (corroborated by the diagnostic base's [ld] data)

The faulting VA is a COMPUTED address: `base + r*row_size` with a garbage
`r` (~60-133GB from the arena base; kernel = the embedding/lm_head row
gather, r = the step's input token index). The idx buffer is an ARENA
buffer; at capture time its content is correct (e.g. 279 = the step's
input token), so the staleness is a CONTENT overwrite between capture and
replay, not a mapping loss:

- the decode queue launch is asynchronous; the NEXT capture's host-side
  `decode_arena_reset_to_floor()` + subsequent arena allocations can
  overwrite the previous launch's still-pending input/idx buffer before
  the launched kernel has read it;
- the graph has zero memcpy nodes — the input tensor's H2D happens outside
  the capture, so replay depends on the arena slot's content surviving.

Candidate next fixes (not in this sweep's budget): (a) synchronize the
dedicated decode queue (completion of the launched step) before the next
capture's arena reset-to-floor; (b) move input/idx (and any
replay-read-before-write) buffers OUT of the bump arena into stable
per-slot allocations.

## Gates

- Lossless 14tok/29K byte-identical: NOT REACHED (probe gate failed).
- A/B decode bench: NOT REACHED.
- CI parity on the deliverable tree (main @ 588e78b):
  `uv run ruff format --check .` = clean (78 files); `uv run pytest -m
  "not gpu"` = 21 passed, 1 skipped.

## Evidence

Probe logs on the build VM: probe-g1pin1 (candidate 1), probe-g1pin2
(candidate 2), probe-g1pin3 (candidate 3), probe-g1pinC1 (pin),
probe-g1v9 (diagnostic base re-verification). Candidate wheel build log:
build-t1071c2 (incremental from the full configured build tree; the
candidate tree needed one compile fix — PinReg definition moved above its
first use in allocator.cpp).
