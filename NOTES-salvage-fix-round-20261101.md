# Salvage fix round (t1097, 2026-11-01)

Branch t1082-exec-replay-fix, commit on top of a8fd141 (parent 1cf60ec). Goal: fix the allocator regression in a8fd141, verify on guest-a, deliver.

## Regression verdict (decisive A/B)

- Clean 1cf60ec wheel (dist-t1097-1cf60ec2, verified t1095-d=0, neutralize=2): PASSES the dead-run probe (STEP 1-6 + ALLTOKS, no crash, debug env).
- a8fd141 as-is: `free(): invalid pointer` (glibc, wrapper delete) in clean env; SEGV at STEP 2 in debug env.
- a8fd141 diff vs 1cf60ec is exactly: allocator.cpp free() `delete buf` in the arena path (item 1) + indexing.hip `n_rows==1` gate (item 2, conservative, not the cause).

## The a8fd141 item-1 bug (both parts)

free() arena path with `delete buf` is wrong for BOTH buffer kinds:

1. Deque wrappers: `arena_alloc` returns `&decode_arena_.wrappers[i]` (deque, stable addresses); the Buffer wrapping it is non-owning. Deleting the wrapper frees a pointer into the deque storage -> glibc `free(): invalid pointer` (p1, clean env).
2. Heap structs: `decode_arena_.contains()` is a 2 GB window check ([base, base+capacity)); unified-pool buffers whose data merely falls in the window are false positives. Instrumented build (t1097): two DISTINCT heap RocmBuffer structs pointing at the same 2 B gather-out data, both deleted -> UAF. The freed memory is reused by a HIP error string ("hipUnknownFunction (worker commit)" from hipLaunchHostFunc in Worker::commit, seen as `this` in the dead-run core) -> SEGV in CopyableHipEvent::record (hipEventRecord, no symbols in libamdhip64). `contains()` cannot distinguish true arena data from pool addresses.

Fix: revert the item-1 delete; the arena path is a plain `return` again (1cf60ec behavior). The per-free 56 B struct leak for false-positive pool buffers is accepted and documented in the code.

## Verification (guest-a, venv-m8, fixed wheel dist-t1097-final)

- Dead-run probe (debug env, N=6): PASS — STEP 1-6 + ALLTOKS, byte-identical tokens to the clean 1cf60ec run (271 8160 513 760 0 0; tokens 4+ nan = known pre-existing stale-replay behavior, not a regression).
- 14-token gate (clean env, GPROBE_N=14): PASS — full stream, no `free(): invalid pointer`, no SEGV (p1 fixed).
- CI parity: ruff check/format, size ratchet (origin/main), pytest 21 passed / 1 skipped.

## Debug env (definition) + [g1h] recorded-line evidence

"Debug env" (as used in the Verification above and in the t1097 probe runs) = `MLX_G1_MEMDUMP=1 MLX_G1_H2DLOG=1`, exactly the env prefix the t1097 dead-run probes were launched with in venv-m8 (`MLX_G1_MEMDUMP=1 MLX_G1_H2DLOG=1 ... python ggraph-probe.py`; the probe script itself sets the graph-decode mode). Quoted verbatim from the t1097 probe log (session record), the [g1h] recorded lines showing the 16-node re-delivery active (eight identical lines, one per recorded 8-byte staging value):

    [g1h] recorded 0x78bdc9806000 <- 0x78c3e9cd8000 (8)

## Pre-existing issues found (not fixed this round, documented)

- 8-byte OOB read in the `read_at` lambda (device.cpp, the G1K block, the MLX_G1_MEMDUMP-gated debug path): present in 1cf60ec (ASan), not introduced by a8fd141.
- a8fd141 REPORT.md "Verify" block cites 12:59 runs that used the older dist-t1074a44 wheel (pre-1cf60ec, no record_neutralize) — doc inaccuracy, corrected in REPORT.md (Verification, wheel identities).

## Repo hygiene fixed this round

- fp16_math.hpp: mirror carried a corrupted token `bfloarun-162float` -> restored to `bfloat162float` (matches fork 75ca881).
- Guest build requires the fast files MERGED (fork base + our K2/K4, keeping GdnPrep/GdnNormGated) — the repo mirror's fast files lack the GdnPrep sections (pre-existing mirror incompleteness, ~280 lines, not fixed here).
- Guest build must wipe the cmake binary dir + touch sources (incremental builds silently skip when git-archive mtimes predate existing .o files).

## F7: K3 zero-slot idx redirect A/B (guest-a, t1100 wheel, 2026-11-01)

Wheel built on guest-a from this branch (ae2c558 + the `rocm::` qualification fix below), fast files restored to the MERGED set (fork base + K2/K4, keeping GdnPrep/GdnNormGated), installed into venv-m8. Protocol: fresh process per run, `gpu-bench-inner.lock`, `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`, venv-m8 python; gates = gd-gate.py (120-tok, 56 steady-state tokens, ctx=14) and ggraph-probe.py (GPROBE_N=14 / 120).

| run | env | result |
|---|---|---|
| eager 14-ctx | — | `321 279 18483 383 ...` sane, 0 HSA errors, RC=0 (byte-identical to t1097 eager) |
| full probe N=120 | K3 on | 0 HSA errors, all 120 steps complete; ALLTOKS `271 8160 513 760 0 0 ...` (the dead-run attractor, as on t1097) |
| 14-tok probe a | K3 on | `271 8160 513 760 0 0 ...`, 0 HSA errors |
| 14-tok probe b | `MLX_GD_K3IDXOFF=1` | **HSA_STATUS_ERROR_MEMORY_FAULT in `gather_rows_kernel`** after STEP 3 (l0=nan) |
| 14-tok probe c | `MLX_GD_K3OFF=1` | **HSA_STATUS_ERROR_MEMORY_FAULT** after STEP 2 |
| gate14 a | K3 on | `321 279 18483 ...` sane, 0 HSA errors, RC=0 |
| gate14 b | `MLX_GD_K3IDXOFF=1` | `271 x 56`, 0 HSA errors, RC=0 |
| gate14 c | `MLX_GD_K3OFF=1` | `271 x 56`, 0 HSA errors, RC=0 |

**Verdict: the zero-slot idx redirect is NOT the cause of the 271 attractor — it is a MASK for a pre-existing bad idx.** Disabling it (b) or all of K3 (c) is strictly worse: the short probe hard-faults in `gather_rows_kernel` (OOB read) and the 120-tok gate degenerates to a uniform 271 attractor instead of running sane. The bad idx is present with K3 fully off, so the root cause is upstream of the redirect — consistent with the t1097 RCA direction (stale replay input): at replay, these gathers' idx buffers hold invalid values and the in-graph per-step H2D does not refresh them. The 271 attractor on short graph runs is the downstream consequence of the redirect converting the would-be fault into a row-0 misread.

**Recommendation: keep K3 (the redirect) as-is; do not change its semantics.** It is the less-bad behavior (no faults; the clean 120-tok gate passes with K3 on). Next RCA step: trace the faulting gather's idx provenance at replay (which buffer the gather reads vs which buffer the per-step H2D writes).

**Build fix found by the t1100 build:** b8eae8e had introduced a latent compile error (`gd_k3_off()`/`gd_k3_idxoff()` used unqualified in global namespace while declared in `namespace mlx::core::rocm`) — never caught before because the b8eae8e-era tree was never compiled on the guest (last built wheel was the 3690886 era). Fixed by qualifying the two use sites (`rocm::gd_k3_off()`, `rocm::gd_k3_idxoff()`).
