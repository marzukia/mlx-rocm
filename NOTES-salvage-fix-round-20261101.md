# Salvage fix round (t1097, 2026-11-01)

Branch t1082-exec-replay-fix, commit on top of a8fd141 (parent 1cf60ec). Goal:
fix the allocator regression in a8fd141, verify on guest-a, deliver.

## Regression verdict (decisive A/B)

- Clean 1cf60ec wheel (dist-t1097-1cf60ec2, verified t1095-d=0, neutralize=2):
  PASSES the dead-run probe (STEP 1-6 + ALLTOKS, no crash, debug env).
- a8fd141 as-is: `free(): invalid pointer` (glibc, wrapper delete) in clean
  env; SEGV at STEP 2 in debug env.
- a8fd141 diff vs 1cf60ec is exactly: allocator.cpp free() `delete buf` in
  the arena path (item 1) + indexing.hip `n_rows==1` gate (item 2,
  conservative, not the cause).

## The a8fd141 item-1 bug (both parts)

free() arena path with `delete buf` is wrong for BOTH buffer kinds:

1. Deque wrappers: `arena_alloc` returns `&decode_arena_.wrappers[i]` (deque,
   stable addresses); the Buffer wrapping it is non-owning. Deleting the
   wrapper frees a pointer into the deque storage -> glibc
   `free(): invalid pointer` (p1, clean env).
2. Heap structs: `decode_arena_.contains()` is a 2 GB window check
   ([base, base+capacity)); unified-pool buffers whose data merely falls in
   the window are false positives. Instrumented build (t1097): two DISTINCT
   heap RocmBuffer structs pointing at the same 2 B gather-out data, both
   deleted -> UAF. The freed memory is reused by a HIP error string
   ("hipUnknownFunction (worker commit)" from hipLaunchHostFunc in
   Worker::commit, seen as `this` in the dead-run core) -> SEGV in
   CopyableHipEvent::record (hipEventRecord, no symbols in libamdhip64).
   `contains()` cannot distinguish true arena data from pool addresses.

Fix: revert the item-1 delete; the arena path is a plain `return` again
(1cf60ec behavior). The per-free 56 B struct leak for false-positive pool
buffers is accepted and documented in the code.

## Verification (guest-a, venv-m8, fixed wheel dist-t1097-final)

- Dead-run probe (debug env, N=6): PASS — STEP 1-6 + ALLTOKS, byte-identical
  tokens to the clean 1cf60ec run (271 8160 513 760 0 0; tokens 4+ nan =
  known pre-existing stale-replay behavior, not a regression).
- 14-token gate (clean env, GPROBE_N=14): PASS — full stream, no
  `free(): invalid pointer`, no SEGV (p1 fixed).
- CI parity: ruff check/format, size ratchet (origin/main), pytest 21 passed /
  1 skipped.

## Pre-existing issues found (not fixed this round, documented)

- 8-byte OOB read in the `read_at` lambda (allocator.cpp, MLX_G1_MEMDUMP-gated
  debug path): present in 1cf60ec (ASan), not introduced by a8fd141.
- a8fd141 REPORT.md "Verify" block cites 12:59 runs that used the older
  dist-t1074a44 wheel (pre-1cf60ec, no record_neutralize) — doc inaccuracy.

## Repo hygiene fixed this round

- fp16_math.hpp: mirror carried a corrupted token `bfloarun-162float` ->
  restored to `bfloat162float` (matches fork 75ca881).
- Guest build requires the fast files MERGED (fork base + our K2/K4, keeping
  GdnPrep/GdnNormGated) — the repo mirror's fast files lack the GdnPrep
  sections (pre-existing mirror incompleteness, ~280 lines, not fixed here).
- Guest build must wipe the cmake binary dir + touch sources (incremental
  builds silently skip when git-archive mtimes predate existing .o files).
