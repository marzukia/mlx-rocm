# t1082 exec-cache replay fix — report

Date: 2026-07-09. Machine: guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest userland).
Model: Qwen3-9B (GDN linear attention). Base: `5254793` (4 KB value/zero slot wrappers).
Branch: `t1082-exec-replay-fix`.

## The fault family (pre-fix)

- r47 (256 B wrappers): capture OK, first replay OK, **second replay faulted**
  (HSA_STATUS_ERROR_MEMORY_FAULT in `gather_rows_kernel`, faulting address ~38 GB
  below the arena). Root cause: a 2 KB gather row (512 x int32) written into its
  256 B arena chunk overflowed into the adjacent chunk, which held the 1-element
  idx buffer shared by three fast_rows gathers; the next replay read the row's
  float bits as an index.
- r50 (4 KB wrappers, K3 out/idx routing without the zero-clear): the zero slot
  (the idx buffer's VA) was clobbered by an 8 KB step buffer before the first
  replay; the gather read a huge negative index (same fault class).
- r51 (K3 + pre-launch zero-clear): the gather fault cleared, exposing the next
  in-graph clobber: `copy_gg_dynamic_nd` (aperture violation, code 0x29) — its
  dynamic offset-in buffer (the zero-slot VA, 0x…9000) was written by an 8 KB
  step buffer earlier in the graph.

Common root cause: with the 5 slot wrappers (0x5000..0x9FFF) cached across
steps, the per-step arena re-allocations restarted at 0x5000 and aliased the
slot region, so any step buffer (gather row, dyn offset target) landed on top of
a slot value the graph later read.

## The fix (this branch)

All in `mlx/backend/rocm/`:

1. **`eval.cpp` (`gd_step_prepare`)**: when the slots are cached (the arena
   already exists), the 5 reserved wrappers are not consumed by slot allocation;
   reserve 5 more so the step layout is identical to the first capture step
   (slots fresh, starting at 0xA000). The VAs the graph baked in stay valid at
   replay, and step buffers no longer alias the slot region.
2. **`indexing.hip` (`Gather::eval_gpu`, K3)**: in graph decode mode a gather
   output of <= 4 KB is routed to the persistent value slot 3, each row at a
   distinct 256 B-aligned offset (per-step cursor, so rows do not clobber each
   other in-graph — a decode step has several fast_rows gathers). The
   fast_rows index is read from the zero slot (zero-cleared before every
   launch by `g1_prelaunch_slot_h2d`). Outputs > 4 KB keep the stable-address
   arena buffer (the row fits its own block there).
3. **`allocator.{h,cpp}`**: `decode_value_slot3_cursor(size)` — the per-step
   cursor into slot 3 (reset with the arena, returns the 4 KB capacity when a
   row does not fit, the caller falls back to a stable-arena buffer).
4. **`device.cpp`** (already in the base): pre-launch zero-slot clear.

The per-step host H2D of the idx array still targets the original step buffer;
only the kernel argument is redirected to the zero slot.

## Verification (guest-a)

- Probes r52/r53 (6-step, `MLX_G1_MEMDUMP=1`): **0 HSA_STATUS_ERROR lines**,
  all steps including the second replay complete; capture step finite; the
  [g1t] dump shows the three fast_rows gathers at distinct slot-3 offsets
  (0x8000/0x8100/0x8200), each row consumed from its own VA.
- Lossless 14-tok gate (120 decode steps, warm=64, 56-token window):
  **graph run 1 = byte-identical to eager** (321 279 18483 383 279 4732 513 …).
  Eager is deterministic (two runs, identical).
- 4096-tok gate (512 MB arena, `MLX_ROCM_GRAPH_DECODE_ARENA_MB=512`):
  120/120 tokens complete in both modes, 0 HSA_STATUS_ERROR; graph = 271 x 56,
  eager = 279 x 56 (both degenerate single-token loops, different attractors).
- Second/third consecutive graph 14-tok gate runs: 271 x 56 (degenerate).

CI parity (host): `ruff format --check` (80 files OK), `ruff check` (all
passed), `size_ratchet --base HEAD` (OK, 2 ratcheted), `pytest -m "not gpu"`
(21 passed, 1 skipped).

## Finding: second replay is fault-free, but not yet token-stable

The fault family is cleared (no HSA_STATUS_ERROR in any run; 120/120 steps
complete in both gates), but token correctness is run-nondeterministic:
1 of 3 graph 14-tok runs matches eager byte-for-byte; 2 of 3 (and the 4k
gate) degenerate to the same 271 attractor. The probe shows the same
perturbation as partial NaN in the first replay's logits (warm region; run 1
self-heals by step 64, the other runs do not).

Diagnosis: the capture step runs eagerly inside the capture window, i.e. *before*
`g1_prelaunch_slot_h2d` (which runs at end_record, then before each replay).
The pre-launch H2D covers the 5 slots + the zero-slot clear only. The graph
additionally contains 16 in-graph H2D memcpy nodes that deliver per-step host
staging values to one non-arena buffer (all 16 share the same dst, paired
8-byte writes, nodes n116..n763); graph-H2D node delivery on this vGPU is
known-unreliable (r15-r19 lineage — the reason the slot pre-launch H2D exists).
At the capture step those values are delivered eagerly (reliable); at replay the
graph's H2D nodes may deliver stale/undelivered values, so the graph kernels
read a process-dependent perturbation (fresh-arena initial memory + the capture
step's own trajectory). Depending on the perturbation the trajectory either
self-heals within the warm window or lands on the 271 attractor.

Next step (not in this branch): re-deliver the 16 staging H2D values pre-launch
(the same pattern as the slots — record each H2D node's (host_src, dev_dst,
size) at capture; blocking H2D before every graph launch). The [g1m] dump
currently prints the host src as (nil) — the record hook must capture it.

## Disposition

Per the t1082 brief: the fix did not clear the "sane tokens, 2 consecutive
runs" criterion (second replay is fault-free; tokens stable in 1 of 3 runs),
so this is the one follow-up cycle. No PR to `marzukia/mlx-rocm` from this
branch; salvage ref = `t1082-exec-replay-fix` pushed to the mirror
(`/srv/fleetwork/mlx-rocm.git`).

Guest-tree diagnostic patches (probe tree only, not part of this branch's diff
except where mirrored in the repo): `device.cpp` pre-launch zero-slot clear
(already in the repo), `eval.cpp` cached-slot reserve (now in the repo),
`gd-gate4k.py` script fixes (`global PROMPT`; removed the 29k leftover lines in
the 4k branch).
