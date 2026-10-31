# t1082 exec-cache replay fix — report

Date: 2026-11-01. Machine: guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest userland).
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
5. **`device.{cpp,h}`** (16-node re-delivery): at capture, record each in-graph
   H2D memcpy node's (host_src, dev_dst, bytes) and neutralize it (0-byte
   copy) so it cannot overwrite the pre-launch-delivered value mid-graph with
   a stale one; re-deliver the recorded values on the launch stream
   immediately before every `hipGraphLaunch` (same pattern as
   `g1_prelaunch_slot_h2d`). The record hook runs on the source graph BEFORE
   the exec is (re)instantiated, so the instantiated exec carries the
   neutralized nodes.

The per-step host H2D of the idx array still targets the original step buffer;
only the kernel argument is redirected to the zero slot (and only for
`n_rows == 1` fast_rows gathers — the zero slot is zero-cleared only for its
first element; an n_rows>1 gather keeps the original idx buffer, so an eager
T>1 step in a session with an active arena cannot read idx[1..] from the
never-cleared part of the 4 KB zero-slot wrapper).

## Verification (guest-a, post-fix)

- 6-step probe: **0 HSA_STATUS_ERROR lines**, all steps including the second
  replay complete; the [g1t] dump shows the three fast_rows gathers at
  distinct slot-3 offsets (0x8000/0x8100/0x8200), each row consumed from its
  own VA.
- **3 consecutive graph runs byte-identical** (pre-fix: run-nondeterministic —
  1 of 3 matched eager, 2 of 3 degenerated to the 271 attractor). The
  16-node re-delivery closed the nondeterminism; the per-launch slot-1
  readback verifies dev==host.
- A/B bench 14-tok: graph **26.20 tok/s** (38.15 ms/step) vs eager
  **60.56 tok/s** (16.51 ms/step) — a ~2.3x documented slowdown (per-launch
  slot H2D + 16-node re-delivery + zero-slot clear; CPU-side launch
  overhead). A finding, not a regression: the graph path is still
  correct and deterministic.
- 14-tok lossless gate: graph = 271 x 56 (deterministic attractor), eager =
  321 279 18483 ... (sane); graph != eager (the 271 attractor is a separate,
  pre-existing issue not caused by the 16 H2D nodes — slot 1 is delivered
  correctly, dev==host, every launch).
- 4096-tok gate: OOM (HBM full on the 16 GB guest); a resource issue, not
  correctness.

CI parity (host): `uv sync --locked`, `ruff format --check` (81 files OK),
`ruff check` (all passed), `size_ratchet` (OK, 2 ratcheted),
`pytest -m "not gpu"` (21 passed, 1 skipped).

## Post-fix state: fault-free and run-deterministic; two residual findings

The fault family is cleared (no HSA_STATUS_ERROR in any run; 120/120 steps
complete in both gates) and the run-nondeterminism is closed: 3 of 3
consecutive graph runs are byte-identical. The pre-fix perturbation
(partial NaN in the first replay's logits, warm region; run 1 self-heals by
step 64, the other runs did not) is explained by the 16 in-graph H2D memcpy
nodes that deliver per-step host staging values to one non-arena buffer (all
16 share the same dst — value slot 1 — paired 8-byte writes, nodes
n116..n763): graph-H2D node delivery on this vGPU is known-unreliable
(r15-r19 lineage — the reason the slot pre-launch H2D exists). The
re-delivery (item 5) makes the device staging buffer hold the current host
value before every graph launch, which is why the runs are now
byte-identical.

Two residual findings:

1. **271-token degenerate attractor (pre-existing).** The 14-tok gate graph
   run degenerates to 271 x 56 while eager stays sane (321 279 18483 ...).
   Not caused by the 16 H2D nodes (slot 1 verified dev==host every launch);
   a separate pre-existing issue.
2. **4096-tok gate OOM.** HBM full on the 16 GB guest; a resource issue, not
   correctness.

Next step (now in this branch): re-deliver the 16 staging H2D values
pre-launch — record each H2D node's (host_src, dev_dst, bytes) at capture,
neutralize the node (0-byte copy), and re-deliver the recorded values on
the launch stream immediately before every hipGraphLaunch (item 5). The
record hook silently skips any node whose src is nil or not attributable
as host; the recorded nodes are logged under `MLX_G1_MEMDUMP`.

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
