# Adversarial independent check — PR #3, marzukia/mlx-rocm

Ticket: t1094. Checker: fresh session, no authoring context (read-only code check;
GPU probes not re-run). Branch `t1082-exec-replay-fix` @ `1cf60ec8845ea459a71d6c587632793c9ac4521a`, base `main` @ `da9eeca`. Worktree head verified == `1cf60ec` (worktree was on `da9eeca` at launch; branch checked out, tip matches the pushed head). 13 files, +1593/-48 as briefed.

## Verdict summary

**PASS** — all three check lines pass; no demonstrated correctness regression in the
verified workloads; claims are consistent with the code (with the staleness notes
below). Four non-blocking findings are flagged for the follow-up cycle.

## Check line 1 — CI parity (exactly as .github/workflows/ci.yml): PASS

Ran locally on the branch head, in order:

| step | result |
|------|--------|
| `uv sync --locked` | OK (pytest 9.1.1, ruff 0.17.2) |
| `uv run ruff check .` | All checks passed! |
| `uv run ruff format --check .` | 81 files already formatted (matches the commit's "81 files") |
| `uv run pytest -m "not gpu"` | 21 passed, 1 skipped (matches the commit exactly) |
| `uv run python scripts/size_ratchet.py --base HEAD` (size job) | SIZE GATE: OK (2 ratcheted) (matches the commit) |

## Check line 2 — persona leak: PASS (0 hits)

Scope: full PR diff (`da9eeca..1cf60ec`, 13 files) + all 12 commit messages
(2,294 combined lines, /tmp/pr3_scope.txt). Patterns and hit counts:

| pattern | hits |
|---------|------|
| `marzuki-*` hostnames (body/diff; author/committer fields exempt) | 0 (only `marzuki.local` in commit author fields = exempt fleet identity; `marzukia/mlx-rocm` is the repo name, not a hostname) |
| `100.64.0.0/10` tailnet IPs (full 100.64–100.127 sweep) | 0 |
| `andryo@*`, `/home/andryo` | 0 |
| `neonvm@`, `192.168.122.x`, `/mnt/models` | 0 |
| `*.junkyard.sh` | 0 |

`guest-a` / `RX 9070 XT` / `gfx1201` appear — fleet/hardware terminology used in
AGENTS.md itself, not on the check list.

## Check line 3 — line-wrap: PASS (layout-is-content exemption)

- New .md (NOTES-gate-29k-infeasible.md, NOTES-research-stale-replay-input-20261031.md,
  vendor/mlx-rocm-upstream/REPORT.md): prose hard-wrapped at 71–80 columns.
- New C++ comments in the vendor files: wrapped at ≤76–80 columns.
- Commit messages: bodies wrapped 72–76 (standard git), subjects single-line.
- The research note's long single-line list items are single logical lines, not
  width-wraps.

The repo's pre-existing convention is the same: `AGENTS.md` at base is
hard-wrapped at ~80 (max 94), base vendor C++ comments wrap at 80/84, and earlier
NOTES files in the range are wrapped the same way. The new files follow the
inherited project style (layout-is-content), so no manual-wrap FAIL. No anomalous
wrap width found in any file.

## Claim checks

### (a) Stream-ordering correctness — PASS (with note)

- Per-launch: `g1_prelaunch_slot_h2d` (4 value-slot H2Ds via `hipMemcpyAsync(st)`
  + zero-slot `hipMemsetAsync(st)`) and `g1_prelaunch_recorded_h2d` (16 recorded
  H2Ds via `hipMemcpyAsync(st)`) both run on the launch stream immediately before
  `hipGraphLaunch` on BOTH paths: `decode_capture_end_record` (first launch) and
  `decode_capture_replay` (every replay). Host buffers are `hipHostMalloc`-pinned,
  so the async H2Ds are safe.
- The per-op slot H2Ds (KVAppend slot 2, SDPA kL slot 0, SliceUpdate slot 1) are
  issued via `encoder.launch_kernel` on the op stream: recorded (or dropped) at
  capture, suppressed at replay — no double-run, no legacy-stream residue.
- The only legacy-default-stream ops: the one-shot initial zero-slot clear
  (`hipMemsetAsync(..., 0)` + `hipStreamSynchronize(0)`) in `decode_zero_slot()`,
  which runs once per arena incarnation inside `gd_step_prepare`, i.e. BEFORE the
  capture window (the slot is pre-allocated there), and is fully synchronized.
  Remaining blocking HIP calls (g1s scratch alloc/memset, G1P/G1Y readbacks) are
  env-gated diagnostics executed outside any capture window.
- NOTE: the claim "none left on the legacy default stream" holds for every
  per-launch op; the one-shot initial clear technically still lands on stream 0
  (synchronized, pre-capture, once per incarnation) — immaterial in practice.

### (b) Layout arithmetic — PASS (with residual-risk notes)

- 4 KB wrapper vs 2 KB gather row: `decode_value_slot3_cursor` advances in 256 B
  steps; two 2 KB rows fit exactly (0, 2048 → off+adv == 4096, not >), a third
  returns the 4 KB capacity → caller falls back to the stable-arena
  `malloc_async` (address-stable per step; the graph bakes the capture-step VA).
  No row can spill past its wrapper — the r47/r50 fault class is closed by
  construction.
- Slot layout: reserve(5) in both capture and replay; slots cached (base
  unchanged) leave offset at 0x5000 → conditional extra reserve(5) → step
  re-allocations start at 0xA000 in both the first capture (slots fresh:
  0x5000 + 5×4096) and every replay (arena reset → same sequence). Slot VAs the
  graph baked in stay valid; step buffers no longer alias 0x5000..0x9FFF.
  `decode_slot3_cursor_` resets in `decode_arena_begin`/`decode_arena_reset`, so
  capture/replay cursor sequences are identical.
- K3 idx routing: `h_indices[0]` → zero slot only for `fast_rows` gathers with
  `out.nbytes() <= 4096` in graph-decode mode; the per-step idx H2D still targets
  the original step buffer (only the kernel arg is redirected), as documented.
- Residual risk 1 (name-only op validation): replay validates the op NAME
  sequence, not shapes. A same-named op with a different `out.nbytes` would shift
  the slot-3 cursor offsets relative to the baked VAs → the step's CPU-visible
  `out` buffer and the graph's write target diverge (silent stale read, bounded,
  no fault). For fixed 1-token decode steps (the verified workload) sizes are
  stable; the r52 [g1t] dump (rows at 0x8000/0x8100/0x8200, each consumed from
  its own VA) and the byte-identical run 1 support this.
- Residual risk 2 (zero-slot 8-byte clear): only `sizeof(int64_t)` is cleared
  (init + per-launch); the fast_rows kernel reads `idx[row]` per row. With
  `n_rows == 1` (all decode steps: big/T>1 steps run eager) only `idx[0]` is
  read — safe. See finding 3 for the n_rows>1 latent case.
- Zero-slot sharing as both `dynamic_offset_in` and `dynamic_offset_out` is safe:
  the copy kernels only READ `*offset_out` (verified in copy_general_dynamic);
  the in-graph memset is skipped on the zero-slot path (only the malloc path
  memsets).

### (c) Eager (non-graph) path — PASS (with one latent hazard, finding 3)

- All new behavior is gated: K3 gather routing + zero-slot idx on
  `graph_decode_mode()` (which also requires the arena active for a valid slot →
  eager with arena inactive falls back to the base path byte-for-byte);
  KVAppend `off_dev = nullptr` outside decode mode → base pre-advanced
  addressing (kernel `off = 0`, new t-stride args contribute 0); SDPA
  `kL_slot = nullptr` default → `params.kL` by value; the zero-slot hook in
  `copy_gpu_inplace` returns null outside decode mode → original
  malloc+memset path. Eager is unchanged in the verified workloads.
- One deliberate eager change (documented in-code): `copy_general_dynamic` iterates
  the shape product instead of `out.size()` — a correctness fix (offset-shifted
  writes would index past the buffer); identical grid for dense copies.
- `scaled_dot_product_attention.hip` has its OWN `AttnParams` (no `kL_slot`) —
  not affected. No other call sites of the re-signed KVAppend kernels or the
  new externs; `g1_check_hip` signatures/decls are consistent across
  device.cpp / copy.hip / indexing.hip / flash_attention.hip; includes
  (cxxabi.h, sys/syscall.h, dlfcn.h) present.

### (d) NOTES/REPORT.md consistency — PASS (with staleness notes)

- 20b0f44 (REPORT.md era) diagnosed the 16 in-graph H2D staging nodes as the
  known-unreliable graph-H2D delivery and set "Next: re-deliver the 16 staging
  H2D values pre-launch"; 1cf60ec implements exactly that (record
  (host_src, dev_dst, bytes) at capture; neutralize to 0-byte via
  `hipGraphMemcpyNodeSetParams` BEFORE the exec is (re)instantiated — verified
  in `decode_capture_end_record`: the hook runs before both the ExecUpdate
  refresh and the fallback re-instantiate; re-deliver via
  `hipMemcpyAsync` on the launch stream before every `hipGraphLaunch`, both
  paths). The record list is cleared and refreshed per (re)capture.
- STALENESS 1: REPORT.md still says "Next step (not in this branch)" for exactly
  the change 1cf60ec adds to this branch, and its Verification section
  (r52/r53, 1-of-3 byte-identical) predates the 16-H2D fix; the 3/3
  byte-identical + A/B 26.20/60.56 + OOM results live only in the 1cf60ec commit
  message. Recommend a REPORT.md refresh before merge (not a code issue).
- STALENESS 2: REPORT.md header "Date: 2026-07-09" is inconsistent with its
  own content (it references the 4 KB wrapper state of 2582cae, 2026-11-01).
- STALENESS 3 (minor): the 16-node shared dst is called "value slot 1" in the
  1cf60ec message but "one non-arena buffer" in REPORT.md — same object,
  imprecise labels.
- No claim in the docs contradicts the code.

### (e) Documented results — consistent

- 0 HSA_STATUS_ERROR: documented for r52/r53 (REPORT.md) and the final 6-step
  probe (1cf60ec) — covers the briefed r52–r54c probe window.
- 3/3 consecutive graph runs byte-identical: 1cf60ec commit (post-fix state,
  superseding the 1-of-3 of 20b0f44 — different probe sets, not a contradiction).
- A/B 14-tok: graph 26.20 (38.15 ms/step) vs eager 60.56 (16.51 ms/step) ≈ 2.3x
  SLOWDOWN — documented as a finding, not a pass claim. Consistent with the
  design (per-step slot H2D + 16 re-delivery H2Ds + memset per launch; the
  CPU-side launch overhead is what the cpu/step logging measures).
- 271-token degenerate attractor: documented as pre-existing / not caused by the
  16 H2D nodes (slot 1 verified dev==host every launch).
- 4096-tok gate OOM (authoring run): documented as "a resource issue, not
  correctness". The diff adds negligible HBM (10×4 KB arena wrappers, ~5×8 B
  pinned host, small diagnostic scratch) — it CANNOT explain a prefill-stage OOM
  on the 16 GB guest (same memory-pressure class as the documented eager
  prefill OOMs in NOTES-gate-29k-infeasible). Sibling t1092 passing 120/120 in
  both modes is consistent with a resource/thermal event. Flag: not
  diff-explainable.

### (f) src/python/mlx_rocm.egg-info/SOURCES.txt — FLAG (non-blocking)

The PR's only SOURCES.txt change adds `vendor/mlx-rocm-upstream/REPORT.md` — an
sdist/build byproduct of building on the authoring machine. egg-info is tracked
build metadata (pre-existing in the tree, only `dist/` is gitignored) and does
not belong in the tree; the hunk should be dropped from the PR (and
`src/python/mlx_rocm.egg-info/` ideally untracked + gitignored). Not a FAIL
trigger on its own.

## Findings (non-blocking, for the follow-up cycle)

1. **Small unbounded leak (new in this PR).** `Gather::eval_gpu` K3 path
   `new rocm::RocmBuffer{...}` per small gather out; `RocmAllocator::free()`
   early-returns when `decode_arena_.contains(buf->data)` WITHOUT deleting the
   control struct (the HBM itself is arena-owned, but the ~56 B struct is
   leaked per freed small gather out — a few per decode step → ~KB/min
   unbounded in a long-lived serve; the deferred-free path leaks the same way).
   Fix: delete the struct in the arena early-return (or reuse the
   `free_rocm_buffer` shape).
2. **G1H2D silent-skip risk / verification gap.** The record hook silently skips
   any H2D node whose src is nil or not attributable as host (no log without
   `MLX_G1_MEMDUMP`); the 20b0f44 [g1m] dump reported host src as (nil) for
   these nodes. If the node params return nil src, neutralization + re-delivery
   are no-ops and the slot-1 dev==host readback cannot distinguish the 16-node
   re-delivery from the slot H2D. The 3/3 deterministic outcome is consistent
   with the fix working, but the quoted probe output does not directly evidence
   the re-delivery (the "[g1m] slot N: K node(s) recorded + neutralized" /
   "[g1h] recorded" lines are not quoted). Low-medium risk; recommend quoting
   those lines in the follow-up probe.
3. **Latent zero-slot idx hazard for n_rows > 1.** `graph_decode_mode()` stays
   true across eager steps once the pilot is prepared (cleared only on
   tear-down), and the arena stays active between steps. An eager T>1 (big)
   step in a session with an active arena therefore runs the K3 path: a
   `fast_rows` gather with `n_rows > 1` and `out.nbytes() <= 4096` would read
   `idx[1..]` from the never-cleared part of the 4 KB zero-slot wrapper
   (garbage index → off-range src row read; not reached by any documented probe,
   since 1-token decode always has n_rows = 1 and the initial prefill runs
   before arming). Fix: replace `h_indices[0]` only when `n_rows == 1` (cheapest)
   or clear the full 4 KB slot per launch.
4. **One-shot initial zero-slot clear on legacy stream 0** (see (a) note) —
   immaterial (synchronized, pre-capture, once per incarnation); listed for
   completeness of claim (a).

## Disposition

VERDICT: PASS — check lines 1–3 pass; no demonstrated correctness regression in
the verified workloads; docs are consistent with the code modulo the staleness
notes. Findings 1–3 + (f) are flagged for the PR author / follow-up cycle.
