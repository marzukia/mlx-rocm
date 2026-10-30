# K3 Phase 1 re-run: GDN segfault fix + numerics (gfx1201, the build VM)

Date: 2026-10-26 (guest UTC). Submitter: submitter (worker an earlier run).
Scope: fix the an earlier run GDN bf16 `copy_general_general` segfault, verify GDN numerics
(recurrent + chunk paths vs the op-loop fallback and an f32 reference). No tok/s
benches (separate 1200 s phase-2 run).

## Guest environment note

All GPU work ran on **the build VM** (<guest-ip> via `operator@<tailnet-ip>`
jumphost) under `flock /models/gpu-bench.lock`. ROCm 10.1, RX 9070 XT
(gfx1201). **The guest runs a 32-CU partition of the 54-CU card** (HIP bind log:
`cus=32 warp=32 lds=64KB`) — every timing/occupancy observation from this guest
is partition-scaled, not whole-card. Build tree `/models/mlx-clone-k3`,
venv `/models/venv-k3`.

## 1. The crash and backtrace status

an earlier run: 4× `Segmentation fault (core dumped)` in the 04:11 wheel during GDN bf16
numerics verify, deterministic at the smallest case via
`flock … python k3_repro.py recurrent` (output ends `graph built / Segmentation
fault`).

Backtrace status: **not captured live.** On the re-run the crash did not
reproduce in any configuration:

| hammer | 04:11 wheel (an earlier run chunked copy, has `chunked_copy_stream_separated`) |
|---|---|
| plain `k3_repro.py recurrent` ×25 | 0 crashes |
| under `gdb` ×15 (`-ex run -ex bt`) | 0 crashes, `Inferior exited normally` |
| accumulation loop ×10 | 0 crashes |
| concurrent (4 procs × 4) | **16/16 crashed** — the crash mode |

No cores were on disk (`ulimit -c 0` in the guest shell; the an earlier run cores were not
preserved). The an earlier run gdb backtrace (release build, imprecise symbolization)
pointed at `allocator.cpp:908/923` — the per-chunk `hipMemcpyAsync`/stream
lifecycle region of `chunked_copy_stream_separated` (bd061d6 lines 894–929).
The crash is **load-gated, not code-deterministic**: it fired 4× in a row at
04:11 (minutes after the 04:00 build, KFD state freshly perturbed) and
reproduced 16/16 under concurrent load, but 0× in ~50 serialized runs across
the day.

## 2. Root cause

1. The an earlier run/an earlier run patch added **chunked host-shadow staging copies**
   (`chunked_copy_stream_separated`, ≤100 MB chunks, `MLX_ROCM_D2H_CHUNK_MB`
   override) to avoid the an earlier run/an earlier run lineage: single large (≥300 MB, 1.2 GB
   model-weight scale) shadow copies intermittently hanging in
   `kfd_wait_on_events` with the GPU idle on gfx1201/ROCm 10.1.
2. That design **created and destroyed a stream per chunk**. On this
   gfx1201/ROCm 10.1 configuration the KFD completion event of an async copy is
   not retired promptly (the an earlier run/an earlier run/an earlier run hang lineage); destroying the
   per-chunk stream while its event is still pending races the runtime
   stream/event bookkeeping → segfault in the copy path (an earlier run, an earlier run backtrace
   at the chunked-copy region).
3. The re-land `85b521e` **dropped the chunked copy entirely** (git: 37e2834
   chunk ≤100 MB, d82ce84 async chunked, bd061d6 an earlier run salvage — all with chunked
   refs; 85b521e and everything after: 0). The 07:44 wheel (an earlier run instrumented
   build) therefore had plain single-shot `hipMemcpy` shadow copies and could not
   crash — but it silently lost the hang-avoidance property for large copies.

So: the an earlier run crash is a **an earlier run stream-lifecycle bug** (per-chunk
`hipStreamDestroy` vs never-retired KFD event), and the re-land masked it by
removing the chunking.

## 3. Fix (this commit)

`mlx/backend/rocm/allocator.cpp`: re-added the chunked host-shadow staging copy
as `shadow_chunked_copy()`:

- ≤100 MB chunks (default; `MLX_ROCM_D2H_CHUNK_MB` override, `0` = single copy,
  i.e. the pre-chunking behavior).
- Each chunk on its own **freshly created** stream: `hipMemcpyAsync` +
  `hipStreamSynchronize` — the per-chunk queue reset is the hang-avoidance
  property (an earlier run, kept).
- **No `hipStreamDestroy`** — the an earlier run/an earlier run segfault fix. One small bounded
  leak per chunk per shadow copy (a 1.2 GB copy leaks 12 streams; negligible for
  a server process) in exchange for no stream-lifecycle race. Streams (and their
  KFD events) live to process exit, matching how the main copy path already
  behaves.
- If `hipStreamCreate` fails, the chunk falls back to synchronous `hipMemcpy`
  (legacy path).
- Applied at all three shadow-copy sites: `ensure_host_shadow` (D2H),
  `flush_host_shadow` (H2D), `flush_all_dirty_shadows` (H2D).

`mlx/backend/rocm/gated_delta_update.cpp`: kept the an earlier run `MLX_GDN_DEBUG`
instrumentation (env-gated fprintf of buffer pointers around the GDN eval; no
behavior change when unset).

## 4. Verification

New wheel built on the guest (incremental, 2 files + relink), installed in
`venv-k3` (note: `venv-k3/bin/pip` has its prefix pointed at `/models/venv`,
so the .so files were copied directly into `venv-k3` site-packages; verified by
md5 + `nm | grep shadow_chunked_copy`).

Crash repro (new wheel):

| run | result |
|---|---|
| `k3_repro.py recurrent` plain ×5 | 0 crashes |
| under `gdb` ×1 | clean exit, no stack |
| 300 MB buffer D2H readback ×20 (3 chunks each) | 0 crashes |
| 300 MB buffer H2D create ×10 (3 chunks each) | 0 crashes |
| 300 MB big-copy under `gdb` | clean exit |
| serialized `k3_repro.py recurrent` ×16 | 0 crashes |
| true-concurrent (4 procs × 4 iters) | **0/16 crashed** (04:11 wheel: 16/16) |
| `MLX_GDN_DEBUG=1 k3_repro.py recurrent` | passes, debug prints sane |

Numerics (B=1, Hk=4, Hv=8, Dk=Dv=128, T=1 recurrent / T=3 chunk; bf16 inputs,
f32 state; separate processes per mode because the op-loop fallback is a
per-process static):

| T | pair | out max_abs | state max_abs | state max_rel (|ref|>0.05) | verdict |
|---|---|---|---|---|---|
| 1 | fused vs fallback | 0.0 (exact) | 1.19e-07 | 1.3e-06 | PASS |
| 1 | fused vs ref | 3.7e-3 | 1.19e-07 | 1.8e-06 | PASS |
| 1 | fallback vs ref | 3.7e-3 | 1.19e-07 | 1.6e-06 | PASS |
| 3 | fused vs fallback | 7.8e-3 (1 bf16 ulp) | 3.5e-3 | 5.5e-2 | PASS |
| 3 | fused vs ref | 1.0e-2 | 3.5e-3 | 5.5e-2 | PASS |
| 3 | fallback vs ref | 3.7e-3 | 2.4e-07 | 2.2e-06 | PASS |

Tolerances: out (bf16) `max_abs ≤ 2e-2`; state (f32) `max_abs ≤ 1e-2` and
`max_rel ≤ 6%` on significant elements (|ref| > 0.05). The T=3 chunk-path state
deviation (3.5e-3 abs, 5.5% on the smallest significant element, mean 1.6e-4) is
the expected bf16 intermediate (the chunk-scan A-matrix is bf16); the op-loop
fallback matches the f32 reference to 2.4e-07, confirming the reference and the
recurrent path are f32-exact. Sums: T=1 fused/fallback out −2.459069 (exact
match), state −35.872292/−35.872284, ref −35.872280; T=3 out
5.386710/5.467781/5.448370, state 16.051422/15.982661/15.982656.

**All checks pass: no segfault (plain/gdb/big-copy/concurrent — the 04:11
wheel's 16/16 crash mode is 0/16 on the fix), recurrent and chunk GDN paths
match the Python fallback and f32 reference within bf16 tolerance.**

## 5. Follow-ups (not blocking)

- The an earlier run trigger was load-gated (16/16 concurrent on the 04:11 wheel, 0×
  serialized); set `ulimit -c unlimited` in the guest bench shell so the next
  occurrence leaves a core.
- Bounded stream leak in `shadow_chunked_copy` (documented above); a shared
  persistent stream would remove it but changes concurrency semantics — revisit
  in phase 2 if stream counts ever matter.
- `venv-k3/bin/pip` prefix points at `/models/venv`; wheel installs there
  land in the wrong venv. Use direct .so copy (as done here) or fix the venv.
