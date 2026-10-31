# WORKSTATE — run-1045 (ent4 final: port run-1044 fixes + per-tensor E<3.9 gating)

Branch `phase3-entropy-compress`, base 7f9d1bd (run-1042 final). Succeeds run-1042
(compressor + lossless gates + A/B, REPORT-entropy.md) and run-1044 (3 compressor
fixes on its own v1-format script; net-zero finding — see below).

## Timeline
- 04:00Z: worktree up; run-1044's 3 commits (764fc08 timing/RSS/traceback, 4f4a9e1
  int32 cumsum, 8ec0580 stream-end OOB) confirmed; manual port required (run-1044
  targets a different v1 format).
- 04:40Z: E discrepancy resolved — run-1044's 3.9707 "dataset" was measured on
  /models/qwen3-9b-mlx-clean/model.safetensors (reproduced its per-tensor
  values bit-for-bit by running its exact script on that file). run-1042's 3.713
  was measured on the final served model /models/qwen3-9b-mlx (independent
  re-measurement: aggregate E_huff 3.7368; per-class el 3.747 / attn 3.728 /
  mlp 3.736; per-tensor max 3.754; 16 distinct symbols everywhere).
- 05:40Z: step 2 done — per-tensor E<3.9 gating (default --gate 3.9; --gate 0 =
  compress-all legacy). Gated .ent4 on the final 9B model (the build VM):
  250/250 tensors compressed (per-tensor E ≤ 3.754, all < 3.9; file size
  4,192,642,800 B = run-1042's 4,192,642,550 B + 250 flag bytes, streams
  otherwise identical) — the gate is a no-op on this dataset. Per-class
  re-measure (compressor's own path): el 3.7472 / attn 3.7331 / mlp 3.7392
  bits/sym; TOTAL raw 4.4764 GB → comp 4.1849 GB (x1.0696, = run-1042).
- 06:20Z: step 3 done — lossless gates on the build VM: rebuilt wheel (mlx-m8r2
  tree: updated ent4.h/ent4_json.h + load.cpp +3 alloc/memcpy, pip wheel OK),
  CPU verifier 250/250 bad=0 on the gated .ent4 (flag byte + +3 pad), GPU
  kernel gate 3/3 bad=0 (ktest updated: flag byte + 3-byte pad; tensors 0/
  1/249), full-stack A/B (fresh wheel, gated file): baseline vs ent4-gated
  logs byte-identical (forward logits + 32-token generation).
- 07:30Z: step 4 done — A/B bench (flock): 64-token arm baseline
  66.20/65.91/65.83 (median 65.91) vs ent4-GATED 65.85/65.90/65.84 (median
  65.85) tok/s → ~0.1 % overhead, texts identical. 29k-ctx arm (A/B/B2,
  both modes, all rc=0): A 72.29/73.16, B (warmed) 61.86/61.60 decode win
  (parity, LOSSLESS-GATE=PASS both), B2 fresh 60.85/57.25 (first-touch
  decode inside the measured window — methodology artifact, warmed arm is
  the steady-state number).
- 08:10Z: step 5 — REPORT-entropy-final.md + final push.

## Key findings
- The two "datasets" never disagreed about the code: they are two different
  model builds — see timeline 04:40Z. Decision C's E<3.9 gate is a no-op on
  the final model (all 250 tensors E ≤ 3.754, all compressed; the 6.5 %
  memory saving from run-1042 stands) but is the safety mechanism for models
  where a class runs above 3.9 (e.g. the -clean build's attn 4.21 / mlp 4.08
  would go plain).

## Gate
- hermetic: ruff format/check (no new findings vs 7f9d1bd), pytest -m "not gpu"
- GPU: green (step 3 — CPU verifier 250/250, kernel gate 3/3, full-stack A/B
  byte-identical; the build VM, flock)
# WORKSTATE — run-1059 (continuation of run-1047, mlx-rocm lever 3a: HIP graph capture of the AR decode step)

Continuation of run-1037 (killed at the 4h wall, 13:48:20Z, zero commits, clean tree) and
run-1024 (09:44Z kill). Guest path placeholder in repo content = /models (real guest FS
/models).

## Status (latest, g1x72-g1x75 era)

Fault characterization (reproduced cleanly, g1x72/g1x75, GPROBE_N=6/4):
- The fault is DETERMINISTIC on the SECOND decode capture+launch (first capture
  always fine), and slot-order-independent (slot-swap experiment g1x70 was
  inconclusive at GPROBE_N=3 — only 1 capture; at GPROBE_N=6 the swap build
  faults identically on the 2nd capture). Both captures are FRESH
  instantiates (two-slot double buffer, each slot's first end_record).
- The `[ndr]` line in node dumps is a mislabeled VA-range map (device.cpp
  ~1551, hardcoded prefix), not a replay dump; `decode_capture_replay` is dead
  code (zero callers). Design = per-step re-capture -> end_record ->
  ExecUpdate/fresh-instantiate -> immediate launch.
- g1t trace: capture 1 arena_hw=0 (no decode-arena allocations — buffers came
  from the normal pool, VA 0x7600f.../0x7606...), capture 2 arena_hw=496640
  (485KB, arena-based VAs 0x760014800000+off). Both ovf=0. Full [nd]
  value-diff of the two captures: 3970/9372 values differ (earlier K686-only
  diff was misleading). The second graph is self-consistent (its baked VAs are
  mapped at launch: arena is one 2GB hipMalloc), so the faulting VA is not a
  direct node arg from the arena.
- kp[5] (arg_reduce_partial in_strides.data_[0]) = 0x7fff014b0740 (host stack
  pointer where a numeric stride is expected) in BOTH captures — by-value struct
  copy is suspicious, but that node has ndim=0 (kp[6] low 4B=0) so strides are
  never read: harmless for that node. Pack path (templated) copies structs
  fully; the 8B G1X deep copy truncates struct args ONLY on the non-pack
  (module/raw) path — check whether any decode node goes there.
- No faulting VA obtainable from HSA (HSA_ENABLE_DEBUG=1 HSA_LOG_LEVEL=5),
  dmesg, or journalctl (no amdgpu page-fault lines). -> Node bisection.
- G1B bisection added (g1x77 build): MLX_G1_BISECT=N re-points kernel nodes
  after the first N at g1b_noop (copy.hip) via
  hipGraphExecKernelNodeSetParams (ROCm 10.1 HIP has no hipGraphRemoveNode).
  Plan: no-bisect baseline (expect fault) -> keep=344 -> narrow. Note:
  neutralizing a node can make downstream nodes fault on garbage (mapped, so
  no VA fault) — direction of bisection error is bounded.

## Guest state at digest time (13:52Z)

- Orphan run-1037 process RUNNING: `flock -w 1800 gpu-bench.lock -c ... timeout 300
  venv-m8/bin/python /tmp/ggraph-probe.py` (started 13:48:20, env
  MLX_ROCM_GRAPH_DECODE=1 MLX_G1Y_DEBUG=1 GPROBE_N=6), writing
  /models/tmp/probe-g1x67.out; timeout -> done ~14:38Z, then the lock frees.
  WAIT for it (sibling run-1026 phase2-gdn may also take the lock); bench only with
  flock -w 1800 max.
- Abench iterations (all in /models/tmp/): A0 (11:27, MEASURED baseline: 71.01 tok/s,
  29k ctx, 147 toks, decode median 13246us, max-step 120959us HSA-fault stall,
  stall-free ~75.4) -> A1 (empty) -> A1d (12:24 debug, 5.5MB err) -> A1e (12:59 node-dump,
  19MB err, 1 MEMORY_FAULT with node dump) -> A1f (13:11) -> A1g (13:25) — A1f/A1g DIED
  with `hipMalloc failed (device VRAM exhausted); managed/GTT fallback disabled:
  out of memory size=8589934592` on the warm pass (8GiB request). OOM mode, not the
  graph fault.
- Latest orphan run probe-g1x67 (g1yf free-trace build): STEP 1/2 ok, STEP 3
  (tok=0) -> `HSA_STATUS_ERROR_MEMORY_FAULT`. **The run-1024 stack-arg-retention fault
  candidate still reproduces with the G1X value-copy fix in place** (the fix copies 24
  arg slots x 8B into a `decode_capture_persist_alloc` block; g1yf2 arena trace shows
  heavy offset wrap/reuse: `same_off_entries` growing, `wrap_n=1330`, repeated
  `free off=0` — the persistent copy's arena space is likely recycled while the graph
  still references it, so at replay the "persistent" block holds garbage -> a computed
  device VA (src + idx*row_size) faults outside the arena. The faulting VA is a
  computed address, not a direct node arg (run-1024 finding, now corroborated, not yet
  root-caused).

## Build / install state (guest)

- Vendor tree: /tmp/mlx-m8r2 = pinned 75ca881b + local patches (graph-decode G1/G1X,
  node-dump [nd], g1t arena trace, g1yf/g1yf2 free trace; 713 insertions across 12
  files incl. device.cpp/device.h/allocator.cpp/eval.cpp/load.cpp/quantized/qmm.hip).
- venv-m8 = /models/tmp/run-1012-29k/venv-m8: mlx 0.32.3.dev20261030+75ca881b
  (wheel sha 440985001a7e..., installed 17:48:27; carries G1/G1X/node-dump
  [nd]/g1t arena trace/G1B bisection; G1B2 = type histogram + memset
  neutralization added by run-1059, build g1xb6 in flight). Needs
  LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib. Probe requires
  MLX_ROCM_GRAPH_DECODE=1 to arm the graph (pilot gate).
- Bench harness: /models/tmp/gd-lev1-bench.py (greedy 200-tok, warm pass
  discarded, per-step argmax host read) + gd-lev1-run.sh. Model /models/qwen3-9b-mlx
  (qwen3-9b 4bit). ggraph-probe.py = 6-step graph probe (GPROBE_N).
- Baseline reference: A0 = 71.01 tok/s @29k ctx; ungraphed AR ref 62.4 tok/s @14tok
  window (9B guest, warm).

## run-1059 continuation (2026-10-30 17:50Z-; wall 3h -> ~20:51Z)

Salvage done: guest state intact (vendor tree /tmp/mlx-m8r2 = pinned 75ca881b +
1056-line uncommitted patch; venv-m8 wheel = last successful pip wheel, g1xac-era
17:48:27 install; builds g1xa8 FAILED, g1xa9/g1xab/g1xac = fast-path reuse of the
same wheel sha 440985001a7e...). G1X copy storage = std::vector blocks kept in
decode_cap_keepalive_[slot] (grows, only cleared by decode_capture_destroy which
is only called by the C API) — blocks ARE alive at replay; the g1xa1-5 "dedicated
pool" idea was redundant; the fault is NOT the G1X block lifetime.

Probe env gotcha: graph decode pilot is gated on MLX_ROCM_GRAPH_DECODE=1
(eval.cpp gd_env_enabled); without it the probe runs eager (~13ms/step) and never
arms — early g1xb0/b1/b2/b3 "clean" runs were graph-OFF (invalid).

Re-verified with current wheel + MLX_ROCM_GRAPH_DECODE=1:
- probe-g1xb4 (no bisect): FAULT at STEP 3 (616ms, HSA_STATUS_ERROR_MEMORY_FAULT),
  hang until timeout. Fault still alive in the latest wheel.
- probe-g1xb5 (MLX_G1_BISECT=0, all 688 kernel nodes -> g1b_noop): STILL FAULTS
  at STEP 3. BISECT=1 also faults (g1xad). So the fault is independent of kernel
  execution (0 or 1 live kernel both fault) — it is in a node type G1B never
  neutralized or in launch-time param machinery.
- Graph has ZERO memcpy nodes (no hipGraphAddMemcpyNode anywhere; no [nd] M lines)
  but hipMemsetAsync exists on the eval stream (matmul.cpp:1461/2040,
  moe_swiglu.cpp) — during stream capture these record GRAPH MEMSET nodes, which
  are invisible to the [nd] dump (dumps only Kernel+Memcpy) and to G1B (kernels
  only). Prime suspect: a memset node writing a buffer whose VA is dead at replay
  (normal-pool buffer freed post-capture, or arena offset recycled).
  (allocator.cpp:685 hipMemset(0x7F) is MLX_GRAPH_POISON_FREE-gated only.)

G1B2 patch applied to /tmp/mlx-m8r2 (device.cpp G1B block): node-type histogram
([g1b] types: kernel/memcpy/memset/other) + MLX_G1_BISECT_MS=1 neutralizes memset
nodes (size=1, value=0). Build g1xb6 in flight (pip install --force-reinstall
--no-deps; log /models/tmp/build-g1xb6.log).

Plan (next, post-rebuild g1xc1):
1. Re-baseline: probe (graph on, no bisect) must fault at step 3 (confirms the
   rebuilt wheel reproduces; the wheel sha changed vs the pre-reboot one).
2. Discriminate launch-param vs stream-side: (a) MLX_G1_NO_LAUNCH=1 (capture
   but never launch — if still faults, the fault is NOT the graph launch:
   deferred free / prewarm / D2H); (b) NO_LAUNCH off + MLX_G1X_OFF=1 +
   BISECT=0 (G1X deep copy off, params = caller stack which is DEAD by replay
   — if this faults and full-G1X+BISECT=0 also faults, the param-copy theory
   needs the exec-instantiate timing, not launch).
3. If NO_LAUNCH is clean -> launch-time param copy; instrument:
   hipGraphExecKernelNodeSetParams with all-NULL kernelParams + BISECT=0
   (if clean -> param copy; if still faults -> grid/func/handle level).
4. If NO_LAUNCH faults -> instrument the deferred free (log every
   hipFreeAsync VA + the graph's referenced VAs at capture end; diff at the
   fault step).
5. Then: fix + probe clears N steps (GPROBE_N=30) + lossless gates 14tok/29K
   (greedy ALLTOKS byte-identical vs ungraphed, fresh process, warm discarded,
   window skip=56) + A/B bench both ctx arms under flock -w 1800.
6. Commit + push (mirror + fork) after EACH verified step. Leak rule + CI gates as
   before. Baselines: A0 = 71.01 tok/s @29k ctx; ungraphed AR 62.4 tok/s @14tok.

## Checkpoints (pushed)

- 13:55Z: run-1047 digest (WORKSTATE + state notes) on phase1-hip-graph.
- ~18:25Z: run-1059 continuation digest (BISECT=0 still faults; memset-node suspect;
  G1B2 patch + build in flight).
- ~19:35Z: probe-g1xb8 (G1B2 build): graph has ONLY 688 kernel nodes
  (kernel=688 memcpy=0 memset=0 other=0) — no memset nodes at all; BISECT=0
  + BISECT_MS=1 STILL faults at step 3. Fault is NOT in any node execution
  and NOT in memset. With HIP_LAUNCH_BLOCKING=1 (g1xb9): step-3 launch
  stalls 611ms = the faulting op; AMD_LOG trace (g1xbA, 31M lines) shows the
  last graph launch SUCCEEDS, then a hipLaunchHostFunc (worker commit) and
  the event-poll loop; queue error = sticky HSA fault. Suspect now:
  launch-time param copy from a dead VA, or a stream-ordered free (graph
  deferred free) of a still-referenced buffer, or slot-0-first-capture state
  (arena reset_to_floor on capture 2; capture 1 buffers were normal-pool,
  capture 2 arena-based).
  G1B2 histogram patch fixed (hipMemsetParams has width/elementSize, no size).
  G1X_OFF + NO_LAUNCH env gates added (g1xba). Build g1xba FAILED (host OOM
  on guest: hipStreamCreateWithFlags out of memory — amdgpu queue stuck after
  the fault; VRAM showed 60MB used; no proc held /dev/kfd). Rebooted the
  the libvirt guest VM (virsh reboot on the operator's tailnet host) at ~19:10Z;
  GPU init OK after reboot. REBOOT SIDE-EFFECT: /models/tmp (vdb-root/tmp,
  bind-mounted to /tmp per fstab line added 2026-10-27 by the operator) was WIPED at
  boot (some unseen boot script); run-1012-29k venv + all guest builds/probes
  gone. Reconstruction: salvage vendor tree (partial patch: g1t/node-dump
  infra) + /tmp/g1x/g1x.diff (G1X/G1V/G1B base; applied clean) + G1B2 +
  g1xba (G1X_OFF/NO_LAUNCH) + g1b_noop in copy.hip; fresh venv-m8; build
  g1xc1 in flight. Local survivors: g1x.diff, g1b*.py, g1xba-patch.py,
  ggraph-probe.py. run-1012-29k.py/run.sh = operator-owned 640, unreadable (recreate
  for the lossless gate later; recipe in WORKSTATE).
- ~20:45Z: vendor fork identified = <op-fork-user>/mlx (NOT marzukia/mlx —
  that repo does not exist); the vendor ref = 75ca881b2e82... exact
  pin match. Full tree cloned on guest at the pin. First full-tree builds
  failed: (a) CMAKE_ARGS not passed (no_gpu stubs linked: symbol
  duplicates); (b) copy.hip KVAppend undeclared — the guest's 1056-line
  patch ALSO carried fast.h/fast.cpp/fast_primitives.h KVAppend ops + 25
  more delta files (K2 fused rms_norm_residual + KV append, r4d_gdn
  kernels, qmm, leak renames). Salvage vendor (task-repo snapshot) =
  fork + those patches; synced all 25 delta files + my 6 patched rocm
  files onto the pin clone; build g1xc4 in flight (CMAKE_ARGS =
  -DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
  -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1).
  Probe plan (next session / budget extension): (1) baseline no-bisect
  must fault at step 3; (2) MLX_G1_NO_LAUNCH=1 discriminator (launch vs
  stream-side: deferred free / prewarm / D2H); (3) G1X_OFF+BISECT=0.
  AMD trace (g1xbA) showed the last graph launch SUCCESS then
  hipLaunchHostFunc (worker commit) then event-poll detecting the sticky
  HSA fault — with all-node neutralization the launch reads only the
  G1X/pack param blocks (alive) + func handles, so the param-copy theory
  weakens; the deferred-free (hipFreeAsync of a still-referenced buffer)
  and prewarm theories strengthen. Fault is deterministic on slot 0's
  FIRST capture (step 3) — the capture where
  decode_arena_reset_to_floor() runs (floor frozen after capture 1) and
  the arena gets its first real use (cap2 arena_hw=485KB vs cap1
  arena_hw=0, normal-pool).
- ~21:40Z: rebuild chain (post-reboot, /models/tmp wiped at boot):
  vendor fork = <op-fork-user>/mlx, the vendor ref (75ca881b exact pin).
  Full-tree build needed the guest's 1056-line patch reconstructed =
  g1x.diff (applied) + G1B2 + g1xba + g1b_noop (my 6 rocm files) + 25
  delta files from the salvage vendor (K2 ops: fast.h/fast.cpp/
  fast_primitives.h rms_norm_residual + kv_append; r4d_gdn kernels;
  qmm.hip incl. qmv_lx_kernel.hip + qmv_wide_mr_kernel.hip) — with 3
  fixes: (1) 6 files had a broken "t1->run-1" leak-rename
  (bfloarun-162float etc.) → restored FORK versions; (2) fast.* merge =
  fork base (gdn_prep/gdn_norm_gated) + salvage K2 blocks (script
  /tmp/merge-fast.py, is_equivalent block appended at EOF — compiles,
  links OK only if present: g1xcA linked but import failed on missing
  KVAppend::is_equivalent → g1xcB with the 1496-line fast.cpp);
  (3) kp.func cast (void*)g1b_noop (ROCm 10.1 HIP: func is void*).
  Build cmd: CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=
  gfx1201 -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1"
  pip install --force-reinstall /models/tmp/mlx-m8r2 (+ mlx-lm).
  Wall OVERRUN (started 17:51Z, wheel built ~21:30Z); probe sequence
  pending: baseline (expect step-3 fault) → MLX_G1_NO_LAUNCH=1
  discriminator → G1X_OFF+BISECT=0.
- ~23:50Z: WHEEL BUILT (g1xcG) + imported; link fixes: is_equivalent
  block inside namespace fast (not EOF — vtable miss), g1b_noop in
  namespace rocm (not global anon — undefined symbol), copy.hip stray
  "namespace {" from my edit removed, CMakeLists +gdn_prep.hip (fork
  has GdnNormGated::eval_gpu there; salvage CMakeLists lacked it).
  Baseline probe (graph ON, no bisect) FAULTS at STEP 3 — REPRODUCED.
  NEW: faulting kernel identified = gather_rows_kernel<hip_bfloat16,int>
  (embedding/lm_head row gather; grid [256,1,1]), faulting addr
  0x7ec11f9e7000, completion_signal=0x0, rptr=1054 wptr=2081.
  NO_LAUNCH probe (capture, skip hipGraphLaunch AND worker commit):
  ALL 6 STEPS CLEAN (no fault; tokens 0 at steps 4-6 = expected, graph
  never runs). Discriminator: fault is in the LAUNCH-or-COMMIT path,
  NOT node execution (BISECT=0) and NOT stream-side direct launches.
  Both baseline and BISECT=0 fault at the SAME gather_rows dispatch →
  the fault is in the gather_rows DISPATCH SETUP (kernarg/signal
  deref at launch) — points at a freed semaphore/signal or kernarg
  backing allocation captured in the graph, not the kernel's own
  memory access. NEXT: NO_COMMIT discriminator (launch runs, commit
  skipped) — build g1xcH in flight; if NO_COMMIT clean → fault is in
  worker commit (deferred hipFreeAsync / hipLaunchHostFunc); if it
  faults → fault is in hipGraphLaunch itself (dispatch resource
  deref). After that: log graph-referenced VAs (param block + node
  params) vs arena free list at step-3 launch; check
  0x7ec11f9e7000 membership.
  Probe env note: the fresh mlx-lm API dropped return_hidden —
  ggraph-probe.py patched (logits = lm(..., cache=cache)).

## run-1062 (2026-10-31): continuation after run-1059 wall-kill

Round-3 results (g1xcH, probe logs on guest /models/tmp/):
- NO_LAUNCH (skip hipGraphLaunch + worker commit): 6 steps CLEAN.
- NO_COMMIT (launch runs, commit skipped): FAULTS on the 2nd capture's
  launch (STEP 5, slot 1) => fault REQUIRES hipGraphLaunch replay;
  commit/deferred-free exonerated. HANG ANALYSIS: gather_rows_kernel,
  faulting addr 0x763e8c5e7000 (arena-family VA 0x76xx...),
  kernarg_address 0x76230df80200, grid/workgroup [256,1,1].
- Kernel (indexing.hip): r = idx[row]; srow = src + r*row_size;
  orow = out + row*row_size. Fault VA is a COMPUTED address (src or out
  base + idx*row_size) ~60GB away from the 0x7600.. region => unmapped,
  not the 2GB arena (arena base 0x7600.., cap 2GB).
- run-1062 work: (1) synced guest g1xcH tree back into vendor (7 files:
  device.cpp, allocator.h, eval.cpp, copy.hip, CMakeLists.txt, device.h,
  rope.hip); (2) added G1V diagnostics: [va] arena owner map (wrappers)
  at end_record + pre-launch, [ld] gather-node exec arg dump with
  hipMemGetAddressRange + DtoH mapping probes at pre-launch, [pd] same
  post-launch after sync; build g1xcI; (3) node-arg diff baked-vs-live
  + VA ownership of the faulting VA; (4) fix = pin KV buffer or
  ExecUpdate the stale node param (G1B SetParams plumbing exists);
  (5) verify probe N=6 clean, lossless gates 14tok/29K, A/B bench.

## t1080 salvage (exec-cache graph decode, 2026-10-31)

Branch `exec-cache-t1080-salvage` (off 7ef78a9). Goal: make same-shape decode
steps REPLAY a cached exec instead of re-capturing 1,740 ops; clear the
graph-decode A/B gap (-60.8%).

- r35-r39 bisection: keep=1 CLEAN; keep=2..125 FAULT (copy_gg_dynamic_nd /
  gather_rows at the 2nd recapture or at a mis-armed replay launch).
- r40/r41 NODEDUMP: the 1,738-op capture's first nodes point only at
  arena slots (base+0x500..+0x900, always mapped) and static weights —
  the stale pointer is not visible in the first 8 nodes.
- **r41 key finding**: "replay diverged at op 1 (Squeeze)" was a 2-op
  argmax MICRO-step (Gather|Squeeze) mis-armed as a replay: the arming
  matched on the first op NAME only. The micro-step then launched the
  1,740-op exec at step end with its first op suppressed → the step's
  token came out NaN (STEP 4 l0=nan), and the same mis-armed launch is
  the r35/r38 faulting launch.
- **Fix 1 (in)**: `gd_step_is_embedding_lookup` — the replay branch now
  requires the first op's first input to be 2-D (the [vocab, embed]
  embedding gather); micro-steps (1-D lm-head gather, ArgReduce, ...)
  run eager without disturbing arming.
- Build r42b (guest t1063-c1 tree: NOTE — the guest device.cpp is the
  t1063 variant with g1s sentinel + its own [nd] dumper; the local repo's
  device.cpp diverged (no g1s). A concurrent session patched the guest
  file's hipMemAlloc/kp.numParams errors at 12:05Z). Probe r42b:
  GPROBE_N=40 (forces a KV-cache 32→64 resize → diverged replay → 2nd
  recapture), G1Y+G1_NAMES+NODEDUMP+MEMDUMP, no bisect.
- Gates next: 14tok (gd-gate.py) + 4096tok (gd-gate4k.py, 4096-token
  prompt, 120 toks / warm 64, ALLTOKS byte compare graph vs eager);
  A/B: gd-benchN.py GD_N=1100, 3 reps graph vs eager.
