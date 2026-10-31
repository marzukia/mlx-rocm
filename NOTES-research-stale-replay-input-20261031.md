# Research note: stale input on exec-cache replay (t1077, read-only research, 2026-10-31)

Full report with all sources: /srv/fleetwork/work/1077/REPORT.md (356 lines).

Symptom class: "replay reads the STALE input slot after the per-step H2D, no fault."

Ranked causes (ROCm/HIP + CUDA analogues):
1. H2D not ordered before the graph launch on the executing queue. Open ROCm bugs with the exact shape: ROCm/HIP#3902 (replay reads one-epoch-old buffer), rocm-systems#11403 (uncaptured SDMA memcpy rides outside the graph dependency chain; retired signal dropped = no wait), #13275 (FIXED: H2D staging race - RX 9070 XT is in its test table), #13877 (gfx1201 dispatch-scope change), HIP#3882 (memcpy-node deps not enforced on first launch after instantiate).
2. Silent hipGraphExecUpdate / SetParams failure -> exec keeps the PREVIOUS AQL packet, API reports success (rocm-systems#10714 at therock-7.14; check whether 10.1/HIP 7.16 has the check). Check EVERY return code; on failure re-instantiate the exec (MLX cuda-backend pattern: vendor/.../backend/cuda/device.cpp:515-525).
3. Kernarg staging slots never reclaimed within an exec lifetime (~exhaustion every ~4.1k updates: #10713/#11434/#14011) -> updates fail -> stale packet.
4. gfx1201 quirk: hipMemUnmap leaves stale GPU translation (TLB) - #13106/#13186.

Workarounds worth testing, in order:
1. CHEAPEST DISCRIMINATOR: after the per-step H2D, before hipGraphLaunch, read the input slot back D2H ON THE SAME STREAM and compare to the host value. Stale pre-launch = H2D staging/ordering class (#13275/#11403). Fresh pre-launch but stale in-graph = in-graph ordering (#3882/#3902) or stale baked pointer (#10714).
2. Do the H2D ON the launch stream, or hipEventRecord after the H2D + hipStreamWaitEvent on the launch stream.
3. H2D into a stable STAGING slot + in-graph D2D copy (tiny copy kernel) into the input slot, instead of the external H2D (named workaround in #3882).
4. Log every ExecUpdate/SetParams return code; re-instantiate on any failure; watch for a rising failure count (kernarg exhaustion).

MLX upstream (cuda backend - the pattern to mirror): #2317 capture-once + LRU exec cache keyed by node+deps; #2358 address stability via a pre-allocated arena (llama.cpp --cuda-graph pattern); #4699/#4700 serialize graph kernels (submission-order chain edges - the rocm port already has these); #4326: a failed commit must not poison the encoder.

---
## Addendum (t1076 report: /srv/fleetwork/work/1076/REPORT.md)

SIGNATURE DECODE: a real 9B-vocab token id (<=152k) read as float32 is ~1e-40 (subnormal). Our ~1e9 pattern is ANOTHER TENSOR'S DATA sitting in the slot - the replay read a DIFFERENT/OVERWRITTEN region, not a stale token. This re-ranks the causes:

1. INPUT SLOT VA MOVED between capture and replay; the captured kernel still references the capture-time pointer (H2D lands at the new VA, replay reads the old VA which now holds other arena data). Most-cited failure mode everywhere (PyTorch "same virtual addresses every replay", NVIDIA "parameters recorded by value", mlx #2358, Megatron-LM #7829). NOTE: matches the worker's own finding (capture advances arena offset by 48B, replay doesn't) - verify the slot VA is byte-identical across capture and every replay.
2. Arena reset/floor reuse re-allocates the slot to another buffer before the replayed read (VA stable, content clobbered).
3. hipGraphExecUpdate silent-success / stale packet (see above, #10714).
4. Ordering race: H2D on a stream the launch doesn't wait on; OR the H2D got captured as a memcpy node that re-copies from captured host staging, shadowing the per-step external H2D. vLLM documents exactly this class (side-stream H2D queued by the previous iteration) and fixes it with an explicit event sync before replay.
5. gfx1201 / ROCm 10.x executor defects (same arch): #13246 repeated hipGraphLaunch randomly corrupts heap (ROCm 10.1, open), #12330 branches serialized, #11499/#11501 dropped cross-stream dependency (input kernel runs before the H2D node = stale read).

DISCRIMINATING TESTS (cheapest first):
(a) read the input slot VA before the first replay and after each arena reset - if they differ, cause 1.
(b) write a canary (token+1) to the slot after the H2D, hipStreamSynchronize before replay - canary survives but model sees 1e9 = kernel reads a different VA (1/2).
(c) env bisect for the gfx1201 executor: DEBUG_HIP_GRAPH_CLASSIC_PATH=1, HIP_GRAPH_MEM_POOL=0, AMD_SERIALIZE_KERNEL=3 - behavior change implicates cause 5.
(d) check hipGraphExecUpdate return code each step (3).
(e) do the H2D on the replay stream, or capture it as a node with a pinned host buffer (4).

UNIVERSAL INVARIANT (all production systems - vLLM, SGLang, TRT-LLM, PyTorch, llama.cpp, MLX): kernel pointers are recorded BY VALUE at capture; per-step inputs reach replay ONLY via (a) write into a FIXED device VA (static pre-allocated buffer + H2D/D2D copy-in before replay) or (b) exec update. Sequence length is a tensor VALUE, never a shape; batch size is the only shape dimension (padding buckets).

---
## Addendum 2 (2026-10-31, t1074 salvage, probes r15-r25 on guest-a)

Confirmed by G1M node histogram (r20b) + G1T data-pointer tag (r18):

- The 1738-op capture step produces a **1038-node graph** (kernel=1022,
  memcpy=16, memset=0). The 16 memcpy nodes are ALL kind=1 (D2H, src=nil,
  one shared host dst ~0x7b9f0e75e000). **Zero H2D memcpy nodes** — the
  slot H2D `hipMemcpyAsync(..., hipMemcpyHostToDevice, capture_stream)`
  calls issued during stream capture never appear as graph nodes
  (the 2-op microstep graph: memcpy=0, kernel=1 — consistent).
  Either ROCm 10.1 vGPU executes them immediately at capture time or
  drops them; in either case the exec has no in-graph H2D, so the slot
  device buffer at first launch holds pre-capture content.
- r16-r18: APERTURE_VIOLATION at copy_gg_dynamic_nd on the **first
  graph launch (the capture-step launch itself)** — the replay step never
  begins (no prepare/slot-dump lines). r15: same wheel, no fault,
  deterministic garbage argmax 3216162876.
- Fix candidate: pre-launch external H2D of the 4 value slots on the
  launch stream immediately before hipGraphLaunch (end_record +
  decode_capture_replay), MLX_G1_NO_PREH2D=1 disables. G1P slot
  readback (host vs device after the pre-launch H2D, blocking D2H) and
  G1K dyn-offset arg dump (mincore-guarded) verify the slot content and
  the kernel's din/dout pointer args at launch time.
- Slot VAs are stable (prepare dump: slot1 = arena_base+0x600 at every
  step; copydyn *dout = same VA; slot data pointer = the VA the kernel
  arg points to). The G1T tag was broken before r18 (compared the
  RocmBuffer control-struct address, not the data VA).

---
## Addendum 3 (2026-11-01, t1080 salvage continuation, probes r35-r38)

Bisection (MLX_G1_BISECT, the exec of the 1738-op capture: 1038 nodes,
kernel=1022 memcpy=16):

| keep | result | probe |
|------|--------|-------|
| 1    | CLEAN  | t1080 |
| 125  | FAULT  | r33   |
| 62   | FAULT  | r34   |
| 31   | FAULT  | r35   |
| 4    | FAULT  | r38   |
| 2    | running| r39   |

- Fault: APERTURE_VIOLATION / HSA MEMORY_FAULT, kernel
  gather_rows_kernel (grid=[256,1,1] — the FIRST op of the decode step,
  the embedding Gather), faulting addr varies per run (0x779ef6e9d000 r35,
  0x78e5d449d000 r38, 0x7853b0a9d000 r34) — a freed pool VA, never in the
  arena (base 0x77c850400000, cap 2GB) nor the weights.
- The bisected exec is the FIRST 1738 capture's exec: it is launched once
  in its own step and re-launched as the "prefix" of the diverged replay
  step. The next step's fresh re-capture never launches (the queue is
  dead). So the faulty node is in the first 4 graph nodes of the 1738
  capture; keep=1 is clean, keep=4 faults.
- Separate (pre-existing) bug, the reason re-captures happen at all:
  "replay diverged at op 1 (Squeeze)" — the captured step's 2nd op is not
  "Squeeze" but the replay step's 2nd op is. Strict op-name sequence
  matching is the trigger for the divergence -> re-capture -> (this) fault.
  Needs op-name trace of capture vs replay step (next probe:
  MLX_G1_NODEDUMP + BISECT=4 for the first-node pointer attribution).
