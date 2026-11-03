# GDN launch-merge probe (48.93 t/s anchor) — negative verdict: no feasible merge design

Goal per task: (1) MEASURE per-phase GDN op time at the 48.93 t/s anchor config
(3-run median, FIRST20 must match); if the GDN ops total < ~1 ms/step →
negative verdict, stop. (2) If justified, DESIGN a merged GDN step (batched
multi-layer pass, persistent kernel, or launch-level batching) with hook points
and a state-correctness argument. (3) If concrete, PROTOTYPE behind an env
gate.

**Verdict: GATE 1 PASSES (GDN total ≈ 4.9 ms/step ≥ 1 ms) but GATE 2 FAILS —
all three merge options are blocked by the residual-stream data dependency
between GDN layers. Stop before prototype.** No prototype was written; no
kernel/model change was made. Deliverables below are the measurement, the
design analysis with the dependency argument, and the one feasible
out-of-scope follow-up candidate.

Environment: guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest userland),
venv-canon wheel (SBCAR md5 `02a54359…` pre and post — unchanged), anchor
protocol = mtp-bench6 @322-iter loop, env
`MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1
MLX_ROCM_QMV_ROWS_MIN_N=1 REFWD_M1=1`, model at `<models-mount>/qwen3-9b-mlx`
(24 GDN + 8 attention layers, full-attention interval 4: layers 0–2 GDN,
layer 3 attention, repeating). All runs under the GPU flock, 25 s idle
between, 3-run median.

## 1. Anchor reproduction (arm A: clean phase harness, 3 runs)

| run | t/s | tokens | verifies | corrections | tok/step | GDN_TAPE_CALLS |
|---|---|---|---|---|---|---|
| A1 | 48.97 | 514 | 161 | 126 | 2.41 | 6888 |
| A2 | 49.00 | 514 | 161 | 126 | 2.41 | 6888 |
| A3 | 49.06 | 514 | 161 | 126 | 2.41 | 6888 |

Median t/s = **49.00** vs anchor 48.93 (+0.15%). FIRST20 identical across all
three runs and matching the anchor exactly:

```
271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279
```

Per-phase medians (ms over the 161-step generation): prefill 5.3,
draft 5249.8 (32.6 ms/step, 50.0%), verify 87.7 (0.55 ms/step, host build),
commit 5095.6 (31.6 ms/step, 48.5%), correction 61.8 (0.49 ms/correction,
host build), ar 0.0. Acceptance: D1 73.3%, D2 46.0%, D3 21.7% (D2/D3 lower
than the r4b 64-verify arm's 62.7/47.3 — the 322-iter window's deeper drafts;
D1, tok/step, FIRST20 and t/s all match, which is what the anchor protocol
requires).

Phase attribution (wheel quirk: `mx.eval()` is a no-op; forwards are deferred
and forced by the first `argmax`/`.item()` read):

- **verify** = S=4 batched-forward host build only (24 GDN tape T=4 ops +
  ~90 GEMMs queued; GPU execution happens at the `argmax(l0)` in the commit
  section).
- **commit** = `argmax` forces the whole S=4 forward on the GPU + tape-row
  rebind (lazy, host) + S=1 correction graph build on rejection.
- **correction** = S=1 graph build only; its GPU execution is deferred into
  the next iteration's draft section (`int(mx.argmax(l))` forces it first).
- **draft** = the deferred S=1 correction GPU + 3×(MTP module GPU +
  lm_head GEMV GPU) + their host builds.

## 2. Per-op GDN cost (arm B instrumented + arm C isolated probe)

**Instrumented harness (arm B, 3 runs):** `GDNHOST total = 12.094 / 12.088 /
12.163 ms` over 6888 `gdn_fused_step_tape` calls = **1.72–1.79 µs per op**,
i.e. ≈ 75 µs per verify step of python-visible host staging (0.11% of the
65.1 ms/step). Instrumentation overhead on t/s: B median 48.96 vs A median
49.00 — negligible.

**Isolated op probe** (`bench/gdn-op-probe.py`, fresh process, per-layer
shapes qwen3-9b: key_dim 2048, value_dim 4096, conv_dim 8192, Hv 32,
Dk=Dv 128; 30-run mean, one forced read per batch):

| op (24 independent ops, serial GPU) | per-op | 24-op total |
|---|---|---|
| `gdn_fused_step_tape` T=4 (verify) | 163.6 µs | 3.93 ms |
| `gdn_fused_step_tape` T=1 (correction) | 108.6 µs | 2.61 ms |
| `gdn_fused_step` T=1 (non-tape) | 56.1 µs | 1.35 ms |

(The 24-op totals include ~0.5–0.6 ms of the 24 sum-reductions + one forced
read; op-only ≈ 3.3 ms and ≈ 2.0 ms respectively. The GDN ops are
partially latency-bound: traffic lower bound for the 24 T=4 ops ≈ 305 MB
(2 MB state read + 2 MB state write + 8.4 MB tape write + conv per op)
≈ 1.16 ms at the measured 264 GB/s, so the measured 3.3 ms ≈ 2.8× the
bandwidth floor — 32 blocks × 256 threads per launch, one block per CU.)

**Per verify step GDN total (amortized over 161 steps, 126 corrections):**

```
24 × tape T=4 (verify)                    3.93 ms   (op-only ≈ 3.3)
24 × tape T=1 (correction) × 126/161      2.04 ms   (op-only ≈ 1.6)
                                          --------
                                         ≈ 4.9 ms/step
```

That is ≈ 7.5% of the 65.1 ms step (≈ 1.54 ms/token = 7.6% of the 20.4
ms/token). **GATE 1: 4.9 ms/step ≥ ~1 ms → design justified and performed.**

Sanity check against the phase split: the S=4 forward (≈ 30 ms GPU in the
commit phase) carries 3.9 ms of GDN (≈ 13% of the forward, consistent with
305 MB of the forward's ≈ 6.4 GB of traffic); the S=1 correction (≈ 20 ms
GPU in the next draft step) carries 2.6 ms (≈ 13%).

## 3. Design analysis — why all three options are blocked

The kernel itself is one launch per layer (32 blocks × 256 threads, T∈{1,2,4}
template; the tape variant adds the per-row state-tape writes; S=4 also adds
3 `contiguous_copy` launches for the row-strided q/k/v slices). The model
(`mlx_lm/models/qwen3_5.py`): `DecoderLayer.__call__` →
`r = self.linear_attn(self.input_layernorm(x), mask, cache)` where
`GatedDeltaNet.__call__` = 4 in_proj GEMMs (qkv/z/b/a) →
`gdn_fused_step_tape` → out_proj GEMM → residual add; layers chained
sequentially through the residual stream.

### (a) Batched multi-layer pass — BLOCKED by the residual chain

A batched launch of 24 (or of one 3-layer group) GDN ops requires all 24
in_proj outputs to be ready before the launch. They are not: GDN layer
i+1 consumes `x_i = x_{i-1} + out_proj_i(gdn_i(...)) + mlp_i(...)`, so
`in_proj_{i+1}` cannot start until `gdn_i` completes. The 24 GDN ops of a
forward form a strict data-dependency chain (interrupted by the 8 attention
layers, which themselves depend on the preceding GDN).

The only construction that removes the launch count without breaking the
chain is a *whole-layer fusion* — one kernel per GDN group computing
in_proj GEMMs + GDN + out_proj GEMM + MLP (and, for a whole-forward kernel,
attention too), i.e. a GEMM engine inside a custom kernel. That is a
redesign with a multi-week effort profile, not a launch merge; it is also
not env-gate-able behind the existing model structure without the same
restructure. **Not pursued** (state-correctness of such a kernel would rest
on the per-layer pointer-table argument below, but the design is out of
scope of this probe and its expected ceiling is the same ≈ 1–2 ms).

### (b) Persistent kernel — BLOCKED by the same dependency

A persistent (grid-stride, all-CU-resident) kernel spanning the 24 layers
faces the identical constraint: it must either (i) receive all 24 in_proj
outputs up front (impossible — the chain), or (ii) compute the intervening
GEMM/MLP/attention inside the kernel (the whole-layer fusion above). Spin-wait
on a flag set by the preceding kernel is not an option on this stack (no
inter-kernel signaling primitive in the wheel's dispatch path; power/latency
cost unbounded). **Not pursued.**

### (c) Launch-level batching — nothing to batch; the costs are already hidden

- **Host staging: 1.7 µs/op (measured, arm B).** The per-op C++ dispatch
  (4–8 `malloc_async`, up to 7 `maybe_copy` checks, one `encoder.launch_kernel`)
  executes while the host is building the rest of the step's graph; the
  per-step host total (≈ 25 ms) is far below the 65.1 ms step, and the
  GDN staging (0.08 ms) is 0.3% of the host budget. Even at 100%
  critical-path it is 0.13% of the step. **No gain to take.**
- **GPU inter-launch gaps: ≈ 2–5 µs × 48 ops ≈ 0.1–0.24 ms/step
  (0.2–0.4% of the step, est.)** in the in-order stream. The launch *count*
  is mandated by the data chain (one op per layer, each waiting on the
  previous layer); gaps between dependent launches cannot be removed without
  reducing the count — which is option (a). **No gain to take.**
- **`contiguous_copy` launches (S=4 only):** each T=4 op adds 3 copy
  launches + 3 kernels (q/k/v are row-strided slices of the qkv GEMM
  output). These 72 extra launches per verify step *are* removable — see the
  follow-up candidate below — but removing them is a kernel input-stride
  change, not a merge, and the ceiling is 0.15–0.36 ms/step.

### State-correctness argument (recorded for the (a) design, for the record)

If a batched op were built (e.g. for a model whose GDN layers are not
residual-chained), the correctness argument is: the kernel's only
cross-layer data path is a read-only per-layer pointer table (24 × 14
pointers ≈ 2.7 KB, passed as a kernel-arg buffer, under the launch-arg
limit); each (layer, head) block reads/writes exclusively its own layer's
buffers (inputs, `state`, `conv_state`, `state_new`, `state_tape`
`[T,32,128,128]`, `conv_state_tape`); all reductions (the 128-way
rms-norm, the 16-way dot products) are intra-block; there is no shared
scratch and no inter-block communication. Block (l, h) in the batched kernel
is bit-identical to the per-layer op invocation on layer l, head h — same
inputs, same code, same thread. Tape-row round-reuse semantics are
unchanged because the tape outputs remain per-layer arrays with unchanged
shapes and allocator paths. This argument does not rescue (a)/(b) for this
model — the dependency chain is in the *input availability*, not in the
kernel.

## 4. Feasible follow-up candidate (out of scope, recorded, not done)

Make the tape kernel accept row-strided q/k/v (pass the qkv row stride; the
slice is row-contiguous with stride `conv_dim × 2` bytes) and delete the 3
`contiguous_copy` ops per T=4 call: 96 → 24 launches per verify step,
≈ 0.15–0.36 ms/step (≈ 0.3–0.6% t/s), low risk (kernel input layout only;
bit-identical reads). Estimated ceiling is below the practical bar on its
own; it composes with nothing else feasible here.

## 5. Verdict

- **GATE 1 (measure):** GDN ops total ≈ 4.9 ms/step (≈ 7.5% of the step) —
  **passes** the ~1 ms threshold, so the design step was performed in full.
- **GATE 2 (design):** all three options blocked — (a) and (b) by the
  residual-stream data dependency between GDN layers (would require a
  whole-layer fusion, out of scope), (c) because the host staging is already
  hidden (1.7 µs/op) and the GPU inter-launch gaps (≤ 0.24 ms/step) are
  unremovable without (a). **No concrete, env-gate-able merge design exists
  for this model.**
- **GATE 3 (prototype):** not reached — negative verdict, stop.

No wheel rebuild, no model patch, no prototype branch work. The vendored
source is untouched (SBCAR unchanged: `02a54359…` pre/post).

## Artifacts

- `bench/mtp-bench6-phase-gdn.py` — instrumented copy of the phase harness
  (per-op host-time wrapping + GDN phase tagging); adds `GDNHOST`/
  `GDNSTEP`/`GDNHOST_PHASE` lines; t/s within 0.2% of the clean harness
  (arm B 48.84–49.12 vs arm A 48.97–49.06 medians 48.96 vs 49.00) with
  identical GDN_TAPE_CALLS=6888 (B's FIRST20 print was skipped by the
  crash noted below; A's three FIRST20 lines all match the anchor).
- `bench/gdn-op-probe.py` — isolated per-op probe (HOST / WALL / PIPE24 with
  forced-read semantics for the wheel's deferred dispatch); the 24-op
  serial numbers came from an equivalent on-guest variant with the same
  one-force-per-24 semantics. The harness copy
  (`bench/mtp-bench6-phase-gdn.py`) crashed on a `GDNHOST_PHASE` print
  bug (F507) *after* all measured lines were emitted in every B run —
  the measured data (GDNHOST totals, PHASE, TOKS, MTP=) is intact; the
  copy is kept byte-identical to the as-run evidence (excluded from
  lint, like the other as-run harnesses).
- Guest copies under the operator's scratch dir; all runs under the GPU
  flock with 25 s idle between runs; SBCAR verified pre/post.
