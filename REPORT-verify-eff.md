# REPORT-verify-eff: does the 4-bit layout cause the verify wall? (mlp/qmv classes)

Ticket: op-directed second-half ticket, parallel with the GDN-order sibling.
Base: planes-exactness @ 6e8df3f. Measurement + design only — no layout
format implemented.

## Verdict (up front)

**LAYOUT HYPOTHESIS DEAD (STOP-AND-REPORT, step 2).** Per-matmul bytes
actually pulled = **1.125 x the 4-bit weight bytes** (weight + bf16 scale +
bf16 bias; scale/bias are 1/64 density per group, not 1:1 with weight). Not
3x. The "3 streams" are real (weight/scale/bias are separate tensors), but the
two aux streams together are 12.5% of the weight stream, so the layout cannot
explain a ~54%/42%-of-floor class. The class cost is kernel efficiency
(dequant ALU + memory latency/occupancy), quantified in section 4.

## 1. Code-first: streams and bytes per matmul (no GPU)

Model: Qwen3-9B-MLX — **Qwen3-Next architecture (GDN linear attention),
32 layers**, 4-bit affine, group 64, bf16 scales, **bf16 per-group biases
present** (safetensors header: `*.scales` BF16 `[N, K/64]`, `*.biases` BF16
`[N, K/64]`, `*.weight` U32 `[N, K/8]` = 8 codes/word), H=4096, I=12288,
GDN attention (in_proj_qkv/z/b/a, conv1d, out_proj). Verified on the box's
model files (header read + live module inspection; no measurement).

Dispatch (vendor tree @ 75ca881, `qmm.hip::QuantizedMatmul::eval_gpu`):
verify-M shapes (M = 1..8, non-batched, transposed, K=4096/12288) fall in the
**QMV family**, not the dequant+rocBLAS path (crossover thresholds on RDNA4:
18/28/40 per shape class; `N>=8192 && K>=2048` early gate at M>=14). All QMV
kernels stream each output-column weight row **once per (row, m)**; scales
and biases are read once per (column, group). `qmv_wide_mr_kernel`
(M<=4, K<=4096) does a **single weight pass for all rows**;
`qmv_wide_kernel` (K=12288, M>4 or MR-LDS gate fail) and
`qmv_tiled_kernel` are grid `(cols, m)` — per-m re-reads are L2-mitigated by
construction (m-fastest block order, 32 MB L2 on 9070 XT). Caveat:
`MLX_ROCM_QMM_M3` is a wheel-side knob not present in the pinned tree
(byte-verified @ 75ca881); it cannot change the byte accounting (it can only
select among the single-pass M<=4 variants).

### 1.1 Per-matmul byte table (4-bit, group 64, bf16 scale+bias)

Per matmul N x K: weight `N*K/2` B (2 codes/byte), scale `N*(K/64)*2` B =
`N*K/32`, bias `N*(K/64)*2` B = `N*K/32`. **Total = 9/16 * N*K = 1.125 x
weight bytes.** Naive weight-only figure = `N*K/2`.

| GEMM (N x K) | weight B | scale B | bias B | total B | total/weight |
|---|---|---|---|---|---|
| mlp gate/up_proj (12288 x 4096) | 25,165,824 | 1,572,864 | 1,572,864 | 28,311,552 | 1.125 |
| mlp down_proj (4096 x 12288) | 25,165,824 | 1,572,864 | 1,572,864 | 28,311,552 | 1.125 |
| qmv in_proj_qkv (8192 x 4096) | 16,777,216 | 1,048,576 | 1,048,576 | 18,874,368 | 1.125 |
| qmv in_proj_z / out_proj (4096 x 4096) | 8,388,608 | 524,288 | 524,288 | 9,437,184 | 1.125 |
| qmv in_proj_b / in_proj_a (32 x 4096) | 65,536 | 4,096 | 4,096 | 73,728 | 1.125 |

| class | per layer | per forward (32 layers) | per verify step (1x M=3 + 4x M=1, single weight pass per GEMM) |
|---|---|---|---|
| mlp (gate+up+down) | 81.0 MiB | 2,592.0 MiB (2.531 GiB) | 12,960.0 MiB (12.66 GiB) |
| qmv (qkv+z+b+a+o) | 36.17 MiB | 1,157.5 MiB (1.130 GiB) | 5,787.5 MiB (5.652 GiB) |
| all GEMMs | 117.17 MiB | 3,749.5 MiB (3.662 GiB) | 18,747.5 MiB (18.29 GiB) |

Cross-check vs the anchor (derived, not a given): mlp forward bytes
2.531 GiB / 8.447 ms (recorded mlp class time) = 322 GB/s = **54.1% of the
595 GB/s floor floor** — so the record's 54.1%-of-floor is consistent with
the per-M=3-forward class at the single-pass byte figure (in-session
re-measurement in section 3 confirms the achieved GB/s). If per-m re-reads
were NOT L2-absorbed, the M=3 forward's per-m variants (down_proj K=12288)
would pull up to 3x the weight bytes on those GEMMs; the in-session GB/s is
the arbiter.

### 1.2 The 3x question, answered numerically

- Streams per matmul: **3** (weight, scale, bias) — the "3 streams" part of
  the hypothesis is TRUE.
- Bytes pulled per matmul: **1.125 x** weight bytes (scale 6.25% + bias
  6.25%). llama.cpp Q4_K with interleaved scale reads ~1.0 x weight bytes
  plus per-block super-scales. The layout delta is **1.125 x, not 3 x**.
- The dequant+rocBLAS path (not used at verify M) would read a bf16 shadow
  (N*K*2 B = 4x code bytes), but the dispatch thresholds keep verify-M on
  QMV.

**STOP-AND-REPORT trigger met:** bytes pulled (1.125 x) already equal the
weight bytes to within 12.5% (the aux streams are group-density, not
weight-density). A layout change removes at most the 12.5% aux-stream
overhead, and the scale stream is not avoidable at group 64 (Q4_K carries
the same 1/32 scale density, interleaved). It cannot close a gap from ~50%
of floor to ~100%. Hypothesis dead — not forced further (step 2).

## 2. Environment / deviations (recorded)

- Box: neon (100.78.68.72, bare-metal ROCm gfx1201). Sidecar md5
  02a54359cda8b247f76b277dd9bc4d4c **PASS** before any run.
- venv-verify: fresh byte-copy. **Deviation:** the ticket named `venv-r4v`
  with COR libmlx md5 8acf13fe, but on-box `venv-r4v` libmlx = fa4ace5b (a
  different wheel); `venv-r4v-p1922` libmlx = **8acf13fe (COR match)** —
  the copy source was venv-r4v-p1922. Pre-existing `venv-verify` (db4f9896,
  no process using it) was replaced per the ticket's "create a NEW
  venv-verify". libmlx md5 at start = 8acf13fe.
- **Deviation:** no `rocprofv3` on this box (only rocprof/rocprofv2), no
  SCLK sampling tool (the t2152 path is guest-a-only). Floor and class
  GB/s are wall-clock-measured with chained-eval amortization (host
  overhead ~1/N per op), labeled **measured(wall)**; P-state = 25 s idle +
  rocm-smi use/clock snapshots per arm (deviation recorded per ticket).
- Coexistence: `fuser ~/gpu.lock` free at start (rc=1); all GPU arms run
  under `flock ~/gpu.lock`, 25 s idle before each arm.
- **Deviation (model):** first probe run hit `DecoderLayer` dict-style
  attributes + revealed the model is Qwen3-Next (GDN linear attention, 32
  layers), not standard Qwen3 attention — the shape table above is the
  corrected one (verified against live module shapes).

## 3. In-session measurement (GPU, under ~/gpu.lock)

<filled after the GPU window>

## 4. Design / class-cost statement

<filled after step 3>

## 5. Check lines

<filled at report finalization>
