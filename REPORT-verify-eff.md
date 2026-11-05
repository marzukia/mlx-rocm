# REPORT-verify-eff: does the 4-bit layout cause the verify wall? (mlp/qmv classes)

Ticket: op-directed second-half ticket, parallel with GDN-order sibling.
Base: planes-exactness @ 6e8df3f. Measurement + design only — no layout format implemented.

## Verdict (up front)

**LAYOUT HYPOTHESIS DEAD (STOP-AND-REPORT, step 2).** Per-matmul bytes actually
pulled = **1.125 x the 4-bit weight bytes** (weight + bf16 scale + bf16 bias,
scale/bias are 1/64 density per group, not 1:1 with weight). Not 3x. The
"3 streams" are real (weight/scale/bias are separate tensors) but the two
aux streams together are 12.5% of the weight stream, so the layout cannot
explain a 54%/42%-of-floor class. The class cost is kernel efficiency
(dequant ALU + memory latency/occupancy), quantified in section 4.

## 1. Code-first: streams and bytes per matmul (no GPU)

Model: Qwen3-9B-MLX 4-bit affine, group 64, bf16 scales, **bf16 per-group
biases present** (safetensors header: `*.scales` BF16 `[N, 64]`,
`*.biases` BF16 `[N, 64]`, `*.weight` U32 `[N, 512]` = 8 codes/word),
lm_head included. H=4096, I=12288, 48 layers, 32 heads x 128, 8 KV heads
(1024). Verified on the box's model files (header read, no GPU).

Dispatch (vendor tree @ 75ca881, `qmm.hip::QuantizedMatmul::eval_gpu`):
verify-M shapes (M = 1..8, non-batched, transposed, K=4096/12288) fall in the
**QMV family**, not the dequant+rocBLAS path (crossover thresholds on RDNA4:
18/28/40 per shape class; `N>=8192 && K>=2048` early gate at M>=14). All QMV
kernels stream each output-column weight row **once per (row, m)**; scales and
biases are read once per (column, group). `qmv_wide_mr_kernel` (M<=4, K<=4096)
does a **single weight pass for all rows**; `qmv_wide_kernel` (K=12288, M>4
or MR-LDS gate fail) and `qmv_tiled_kernel` are grid `(cols, m)` — per-m
re-reads are L2-mitigated by construction (m-fastest block order, 32 MB L2 on
9070 XT). Caveat: `MLX_ROCM_QMM_M3` is a wheel-side knob not present in the
pinned tree (byte-verified @ 75ca881); it cannot change the byte accounting
(it can only select among the single-pass M<=4 variants).

### 1.1 Per-matmul byte table (4-bit, group 64, bf16 scale+bias)

Per matmul N x K: weight `N*K/2` B (2 codes/byte), scale `N*(K/64)*2` B =
`N*K/32`, bias `N*(K/64)*2` B = `N*K/32`. **Total = 9/16 * N*K = 1.125 x
weight bytes.** Naive weight-only figure = `N*K/2`.

| GEMM (N x K) | weight B | scale B | bias B | total B | total/weight |
|---|---|---|---|---|---|
| q_proj / o_proj (4096 x 4096) | 8,388,608 | 1,048,576 | 1,048,576 | 10,485,760 | 1.125 |
| k_proj / v_proj (1024 x 4096) | 2,097,152 | 262,144 | 262,144 | 2,621,440 | 1.125 |
| gate/up_proj (12288 x 4096) | 25,165,824 | 1,572,864 | 1,572,864 | 28,311,552 | 1.125 |
| down_proj (4096 x 12288) | 25,165,824 | 1,572,864 | 1,572,864 | 28,311,552 | 1.125 |

| class | per forward (48 layers) | per verify step (1x M=3 + 4x M=1, single weight pass per GEMM) |
|---|---|---|
| mlp (gate+up+down) | 3,888.0 MiB (3.795 GiB) | 5 x 3,888 = 19,440.0 MiB (19.00 GiB) |
| qmv (q+k+v+o) | 1,080.0 MiB (1.054 GiB) | 5 x 1,080 = 5,400.0 MiB (5.273 GiB) |
| all GEMMs | 4,968.0 MiB (4.846 GiB) | 24,840.0 MiB (24.27 GiB) |

(If per-m re-reads were NOT L2-absorbed, mlp step bytes would be
(3+4x1) passes = 5x single-pass anyway for the M=3 part counted once each; the
per-m kernel re-read factor is at most the M factor on the M=3 forward,
+3x3,145,728 B per down_proj per layer — bounded, and the in-session GB/s
is the arbiter.)

### 1.2 The 3x question, answered numerically

- Streams per matmul: **3** (weight, scale, bias) — the "3 streams" part of the
  hypothesis is TRUE.
- Bytes pulled per matmul: **1.125 x** weight bytes (scale 6.25% + bias 6.25%).
  llama.cpp Q4_K with interleaved scale reads ~1.0 x weight bytes plus
  per-block super-scales. The layout delta is **1.125 x, not 3 x**.
- The dequant+rocBLAS path (not used at verify M) would read a bf16 shadow
  (N*K*2 B = 4x code bytes), but the dispatch thresholds keep verify-M on QMV.

**STOP-AND-REPORT trigger met:** bytes pulled (1.125 x) already equal the weight
bytes to within 12.5% (the aux streams are group-density, not weight-density).
A layout change removes at most the 12.5% aux-stream overhead (and the scale
stream is not avoidable at group 64 — Q4_K carries the same 1/32 scale
density interleaved). It cannot close a gap from ~50% of floor to ~100%.
Hypothesis dead — not forced further (step 2).

## 2. Environment / deviations (recorded)

- Box: neon (100.78.68.72, bare-metal ROCm gfx1201). Sidecar md5
  02a54359cda8b247f76b277dd9bc4d4c **PASS** before any run.
- venv-verify: fresh byte-copy. **Deviation:** ticket named `venv-r4v` with
  COR libmlx md5 8acf13fe, but on-box `venv-r4v` libmlx = fa4ace5b (a
  different wheel); `venv-r4v-p1922` libmlx = **8acf13fe (COR match)** — the
  copy source was venv-r4v-p1922. libmlx md5 at start = 8acf13fe.
- **Deviation:** no `rocprofv3` on this box (only rocprof/rocprofv2), no SCLK
  sampling tool (the t2152 path is guest-a-only). Floor and class GB/s are
  wall-clock-measured with chained-eval amortization (host overhead ~1/N per
  op), labeled **measured(wall)**; P-state gating = 25 s idle + idle/swap gate
  (deviation recorded per ticket).
- Coexistence: `fuser ~/gpu.lock` free at start (rc=1); all GPU arms run
  under the lock, 25 s idle before each arm.

## 3. In-session measurement (GPU, under ~/gpu.lock)

<filled after the GPU window>

## 4. Design / class-cost statement

<filled after step 3>

## 5. Check lines

<filled at report finalization>
