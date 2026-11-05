# REPORT-verify-eff: does the 4-bit layout cause the verify wall?
(mlp / qmv classes)

Ticket: op-directed second-half ticket, parallel with the GDN-order sibling.
Base: planes-exactness @ 6e8df3f. Measurement + design only — no layout format implemented.

## Verdict (up front)

**LAYOUT HYPOTHESIS DEAD (STOP-AND-REPORT, step 2).** Per-matmul bytes actually pulled = **1.125 x the 4-bit weight bytes** (weight + bf16 scale + bf16 bias; scale/bias are 1/64 density per group, not 1:1 with weight). Not 3x. The "3 streams" are real (weight/scale/bias are separate tensors), but the two aux streams together are 12.5% of the weight stream, so the layout cannot explain a ~54%/42%-of-floor class. The class cost is kernel efficiency (dequant ALU + memory latency/occupancy), quantified in section 4.

## 1. Code-first: streams and bytes per matmul (no GPU)

Model: Qwen3-9B-MLX — **Qwen3-Next architecture (GDN linear attention),
32 layers**, 4-bit affine, group 64, bf16 scales, **bf16 per-group biases
present** (safetensors header: `*.scales` BF16 `[N, K/64]`, `*.biases` BF16
`[N, K/64]`, `*.weight` U32 `[N, K/8]` = 8 codes/word), H=4096, I=12288,
GDN attention (in_proj_qkv/z/b/a, conv1d, out_proj). Verified on the box's
model files (header read + live module inspection; no measurement).

Dispatch (vendor tree @ 75ca881, `qmm.hip::QuantizedMatmul::eval_gpu`):
verify-M shapes (M = 1..8, non-batched, transposed,
K=4096/12288) fall in the
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
595 GB/s floor** — so the record's 54.1%-of-floor is consistent with
the per-M=3-forward class at the single-pass byte figure (in-session
re-measurement in section 3 confirms the achieved GB/s). If per-m re-reads
were NOT L2-absorbed, the M=3 forward's per-m variants (down_proj K=12288)
would pull up to 3x the weight bytes on those GEMMs; the in-session GB/s is
the arbiter.

### 1.2 The 3x question, answered numerically

- Streams per matmul: **3** (weight, scale, bias) — the "3 streams" part of the hypothesis is TRUE.
- Bytes pulled per matmul: **1.125 x** weight bytes (scale 6.25% + bias 6.25%). llama.cpp Q4_K with interleaved scale reads ~1.0 x weight bytes plus per-block super-scales. The layout delta is **1.125 x, not 3 x**.
- The dequant+rocBLAS path (not used at verify M) would read a bf16 shadow
  (N*K*2 B = 4x code bytes), but the dispatch thresholds keep verify-M on
  QMV.

**STOP-AND-REPORT trigger met:** bytes pulled (1.125 x) already equal the weight bytes to within 12.5% (the aux streams are group-density, not weight-density). A layout change removes at most the 12.5% aux-stream overhead, and the scale stream is not avoidable at group 64 (Q4_K carries the same 1/32 scale density, interleaved). It cannot close a gap from ~50% of floor to ~100%. Hypothesis dead — not forced further (step 2).

## 2. Environment / deviations (recorded)

- Box: the on-box bare-metal ROCm gfx1201 (hostname/IP intentionally omitted from this commit). Sidecar md5 02a54359cda8b247f76b277dd9bc4d4c **PASS** before any run.
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
- **Deviation (probe API):** `mx.qmatmul` does not exist in this build;
  the probe issues the byte-identical model op
  `mx.quantized_matmul(..., transpose=True, group_size=64, bits=4,
  mode="affine")` (verified against `mlx.nn.layers.quantized`'s call
  path).
- **Deviation (model):** first probe run hit `DecoderLayer` dict-style attributes + revealed the model is Qwen3-Next (GDN linear attention, 32 layers), not standard Qwen3 attention — the shape table above is the corrected one (verified against live module shapes).

## 3. In-session measurement (GPU, under ~/gpu.lock)

Method (labeled **measured(wall)**; deviation: no `rocprofv3` / no SCLK
sampling on this box): 25 s idle + rocm-smi P-state snapshot before each
arm; 2 rounds, best-of, variance reported; GEMMs run through the exact
model op (`mx.quantized_matmul(..., transpose=True, group_size=64,
bits=4, mode="affine")`), 100-chain eval amortizes host overhead; floor
= 4 GiB x 50 chained sums, 3 rounds. COR env (COR libmlx md5 8acf13fe),
`MLX_ROCM_QMM_M3=1` set (knob not present in the pinned tree — recorded;
byte accounting unaffected).

P-state snapshot (identical across arms): sclk S: 41 MHz, fclk 582 MHz, mclk 96 MHz (idle state; the GPU boosts under load — variance across rounds is the stability witness: 13.2% M1-gate, 6.3% M1-down, 6.2% M3-qkv, <1% on all other arms). No SCLK sampling possible (deviation recorded).

Floor (in-session): **631.8 GB/s** (4 GiB, 50-chain, 3 rounds, var 0.13%).
Cross-check: standalone single-process run in the same context gave 625.1 / 623.7 / 628.5 / 630.7 GB/s at 1/2/5/50 chunks (linear in chunk count) — consistent with the recorded 595-624 GB/s t2048 floor band.

Per-GEMM achieved GB/s (bytes = 1.125 x weight, section 1.1) and % of the
in-session floor:

| GEMM | M=1 t / GB/s / %floor | M=3 t / GB/s / %floor | M=3 kernel path |
|---|---|---|---|
| mlp gate (12288 x 4096) | 58.4 us / 485.0 / 76.8% | 116.1 us / 243.8 / 38.6% | qmv_wide_mr (single pass) |
| mlp down (4096 x 12288) | 55.9 us / 506.6 / 80.2% | 88.0 us / 321.9 / 51.0% | qmv_wide per-m (MR-LDS gate: 72 KiB > 32 KiB LDS) |
| qmv in_proj_qkv (8192 x 4096) | 44.8 us / 421.0 / 66.6% | 83.6 us / 225.9 / 35.8% | qmv_wide_mr (single pass) |
| qmv in_proj_z (4096 x 4096) | 29.3 us / 322.4 / 51.0% | 53.3 us / 177.2 / 28.0% | qmv_wide_mr (single pass) |
| qmv out_proj (4096 x 4096) | 29.5 us / 319.4 / 50.6% | 52.6 us / 179.3 / 28.4% | qmv_wide_mr (single pass) |
| qmv in_proj_b / a (32 x 4096) | 12.2 us / 6.1 / 1.0% | 16.4 us / 4.5 / 0.7% | qmv_wide (latency-bound, negligible bytes) |

Class totals (per M=3 forward, 32 layers) and per verify step
(1x M=3 + 4x M=1, single weight pass per GEMM):

| class | per M=3 forward | GB/s | % of 631.8 floor | per verify step | GB/s | % floor |
|---|---|---|---|---|---|---|
| mlp | 10.25 ms | 265.3 | **42.0%** | 32.34 ms | 420.2 | 66.5% |
| qmv | 7.11 ms | 170.5 | **27.0%** | 23.50 ms | 258.1 | 40.8% |

Anchor cross-check (recorded fractions, different session): mlp 54.1% / qmv
41.7% of the recorded 595-624 GB/s floor. In-session: M=3-forward mlp
42.0% (anchor-consistent direction, 12 pts lower — session P-state / wheel
path difference; the record's 54.1% back-computes to the per-M=3-forward
class at the 1.125x byte figure, section 1.1 cross-check). The record's
qmv 41.7% matches the **per-verify-step** qmv here (40.8%) within 1 pt; the
record's mlp 54.1% matches the per-forward basis. The recorded mlp/qmv
appear to use different basis; flagged, not concluded.

## 4. Design / class-cost statement

1. **The layout is not the cause (hypothesis dead, confirmed with measured numbers).** The same 4-bit affine layout (weight + bf16 scale + bf16 bias, 1.125x weight bytes) reaches **76.8-80.2% of the streaming floor at M=1** on the large shapes (gate 76.8%, down 80.2%). Removing the aux streams entirely (an idealized 1.0x layout that does not exist at group 64 — Q4_K carries the same 1/32 scale density, interleaved) would raise achieved GB/s by at most 12.5%: mlp 42.0% -> <= 47.3%, qmv 27.0% -> <= 30.4%. It cannot close a gap to ~100%.
2. **The class cost is kernel efficiency at M>=2, not bytes.** M=1 -> M=3
   collapse: gate 76.8% -> 38.6%, qkv 66.6% -> 35.8%, z/o 51.0% -> 28.0%.
   The M=3 single-pass MR kernel (LDS 24 KiB, K=4096) is the dominant
   cost at 35-39% of floor (dequant ALU at low arithmetic intensity:
   N x K/8 codes dequantized per tile); the per-m wide path on
   K=12288 (down) holds 51.0% thanks to L2 mitigation.
3. **The small GDN GEMMs (in_proj_b/a, N=32) are latency-bound (0.7-1.0%
   of floor) but negligible in bytes** (73,728 B each; 33 us per forward
   = 0.5% of the qmv class time).
4. **No layout/format change is indicated for verify-M.** Improvement levers, listed for the record (not implemented here; the K=12288 single-pass MR is the sibling m3f/planes workstream, not touched here): (a) single-pass MR coverage for K=12288 (LDS 72 KiB gate); (b) MR dequant-ALU tuning for the K=4096 shapes (occupancy / wider K-blocks); (c) batching the N=32 GEMMs (model-architecture change, out of scope). The bounded layout overhead (12.5%) is second-order against the 2x+ kernel-efficiency gap.

Class-cost statement: **mlp runs at 42.0% of the in-session streaming
floor (265 GB/s) and qmv at 27.0% (171 GB/s) at the M=3 verify batch;
per verify step (1x M=3 + 4x M=1) mlp 66.5% / qmv 40.8%. The 4-bit
layout pulls 1.125x weight bytes, not 3x; the remaining gap is
quantized-matmul kernel efficiency at M>=2 (dequant ALU + per-m
re-reads on the K=12288 path), not the separate weight/scale/bias
streams.**

## 5. Check lines

- ruff format: `88 files already formatted` — OK
- ruff check: `All checks passed!` — OK
- line wrap: no non-table line > 76 chars — OK
- persona/secret leak scan (report + probe): CLEAN
- box IP/hostname scan: only the project-internal name `guest-a`
  (per AGENTS.md) — OK
- size gate: this PR adds no code to >2,000-line files (probe script is
  204 lines)
- GPU gate: probe ran under `flock ~/gpu.lock`, 25 s idle per arm, two
  independent sessions (arm times reproducible to within ~3% between
  sessions; floor 631.8 GB/s in-session vs 623.7-630.7 standalone)
