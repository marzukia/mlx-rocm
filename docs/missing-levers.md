# Missing levers — reference-anchored

Anchored to the two implementations that already reach the target on this GPU class (llama.cpp's MTP anchor, and the vLLM radiance serve), not to our own history. Every claim is labelled READ (file:line), MEASURED (a run) or DERIVED (arithmetic shown).

## The measured budget

Sources, stated precisely so a reader can land on them:

- The three rows are READ from `reports/opt-gauntlet-areaB-backend.md:121`: "Per token: QMV 9.46 ms; non-QMV kernels 4.9 ms; the rest is gaps", captured on a 32-token decode plus a 16-token warmup with rocprofv3 1.4.1 on an older wheel than the current one.
- The per-kernel detail is READ from the same file, section 3, and is reproduced below because it settles a question this document originally left open.
- The 636 GB/s ceiling and the ~88% realistic target are **not** in this repository. They come from a box-local report and are marked DERIVED here; the measurement behind them is a median of 4 GiB streaming passes with <1% spread over 3 passes. Landing that measurement is a small, worthwhile follow-up.

| bucket | now | realistic | headroom | source of "now" |
| --- | --- | --- | --- | --- |
| quantized weight stream (GEMV/QMM) | 9.46 ms | 8.22 ms | 1.24 ms | areaB:121 |
| non-GEMM kernels (norms, copies, elementwise, GDN, attention) | 4.9 ms | ~1.5 ms | ~3.4 ms | areaB:121 |
| launch / commit / python gaps (eager, ~1070 dispatches) | ~1.5 ms | ~0.3 ms graphed | ~1.2 ms | areaB:120 ("0 HIP graph launches"); the 1070 figure is validated at `reviews/pr-2-review.md:149` (43,838/41 = 1069.2 per step) |
| total | 15.9 ms = 63 tok/s | ~10.0 ms = ~100 tok/s | ~5.9 ms | areaB:119 (no-profiler bench, 15.9 ms/token) |

DERIVED consequence: the largest single recoverable block is **not** the weight-streaming maths. Over half of the available headroom sits in the non-GEMM bucket. That is one bucket, not a long tail of small issues.

A warning about denominators: earlier framings in this repo compared against 960 GB/s and reported "51% of floor". The measured ceiling is 636 GB/s, which puts the weight-stream at ~76% of what the card does rather than half. Any percentage quoted against 960 GB/s should be treated as void.

## Lever 1 — non-GEMM kernel cost (the headline)

READ: `reports/opt-gauntlet-areaB-backend.md:187`, which titles the same item "Non-GEMM kernel fusion: 4.9 → 1–2 ms/token — expected +20–30%".

Our tree already carries a fusion framework for exactly this: `vendor/mlx-rocm-upstream/mlx/fast.h` declares fused primitives for `rms_norm`, `rms_norm_residual`, `layer_norm`, `rope`, `kv_append`, `scaled_dot_product_attention`, `gdn_*` and `lm_head_int2`.

READ: the one dense-MLP-shaped fusion in the tree is for MoE only — `vendor/mlx-rocm-upstream/mlx/backend/rocm/moe_swiglu.cpp` ("Fused sorted-MoE SwiGLU as a compile-safe Primitive"), with `silu(g)*u` fused at `gemms/naive_gemm.h:206`. There is no fused dense gate/up path.

So the shape of the lever: the framework and a worked example exist; the dense case has no fused primitive. Adding one removes a read of the input plus the separate activation and multiply per layer per step.

### The bucket is already broken out

READ, `reports/opt-gauntlet-areaB-backend.md` section 3, rocprofv3 over a 32-token decode plus a 16-token warmup:

| kernel | time | calls | per token | avg |
| --- | --- | --- | --- | --- |
| `qmv_wide_kernel` | 454.3 ms (76.8%) | 12,450 | 260 | 36.5 µs |
| `rms_norm_kernel` | 35.5 ms | 7,345 | 153 | 4.8 µs |
| `row_reduce_simple` | 14.3 ms | 2,544 | 53 | 5.6 µs |
| fused elementwise (swiglu/residual/exp/sigmoid) | ~30 ms | — | — | — |
| copies (`copy_v`, `copy_gg_byval`, `copy_g_byval`) | 28.1 ms | 6,937 | 145 | 1.8-15 µs |
| `rope_single_1d` | 2.3 ms | 800 | 17 | 2.8 µs |
| `kernel_sdpav_1pass` | 1.8 ms | 400 | 8 | 4.4 µs |

Two things this settles:

1. **The seam is small, not large.** Kernel time across that profile is 722 ms. The same file records the no-profiler bench at 15.9 ms/token, so on the profiled token count the kernels are the large majority of the step and the non-kernel remainder is sub-millisecond to ~1.5 ms. The "GPU busy 41.7%" line in that file is computed against the profiler-inflated span and should not be read as real idle. So the 5-15 ms launch estimate attached to graph capture elsewhere is unsupported, and graph capture is a ~1 ms item at best on this evidence.

2. **The bucket is dominated by call count, not by heavy kernels.** 153 norm launches, 145 copy launches and 260 GEMV launches per token. Individually they are 2-5 µs; collectively they are the 4.9 ms. For a dense 24-layer model the GEMV call count is roughly double what the layer count implies, which is its own question.

CAUTION: do not size the fusion work from the MoE example alone. The MoE fusion is a sorted-gather problem with different economics. And these numbers come from an older wheel; a profile on the current wheel should confirm or replace them rather than be skipped.

## Lever 2 — weight-stream efficiency

READ: `reports/opt-gauntlet-areaB-backend.md:122` reports the QMV effective bandwidth as ~486 GB/s and calls it "51% of 960". DERIVED: against the measured 636 GB/s ceiling that is ~76%; the 960 denominator is spec-sheet, not measured, and it is why this repo's percentages read far worse than the hardware does. Realistic target ~88%, worth ~1.24 ms.

READ: llama.cpp does this differently at the arithmetic level — `ggml/src/ggml-cuda/mmq.cu` quantises activations to q8_1 (`quantize_mmq_q8_1_cuda`, line ~285) and multiplies integers; `mmvq.cu` provides the small-M vector-dot family. We dequantise weights to bf16 and accumulate in float (`quantized/qmv_kernel.hip`, `dequant_and_dot`). An integer path exists in our tree but lost its A/B (opt-in `MLX_QMV_LX`, measured -15-20%), so the arithmetic swap is NOT the lever — the layout and dispatch are.

## Lever 3 — per-step submission

READ (audit, `REPORT-gap-dive-audit.md` section C item 2, on branch `gap-dive-audit` @ `989c52e` — that file is not on this base): llama.cpp's anchor runs with **auto-captured HIP graphs** — `ggml/CMakeLists.txt:217` declares `GGML_HIP_GRAPHS` ON by default, the HIP backend compiles the `ggml-cuda` sources so `iface.graph_compute` is `ggml_backend_cuda_graph_compute`, and `is_enabled()` defaults true on gfx1201. The anchor therefore launches one captured graph per step while our wheel submits per operation.

STATUS: the section above settles this in favour of the measured figure. The audit's 5-15 ms is priced by the audit itself as a lower bound; the measured non-kernel remainder is ~1.5 ms, and the call-count table shows where the time actually is. Capture remains worth ~1 ms and is not a headline.

READ: our tree has decode capture machinery — `backend/rocm/eval.cpp` (`gd_begin_capture`, `gd_begin_replay`) driven by the pure state machine in `graph_decode_pilot_logic.h`. Note its gating: a first op with T > 1 runs the step **fully eager**, and a step whose op-name sequence diverges from the captured reference un-suppresses to eager. Spec decode with an S>1 verify contains a T>1 op, so that path is a poor fit for capture as written.

## Lever 4 — the draft head

READ (audit section C item 6): `lm_head_int2.hip` exists in the vendor tree lineage but is **absent from the canonical wheel** and was `INT2=off` in every measured run. That makes it a wheel build plus an A/B, not new kernel work.

## Not levers, with reasons

- Draft depth beyond 3: READ (audit C#1) llama.cpp's `n_max=3`; radiance K=4 measured 95.05 against K=3 at 142.31. K=3 is the operating point.
- Integer arithmetic swap (LX): MEASURED -15-20% on gfx1201.
- Weight format change (affine 4-bit vs MXFP4): no measurement supports it; the weight-stream is at 76% of the measured ceiling, so format is not the observed loss.
- Our own record as a yardstick: our history answers regression questions only. It cannot answer what is missing.

## The order of work

1. Confirm the call-count table on the current wheel. The shape is known from an older wheel; this run should confirm or correct it and be short, not a rediscovery.
2. Do NOT attack the call count: it reconciles exactly to 249 expected launches and the launch axis is a few percent (see the correction section).
3. Dense fused gate/up if the elementwise and norm share confirms it is worth it; the framework exists (`fast.h`) and only the dense case is missing.
4. Wheel build plus A/B for the int2 head, in parallel if the pool allows.

## Correction — the GEMV call count is not a defect

An earlier draft of this document treated "260 quantized-matmul launches per token where a 24-layer dense model implies ~121" as a lead. That inference was wrong, and the error was mine: 121 assumed one matmul per layer and omitted the 96 MLP linears, the 32 full-attention linears and the GDN gate/beta/alpha projections.

READ, `reports/gemv-callcount.md` (branch `gemv-callcount-analysis` @ `ae69da1`): the in-tree architecture is 32 hybrid layers (24 GDN, 8 full attention at interval 4). Each GDN layer issues in_proj_qkv (fused) + in_proj_z + in_proj_b + in_proj_a + out_proj + 3 MLP linears; each full-attention layer issues q/gate + k + v + o + 3 MLP linears; plus lm_head. Expected = **249 launches per forward**. Observed 12,450 = 249 x 50 exactly, and the "260 per token" was a divisor artifact (the profiled region held 50 single-token forwards, not 48). Delta from expected: zero. The count is wheel-invariant.

The same report prices the launch axis honestly: fusing qkvz (-24), beta/alpha (-24), gate_up (-32) and the full-attention KV writes (-16) takes 249 to 153 launches, a 24% cut, all load-time weight merges that are mathematically identical — and it measures the total launch overhead at only ~0.2-0.5 ms of a ~4.65 ms per-token fixed floor. A few percent.

Consequence for this document: the launch axis is a few percent, not a headline. Graph capture and call-count reduction drop to the bottom of the list, and the two live levers are the weight stream's 76%-of-ceiling efficiency and the non-GEMM kernel cost itself. The 9.46 ms GEMV row is bandwidth and layout on 249 real matmuls, not a redundancy.
