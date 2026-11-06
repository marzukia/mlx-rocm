# Missing levers — reference-anchored

Anchored to the two implementations that already reach the target on this GPU class (llama.cpp's MTP anchor, and the vLLM radiance serve), not to our own history. Every claim is labelled READ (file:line), MEASURED (a run) or DERIVED (arithmetic shown).

## The measured budget

From `REPORT.md` section 4, rescaled to a **measured** streaming ceiling of 636 GB/s. That report is a K2-era artifact based on an older base, so treat these as last-known-measurement, not current: a profile on the current wheel must confirm or replace them.

| bucket | now | realistic | headroom |
| --- | --- | --- | --- |
| quantized weight stream (GEMV/QMM) | 9.46 ms | 8.22 ms | 1.24 ms |
| non-GEMM kernels (norms, copies, elementwise, GDN, attention) | 4.9 ms | ~1.5 ms | ~3.4 ms |
| launch / commit / python gaps (eager, ~1070 dispatches) | ~1.5 ms | ~0.3 ms graphed | ~1.2 ms |
| total | 15.9 ms = 63 tok/s | ~10.0 ms = ~100 tok/s | ~5.9 ms |

DERIVED consequence: the largest single recoverable block is **not** the weight-streaming maths. Over half of the available headroom sits in the non-GEMM bucket. That is one bucket, not a long tail of small issues.

A warning about denominators: earlier framings in this repo compared against 960 GB/s and reported "51% of floor". The measured ceiling is 636 GB/s, which puts the weight-stream at ~76% of what the card does rather than half. Any percentage quoted against 960 GB/s should be treated as void.

## Lever 1 — non-GEMM kernel cost (the headline)

READ: `REPORT.md` section 4, and `reports/opt-gauntlet-areaB-backend.md:187`, which titles the same item "Non-GEMM kernel fusion: 4.9 → 1–2 ms/token — expected +20–30%".

Our tree already carries a fusion framework for exactly this: `vendor/mlx-rocm-upstream/mlx/fast.h` declares fused primitives for `rms_norm`, `rms_norm_residual`, `layer_norm`, `rope`, `kv_append`, `scaled_dot_product_attention`, `gdn_*` and `lm_head_int2`.

READ: the one dense-MLP-shaped fusion in the tree is for MoE only — `vendor/mlx-rocm-upstream/mlx/backend/rocm/moe_swiglu.cpp` ("Fused sorted-MoE SwiGLU as a compile-safe Primitive"), with `silu(g)*u` fused at `gemms/naive_gemm.h:206`. There is no fused dense gate/up path.

So the shape of the lever: the framework and a worked example exist; the dense case has no fused primitive. Adding one removes a read of the input plus the separate activation and multiply per layer per step.

CAUTION: do not size this from the fusion example alone. The MoE fusion is a sorted-gather problem with different economics. The profile must say how much of the 4.9 ms is norms, how much is elementwise, how much is copies, how much is GDN, and how much is attention before any of them is fused.

## Lever 2 — weight-stream efficiency

MEASURED (audit): our quantized QMV sits at **76.4%** of the 636 GB/s ceiling (`REPORT.md` section 3 cross-check, which itself notes the "51% of 960" was a wrong-denominator artifact). Realistic target ~88%. Worth ~1.24 ms.

READ: llama.cpp does this differently at the arithmetic level — `ggml/src/ggml-cuda/mmq.cu` quantises activations to q8_1 (`quantize_mmq_q8_1_cuda`, line ~285) and multiplies integers; `mmvq.cu` provides the small-M vector-dot family. We dequantise weights to bf16 and accumulate in float (`quantized/qmv_kernel.hip`, `dequant_and_dot`). An integer path exists in our tree but lost its A/B (opt-in `MLX_QMV_LX`, measured -15-20%), so the arithmetic swap is NOT the lever — the layout and dispatch are.

## Lever 3 — per-step submission

READ (audit, `reports/REPORT-gap-dive-audit.md` section C item 2): llama.cpp's anchor runs with **auto-captured HIP graphs** — `ggml/CMakeLists.txt:217` declares `GGML_HIP_GRAPHS` ON by default, the HIP backend compiles the `ggml-cuda` sources so `iface.graph_compute` is `ggml_backend_cuda_graph_compute`, and `is_enabled()` defaults true on gfx1201. The anchor therefore launches one captured graph per step while our wheel submits per operation.

TENSION to resolve, not paper over: the audit estimates this at 5-15 ms; the budget row above measures our eager launch/commit/python gaps at only ~1.5 ms. One is wrong. The measured figure wins until a profile adjudicates.

READ: our tree has decode capture machinery — `backend/rocm/eval.cpp` (`gd_begin_capture`, `gd_begin_replay`) driven by the pure state machine in `graph_decode_pilot_logic.h`. Note its gating: a first op with T > 1 runs the step **fully eager**, and a step whose op-name sequence diverges from the captured reference un-suppresses to eager. Spec decode with an S>1 verify contains a T>1 op, so that path is a poor fit for capture as written.

## Lever 4 — the draft head

READ (audit section C item 6): `lm_head_int2.hip` exists in the vendor tree lineage but is **absent from the canonical wheel** and was `INT2=off` in every measured run. That makes it a wheel build plus an A/B, not new kernel work.

## Not levers, with reasons

- Draft depth beyond 3: READ (audit C#1) llama.cpp's `n_max=3`; radiance K=4 measured 95.05 against K=3 at 142.31. K=3 is the operating point.
- Integer arithmetic swap (LX): MEASURED -15-20% on gfx1201.
- Weight format change (affine 4-bit vs MXFP4): no measurement supports it; the weight-stream is at 76% of the measured ceiling, so format is not the observed loss.
- Our own record as a yardstick: our history answers regression questions only. It cannot answer what is missing.

## The order of work

1. Profile the non-GEMM bucket on the current wheel. Per-kernel ms, totalled, cross-checked against the 4.9 ms row. No fusions in that run.
2. Adjudicate the launch-gap tension with the same profile (kernel sum against step wall).
3. Then fuse what the profile says is fat, dense gate/up first if it is the fat one.
4. Wheel build plus A/B for the int2 head, in parallel with 3 if the pool allows.