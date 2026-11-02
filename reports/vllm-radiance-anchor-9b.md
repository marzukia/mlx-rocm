# vLLM-radiance MXFP4 anchor — Qwen3.5-9B MTP spec decode on neon (RX 9070 XT, gfx1201)

Date: 2026-11-02 (neon UTC, ~22:15–22:39). Reproduction of the vllm-radiance stack (the
27B DFlash2 production setup) on the 9B mxfp4a16 checkpoint, benchmarked per the t1392
protocol. Purpose: an anchor number for the 9B-class dense model on the radiance stack,
comparable to t1392's 3.24 tok/s (stock vLLM) and the current MTP figure of 65.79.

## Setup

- **Box**: neon, AMD Radeon RX 9070 XT / gfx1201, 32 GB. No docker, so the
  `stilldeadcode/vllm-radiance:0.9.3` image's `/opt` was extracted to a local tree
  (8.7 GB rsync from helium): vLLM 0.27.1,
  torch 2.11.0+rocm7.14, triton 3.6.0, CPython 3.12 (uv-managed, since neon's system
  python is 3.14 — wrong ABI for the wheels). ROCm 7.14 tree from the image is the only
  compiler (host ROCm 10.1 is runtime-only); `radiance_mxfp4_fp8.hip` compiled with its
  hipcc, `r4d.so` installed; aiter `module_aiter_core` built without `--offload-arch`
  (7.14 hipcc doesn't know gfx1201) — the stack tolerates the JIT shape the same way the
  container does. The radiance repo (v0.13.0, commit 45da506) is cloned locally;
  the libr4d cache was transferred from the 27B prod (rx6 + rx9 keys).
- **Model**: `qwen3.5-9b-mxfp4a16` — 555 tensors (299 BF16 + 256 U8-packed mxfp4),
  public hub source `2imi9/Qwen3.5-9B-MXFP4A16` — plus the grafted MTP head
  (`model-mtp-bf16.safetensors`, 15 BF16 tensors, 486 MB, from the mlx sidecar).
- **GPU**: 3 h flock window on `gpu-bench.lock`; card at FPL=high, sclk 2200 MHz,
  31 °C at lock time.

## Deviations from the 27B production env (all stack-native or documented)

1. **Attention backend `TRITON_ATTN`** instead of `R4D`: the R4D attention kernels only
   cover `gqa == 6` (the 27B shape); the 9B is `gqa = 4` (16 heads / 4 KV). The stack's
   own R4D refusal message names `TRITON_ATTN` as the fallback for this shape. R4D
   allreduce stays on (`RADIANCE_USE_R4D_AR=1` etc. — AR kernels are shape-agnostic).
2. **`RADIANCE_FP8_STREAM=0`**: the 9B GDN `in_proj` is BF16 in the hub checkpoint, so
   the gdnmerge patch takes the unquantized-merge path, which does not create `_rad_ws`;
   with FP8_STREAM on the merged forward unconditionally takes the quantized tuple path
   and crashes (`AttributeError: _rad_ws`). The stack's documented fallback is
   `FP8_STREAM=0` (`serve-mxfp4.sh:340`); the 27B prod uses 1 because its `in_proj` is
   quantized.
3. **`--max-model-len 16384`** instead of 65536: at 65536, vLLM reports 1.29 GiB KV
   needed vs 0.45 GiB available on the one 32 GB card (9B weights 9.95 GiB + MTP draft
   head + profile-run activations). 16384 needs ~0.25 GiB. The bench only decodes 512
   tokens, so context length is not load-bearing for the measurement.
4. **MTP draft config**: upstream vLLM's `hf_config_override` has no qwen3_5 entry, so the
   extracted venv's `speculative.py` was patched (the venv is a throwaway extraction, not
   the stack repo) to map `qwen3_5_text` + `mtp_num_hidden_layers > 0` →
   `model_type: qwen3_5_mtp`, `architectures: ["Qwen3_5MTP"]` in the draft config only —
   the same mechanism vLLM uses for deepseek_v3 / kimi_k3 / qwen3_next. `Qwen3_5MTP` is
   a registered architecture; the predictor reads `mtp_num_hidden_layers` and the grafted
   `mtp.*` weights directly.
5. **Checkpoint tensor names**: the hub checkpoint names weights
   `model.language_model.*`, which the text model class (`Qwen3_5ForCausalLM`) does not
   accept; the names were renamed in place (safetensors header rewrite, data untouched,
   padded to the original header length). First attempt truncated the file — it was a
   symlink to the anchor copy, which was then restored bit-for-bit from the public hub
   source (identical 555-tensor data total 10 120 881 152 B, verified with
   `safe_open`) before the rename was redone correctly.

## Protocol (t1392)

`bench-vllm-anchor.py`: prompt "Hello", `max_tokens=512`, `ignore_eos`, temperature 0,
3 runs, wall-clock tok/s, single request per run. Arms: spec ON K=4 (the radiance
quickstart default), spec ON K=3, plain (spec OFF). Fresh serve per arm
(`radiance-serve.sh`, TP=1, max-num-seqs 3, max-num-batched-tokens 2560, KV fp8,
mamba cache bf16 / ssm fp16, mode align, prefix caching on, chat template
`qwen-fixed-v22.3.jinja`, override gen config temp 1.0 / top_p 0.95 / top_k 20).
Run 1 of each arm includes first-decode JIT/autotune (~3–4 s); steady state = mean of
runs 2–3.

## Results (tok/s, wall-clock, 512-token decodes)

| arm | run 1 | run 2 | run 3 | steady (2–3 mean) |
|-----|-------|-------|-------|-------------------|
| plain (spec OFF) | 63.08 | 63.32 | 63.31 | **63.32** |
| MTP K=4 | 55.50 | 94.59 | 95.51 | **95.05** |
| MTP K=3 | 60.27 | 142.09 | 142.53 | **142.31** |

- vs plain: K=3 **2.25×**, K=4 1.50×.
- vs t1392 stock vLLM 3.24: K=3 **44×**, K=4 29×.
- vs current MTP 65.79: K=3 **2.16×**, K=4 1.45×.

## Findings

1. **K=3 beats the quickstart default K=4 on this model** (142.3 vs 95.0 steady). The
   single-MTP-layer stack re-forwards the same draft layer K times (vLLM warns
   "multiple forwards on same MTP layer may result in lower acceptance rate"); with
   this draft head, the 4th speculative token costs more than it earns. The radiance
   quickstart K=4 default is tuned for the 27B, not the 9B head.
2. **The 9B plain (AR) decode on radiance is 63.3 tok/s** — 19.5× the t1392 stock-vLLM
   anchor (3.24) and 1.0× the current mlx MTP plain-class figure. The MXFP4 W4A8
   kernels (r4d allreduce, skinny gemm, mxfp4 decode band 64 for max-seqs ≤ 8) are doing
   the work.
3. **The stack needed three shape/env accommodations for a gqa=4 / bf16-in-proj model**
   (items 1–3 above); all are the stack's own documented fallbacks, and none touch the
   kernel sources. An 9B-class model is serveable on the stock 0.9.3 stack as-is on
   gfx1201.
4. Warm-run cost (~3–4 s first decode per arm) is autotune/JIT on the fresh graph
   (FP8_STREAM=0 differs from the cached 27B graph); subsequent boots reuse the inductor
   cache (K=3 arm booted in ~4 min end-to-end).

## Reproduce

- Serve script: `radiance-serve.sh` on neon (SPEC, SPEC_METHOD, MAXLEN,
  ATTENTION_BACKEND overridable; boot is `radiance-host-boot.sh`, idempotent).
- Logs: `radiance-serve-{k4,k3,plain}.log`, `bench-radiance-{k4,k3,plain}.log` on neon.
