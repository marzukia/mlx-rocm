# LEVER-FINDING - NEON 2.5-bit Q2_K vehicle (leg 2.6 ROUGH baseline + oracle cross-check)

jimmy - 2026-11-26 14:0xZ - lane: NEON 2.5-bit 27B front (op 11-26) - this card is the LEAD for monky's 4-item record (helium).

## Verdict

The vehicle runs Q2_K end-to-end on the 16GB dGPU. Both pre-registered ROUGH gates trip (HBM peak 15.84G > 14G; 0.28 tok/s < 20). A 147x gap vs llama.cpp on the SAME file and SAME dGPU localizes the lever to the vehicle's Q2_K file-GEMM path plus the HBM arena retain/reclaim interaction - not to an inherent Q2_K cost on gfx1201.

## Numbers (one run, ROUGH class, 2026-11-26 12:58:39Z -> 14:02:50Z, rc=0)

- Load: 866 tensors in 1.9s (506 raw uint8, 360 f32). MTP-ON load fault (shared-head design: file has no standalone self_attn/mlp for nextn, only nextn.{eh_proj,enorm,hnorm,shared_head_norm}) -> MTP-OFF fallback (pre-registered fault class). Build 0.5s.
- Prefill 4096 tokens: 150.0s (512-token chunks, kv q8_0 on 17 layers).
- Decode 1024 tokens: 3693.86s = 0.28 tok/s. ms/step p50 3607.2 (min 3600.3, max 3622.3 - flat). tokens/step 1.
- HBM: peak 99% = 15.84G at prefill; AR steady 94-96% = 15.04-15.36G. Gate < 14G: TRIPPED.
- Coherence: 5gram-dup-rate 0.000 (0/255) - coherent multilingual word-salad class (d500x prompt), same class as the 128-token smoke.
- dGPU mid-run sample: SCLK 3270MHz, 227W, busy 98% - not launch-starved. Flat per-step cost = stable bottleneck; 9.4G weight read per step at 0.28 tok/s = ~2.6 GB/s effective HBM.
- Regime: 3s HBM sampler (400 samples = 20min loop cap - going quiet after 13:19Z is loop exhaustion by design, not a vehicle bug), rocm-smi clocks at start (sclk 1009MHz idle, mclk 772MHz) and exit (sclk 3239MHz post-boost, mclk 96MHz), gpu.lock run-only (held full window, released at exit), exit VRAM 0%.

## Oracle cross-check (llama.cpp b12062, commit 29c720eaf, GGML_HIP + MMQ_MFMA)

- Same file + same prompt bytes (llama-cli reads the file raw; the vehicle replaces newlines with spaces before tokenizing).
- Full llama-cli invocation (control): env HIP_VISIBLE_DEVICES=0 HCC_AMDGPU_TARGET=gfx1201 llama-cli (perclass-probe/llama.cpp build dir on the box) -m models/Qwen3.8-27B-Q2_K-plain.gguf -f d500x-20261115/prompt.oneline -n 1024 --temp 0 --seed 42 -ctk q8_0 -ctv q8_0 -c 6144 -ngl 99 -t 16 (b12062, 29c720eaf, GGML_HIP + MMQ_MFMA build; GPU[0] only).
- Prompt 54.0 t/s, generation 41.2 t/s = 147x the vehicle's 0.28 tok/s, rc=0.
- First divergence: TOKEN 1 (vehicle "ജ" vs llama "[Start"). Tokenizer-handling class, expected given the newline treatment - not a numerics signal.
- llama output class: coherent thinking-mode for ~2/3 of the 1024, then degrades into a "b" run (5gram-dup-rate 0.234 (100/427), 105 b-lines of 431 words). Q2_K-class behavior on an adversarial prompt; recorded, not a gate.

## G1 (bit-exact dequant, fresh re-run on the installed artifact)

- Oracle: hand-computed Q2_K block reference (numpy, GGUF spec layout: d@80, dmin@82, 64 scale/min bytes, 2x4 sub-quant scale pairs, 2-bit indices) on a real file tensor, blk.3.attn_v.weight dims (5120,1024), nb=20480, read at raw GGUF offsets (not a ggml library ref).
- G1a kq_dequant (flat stream): BIT-EXACT (n_mismatch=0, max|d|finite=0.0, nan-aware; 152832 nans in both).
- G1b kquant_file_dequant (the vehicle path): BIT-EXACT (same numbers).
- G1c kquant_file_gemm end-to-end vs x@ref (f32, seed 42): max|d|finite=0.0, n_eq=0/16384 = all-nan-both comparison - CLOSED with a different answer than first reported: the 152832-nan face was a 4-byte PARSER shift in my extraction scripts, not file content. GGUF pads the data-section start to 32-byte alignment (true base = DS + (-DS % 32)); the unpadded header end shifts every 84-byte block by 4 and makes qs bytes parse as d/dmin halves (random bytes hit the NaN-half pattern at ~2.9%, matching the phantom 597 "all-nan blocks"). THE FILE IS CLEAN: block census at the true base (tools/q2k_block_census.py, committed) = 0 NaN halves / 0 Inf halves over all 506 Q2_K tensors (106,711,040 blocks), rc=0. The "source Q4_0 contamination" (2939/52 NaN-d blocks) was the same artifact on the source file. The instrumented real binary (rebuilt from the tree, byte-identical output sha256 3b8d3c60...) confirms the quantizer never wrote a NaN: _ref/_impl write-scan 0, work-buffer scan 0 on all 506 tensors, post-write file scan 0 on blk.3.attn_v. G1a/G1b/G1c verified the dequant/GEMM math on the given byte buffer (math is position-independent) - the bit-exact verdicts stand; the "file-data property" explanation is retracted.
- G1 gate: PASS - the wheel ships.

## Profile lead (top-2 kernel cost, measured 14:1xZ on the installed artifact)

- Method: rocprof v1 unsupported on gfx1201, rocprofv2 preload lib missing (host change not allowed) - direct per-call timing of kquant_file_gemm over all 410 large Q2_K tensors, two consecutive passes, steady state (script on the box: gemmtiming.py).
- TOP-1: kquant_file_gemm (Q2_K, ktype-10), 410 calls/step: 3785ms (pass 1) / 3689ms (pass 2) = 102% of the 3607ms/step step - the GEMM family IS the entire step cost. 9.0ms per call, effective 2.55 GB/s per weight read (9.4G/step).
- TOP-2: all other per-step work (96 small-tensor dequant-once gemms, conv/ssm/attn/norm, HBM arena churn): ~0ms (GEMM-dominated step; pass 2 = pass 1, no warmup/pool effects).
- Comparison: llama.cpp mmq on the same file = 24.3ms/step (41.2 tps) ~ effective 387 GB/s = 152x kernel-path gap over the vehicle ktype-10 GEMM.
- Mechanism class: the Q2_K file-GEMM kernel reads weights at ~0.4-0.8% of HBM peak - an mmq miss (ktype-10 non-mmq fallback, or an inefficient Q2_K mmq instantiation on gfx1201); monky items 1+2 identify which. Arena 94-99% residency is a consequence (9G weights + KV + graph), not the per-step cost driver.

## LEVER (monky - 4-item record, helium)

0.28 tps vs 41.2 tps on the same file, same dGPU: the vehicle path owns the cost.

1. Which GEMM path does ktype-10 (Q2_K) take on gfx1201 - mmq routing or a dequant->bf16 fallback? Evidence banked: 1031 kqfg launches per step (smoke timing), flat 3.607s/step.
2. Effective HBM bandwidth per weight read on the vehicle path (MEASURED: 2.55 GB/s at 9.4G/step, 9.0ms/call, gemmtiming.py pass 2).
3. HBM arena retain/reclaim cost per step (arena at 94-99% of 16G; t3437 dive: weights NOT host_backed, arena = driver backing for hipHostMalloc - retention-at-stock confirmed).
4. The stack lever itself: i-quant conversions + Q2_K routed to the rdna4 mmq kernels (12 refs) - quantify what the vehicle path leaves on the table vs the 41.2 tps mmq class.

## Provenance (identity block)

- File: Qwen3.8-27B-Q2_K-plain.gguf, 8985410528 B, sha256 3b8d3c602f63414794d5f43ceb22bbb61eb586af2cddf07cbf10656860f5b80a. Built host-side on neon 11-26 11:43Z: the unsloth Q2_K_XL (16056478688 B) carries a distinct type not in the raw-passthrough list (Q2_K/Q5_K/Q6_K), so a PLAIN Q2_K was built from the 4-bit source (recipe + source tree on the box).
- Wheel (vehicle print, baseline of record): mlx-0.32.3.dev20261126+b9cb137-cp314-cp314-linux_x86_64.whl, wheel md5 fc4bd14850cddee8e11df1d24fde1d9c (the 12:27Z j-fix build, logged by the runner header at run start), tree sha b9cb137. One-line note: the installed artifact at run time verified as the wheelout3 build (libmlx.so md5 3cc8483f3c44fe22a895aac5c4edd47d, installed 12:41:01Z) - all three builds share tree sha b9cb137 with identical core.so (424527c9e8483c871d6b1f0094608b0f); G1 re-run was on the installed artifact. md5 differs from the prior (pre-Q2_K) wheel. Built on neon.
- Tree: mxr-q2k branch q2k-ktype10-vehicle (4 code commits + tooling over origin/main 3e84320e: d88bc82 ktype-10 Q2_K file-GEMM family at the 3549-named sites; 693d4f0 G1 j-term fix in kq_dequant_q2k; 067b7fb gguf raw-load bsize tolerance + the missing copy; dce81ac persona-scrubbed branch copy of this card at docs/lever-finding-q2k-vehicle.md; + tools/q2k_block_census.py, the corrected 32-byte-aligned GGUF block census).
- Box: neon, gfx1201 RX 9070 XT 16GB, ROCm 7.2.0, python 3.14, venv venv-ar.

## Check lines

- style-format: `uv run ruff format --check .` -> 107 files already formatted (PASS, post card-commit).
- persona-leak: 0 hits over diff 3e84320..HEAD (added lines, tree content, re-verified after the card commit dce81ac).
- line-wrap: 0 hits over this card body (detector hm2 scripts/line_wrap_check.py, b0ad844 class, rc=0).
