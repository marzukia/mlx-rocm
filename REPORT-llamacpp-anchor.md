# MTP MISSION ANCHOR — llama.cpp + GGUF q4 (Qwen3.5-9B, 9B work model)

Date: 2026-11-02 (Z). Guest: bare-metal fleet box (tailscale, RX 9070 XT 16GB, gfx1201),
ROCm 10.x userland (apt `clang-rocm 10.0+dfsg-0ubuntu1`, gcc 15.3.0, HSA target list includes gfx1201).
Companion to t1392 (vLLM anchor: plain 3.24 tok/s, MTP spec-init crashed on the GDN dynamo view).

**Headline: the llama.cpp path is the working MTP anchor. Native MTP spec decode RUNS on this GDN
hybrid (vLLM died the same model), K up to 3, mean acceptance 2.15/3. Decode 144.3 t/s (MTP) vs
86.9 t/s (plain AR) — vs 3.24 t/s for the vLLM anchor on the same box.**

## Per-arm results (medians first, n=3 per arm)

Protocol = t1392: 1-prompt-token `Hello`, 512 generated tokens, `--ignore-eos`, greedy
(`--temp 0 --top-k 1`), ctx 1024, all 33 layers on GPU (`-ngl 99`), one model load per run.
"Wall" = whole process wall (load + prompt + 512-token generation); "gen t/s" = llama.cpp's own
decode-only measurement from its summary line.

| arm | flag | wall (median, s) | wall all runs (s) | tok/s (512/median wall) | gen t/s (median) | gen t/s all |
|---|---|---|---|---|---|---|
| A plain AR | none | 6.831 | 7.276 / 6.831 / 6.828 | 74.95 | 86.9 | 86.3 / 86.9 / 86.9 |
| B MTP spec | `--spec-type draft-mtp -md <same gguf>` | 5.033 | 5.042 / 5.033 / 5.037 | 101.73 | 144.3 | 144.5 / 144.3 / 144.1 |

Decode-only speedup A→B: 1.66×. Wall speedup (incl. load): 1.36×.
vs t1392 vLLM plain 3.24 tok/s: **llama.cpp plain AR ≈ 27×, MTP ≈ 45×** (vLLM number was the
emulated-GEMM lower bound on an unlocked-clock guest; see caveats).

### Exact flags

```
# common
llama-cli -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 1024 -p "Hello" -n 512 --temp 0 --top-k 1 --ignore-eos
# arm B adds:
--spec-type draft-mtp -md Qwen3.5-9B-Q4_K_M.gguf
```
Env: `HCC_AMDGPU_TARGET=gfx1201` (source build, the fix for the stock docker arch-list gap,
ggml-org/llama.cpp#15106). No LD_PRELOAD needed for llama.cpp (that preload is the torch-specific
knob from the vLLM known-good env).

### MTP arm detail (achieved K)

- llama.cpp init: `spec common_specu: adding speculative implementation 'draft-mtp'`,
  `n_max=3, n_min=0, cache_k=f16, cache_v=f16, ctx_tgt=yes, ctx_dft=yes` — the model's 1 MTP
  layer is chained by the engine to draft up to **3 tokens per step** (n_max=3); no separate
  draft weights needed, the MTP head lives in the same GGUF (`qwen35.nextn_predict_layers=1`).
- Verbose 512-token run: **161 spec steps, 346 draft tokens accepted, mean 2.15/3 per step**
  (early steps: `accepted 3/3 draft tokens`).
- KV: f16 KV cache, ctx 1024 (target + draft contexts).

## STEP 1 — model provenance

**Used an existing q4 GGUF; no local conversion needed.**

- Repo: `https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF` (repo sha 9716a636ee4bddc3fed678220b7a33dd2a4160ae, last modified 2026-05-16). Chosen over the plain `unsloth/Qwen3.5-9B-GGUF` / `lmstudio-community` / `axteam` repos because this one is the MTP-inclusive publish of the same quant ladder.
- File: `Qwen3.5-9B-Q4_K_M.gguf`, 5,868,826,976 bytes (matches the HF LFS size exactly),
  **sha256 `e8dd94817e95d6c0939102049d068418269978377b13616c4726235e232841fe`**, downloaded to the
  agent home on the guest and verified complete by byte size before use.
- Base model: `Qwen/Qwen3.5-9B` (HF public) — 9B, 32 layers repeating [linear_attention ×3,
  full_attention] (GDN hybrid, `full_attention_interval=4`), hidden 4096, `mtp_num_hidden_layers=1`,
  `mtp_use_dedicated_embeddings=false`, vocab 248320, 262144 ctx, M-RoPE partial 0.25.
- **Tensor inventory (442 tensors, via `gguf` package):** MTP head = block 32, 15 tensors, all present:
  `blk.32.{attn_q,attn_k,attn_v,attn_q_norm,attn_k_norm,attn_norm,attn_output,ffn_gate,ffn_up,ffn_down,post_attention_norm}.weight`
  + `blk.32.nextn.{eh_proj, enorm, hnorm, shared_head_norm}.weight` (+ quant scales).
  Metadata: `qwen35.nextn_predict_layers u32 = 1`. Spec arm confirmed these tensors are actually
  loaded and executed at runtime (`create_tensor: loading tensor blk.32.nextn.*`, graph nodes
  `mtp_shared_head_norm`, `mtp_tok_embd-32`).
- On-guest checkpoints (the on-guest model dir: qwen3.5-9B fp8, qwen3.5-9B bf16 src, qwen3-9b mlx)
  were inspected as fallback sources; not needed since the HF GGUF carried the MTP module.

## STEP 2 — build

- llama.cpp **master commit `29c720eaf72fcf8e4e61a56a88b994d92668261d`** ("docs: update s390x build
  docs (#31546)") — native MTP for Qwen3.x IS in master; no special branch needed.
- CLI surface verified in the built binary:
  `--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-*`,
  `--spec-draft-model/-md`, plus `--mtp` (download the MTP head) on the toolchain side.
- Converter support: `conversion/qwen.py` resolves `Qwen3_5TextModel` with `supports_mtp_export=True`
  (MTP block remapped to standard `blk.<N+1>.nextn.*` naming) — so if the HF GGUF had not existed,
  `convert_hf_to_gguf.py` + `llama-quantize --outtype q4_k_m` on the on-guest bf16 checkpoint would
  have worked.
- Build (source, on guest): `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON` +
  `make -C build -j16 llama-cli llama-quantize llama-server`, env `HCC_AMDGPU_TARGET=gfx1201`.
  CMake auto-detected the apt ROCm clang (`/usr/lib/rocm/llvm/bin/clang++`, Clang 23.0.0),
  `hip::amdhip64 SHARED_LIBRARY`, FlashAttention KV types f16/q4_0/q8_0/bf16.
  Binary self-id: `version 0.8.0-dev (build 1, commit 29c720e)`, ggml 0.28.0, libggml-hip.so linked.
  `rocminfo` HSA targets list includes `amdgcn-amd-amdhsa--gfx1201` (no arch-list gap with the
  source build).

## GPU-lane / lock window evidence

- Sibling anchor run (t1433 sampler) held the fleet GPU lock from before 21:01Z; polled at 30 s
  cadence from 21:01:19Z, **lock free at 22:05:47Z**.
- Bench runs executed under `flock` on the lock file: window **22:08:24Z → 22:09:00Z** (6 protocol
  runs; per-run log mtimes 22:08:31 → 22:09:00), plus one 512-token verbose spec-stats run
  **ending 22:09:43Z** (same lock, separate flock invocation).
- Clock lock verified before the window: `power_dpm_force_performance_level=high` (node
  `power_dpm_force_performance_level`, the actual sysfs spelling), `pp_dpm_sclk` active level 2 =
  2200 MHz.
- Release verified after the window: `fuser` on the lock file returned no holders; no stray
  llama/bench/sampler processes of ours remained (all runs were `timeout`-wrapped and exited).

## Caveats

- **GEMM path:** native dequant GEMM — Q4_K_M weights run through the standard q4 dequant kernels
  with f16 compute (llama.cpp's stock HIP path; no emulated/int8 fallback in play, no MXFP4
  emulation unlike the vLLM anchor). This is the main reason the plain AR number is ~27× the
  vLLM anchor: native q4 + locked 2200 MHz clocks vs emulated GEMM at the ~600 MHz DPM floor.
  The comparison to t1392 is therefore an engine+quant+clock comparison, not like-for-like.
- **MTP K:** model ships 1 MTP layer (`nextn_predict_layers=1`); the engine chains it to
  `n_max=3` draft tokens per step. Observed mean acceptance 2.15/3 (≈72% of max).
- **Greedy non-bit-identity:** Arm A vs Arm B greedy outputs diverge from ~token 29 (different
  numerics paths: fused vs split verify). Token streams are not bit-identical across arms;
  expected for spec decode, noted for completeness.
- **Wall vs decode:** "wall" includes model load (~1–2 s from the 5.5 GB file per run) and prompt
  eval; the decode-only `gen t/s` is the like-for-like figure. 512 tokens at 86.9/144.3 t/s
  imply 5.9 s / 3.5 s pure decode, consistent with the walls (6.83 / 5.03 s incl. load).
- **Cards:** the box has two amdgpu cards; the 9070 XT (2200 MHz sclk DPM, the clock-locked one)
  is the device loaded and benchmarked. Card-0 DPM sysfs was busy (iGPU/driver quirk) and unused.
- **Context:** 1024 tokens (bench prompt is 1 token + 512 generated; no pressure on KV).
- llama.cpp master as of this commit prints no aggregate spec-acceptance counter in non-verbose mode;
  acceptance figures come from the verbose run's per-step `accepted N/3` log lines.

## Deliverable status

- Arm A: PASS (74.95 tok/s whole-request, 86.9 t/s decode).
- Arm B: **PASS — MTP spec decode runs on the GDN hybrid** (no vLLM-style init crash),
  n_max=3, mean acceptance 2.15/3, 101.73 tok/s whole-request, 144.3 t/s decode, +66% over A.
- Lock: held for the whole window, released and verified free; no orphan GPU/sampler procs.
