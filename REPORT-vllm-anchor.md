# vLLM radiance anchor bench (RX 9070 XT / gfx1201)

Reference decode throughput from the vLLM side, same GPU + model class as the
mlx-rocm MTP work. This number feeds the 100+ tok/s MTP target math.

## Setup

- **GPU box**: guest box, AMD RX 9070 XT (gfx1201), 16 GiB VRAM, ROCm 7.x
  userland, Ubuntu. All GPU env from the known-good `anchor-serve.env`
  (verified by the prior agent): `MODELS`, `LD_PRELOAD=libhsa-runtime64`,
  `LD_LIBRARY_PATH` incl. the hand-installed OpenMPI 4 dir (torch needs
  `libmpi_cxx.so.40`), HCC target var = gfx1201, `PATH` prefixed with the
  compiler wrapper dir.
- **vLLM**: 0.28.0 ROCm wheel (`vllm-0.28.0+rocm723`, torch
  `2.12.0+git6bbd260` / hip `7.2.53211`) in `venv-vllm-anchor`, installed from
  wheels.vllm.ai.
- **Model**: `qwen3.5-9b-mxfp4a16` (9.43 GiB, MXFP4A16, single
  safetensors, `mtp_num_hidden_layers: 1` in config) — the text-only anchor
  copy of the same 9B Qwen3.5 class the mlx MTP benches run; the `$MODELS`
  mount carries the mlx-format sibling `qwen3-9b-mlx`.
- **Compiler wrappers**: the prior-run `ccwrap` scripts (c++/g++/gcc/hipcc)
  strip `-isystem /usr/include` and inject `cuda_shim.h` for the quark JIT.

## The blocker fix (one line)

The 2026-11-02 13:07Z death was NOT missing ninja: ninja is on PATH
(`ninja` resolves in the venv + `/usr/bin`); the exit-127 subcommand was the
compiler — torch's `ROCM_HOME` guess resolves the wrapped `hipcc` on PATH up to
the agent home, so the build invoked the nonexistent `<home>/bin/hipcc`;
fixed with `~/bin/{c++,g++,gcc,hipcc}` symlinks to the existing `ccwrap`
wrappers, and the relaunch additionally needed the ompi `LD_LIBRARY_PATH`
dir, `--gpu-memory-utilization 0.95` (0.5 left a −6.68 GiB KV budget) and
`--max-num-seqs 16` (GDN/Mamba cache allows only 29 blocks at that budget,
default 256 seqs cannot be captured).

After the fix the quark Stable-ABI `hw_emulation` JIT compiled cleanly
(3.9 s, cached thereafter); server ready in ~2 min.

## Per-run results (median first)

Decode bench: 1 prompt token ("Hello"), `max_tokens=512`, `ignore_eos=true`,
temperature 0, wall-clock of the whole request = 512 decode tokens.

| run | serve | wall (s) | tok/s |
|---|---|---|---|
| **median** | plain | **158.01** | **3.24** |
| 1 | plain | 158.07 | 3.24 |
| 2 | plain | 158.01 | 3.24 |
| 3 | plain | 157.93 | 3.24 |

**Speculative (MTP) serve: not operational** on this wheel + model. Both
attempts crashed at engine init before serving:

1. compiled mode: dynamo fake-tensor `view` failure in the GDN path
   (`shape [3*s, -1, 256] invalid`).
2. `--enforce-eager`, `--speculative-config '{"method":"qwen3_5_mtp",...}'`
   and `'{"method":"mtp","num_speculative_tokens":1}'`: custom RoPE kernel
   `rotary_embedding ... query, key and positions must have the same
   batch_size and seq_len` (pos_encoding_kernels.hip:141), same crash with
   both method names.

The model does expose an MTP head (`mtp_num_hidden_layers: 1`) and vLLM 0.28
has the spec methods, but the spec engine for this hybrid (GDN) model does
not start on this build. No spec number is reported (none was produced); the
plain number above is the reference.

## Exact flags

```
# plain serve (as benched)
vllm serve <model> --port 8601 --max-model-len 4096 \
  --gpu-memory-utilization 0.95 --max-num-seqs 16

# speculative serve (failed to start; recorded verbatim)
vllm serve <model> --port 8601 --max-model-len 4096 \
  --gpu-memory-utilization 0.95 --max-num-seqs 16 --enforce-eager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
# (also tried method "mtp"; and the same config without --enforce-eager)
```

## GPU-lane lock window

`flock` on the guest's `gpu-bench.lock` held for the whole bench session:
acquired **2026-11-02 15:33:21Z** (uncontended, acquired immediately), held
through all runs and server restarts, released ~**15:50Z**. No sibling held
it during the window.

## Caveats (read the number with these)

1. **No native MXFP4 on gfx1201** — vLLM fell back to the emulation kernel
   (simulated weight dequant + activation QDQ, high-precision GEMM) and GDN
   decode fell back to Triton (`fused_gdn_decode_post_conv_mtp` not built in
   the wheel). Emulated GEMM is the structurally slow part.
2. **Guest clock state**: the prior mlx-side diagnosis (t1379) found the
   guest's DPM `sclk` sitting at the ~600 MHz floor through active
   generation (force knobs are root-only, no rocm-smi in this userland to
   re-verify). Absolute tok/s on this box is a **floor**, not the silicon's
   ceiling.
3. The vLLM plain-serve number is the *reference floor*: mlx-rocm
   bit-exact canonical B0 on a healthy session was ~69.5 tok/s (see
   re-baseline report: guest-state drift has since invalidated absolute
   gates on this box, per the warden ruling).

## Bottom line

**vLLM reference decode throughput on this GPU + 9B class: 3.24 tok/s
(median of 3 runs, plain serve, 512-token decode).** MTP speculative
decoding in vLLM 0.28 ROCm is not operational for this model, so the
reference is the plain number. The 100+ tok/s MTP target is ~31× this
reference; with the clock-state and MXFP4-emulation caveats above, the
reference should be read as a lower bound, and the mlx-side comparison
should use the re-baselined (post clock-fix) numbers, not this absolute.
