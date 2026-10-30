# an earlier runc: TAIL FUSION lever 2 — norm/GDN tail (gdn_prep + gdn_norm_gated)

Salvage/finish of the burned an earlier run run. Goal: `MLX_GDN_TAIL_FUSION=1` on the ROCm
wheel, **≥100 tok/s at T=1, LOSSLESS** (bit-exact stream vs reference m0), on
the build VM (RX 9070 XT / gfx1201, ROCm 10.1 guest userland).

## What the lever does

`gdn_prep` fuses the GDN pre-chain (depthwise conv1d + silu, q/k RMS-norm,
beta sigmoid, compute_g) into one kernel; `gdn_norm_gated` fuses the GDN
post-chain (RMS-norm × precise_swiglu) into one kernel. The model
(`mlx_lm/models/qwen3_5.py`, `GatedDeltaNet.__call__`) takes the fused branch
when `MLX_GDN_TAIL_FUSION=1` and (this run) `S == 1` — decode only; prefill and
MTP-verify (S>1) fall back to the non-fused path, which is what the reference
m0 stream used.

## The v5 fault (an earlier run) and its root cause

The 07:40 wheel (built from `fa18fbf`, pre-fix source) faulted at prefill
(S=27): `grid = conv_dim = 27*8192 = 221184` workgroups with **no
`c < conv_dim` guard** → conv_state/conv_weight reads up to index
`2*221184+221183` / `4*221184-1` against 24576/32768-element buffers →
unmapped-VA read → HSA memory fault (addr 0x1e30b000).

Isolation on the 07:40 wheel (fresh process per test, flock):

| test | result |
|---|---|
| `gdn_prep` alone (T=1 shapes, zero + random) | **FAULT** — grid=8192=conv_dim, no guard; addr 0x1a2db000 |
| `gdn_norm_gated` alone (T=1 shapes) | **FAULT** — grid=8192 (pre-extraction `num_heads` from 1-D weight [128]); addr 0x40e5c000 |
| fused T=1 layer forward (fused branch) | **FAULT** in `gdn_prep` (same root cause) |

The source at `fbb7dc9` already carried the grid/guard fixes
(`grid=(conv_dim+255)/256` + `if (c < conv_dim)`; norm_gated
`head_dim = w.size()`, `num_heads = o.size()/head_dim` — correct for this
model's 1-D `[128]` norm weight). The remaining defects were bit-exactness
and (found by an earlier run) a launcher pointer bug — see below.

## The an earlier run fault (dist-g4c wheel) — root cause (an earlier run)

The dist-g4c wheel (built 09:01 from the 96a1677 tree — verified: the
installed .so's kernel symbol carries the f32 `g_out` param, and the
launcher disassembly computes `grid=(conv_dim+255)/256`) still faulted
deterministically in fresh processes, in **both** size variants (an earlier run
minprobe: conv_dim=512 AND 8192), with varying fault addrs (0x1a150000 /
0x29db5000 / 0x39a4f000). The non-fused control passed.

**The grid(8192) analysis was a misattribution.** The ROCm hang-analysis
"grid" field is total work-items (blocks × 256), not blocks: grid=[8192]
= 32 blocks, grid=[512] = 2 blocks — exactly what the fixed launcher
produces. The kernel indexing is fully guarded (`c < conv_dim`,
`t < value_dim/128`); no OOB at any size.

**Root cause (surgical):** all 14 pointer args in the three launchers
were `arr.data_shared_ptr().get()` — a **host-side `array::Data*`**
(struct: `allocator::Buffer` + `Deleter`), not a device pointer. array.h
says so explicitly ("When accessing the data for GPU kernels, be sure to
use the correct method / function for the given backend to access the GPU
pointer"). Every other ROCm kernel uses `gpu_ptr<T>(arr)` =
`buf->data + arr.offset()` (device ptr, + host-shadow flush). The GPU
dereferenced a host-heap address as an element array → unmapped-VA read →
HSA memory fault. Host addrs vary per run (fault-addr variance) and the
first access (`conv_state[0]`) faults at any size (size-independence) —
matching the an earlier run signature exactly.

**Fix (commit 75ca881):** `gpu_ptr<void>(arr)` / `gpu_ptr<float>(arr)` in
all three launchers (gdn_prep, gdn_qk_norm, gdn_norm_gated). One file,
11 lines.

1. **g math**: fused chain used f32 sum, f32 `LogAddExp`, bf16 g output —
   three rounding mismatches vs non-fused
   `(bf16)(f32(a)+f32(db))` → `(bf16)(max+log1p(exp(min-max)))` → f32.
2. **g dtype**: non-fused feeds the recurrent GDN kernel an **f32** g (the
   kernel reads it as u16 pairs — the even v-heads get eg≈0, odd heads reuse
   the even head's g; baked into m0). The fused path cast g to bf16.
3. **qk scale**: fused `gdn_qk_norm` multiplied by an f32 scale without
   pre-rounding; non-fused python `float * bf16-array` casts the scalar to
   bf16 first (inv_scale is not exact in bf16, so k's scale differs).
4. **prefill OOB**: kernels are T=1-only but the fused branch ran for all S.

## Fixes (commit 96a1677 on the vendor fork:tail-fusion-gdn)

- `gdn_prep.hip`: bit-exact compute_g chain (bf16 sum, bf16 softplus via
  `max+log1p(exp(min-max))`, f32 `expf`); g output stays **f32** (kernel param
  `float*`, launcher cast, `fast.cpp` 5th output dtype `float32`).
- `gdn_prep.hip` qk_norm: scale pre-rounded to bf16 before the f32 multiply.
- `qwen3_5.py` (venv-d model): fused branch gated on `S == 1`.

Verified correct as-is (no fix needed): `gdn_norm_gated` extraction (norm
weight is 1-D [128] → head_dim=128, num_heads=32, no OOB); conv1d weight
layout [8192,4,1] = [C,K,1] matching `nn.Conv1d`; depthwise conv k-order f32
accumulation; silu = `Multiply{}(x, Sigmoid{}(x))` op-struct reuse.

## Gates (venv-d, dist-decode wheel +75ca881, fresh process, flock, LD_LIBRARY_PATH)

- [x] G1 primitives: gdn_prep / gdn_norm_gated vs non-fused chain, zero + random
  inputs → maxdiff 0.0
- [x] G2 fused T=1 layer forward: TAIL_FUSION=1 vs =0 → maxdiff 0
- [ ] G3 1024-token T=1 bench, TAIL_FUSION=1 → tok/s ≥100, stream bit-exact vs
  m0 (`/mnt/models/tmp/bench4/m0.out`, 64.49 tok/s non-fused reference) —
  **PENDING (next run)**
- [ ] G4 TAIL_FUSION=0 vs =1 → streams identical (toggle is a no-op when off) —
  **PENDING (next run, with G3)**

RESULTS (an earlier run, 2026-10-28, dist-decode wheel +75ca881):

- G1a `gdn_prep` alone, T=1, zero+random: **no fault** (was: deterministic HSA
  memory fault), qn/kn/vc/beta/g maxdiff = 0.0 (10/10), g_dtype=float32
- G1b `gdn_norm_gated` alone, zero+random: **no fault**, maxdiff = 0.0 (2/2)
- G2 fused T=1 layer forward (qwen3-9b layer 0): fused vs unfused
  maxdiff = 0.0
- Note: an earlier run's orphaned minprobe (hung 74 min on the faulted queue) was
  killed before the gates; all gates ran on a clean GPU (flock).
- G3/G4 (bench ≥100 tok/s + lossless stream vs m0) are the next run.

## Vendor pin

`vendor.json` primary re-pinned to `the vendor fork` @ `75ca881`
(branch `tail-fusion-gdn`) — the exact source dist-decode was built from
(96a1677 + an earlier run launcher pointer fix; same CMAKE_ARGS as before:
`-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS`,
`CMAKE_PREFIX_PATH=/opt/rocm/core-10.1`).
