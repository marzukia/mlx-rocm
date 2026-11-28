# S5-1 spec: collapse the per-step dispatch count on the mmq M=1 path

Date: 2026-11-28 (spec phase, 02:19-02:40Z). Source of record: box tree
`/home/andryo/build-p2seq` at b9cb137 + dirty leg-4 patch (11 modified files,
all mtimes <= 09:44:56Z build start; md5 ledger recorded in the session log).
Control arm of record: venv-ar `libmlx.so` md5 `cdb74a5794` (re-verified 02:22Z).

## Per-call dispatch model (source-read, not inferred)

**C arm (mlx 0.6.1-rc.1+rocm10.1, venv-ar = leg-4 wheel), per Q2_K gemm call
(M=1, ktype 10, in-model bf16 activations, gate passes — all 410 gemm-family
tensors pass the 84 remainder check per S4(1)):**

1. `zeros()` f32 temp fill (dead: `KQuantFileGemm::eval_gpu` builds the f32
   temp via `mlx::core::zeros(...)` then `set_data(malloc_async(...))`
   re-allocates it — the filled buffer is discarded; dead in BOTH arms)
2. `hipMemsetAsync` pad region (15 blocks x 144B; M=1 < J=16 always)
3. `mmq_quant_in_q8_1_mmq` launch
4. `mul_mat_q` (x_file) single launch (no stream-K fixup at M=1)

= **4 dispatches/call**. The dead zero-fill is confirmed by the profiled C mix:
`fill 171.9ms/step` ~= 404 fills x 0.43ms.

**B arm (MLX_KQUANT_FILE_MR=0), same call:** zeros() fill (dead) +
`kqfg_mr_gemm_kernel` (M<=16 path, grid NBS*4 x 256, direct store
`op[...]=acc[j]` with in-register acc from 0 — every op element overwritten)
+ `kqfg_cast` (bf16 out) = **3 dispatches/call**.

C-B = +1/call = +404/step by this model; measured gap 3656-2514 = 1142.
The residual ~738 is not attributable to the per-call source (attention/sdpav, copy, reduce buckets are profiled-cost-inflated in the C arm: 34% inter-kernel gap vs 0.9% in B, per-dispatch cost 118-270us vs 9us — S4(1) flags the C
profiled wall +12.5% vs unprofiled). The GATE is the unprofiled tps, which is what decides.

## Change set (minimal, in-family — NO new kernel, NO new kernel family)

### C1: fold the pad-zero into the quant kernel (removes dispatch #2)

`mlx/backend/rocm/quantized/mmq/mmq.hip` — `mmq_quant_in_q8_1_mmq`:
add trailing param `int64_t real_total = -1`. After the existing
`if (idx >= total) return;`:

```cpp
if (real_total >= 0 && idx >= real_total) {
  // S5-1: pad block — zero 144B (replaces the hipMemsetAsync dispatch).
  char* yb = y + idx * 144;
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  for (int b = 0; b < 9; ++b) *reinterpret_cast<uint4*>(yb + b * 16) = z;
  return;
}
```

All existing call sites (mmq.hip:531, 757 GLU; 1071/1075 Q4_K launcher)
use the default `real_total = -1` — behavior unchanged.

`mlx/backend/rocm/quantized/qmm.hip` — `mmq_q2k_file`:
- DELETE the `hipMemsetAsync` pad block (11 lines: offset/size compute +
  memset + error check).
- Quant launch: `total` arg becomes `x_tiles_pad`, add `x_tiles` as
  `real_total`. Grid becomes `(x_tiles_pad + tb - 1) / tb`.

Correctness: the pad region bytes `[x_tiles*144, x_tiles_pad*144)` are
exactly the bytes the removed memset zeroed; the kernel zeroes them EVERY
call (no buffer-reuse state dependency). Real-block writes untouched.
Bit-identical to the removed memset.

### C2: replace the dead `zeros()` f32 temp with an uninitialized buffer
(removes dispatch #1)

`mlx/backend/rocm/quantized/kquant_file_gemm.hip` — `eval_gpu`:

```cpp
// OLD:
  tmpf32 = std::make_unique<array>(
      mlx::core::zeros(Shape{...}, float32, s));
  encoder.add_temporary(*tmpf32);
  f32_target = tmpf32.get();
  ...
  if (f32_target != &out) {
    f32_target->set_data(rocm::malloc_async(f32_target->nbytes(), encoder));
  }
// NEW:
  tmpf32 = std::make_unique<array>(
      rocm::malloc_async((int64_t)M * NBS * 256 * sizeof(float), encoder),
      Shape{...}, float32);
  encoder.add_temporary(*tmpf32);
  f32_target = tmpf32.get();
  // (the set_data reallocation is GONE — the buffer IS the device buffer)
```

Safety (verified by source read): every kqfg kernel path
(`kqfg_gemm_kernel` grid (M+63)/64 x NBS x 256; `kqfg_mr_gemm_kernel`
grid NBS*4 x 256) writes ALL op elements before the cast kernel reads them
(direct store; in-register accumulation from 0; no read-modify-write).
On the mmq path the temp is unused entirely. Lifetime identical:
`add_temporary` + array owns the buffer until encoder commit.
`f32_target == &out` (f32 output) path: no temp, unchanged.

### Named structural sources — disposition

- **(a) per-op `flush_all_dirty_shadows`**: verified ALREADY "only-when-dirty"
  (memcpys issued only for dirty buffers; clean cost = mutex + O(1) empty-set
  iteration; no dispatch when clean). The global `get_command_encoder`
  (with flush) is used by load/reduce/matmul/binary/ternary/event paths —
  not by the 404 kqfg calls (those use the Device member, no flush).
  Host-side microsecond cost; S4(1) already ruled the host class out
  (3.2ms/step). **No change.**
- **(b) per-call xq81 staging**: addressed by C1 (the staging itself = 1
  malloc from the pool, no dispatch; the pad memset is C1's target).
- **(c) per-chunk stream in `shadow_chunked_copy`**: only for >100MB copies;
  per-step token embedding upload is a small one-shot memcpy (no stream);
  model load is one-time. NOT a per-step dispatch source. **Out of scope
  (S5-2 lane with the 1384 leak).**

## Predicted per-step dispatch count

- Baseline (profiled, S4(1) class): C 3656, B 2514.
- C1: -404 (one pad memset per gemm-family call; -403 if the lmhead N%256
  edge falls to kqfg — the 410-tensor 84-remainder table says all pass).
- C2: -404 (one dead zero-fill per gemm-family call, both paths).
- **Predicted test arm: 3656 - 404 - 404 = 2848 (class 2800-2900).**
- Residual vs 2514: 334 = the essential per-call quant dispatch (404;
  fusing it would need a NEW fused quant+GEMM kernel = guardrail 2,
  out of scope) minus ~70 of profiled-measurement asymmetry.
- Consistency check: 2848 - 404 = 2444 ~= 2514 (within measurement noise) —
  i.e. the C-B gap is ~fully explained by the 2 removed dispatches + the
  essential quant, with no unexplained C-specific structure left.

## Build protocol

- No .cuh edits (C1: mmq.hip + qmm.hip; C2: kquant_file_gemm.hip).
  Trap protocol applied anyway: `touch mmq-host.hip qmm.hip` before build.
- Build: `MLX_VENDOR_DIR=/home/andryo/build-p2seq/vendor-mlx
  bash scripts/build-wheel.sh` (box), new wheel -> new venv (venv-s51),
  control venv-ar untouched.

## Gate

Measured tps improvement, test vs control, SAME session, UNPROFILED
(arm protocol per P2 card). Dispatch count is the mechanism, not the gate.
