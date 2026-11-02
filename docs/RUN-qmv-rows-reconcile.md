# RUN card — QMV-ROWS reconcile (t1281): build + GPU verify the ws-fix kernel

Purpose: t1265 verdict was FAIL — with `MLX_ROCM_QMV_ROWS=1` engaged, the
multi-row kernel (`qmv_rows_msg_kernel`) produced garbage from token 0 on the
real 9B f32 path, in BOTH the V1 and V2 wheels. Root cause (t1281 audit,
REPORT.md §QMV-ROWS reconcile): a toolchain codegen defect in ROCm 10.0 /
clang 23.0.0rocm7.14 that compiles the 32-element `wn[V]` weight-decode
array in this kernel into a no-op (qnan-poisoned accumulator, k-loop body
deleted — 0 global loads in the SASS of all 18 instantiations). The source
itself was audited line-by-line against `qmv_wide_kernel` and is
arithmetically faithful (no divergence). The scoped fix restructures the
decode to a 4-element word array (`ws[4]`) with the identical
shift+mask+cvt chain run inline in the MAC loop — bit-identical arithmetic
(same q-major j-minor FMA order, same int→float conversion). This card
verifies the fix on the GPU.

## Guest state (already done, host-side)

- Fixed kernel installed in the V2 build tree:
  `/home/andryo/vendor-v2/mlx/backend/rocm/quantized/qmv_rows_msg_kernel.hip`
  md5 `3dc576ba9733b61415ea58587ae087c7`.
  Pre-fix backup: `qmv_rows_msg_kernel.hip.pre-wsfix` in the same directory.
- `patches/qmv-rows-msg-kernel-wsfix.diff` = V2-kernel → fixed diff
  (applied to the guest tree; also the authoritative copy is committed at
  `vendor/mlx-rocm-upstream/mlx/backend/rocm/quantized/qmv_rows_msg_kernel.hip`).
- Dispatch: UNCHANGED. The V2 tree's `qmm.hip` already carries the
  rows-msg dispatch (opt-in `MLX_ROCM_QMV_ROWS=1`, band `M in 2..4`,
  `N >= MLX_ROCM_QMV_ROWS_MIN_N` default 16384, `K % 32 == 0`, f32 branch
  `T=ST=float`). The fix touches the kernel file only.
- Compile verification (host-side, guest toolchain, done):
  - bar (`clang++ -x hip -D__HIP_PLATFORM_AMD__=1 --offload-arch=gfx1201
    -fsyntax-only -I/usr/include`): fixed kernel = identical profile to the
    pre-fix kernel (3 pre-existing host-side `__launch_bounds__` parse notes;
    0 device-side errors).
  - `-O3 -Xclang -emit-llvm`, all 18 instantiations (f32/bf16/f16 × GS
    32/64/128 × BITS 4/8): fixed = 56 (4-bit) / 40 (8-bit) `addrspace(1)`
    loads + 8 scratch loads per kernel; pre-fix control in the same harness
    = 0 global loads in every one of the 18 (reproduces the defect).
  - `-O3 -S` SASS: fixed = 14–20 `global_load` + 4 `global_store` per
    instantiation + 1680 `v_fmac_f32_e32` / 240 FMA-class ops total;
    pre-fix control = 0 global loads.
  - Trigger bisection (probe matrix, `-O3` IR): 1-word/8-value decode array
    survives; 2-word/16-value survives; 4-word/32-value degenerates —
    independent of 16B vs 4B weight load and of `xn` array vs scalar x
    loads. The 4-element `ws[4]` form (with direct-into-la control)
    survives.

## Step 1 — rebuild the V2 wheel (ON guest, GPU lock)

Same provenance as the t1265 V2 wheel
(`mlx-0.32.3.dev20261101+75ca881-cp314-cp314-linux_x86_64.whl`), rebuilt
from the now-fixed tree. Build the wheel from
`/home/andryo/vendor-v2` exactly as the t1265 V2 wheel was built
(`python -m pip wheel . --no-deps -w <wheelout-v3>/` under the v2 venv;
`-O3` build flags per `compile_commands.json` in the build tree), output to a
FRESH directory (e.g. `/home/andryo/wheelout-v3/`) so the t1265 artifact is
not clobbered. Install into a fresh venv (or the t1265 v2 venv after
recording its current wheel name).

Post-build SASS spot-check (cheap, no GPU needed): carve the GPU ELF out of
`qmm.o`'s `.hip_fatbin` (the t1265 carve recipe), `llvm-objdump -d` it, and
confirm `qmv_rows_msg_kernel` instantiations now show `global_load_b128` /
`global_load_b32` instructions (expect the 14–20 per-kernel profile above;
t1265's wheels showed 0). If a kernel still shows 0 global loads, STOP —
the build picked up a stale object; check `qmm.o` mtime vs the source
mtime (the t1265 provenance lesson).

## Step 2 — probe 2: f32 bit-exactness at K=4096 N=248320 M=4 (the t1265 shape)

The 9B f32 path (the op promotes bf16 x when scales are f32). The lm_head
shape: N = 248320, K = 4096, M = 4.

1. **Engagement bracket** (proves the rows kernel is actually engaged):
   - `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=16384` → band floor
     16384 ≤ 248320 → rows kernel engaged.
   - `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=400000` → floor above N →
     rows kernel NOT engaged (per-row wide path).
   - `MLX_ROCM_QMV_ROWS` unset → not engaged (default off).
2. **Bit-exact A/B**: run the same 4-row f32 QMV (the t1265 probe 2
   harness: 4 rows of f32 activation × the 4-bit lm_head, scales f32) with
   rows engaged (line 1 above) and rows disengaged (line 2 or 3 above);
   compare the 4×248320 outputs: **maxdiff must be 0.0 and md5 equal**
   (the row-exact contract: output row r == `quantized_matmul(x[r:r+1])`
   bit for bit, per row, against the per-row wide path).
3. **The t1265 failure check**: t1265 saw garbage from token 0 with the
   rows kernel engaged. With the ws-fix wheel, the engaged run must match
   the per-row run bit for bit — the garbage must be gone.

PASS: md5(rows-on) == md5(rows-off) and maxdiff 0.0; the bracket runs
confirm engagement (e.g. via the t1265 timing delta or a dispatch
counter/log if the probe prints one).

## Step 3 — N=12288 shape (gate_up band)

N = 12288 is below the default 16384 band floor, so engage the rows kernel
there explicitly:

- `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=8192` (floor 8192 ≤ 12288,
  M = 4, K = 4096 on the 9B f32 path — `K % 32 == 0` holds) vs the same
  run with `MLX_ROCM_QMV_ROWS_MIN_N=20000` (not engaged).
- Compare: **md5 equal, maxdiff 0.0** (row-exact, 4 rows × 12288).

This shape exercises the `N < 16384` band with an explicit floor — the
band logic itself is unchanged by the fix, but the check confirms the
kernel is correct at a shape where the default config would never engage
it (guards against the fix only working at the large-N shape).

## Step 4 — K=3 lossless A/B (the t1228 method)

Full-model spec-decode A/B, the t1228 protocol:

- Run A (rows off, model default): the t1228 K=3 arm
  (`bench/mtp-bench6.py`, K=3, MAX_TOKENS per t1228), the 9B f32 path.
  With the t1265-era band, the default run does not engage rows (M=4,
  N=248320 ≥ 16384 — but `MLX_ROCM_QMV_ROWS` unset → off).
- Run B (rows on): same run + `MLX_ROCM_QMV_ROWS=1` (default floor 16384;
  the lm_head N=248320 engages the rows kernel on the M=4 forwards).
- **Verdict: `md5(ALLTOKS)` of Run A and Run B must be byte-identical**
  (lossless: the token stream cannot change if the QMV outputs are
  bit-exact), and record tok/s for both (rows-on should be ≥ rows-off if
  the morphology wins on this shape; t1253's lab number was 2.59 ms vs
  4.00 ms per-row at N=248320 M=4 — a win, but the A/B only REQUIRES
  the md5 match; the timing is reported, not gated).

PASS: md5 match + both runs complete; tok/s delta recorded.

## Verdict

- **PASS**: probe 2 bit-exact (Step 2) AND N=12288 bit-exact (Step 3)
  AND K=3 md5 match (Step 4), with the Step 1 SASS spot-check showing
  global loads present.
- **FAIL**: any mismatch → (a) confirm engagement (Step 2 bracket) — a
  "mismatch" with rows never engaged is a harness error, not a kernel
  error; (b) if engaged and mismatched, re-run the SASS spot-check on the
  built `qmm.o` (0 global loads = stale/miscompiled object, redo Step 1;
  loads present but wrong numbers = report the SASS + the probe outputs
  for re-audit).

## Notes

- The fix is kernel-file-only; the V2 dispatch, the f32 branch, and the
  band logic are untouched. Default-off (`MLX_ROCM_QMV_ROWS` unset) means
  a misbuilt wheel is not a regression risk for the default path.
- If the guest's ROCm 10.1 toolchain (per vendor.json) differs from the
  10.0 userland this was compile-verified under, the SASS spot-check
  (Step 1) is the arbiter — the pass signature (global loads present in
  all 18 instantiations) is toolchain-independent.
