# Adversarial review — PR #2 (k2: non-GEMM fusion slice)

- Repo: marzukia/mlx-rocm, diff `c51a835..8185109` (15 files, +1576, all additions)
- Method: read-only, file-by-file diff read; base versions diffed from `c51a835`;
  size gate run locally (`uv run python scripts/size_ratchet.py --base HEAD`);
  no build, no GPU. Bench evidence taken from `reports/k2-non-gemm-fusion.md` (not re-run).
- Reviewer: fresh session, read-only. Submitter of record: submitter.

## Verdict summary

Core fusion logic is sound: the fused rms_norm+residual kernel is a faithful port of
the existing (bench-validated) block-reduction pattern, the uint4/D=8 fix is verified
in the final state, and the in-place KV append is safe for the qMLX flow. However,
the slice has **5 MAJOR gaps**: one ungated default-path change to the existing
`rms_norm` op (violates the "env unset → bit-identical" contract), one latent
layout bug in `RMSNormResidual`, and three missing validations / one broken fallback
in `kv_append`. None break the benchmarked qMLX flow (contiguous, same-dtype, GPU
stream), so there are no BLOCKERs — but per the rubric (PASS = no BLOCKER/MAJOR)
the verdict is FAIL. All fixes are small.

## Findings

| # | Sev | Location | Finding |
|---|-----|----------|---------|
| 1 | MAJOR | `vendor/.../rms_norm.hip:461-470` | Ungated default-path change: new warp fast-path in existing `RMSNorm::eval_gpu` |
| 2 | MAJOR | `vendor/.../rms_norm.hip:500-510` | `RMSNormResidual::eval_gpu` packed check ignores `rin` layout |
| 3 | MAJOR | `vendor/.../fast.cpp:221-225` | `kv_append` CPU fallback uses stop=0 → empty slice → always throws |
| 4 | MAJOR | `vendor/.../fast.cpp:174-219`, `copy.hip:343-347` | `kv_append` has no dtype check; kernel templated on `k.dtype()` writes into cache buffer of possibly different dtype |
| 5 | MAJOR | `vendor/.../copy.hip:287-299` | `kv_append` assumes standard [B,H,*,D] layout (no contiguity validation) |
| 6 | MINOR | `scripts/verify-k2.py`, `scripts/k2-d8-probe.py` | Unit coverage gaps: block path, scalar (D%VEC≠0) path, non-contig residual |
| 7 | MINOR | `reports/k2-non-gemm-fusion.md:44-52` | Per-kernel counts don't reconcile with stated step count; -21.7/step vs naive -72/step expectation unexplained |
| 8 | MINOR | `vendor/.../fast.cpp:146` | `rms_norm_residual` out dtype = result_type(x, weight) ignores residual dtype (undocumented) |
| 9 | NIT | `vendor/.../copy.hip:225-231` | `if constexpr (VEC == 8)` branches are byte-identical (both copy uint4) |
| 10 | NIT | `reports/k2-non-gemm-fusion.md:3`, `reports/opt-gauntlet-areaB-backend.md:6` | Machine identifiers in reports (<guest-ip>, guest-a) — docs, not code; no secrets or personal home paths in code/scripts/patches |
| 11 | NIT | `patches/k2-mlx-lm-model-patches.diff`, `patches/k2-rocm-python-binding.patch` | Patched model files reference `mx.rocm` unconditionally (ROCm builds only); binding forces `Device(gpu)` with no stream (consistent with existing rocm binding pattern) |

### MAJOR-1 — `rms_norm` default path changed, ungated

`rms_norm.hip:461-470` inserts, into the **pre-existing** `RMSNorm::eval_gpu` (the
only modification to existing code in the whole PR — fast.cpp/fast.h/
fast_primitives.h are pure additions, 0 removed lines):

```cpp
} else if (n_rows <= 64 && axis_size <= 8192) {
  // K2: decode fast path — one warp per row (no block reduction).
  ...
  &rocm::rms_norm_warp_kernel<DT>, ...
```

This is not env-gated. Every decode-shaped `rms_norm` call — per-head q/k norm
(n_rows=32, axis=128) and input/post-attention norm (n_rows=1, axis=4096) — now
routes to the new warp kernel instead of the block kernel, in **both** bench arms
(the report's own OFF trace shows `rms_norm_warp` present in the unpatched-model
arm). The two kernels accumulate the sum of squares in different orders (block:
per-thread stride 1024 + two-stage warp/shared reduce; warp: per-lane stride 128
+ single shfl_xor tree), so the normalizer can differ by ~1 ulp. Not bit-identical.
Impact is small (same algorithm, ulp-level; the A/B delta is clean because the OFF
arm includes the change), but it violates the stated contract "with the env unset
the original path must be bit-identical". Fix: gate the branch on an env var
(e.g. `MLX_ROCM_RMSNORM_WARP`, default off) or explicitly document it as an
intentional default-path change in the PR description.

### MAJOR-2 — `RMSNormResidual` packed check ignores the residual's layout

`rms_norm.hip:500-510`:

```cpp
bool last_contig = ndim >= 1 && xin.strides()[ndim - 1] == 1;
bool packed = xin.flags().contiguous && last_contig;
...
array x = xin;
array r = rin;
if (!packed) { x = contiguous_copy_gpu(xin, s); r = contiguous_copy_gpu(rin, s); ... }
```

`packed` is decided from `xin` only. If `xin` is packed but `rin` is not
row-contiguous (e.g. a transposed or sliced residual — `astype` in fast.cpp:174-177
preserves non-contiguous strides), `r = rin` is used directly and the kernel reads
residual rows at `row * axis_size` (`rms_norm.hip:107-121`, `:152-160`) → silent
misread. The sibling `RMSNormVJP::eval_gpu` in the same file checks row
contiguity of **both** inputs via `check_input` (`rms_norm.hip:573-589`); the
upstream Metal `rms_norm_residual` does the same. Latent for the qMLX flow (the
model passes a contiguous residual) but reachable through the public API. Fix:
apply the same contiguity check to `rin` (copy if not row-contiguous).

### MAJOR-3 — `kv_append` CPU fallback is broken

`fast.cpp:221-225`:

```cpp
auto start = Shape{(int)0, (int)0, (int)offset, (int)0};
auto stop  = Shape{(int)0, (int)0, (int)(offset + inputs[0].shape(2)), (int)0};
auto ok = slice_update(inputs[2], inputs[0], start, stop, strides, s);
```

Per `normalize_slice` (`mlx/ops.cpp:729-760`), for a positive stride,
`start=0, stop=0` normalizes to an **empty** (size-0) slice in that dimension
(`out_shape[i] = (0 - 0 + 1 - 1)/1 = 0`). So the B, H and D dims become 0-sized:
`upd_shape = [0, 0, T, 0]`, and `broadcast_to` of the `[B, H, 1, D]` update throws
for any H>1 or D>1 (i.e. effectively always). The fallback — used whenever
`KVAppend::use_fallback` (explicit CPU stream, `copy.hip:268-270`) — therefore
always throws instead of computing. Fix: `stop = {-1, -1, offset+T, -1}` (or the
full dim sizes). Note the Python-side `update_and_fetch_fused` T>1 branch
(`patches/k2-mlx-lm-model-patches.diff`, cache.py) uses proper `...`/`:` slices, so
only the C++ fallback lambda is affected.

### MAJOR-4 — `kv_append` missing dtype validation (silent corruption risk)

`fast.cpp:174-219` validates ndim, shapes, T=1 and offset range — but not dtype.
`KVAppend::eval_gpu` launches the kernel templated on `k.dtype()`
(`copy.hip:343-347`) while the destination is the **cache** buffer
(`ok.copy_shared_buffer(ck)`, `copy.hip:281-282`) and the output dtype is the
cache's dtype (`fast_primitives.h` `KVAppend::output_shapes`). If keys/values and
the cache have different dtypes (e.g. f32 keys into a bf16 cache), the kernel
writes the wrong element size into the cache buffer → silent memory corruption.
The qMLX flow always matches dtypes (cache allocated with `keys.dtype`), but the
public API should either check `keys.dtype() == cache_keys.dtype()` (and values vs
cache_values) or cast.

### MAJOR-5 — `kv_append` implicit layout assumption

`copy.hip:287-299`: the flattened `[B*H, D]` addressing
(`src = k + row * k.strides()[1] + col`, `dst = ck + offset*strides[2] +
row*strides[1] + col`) is only correct when `strides[0] == H * strides[1]` — i.e.
standard [B,H,1,D] / [B,H,T,D] layout — for **both** keys/values and the caches.
A non-contiguous view (transpose, batch slice) is silently misaddressed, while the
`slice_update` fallback handles arbitrary strides. No contiguity check in
`fast.cpp:174-219`. Latent for the qMLX flow (rope output and zero/concat-allocated
caches are contiguous) but reachable through the public API. Fix: require
`row_contiguous` (or the specific `stride[0] == H*stride[1]`) for all four inputs,
else fall back.

### MINOR-6 — unit coverage gaps

`scripts/verify-k2.py` covers shapes (1,4096)/(1,128)/(8,4096)/(1,256) — all
n_rows ≤ 8, i.e. only the **warp** path of `rms_norm_residual`; the block path
(n_rows > 64 or axis > 8192) is untested. `kv_append` is tested at D ∈ {256,128,96,
120} — all divisible by 8 (and by 4 for f32), i.e. only the **vector** path; the
scalar path (D % VEC ≠ 0, e.g. D=100 bf16) is untested. No test passes a
non-contiguous residual (MAJOR-2 case) or mismatched dtypes (MAJOR-4). The D=8
probe (`scripts/k2-d8-probe.py`) does cover the fixed uint4 case, in-place
semantics, offsets 0/1/3, and untouched rows — good.

### MINOR-7 — report arithmetic doesn't fully reconcile

`reports/k2-non-gemm-fusion.md:44-52`: per-kernel counts imply ~9.3 decode steps,
not 41: `rms_norm_warp 1344 / 36 layers = 37.3/step → 1344/37.3 = 9.33 steps`;
`kv_append 336 / 36 = 9.33 steps` (consistent), but `1344/41 = 32.8/step` and
`336/41 = 8.2/step`. The per-step dispatch total (43,838/41 = 1069.2) is validated
against an independent prior baseline (1070), and the delta is self-consistent
(−890/41 = −21.7), so the A/B conclusion stands — but the absolute per-kernel
counts don't reconcile with the stated step count, and the naive expectation for
KV fusion is −2 copies/layer × 36 layers = −72/step vs the measured −21.7/step;
the report should explain the gap (e.g. how the OFF slice-assign path dispatches).

### MINOR-8 — output dtype ignores residual

`fast.cpp:146`: `out_type = result_type(x, weight)` — the residual's dtype is not
considered. Self-consistent between the CPU fallback and the GPU path (both
produce `out_type`), but differs from the naive `add(x, residual)` + `rms_norm`
composition when the residual has a higher dtype (e.g. bf16 x + f32 residual:
fused → bf16, unfused → f32). Worth a doc comment on `fast.h:27-33`.

### NIT-9 — redundant `if constexpr`

`copy.hip:225-231`: both branches of `if constexpr (VEC == 8)` are byte-identical
(8×16-bit and 4×32-bit are both 16-byte `uint4`); VEC only affects addressing.
Collapse to a single block.

### NIT-10 — machine identifiers in reports

`reports/k2-non-gemm-fusion.md:3` (<guest-ip>),
`reports/opt-gauntlet-areaB-backend.md:6` (guest-a). Documentation, not code;
no secrets, tokens, or personal home paths anywhere in code, scripts, or patches
(grepped the full diff). Acceptable for bench records.

### NIT-11 — ROCm-only model patches

The patched mlx_lm files call `mx.rocm.*` unconditionally and the binding forces
`Device(gpu)` when no stream is given; the patches only apply to the ROCm venv, so
this is fine in practice — noted for completeness.

## Verification against review focus

1. **Fused rms_norm+residual correctness** — OK. In-place semantics safe: in the
   donatable branch `out` aliases `xin` (`rms_norm.hip:519-521`), but the
   sum-of-squares pass runs before the write pass and each element is read before
   being written (`rms_norm.hip:118-121`), so aliasing is safe. dtype/bf16-f32
   handling matches the existing kernel: f32 accumulation, weight multiply in T
   after truncation (`w * T(t * normalizer)`, same order as the pre-existing
   `rms_norm_row`, `rms_norm.hip:88-93`). Block reduction is correct for
   BLOCK_DIM=256: 8 warps, `shared_sum[9]`, second stage reads exactly the 8 warp
   sums (`(256+31)/32 = 8`), broadcast via `shared_sum[0]` after sync. Warp
   reduction is the standard shfl_xor tree; the `warp >= n_rows` guard
   (`rms_norm.hip:141-143`) handles n_rows not a multiple of 8 (grid =
   ceil(n_rows·32/256)). Boundary D values are guarded by `i + j < axis_size` in
   all four kernels; axis_size=1..8192 all covered. `1.0f/sqrtf` (not `rsqrtf`)
   matches the pre-existing precision choice. Only issues: MAJOR-2 (rin layout)
   and MINOR-8 (dtype contract).
2. **uint4 vector copy for all D/VEC** — OK, no residual tail bug. Final state:
   16-bit → VEC=8 (16 B), f32 → VEC=4 (16 B), both copy one `uint4`
   (`copy.hip:217-232`); D=8 f32 → 2 vectors/row = all 8 elements (was the
   pre-fix bug; verified by `scripts/k2-d8-probe.py` per the report, incl.
   offsets 0/1/3 and untouched-row checks); D % VEC ≠ 0 → scalar kernel covering
   every element (`copy.hip:235-250`); alignment holds under the standard-layout
   assumption (row stride multiple of VEC elements = 16 B; dst base
   `offset·stride[2]` multiple of 16 B). The redundant `if constexpr` is NIT-9.
3. **KV copy in-place aliasing** — OK for the flow. `ok/ov` share the cache
   buffers (`copy.hip:281-282`); src (new k/v from rope) and dst (cache at
   `offset`) are distinct buffers in the qMLX flow; per-element load-then-store is
   safe even if a caller aliases the same region; prior `keys_and_values()` views
   are sliced to `[:old_offset]` so they are unaffected by the append. The
   generic API's missing dtype/layout validation is MAJOR-4/5.
4. **Env-gating completeness** — PARTIAL. fast.cpp/fast.h/fast_primitives.h are
   pure additions (0 removed lines); the new APIs run only when called; the model
   patch (patched vs pristine venv) is the actual ON/OFF switch, and the bench
   isolates it correctly (both arms carry the same wheel). `MLX_COUNT_COPIES`
   (copy.hip:25) is pre-existing (present in base `c51a835`), not part of this
   PR. The one default-path change is MAJOR-1 (ungated `rms_norm` warp branch) —
   the OFF arm demonstrably runs it, so the A/B delta is clean, but "env unset →
   bit-identical" is not met.
5. **Build wiring / size gate / hygiene** — OK. No new source files: `copy.hip`
   and `rms_norm.hip` are already listed in
   `vendor/mlx-rocm-upstream/mlx/backend/rocm/CMakeLists.txt` (lines 145, 158);
   the python binding patch extends the existing `init_rocm`. Size gate:
   `SIZE GATE: OK (0 ratcheted)` (gate is .py-only; largest C++ file fast.cpp is
   1210 non-blank lines, out of gate scope — unchanged by this PR). vjp/jvp/vmap
   for both new primitives work via `Custom::fallback_` (fast.cpp:13-27);
   `is_equivalent`/`state()` implemented (fast.cpp:299-308, fast_primitives.h).
   No secrets or personal machine paths in code/scripts/patches (NIT-10 covers
   reports only).

## Evidence assessment

`reports/k2-non-gemm-fusion.md`: protocol is sound (flock + util<1% pre-bench,
fresh process per arm, same runner, 16-tok warmup, re-anchored baseline with the
confounded old baseline explicitly identified). The 9B "neutral" call is
defensible (+4.0%/+7.4% within run variance); the 0.6B +17-18% is consistent
across both runs; the dispatch protocol is validated against an independent prior
baseline (1069.2 vs 1070). The ADOPT verdict is supported, subject to MINOR-7
(reconcile per-kernel counts / step count) and the MAJOR fixes above (none of
which affect the benchmarked flow).

## Final verdict

FAIL — 0 BLOCKER, 5 MAJOR (MAJOR-1 ungated rms_norm default-path change;
MAJOR-2 residual layout unchecked; MAJOR-3 broken kv_append CPU fallback;
MAJOR-4 missing kv_append dtype check; MAJOR-5 implicit kv_append layout
assumption). Core fusion is correct and the evidence is good; all five are
small, mechanical fixes — re-review recommended after they land.
