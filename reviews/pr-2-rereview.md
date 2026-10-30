# K2 PR #2 re-review (non-GEMM fusion) — adversarial, post-fix

Reviewer: <agent-gh> (fresh session, read-only). Submitter of record: submitter.
Scope: `main..19e76d9` (task named head `d812947`; the branch advanced to `19e76d9`
"k2: ruff import-sort on identity probe (CI I001)" before this review — the delta is
ruff-only on `scripts/k2-identity-probe.py`, no product/test semantics change).
Method: full source read of the fix round (`4c45978`), line-by-line diff
`main..19e76d9` (rms_norm.hip, copy.hip, fast.cpp, fast.h, scripts/), MLX flag
semantics cross-check against `array.h` / `common.cpp`, static verification of every
fix, adversarial pass for new bugs (edge shapes, CPU/GPU split, dtype/stride).

## Verdict: FAIL — 0 BLOCKER / 2 MAJOR / 4 MINOR

All five original MAJORs are addressed; four are correctly and fully fixed
(MAJOR-1, -2, -3, -4). MAJOR-5 is only partially fixed (gate too weak — see
MAJOR-A below), and the fix-verification suite crashes on its own fix test cases
(MAJOR-B). Both remaining MAJORs are small, mechanical fixes; re-review after.

## Per-original-MAJOR verification

### MAJOR-1 (ungated default-path change) — FIXED

- The ungated `warp` branch is removed from `RMSNorm::eval_gpu`. The fixed function
  is **code-identical to the base** (`c51a835`): I extracted both functions and
  diffed them — identical except a comment. The decode fast path now exists only in
  the new `RMSNormResidual` op, which is reachable exclusively via the patched
  mlx_lm model files (env-gated at the venv level: patched venv-k2 vs pristine venv,
  per the A/B protocol in `reports/k2-non-gemm-fusion.md`).
- This is not "a gate that hides the path": the pre-existing op is restored to
  byte-identical behavior, and the fast path is genuinely new surface.
- Evidence: identity probe 24/24 `refclose=True` on the fixed wheel (8 shapes ×
  bf16/f16/f32, `scripts/k2-identity-probe.py`); the rebuilt wheel's `.so` is
  bit-identical to the pre-burn fixed build, so the fix round is what was tested.
  The baseline-wheel sha diff is still pending on the guest — but bit-identity is
  independently established by code identity: the vendor diff `main..19e76d9` is
  purely additive (0 deletions in `rms_norm.hip`/`copy.hip`/`fast.cpp`), so the
  compiled `rms_norm` block/strided kernels are unchanged. The pending sha diff is
  confirmatory, not load-bearing.

### MAJOR-2 (residual layout unchecked) — FIXED

- `packed_check` is now applied to **both** `xin` and `rin`; if either fails, both
  are `contiguous_copy_gpu`'d and `encoder.add_temporary`'d (always —
  `contiguous_copy_gpu` allocates fresh, so the `data_shared_ptr !=` check is a
  harmless no-op), and `out.copy_shared_buffer(x)` aliases the fresh copy, keeping
  the in-place kernel safe (per-element read-before-write; `r` is read-only).
- The packed branch is unchanged and correct; the non-packed branch reuses the
  base's proven `rms_norm_strided_kernel` path.
- Residual weakness: the same flag-class issue as MAJOR-5 (see MINOR-1).

### MAJOR-3 (broken CPU fallback) — FIXED

- `stop` now uses full dims: `stop = {B, H, offset + 1, D}` (was `{0,0,0,0}` →
  always threw on non-empty updates). `T = 1` is enforced on `keys`/`values`, so
  the update region `[0:B, 0:H, offset:offset+1, 0:D]` matches the update shapes
  exactly; `offset` is range-checked. The in-place fast path is unchanged.
- Covered by the suite (this test runs *before* the crash point of MAJOR-B).

### MAJOR-4 (missing dtype check) — FIXED

- Clean `invalid_argument` for `keys.dtype() != cache_keys.dtype()` and
  `values.dtype() != cache_values.dtype()` (the original finding's `keys` vs
  `values` comparison was itself a typo — the real contract is per-operand).
- Covered by the suite (runs before the crash point).

### MAJOR-5 (implicit standard-layout assumption) — PARTIALLY FIXED (→ MAJOR-A)

- A gate was added: `standard_layout` requires `flags().contiguous` on all four
  inputs, else the `slice_update` fallback. But MLX's `contiguous` flag means
  **no gaps** — `prod(shape[i] for strides[i] > 0) == data_size` (`array.h` doc;
  computed in `AsStrided::eval`, `common.cpp`) — **not** row-major.
- The kernel's actual requirements are `strides[0] == H*strides[1]` (or B==1) and
  `strides[3] == 1` (or D==1) — i.e. `row_contiguous` (whose documented definition
  implies exactly those, or the original finding's explicit stride check).
- The original review's recommended fix was `row_contiguous`; the implemented fix
  used the weaker flag, leaving the canonical counterexample open (MAJOR-A).

## New findings (adversarial pass)

### MAJOR-A: MAJOR-5 gate too weak — gap-free permuted views pass and misaddress

The `contiguous` (no-gaps) gate admits views that are not standard layout.
Concrete counterexample: a square `[Tmax=D]` cache with the last two axes
transposed — shape `[B,H,D,D]`, strides `[H·D², D², 1, D]`. No gaps
(`no_bsx_size = B·H·D² = data_size` → `contiguous == true`), shape matches
`[B,H,T',D']` with `T'=D, D'=D`, so it passes the gate. The fast kernel uses raw
element offsets for the last dim (assumes last-dim stride 1; actual is `D`) and
`strides[2]` as the row stride (actual row stride is 1) — it writes the D new
values into the first D positions of the row region, corrupting the start of
subsequent rows. Silent data corruption on the public API (the qMLX flow is
unaffected: all four inputs there are standard contiguous, so the bench results
stand).

Fix (one line): require `flags().row_contiguous` (or explicit
`strides[0]==H*strides[1] && strides[3]==1`, with the B==1/D==1 degenerate
allowances) for all four inputs, else fall back. Note `row_contiguous` is already
maintained on the flag set (recomputed in `AsStrided::eval`), so this is free.

### MAJOR-B: verify-k2.py crashes on its own fix test cases

Both "non-contiguous" test setups are actually **contiguous** per MLX's flag
definition, so the script dies with `AssertionError` before the MAJOR-2 test and
the final summary:

1. MAJOR-5 test: `base (2,1,8) → transpose(1,0,2) → expand_dims` → shape
   `[1,2,1,8]`, strides `[8,8,1,1]`. `Transpose::eval` inherits
   `flags.contiguous` from the input (a fresh allocation → true) and never
   clears it; `ExpandDims::eval` passes flags through. `no_bsx_size = 16 =
   data_size` → `contiguous == true` → `assert not keys.flags.contiguous` raises.
   Additionally, with **B=1** the view is correctly addressed by the fast kernel
   (`strides[0] != H*strides[1]` is harmless when B=1), so even without the
   crash the test would not exercise the fallback it claims to.
2. MAJOR-2 test: `base_r (8,2) → transpose(1,0)` → shape `[2,8]`, strides
   `[1,2]` — likewise no gaps → `contiguous == true` →
   `assert not r.flags.contiguous` raises (never reached; the script already
   died at test 1).

Consequence: the designated fix-verification suite exits non-zero with a
traceback (not a failure list), and the MAJOR-2/MAJOR-5 unit checks never run —
the "verify" claim currently rests on the identity probe alone.

Fix: MAJOR-5 test — use **B=2** (e.g. `base (2,2,8)`) and assert
`not keys.flags.row_contiguous` (a B=2 B/H-permuted view is no-gap but
not row-contiguous, so it both survives the assert and exercises the fallback
once MAJOR-A is fixed). MAJOR-2 test — assert `not r.flags.row_contiguous`
(a transposed 2-D is row-contiguous=false).

### MINOR-1: same weak-flag class in RMSNormResidual's packed_check

`packed_check` (used for both `xin` and `rin`) requires `contiguous` (no gaps) +
last stride 1 + second-to-last stride == axis (or 0). For ndim ≤ 3 this is
sufficient (no-gaps then forces the leading strides — verified by construction),
so the 3-D model flow `[B,L,D]` is provably safe. For ndim ≥ 4, a leading-dim
permuted view slips through: e.g. shape `[2,2,1,8]`, strides `[8,16,8,1]`
(B1/B2 permute, T=1) — no gaps, `strides[2] = 8 = axis`, last stride 1 → packed,
but row bases are `b1*8 + b2*16`, not `(b1*2+b2)*8` → fast kernel misaddresses.
Exotic (public API only; the same logic pre-exists in the base `RMSNorm` op).
Fix together with MAJOR-A: use `row_contiguous` in `packed_check`.

### MINOR-2: MINOR-8 (carried) — output-dtype contract still undocumented

`fast.h` now documents `rms_norm_residual` in one line
("K2: fused residual add + RMSNorm: w * rmsnorm(x + residual)"), but the
original MINOR-8 asked for the **dtype contract**: output dtype =
`result_type(x, weight)`, with the residual astype'd to it. Still unstated.

### MINOR-3: kv_append "in-place" doc vs fallback path

The `fast.h` doc says the append is in-place. The CPU/fallback path (CPU stream,
or non-standard-layout inputs) returns a **fresh buffer**
(`slice_update` result) — values are correct, but the cache buffer is not
modified in place. Doc inaccuracy on the fallback path; acceptable behavior,
wording should say "in-place on the GPU fast path".

### MINOR-4: dead code — rms_norm_warp_kernel

After the MAJOR-1 fix, `rocm::rms_norm_warp_kernel` is defined but never launched
(the pre-existing op uses block/strided kernels; the residual op uses
`rms_norm_residual_warp_kernel`). Harmless; delete or keep with a comment.

## Resolved / no-change items from the original review

- **NIT-9 (redundant if constexpr)**: fixed — single uint4 block with a comment
  ("always 4; the D=8 f32 case is the same 4×f32 = 16B, no separate path").
- **MINOR-7 (report arithmetic)**: addressed — the report now carries a
  reconciliation note: per-kernel counts cover a ~9.33-decode-step kernel-trace
  window (1344/36, 336/36, 672/36/2 all = 9.33), separate from the 41-step
  dispatch total (43,838/41 = 1,069.2, matching the prior 1,070 baseline); the
  naive −72/step expectation overstates the measured −21.7/step because the OFF
  slice-assign dispatch pattern differs; the A/B conclusion rests on the
  validated dispatch total + tok/s A/B. Reasonable.
- **NIT-10 (machine identifiers in report)**: unchanged, acceptable.
- **NIT-11 (ROCm-only model patch)**: unchanged, acceptable.
- **CI lint at d812947** (found this review): `scripts/k2-identity-probe.py`
  failed `ruff check` (I001 import sort) and `ruff format --check` — fixed by
  `19e76d9` (the branch tip). At `19e76d9` I verified locally: `ruff check .`
  clean, `ruff format --check .` clean, `size_ratchet.py --base HEAD` OK
  (0 ratcheted), all new scripts `py_compile` clean.

## Evidence assessment

- Rebuilt wheel `.so` bit-identical to the pre-burn fixed build → the fix round
  is what the guest evidence tested. (Carried, accepted.)
- Identity probe 24/24 `refclose=True` on the fixed wheel → fixed `rms_norm`
  correct across all 8 shapes × 3 dtypes. (Accepted.)
- Baseline-wheel sha diff: **pending on the guest**. Not load-bearing: bit-
  identity is established by code identity (purely-additive vendor diff;
  `RMSNorm::eval_gpu` code-identical to base). Recommend completing it for the
  record, but it does not gate the verdict.
- uint4/D=8 fix: covered by `k2-d8-probe.py` (all 8 elements, f32/bf16/f16,
  offsets 0/1/3, untouched-row isolation) per the original review + report.

## Suggested fix list (all mechanical)

1. `fast.cpp` `kv_append`: replace `flags().contiguous` with
   `flags().row_contiguous` in `standard_layout` (MAJOR-A).
2. `rms_norm.hip` `RMSNormResidual::eval_gpu` `packed_check`: same
   `row_contiguous` swap (MINOR-1).
3. `verify-k2.py`: MAJOR-5 test → B=2 + assert `not keys.flags.row_contiguous`;
   MAJOR-2 test → assert `not r.flags.row_contiguous` (MAJOR-B).
4. `fast.h`: document the `rms_norm_residual` dtype contract (MINOR-2) and the
   in-place-on-fast-path wording for `kv_append` (MINOR-3); optionally delete
   `rms_norm_warp_kernel` (MINOR-4).
5. Re-run `verify-k2.py` + `k2-identity-probe.py` on the guest; complete the
   baseline sha diff.
