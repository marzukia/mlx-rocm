# REPORT-refwd — the MTP correction re-forward (refwd) speedup (CPU-side design + default-off toggle)

**Scope.** Bring the K=3 MTP (DRAFT=3) decode pair (verify+refwd) below 41.5 ms,
i.e. get the refwd ≤ 7 ms — by a CPU-side (no-GPU-bench) design that ships as a
default-off env toggle on `worker/refwd-fold` off `da9eeca`, validated by a
hermetic gate test and a canonical GPU battery the next round runs.

**Baseline (t1307 / t1334, the canonical method, K=3 / DRAFT=3 / M=4 verify):**

| config | tok/s | verify | refwd | pair | n_refwd |
|---|---|---|---|---|---|
| B0 — base, no rows (the M=1 QMV **wide default**) | 69.48 | 22.13 ms | ~9.9 ms / 12.6 ms fwd | ~41.3 | 100 |
| R1 — rows ON (current best; the `worker/rows-statistical` wheel) | 75.00 | 17.41 ms | 9.31 ms | 41.35 | 102 |

The refwd (the single correction re-forward on a rejected verify) is the lever
this task targets. The verify is already rows-bounded at 17.41 ms (the rows
work, REPORT-rows-bitexact) and is NOT the bottleneck; the pair floor is set by
verify+refwd.

---

## 1. What the refwd is, and why it is what it is

The K=3 spec-decode loop (per verify, `qwen3_5.py:467`):
1. **verify** (M=4): the 3 draft tokens + 1 carry, one QMM batch, per-layer
   argmax accept/reject against the draft.
2. **refwd** (M=1): on a rejected verify, the correction token
   `argmax(l_{keep-1})` from the verify's final output is re-forwarded through
   the full 9B model (one token) to produce the next draft's starting state.
   n_refwd ≈ 100/128 verifies (the acceptance tail).

The refwd is a **genuine full M=1 9B forward** — 249 quantized GEMMs
(32 layers × 3 MLP + 24 GDN × 5 + 8 attention × 4 + 1 lm_head = 96+120+32+1),
at M=1. The M=1 QMV default is the scalar-FMA wide kernel
(`qmv_wide_mr_kernel`, M≤4), which REPORT-gemms-m4 found is **FMA-throughput
bound at ~175 GB/s effective (~27 % of the 640 GB/s peak)** on gfx1201 — the
same FMA wall that bounds the M=4 default.

The pair floor: with rows ON the verify is 17.41 ms and the refwd ~9.3–12.6 ms,
so the pair sits ~41 ms. To get the pair < 41.5 ms with the verify fixed at
17.41 ms requires the refwd to drop to ≤ ~7 ms. **The refwd is the only
remaining lever** (the verify is rows-bounded; the GDN step is the fused kernel,
already benched).

---

## 2. The three options

### (a) Fold the refwd into the verify batch — **infeasible (data-dependent)**

The correction token is `argmax(l_{keep-1})`, i.e. it is produced **by** the
verify's final output. It cannot be known before the verify forward completes, so
it cannot be appended to the verify's M=4 batch up front. "Folding" would mean
running the verify at M=5 (the 4 tokens + a speculative 5th), but the 5th token
is undefined until the 4th's argmax is known — a chicken-and-egg within one
forward. The only way to fold is a **method change** (a 2-pass verify that
speculates the correction), which changes the accept/reject semantics and the
losslessness contract. **Out of scope for a CPU-side, method-preserving change.**

### (b) Cache the verify's per-layer state and reuse it in the refwd — **infeasible (sequential recurrence)**

The natural idea: the refwd re-computes the forward for the new correction
token; if we had cached the verify's per-layer intermediate state, we could
incrementally update it. But:

1. The **GDN state recurrence is sequential**: `h_t = a_t·h_{t-1} + b_t·(v_t·x_t)`.
   A new token's `h` depends on the **full** GDN chain from the layer's input;
   there is no per-layer checkpoint that lets you skip the chain. Replaying the
   correction token requires re-running every GDN layer.
2. The **MTP module does not preserve per-layer activations** — it only keeps the
   state buffers (`conv_state`, `ssm_state`, the KV cache) via the tape commit
   (the O(1) rebind, `gdn_tape_commit.py`). The verify's per-layer hidden
   tensors are discarded after the forward.
3. The **KV cache** IS preserved (the tape commit trims it in place), but the KV
   for the new token must be **computed** by the new token's attention forward —
   it is not recoverable from the verify's KV.

So there is no reusable per-layer intermediate; the refwd is a genuine
recomputation. Caching would require a method change (persisting per-layer
activations across the verify→refwd boundary). **Infeasible without a method
change.**

### (c) Tune the M=1 QMV kernel path — **the only feasible CPU-side lever (chosen)**

The M=1 QMV currently dispatches to the scalar-FMA wide default. There is an
**existing, already-compiled, already-correct** alternative in the same file: the
**int8-activation `v_dot4` "LX" path** (`LAUNCH_LX`, gated by `MLX_QMV_LX`),
which replaces the f32-FMA wide loop with in-block int8 quant +
`__builtin_amdgcn_sudot4` + LDS x-staging. That is a **different MAC path**
(integer dot-product rather than scalar FMA) that may have higher throughput for
the M=1 case on gfx1201 — exactly the FMA-bound case the wide default suffers.

**Option (c) is implemented as a default-off toggle** that re-routes **only the
M==1 QMMs** (the decode/refwd) to the existing LX path, so the next GPU round
can A/B the M=1 scalar-FMA default vs the integer-dot path. It is the only
option that (i) is method-preserving, (ii) is CPU-side (no GPU bench to design),
and (iii) ships with zero default-on risk (default-off, gated by A/B).

---

## 3. The implementation (option c)

**The change is a 2-line, default-off gate in `qmm.hip`** (the M=1 QMV dispatch,
`QuantizedMatmul::eval_gpu`):

```cpp
static const char* lx_env = std::getenv("MLX_QMV_LX");
// REFWD-M1 (default off): re-route ONLY the M==1 QMMs to the existing LX path.
static const bool refwd_m1 = (std::getenv("MLX_ROCM_REFWD_M1") != nullptr);
const bool lx_engage = (lx_env != nullptr) || (refwd_m1 && M == 1);
if (lx_engage && /* ...the original LX shape gate... */) {
    // guarded cols read (lx_env may be null when engaged via refwd_m1):
    int lx_cols = 32;
    if (lx_env != nullptr && lx_env[0] == '1' && lx_env[1] == '6') lx_cols = 16;
    ...  // the pre-existing LAUNCH_LX dispatch (unchanged)
}
```

Properties:
- **Default-off, zero behavior change:** with `MLX_ROCM_REFWD_M1` unset,
  `lx_engage == (lx_env != nullptr)` — the original gate — and the cols read is
  backward-compatible (null → 32, `=16` → 16). The default build path is
  byte-identical in behavior.
- **M==1-only:** the toggle re-routes only `M == 1`, so the M=4 verify (the rows
  path / the wide-MR M≤4 path) is untouched.
- **Reuses an existing kernel:** the LX path is the pre-existing `MLX_QMV_LX`
  kernel (compiled, tested via `test_qmv_mr2_lx_instantiation.py`). The toggle
  only adds a gate line and a null-guard on the cols read — no new kernel.

**Gate scope (all 249 refwd QMM shapes):** the original LX shape gate requires
`K%32==0`, `K%group==0`, `group ∈ {32,64,128}`, 4/8-bit affine, bf16/f16/f32
x, non-batched, transpose, `host_warp==32`. For the 9B target (4-bit group 64,
bf16) **every refwd shape satisfies it** (verified by
`tests/test_refwd_m1_gate.py` over all 249 shapes: K ∈ {4096, 12288, 4096, 8192,
32768, 512, 4096, 12288, 8192, 248320} — all %32==0 and %64==0). So the toggle
re-routes the **entire** M=1 refwd QMV set, uniformly.

---

## 4. Analysis: expected effect and risk to losslessness

**Expected effect (to-be-measured by the A/B — this is the honest framing).**
The M=1 QMV is memory-bound: it reads the 9B 4-bit weight matrix once and does a
vector-matrix product. The theoretical floor is weight-bandwidth-bound
(~4.5 GB / 640 GB/s ≈ 7.0 ms); the current wide default runs at ~175 GB/s
effective. The int8-dot4 LX path is a different MAC path; whether it is faster
for M=1 on gfx1201 is **not known a priori** — the A/B measures it. The lever
addresses only the **QMV fraction** of the refwd; the GDN layers run the fused
decode-step kernel (~230 GB/s, REPORT-gemms-m4) and the attention runs the SDPA
S4VA — those are unaffected by this toggle. So the achievable refwd reduction is
bounded by the QMV fraction of the 249-GEMM set.

**Honest expectation:** the A/B will show one of (i) the int8 path is faster for
M=1 → the refwd drops (possibly toward 7 ms), or (ii) it is not faster → the
toggle stays off (zero cost, the M=1 wide default is kept). **Either way there is
no downside to shipping the default-off toggle.**

**Risk to losslessness (contained).** The int8-dot4 path is **statistical, not
bit-exact** (it quantizes x to int8 and uses `__builtin_amdgcn_sudot4`; the wide
default uses f32 FMA). So R2 (the toggle ON) changes the token distribution — it
is NOT a lossless-serve candidate. This is contained by:
1. The toggle is **default-off** — the serve path (B0 / the R1 rows config) is
   the unchanged M=1 wide default.
2. The A/B gate (benches) gates R2 on *runs + refwd faster + within the ULP-jitter
   class*, **not** on bit-exactness — and R2 is an A/B arm only, never shipped.

If the A/B shows R2 is both faster AND lossless within tolerance, a follow-up
decision (a new PR) would be required to consider it for the serve path. This
PR ships only the default-off toggle + the bench.

---

## 5. Validation

**CPU (this PR, hermetic, no GPU):**
- `tests/test_refwd_m1_gate.py` — mirrors the dispatch gate, enumerates all 249
  refwd GEMM shapes, and pins: default-off is a no-op; M=1 re-routes to LX (all
  249 shapes); M=2/M=4 stay on wide (the verify is protected); the cols guard is
  backward-compatible. **7 tests pass; 28 hermetic tests pass; size gate OK; ruff
  clean.**
- Static diff review: the change is a single gate line + a null-guard on the
  cols read; the default path reduces to the original gate.

**GPU (next round, `bench/run-refwd.sh` — env-only arms, canonical md5-pinned
bench, 2 runs/arm, flock per arm):**
- **B0** (base, the M=1 QMV wide default): gate — reproduce the 69.48 control
  (±3 %), n_refwd ~100, TAPE 128/128 ok.
- **R2** (B0 + `MLX_ROCM_REFWD_M1=1`): gate — runs (no collapse), refwd < B0
  refwd (the lever helps), verify == B0 verify (±2 %, M==1-only), n_refwd in
  [95,105], within the ULP-jitter class (not bit-exact).
- **R1** (the current rows config, 75.00 reference): requires the rows kernel in
  the wheel — i.e. a wheel built from the **combined tree**
  (`worker/rows-statistical` + this branch). On this branch's wheel the
  `MLX_ROCM_QMV_ROWS` env is a no-op, so R1 degenerates to B0; the real 75.00
  R1 is benched on the combined tree.

**Build prerequisite (next round):** build + install the wheel from this branch's
vendor tree (`MLX_VENDOR_DIR=<this branch's vendor tree> bash scripts/build-wheel.sh`).
This branch's vendor tree has **no rows kernel** (rows is on
`worker/rows-statistical`), so the pure A/B on this wheel is **B0 vs R2**
(the M=1 scalar-FMA default vs the int8-dot4 re-route, no rows confound).

---

## 6. Summary

- **The refwd is the only remaining lever** (the verify is rows-bounded at
  17.41 ms; the GDN step is the benched fused kernel).
- **Options (a) and (b) are infeasible** without a method change (the correction
  token is data-dependent; the GDN recurrence is sequential and no per-layer
  activation is preserved).
- **Option (c) is the only feasible CPU-side lever**: re-route the M=1 QMMs to
  the existing, already-compiled int8-dot4 LX path, via a **default-off toggle**
  (`MLX_ROCM_REFWD_M1`) that is M==1-only and reuses an existing kernel.
- **Zero default-on risk**: default-off; if the A/B shows the int8 path is not
  faster or breaks losslessness, the toggle simply stays off.
- **The expected refwd reduction is to-be-measured** (A/B), bounded by the QMV
  fraction of the 249-GEMM refwd; the GDN/attention portions are unaffected.
