# RUN-verify-attention — GPU RUN card (verify2 branch, env-gated)

Single card, run top-to-bottom ON guest-a (RX 9070 XT / gfx1201, ROCm 10.1
guest userland). Companion: `patches/verify-attention-s4/`,
`patches/gdn-fused-step-tape/`, `patches/rocm-vendor-delta/`.

**Goal:** package the S=4 (K=3 verify) attention lever **default-off,
env-gated** (`MLX_ROCM_SDPA_S4VA=1`), and prove
(1) env-off is bit-identical to the no-graft wheel, (2) env-on is
bit-lossless vs the AR S=1 path, (3) per-pair wall A/B gain, on the
**full-stack wheel** (vendor delta + conv-row-fix + env-gated graft, with the
state-None model guard installed in the venv).

Wheel name: **`v4-fullstack`** (vendor tree = `rocm-vendor-delta` +
`conv-row-fix` + `sdpa_s4_env`).

## Step 0 — worktree chain (the reproducibility gate)

```
git clone --filter=blob:none https://github.com/lifeofpiboy/mlx v
cd v && git checkout 75ca881b2e82f7b4069c8564a203792aa16d8a29
git apply --check ../../<repo>/patches/rocm-vendor-delta/delta.patch          # silent
git apply ../../<repo>/patches/rocm-vendor-delta/delta.patch
cp -r ../../<repo>/patches/rocm-vendor-delta/new_files/. mlx/backend/rocm/
git apply --check ../../<repo>/patches/gdn-fused-step-tape/conv-row-fix.diff  # silent
git apply ../../<repo>/patches/gdn-fused-step-tape/conv-row-fix.diff
git apply --check ../../<repo>/patches/verify-attention-s4/sdpa_s4_env.patch   # silent
git apply ../../<repo>/patches/verify-attention-s4/sdpa_s4_env.patch
grep -c "MLX_ROCM_SDPA_S4VA" mlx/backend/rocm/scaled_dot_product_attention.hip # expect 3
diff -q mlx/backend/rocm/scaled_dot_product_attention.hip \
  ../../<repo>/patches/verify-attention-s4/scaled_dot_product_attention.hip      # identical
```

(Verified on this branch: all `--check`s pass on a fresh 75ca881 worktree.)

## Step 1 — G1 compile bar

```
hipcc --offload-arch=gfx1201 -fsyntax-only -I . -I mlx \
  mlx/backend/rocm/scaled_dot_product_attention.hip   # 0 errors
```

## Step 2 — wheel build + install

Build as in the vendor-delta run (incremental `pip wheel --no-deps`, 16 cores),
install into the verify venv (venv has no pip: unzip the wheel over
`site-packages` after removing the old `mlx*` entries). Install the
state-None model guard (`patches/gdn-fused-step-tape/state-None-guard.diff`)
in the venv's `mlx_lm/models/qwen3_5.py`; `python -m py_compile` must pass.

## Step 3 — G0 job gates (before the battery)

- **G0.1 (Job 2, crash guard):** fresh cache, 4-token first forward
  (S=4, state=None → tape branch, guarded → default chain): no pybind error.
  Fresh cache, 1-token first forward (S=1, state=None → F branch): no error.
- **G0.2 (Job 1, rows 0/1):** rowprobe — prefill 16, then 1×S=4 tape vs
  4×S=1 fused: per-row conv md5, **all four rows identical** (pre-fix: rows
  0/1 differ in k+v channels). Plus full-state parity (conv + recurrent
  md5-identical after 4 tokens).
- **G0.3 (control):** GDN-ops-OFF, S=4 default chain vs 4×S=1 default chain
  on the no-graft wheel: record the logit diff (expected > ulp — the
  prefill-attention baseline the graft fixes; documents that G0.2's parity
  isolates the GDN path).

## Step 4 — G2 kernel gate (env ON): per-row bit-exact vs AR S=1

`test_s4va.py` (t1274 method): direct SDPA, B=1, H=16, qL=4, D=128, bf16,
`mask="causal"`, kL ∈ {33, 67, 129, 257, 512}, 16 random trials per kL:

- **G2.1 (the gate, env ON):** per-row bit-exact vs the AR S=1 no-mask row
  over the same visible prefix (n=0), all 5 kL × 16 trials × 4 rows.
- **G2.2 (regression, env ON vs the no-graft wheel):** qL=1 no-mask and
  qL=1 causal rows bit-exact (n=0); qL=2 causal row 1 bit-exact (n=0).
- **G2.3 (sanity, env ON):** qL=4 causal vs fp32 numpy reference
  `max|Δ| ≤ 2e-2` (bf16) — guards a masking bug (row seeing future keys).

## Step 5 — the 4-arm battery (400 toks, 128 verifies, fresh AR reference)

One fresh AR run (300+ tok, fixed seed) is the reference. Arms (all
`FUSE=1`, QMV-ROWS off):

| arm | GDN op | MLX_ROCM_SDPA_S4VA |
|---|---|---|
| A0 | fused (S=1 decode + S=4 verify via prefill kernel) | unset |
| A1 | fused | `=1` (vector S=4) |
| A2 | tape | unset |
| A3 | tape | `=1` (vector S=4) |

- **G4.1 (lossless gate, the one that must pass):** A1 and A3 K=3 streams
  bit-match the AR stream (FIRST20 prefix + ALLTOKS prefix).
- **G4.2 (default-off gate):** A0/A2 behavior vs the no-graft baseline wheel
  (v2c): same known pre-graft attention signature (G0.3); A1 vs A0 and A3 vs
  A2 show the graft flipping the verify path (verify wall drops, stream
  becomes bit-lossless).
- **G4.3 (accept/throughput):** per-arm accept rate and tok/s recorded;
  A1/A3 expected ≥ A0/A2 (per-pair wall gain, t1274 G5 method).

## Step 6 — G5 AR regression

300-token fused-AR on `v4-fullstack`: tok/s within ±2 of the no-graft wheel
baseline; AR stream md5 (fixed seed) recorded (the graft must not touch S=1
rows — G2.2 n=0 is the proof).

## Verdict

- **LOSSLESS PASS** iff G4.1 (A1+A3) + G2.1–G2.3 all pass.
- **DEFAULT-OFF PASS** iff G2.2 (n=0) + G4.2 signature match.
- **LEVER CONFIRMED** iff A1/A3 verify-wall ≤ A0/A2 (record exact A/B).

## Rollback

`git apply -R sdpa_s4_env.patch` (or reinstall the no-graft wheel). With the
env unset there is no state on disk outside the wheel; unset the env to
revert behavior of the grafted wheel.
