# Phase 2 lever 3 — fused GDN decode-step kernel (continuation of t1026)

Date: 2026-10-30 (guest UTC). Worker: kernel-perf (an earlier run). Machine:
the build VM, RX 9070 XT / gfx1201, ROCm 10.1 guest userland. Model:
/models/qwen3-9b-mlx (4bit, 9B GDN-hybrid = 24 GDN + 8 attention layers).

## Result (one line)

The per-step GDN decode chain (4 launches: `gdn_prep` + `gated_delta_update`
+ `gdn_norm_gated` + conv) is fused into ONE `mx.fast.gdn_fused_step` launch
for the AR T=1 path. After fixing the `g`-dtype bug, the fused kernel is a
**lossless drop-in** (byte-identical greedy stream + bit-identical per-layer
GDN state, both contexts) and is **~4% faster** in decode (window tok/s
+4.8% at 14-tok, +4.1% at 29K; median ms/step −4.6% / −3.5%).

## 1. The bug (multi-step state explosion)

Single-step fused == chain was already bit-identical, but the 8-step probe
showed the fused path **exploded** (step 1 state 7.6e26, step 2 all nan/inf)
while the chain stayed stable (state 3.15 → 3.38). An egprobe (q=k=v=0,
β=1, h0=1.0, so the per-head state IS the applied decay) localised it:

- chain per-head eg ≈ 0.992–1.0 (tame, == bf16(g))
- fused per-head eg ≈ 1e23 (== low-16-bits-of-f32(g) pattern)

**Root cause:** `fast.cpp`'s `gated_delta_update` does
`g = astype(gates, out_dtype)` at the op boundary, where `out_dtype =
promote(q, k).dtype = bf16`. So the chain's r4d kernel receives `g` as a
**full bf16 value** (its u16 read = the bf16 bit pattern → gentle decay).
The fused kernel computed `s_g` in f32 and read its **low 16 bits** — which
are f32 mantissa bits, not a bf16 encoding — so `eg = exp(log(~1e16)) ~ 1e23`
saturated the recurrent state within a few steps. The "low-u16 quirk" was not
a quirk: r4d was designed to read `g` as bf16.

## 2. The fix

Round `s_g` to bf16 **before** the u16 read, using the exact rounding MLX's
`astype` uses (`_MLX_BFloat16` float ctor, RNE: `u += (u>>16 & 1) + 0x7FFF;
u >> 16`):

```
eg = __expf(fmaxf(__logf(gdnf_bf2f16(gdnf_f2bf_rne(s_g))), -80.0f));
```

Two f2bf variants are now both needed and both verified:
- `gdnf_f2bf` (round-half-away, `(bits + 0x8000) >> 16`) — kept for the **on
  store** (matches r4d's own f2bf in `r4d_gdn_wmma.h`, i.e. r4d parity).
- `gdnf_f2bf_rne` (RNE) — the new `s_g → bf16` cast (matches `astype`
  bit-for-bit). They differ only on exact-tie cases (low 16 bits = 0x8000).

All other phases (conv+silu, beta, g, qk norm incl. bf16 pre-rounded scale,
r4d state-update layout + arithmetic, norm_gated) were re-verified line-by-line
verbatim against the 75ca881 chain kernels.

## 3. Verification (all on the build VM, Qwen3-9B 4bit, greedy)

**Kernel-level (fast, same process):**
- egprobe: per-head `fused_eg == chain_eg` **exactly** for h0–h31 (all tame
  0.99–1.0; pre-fix fused was 1e23).
- bisect: single-step fused == chain bit-identical (out + state + dbg
  intermediates); fused 8-step now **stable and matches the chain at every
  step** (state 3.15 → 3.38, no inf/nan).

**Lossless gate (fresh process per arm, WARM pass discarded, skip=56,
both arms the SAME build `venv-fused` which contains both the chain and the
fused op — the only difference is `MLX_ROCM_GDN_FUSE`):**

| ctx | stream | per-layer GDN state | non-vacuity |
|-----|--------|--------------------|-------------|
| 14    | byte-identical (256 tok) | MATCH (24 layers, steps 0–7 + 256) | fused_calls=6144 |
| 29000 | byte-identical (256 tok) | MATCH (24 layers, steps 0–7 + 256) | fused_calls=6144 |

Two cross-process observations that shaped the gate design:
- The **chain is deterministic**: two fresh chain processes give identical
  state shas at every step (no baseline noise to hide behind).
- **Within one build, chain == fused is bit-identical**: a same-process
  chain-vs-fused state-diff at the final step (256) reports
  `max_abs = max_rel = 0.000000` across all 24 layers. The earlier
  cross-process state mismatch (15–18 layers at step 256) was a **build
  artifact** — comparing the `venv-m8` baseline (t1012 wheel) against the
  rebuilt `venv-fused` (75ca881 + patch); the *chain path itself* differs
  slightly between the two .so builds (codegen) and that drift accumulates by
  step 256. It is not a fused-kernel difference. The correct lossless gate
  compares within one build.

**A/B decode (graph decode OFF on both arms):**

| ctx | arm | window tok/s | all tok/s | median ms/step |
|-----|-----|-------------|-----------|----------------|
| 14    | UF (chain) | 77.22 | 79.44 | 12.946 |
| 14    | F  (fused) | 80.95 | 83.21 | 12.348 |
| 29000 | UF (chain) | 62.63 | 64.52 | 15.852 |
| 29000 | F  (fused) | 65.19 | 67.03 | 15.295 |

Delta: window tok/s **+4.8%** (14-tok) / **+4.1%** (29K); median ms/step
**−4.6%** / **−3.5%**. The fused kernel saves 3 of the 4 GDN sub-launches per
layer per step; because GDN is only part of each decode step (8 attention
layers + MLP remain separate), the end-to-end win is modest but consistent.

## 4. Build config notes (repro gotchas)

- `setup.py` reads **only** the `CMAKE_ARGS` env var (not
  `--config-settings=cmake.define.*` — that is a scikit-build convention and
  is inert here since the backend is setuptools).
- The ROCm prefix must be **`-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1`** (the
  prefix), **not** `.../lib`; otherwise `find_package(hip CONFIG)` fails
  (the config lives at `core-10.1/lib/cmake/hip/`).
- Omitting the ROCm args makes CMake compile the `no_gpu` backend, which
  re-defines `fast.cpp` symbols (`CrossEntropy::eval_gpu`, etc.) and fails the
  link with multiple-definition errors.
- Vendor cmake_args: `-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
  -DBLA_VENDOR=OpenBLAS` (from `vendor.json`); add the prefix above.

## 5. Files

- `patches/gdn_fused_step/gdn_fused_step.hip` — the fused kernel (single launch).
- `patches/gdn_fused_step/gdn_fused_step.patch` — 811-line patch vs the pinned
  75ca881 tree (applied at wheel-build time).
- `scripts/p2-model-patch.py` — venv `qwen3_5.py` F-branch dispatch +
  `GDN_FUSE_CALLS` counter (early-return before the chain when FUSE=1).
- `scripts/phase2-gdn-fuse-bench.py` — A/B + lossless-gate data capture.
- `scripts/phase2-gdn-fuse-runner.sh` — the gate + A/B protocol (same-build
  arms).
- `scripts/p2-gdn-egprobe.py`, `scripts/p2-gdn-bisect.py`,
  `scripts/p2-state-diff.py` — kernel-level verification probes.
- `tests/test_gdn_fuse.py` — hermetic static pin (RNE expression present,
  old low16-of-f32 helper absent).

## 6. CI

ruff format (71 files) + ruff check clean; hermetic `pytest -m "not gpu"`
21 passed + 1 skipped; size gate OK (2 ratcheted). Leak rule clean: the
branch diff against fa9fa30 plus every new file was grepped (tree content;
git author fields exempt — the mandated fleet identity) with:
`grep -RniE 'marzuki|100\.64\.[0-9]{1,3}\.[0-9]{1,3}|andryo|neonvm@|192\.168\.122\.[0-9]{1,3}|/mnt/models|\.junkyard\.sh|@ag\.mrzk\.io'`
(the operator-handle hostnames, the 100.64/10 tailnet IPs, the andryo
identity incl. its @-form and home dir, the neonvm@ mailbox prefix, the
192.168.122.x libvirt-bridge IPs, the guest models root, the *.junkyard.sh
scripts, and the ag.mrzk.io host domain) — 0 hits (the only grep matches in
the tree are the pattern names quoted in this very list; every other
new/changed file is clean).
