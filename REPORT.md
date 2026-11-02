# REPORT — GPU verification round (verify2 branch)

Branch `worker/gpu-verify2`, base `ec63453`. All GPU work on guest-a
(RX 9070 XT / gfx1201, ROCm 10.1 guest userland). Wheel chain: stock
`75ca881b2e82f7b4069c8564a203792aa16d8a29` (the `lifeofpiboy/mlx` fork; see
`patches/rocm-vendor-delta/README.md` for why the repo's committed vendor
mirror is not the build base) + 81-build vendor delta + conv-row-fix +
env-gated S=4 attention graft. Model: Qwen3-9B-MLX (fp16 + MTP sidecar
pinned, 4bit/fixed parked).

**Verdict: ALL GATES PASS.** Conv-tape rows 0/1 fixed and bit-exact; fresh
`state=None` crash guarded; S=4 attention lever packaged default-off
(`MLX_ROCM_SDPA_S4VA=1`), bit-lossless when on, stock-identical when off,
+5.4 to +10.4 tok/s on the K=3 verify loop.

(The previous top-level report, the m2 draft-collapse RCA, is preserved as
`REPORT-m2-rca.md`.)

## Job 1 — conv-tape rows 0/1 bit-exactness

**Root cause.** `gdnt_cs_tap()` in `gdn_fused_step_tape.hip` took one channel
argument for both the conv_state and raw index. The q branch is correct
(conv channel == raw channel), but the k/v branches passed the raw-side
offset (`kch`, `vch`) into the conv_state, which is laid out in conv
channels (v block offset `2*key_dim`). Only rows 0/1 read the mis-indexed
k/v conv blocks on cs taps → rows 0/1 diverged in exactly the k+v channels
(2048–8191 of the 8192-wide conv state), rows 2/3 bit-exact. See
`patches/gdn-fused-step-tape/README.md`.

**Fix.** Separate `raw_ch` / `cs_ch`; call sites pass `(kch, ck)`,
`(vch, cv)`, `(qch, cq)`. `patches/gdn-fused-step-tape/conv-row-fix.diff`.

**Evidence (post-fix wheel, v2c).**

- rowprobe (prefill 16 → 1×S=4 tape vs 4×S=1 fused): per-row conv md5,
  **all four rows identical**: `6d60cddc / bc357dbd / f6231042 / f5c0a6fd`.
  Pre-fix (m1005): rows 0/1 differ, rows 2/3 identical.
- Full-state parity (4 tokens after prefill 16): conv state **and**
  recurrent state md5-identical between the two arms.
- Control (GDN ops OFF, S=4 default chain vs 4×S=1 default chain): logit
  diff max 6.94 / mean 0.67 — the prefill-attention baseline the graft
  fixes; the GDN-ops ON case (max 4.88) is the same signature, i.e. the
  end-to-end S=4-vs-4×S=1 logit mismatch on the *no-graft* wheel is
  attention-provenanced, not GDN. With the graft on, the K=3 stream
  bit-matches AR (Job 5, G4.1).

## Job 2 — fresh-cache `state=None` crash

Both op pybinds (`gdn_fused_step`, `gdn_fused_step_tape`) take `state` as a
non-optional array; the first S≤4 forward of a fresh cache (`cache[1]` is
`None`) crashed in the pybind conversion. The guard
(`patches/gdn-fused-step-tape/state-None-guard.diff`, applied to the venv's
`mlx_lm/models/qwen3_5.py`, `py_compile` OK) computes
`_state = cache[1] if cache is not None else None` once and adds
`and _state is not None` to **both** the tape and the F branch conditions —
a fresh-cache S≤4 forward falls to the default 4-op chain, which handles a
`None` state.

**Evidence.** GATE2A: fresh cache, 4-token first forward (S=4, state=None →
tape branch, guarded): no error. GATE2B: fresh cache, 1-token first forward
(S=1, state=None → F branch, guarded): no error. Pre-fix: both raised in the
pybind conversion (m1005 repro, `freshcrash`).

## Job 3 — S=4 attention graft, packaged default-off / env-gated

Ported from the t1274 branch and re-worked (see
`patches/verify-attention-s4/README.md`):

- Hunk 1 (m8 single-kernel runtime causal mask): active in both env states;
  bit-identical to the stock kernels on every row the stock kernels serve
  (m8 wheel G4: M1-M0-CONTROL MATCH).
- Hunk 2 (vector gate): stock `query_sequence_length < 4` becomes
  `query_sequence_length < 4 || (query_sequence_length == 4 && s4va_enabled())`;
  `s4va_enabled()` reads `MLX_ROCM_SDPA_S4VA` **once per process lifetime**
  (`static` init, `=1` engages). Env unset → exactly the stock gate → qL=4
  rows take the prefill kernel exactly as on the no-graft wheel.

**G1 compile bar**: `hipcc --offload-arch=gfx1201 -fsyntax-only` on the exact
post-patch file: 0 errors.

**Chain gate (fresh worktree, `git apply --check` + apply, in order):**

1. `patches/rocm-vendor-delta/delta.patch` — OK
2. `patches/rocm-vendor-delta/new_files/` — copied
3. `patches/gdn-fused-step-tape/conv-row-fix.diff` — OK
4. `patches/verify-attention-s4/sdpa_s4_env.patch` — OK

Resulting `scaled_dot_product_attention.hip` and `gdn_fused_step_tape.hip`
are byte-identical to the guest-tree files that build the wheel.

**G2 kernel gate (env ON, v4 wheel)** — `test_s4va.py`, direct SDPA,
B=1 H=16 qL=4 D=128 bf16 causal, kL ∈ {33, 67, 129, 257, 512}, 16 trials:

| gate | bar | result |
|---|---|---|
| G2.1 per-row bit-exact vs AR S=1 | n=0 | **n=320, bad=0 PASS** |
| G2.2 qL=1/2 regression vs no-graft wheel | n=0 | **bad=0 PASS** |
| G2.3 fp32-reference sanity (bf16) | max\|Δ\| ≤ 2e-2 | **3.75e-3 PASS** |
| G2.4 dispatch proof | vector wall ≪ prefill ~4.7 ms | **1000-call wall 0.4 ms (0.39 µs/call) PASS** |

## Job 4 — full-stack wheel

`v4-fullstack` = vendor delta + conv-row-fix + env-gated graft (cp314,
22.4 MB), built from the guest vendor tree (incremental 16-core, clean, 0
errors); the `v2c` no-graft baseline wheel (vendor delta + conv-row-fix only)
was built for the G0.3 control and the default-off comparison. Installed into
the verify venv (no pip in venv: wheel unpacked over `site-packages`);
state-None guard installed and compile-checked.

## Job 5 — 4-arm K=3 battery (lossless gates)

One fresh AR reference (mtp-bench6, FUSE=1): **400 toks, 74.47 tok/s**
(G5: the S=1 AR path is untouched by the graft — G2.2 n=0; 74.47 tok/s
recorded on the v4 wheel).

400-token budget arms (mtp-bench7, MLX_MTP=4, DRAFT=3, 200 verifies each):

| arm | GDN op | S4VA | tok/s | FIRST20 | stream vs AR |
|---|---|---|---|---|---|
| A0 | fused | off | 42.72 | MATCH | diverges @299 (974 vs 13081) — pre-graft signature |
| A1 | fused | **on** | 49.69 | MATCH | **BIT-EXACT (400/400) — G4.1 PASS** |
| A2 | tape | off | 42.00 | MATCH | diverges @299 (974 vs 13081) — pre-graft signature |
| A3 | tape | **on** | 48.88 | MATCH | **BIT-EXACT (400/400) — G4.1 PASS** |

- **G4.1 lossless (the gate): PASS** — A1 and A3 K=3 streams bit-match the
  AR stream (FIRST20 + ALLTOKS prefix, 400/400 tokens).
- **G4.2 default-off: PASS** — A0/A2 carry the known pre-graft
  prefill-attention signature (identical divergence point/token in both arms,
  matching the G0.3 control on the no-graft wheel); A0/A2 vs A1/A3 show the
  graft flipping the verify path (wall −8.5 ms, stream becomes bit-lossless).
- Acceptance is verify-path-dependent (A0/A2 D3 56.3 % vs A1/A3 48.2 %) —
  recorded, not a gate input; throughput is the gate.

Instrumented 128-verify A/B (mtp-bench-k3-tape, 256-token budget, 128
verifies, tape arm: `TAPE n_ok=128 n_fb=0`, fused arm: FUSE_CALLS only):

| arm | tok/s | verify wall | refwd wall | pair wall |
|---|---|---|---|---|
| A0 (fused, S4VA off) | 45.62 | 33.83 | 24.79 | 68.50 |
| A1 (fused, S4VA on) | 50.98 | 25.33 | 26.39 | 61.60 |
| A2 (tape, S4VA off) | 59.10 | 34.66 | 9.30 | 53.93 |
| A3 (tape, S4VA on) | 69.48 | 25.99 | 9.23 | 45.20 |

**Lever confirmed (G4.3 / RUN-card G5):** S4VA on cuts the verify wall
−8.5 ms (fused) / −8.7 ms (tape) and the pair wall 68.50→61.60 / 53.93→
45.20; tape op removes the per-row re-forwards (refwd 24.79→9.30,
n_refwd 272→101). Net: **A3 69.48 tok/s = +52 % vs A0, +36 % vs A1,
+17.6 % vs A2.**

## Gate summary

| gate | result |
|---|---|
| G0.1 fresh-cache guard (S=4, S=1) | PASS (no crash) |
| G0.2 rows 0–3 conv parity + full-state parity | PASS (4/4 md5, conv+rec md5) |
| G0.3 no-graft attention signature (control) | PASS (max 6.94 documented) |
| G1 hipcc syntax bar (post-patch SDPA) | PASS (0 errors) |
| G2.1–G2.4 kernel gate (env ON) | PASS (n=0 ×320, 3.75e-3, 0.39 µs/call) |
| G4.1 lossless (A1, A3 vs AR) | **PASS (400/400 bit-exact)** |
| G4.2 default-off (A0/A2 signature) | PASS (pre-graft signature, both arms) |
| G4.3 lever (per-pair wall A/B) | CONFIRMED (−8.5/−8.7 ms verify; +5.4/+10.4 tok/s) |
| G5 AR regression | 74.47 tok/s, S=1 rows n=0 (untouched) |
| chain gate (fresh 75ca881, 4 steps `git apply --check`) | PASS (byte-identical to guest tree) |

## Reproduce

`docs/RUN-verify-attention.md` (full card: chain, G1, build, G0–G5, verdict,
rollback). Patch layout: `patches/rocm-vendor-delta/`,
`patches/gdn-fused-step-tape/`, `patches/verify-attention-s4/`.

**Rollback.** `git apply -R sdpa_s4_env.patch` + rebuild, or reinstall the
no-graft wheel; with the env unset the grafted wheel behaves stock.
