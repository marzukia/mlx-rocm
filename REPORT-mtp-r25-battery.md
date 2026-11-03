# R2.5 diagnostic battery — per-arm ms/token decomposition, wheel diff, locked-regime re-run, tape state-identity
 **Scope**: diagnostics only. No mission path changes, no R3 lever picks. All GPU work on guest-a (`<tailnet-ip>`, RX 9070 XT / gfx1201, ROCm 10.1 guest userland) under `flock ~/gpu.lock`, DPM high, 25 s idle between runs, sidecar md5 pinned before/after.
 Model: `qwen3-9b-mlx` + sidecar `model-mtp.safetensors` (pin `02a54359cda8b247f76b277dd9bc4d4c`, verified pre/post every lock session).

---

## Item 1 — CPU: where does the ~50–85 ms/step go?

### 1.1 Per-arm totals (locked regime, per verify step; 1 step = 1 S=4 verify + 3 MTP draft calls)

| Arm | Wheel / venv | toks/s | ms/step | ms/tok | tok/step | S=1 fwrds/step |
|---|---|---|---|---|---|---|
| AR (greedy, no MTP) | graft (93810e87) | 84.8 | 11.79 | 11.79 | 1.0 | 1.0 |
| Canonical savestate | 75ca881 / venv-canon | 37.27–37.47 | 85.0 | 26.83 | 3.17 | 2.09 re-fwd + corr |
| Canonical tape (r2-hybrid) | 75ca881 / venv-canon | 38.94 | 83.1 | 25.68 | 3.236 | 0.79 corr |
| Tape (m4r2 P) | **ed40b1a / venv-patch1** | 23.73–23.76 | 136.4 | 42.14 | 3.237 | 0.79 corr |
| Graft R0 | 93810e87 / venv-ra | 60.77 / 61.08 | 52.2 | 16.40 | 3.188 | 0.79 corr |
| Graft R3 | 93810e87 / venv-ra | 65.68 / 65.64 | 49.0 | 15.22 | 3.219 | 0.77 corr |
 ms/step = GEN/512 (bench) or GEN/128 (graft, per verify step). tok/step = t/s × ms/step / 1000, ms/tok = ms/step / tok/step. Graft rows are the **R2.5 re-run** (item 3); canonical/tape rows are the t1500 round-2 + m4r2 logs.

### 1.2 Per-phase decomposition (ms/step, ms/tok)

Graft rows use the bench's explicit `mx.eval()` syncs (clean). Canonical/tape rows use `mtp-bench6-phase` phase timers, which are **contaminated by lazy evaluation** (see §1.3): the "verify" timer measures graph construction only; the S=4 forward is paid in the commit window, and the previous step's pending re-forwards/correction are paid at the top of the **draft** window.

| Phase | Canonical savestate (37.3) | Tape, 75ca881 (38.9) | Tape, ed40b1a (23.7) | Graft R0 (60.9) | Graft R3 (65.7) |
|---|---|---|---|---|---|
| Draft chain (3× MTP fwd + 3× head) | 31.8 (10.0) | — (in commit) | 82.2 (25.7) | 9.7 (3.1) | 9.7 (3.0) |
| — of which prior-step S=1 fwrds (lazy spillover) | ≈24.7 (2.09 fwrds × ~11.8) | — | ≈7 (0.79 corr × ~9) | — (synced) | — (synced) |
| S=4 verify | in commit | in commit | in commit | 33.1 (10.4) | 30.0 (9.3) |
| Commit (verify eval + rebind/restore + corr fwd) | 52.7 (16.6) | 52.8 (16.5) | 52.8 (16.5) | 9.2 (2.9) | 9.1 (2.8) |
| Other (glue, argmax, embed) | 0.5 (0.2) | ~1.2 | 1.2 (0.4) | 0.2 (0.1) | 0.2 (0.1) |
| **Total/step** | **85.0** | **83.1** | **136.4** | **52.2** | **49.0** |
| **Total/tok** | **26.8** | **25.7** | **42.1** | **16.4** | **15.2** |

### 1.3 Where the gap is (verified, not estimated)

1. **Canonical's 2.09 S=1 re-forwards/step are the canonical-specific cost** (≈25 ms/step ≈ 7.9 ms/tok): savestate reject = restore + 1–3 S=1 re-forwards + 1 correction; tape/graft reject = tape-row rebind (zero forward) + 1 correction. This is the dominant canonical-vs-graft difference.
2. **S=4 verify itself is NOT the driver**: canonical's S=4 (4-op chain, paid in commit) ≈ graft's 33.1 ms tape-verify — comparable. The 4-op vs fused kernel difference on the verify forward is real (item 4 quantifies it bit-wise) but small next to the re-forward policy difference.
3. **The ed40b1a (patch1) tape arm is 1.64× slower than the 75ca881 tape arm** (23.73 vs 38.94 t/s; identical D1/D2/D3 counts 392/241/117, identical REFWDS=395, identical GDN_TAPE_CALLS=21768). This is **not a machine effect** — same box, same session, same locked regime. But attributing the 1.64× to the *wheel* (ed40b1a vs 75ca881) specifically is **not established**: the m4r2-P arm ran with the bench's default draft head `MTP6_DRAFT_HEAD=int2` (`bench/mtp-bench6.py:179`), and the round-1 A/B measured tape+int2 = 23.77 vs tape+4-bit = 39.16 — ratio 1.647 ≈ 1.641, i.e. the int2-head penalty alone accounts for the full gap, and round-1 reports the int2 and 4-bit runs bit-identical end-to-end, so the shared "identical acceptance" cannot distinguish wheel from head. The m4r2-P arm's head choice is recorded in no in-branch report; the "savestate is unaffected (37.42 vs 37.47)" control compares two venv-canon runs (round-1 PRE vs round-2 PRE), not a wheel A/B — the only recorded venv-patch1 savestate arm is round-1's 22.72 (int2 head). → the "23.4-class" tape number in the t1500 report is at minimum a **draft-head (int2) effect; the wheel-specific share is unmeasured**. The 75ca881-class tape path runs 38.9.
4. **Wheel confound in the t1500 round-2 data**: the phase-tape/m4r2 runs ran on **venv-patch1 (ed40b1a)**, not venv-canon; the savestate phase runs ran on venv-canon. The t1500 report's "all runs: venv-canon 75ca881" is not correct for the tape arms.

### 1.4 Lazy-eval caveat (method)
 `mtp-bench6-phase` phase timers: `verify` = graph construction only (~0.5–1.2 ms); the S=4 forward is evaluated on the first `argmax` in the commit window; prior-step pending forwards (re-forward cascade / correction) are evaluated at the first `argmax(l)` in the next draft window. Graft `mtp-bench-k3-tape` uses explicit `mx.eval()` per phase → clean numbers. Totals (GEN) are exact for both; only the phase split for mtp-bench6 carries the spillover ambiguity above (decomposed in §1.2 with the AR S=1 rate as the re-forward unit cost).

---

## Item 2 — CPU: graft vs canonical model-side MTP plumbing diff
 **Verdict: the model side is byte-identical; the "D1 95→72.7" is a units misreading; the real acceptance difference is GDN-kernel FP reordering, not model plumbing.**

1. `mlx_lm/models/qwen3_5.py` is **byte-identical** between venv-canon (75ca881) and venv-ra (graft): md5 `7763fbbb…` on both. The entire `mlx_lm` package has **zero .py diff**.
2. The only binary difference is `libmlx.so`: graft = 75ca881 + **456 bytes** = the QMM REFWD_M1 patch (M=1 int8-dot4 LX path, env-gated, off by default). **No GDN kernel changes, no MTP plumbing changes.** All GDN paths (fused, fused-tape, 4-op chain) exist in both wheels; dispatch is per-call `os.environ.get` (Python) / `std::getenv` (C++) — toggleable mid-run.
3. **"D1 95 → 72.7" is a units misreading.** Graft bench prints D1ACC as a **raw count over 128 verifies** (95/128 = 74.2%); mtp-bench6 prints **percent over 512** (372/512 = 72.7%). Normalized, D1 is 74.2% (graft) vs 72.7% (canonical) — nearly identical. The real profile:

   | | Graft R3 (95/128) | Graft R0 (93/128) | Canonical savestate (372/512) | Tape 75ca881 (392/512) |
   |---|---|---|---|---|
   | D1 | 74.2% | 72.7% | 72.7% | 76.6% |
   | D2 (cond.) | 64.2% | 63.4% | 61.0% | 61.5% |
   | D3 (cond.) | 47.5% | 45.8% | **60.8%** | **48.5%** |
 The acceptance gap is **D3: graft 47.5% vs canonical 60.8% (−13.3 pt)**; D1/D2 are graft-equivalent or slightly better.
4. **Cause of the D3 gap: GDN kernel FP reordering (fused T=4 vs fused T=1×3).** Canonical savestate runs the fused GDN kernel for every forward (draft chain has no GDN — the MTP module is a single full-attention layer; `bench/mtp-bench6.py:166` sets `MLX_ROCM_GDN_FUSE=1` unconditionally for all mode-4 arms, including canonical savestate) — i.e. three T=1 fused re-forwards + one T=1 correction. Graft/tape run the fused tape kernel as one T=4 batched verify + one T=1 fused correction. Fused T=4 batched recurrence vs three T=1 re-forwards of the same fused kernel = different FP operation order → different hidden states after the first verify step → different later draft chains → different D3 acceptance. Item 4 measures the resulting state divergence bit-wise. (The MTP module itself — 1 layer, full attention, `mtp_num_hidden_layers=1` — contains **no GDN layer**; the GDN env toggles never touch the draft chain.)

---

## Item 3 — GPU (locked regime): t1434 protocol re-run of the old graft stack
 **Verdict: 65-class. The machine is NOT regressed; the 37-class is the canonical stack, not machine state.**

Protocol: `flock`, DPM high (caveat: DPM sysfs read EBUSY this session — see below), 25 s idle between runs, 2 runs/arm, sidecar md5 pre/post, FIRST20 signature, TAPE n_ok/n_fb.

| Arm | Run | toks/s | wall ms/step | single | verify | refwd | n_refwd | TAPE | FIRST20 |
|---|---|---|---|---|---|---|---|---|---|
| R0 (R0 gate + nothing) | 1 | 60.77 | 52.45 | 9.75 | 33.19 | 9.29 | 101 | n_ok=128 n_fb=0 | = t1434 |
| R0 | 2 | 61.08 | 52.18 | 9.72 | 33.02 | 9.21 | 101 | n_ok=128 n_fb=0 | = t1434 |
| R3 (stacked REFWD_M1) | 1 | 65.68 | 49.00 | 9.73 | 29.96 | 9.09 | 99 | n_ok=128 n_fb=0 | = t1434 |
| R3 | 2 | 65.64 | 49.03 | 9.72 | 30.00 | 9.08 | 99 | n_ok=128 n_fb=0 | = t1434 |

Gates:
- R0 window 57.33–60.98, drift headroom 61.5: R0 medians 60.93 (R0-2 at 61.08 is +0.10 above the window top, inside the 61.5 headroom — same disposition as t1434's 61.10/61.11).
- In-run spread: R0 0.52%, R3 0.06% (t1434: 0.02% / 0.15%).
- vs t1434 (R0 61.10/61.11; R3 65.74/65.84): R0 delta −0.54%/−0.05%, R3 delta −0.09%/−0.30%
  — inside the −0.55% t1434 band. FIRST20 bit-identical to the t1434/t1426 signature (271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 …).
- Pins: bench md5 2c2d6c90 ✓, guard 4a262a0d ✓, sidecar 02a54359 pre/post ✓, venv-ra .so 93810e87 ✓.
 **Interpretation.** Same machine, same locked regime, same stack: still 65.6/60.9. The 37.47-class canonical (item 1) runs on the same machine in the same session window. Therefore the 37-class is a **stack property** (4-op GDN chain + savestate re-forward policy + full bf16 head on 75ca881), not a machine-state regression. Nothing about the box moved between t1434 and now.
 Caveat: the `pp_dpm_force_performance_level`/`pp_dpm_sclk` sysfs reads returned EBUSY at session start and mid-session (amdgpu transient state); the background sampler did capture sclk during the runs (1476→3249 MHz ramp under load, 0 MHz in the 25 s idle gaps). The runs themselves produce bit-identical FIRST20 and t1434-band timings, which is the regime's actual acceptance criterion.

---

## Item 4 — GPU: tape state-identity test (one S=4 verify forward per path)
 **Verdict: NOT bit-identical — the tape row state and the canonical re-forwarded state differ at bf16/f32 ULP scale in every GDN layer; all shapes match, the tape op is internally consistent, and the correction argmax is identical across the full {state} × {kernel} matrix. The D3 acceptance gap is explained by GDN-kernel FP reordering, not an indexing bug.**
 Protocol: single process, venv-canon (75ca881), env toggled per call (per-call getenv confirmed). Deterministic shared prefix (prefill + single-token step + 3 MTP draft forwards, env off / 4-op chain) — prefix tokens `t=271 d1=8160 d2=369 d3=264`, `t=271` = t1434 FIRST20[0] ✓. One S=4 verify forward on each path (GDN_TAPE_CALLS=24 = all 24 GDN layers tape-dispatched). Bench decision tree gave krow=0 (d1 rejected: true next token 1919 ≠ d1=8160), so the comparison is tape row 0 (state after 1 token) vs canonical 1× S=1 re-forward; ctoken=1919. Comparison via `arr.__array__()` (bf16→f32 lossless upcast; `np.array(mlx_arr)` is a 0-d object wrapper in this build) → sha256 / byte-equality + max-abs-diff per layer.

| Check | Result |
|---|---|
| Tape row 3 vs op's returned final state (op self-consistency) | **BIT_EXACT=YES** (all 24 GDN layers) |
| Tape row 0 (krow) vs canonical re-forwarded state | **BIT_EXACT=NO** — ssm (f32) max diff 0.0013–0.034; conv (bf16) max 0.031–0.25; KV k/v (bf16) max 0.06/0.02–0.25; **all shapes equal** (24× (1,3,8192) conv, 24× (1,32,128,128) ssm, 8× (1,4,28,256) KV) |
| Correction forward, 4-way matrix {tape_state, canon_state} × {tape kernel, 4-op kernel} | argmax **identical (369) in all 4 cells**; logits max diff 0.125 (bf16 ULP scale) |
| Post-correction state (state after S=1 correction) | BIT_EXACT=NO, same ULP-scale magnitudes |
| L0 conv specifically | **bit-exact** on the state compare — expected: after 1 token the layer-0 conv state is raw input shift (no recurrence), so it is kernel-invariant; layers ≥1 diverge from the layer-0 ULP diff |

Interpretation:
- An indexing bug (wrong tape row, off-by-N trim, stale conv) would show O(1+) relative errors, zeroed/garbage regions, or shape/offset mismatch. Observed: a few ULP per tensor, equal shapes, L0-conv invariant, op-internal row 3 = final state. → **FP reordering** (fused T=4 batched recurrence vs three T=1 fused re-forwards of the same kernel; this probe additionally ran the 4-op chain — gdn_prep + gated_delta_update + kkt_solve + recurrent_update — in the 4-way matrix), with the conv-state diff from in-kernel conv rounding at different points in the T=4 batched vs T=1×3 recurrence.
- The one-step correction is robust to the state/kernel ULP diff (no argmax flip in any of the 4 matrix cells), so the divergence does not change individual correction decisions at ULP scale; what it changes is the **trajectory** — recurrent states compound the ULP diff over the generation, which is exactly the mechanism that shifts the D3 draft distribution (item 2: D3 47.5% graft/tape vs 60.8% canonical savestate).
- The tape op's per-row tapes are faithful to the op's own recurrence (row 3 = final state, bit-exact), so the commit path commits the *true* state of the fused kernel — the difference vs the canonical path is the kernel's FP order, not a commit defect.
 Caveat: this test covers the krow=0 reject (the first verify step on the deterministic prefix). The magnitude and character (ULP scale, equal shapes, L0-conv invariant) are the same across all 24 layers and all 8 KV layers; a krow=1/2 probe would follow the same machinery (the script is data-driven: `mtp-tape-state-identity.py` 18057e7f).

---

## Session hygiene

- All GPU work under `flock ~/gpu.lock`; 25 s idle between runs.
- Sidecar md5 `02a54359cda8b247f76b277dd9bc4d4c` verified pre/post both lock sessions.
- No files in the repo were modified by the test runs; test scripts live in /home/<host-user> (battery: run-r25-battery.sh 810495fa; state-identity: mtp-tape-state-identity.py 18057e7f).
- sysfs clock reads (pp_dpm_*) intermittently EBUSY this session — documented, did not affect the benchmark acceptance (FIRST20 + t1434-band gate).
