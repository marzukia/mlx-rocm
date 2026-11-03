# MTP fix round 2 — report (2026-11-03)

## Verdict

- **A/B (canonical vs hybrid vs pure tape): STOPPED at the gate.** PRE (locked regime, 3-run median) = **37.47 t/s**, not in the 65-66 class. Per protocol, the A/B was not run; the regime delta is reported below.
- **Hybrid losslessness gate: FAIL.** The hybrid (tape state commit + one S=1 correction forward, l/h from the correction forward) diverges from canonical at **token 32** under *both* mtp_cache protocols. Changing the h source (batched row → correction forward) does not fix the divergence.
- **39.16 run identified and reproduced.** Round 1's "39.16 measured" run was the 059245e *hybrid* behavior (correction forward + batched-row h). My H arm reproduces it: 38.88 t/s vs 39.16, 1657 vs 1657 tokens, D3PCT 48.5 vs 48.5, REFWDS 395 vs 395.
- **The 02:40→03:00 "regime drift" (D3PCT 60.8→51.1) was a bench-file artifact, not machine drift** — see Finding 1.

All runs: venv-canon wheel `75ca881` (installed Nov 2 11:56, unchanged), model `qwen3-9b-mlx` + sidecar `model-mtp.safetensors` (Oct 27, unchanged), greedy, 512 verifies, locked regime `force_performance_level=high` (sclk = 2200 MHz = DPM table max; mclk 3000 MHz; confirmed at every run start; power 34-85 W / 32-35 C under load, no throttling).

## STOP gate and regime delta

| Arm | t/s | notes |
|---|---|---|
| PRE canonical, run 1/2/3 | 37.70 / 37.47 / 37.40 | median **37.47**; 1623 toks; D1PCT 72.7 D2PCT 61.0 D3PCT 60.8 |
| AR, run 1/2/3 | 84.86 / 84.81 / 45.41 (contaminated) | median **84.81**; 400-token arstep 74.4-75.3 (73-74 class ✓) |

**PRE = 37.47 → STOP gate triggered.**

Regime delta: the 65.88 anchor (t1387, Nov 2 17:29-19:43Z) was measured on the **VM** (the previous guest, now dead). This bare-metal box (guest-a, booted Nov 2 ~20:23Z) has **never** produced 65-class MTP in any clock state: best ever 37.71 (DPM-era 00:15Z), 37.42 (locked 02:15Z), 37.47 (locked, current session). MTP/AR ratio: 0.50 on this box vs 1.08 on the VM. The "65-66 class" expectation is a genealogy error — the S=4 batched-verify path is pathologically slow relative to S=1 on bare-metal. Operator judgment required before the A/B can proceed.

## Losslessness gate results (median-of-3 not applicable: gate stopped; 1-run arms with the controlled variable isolated)

| Arm | config | t/s | toks | D1/D2/D3 PCT | REFWDS |
|---|---|---|---|---|---|
| A0 | canonical + mtp_cache trim (new bench) | 36.76 | 1636 | 75.6 / 58.1 / 51.1 | 397 |
| A0n | canonical, no trim (old bench protocol) | 37.43 | 1623 | 72.7 / 61.0 / 60.8 | 374 |
| H | hybrid + trim | 38.88 | 1657 | 76.6 / 61.5 / 48.5 | 395 |
| Hn | hybrid, no trim | 38.67 | 1635 | 73.4 / 62.5 / 52.3 | 389 |

**A0n exactly reproduces the r2-pre PRE run (37.43 vs 37.47 median, 1623 toks, identical D*PCT, REFWDS 374)** — the old-bench canonical protocol is bit-reproducible.

ALLTOKS (programmatic diff, same prompt, greedy):

- A0n vs Hn: **first divergence token 32**
- A0 vs H: **first divergence token 32**
- A0n vs H: **first divergence token 32**

Divergence signature (identical across all pairs): canonical `...52046 8765 314 7752 329 2978 11 24296 11 321 21461 5306 1083 264 3545 8882...` tape `...52046 8765 314 12103 6599 1083 264 3545 8882...` —the tape branch skips 8 tokens, emits 2, then re-merges onto a shared tail. FIRST20 anchor: intact in all arms (divergence is at token 32 > 20). Extra tokens vs canonical: H +34, Hn +12 (duplicated correction emission).

## Findings

1. **The apparent 02:40→03:00 canonical drift (D3PCT 60.8→51.1, 1623→1636) was my round-2 bench, not the machine.** The round-2 bench's unified `reject()` added `mtp_cache[0].trim(ktrim)` to the canonical reject path; the old bench (K=2-arm protocol) never trims mtp_cache. Now controlled via `MTP6_MTPCACHE_TRIM`. A0n (no trim) reproduces the old-bench canonical exactly. Wheel and sidecar verified unchanged in the window.

2. **Hybrid does not fix the losslessness gate.** H and Hn diverge at token 32 with the same signature; the h-source change (batched row → correction forward) is a no-op for the token sequence. The divergence is caused by the tape commit state-selection path (GDN `last_tape` row select + KV trim + correction forward), not the (l, h) source. The mtp_cache trim slightly worsens tape mode (Hn is closer to canonical than H).

3. **My "pure" mode (tape commit + correction forward + l,h = batched row) degenerates** to a 1919-token loop (1822 toks, D1PCT 99.6%, D3PCT ≈ 0). The batched-row l protocol (argmax re-emits the correction token) is not viable on this wheel, and is not a faithful reproduction of the 7455906 "39.30" measurement (that code had no correction forward at all).

4. **The tape kernel is math bit-identical to the fused kernel.** `diff gdn_fused_step.hip gdn_fused_step_tape.hip` (normalized): only the per-row tape stores (SSM `state_tape[t]` = register-resident post-update `h`, `state_tape[3] == state_new`; conv window `conv_state_tape[t] = {E[t+1], E[t+2], E[t+3]}` — verified correct against the kernel's row-t conv inputs), helper renames (`gdnf_`→`gdnt_`), and debug-out removal. So the S=4 verify forward is bit-identical in both modes; the divergence comes from the *state commit/rebind + correction-forward path*.

## Diagnosis

The canonical and tape reject paths are **mathematically equivalent but not bit-identical in floating point**:

- canonical: GDN state after the accepted prefix = product of **three separate S=1 fused forwards** (restore to pre-verify state, then forward prev_token, d1, d2 sequentially);
- tape: state = the **register-resident state of the S=4 verify forward** at row 2 (one continuous T-loop, different fp accumulation order).

Near-identical states → near-identical acceptance (D1PCT 73.4-76.6 tape vs 72.7-75.6 canonical; D3PCT 48.5-52.3 vs 60.8) but not bit-identical → early greedy argmax flips → divergence at token 32. The tape path is therefore a *near-lossless* numeric variant, not a bit-lossless one — **unless** the rebind/trim has an indexing bug (hypothesis b). Inspection of the kernel row/conv indexing and the bench rebind (`st[krow]` with the B-dim re-shape, KV `trim(3-krow)`, mtp_cache trim) found no defect, but the two hypotheses are not yet separated.

## Next steps (recommended)

1. **Decisive tape-state identity test** (small GPU script, one verify forward on each path): compare (a) tape row-2 state vs the canonical re-forwarded state — bit-identical? (b) the correction-forward output from each state — identical? Separates hypothesis (a) fp reordering (tape can never be bit-lossless under greedy → either accept "near-lossless" with a documented divergence rate, or retire the tape path) from (b) indexing bug (fixable in kernel/glue).
2. A/B speed comparison remains stopped by the gate; operator judgment on the VM-vs-bare-metal genealogy is required. Informational (uncontrolled, 1-run): hybrid 38.67-38.88 t/s vs canonical 37.43 t/s (~3.5% up, unvalidated).

## Artifacts

- Bench: `bench/mtp-bench6.py` (local, 3-way `MTP6_TAPE` switch + `MTP6_MTPCACHE_TRIM` control + REFWDS/GDN_TAPE_CALLS/INT2 INFO line); remote `~/mtp-bench6.py` md5 4475be5d… (plus the MTPCACHE_TRIM patch).
- Phase bench: `bench/mtp-bench6-phase.py` (unified reject + commit/correction phase timers), remote md5 608a7770.
- Remote logs: `/tmp/r2-pre.log`, `/tmp/r2-hybrid.log`, `/tmp/r2-canon2.log`, `/tmp/r2-arm.log`.
