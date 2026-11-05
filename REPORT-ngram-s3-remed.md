# REPORT-ngram-s3-remed: MTP6_NGRAM_S3 re-measurement on a non-degenerate trajectory (t1996, guest-a RX 9070 XT / gfx1201, ROCm, 2026-11-05)

**Verdict: the candidate is REJECTED (net loss) on the non-degenerate trajectory it was re-measured for. Lever (a) (the t1436 anchor "Hello" prompt, config-of-record verbatim) FAILED: the S=3 pure-tape protocol (MTP6_TAPE=1) locks into the 883 attractor on the anchor prompt just as it locks into the 1919 attractor on the canonical prompt — the loop is prompt-independent under this protocol. Lever (b) (minimal knob ablation) found the single knob that breaks the loop: MTP6_TAPE 1 → hybrid (the S=4 escape pattern; one knob moved, all other config-of-record envs verbatim). On that ablation trajectory (canonical smartphone prompt, 322 anchor): control 58.00 t/s median (3 runs bit-reproducible, spread 0.10%), A 58.00 vs B 57.22 t/s = **−1.34%**; tok/step 2.193 → 2.168 (down); the D1 leg DID move in the expected direction (D1 114 → 117 accepts, +1.9 pt, the position-1 fill replacing non-greedy MTP drafts) but the D2 leg moved down (78 → 71 accepts, +7 correction re-forwards) and the net is negative. Against the ladder (tok/step 3.15 anchor shape / 140+ t/s target / 120+ merge gate) both arms fail; the t1986 +34.2% number is confirmed as a degenerate-attractor artifact, not a transferable lever. All 23 runs rc=0 (9 session 1 + 14 session 2), SCLK 53/53 + 84/84 = 137/137 samples at 2200, wheel + sidecar md5s verified pre/post both sessions, FIRST20 bit-identical across every A/B run in each session.**

## 0. Mission and why

t1986 measured the candidate (MTP6_NGRAM_S3=1, wired into the S=3 draft chain, default off = canonical S=3 verbatim) on the degenerate 1919-loop trajectory: A 89.68 vs B 120.33 t/s = +34.2% (tok/step 2.907 → 2.969, D2 93.1 → 99.4, D1 unchanged 98.8). It was DECLINED against the bar (tok/step >= 3.0) — and the failure was structural: the 3.0 gate is unreachable on that trajectory by construction (ceiling 2.9876 at D1 98.8%). The leg that matters on real trajectories — the D1 leg (position-1 fill replacing non-greedy MTP drafts) — was UNMEASURED because the loop cannot show it. Worker recommendation carried into this ticket: re-measure on a session that does not lock into the 1919 loop before any adopt decision.

## 1. Setup and identity

- Base ngram-s3-probe @ f2f49d7 (already carries the candidate; harness A 1d9bc556 / B 7d2ace6b record from t1986). Machine: guest-a (<tailnet-ip>, RX 9070 XT / gfx1201). GPU lane: flock on ~/gpu.lock (queued behind sibling run t1989 in session 1; free before session 2; never killed another ticket's holder; released and verified free after each session).
- Wheel: venv-p3 for ALL arms, libmlx.so md5 460c803d66c43ebfce8997ef08c912db verified before AND after both sessions (untouched, no rebuild). Model sidecar <models-mount>/qwen3-9b-mlx/model-mtp.safetensors md5 02a54359cda8b247f76b277dd9bc4d4c verified before AND after both sessions (the 02a54359 pin only).
- GPU env: MODELS=<models-mount>, LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 (the t1981 pattern), HCC_AMDGPU_TARGET=gfx1201. SCLK: idle reading 2200 verified at the start of both sessions; sampled every 10 s (rocm-smi at /opt/rocm-7.2.0/bin) — 53/53 @ 2200 through session 1 (18:56:41–19:05:29Z) and 84/84 @ 2200 through session 2 (19:11:18–19:25:20Z).
- Harness: one file for both arms (bench/mtp-bench6.py at f2f49d7 + the MTP6_PROMPT env override added this ticket, md5 54ce16ac29e6c6aad2c48a233ff70d85, verified on the box before use). MTP6_PROMPT defaults to the canonical smartphone prompt verbatim (default-off = byte-identical behavior to the t1986 harness); arm B = same file + MTP6_NGRAM_S3=1 in the env. Protocol: 322 anchor (MTP_MAX_TOKENS=322), config-of-record envs verbatim (MLX_MTP=3 MTP6_TAPE=1 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1; the DRAFTGATE/NGRAM entries are no-ops in this harness, kept for config-of-record parity), 25 s idle before/after every arm, per-arm timeout 900 (none hit).
- Loop-break gate (session 2): NO run of ANY token longer than 3 in the first 64 tokens (supersedes the 1919-only check after session 1's 883 attractor), FIRST20 diverse, FIRST20 bit-identical across the 3 controls, t/s within a 1% band, before any candidate measurement.

## 2. Investigation (code-first): the t1436 anchor and what breaks the S=3 collapse

**The t1436 anchor protocol (repo record: /srv/fleetwork/work/1436/REPORT-llamacpp-anchor.md, same box, same model family, 2026-11-02):** llama.cpp 0.8.0-dev commit 29c720e native HIP, q4_K_M GGUF with the MTP head, 1-prompt-token `Hello`, 512 generated tokens, greedy (`--temp 0 --top-k 1`), `--ignore-eos`, ctx 1024, MTP draft K=3. Result: AR 86.9 t/s, MTP 144.3 t/s, **mean acceptance 2.15/3 over 161 spec steps** — a diverse, non-looping trajectory (a looping trajectory would sit at ~3.0 acceptance; 2.15 is real-text acceptance). So the anchor prompt is `Hello` and the anchor did NOT loop. (The anchor used its own 1-token prompt, not the repo's canonical smartphone prompt; matching the anchor prompt is therefore the most comparable non-degenerate trajectory, per the ticket.)

**The S=3 loop is config-level and path-specific (repo records):** REPORT-acceptance-t3.md documents the S=3 arms locking into `271 + 19x1919` from position 3 while the S=4 arm escapes at position 3 on the SAME prompt and box (`271 1919 369 264 15352 ...`), attributed to the S=3 correction path in a near-tie degenerate region (the S=4 4-row verify does not lock; plain AR is also diverse on the same prompt — REPORT-mtp-r4c-int2-head.md AR arm). REPORT-mtp-fix-round2.md documents the S=3 "pure" protocol (tape commit + correction forward + l/h = the batched-row logits/hidden, i.e. MTP6_TAPE=1 in mtp-bench6.py mode 3) degenerating to the 1919 loop explicitly. Code-first: in bench/mtp-bench6.py mode 3, `MTP6_TAPE=1` (the config-of-record value) selects `l, h = lk, hb` (the batched verify-row values) as the next draft chain's inputs on rejection; `MTP6_TAPE=hybrid` instead uses the S=1 correction forward's own l/h; `MTP6_TAPE=r5` defers the correction (ctoken = row 0 of the next verify, no re-forward). The candidate knob candidates the ticket names map to: tape = MTP6_TAPE (1/hybrid/r5), postnorm = MTP6_HEAD_H, head = MTP6_DRAFT_HEAD, gate knobs = the no-op MTP6_DRAFTGATE family.

## 3. Session 1 — lever (a): the anchor prompt on the config-of-record verbatim (FAILED the diversity gate)

Lever (a) test: MTP6_PROMPT=Hello, config-of-record verbatim (MTP6_TAPE=1), 322 anchor, 3 controls + 3x2 battery, all venv-p3.

| run | t/s | toks | tok/step | D1/D2 | REFWDS | FIRST20 |
|---|---|---|---|---|---|---|
| ctl1/2/3 (A, Hello, tape=1) | 88.48 / 88.80 / 88.44 (spread 0.41%) | 480 | 2.944 | 98.1 / 98.1 | 6 | `11 353 353 599 599 264 3296 3296 883 x12` (bit-identical) |
| bat A r1/r2/r3 | 88.62 / 88.43 / 88.24 (**median 88.43**) | 480 | 2.944 | 98.1 / 98.1 | 6 | reference |
| bat B r1/r2/r3 (+MTP6_NGRAM_S3=1, NNG=312) | 112.42 / 112.47 / 112.22 (**median 112.42**) | 480 | 2.950 | 98.1 / 98.7 | 5 | reference |

B vs A = **+27.2%** on this session. The literal 1919-run gate PASSED (zero 1919 in the first 64 tokens) but the trajectory is DEGENERATE: 472 of 480 tokens are token 883 (a new attractor; the same class as the 1919 loop — the stream locks onto a single-token fixed point from position ~9). **Lever (a) verdict: the S=3 pure-tape protocol is prompt-independent in its collapse — the anchor prompt does not break it, it just changes the attractor.** The t1436 anchor's non-looping "Hello" trajectory is therefore a property of the llama.cpp engine (K=3, its own draft/verify structure), not of the prompt under our S=3 path. (Session 1 data stands as the second degenerate-trajectory datapoint for the candidate: +27.2% here vs +34.2% in t1986, both attractor-dominated.)

## 4. Session 2 — lever (b): minimal knob ablation (the loop-break knob: MTP6_TAPE 1 → hybrid)

Four one-knob probes on the canonical smartphone prompt (config-of-record + exactly one knob moved, 322 anchor, arm A config), in comparability order:

| probe | knob moved | loop gate (first 64 / full 480) | t/s | tok/step | D1/D2 | REFWDS |
|---|---|---|---|---|---|---|
| P1 | MTP6_TAPE: 1 → **hybrid** | **BREAKS (1 / 1)** | 58.26 | 2.193 | 70.8 / 68.4 | 83 |
| P2 | MTP6_TAPE: 1 → r5 | breaks (1 / 1) | 50.77 | 2.081 | 67.1 / 61.1 | 0 (CDEFERRED 95) |
| P3 | MTP6_HEAD_H: postnorm → prenorm | still loops (63 / 480, 1919 x480) | 83.09 | 2.882 | 98.8 / 90.6 | 17 |
| P4 | MTP6_DRAFT_HEAD: int2 → bf16 | still loops (63 / 480, 1919 x480) | 84.26 | 2.907 | 98.8 / 93.1 | 13 |

**Winner: P1 (ONE knob moved: MTP6_TAPE 1 → hybrid)** — the S=4 escape pattern (the S=4 hybrid arms in REPORT-r5.md / REPORT-mtp-r4c-int2-head.md run exactly this tape mode and escape the loop; P4's 84.26 t/s also confirms the t1986 config-of-record A arm is reproduced on this box, 84.26 ≈ the 84.11/89.68 class). P3/P4 (head-flavour / head-quant knobs) do NOT break the loop — the collapse is a protocol property of the tape=1 batched-row l/h, not of the head. **Anchor comparability probe (P5):** the winning ablation on the anchor "Hello" prompt is ALSO non-degenerate (69.11 t/s, 2.509 tok/step, D1 83.2 / D2 81.3, REFWDS 52, FIRST20 `11 353 599 264 3296 883 279 2614 1970 13 353 599 264 523 421 682 264 1654 421 4523`, maxrun_all=1) — the hybrid ablation is the trajectory on which the anchor prompt becomes comparable. The battery below runs the canonical prompt (config-of-record prompt, one knob moved) so the A arm is directly comparable to t1986's A arm on the same box; the ablation numbers are ablation datapoints, not the config-of-record numbers.

## 5. Phase 0 — control (the first gate): REPRODUCED, bit-reproducible

3 control A runs (hybrid ablation, canonical prompt, 322, config-of-record + MTP6_TAPE=hybrid):

| run | t/s | toks | tok/step | D1/D2 | REFWDS | FIRST20 |
|---|---|---|---|---|---|---|
| ctl1 | 58.10 | 436 | 2.193 | 70.8 / 68.4 | 83 | `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` (reference) |
| ctl2 | 58.05 | 436 | 2.193 | 70.8 / 68.4 | 83 | reference |
| ctl3 | 58.04 | 436 | 2.193 | 70.8 / 68.4 | 83 | reference |

FIRST20 bit-identical across the 3 runs, t/s spread 0.10% (< 1% band), NO token run > 3 in the first 64 tokens (maxrun64=1; the full 436-token stream has maxrun_all=1 — zero consecutive repeats). GATE PASS before any candidate measurement. The reference line is the known lossless/diverse anchor line (identical to the AR and S=4 baseline prefixes in the repo records).

## 6. A/B battery (3 rounds x 2 arms interleaved, 322 anchor, hybrid ablation, venv-p3 both arms)

| run | A (ablation, ngram off) | B (ablation + MTP6_NGRAM_S3=1) |
|---|---|---|
| r1 | 58.10 (D1 114, D2 78, 2.193, REFWDS 83) | 57.23 (NNG=14, D1 117, D2 71, 2.168, REFWDS 90) |
| r2 | 58.00 (114, 78, 2.193, 83) | 57.22 (14, 117, 71, 2.168, 90) |
| r3 | 57.99 (114, 78, 2.193, 83) | 57.16 (14, 117, 71, 2.168, 90) |
| **median** | **58.00** | **57.22** |
| spread | 0.19% | 0.12% |

**B vs A = −0.78 t/s = −1.34% (same wheel, same session, interleaved).** All 6 runs rc=0; FIRST20 bit-identical (the diverse reference line) in all 6; full-stream loop-free (maxrun_all=1) in all 6. SCLK 84/84 @ 2200 over the session window.

## 7. The D1-leg answer (the unmeasured leg from t1986)

- **The D1 leg moved, in the expected direction:** D1 114 → 117 accepts (+3, 70.8% → 72.7%): on real text the MTP module's position-1 draft is often non-greedy (D1 70.8% for arm A), and where the n-gram proposal hit (NNG=14 positions over 322 steps), it supplied the greedy continuation and flipped 3 rejects to accepts. This is exactly the leg the t1986 loop trajectory could not show.
- **But the D2 leg moved down:** D2 78 → 71 accepts (−7, 68.4% → 60.7% of D1), adding 7 correction re-forwards (REFWDS 83 → 90, ~+80 ms at the 11.7 ms/refwd class). The position-2 n-gram proposal (key = 3 confirmed context tokens + draft1, most-frequent continuation) is a WORSE prior than the MTP module's draft2 on this text: the MTP module has consumed draft1's embedding (a 2-token richer context); the n-gram map has only the frequency prior over n=4 windows of a 436-token context.
- **Net: tok/step 2.193 → 2.168 (down), t/s −1.34%.** On non-degenerate text the fill is a net loss: it helps position 1 (the unmeasured leg, +3) and hurts position 2 (−7, plus re-forward cost). NNG=14/322 (4.3% of steps) vs 316/322 (98%) on the t1986 loop — n=4 n-gram repetition is rare on real text, so the fill is almost never active and where it is active the D2 regression outweighs the D1 gain.
- Lossless: all emitted tokens remain target-greedy (verify pass untouched); FIRST20 bit-identical across arms; full streams loop-free and bit-stable per arm across rounds.

## 8. Adoption recommendation against the ladder

Ladder: tok/step 3.15 anchor shape (llama.cpp 2.15/3 acceptance), 140+ t/s target, 120+ merge gate; t1986 bar (t/s >= +5% same-wheel AND tok/step >= 3.0 AND FIRST20 identical).

| gate | A (ablation) | B (ablation + candidate) | verdict |
|---|---|---|---|
| tok/step vs 3.15 | 2.193 | 2.168 | both FAIL; candidate moves it the wrong way |
| 140+ t/s target | 58.00 | 57.22 | both FAIL |
| 120+ merge gate | 58.00 | 57.22 | both FAIL |
| t/s vs A (>= +5%) | — | −1.34% | FAIL |
| tok/step >= 3.0 | 2.193 | 2.168 | FAIL |
| FIRST20 identical | reference | reference (all 6 runs) | PASS |

**REJECTED (declined).** On the non-degenerate trajectory the candidate is a net loss (−1.34% t/s, tok/step down, D2 regression + re-forward cost > D1 gain). The t1986 +34.2% is confirmed as an attractor artifact: both degenerate-trajectory measurements (t1986 +34.2% on the 1919 loop; this session +27.2% on the 883 loop) are dominated by the fill's D2 substitution inside a single-token fixed point, and neither transfers to a real trajectory. The candidate remains env-gated (default off = canonical verbatim); nothing changes for the config-of-record. Recommendation: do not adopt MTP6_NGRAM_S3 for the S=3 ladder; if the n-gram-fill idea is revisited, the position-1-only fill (skip the position-2 lookup) is the mechanism this measurement supports (+3 D1 accepts, no D2 regression, no extra re-forwards) — noted as a future probe, not measured here. Separately, the loop-break finding is the load-bearing one: the S=3 config-of-record (MTP6_TAPE=1, pure batched-row protocol) is degenerate on EVERY prompt tested (1919 on canonical, 883 on the anchor prompt) while MTP6_TAPE=hybrid is non-degenerate on both prompts and reproduces the known lossless anchor line — the tape-mode defect is a config-level property worth its own ticket.

## 9. Provenance

- Two sessions on guest-a, one flock window each (session 1 18:56:41–19:05:29Z queued behind sibling t1989; session 2 19:11:18–19:25:30Z on a free lock), 23/23 runs rc=0, per-arm timeout 900 (none hit), 25 s idle before/after every arm, SCLK 53/53 + 84/84 = 137/137 samples at 2200 MHz (idle verified both sessions), lock released and verified free after each session.
- md5s: venv-p3 libmlx 460c803d (pre + post, both sessions); sidecar 02a54359 (pre + post, both sessions); harness 54ce16ac (= the f2f49d7 candidate-carrying file + the MTP6_PROMPT default-off override; byte-identical behavior at default prompt); gate/gatecheck/gatepick + runner scripts md5s 94fba100 / 163e9d79 / 88875850 / e170162e / df74ca61; all verified on the box before use.
- Config-of-record env set verbatim in every run (the DRAFTGATE/DGATE_T0/NGRAM entries are no-ops in this harness — t1986-confirmed); the ablation moved exactly ONE knob (MTP6_TAPE 1 → hybrid); arm B = ablation + MTP6_NGRAM_S3=1.
- t1436 anchor record: /srv/fleetwork/work/1436/REPORT-llamacpp-anchor.md (prompt "Hello", 512 tokens, greedy, --ignore-eos, K=3, acceptance 2.15/3 = non-looping); the anchor used its own 1-token prompt, not the repo's canonical prompt.

## 10. Check lines (reported separately)

- **Persona-leak: CLEAN.** Grepped tree + this commit + this report for the standard persona-term list (fleet hostnames, tailnet IPv4 range, box-user identity + home, the box-adjacent service-account identity, LAN IPv4 range, the two personal domain suffixes): zero hits in the report or in the two commits' non-git metadata (git author lines exempt per protocol). The report and the harness comment use the placeholder convention (guest-a / <models-mount> / <tailnet-ip>).
- **Line-wrap: CLEAN.** No in-tree scripts/line_wrap_check.py; fleet-detector fallback applied — the report is flowing lines (no hard-wrapped prose); the only lines that end early are markdown table rows and list items.
- **Format: APPLIES + PASS.** The repo CI (.github/workflows/ci.yml) defines a .py format check (`uv run ruff format --check .` + `uv run ruff check .`): both run green with this ticket's changes in-tree (110 files already formatted; all checks passed). The only .py touched is bench/mtp-bench6.py, which is ruff-excluded in pyproject.toml (as-run evidence convention); the report is a new .md. `uv run pytest -m "not gpu"`: 25 passed, 1 skipped.
