# REPORT-rows-statistical.md — QMV-rows under the STATISTICAL bar (t1307)

Branch `worker/rows-statistical` off `worker/qmv-rows-bench` @2d65738 (the
t1297 verdict: ws[4] rows kernel 3.14x faster, 1–2 ULP off the per-row
reference, K=3 diverges at token 172 — "do not serve" **under the
bit-exact bar**). Per the 11-02 op order this branch ships under the
**statistical bar** (MTPLX: distributionally identical, matched within the
plain path's own noise — NOT token-identical). The bit-exact track
(t1305) is a separate in-flight job; this branch does not touch it.

Full-stack wheel = 81-build vendor delta + gdn-fused-step-tape (conv-row-fix
+ state-None guard) + verify-attention-s4 (`MLX_ROCM_SDPA_S4VA=1`) +
qmv-rows (ws[4], `MLX_ROCM_QMV_ROWS=1` gate, **default off**). A3 = the
first three with rows OFF = the t1292 69.48 tok/s reference config.

- [x] wheel built + md5-pinned
- [x] engagement evidence
- [x] statistical gates (acceptance parity / distribution parity / first-divergence data)
- [x] bench table + position vs the 100+ target
- [x] verdict

## Build (guest-a, ROCm 10.1, gfx1201)

- Vendor clone: `vendor-v2` (fork delta + gdn-fused-step-tape +
  verify-attention-s4 + qmv-rows wsfix layers, all applied and build-verified
  in prior tasks) → copied to `vendor-rs` for this task.
- CMake cache path mismatch (two rounds of stale `CMakeCache.txt` from the
  clone location) fixed with a `sed` path rewrite across all caches.
- Wheel: `mlx-0.32.3.dev20261102+75ca881`
  (md5 `2442e063…`), installed into `venv-rs`
  (`libmlx.so` md5 `e6d3d182…`).
- SASS on the **installed** .so: `qmv_rows_msg` symbol present, 7276 FMA
  instructions in the rows kernel — the ws[4] path is compiled in.

## Engagement (the gate must actually fire)

`qmv_rows_engage.py`, N=248320 V=512 M=4 (the K=3 verify shape):

| shape | rows OFF | rows ON | speedup |
|---|---|---|---|
| N=248320 M=4 | 2.642 ms | 1.668 ms | **1.58x** |
| N=12288 M=4 | 0.142 ms | 0.121 ms | 1.17x |

Rows gate engages at M=4 with `MLX_ROCM_QMV_ROWS=1
MLX_ROCM_QMV_ROWS_MIN_N=1`; default-off wheel unaffected (OFF arm = A3).

## K=3 greedy A/B (non-canonical t1228-v2 run — RETIRED; raw data below)

The t1228-v2 per-token re-forward variant (`bench/mtp-bench-k3.py`,
guest copy `rs-mtp-bench-k3.py`), same session, 128 verifies per arm.
Restated from the script's raw printed lines (guest logs `rs-B0b.log` /
`rs-R1b.log`) — no re-derivation. The raw `TOKS` count is the total
generated (decoded) tokens emitted over the run — one per single-token
step plus the 1–3 tokens per verify (accepted drafts included, prompt
excluded); `TOKS_S` = TOKS ÷ total generation wall; the loop budget is
256 *iterations*, not 256 tokens, so 402/401 tokens over 128 verifies
is what the run actually produced. The original table's run-1 values
(50.58/50.89 tok/s, verify 25.46/22.11 ms, pair 62.09/61.57 ms) have
no surviving raw log on the guest (a guest-wide search found none) and
are struck as unverifiable; the original "402/401 tokens @ 128
verifies" header was attached to both runs and is not reproducible from
the surviving data.

| raw printed line | B0b (rows OFF, A3) | R1b (rows ON) |
|---|---|---|
| TOKS | 402 | 401 |
| VERIFIES | 128 | 128 |
| TOKS_S | 50.39 | 50.73 |
| D1PCT / D2PCT / D3PCT | 70.3 / 62.2 / 50.0 | 73.4 / 54.3 / 39.2 |
| WALL_MS | single 9.68 / verify 25.66 / refwd 26.80 (n_refwd 290) | single 9.68 / verify 22.24 / refwd 29.65 (n_refwd 321) |
| PER_PAIR_MS | wall 62.32 / fwd_only 62.15 | wall 61.75 / fwd_only 61.57 |

Raw-acceptance observation (b-run only): D3 50.0 → 39.2 (−10.8 pp) with
n_refwd 290 → 321 (+31) — the t1297 ULP-jitter signature at work (1–2
ULP verify logit jitter flips argmaxes → more draft mismatches → more
re-forwards); superseded as the speed/parity record by the canonical
table in CORRECTION. The original "same-session noise floor" claim is
not reproducible (run-1 raw log not preserved).

**Acceptance parity (b-run only):** D1/D2/D3 of rows-ON vs rows-OFF
differ only by the documented ULP-jitter signature (same direction and
class as t1297's −5.0 pp D3 on the plain-path comparison), no novel
degradation mode.

## First-divergence (data, not a gate)

- Greedy K=3, same session: B0b vs R1b diverge at **token 32** (t1297: 172). Divergence point varies across
  sessions — consistent with 1–2 ULP jitter, not a structural break.

## Sampling gate (distribution parity; temp 1.0, top-p 0.95, 4 seeds)

`mtp-sample-k3.py` (same S=4 verify loop, target token sampled instead of
argmax; per-step `mtp_cache` reset — under sampling the drafter's own KV
context accumulates rejected drafts and poisons subsequent drafts).

- **First divergence, same seed: token 2 in all 8 pairs (64- and 256-token
  runs).** Mechanism: prompt forward is rows-unaffected (N=40) so token 1
  is bit-identical; the first *verify* forward is rows-affected (N=248320
  M=4), and any 1-ULP logit jitter flips a categorical draw sitting near a
  cumulative-probability boundary → full cascade. The sampling path
  **amplifies** ULP jitter exactly as expected; this is why the
  statistical bar, not token-identity, is the correct bar for rows.
- **Token-id overlap, rows-ON vs rows-OFF (same seed):** 0.2–0.4 % set /
  0.4 % positional (256 tok); 0.8 % set / 1.6 % positional (64 tok).
- **Plain path's own noise band (within-arm, cross-seed):** 0.0–0.8 % set,
  ~0 % positional.
- **Distribution parity gate: not PASS on overlap evidence — the
  token-id-overlap gate is unresolvable in this chaotic regime.** All 8
  same-seed pairs diverge at token 2, so the overlap cannot
  discriminate: ON is not distinguishable from OFF beyond seed noise
  (the 0.2–0.8 % overlap sits inside the plain path's 0.0–0.8 %
  seed-noise band — parity at face value, not a gate). The
  distributional argument instead rests on (i) the direct 1–2 ULP kernel
  measurement (t1297 probe: maxdiff 9.54e-07 at N=248320 and
  7.15–9.54e-07 at N=12288, both shapes) and (ii) the acceptance-parity
  tables (this section's degenerate-sampling result; the greedy b-run D
  deltas above; the canonical D −1.5/−2.0/−0.9 pp with FIRST20
  bit-identical).
- **Acceptance parity (sampling): PASS (degenerate).** D1 0.0 % in both
  arms across 8 runs × 128 verifies. The MTP drafter (a small head on the
  target prenorm hidden) cannot track an off-manifold sampled stream; both
  arms are equally zero. Verifies still run S=4, so rows still engages.
- Sampling tok/s: OFF 31.85–32.15 vs ON 33.61–33.69 → **+5.3–5.8 %**
  (larger relative gain than greedy because the verify is a bigger share
  of the slower sampled step).

## CORRECTION (11-02 re-bench, canonical t1292 method) — op order: fix the
benchmarks to match the existing baselines, do not vary the method

The speed table in this report (50.58/50.89 tok/s, pair 62.09/61.57 ms,
n_refwd 290/321, D3 −10.8 pp) came from `bench/mtp-bench-k3.py` — the
t1228-v2 per-token re-forward variant (byte-identical to the guest copy
`rs-mtp-bench-k3.py`), **not** the 11-01 canonical script — and is
**RETIRED**. Its B0 arm did not
reproduce the t1292 baseline (50.58 / 62.09 ms / n_refwd 290 vs the
established 69.48 / 45.20 ms / n_refwd 101) while its same-session verify
wall matched (~25.5 vs 25.99 ms): the kernels are fine, the re-forward
accounting differs between script generations. The canonical script
(`mtp-bench-k3-tape.py`, the instrumented 128-verify A/B: 256-token
budget, 128 verifies, A3 = tape + S4VA, md5 `2c2d6c90…` pinned source
== guest) is the only method used for the numbers below; arm differences
are env vars only.

**B0 in-session reproduction of the t1292 baseline: PASS.** 69.95 tok/s
(+0.67 % vs 69.48, inside the ~3 % bar), pair wall 44.90 ms (vs 45.20),
verify 25.71 ms (vs 25.99), n_refwd 100 over 128 verifies (vs 101),
`TAPE n_ok=128 n_fb=0`. The method is validated before R1 is trusted.

### Canonical-method speed table (this wheel, `venv-rs`, 256-token budget,
128 verifies per arm, 2 runs each, same session, `TAPE n_ok=128 n_fb=0`
in all four runs)

| | B0 (rows OFF = A3) | R1 (rows ON) | delta (run 1 / run 2) |
|---|---|---|---|
| tok/s | 69.95 / 69.72 | 75.00 / 74.98 | **+5.05 / +5.26 (+7.2 % / +7.5 %)** |
| verify wall ms | 25.71 / 25.82 | 22.13 / 22.15 | −3.58 / −3.67 (−13.9 % / −14.2 %) |
| refwd wall ms | 9.22 / 9.22 | 9.31 / 9.31 | +0.09 |
| n_refwd (of 128) | 100 / 100 | 102 / 102 | +2 |
| pair wall ms | 44.90 / 45.05 | 41.35 / 41.36 | **−3.55 / −3.69 (−7.9 % / −8.2 %)** |
| D1 / D2 / D3 % | 70.3 / 62.2 / 50.0 | 68.8 / 60.2 / 49.1 | −1.5 / −2.0 / −0.9 pp |
| t1292 A3 reference | 69.48 | — | — |

Same-session noise floor (run 1 vs run 2, per arm): 0.03 % tok/s,
0.11–0.15 ms wall, 0 pp acceptance — the deltas above are all well above
noise. Vs the t1292 A3 baseline: B0 reproduces it (+0.67 %); R1 =
**+7.9 % tok/s vs 69.48**, pair 41.35 ms vs 45.20 (−8.5 % vs reference).

**Acceptance parity (canonical): PASS.** D deltas −1.5/−2.0/−0.9 pp with
n_refwd 100 → 102: a small, direction-consistent ULP-jitter signature —
materially milder than the retired script's −10.8 pp / +31 refwd
(read that as a property of the retired script's re-forward accounting,
not of the rows kernel). No novel degradation mode; FIRST20 bit-identical
across arms.

**Position vs the 100+ tok/s target: NOT MET, but the gap is ~10 ms, not
~30.** R1 = 75.00 tok/s, pair 41.35 ms vs target pair ≤ 31.4 ms.
Decomposition: verify 22.13 ms (rows −13.9 %) + single ≈ 9.68 ms +
refwd 9.31 ms/verify-cycle (n_refwd 102/128). The remaining gap is the
single-step + re-forward regime — out of scope for the rows branch.

**Standing as-is (not re-run, unaffected by the method change):** the
engagement evidence (1.58x at the K=3 verify shape) and the sampling gate
(`mtp-sample-k3.py`: +5.3–5.8 % tok/s; overlap gate unresolvable — all
8 same-seed pairs diverge at token 2, distributional argument = direct
1–2 ULP kernel measurement + acceptance parity; acceptance parity
degenerate-PASS in both arms). The greedy first-divergence data point
(@32, session-varied) is retired with the old script's numbers; under the
canonical method FIRST20 matches and the acceptance deltas above are the
parity record.

## Bench table (RETIRED — raw restatement; see CORRECTION above)

The table below came from the t1228-v2 per-token re-forward variant
(`bench/mtp-bench-k3.py`; re-forward accounting differs from the
canonical t1292 method). Retained for the record only; the CORRECTION
section's canonical-method table replaces it. Restated from the raw
printed lines of the surviving logs (guest `rs-B0b.log`, `rs-R1b.log`,
and the t1292 `inst_A3.log`) — no re-derivation. The run-1 values
(50.58/50.89 tok/s, verify 25.46/22.11 ms, pair 62.09/61.57 ms) have
no surviving raw log on the guest and are struck.

| config | TOKS | VERIFIES | TOKS_S | D1/D2/D3 % | verify ms | pair ms | n_refwd |
|---|---|---|---|---|---|---|---|
| t1292 A3 (reference, prior build, `inst_A3.log`) | 402 | 128 | 69.48 | 70.3 / 62.2 / 50.0 | 25.99 | 45.20 | 100 |
| B0b = A3 this session (rows OFF) | 402 | 128 | 50.39 | 70.3 / 62.2 / 50.0 | 25.66 | 62.32 | 290 |
| R1b = A3 + rows ON | 401 | 128 | 50.73 | 73.4 / 54.3 / 39.2 | 22.24 | 61.75 | 321 |

Raw `TOKS` count: total generated (decoded) tokens emitted over the run
— one per single-token step plus the 1–3 tokens per verify (accepted
drafts included, prompt excluded); `TOKS_S` = TOKS ÷ total generation
wall; the loop budget is 256 *iterations*, not 256 tokens. (The original
table's reference row read "n_refwd 101, D3 48.2 %"; the raw t1292
`inst_A3.log` prints n_refwd=100 and D3PCT=50.0 — those cells are struck
as inconsistent with the raw log.)

## Verdict (re-derived on the canonical-method numbers)

**PASS under the statistical bar; ship the gate (default off).**

- B0 (this wheel, canonical method) **reproduces the t1292 A3 baseline**
  (69.95 vs 69.48 tok/s, pair 44.90 vs 45.20 ms, n_refwd 100 vs 101,
  `TAPE n_ok=128 n_fb=0`) — the method is validated in-session before R1
  is trusted.
- Rows ON (canonical, both runs above noise): verify **−13.9/−14.2 %**,
  pair **−7.9/−8.2 %**, tok/s **+7.2/7.5 %** vs B0, **+7.9 %** vs the
  69.48 baseline; n_refwd 100 → 102, D deltas −1.5/−2.0/−0.9 pp —
  acceptance parity holds with a small ULP-jitter signature, FIRST20
  bit-identical across arms. (The retired script's −13.2 % verify /
  −0.8 % pair / D3 −10.8 pp numbers are superseded: the method, not the
  kernel, was the difference.)
- Engagement proven (1.58x at the K=3 verify shape, SASS-verified on the
  installed .so) — stands as-is.
- Distribution parity (sampling gate): the token-id-overlap gate is
  unresolvable (all 8 same-seed pairs diverge at token 2; ON not
  distinguishable from OFF beyond seed noise) — not a PASS on overlap
  evidence; the distributional argument rests on the direct 1–2 ULP
  kernel measurement (t1297: maxdiff 9.54e-07, both shapes) + the
  acceptance-parity tables — stands as-is; sampling +5.3–5.8 % tok/s
  stands as-is.
- 100+ tok/s target **not met** in this config: R1 = 75.00 tok/s,
  pair 41.35 ms vs target pair ≤ 31.4 ms (gap ≈ 10 ms; the remainder is
  the single-step + re-forward regime, out of scope for the rows
  branch).
- Gate `MLX_ROCM_QMV_ROWS` defaults OFF in the wheel; ON is opt-in.

## Operational notes

- GPU lock window 02:08–02:30Z (this task). t1305's 02:13 pip-install
  phase was lock-blocked by this task's flock and released at 02:30:06Z —
  flagged to warden; t1305 unblocked, its run continues.
- Scripts uploaded to guest: `mtp-sample-k3.py`, `qmv_rows_engage.py`;
  the non-canonical bench run used the guest copy of `bench/mtp-bench-k3.py`
  (the t1228-v2 per-token re-forward variant, `rs-mtp-bench-k3.py` —
  **not** the 11-01 canonical script `mtp-bench-k3-tape.py`). Logs
  `rs-B0b/rs-R1b/rs-samp-*/rs-s64-*.log` on guest (only the b-run logs
  survive; the run-1 logs are not preserved).
- 11-02 re-bench: canonical `mtp-bench-k3-tape.py` (md5 `2c2d6c90…`,
  source == guest copy) + `gdn_tape_commit.py` (md5 `4a262a0d…`) in a
  fresh guest dir; wheel `2442e063…` re-verified on the guest before
  running; sidecar hash-pin (only `model-mtp.safetensors` active)
  confirmed. Ran after t1305 reached terminal (02:54Z). Logs
  `B0/R1/B0b/R1b.log` on guest; flock held per arm, released after each.

## FIX ROUND (read run finished 02:56Z @ 3615777: findings 1–5 → done)

1. **TABLE ARITHMETIC (major)** — the retired t1228-v2 table was
   internally inconsistent: "402/401 tokens @ 128 verifies" was attached
   to both runs of each arm, but only the b-run raw logs survive on the
   guest (`rs-B0b.log`, `rs-R1b.log`; a guest-wide search found no
   surviving log for the run-1 values 50.58/50.89 / 25.46/22.11 /
   62.09/61.57). Both retired-table sections were restated from the
   script's raw printed lines only (TOKS, VERIFIES, TOKS_S,
   D1PCT/D2PCT/D3PCT, WALL_MS, PER_PAIR_MS) — no re-derivation — with a
   note on what the raw `TOKS` count includes: total generated (decoded)
   tokens emitted over the run (one per single-token step plus the 1–3
   tokens per verify, accepted drafts included, prompt excluded;
   `TOKS_S` = TOKS ÷ total generation wall; the loop budget is 256
   *iterations*, not 256 tokens). The run-1 values are struck as
   unverifiable. The canonical-method table (69.95/75.00) stands
   untouched. Raw-log cross-check: the t1292 reference row's "n_refwd
   101, D3 48.2 %" does not match the raw `inst_A3.log` (n_refwd=100,
   D3PCT=50.0) — corrected in the retired table; the CORRECTION
   section's "vs 101" wording was left untouched per the fix scope and
   is flagged here for the orchestrator.
2. **PROVENANCE (major)** — CORRECTION now states the canonical script
   is `mtp-bench-k3-tape.py` (md5-pinned) and `bench/mtp-bench-k3.py`
   is the t1228-v2 per-token re-forward variant, not the 11-01
   canonical script (branch file byte-identical to the guest
   `rs-mtp-bench-k3.py`). The stale operational-notes line
   "(bench mtp-bench-k3.py = t1292 script, unmodified)" was corrected.
3. **DISTRIBUTION GATE WORDING (major)** — the token-id-overlap gate is
   degenerate (all 8 same-seed pairs diverge at token 2). Restated as:
   "not distinguishable beyond seed noise; the overlap gate is
   unresolvable in this chaotic regime" — no longer called PASS on
   overlap evidence. The distributional argument now rests on the direct
   1–2 ULP kernel measurement (t1297 probe: maxdiff 9.54e-07 at
   N=248320, 7.15–9.54e-07 at N=12288) + the acceptance-parity tables
   (sampling degenerate-PASS both arms; greedy b-run D deltas; canonical
   D −1.5/−2.0/−0.9 pp, FIRST20 bit-identical). Verdict bullet
   updated to match.
4. **LINT (major)** — `uv run ruff format .` (3 files reformatted) +
   fixed all 25 `ruff check` errors in `bench/mtp-bench-k3.py`,
   `bench/mtp-sample-k3.py`, `bench/qmv_rows_engage.py` (E501,
   E741 x10, F841 x6, E402, I001, UP031 x4). Behavior-neutral: local
   `l`→`lgrow` renames, unused unpacks → `_`, import move (E402) and
   sort (I001), long lines wrapped, % -format → f-string with
   byte-identical print output. No semantic touch was required. End
   state: `uv run ruff format --check .` clean, `uv run ruff check .`
   clean (0 errors), `uv run pytest -m 'not gpu'` 22 passed / 1 skipped,
   size gate OK.
5. **GATE CODE IN BRANCH (minor)** — the rows dispatch/gate code (reads
   `MLX_ROCM_QMV_ROWS` / `MLX_ROCM_QMV_ROWS_MIN_N`) was only in the
   guest vendor layer. Extracted read-only from the guest vendor-rs
   delta (`git diff` of `mlx/backend/rocm/quantized/qmm.hip` vs fork
   base `75ca881`, guest file md5 `4dec02b00e3c750d683fbc0dc6ce8231`)
   and committed as `patches/qmv-rows-dispatch-gate.diff` (the
   rows-gate dispatch block as a zero-context insertion hunk anchored
   after the host_warp `else {`, plus the rows-kernel `#include`
   hunk). With `patches/qmv-rows-msg-kernel-wsfix.diff` the kernel +
   gate are now source-verifiable in the branch; README note added.
   The committed vendor tree is the build base per project convention;
   the guest layers are the delta.
