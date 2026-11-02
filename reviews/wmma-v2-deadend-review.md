# Adversarial review — REPORT-wmma-m4-v2 dead-end verdict (fresh session, CPU-only)

Target: the claim in REPORT-wmma-m4-v2.md (worker/wmma-m4 @2d15d4b): the WMMA M=4
QMV kernel (v1 + v2 multi-wave + v3 barrier-free) is a DEAD END as a lever for
the K=3 MTP verify GEMM — correct but loses to the scalar rows kernel, not
shipped. Method: read all three reports (v2, v1, gemms-m4) in full, read the
actual kernel + dispatch in `patches/qmv-wmma-m4/` and the vend tree
(`vendor/mlx-rocm-upstream/`), cross-check every quantitative claim, stress-test
the verdict per the brief. No GPU available (lane occupied); unresolvable
items are marked `unresolved-needs-gpu` with the settling micro-bench.

## 1. Code-vs-report verification (kernel + dispatch)

All structural claims check out against `patches/qmv-wmma-m4/qmv_wmma_m4_kernel.hip`
(the in-repo kernel is the v3 variant: no barriers, direct acc->global store)
and the two dispatch diffs:

- Tile config: template `<T, ST, GROUP_SIZE, WAVES>`, `__launch_bounds__(32*WAVES)`,
  wave g owns n-tile `blockIdx.x*16*WAVES + g*16`, one 16-column n-tile x all M
  rows per wave. Matches the report.
- LDS staging: `__shared__ __half A[WAVES][2][16][16]` + `B[WAVES][2][16][16]`
  = 2048*WAVES bytes = 2/4/8 KB for 1/2/4-wave. Matches "2/4/8 KB LDS".
- Per-chunk pipeline: each 32-k-chunk iteration = stage 16 x values + dequant 16
  weight values (scalar `__float2half(s*w4+b)`) into LDS per thread, 4
  `load_matrix_sync`, 2 `mma_sync` (ILP=2), next-chunk global prefetch into
  registers. Exactly the "per-32-k-chunk scalar dequant + LDS staging" pipeline
  the report blames. At K=4096: 128 such iterations (256 16-wide tiles — the
  report's "256 chunks/GEMM" counts 16-wide tiles, see finding F6).
- 12 dead rows: A tile rows 4..15 are zero-filled (`load_x16` returns zero for
  `n >= M`); the 16x16x16 mma still computes all 16 rows; store loop writes only
  lanes 0..15, elements 0..M-1 (acc layout lane L, element e -> row
  8*(L>>4)+e, col L&15). 12 of 16 accumulator rows dead at M=4: confirmed, and
  structural (section 2).
- v3 barrier removal: A/B LDS indexed per wave `g`; a wave is a SIMD lockstep
  unit, so the removed barriers order nothing between threads; direct
  accumulator->global store matches the code. Sound as written.
- Dispatch (v2 diff): `MLX_QMV_WMMA=1` (default off), `M == 4` only,
  N%16==0, K%16==0, f32 x declines; `MLX_QMV_WMMA_WAVES` 1/2/4 (default 2)
  with power-of-2 fallback dividing N/16. So `MLX_QMV_WMMA=1` engages every
  M=4 GEMM of the model (all N, K are 16-multiples) — the verdict's scope
  "WMMA for all M=4 GEMMs" is exactly what was gated and benched.
- Provenance note: only the v3 kernel source is preserved in `patches/` (v1 and
  v2 kernel variants are reachable only via git history); the v2 report says
  "v2/v3 kernel + dispatch kept in patches/" (F9, minor).

## 2. Hardware: is the 16-row M fragment structural on RDNA4 (gfx1201)?

**Yes — structural to the instruction, not a tiling choice.** Cited in-repo
evidence:

- The only AMDGCN WMMA builtin used anywhere in the vend tree is
  `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`
  (`vendor/mlx-rocm-upstream/mlx/backend/rocm/r4d_gdn_wmma.h:130` — the `gfx12`
  builtin, i.e. gfx120x/RDNA4); a repo-wide sweep of wmma builtins finds only
  the 16x16x16 shape.
- The tested kernel uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
  (kernel header) and REPORT-wmma-m4 §4 is SASS-verified:
  `v_wmma_f32_16x16x16_f16` present in the compiled kernel.
- `flash_attention_wmma.hip` (vend tree, comments at ~L106 and ~L524):
  "A rocWMMA 16x16x16 fragment op is executed collectively by one full
  hardware wavefront ... 32 lanes on RDNA (gfx10/11/12)", "One full hardware
  wavefront per 16-row M-tile" — the whole codebase tiles M in 16-row
  increments; no 8- or 4-row fragment op exists on RDNA (unlike CDNA MFMA or
  NVIDIA's m16n8k16 HMMA variants).

Consequence: there is NO untested M tiling. Every WMMA-based M=4 path must
compute 16 rows and discard 12 (75% of mma throughput wasted), and no packing
escape exists (a single mma carries one K-slice x one N-tile; the 16 A-rows are
the only place to put x-rows, and 4 live x-rows + 12 zero rows is the only
M=4 packing). Dead rows = 12/16 structural. This makes the dead-end case
stronger: "the config that was never tested" does not exist for the M
dimension. (N tiling: 16-wide per mma is fixed; 1/2/4 waves-per-block were all
tested — "N=32 per wave" is arithmetically the 2-wave block.)

## 3. Confounds

- **B0 baseline mismatch (this session 54.47 pair vs gate 44.74-45.20 / 69.48
  tok/s):** the report explicitly flags the wheel-base difference and makes the
  within-run A/B the decision basis — correct methodology. The mismatch biases
  absolute-vs-gate comparisons only, not the within-run relative comparison
  (same wheel, same env, all arms). Corroboration across bases: the v1 session
  (different wheel, B0 44.70/44.97 — gate-consistent) shows R1 (rows+wmma)
  33.69-33.90 vs ROWSONLY 22.25 = +51..53% verify; the v2 session shows
  +55..68%. Same sign and same order across two independent bases/wheels:
  the relative result is base-robust.
- **Clock/thermal drift:** a sustained 55-68% gap across 3 arms x 2 sessions x
  3 benchmark contexts (canonical A/B, fwd4, 249-GEMM stream) is far outside
  drift (canonical-record noise floor: 0.03% tok/s, 0.11-0.15 ms between
  back-to-back runs). The one anomaly — canonical pair non-monotonic in wave
  count (1w 81.72 > 4w 71.46 > 2w 67.61) — suggests ±10-15% run-to-run
  variance, which the report's "monotonic" wording overstates (F3); it cannot
  close a >=55% gap (rows 30.94 vs worst-wmma 47.82).
- **fwd4 vs canonical consistency:** consistent (both show rows < 32 ms, wmma
  1-wave 43-46 ms, penalty growing with waves in fwd4 and in the micro
  stream); no arm where wmma wins in one context and loses in the other.
- **n_ok/TAPE_CALLS:** the v2 report states tape engaged "n_ok=128 n_fb=0,
  TAPE_CALLS~5496" for the canonical runs but does not tabulate per-arm counts
  (F7, minor); n_fb=0 and n_ok=128 match the canonical convention (128/128).
- **Arm ordering:** B0 -> ROWSONLY -> 1w -> 2w -> 4w sequential; the non-
  monotonic canonical result (2w beats 1w) breaks a pure time-drift
  explanation; fwd4 rows arms are stable (31.54-31.60) while wmma arms move
  across builds, i.e. the ordering effect is not the driver.

No confound explains the loss; the claimed mechanism (per-chunk dequant + LDS
staging pipeline at K=4096 + multi-wave co-scheduling) is coherent with the
code and with the K=512-vs-K=4096 split, and is independently corroborated by
two sessions.

## 4. Counter-options

(a) **Hybrid dispatch (wmma for the short class + rows for K>=4096): not shown
to be a net win by in-repo evidence — neutral-to-negative, unresolved.**
- The v1 N-gate bisection (same session that measured the isolated short-class
  win): min_n=249000 (largest short-class call, lm_head) -> 31.59 vs 31.63 =
  NEUTRAL; min_n=29000 (+N=8192/12288, 96 calls, all the short-K class) ->
  +4.2 ms REGRESSION. Even in the session that reported the isolated win, the
  in-stream bisection shows the win does not materialize as a stream win.
- The only stream-level wmma win on file (v2: "K=512 chained 13.21 vs 14.56",
  "GDN co-schedule 5.48 vs 6.48") is a short-K-only synthetic stream, not the
  model stream; the 249-GEMM real-mix stream (the model-relevant one) shows
  rows 19.82/30.74 vs wmma 33.65-41.34.
- The v2 dispatch has no K/N-class gate (only wave count), so the true
  hybrid (wmma only for N in {32, 1024, 248320}; rows for N>=4096) was never
  tested. Given the v1 isolated win is ~5-15 us on ~65 small calls (and the
  bisection shows the largest such call is stream-neutral), the expected
  upside is ~0-1 ms on a ~31 ms forward — below the demonstrated run variance
  and not a demonstrated net win.
- **Settling micro-bench (unresolved-needs-gpu):** 249-GEMM real-mix stream +
  fwd4, ONE wheel, 10-iter medians, arm order balanced, arms: (i) rows-only,
  (ii) hybrid — wmma only for N in {32, 1024, 248320}, rows for N>=4096,
  (iii) all-wmma 1-wave. Pass bar for the hybrid: (ii) beats (i) by >= 1 ms on
  BOTH the stream and fwd4. Report the per-shape isolated table in the same
  wheel to close the session disagreement of finding F1.

(b) **Non-16-row M tiling / different N tiling:** ruled out by the hardware
  (section 2: 16x16x16 is the only RDNA4 WMMA shape); WAVES 1/2/4 all tested;
  nothing untested remains on the tiling axis.

(c) **The report's "next lever" (M=4-native fragment path):** NOT a genuinely
  different instruction class — it is the same dead-row problem in different
  clothing. "Two n8 tiles per wave" and "a WMMA kernel that skips the 12 dead
  accumulator rows" both presuppose a fragment op smaller than 16 rows, which
  does not exist on RDNA4 (section 2 citations). An mma always computes all 16
  rows; "skipping" dead rows is not expressible. The only M=4-native path is
  scalar-FMA custom — which is exactly what `rows` already is. This makes the
  verdict stronger: "accept rows for M=4" is the correct branch of the report's
  own dichotomy (F2, minor — the next-lever wording should say so).

## 5. Findings

- **F1 (major, unresolved-needs-gpu) — cross-session contradiction on the
  isolated (N=12288, K=4096) shape; the GEMM-inventory K column is mislabeled.**
  REPORT-gemms-m4 §4 lists 217 of 249 GEMMs at "K=512"; the only consistent
  reading is that the K column is 4-bit packed words per row (512 words =
  K=4096; 1536 words = K=12288), corroborated in-repo by
  reports/mtp-s62-layerdiff.md ("in_proj_qkv ... (4096 -> 8192)") and by the
  64-count N=12288 class = MLP gate+up (2/layer x 32 layers). Under that
  reading, v1's isolated "K=512 N=12288: rows 0.067 vs wmma 0.051 (wmma
  wins 24%)" (REPORT-wmma-m4 §4) and v2's isolated "K=4096 N=12288: rows 142
  us vs wmma 237 us (1.67x loss)" (REPORT-wmma-m4-v2 §mechanism) are the SAME
  shape measured 4.6x apart across sessions (rows also moved 2.1x: 0.067 ->
  0.142). Why it matters: the short-K "win" is the only pro-hybrid datum on
  file, and if "K=512" = true K=4096 (the model's dominant class), v2's own
  measurements contradict its "wmma wins K=512" bullet. It does NOT overturn
  the verdict: every in-stream datum in both sessions (canonical A/B, fwd4,
  249-GEMM stream, v1 N-gate bisection) agrees wmma loses. Settled by the
  finding F4 bench (isolated (N=12288, K=4096) and (N=12288, K=512) on one
  wheel) plus fixing the inventory K label.
- **F2 (minor) — "next lever" wording overstates the option space.**
  REPORT-wmma-m4-v2 Disposition proposes "an M=4-native path ... two n8 tiles
  per wave, or a WMMA kernel that skips the 12 dead accumulator rows". On
  RDNA4 no sub-16-row WMMA shape exists (section 2); both proposals are the
  same dead-row problem in different clothing. The lever that exists is
  non-wmma scalar (rows already is that). Verdict-strengthening, not a hole.
- **F3 (minor) — "monotonically worse as waves per block grow" is overstated
  for the canonical bench.** Canonical pair: 1w 81.72 > 4w 71.46 > 2w 67.61
  (verify: 50.89 / 51.77 / 47.82) — non-monotonic, consistent with ±10-15%
  run-to-run variance; the monotonic claim holds only for fwd4 and the 249-
  GEMM micro stream. No effect on the verdict (all wmma arms lose by >=55%).
- **F4 (major, scope — open counter-option; does not FAIL the verdict) —
  hybrid dispatch was never tested and the in-repo evidence for it is
  neutral-to-negative.** See section 4(a). The v2 dispatch has no K/N-class
  gate; the v1 N-gate bisection is the closest in-stream test and shows
  neutral (min_n=249000) to regression (min_n=29000, +4.2 ms). Settling
  micro-bench: as specified in section 4(a).
- **F5 (minor) — "N=12288 in_proj class (64 calls)" mislabeled.** In-repo
  evidence (reports/mtp-s62-layerdiff.md: in_proj_qkv is 4096 -> 8192) says
  the 64-call N=12288 class is the MLP gate+up, not in_proj. The dominance
  arithmetic (+95 us/call x 64 ~ 6.1 ms of the 7.1 ms micro delta) is
  unaffected — the other losing classes add ~0.6 + 0.9 ms (11 us x 56 + 28 us
  x 32), so "essentially the full stream delta" still holds (~7.6 vs 7.1).
- **F6 (minor) — "256 chunks/GEMM" conflates chunk sizes.** K=4096 = 256
  16-wide tiles = 128 per-32-k-chunk iterations (the loop period). Wording
  only; the mechanism claim (per-chunk scalar dequant + LDS staging) is
  exactly what the code does.
- **F7 (minor) — tape engagement asserted, not tabulated per arm.** "n_ok=128
  n_fb=0, TAPE_CALLS~5496" is a summary; per-arm tape counts are not shown, so
  per-arm n_ok equality is asserted rather than evidenced. Consistent with the
  canonical convention (128/128) and with n_fb=0; not a hole.
- **F8 (minor, style) — hard-wrapped prose: 90 hits** (68 in the four report
  .md files in the range, 20 in the five new commit messages). Style only.
- **F9 (minor) — provenance.** Only the v3 kernel source is in
  patches/qmv-wmma-m4/; the v2 report's "v2/v3 kernel + dispatch kept in
  patches/" is accurate only for v3 (v1/v2 kernel variants live in git
  history). Record-keeping only.
- **Context note (informational):** the brief's canonical "verify 17.41 ms"
  does not appear in the in-repo canonical record
  (worker/rows-statistical @91d7df5: rows ON verify 22.13 ms / B0 25.71 ms /
  pair 41.35 ms / 75.00 tok/s — the 75.00/41.35 match). The v2 session's
  ROWSONLY 30.94 vs 22.13 is explained by the (flagged) wheel-base mismatch,
  not used in the relative verdict.

## 6. Check lines

- Format: `uv run ruff format --check .` — **PASS** (90 files already formatted, 0 changes).
- Persona-leak: grep of `main..2d15d4b` — full diff, all five commit
  messages, and all 10 touched files, for the seven pattern families named in
  the task brief (the fork-hostname family, the 100.64/10 IP family, the
  three username forms, the address form, the 192.168.122 LAN block, the
  model-mount path, and the junkyard shell-file suffix; git author lines
  exempt) — **PASS, 0 hits** in all three scopes.
- Line-wrap: report .md files in the range + new commit messages (docs scope)
  — **90 hits** (REPORT-gemms-m4.md 28, REPORT-wmma-m4-v2.md 12,
  REPORT-wmma-m4.md 28, SALVAGE-NOTE-1339.md 2, commit messages 20).

## 7. Persona scrub of this review

Scrubbed this file against the same seven pattern families from the brief
before commit; the check line above is written descriptively so the file does
not self-match its own scrub patterns. Final scrub: 0 hits, 0 lines changed
(scrub count: 0).

## Verdict

The dead-end claim is sound. It rests on in-stream evidence from two
independent sessions (three wheel bases, three benchmark contexts) that is
consistent in sign and order (rows 22.25-31.60 vs wmma 33.7-54.8 ms, a
sustained +41..68% penalty across all wave configs), on a mechanism that
matches the actual kernel code (per-32-k-chunk scalar dequant + LDS staging
with 12 structurally dead accumulator rows out of 16 — a structural property
of the only WMMA shape RDNA4 exposes, so no untested tiling remains), and on
a correctness record (unit 24/24, maxdiff 0.007812, A/B 200/200). The one
disputed datum (the isolated short-K "win", F1) is contradicted by v2's own
same-session measurements and by v1's own in-stream bisection, and the
untested hybrid counter-option (F4) has neutral-to-negative in-repo evidence
with a specified settling bench — a scope finding, not a hole in the verdict.

VERDICT: PASS
