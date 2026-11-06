# REPORT: var-diff, record 13-var COR vs t2213 battery env (cloudy task 1, 00:5xZ)

Doc task, no GPU. Sources: t2027 REPORT-fold-qmm-hybrid.md (the 69.43 gate: 13-var verbatim list + the bench md5), the on-box battery script <box-home>/t2190-oobfix-battery.sh (t2213's PRE/POST gate, 47.05/47.1 t/s arms), the on-box bench <box-home>/m3k/bench/mtp-bench6.py (md5 54ce16ac29e6c6aad2c48a233ff70d85 = THE RECORD BENCH, same file as the 69.43 gate), and the wheel identity check below.

## Verdict: DIFFERENT CONFIG. The t2213 "COR-verbatim" battery is NOT the record's config.

### Side-by-side (record 13-var verbatim vs t2213 battery effective env)

| var | record (t2027, 69.43) | t2213 battery (47.05) | status |
|---|---|---|---|
| MLX_MTP | **3** (K=2, S=3 verify) | **4** (K=3, S=4 verify) | **DIFFERS - the big one** |
| MTP6_TAPE | hybrid | hybrid | same |
| MLX_ROCM_GDN_TAPE | 1 | unset -> bench sets 1 internally (tape mode) | same (mechanism, not env) |
| MLX_ROCM_GDN_T3 | 1 | unset -> bench sets 1 internally (tape mode) | same (mechanism, not env) |
| MLX_ROCM_GDN_FUSE | (not in list; bench sets 1 internally) | 1 (explicit) | same (mechanism) |
| MTP6_HEAD_H | postnorm | postnorm | same |
| MTP6_DRAFTGATE | 1 | unset (bench does NOT read it; code default unknown) | **DIFFERS (if code default != 1)** |
| MTP6_DGATE_T0 | 16 | unset | **DIFFERS (if code default != 16)** |
| MTP6_NGRAM | 1 | unset (bench reads MTP6_NGRAM_S3, default 0 - name mismatch, see note) | **DIFFERS or INERT (see note)** |
| MTP6_NGRAM_N | 4 | unset (bench default 4) | same (if inert) |
| MLX_ROCM_SDPA_S4VA | 1 | 1 | same |
| MLX_ROCM_QMV_ROWS | 1 | 1 | same |
| MLX_ROCM_REFWD_M1 | 1 | 1 | same |
| MLX_QMV_MR_LX | 0 | 0 | same |
| MLX_ROCM_QMM_M3 | 1 | 1 | same |
| MLX_ROCM_QMV_ROWS_MIN_N | (not in list; bench has no setdefault - grep-clean) | 1 (explicit) | **DIFFERS (battery extra)** |
| MTP_MAX_TOKENS | 322 | 322 | same |

Notes: (1) The bench (md5 54ce16ac, the record bench) internally sets GDN_TAPE=1/GDN_T3=1/GDN_FUSE=1 under tape modes, so those are same-mechanism, not env differences. (2) MTP6_NGRAM vs MTP6_NGRAM_S3: the record's list names MTP6_NGRAM=1 but the record bench reads MTP6_NGRAM_S3 (default 0). If the model code does not read MTP6_NGRAM, it was INERT in the record too - either way the two arms of the upcoming dual-venv A/B get the IDENTICAL env (the record list verbatim in both), so any effect is symmetric and cancels. (3) QMV_ROWS_MIN_N=1 is the battery's extra: cloudy's "rows kernel byte-grep-absent from the 8acf13fe wheel -> should be a no-op" stands as a claim to verify, not a fact.

### What the config difference explains - and does not

- MLX_MTP 4 vs 3 changes the verify matrix S=4 vs S=3. t2207 microbench (M=4/M=3) time ratio = 1.33, measured. That accounts for ~1.33x of the 2.16x per-step gap (cloudy's framing: 68.1 vs 31.59 ms).
- 2.16 / 1.33 = ~1.6x REMAINS UNEXPLAINED by the named config deltas. That residual is the actual regression question, and it is now measurable same-session (below), because:

## WHEEL IDENTITY (fresh check, 00:5xZ): the record wheel is ALIVE on the box

- venv-m3k libmlx.md5 = a0057c220d31a0ec87a8b373a97625e8, dist tag mlx-0.32.3.dev20261104+75ca881. t2027's report names venv-m3k a0057c22 as the wheel the 69.43 gate ran on. -> the record's libmlx is on the box.
- venv-verify (lane wheel) libmlx.md5 = 8acf13fec6fc1636026d8e1dc69cd497, SAME dist tag 75ca881. Same tag, different libmlx - the tag does not identify the build.
- e9a1b89d (the "record wheel gone" claim, propagated from t2212): NO file, wheel, or venv carrying e9a1b89d found on the box (<box-home>/wheelout = one Nov-1 +75ca881 wheel; find <box-home> maxdepth 2 = nothing). The identity was a tag confusion; "69.43 unverifiable" is PREMATURE.
- Consequence: the stage-1 binary (cloudy) is answerable in ONE same-session battery: true 13-var verbatim env, bench 54ce16ac, MTP_MAX_TOKENS=322, run on BOTH venvs - venv-m3k (record wheel, as-is, md5-verify pre/post, do not modify) and venv-verify (lane wheel). If venv-m3k reproduces ~69.43: record VERIFIED (not a memory) and the wheel delta is quantified same-session; the 1.6x residual becomes a file-level diff between two live wheels (name suspects, no bisect in-run). If venv-m3k does NOT reproduce: the record is a memory and the box itself moved - report that, stop.
