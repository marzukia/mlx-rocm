# MTP R3 — REFWD_M1 + GDN-tape A/B on the canonical 512 protocol

**Verdict (up front): the 65-class does NOT reproduce on the canonical 512 protocol. The 512-protocol number is the 40.4-class, and it is wheel-independent (A ≡ B ≡ C, 9/9 bit-identical token streams). NO PORT — the REFWD_M1 hunk is not ported into the canonical vendor tree, per the task's stop branch.**

Same session, same box (neon, RX 9070 XT), same wheel (93810e87): the 128- protocol graft protocol returns 65.69 t/s (anchor re-run, below) while the canonical 512 protocol returns 40.38 t/s. The gap is ms-per-step (49.0 → 78.8), which tracks protocol length (KV growth), not acceptance (flat 3.16–3.22 tok/step) and not the wheel.

## Protocol note (bench md5 discrepancy, resolved)

The task pinned bench `452d363` (plain canonical 512 bench) and requested `TAPEMODE=hybrid`. The 452d363 bench on the box (and the vendored copy under `vendor/`) has **no tape switch** — `MTP6_TAPE=hybrid` exists only in the round-2 bench `700f1810207ccf8837be58f81c5eb944`, which is bit-identical to the repo's `bench/mtp-bench6.py` on this branch. That bench's `MTP6_TAPE=0` branch is exactly the 452d363 protocol (round-2 A0n = PRE bit-repro, 37.43/1623), so the round-2 bench is the correct instrument: `MTP6_TAPE=hybrid` runs the hybrid tape path and `MTP6_TAPE=0` reproduces the plain canonical path.

**v1 session mistake, corrected in v2 (numbers of record):** the first session ran `MTP_MAX_TOKENS=512` → 256 verify steps (the 46.4-class, 808 tokens). The canonical "512" runs (38.94 tape / 37.47 savestate, 1623–1657 tokens) are **512 verify steps**, i.e. the bench's default `MAX_TOKENS=1024`. v2 re-ran A/B/C at the default (512 verifies, 1628 tokens). v1 numbers are tabulated only as a protocol-length datapoint.

## Regime

- `flock ~/gpu-bench.lock` held for every run; 25 s idle before and between runs.
- DPM: non-root sysfs read is EPERM on this box (worse than the battery's EBUSY); the v2 root wrapper had a quoting bug, so the force-level was never read/written this session. **Root sampler instead:** sclk under load ramps to ~3.30 GHz (table max) and drops to 0 MHz in the 25 s idle gaps — exactly the t1508 battery pattern under which 65.68/65.64 were bit-reproduced. Per the battery's documented caveat, FIRST20 + timing band are the acceptance criterion.
- **FIRST20 = bit-identical to the 65-class anchor line in 10/10 runs** (9 A/B/C runs + the 128 anchor). Token streams: 9/9 A/B/C runs bit-identical (single md5). SBCAR `model-mtp.safetensors` md5 `02a54359…` pre/post every session.
- Wheels pinned: venv-ra `libmlx.so` `93810e871f7bc977e723d5288a3b73e2` (= canon `b1de59fda0e4fc607d620e287deeaaf8` + 456 bytes, i.e. 75ca881 + REFWD_M1); venv-canon `b1de59fd…`.
- Heads: both benches use the full bf16 `lm_head` (the 128-graft bench is bf16 by code; the 512 bench prints `INT2=off` because `mx.fast.lm_head_int2` is absent from the wheel). No head lever gap between the two protocols.

## Numbers (3-run medians, 512 verifies, `MLX_MTP=4 MTP6_TAPE=hybrid`,
`GDN_FUSE=1 QMV_ROWS=1 QMV_ROWS_MIN_N=1`)

| arm | venv | REFWD_M1 | t/s | tokens | verifies | D1/D2/D3 % | REFWDS | FIRST20 |
|---|---|---|---|---|---|---|---|---|
| A | ra | 1 | **40.38** (40.40/40.38/40.34) | 1628 | 512 | 72.9/61.9/57.1 | 395 | OK (3/3) |
| B | ra | 0 | **40.37** (40.37/40.35/40.38) | 1628 | 512 | 72.9/61.9/57.1 | 395 | OK (3/3) |
| C | canon | 1 (no-op on canon wheel) | **40.37** (40.37/40.41/40.36) | 1628 | 512 | 72.9/61.9/57.1 | 395 | OK (3/3) |

**A−B = +0.02 %; A−C = +0.02 %.** (RUN_C fired because |A−B| < 10 %; C confirms the wheels are indistinguishable on this protocol.)

v1 session (256 verifies, same arms): A 46.37 / B 46.28 / C 46.35, 808 tokens, bit-identical streams — retained as the 256-length datapoint only.

## Phase-timer delta decomposition (the stop-branch deliverable)

512 hybrid+rows phase arm (`mtp-bench6-phase.py` 608a7770, `MTP6_TAPE=hybrid MTP6_MTPTRIM=1`, venv-ra, REFWD_M1=1): 512 verifies, D1/D2/D3 373/231/132, TAPE_CALLS 21408, INT2=off, FIRST20 anchor OK. Cumulative over the 40.3 s generate window:

| phase | ms | % | ms/step | what it is (lazy-eval caveat, battery §1.4) |
|---|---|---|---|---|
| prefill | 5.3 | 0.0 | — | (outside window) |
| draft | 14773.3 | 36.6 | 28.85 | 3× MTP draft forwards + t-argmax (evaluates the **previous step's S=1 correction forward** here) + 3× head argmax |
| verify | 277.3 | 0.7 | 0.54 | S=4 verify graph construction only |
| commit | 25102.2 | 62.2 | **49.03** | S=4 verify forward eval + tape rebind (m_ok, n_fb=0 class) |
| correction | 186.3 | 0.5 | 0.36 | S=1 correction forward construction only |

**78.8 ms/step total** → 1628/512 = 3.18 tok/step → 40.4 t/s.

Comparison, same wheel/regime/head:

| protocol | verifies | ms/step | t/s | tok/step | D1/D2/D3 % |
|---|---|---|---|---|---|
| 128 graft (t1508 battery R3, clean synced phases) | 128 | 49.0 (9.73 single + 29.96 verify + 9.09 refwd + 0.2) | 65.68 | 3.219 | 74.2/64.2/47.5 |
| 128 graft (same-session anchor re-run) | 128 | 48.9 (6.27 s/128) | **65.69** | 3.219 | 74.2/64.2/47.5 |
| 256 (v1 session) | 256 | 68.1 | 46.37 | 3.156 | 71.5/61.7/51.3 |
| **512 canonical (v2, numbers of record)** | 512 | **78.8** | **40.38** | 3.179 | 72.9/61.9/57.1 |

**Reading:** tok/step is flat (3.16–3.22) and the D*PCT tiers are comparable across all four — acceptance is not the delta. ms/step is monotone in protocol length: 49.0 → 68.1 → 78.8. The 128-vs-512 split: verify 29.96 → 49.03 (+19.07) and draft-window 18.82 → 28.85 (+10.03). Both benches run the same wheel/regime, the same `return_hidden=True, return_prenorm=True` S=4 call, the same bf16 head; the 128 bench syncs per phase, the 512 bench is lazy (battery §1.4 caveat). The dominant candidate is KV-length scaling of the S=4 verify attention (average KV ~2–3× larger over a 512 run than a 128 run), with the lazy-eval spillover structure as a secondary contributor; isolating the two needs a controlled KV-length A/B, which is outside this task's scope. Flagged as an estimate.

## Wheel verdict (exoneration)

- `venv-canon` (75ca881 wheel, `b1de59fd`) = 40.37 = `venv-ra` (93810e87) with REFWD_M1 on (40.38) and off (40.37), bit-identical token streams across all nine runs.
- Per the graft source (t1391 gate, lines 688–700), with `QMV_ROWS=1` the `REFWD_M1` M=1 lx re-route is off by design, so the 456-byte patch is an inactive variable in this regime — the zero A−B/C delta is the expected behavior, not a measurement artifact.
- The 65.68 (128 protocol) vs 40.4 (512 protocol) gap is therefore a protocol/stack property, not a wheel property. **Nothing to port; the canonical tree is left untouched.**

## Losslessness

SILENT gate: the tape path is near-lossless by the battery's ULP judgment (not bit-exact vs savestate). No FAIL signal this session: 9/9 streams bit-identical to each other, FIRST20 = the 65-class anchor line in 10/10 runs. Per the task, divergences of this class are recorded, not chased; there is nothing to report beyond this line.

## Session hygiene

Locks: flock held for every run (v2 A/B/C 06:33–06:44Z, phase arm 06:44–06:45Z, 128 anchor 06:51–06:53Z), 25 s idle before and between, released cleanly. Pins: bench `700f18…` (512), phase bench `608a77…`, 128 bench `2c2d6c…` + guard `4a262a…`, SBCAR `02a54359…` pre/post every session, .so md5s as in the regime section. Root sampler 921 samples (sclk under load ~3.30 GHz max, 0 MHz in idle gaps; temp sysfs node absent on this box, as in the battery).

## Stop branch taken

A ≈ 40.4 < 60 → the graft 65-class is not reproduced on the canonical 512 protocol → per the task: report the phase-timer delta decomposition (above) and **stop; no port**. The `refwd-m1-port.patch` (prepared, verified applicable to the canonical 1ca31eea `qmm.hip`) is **not applied** to the vendor tree, and no wheel is rebuilt.
