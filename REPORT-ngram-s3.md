# REPORT-ngram-s3: n-gram fill wired into the S=3 tape path (t1986, guest-a RX 9070 XT / gfx1201, ROCm 10.1, 2026-11-04)

**Verdict: the candidate works and is a +34.2% t/s lever (89.68 -> 120.33 median, tok/step 2.907 -> 2.969, FIRST20 bit-identical, all 12 runs rc=0), but it is DECLINED against the stated adopt bar (t/s >=+5% AND tok/step >= 3.0 AND FIRST20 identical) because tok/step 2.969 < 3.0 — the 3.0 gate is unreachable on this (degenerate 1919-loop) trajectory by construction (see §6); the measured D1/D2 decomposition is the complete answer. The t1959 no-op claim is CONFIRMED (arm A ran with MTP6_NGRAM=1 in the env set and it is dead code in mtp-bench6.py); the candidate MTP6_NGRAM_S3=1 wires the mode-4 NgramFill into the S=3 draft chain and is the first measurement of the ngram fill on the live S=3 tape path.**

## 0. Mission

MTP ladder lever: the tok/step lift 2.907 -> 3.15 anchor shape (llama.cpp MTP anchor 3.15 tok/step, our 2.907). Phase 0: reproduce the canonical S=3 control (first gate, per the t1970 lesson). Phase 1: code-first naming of what is missing for the S=3 tape path to get the ngram fill active. Phase 2: full 322 A/B battery (3 rounds x 2 arms interleaved).

## 1. Setup and identity

- Base m3-verify-eff-probe @ 4319a542 (verified: local HEAD == bare remote head at run start). Machine: guest-a (<tailnet-ip>, RX 9070 XT / gfx1201, ROCm 10.1 guest userland); flock on ~/gpu.lock (free at start, acquired by this run, released at end).
- Wheels: venv-p3 for BOTH arms, libmlx.so md5 460c803d66c43ebfce8997ef08c912db verified before and after the battery (untouched). Model sidecar md5 02a54359cda8b247f76b277dd9bc4d4c verified (= model-mtp.safetensors, the MTP module loaded by the default config).
- Harness: arm A = mtp-bench6.py @ 4319a54 (md5 1d9bc556c0d42efa672a6da2350bd634 = byte-identical to the t1981 battery harness in ~/m3verify); arm B = the same file + the MTP6_NGRAM_S3 candidate (md5 7d2ace6b26de003ef4d7cf92760bc623). Config-of-record envs (all arms): MLX_MTP=3 MTP6_TAPE=1 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0, anchor MTP_MAX_TOKENS=322, 25 s idle between arms, sclk sampled every 10 s.
- SCLK: locked 2200 and VERIFIED this session — idle reading 2200 at session start (rocm-smi at /opt/rocm-7.2.0/bin, the t1981 pattern; the t1970 session could not verify) and 64/64 samples at 2200 MHz through the whole battery window (17:55–18:10Z, all under load).
- Session state: the degenerate "271 1919x19" loop trajectory (the t1970 environmental/model artifact, affecting all arms equally).

## 2. Phase 0 — control (the first gate): REPRODUCED

| run | t/s | D1/D2 | tok/step | verifies | refwd | GDN_TAPE_CALLS | FIRST20 |
|---|---|---|---|---|---|---|---|
| ctrl1 | 89.97 | 98.8 / 93.1 | 2.907 | 161 | 13 | 4176 = 24 x (161 + 13) | 271 1919 x19 (reference) |
| ctrl2 | 89.79 | 98.8 / 93.1 | 2.907 | 161 | 13 | 4176 | reference |
| ctrl3 | 89.90 | 98.8 / 93.1 | 2.907 | 161 | 13 | 4176 | reference |
| **median** | **89.90** | in band | 2.907 | | | | |

All three controls reproduce the record within session noise (89.5-90.0 t/s band; D1 ~98.8%; D2 ~93%; 2.907 tok/step; FIRST20 = 271 1919 x19 reference). The runner hard-gated the battery on this (FIRST20 + median in band); the gate passed before any candidate measurement.

## 3. Phase 1 — code-first: what is missing in the S=3 path (and the candidate)

**The t1959 claim, confirmed by reading the code:** the NgramFill class exists ONLY in the mode-4 gate harness (mtp-bench6-dg.py:74), and there `ngram_on = dgate and mode == "4"` (mtp-bench6-dg.py:315). The canonical S=3 bench (mtp-bench6.py) reads NONE of MTP6_DRAFTGATE / MTP6_DGATE_T0 / MTP6_NGRAM / MTP6_NGRAM_N — its mode-3 draft phase unconditionally runs 2 MTP forwards + 2 draft-head argmaxes per step, and its mode-3 reject trims mtp_cache by fixed 1 (d2 reject) / 2 (d1 reject) assuming exactly 2 rows were pushed. So MTP6_NGRAM=1 in the env set is a no-op in the live S=3 tape path (arm A of this run proves it empirically: identical to the t1981 A arms).

**What is missing (named):** (1) the NgramFill class in mtp-bench6.py; (2) fill state init from prompt_ids (n = MTP6_NGRAM_N, default 4) and per-emitted-token updates (verify-step emissions + the single-token step's t, updated at append time before the lookups that consume it); (3) fill lookups at draft positions 1 and 2 in the mode-3 chain (a hit skips the MTP forward for that position — the same stale-mtp_cache approximation class as the mode-4 fill); (4) mtp_cache reject-trim accounting for skipped forwards (n_fwd = fw1 + fw2; d2 reject trims max(0, n_fwd - fw1), d1 reject trims n_fwd — with fw1=fw2=True this reduces to the canonical 1/2 constants, so default-off is bit-identical).

**The candidate (MTP6_NGRAM_S3=1, default off = canonical S=3 verbatim, zero behavior change):** implements exactly (1)-(4) in mtp-bench6.py (+ INFO NGRAM_S3 n=%d NNG=%d print). The NgramFill class body is AST-identical to the adopted mode-4 one. All emitted tokens remain target-greedy (lossless by construction); a fill hit only skips draft forwards, never the verify.

## 4. Phase 2 — A/B battery (3 rounds x 2 arms interleaved, 322 anchor, config-of-record, venv-p3 both arms)

Pre-battery sanity B (run twice — the runner script edit duplicated the sanity call; both are consistent with the battery and serve as stability checks): 120.37 / 120.45 t/s, NNG=316, FIRST20 reference, rc=0.

| run | A canonical | B candidate (+MTP6_NGRAM_S3=1) |
|---|---|---|
| r1 | 89.68 | 120.15 |
| r2 | 89.86 | 120.33 |
| r3 | 89.66 | 120.01 |
| **median** | **89.68** | **120.33** |
| spread | 0.23% | 0.26% |

**B vs A: +30.65 t/s = +34.2% (same wheel, same session, interleaved).** All 12 runs rc=0; FIRST20 = 271 1919 x19 (bit-identical) in every run — the silent gate never fired. SCLK 64/64 @ 2200.

| arm | gen | tok/step | D1/D2 | verifies | refwd | NNG | GDN_TAPE_CALLS |
|---|---|---|---|---|---|---|---|
| A | 5.35 s (33.2 ms/step) | 2.907 | 98.8 / 93.1 | 161 | 13 | n/a | 4176 |
| B | 4.00 s (24.8 ms/step) | 2.969 | 98.8 / 99.4 | 161 | 3 | 316 | 3936 = 24 x (161 + 3) |

## 5. The D1/D2/tok-step decomposition (why the number is what it is)

- **D1 is unchanged (98.8% both arms):** on this degenerate 1919-loop trajectory the ngram position-1 proposal (the most-frequent continuation of the all-1919 key = 1919) equals the MTP module's draft (target-greedy 1919 except 2/161 steps), so the D1 acceptance mix is identical: 159/161.
- **D2 is the lift (93.1% -> 99.4%):** the canonical MTP module's draft2 deviates from 1919 on 10/159 steps (those drafts are rejected); the ngram fill proposes the most-frequent continuation (1919) at position 2, which equals target-greedy on 158/159 steps (one miss). The fill's n=4 context lookup is doing exactly what llama.cpp's ngram-map draft is for: substituting a zero-cost, context-consistent proposal for a drafter-forward proposal.
- **tok/step = 1 + (D1ACC + D2ACC)/VERIFIES: 2.907 -> 2.969.** The 3.0 gate requires D2ACC >= 163 given D1ACC = 159 — impossible, since D2ACC <= D1ACC. **On this trajectory the achievable ceiling at D1 98.8% is 1 + 159/161 + 159/159 = 2.9876 < 3.0, regardless of fill quality.** 3.0 tok/step on this config requires D1 >= ~99.3% (or a non-degenerate trajectory where the fill improves D1 as well — on real text the MTP draft is often non-greedy while the ngram proposal can be the greedy continuation; this session's degenerate loop cannot show that).
- **The +34.2% t/s mechanics (8.4 ms/step saved, 25% of the A step):** B skips 2 MTP forwards + 2 int2-head GEMVs on 316/322 draft positions (NNG=316 = 158 full two-position hit steps + warm-up misses) — the draft section (4.9 ms at K=2 per the t1970 roofline) plus the sync round-trips — and D2 rejections drop 10 -> 1 (REFWDS 13 -> 3, ~0.7 ms/step amortized at 11.7 ms/refwd). All emitted tokens are still target-greedy (the verify pass is untouched), so the stream stays lossless.
- NNG=316: 158 steps where both draft positions were fill hits; the 6 misses are the first ~3 steps before the 1919 loop is established (the all-1919 key does not exist yet) — no misses once the trajectory settles, consistent with the D2 158/159.

## 6. Adoption decision

Bar: t/s >=+5% same-wheel AND tok/step >= 3.0 AND FIRST20 identical. Measured: +34.2% t/s (pass), tok/step 2.969 (FAIL, < 3.0), FIRST20 identical (pass) -> **DECLINED against the stated bar** — but note the failure is structural, not marginal: 3.0 is unreachable on this trajectory at D1 98.8% (ceiling 2.9876, §5). The candidate is env-gated (MTP6_NGRAM_S3, default off = canonical verbatim) and default-off causes no harm; the measured structure is a real +34% t/s lever with a D2 93.1 -> 99.4 acceptance lift on the degenerate loop. For the 2.907 -> 3.15 ladder target, the lever that matters on a non-degenerate trajectory is the D1 improvement (fill at position 1 replacing non-greedy MTP drafts) — this session's loop trajectory cannot measure that leg. Recommendation: re-measure the candidate on a session that does not lock into the 1919 loop (or on a non-degenerate prompt) before any adopt decision against the tok/step leg; the t/s leg is already cleared by a 6x margin.

## 7. Provenance

- Session: single flock window 17:55-18:10Z, all 12 runs rc=0, per-arm timeout 900 (none hit), 25 s idle between arms, sclk 64/64 @ 2200.
- md5s: venv-p3 libmlx 460c803d (before + after); sidecar 02a54359 (model-mtp.safetensors); A harness 1d9bc556 (== t1981 battery harness); B harness 7d2ace6b; runner fad2e260.
- The session's degenerate 1919-loop trajectory (271 1919x19) is the t1970 environmental/model artifact, present in all arms equally (control and candidate FIRST20 identical); acceptance statistics remain valid by the t1970 protocol.
