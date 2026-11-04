# REPORT-cheap-head.md — K=3 MTP chain: cheap-head levers (4-bit MTP module, int2 shared head, batched argmax)

Branch: `cheap-head-chain` (from `roofline-draft-step` @ dfd6bf4).
Box: neon (RX 9070 XT, gfx1201, ROCm 10.1 guest userland). Wheels: venv-r4v
(libmlx md5 8acf13fe...), venv-kfix (md5 69d73260...), venv-int2w (md5
0b8aa95d..., adds `mx.fast.lm_head_int2_wide`). Harness:
`bench/roofline-draft-perop.py` (per-op + draft-chain wall, AR=512 warmup, 3 reps,
median) and `bench/mtp-bench6.py` mode 4 (256-token acceptance runs). Config B
regime: GDN_TAPE=1 GDN_FUSE=1 QMV_ROWS=1 QMV_ROWS_MIN_N=1 REFWDM1=1. Protocol:
per-run flock (loop, -w 1800, sleep 60), 25 s idle before each run, SCLK sampled
before/after each run (direct `pp_dpm_sclk` read; `clkcheck.sh` was non-functional
— sysfs busy-loop; clocks never changed), 3 reps, median.

## 0. TL;DR

| Lever | Verdict |
|---|---|
| B+4bit-MTP | **ADOPT (r4v)**; large chain win on all wheels; D drift mixed (r4v improved, kfix-class D1 −5.5pp); one stop-boundary token divergence /400. See §3. |
| B+int2w | **NOT ADOPT** — 0% draft acceptance (D1=D2=D3=0), tok/s −37%, chain only −1.58 ms (< 2 ms gate). Reproduces the t1796 zero-acceptance finding on the new wheel. |
| B+batched-argmax | **INFEASIBLE as specified** — the K=3 chain is a strict serial dependency chain; no two head GEMVs can have inputs ready simultaneously. Analysis in §2. |
| B+combined (4bit+int2w) | Chain reaches 3.83 ms (inside the 3–6 ms 144 budget) but inherits int2w's 0% acceptance — no practical benefit over 4bit alone. |

## 1. Arms and code

- 4-bit MTP arm: the switch is `CH_MTP_4BIT` (NOT the wheel's `MTP_4BIT`).
  Reason: the wheel reads `MTP_4BIT` at model-init, before `load()`; with it
  set, the wheel builds the quant MTP module and then `load()`
  shape-mismatches the globbed bf16 sidecar. `CH_MTP_4BIT` rebuilds
  `MTPModule(quant=True)` AFTER load and loads the 4-bit sidecar directly
  (path via `MTP4_SIDECAR`, default `model-mtp-4bit.safetensors`; the box file
  `model-mtp-4bit.safetensors.disabled` is used by path, NOT renamed — the
  r4c-sidecar-surgery lesson). Sidecar verified (in the prior session):
  152,130,753 bytes, md5 e34f85e4, a genuine gs=64 4-bit quantization of the
  canonical `model-mtp.safetensors` (max rel error ~4–5%, norms exact).
- `MTP6_DRAFT_HEAD=int2w` (pre-existing switch, now wired in the per-op
  harness too): int2 g128-group asymmetric lm_head (quarter-split 2-bit pack,
  radiance port) + `lm_head_int2_wide` + exact bf16 re-rank of the top-32
  candidates. Int2 head = 317,857,920 bytes (303 MB) vs 4-bit head 2,163,367,936
  bytes (2,064 MB) — 7× smaller.
- M=3/M=4 head probes added to the harness for the batched-argmax record.

## 2. B+batched-argmax: INFEASIBLE as specified

The draft report's premise (§4): "K=3 = 3 independent head calls ... one M=3
call". The dataflow is not 3 independent calls. It is a strict serial chain:

```
mtp_1 = MTP(h_prev, e_prev)
d_1   = argmax(head(h_1));  e_1 = embed(d_1)
mtp_2 = MTP(h_1, e_1)       # requires d_1 (the argmax output)
d_2   = argmax(head(h_2));  e_2 = embed(d_2)
mtp_3 = MTP(h_2, e_2)       # requires d_2
```

A single M=3 head call requires h_1, h_2, h_3 ready simultaneously. But h_2
exists only after argmax(head(h_1)), and h_3 only after argmax(head(h_2)).
The MTP layer input is [hidden, embedding], and the embedding depends on the
previously SELECTED token — the argmax output. Batching the GEMVs would
require materializing mtp_2/mtp_3 without the selections, which does not exist
as a linear-algebra pass (embedding is a per-token lookup, not a linear map
of the inputs).

Even ignoring feasibility, the stated savings do not exist in the measured
regime:
- "per-call launch overhead 0.43 ms": launches are asynchronous in this
  wheel; the measured draft wall (7.3–11.0 ms) is HBM-bound (2.67 GiB of head
  traffic at ~634 GB/s ≈ 4.2 ms). There is no per-call wall to remove.
- "shared input HBM reduction": the only input shared by the 3 GEMVs is the
  head weight matrix itself (242 MB packed), which is read once PER ROW — an
  M=3 call reads exactly 3 × (K×N/4) bytes, the same as three M=1 calls. No
  traffic is saved.
- Measured M=3/M=4 head probe costs (same run, same idle protocol; for the
  record): r4v: M=3 2.18–2.23 ms, M=4 2.90–2.93 ms; i2w: M=3 1.31–1.32 ms,
  M=4 1.64–1.65 ms; kfix (clean): M=3 1.80 ms, M=4 2.24 ms. (An M=3 call is
  ~2.2× a single head on r4v — consistent with the 3×M=1 byte cost plus
  layout overhead; it does not get you 3 heads for 1.)

A genuinely batched head (e.g., branching over top-2 candidates per draft step)
is a different algorithm (different acceptance model, different cache handling)
— out of scope for this lever.

## 3. Measurements

Harness: per-op + chain wall (AR=512 warmup, 3 reps, median; CTX=539; config B).
Acceptance: bench6 mode 4 (256-token cap; B-class runs ran to 400–414 tokens,
int2w runs stopped at 256), hybrid tape.

### 3.1 Chain / fused wall (median of 3 reps)

| run | arm | chain ms | Δ vs B | corr ms | fused ms | mtp_head ms | head ms |
|---|---|---|---|---|---|---|---|
| r4v-b | B | 10.98 | — | 12.60 | 23.47 | 1.835 | 0.969 |
| r4v-4b | 4bit-MTP | 7.37 | **−3.61 (−33%)** | 12.56 | 19.96 | 0.691 | 0.956 |
| kfix-b | B | 9.54 | — | 11.99 | 21.40 | 1.838 | 0.951 |
| kfix-4b | 4bit-MTP | 5.57 | **−3.97 (−42%)** | 11.98 | 17.59 | 0.488 | 0.949 |
| i2w-b | B | 9.60 | — | 11.95 | 21.50 | 1.811 | 0.949 |
| i2w-2 | int2w | 8.02 | −1.58 (−16%) | 11.99 | 19.99 | 1.809 | 0.780 |
| i2w-4b | 4bit-MTP | 5.58 | **−4.02 (−42%)** | 11.98 | 17.58 | 0.489 | 0.949 |
| i2w-c | 4bit+int2w | 3.83 | −5.77 (−60%) | 12.15 | 16.10 | 0.486 | 0.782 |

The 4-bit MTP module removes 2.51 GiB of mtp traffic per call (mtp bytes
2,919,911,424 → 410,956,800) and is the dominant lever: mtp_head
1.81–1.84 → 0.49–0.69 ms. int2w shaves 0.17 ms per head call (0.949 → 0.780).

MTP head traffic (bf16 vs 4bit mtp): 2.92 GiB → 411 MB per call, 3 calls/chain.

### 3.2 Acceptance (bench6 mode 4, 256-token cap)

| run | t/s | Δ vs B | D1% | D2% | D3% | tok/step |
|---|---|---|---|---|---|---|
| r4v-b6 | 43.97 | — | 75.0 | 58.3 | 46.4 | 2.391 |
| r4v-b6-4b | 47.52 | **+8.1%** | 75.0 | 63.5 | 54.1 | 2.484 |
| kfix-b6 | 45.41 | — | 76.6 | 61.2 | 41.7 | 2.430 |
| kfix-b6-4b | 46.18 | +1.7% | 71.1 | 58.2 | 41.5 | 2.297 |
| i2w-b6 | 49.62 | — | 76.6 | 61.2 | 41.7 | 2.430 |
| i2w-b6-4b | 50.89 | +2.6% | 71.1 | 58.2 | 41.5 | 2.297 |
| i2w-b6-2 | 31.30 | **−37.0%** | 0 | 0 | 0 | 1.000 |
| i2w-b6-c | 33.57 | −32.7% | 0 | 0 | 0 | 1.000 |

Note the kfix-class wheel (kfix, i2w) B baseline (45.4–49.6 t/s, tok/step 2.43)
is a markedly faster regime than the t1796 report (~27 t/s) — the wheel moved.

### 3.3 Token identity (256-token stretch, full stream)

- FIRST20 identical across all arms (all runs share the first 31 tokens).
- 4bit vs B (i2w): identical through token 61, diverge at token 62 — a
  stop-boundary token (B emits the stop sequence, 4bit continues 16+ tokens).
  Not bit-identical over the full stream.
- int2w vs B (same wheel): diverge at token 32. The draft never feeds the
  committed stream (verify/commit is exact); with 0% acceptance the int2w run
  is pure M=1 commit forwards while B's run uses M=4 verify batches — the
  divergence is M=4-vs-M=1 kernel numerics, not a draft effect. (The r4v and
  kfix-class wheels' B runs also diverge from each other at token 32 — wheel
  kernel difference.)
- combined vs int2w: 256/256 identical.

### 3.4 Gates

- **int2w**: FAIL. Chain −1.58 ms (< 2 ms), tok/s −37% (not +4%), acceptance
  0% vs 76.6% (not neutral). The t1796 zero-acceptance finding holds on the
  new wheel: int2 draft quality for this 9B model is zero.
- **4bit-MTP**: r4v — chain −3.61 ms, tok/s +8.1%, D1 identical (75.0→75.0),
  D2/D3 improved (58.3→63.5, 46.4→54.1): gate PASS, **adopt**. kfix-class —
  chain −4.0 ms, tok/s +1.7/+2.6%, D1 −5.5pp (76.6→71.1, D2 61.2→58.2,
  D3 41.7→41.5), one stop-boundary divergence /400 tokens: D drift is not
  strictly neutral on this wheel; adopt with the drift caveat (the r4v
  shipping wheel is the adoption basis).
- **combined**: no practical benefit over 4bit alone (0% acceptance).

## 4. 144-shape contribution

144 t/s at tok/step T=2.4 requires spec step wall ≤ 16.7 ms. Measured spec step
wall (i2w-b6): 2.430/49.62 s = 49.0 ms; (i2w-b6-4b): 2.297/50.89 s = 45.1 ms.
Decomposition: fused draft 21.5 (B) / 17.6 (4bit) + verify+commit ≈ 27.5 ms.
The 4bit-MTP lever removes 3.9–4.0 ms of the 28.5 ms gap (≈14% of the gap);
the chain at 5.57–5.61 ms is inside the 3–6 ms draft budget, and combined
reaches 3.83 ms. The binding constraint to 144 is the verify/commit wall
(~27.5 ms), a different lever — the draft chain is no longer the bottleneck
once 4bit-MTP is adopted.

## 5. SCLK / lock / anomaly log

- SCLK sampled before/after each run (direct `pp_dpm_sclk`; `clkcheck.sh` was
  non-functional — sysfs busy-loop, no output). Samples during/after runs:
  2653–2697 MHz (transient boost above the 2400 MHz level; levels 500/827/2400),
  TEMP 32–35 °C. No throttling observed in the clean runs.
- Lock: 26 flock acquisitions across smoke+battery+supplemental, zero busy-wait
  retries (all runs started on the first attempt; sibling runs interleaved
  cleanly between ours).
- Anomaly: the first kfix-4b harness run was contaminated — all non-MTP class
  times inflated 5–27× (qmv 0.64 vs 0.10 ms/layer, embed 1.38 vs 0.05,
  per-run PRE_S 9.7 vs 2.9), chain 9.29 ms. Rerun (same protocol) gave clean
  class times and chain 5.57 ms — consistent with i2w-4b (5.58). The clean
  rerun is the recorded kfix-4b number. Likely a non-locking sibling process
  or a clock event during that window (no SCLK sample captured in that run —
  the first-sample read returned empty).

## 6. Files

- `bench/roofline-draft-perop.py` — `CH_MTP_4BIT`/`MTP6_DRAFT_HEAD` arm
  switches, `quantize_int2`/`int2_argmax` (radiance port), M=3/M=4 head
  probes, `int2_head_bytes` in RESULT_JSON.
- `bench/mtp-bench6.py` — 4-bit MTP module arm (post-load rebuild + sidecar
  load, no file surgery).
- Box `~/cheap-head/` — driver, per-run logs (chain/acceptance/SCLK), lock log.
