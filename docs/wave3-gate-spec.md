# WAVE 3 AUDIT — G1 gate spec + routing decision (zero-GPU, independent re-derivation)

Wave 3/3 of the failure-wave trigger (leg-3 terminal FAIL on measured data). Scope:
the G1 gate SPEC and the named routing next step — the measurement facts (wave 1)
and kernel correctness (wave 2) are taken as on-the-record; this run rules on the
spec. All verification is CPU-side: independent numpy re-derivation on this host
(the real tensor bytes were pulled read-only from the box), source reading of the
mirror at 42dd9c6, arithmetic re-checks. No GPU touched.

Method note (independence): the d2s6 x-quant and the ktype-10 dequant were
re-implemented from the source (mmq.hip `mmq_quant_in_q8_1_mmq` d2s6 branch,
lines 314-356; kquant_file_gemm.hip `deq_elem` ktype==10, line 193+) — not by
re-running the card's scripts. The re-implementation reproduces the card's G1
battery floor to 4 digits (below), which also validates the ground-truth W
(max|W| = 0.226551; max|ref1| = 0.4130, matching the card-implied 0.4129-0.4130).

## Claim-by-claim verdict

### C1 — "the G1 gate is mis-specified for any mmq path" — **CONFIRMED**

The literal G1 (M=1 total rel <= 1e-3 vs the f32 reference) is unmeetable by ANY
q8_1 mmq path. Independently derived on the real tensor (blk.3.attn_v, K=5120,
N=1024, x ~ N(0,0.1), the card's exact seed-42 battery vector):

| x-quant variant | max|d| | rel |
|---|---|---|
| current d2s6 (as built; option A and B, same host quantizer) | 9.458e-02 | **2.290e-01** (exact 4-digit match to the card) |
| plain q8_1 (min-term = 0, valid in the mma structure y = d·t + s/16) | 2.701e-03 | 6.539e-03 |
| d2s6 layout, proper mean-correction (see C2) | 2.215e-03 | 5.363e-03 |
| robustness check, seed 7 (current d2s6) | 1.044e-01 | 2.107e-01 |

Even the minimum-achievable q8_1 floor (5.36e-3) is 5.4x the 1e-3 gate; the
as-built floor (2.29e-1) is 229x. The kqfg-class calibration is confirmed
arithmetically: KQFG M=2 = 2.454e-6 (record) << 1e-3. The 1e-3 tolerance was
evidently calibrated to kqfg-class accuracy and does not apply to mmq. The
recommended corrected spec (C5) is sound, with the two modifications below.

### C2 — "2.29e-1 is the irreducible architectural floor" — **REFUTED as a floor;
CONFIRMED as the as-built behavior**

This is the load-bearing correction of the wave. The 2.29e-1 number is the
behavior of the **current d2s6 min-term design, not an irreducible property of
q8_1**. Decomposition (same battery, same W):

- grid-only (no min-term): 6.539e-03
- + the min-term (s_g/16 with s_g = f16(d_g·Σt)): **2.290e-01** — the min-term is
  the entire 35x amplification
- f16 scale rounding alone: negligible (2.290e-01 → 2.288e-01)

Mechanism (verified in the source of both the host quantizer and the option-B
mma, mmq-vec-dot.cuh `ggml_cuda_mmq_vec_dot_q2_K_file`): the quantizer computes
`t = rint(v·127/amax)` on the **raw** values and stores `s_g = f16(d_g·Σt)`;
the mma reconstructs `v' = d_g·t + s_g/16 = d_g·(t + mean(t))`. Since t
quantizes v (not v − mean), the reconstruction is **v + mean(group)** — the
group mean is *added on top of* the raw-value quantization instead of the
quantization being done on the mean-subtracted residual. The E0 unit probe
mechanism is confirmed independently: x=e0 → x' = f16(1/127)·127 + f16(f16(1/127)·127)/16
= 1.0625x (the 1.0625x "min-correction" the card names is exactly this add-the-mean
artifact; f16(f16(1/127)·127) = 1.000000).

The floor is avoidable within the d2s6 layout: a proper mean-correction
(per-16-group mean subtract, the two per-64-half scales recomputed on the
residual, s_g = raw group sum, last-32 plain — the mma reads s_g/16 unchanged,
so **only the host-side quantizer changes**) gives **5.363e-03** — a 43x
improvement, with the 41.2 t/s mma cost untouched. Plain q8_1 (s=0) gives
6.539e-03. Caveat: upstream GGML's d2s6 implementation is not present in this
tree to compare (headers only); 2.29e-1 is verified as the in-tree (and
therefore option A's) behavior, and the oracle (llama.cpp, 41.2 t/s) is the same
mmq d2s6 class, but the upstream behavior is unverified here. Either way the
word "irreducible" does not survive: 5.36e-3 is achievable in the same mma.

### C3 — "mmq quantizes x to q8_1 d2s6 architecturally (option A included)" —
**CONFIRMED (for the as-built paths)**

The in-tree mma x-side is structurally q8_1 for all 10 weight types
(`ggml_cuda_mmq_vec_dot_qX_q8_1_dp4a`, mmq.cuh); every mmq host entry point
(mmq.hip:511/737/1051/1055) feeds `mmq_quant_in_q8_1_mmq`, and the d2s6 branch
is the Q2_K path. Option A (in-tree dp4a) and option B (file-layout vec-dot)
share the host quantizer, so both carry the same 2.29e-1 as-built behavior.
"Architectural" holds for the as-built format spec; the minimum-achievable floor
of the mma structure is 5.36e-3 (C2), so the q8_1 format itself is not floored
at 2.29e-1.

### C4 — E0 line "max|d| = 1.409e-02 = 0.0625 · max|W[0,:]|" — **MODIFIED
(inconsistent as stated; re-verify on the next GPU run)**

For this tensor (independently dequanted two ways, bit-exact on sampled blocks):
max|W[0,:]| = **0.046088**, so 0.0625 · max|W[0,:]| = **2.8805e-03** — 4.9x
smaller than the card's 1.409e-02. The card's equation only closes if
max|W[0,:]| = 0.22544 ≈ max|W| (full-matrix max 0.226551; 0.0625·max|W| =
1.4159e-02, 0.49% off — not 4 digits). The 1.0625x mechanism itself is
confirmed (C2), and the *expected* E0 error for x=e0 on this tensor is 2.88e-03
(= 0.0625 · max|W[0,:]|). If the measured E0 really is 1.409e-02, the relative
E0 error is 30.6% of the row max — not "small + exact" — which would also
contradict the kernel-own-error class (6.6e-5). Most likely a mislabel (full
matrix vs row 0 max) or a different probe; it is not load-bearing for the gate
spec, but the E0 line as written is arithmetically inconsistent with the file
data. Cheap re-verify: one E0 re-run printing max|W[0,:]| alongside max|d|.

### C5 — the corrected G1 spec for the mmq path — **CONFIRMED, with two modifications**

The proposed shape (M=1 mmq gate = kernel-own rel <= 1e-3 vs the d2s6-quantized-x
numpy replica + a floor line + the E0 exactness check; literal total-rel gate
reserved for kqfg-class paths) is the correct "gate the thing under test, pin the
fixed floor" structure. Modifications:

1. **The floor must be re-checked in-band, not merely documented.** The floor is
   a property of the (mutable) x-quantizer: if the quantizer changes, the floor
   moves. Gate line: *measured total floor on the G1 battery within
   0.9x..1.1x of the numpy-replica floor* (as-built: 2.29e-1; after the C2 fix:
   5.36e-3). A floor that jumps outside the band = quantizer regression, FAIL.
   This also makes the gate self-updating when the min-term fix lands.
2. **E0 check with the corrected expectation**: E0 max|d| within 1.1x of
   0.0625 · max|W[0,:]| (= 2.88e-03 for this tensor), re-verify the card's
   1.409e-02 (C4).
3. **New, named follow-up the card misses (from C2)**: fix the d2s6 min-term
   (host-side quantizer only: per-16-group mean subtract, half-scale on the
   residual, store s_g = raw sum; mma unchanged) → floor 2.29e-1 → 5.36e-3 (43x),
   41.2 t/s preserved. This is the highest-leverage next action: it turns the
   mmq M=1 path from a 229x-gate-miss into a 5.4x-gate-miss and materially
   changes the op/cloudy accuracy ruling (see C6).

### C6 — routing: "dispatch M=1 to kqfg; option-B mmq env-gated fast path" —
**CONFIRMED in shape; mechanism mis-specified; the ruling's framing is off**

- **kqfg M=1 cost = 0.28 t/s — CONFIRMED from the record** (lever-finding card:
  decode 1024 tokens = 3693.86 s = 0.28 tok/s, ms/step p50 3607.2; profile
  lead: kquant_file_gemm = 102% of step cost, 2.55 GB/s effective). It is 147x
  below the 41.2 t/s mmq control class (same card; also the "41.2 t/s control
  class" comment at kquant_file_gemm.hip:1202). Consequence the card undersells:
  the kqfg M=1 default is accuracy-correct but a **latency dead-end for serve**
  (mission target AR 87 t/s / MTP 144 t/s). The env-gated mmq M=1 opt-in is the
  actual serve path; the "interim state" is correct only as a *safety* default.
- **The named mechanism is mis-specified (MODIFIED)**: "raise `mr_gate` to >= 1"
  is wrong — the default is *already* mr_gate=1, and the dispatch is
  `M <= mr_gate → mmq` (kquant_file_gemm.hip:1205). To send M=1 to kqfg the
  existing knob is `MLX_KQUANT_FILE_MR=0` (lower, not raise). Side effect:
  `use_mr = (mr_gate==2)||(mr_gate==1 && M<=16)` (line 702) — mr_gate=0 also
  disables the kqfg multi-row path for M=2..16. Cleaner: the card's own
  `MLX_KQFILE_MMQ_M1` env gate (default off → M=1 to kqfg; on → option-B mmq)
  without touching mr_gate. Note the safe fallback already exists: if
  `mmq_q2k_file` returns false (e.g. N%256 != 0) the dispatch falls through to
  kqfg (line 1205-1216).
- **The op/cloudy ruling should not be framed as "is 2.29e-1 acceptable."** The
  as-built floor is 2.29e-1, but the *achievable* floor is 5.36e-3 (C2/C5).
  The decision is: (a) kqfg M=1 — 2.45e-6-class accuracy, 0.28 t/s (serve dead);
  (b) as-built mmq M=1 — 2.29e-1 total floor, ~41 t/s class; (c) **fixed-mmq
  M=1 (C5.3) — 5.36e-3 total floor, ~41 t/s class** — the one that could make
  the mmq path serve-credible. (c) should be in the ruling.

### C7 — FLIP DECISION RULE (P1-e vehicle-flip A/B, real 27B decode)

Pre-registered, before the run fires. Setup: arm A = `MLX_KQFILE_MMQ_M1=1`
(option-B mmq M=1), arm B = kqfg M=1 (gate off); same 27B Q2_K model, same
prompt bytes, temperature 0, >= 1024 decode tokens, 3 runs per arm, interleaved,
warmup excluded. If the C5.3 min-term fix lands first, add arm C = fixed-mmq M=1.

Must measure (per arm):
1. **Throughput**: decode t/s + ms/step p50.
2. **Coherent output class**: 5gram-dup-rate over the full window, "b-run"
   onset token index, FIRST20 tokens (precedent: the S4VA A/B gate), word-salad
   class vs arm B.
3. **Real-x floor (the number the battery cannot give)**: over the first 1000
   steps, per-step output max|d| and rel vs arm B's same-step output; report
   max/median. This measures the realized floor on REAL activations (the G1
   battery is adversarial-random; the real-x floor is currently unmeasured).
4. **Divergence**: first divergent token index vs arm B, divergence class
   (1-ulp/rounding vs semantic), token-md5 of the first 100 tokens.

Decision rule (thresholds are PROPOSALS for the op/cloudy ruling):
- G1 COHERENCE: arm A 5gram-dup-rate <= 0.05 over the window, no sustained
  b-run before token 256, class == arm B's class.
- G2 REAL-X FLOOR: max rel over the first 1000 steps <= 5e-1 (hard cap), AND
  kernel-own rel <= 1e-3 (vs the arm-B reference, i.e. the error the kernel adds
  on top of the shared x-quant floor stays in the 6.6e-5 class).
- G3 DIVERGENCE: first divergence at/after token 64, or first-100 token-md5
  identical (a 1-ulp flip class is acceptable per the S4VA precedent: divergence
  at token 32 with coherent text passed).
- G4 LATENCY: t/s_A >= 20 (>= 70x arm B's 0.28) and the mmq path is the dominant
  step cost (profile check, per the lever-finding method).
- **FLIP if G1 ^ G2 ^ G3 ^ G4 all PASS; HOLD otherwise** (kqfg M=1 stays the
  default, mmq M=1 remains opt-in). With arm C (fixed-mmq): G2's real-x floor
  expectation tightens to the 5.36e-3 class, and a PASS makes (c) the serve
  path.

## NAMED RECOMMENDATION (for the lane; orchestrator + cloudy rule on it)

1. **Re-spec G1 now (no GPU needed)**: for the mmq path — kernel-own rel <= 1e-3
   vs the d2s6-quantized-x numpy replica (same battery) + in-band floor check
   (0.9x..1.1x of the replica floor; as-built 2.29e-1) + E0 mechanism check
   (expected 2.88e-03 for this tensor; re-verify the card's 1.409e-02). The
   literal total-rel <= 1e-3 gate applies to kqfg-class paths only.
2. **Routing interim state**: default M=1 → kqfg via the existing
   `MLX_KQUANT_FILE_MR=0` (or the cleaner new `MLX_KQFILE_MMQ_M1=0` default; do
   not "raise mr_gate"); option-B mmq M=1 stays opt-in. Frame the op/cloudy
   ruling on (a) kqfg 2.45e-6-class @ 0.28 t/s vs (b) as-built mmq 2.29e-1 @
   ~41 t/s vs (c) fixed-mmq 5.36e-3 @ ~41 t/s — (c) changes the decision.
3. **Fire the C5.3 min-term fix** (host-side quantizer only, ~1 function, mma
   unchanged): floor 2.29e-1 → 5.36e-3. Then one GPU build to land the
   corrected G1 (kernel-own rel + in-band floor at 5.36e-3). This is the
   highest-leverage action in the lane.
4. **P1-e flip A/B with the pre-registered rule in C7** (arm C if the fix lands
   first) to measure the real-x floor and the throughput on real 27B decode.

## VERDICT: MODIFIED

- C1 gate mis-specified for mmq: CONFIRMED (min-achievable q8_1 floor 5.36e-3
  is 5.4x the 1e-3 gate; as-built 2.29e-1 is 229x; kqfg 2.45e-6 calibration
  confirmed).
- C2 "irreducible 2.29e-1 floor": REFUTED as a floor (it is the current
  add-the-mean min-term design; 5.36e-3 achievable in the same mma), CONFIRMED
  as the as-built behavior (reproduced to 4 digits on the real tensor).
- C3 "architectural q8_1 d2s6, option A included": CONFIRMED for the as-built
  paths (shared host quantizer; structurally q8_1 mma).
- C4 E0 line: MODIFIED (equation inconsistent with the file data; 1.0625x
  mechanism confirmed; expected 2.88e-03 vs card 1.409e-02; re-verify).
- C5 corrected gate spec: CONFIRMED + in-band floor check + new 43x min-term fix.
- C6 routing: CONFIRMED in shape (0.28 t/s record-confirmed; kqfg default is a
  latency dead-end), MODIFIED on the mechanism (mr_gate is *lowered* to 0, not
  raised; use_mr side effect; use the dedicated M1 env gate) and on the ruling
  framing (include the 5.36e-3 fixed-mmq arm).
- C7 flip rule: specified above with pre-registered proposal thresholds.

Provenance: zero-GPU; tensor bytes + measurement-script listings read-only over
ssh from the box (md5 92eb0db4bef078ac4c113d0c9831f9ab, 1,720,320 B); kernel
and ground-truth source read from the mirror at 42dd9c6; all numbers
re-derived on this host in /tmp/wave3_audit (derive.py, e0check.py, decomp.py,
crux.py, minfloor.py, d2s6fix.py).
