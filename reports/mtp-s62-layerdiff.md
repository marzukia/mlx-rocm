# m3: per-layer FULL-STATE diff at S62 — T=2 verify vs 2×T=1 (m2-vs-m0 mtp divergence)

Date: 2026-10-29 (UTC). Run on guest-a (RX 9070 XT / gfx1201, ROCm 10.1 guest,
venv-d, model `/models/qwen3-9b-mlx`). Script: `scripts/diag-s62-layerdiff.py`
(MODE=m3), guest log `/models/tmp/run-785/diag-m3.log`, captures
`/models/tmp/run-785/cap-m3-s62.pkl`.

## What m3 does

Extends the run-809 m2 flow (same prompt, same 30-step m2 verify loop, same ARM
condition). At the S62 arm it:

1. snapshots the ENTIRE main-model cache (all 32 layers: ArraysCache conv/ssm/
   left_padding/lengths + KVCache keys/values/offset; arrays deep-copied as f32
   numpy with the original dtype recorded),
2. runs **variant A**: one T=2 verify forward `lm([[279, 3545]])` and harvests
   every layer's post-forward state,
3. restores the snapshot and **verifies the fork bit-exact** (32/32),
4. runs **variant B**: two T=1 forwards `lm([[279]])` then `lm([[3545]])` and
   harvests,
5. diffs every layer in order — one table, not stop-at-first — plus the next-t
   logits after 279 (A[0,0] vs B1[0,0]) and after 3545 (A[0,1] vs B2[0,0]).

The MTP draft module's own cache is not touched by either variant and is not
part of the diff.

## Arm confirmation

- ARM reached: `step=60 prev=279 draft=3545 n_out=61` — identical to the run-809 m2
  run (out[-8:]=[3418, 424, 11, 567, 1902, 401, 7355, 279]).
- All 8 attention layers at offset 87 (87 prior tokens + the arm's own token
  history), i.e. the pre-verify state is the 87-token history.
- Variant A act = **ACC** (t2=3545 == draft=3545). The m2-vs-m0 divergence
  regime (n_out==61) is reproduced.
- Restore verified: **32/32 entries bit-exact** — the fork is sound; the A/B
  comparison is from the identical pre-verify state.

## Per-layer full-state diff (variant A T=2 vs variant B 2×T=1)

Layer map: 24 GDN layers (0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30) +
8 full-attention layers (3, 7, 11, 15, 19, 23, 27, 31).

```
layer  type  sub    n_diff/total     maxabs       first_idx
0      gdn   conv   0/24576          -            -
0      gdn   ssm    0/524288         -            -
1      gdn   conv   0/24576          -            -
1      gdn   ssm    0/524288         -            -
2      gdn   conv   0/24576          -            -
2      gdn   ssm    0/524288         -            -
3      attn  k      0/91136          -            -
3      attn  v      0/91136          -            -
4      gdn   conv   2059/24576       1.562e-02    16392
4      gdn   ssm    515560/524288    7.177e-04    0
5      gdn   conv   4892/24576       3.125e-02    16386
5      gdn   ssm    523310/524288    4.580e-04    0
6      gdn   conv   5623/24576       1.562e-02    16384
6      gdn   ssm    523581/524288    1.580e-03    0
7      attn  k      645/91136        6.250e-02    22530
7      attn  v      697/91136        1.562e-02    22528
8      gdn   conv   6131/24576       3.906e-02    16384
8      gdn   ssm    523869/524288    6.262e-03    0
9      gdn   conv   5998/24576       3.223e-02    16384
9      gdn   ssm    523943/524288    2.716e-03    0
10     gdn   conv   6122/24576       3.125e-02    16384
10     gdn   ssm    523316/524288    3.472e-03    0
11     attn  k      725/91136        6.250e-02    22528
11     attn  v      813/91136        1.562e-02    22528
12     gdn   conv   6280/24576       6.250e-02    16384
12     gdn   ssm    523868/524288    5.972e-04    0
13     gdn   conv   6362/24576       3.125e-02    16384
13     gdn   ssm    524185/524288    9.415e-04    0
14     gdn   conv   6237/24576       3.125e-02    16384
14     gdn   ssm    523528/524288    2.385e-03    0
15     attn  k      745/91136        6.250e-02    22528
15     attn  v      823/91136        1.953e-02    22528
16     gdn   conv   6430/24576       3.125e-02    16384
16     gdn   ssm    523974/524288    1.020e-03    0
17     gdn   conv   6396/24576       6.250e-02    16384
17     gdn   ssm    524206/524288    2.161e-03    0
18     gdn   conv   6121/24576       3.125e-02    16388
18     gdn   ssm    524008/524288    9.623e-04    0
19     attn  k      673/91136        3.125e-02    22529
19     attn  v      812/91136        3.125e-02    22529
20     gdn   conv   6057/24576       6.250e-02    16384
20     gdn   ssm    524118/524288    3.415e-03    0
21     gdn   conv   5963/24576       6.250e-02    16384
21     gdn   ssm    523916/524288    2.583e-03    0
22     gdn   conv   5999/24576       6.250e-02    16384
22     gdn   ssm    524229/524288    2.011e-03    0
23     attn  k      749/91136        6.250e-02    22528
23     attn  v      841/91136        3.906e-02    22528
24     gdn   conv   5837/24576       6.250e-02    16384
24     gdn   ssm    524240/524288    5.269e-03    0
25     gdn   conv   5747/24576       6.250e-02    16384
25     gdn   ssm    524261/524288    5.988e-03    0
26     gdn   conv   6127/24576       6.250e-02    16384
26     gdn   ssm    524251/524288    1.285e-02    0
27     attn  k      733/91136        9.375e-02    22529
27     attn  v      865/91136        1.250e-01    22529
28     gdn   conv   5942/24576       1.250e-01    16384
28     gdn   ssm    524260/524288    2.850e-03    0
29     gdn   conv   6064/24576       6.250e-02    16384
29     gdn   ssm    524261/524288    6.290e-03    0
30     gdn   conv   6254/24576       1.250e-01    16384
30     gdn   ssm    524254/524288    5.499e-03    0
31     attn  k      761/91136        6.250e-02    22528
31     attn  v      713/91136        8.203e-02    22528
```

- **first diverging layer: 4 (gdn, conv)** — and every layer from 4 to 31
  diverges (all GDN conv+ssm, all attention k+v).
- Layers 0-3 (gdn 0,1,2 + attn 3) are **bit-exact**.
- logits after 279 (A[0,0] vs B1[0,0]): **MATCH** — all 248320 entries bit-exact.
- logits after 3545 (A[0,1] vs B2[0,0]): **DIFF** — n=192967/248320, maxabs
  1.250e-01, first idx 2.

## Suspect verdict

**The first diverging STATE is GDN layer 4, but the first diverging COMPUTATION
is the attention (SDPA) output of layer 3 for token 3545 — the second token —
computed as row 1 of a 2-row causal SDPA (variant A) vs a 1-row unmasked SDPA
(variant B).** The divergence then cascades through every subsequent layer.

Chain of evidence:

1. **Not the KV page layout.** Attention layer 3's k/v state is bit-exact
   (0/91136 both). The paged KV writes for both tokens are identical in A and B.
   (The k/v divergence at layers 7-31 is downstream: it is a function of the
   already-diverged input hidden states.)
2. **Not the GDN computation (again).** GDN layers 0-2 and the attention layer 3
   are bit-exact end-to-state. For GDN layer 2, the conv state after 3545 is
   `[old_row2, qkv_279, qkv_3545]` — the old conv rows are the shared snapshot,
   so the conv match means `qkv_279` and `qkv_3545` are bit-exact in A and B.
   `qkv = in_proj_qkv(post_layernorm(hidden))` is a deterministic GEMM (4096→
   8192, effectively injective), so the hidden states ENTERING GDN 2 for both
   tokens are the same in A and B. The same induction holds up the stack: the
   input hidden states of GDN 0, 1, 2 are all the same, so their outputs are the
   same (deterministic layers), so the input to attention layer 3 is the same.
   With the same input and (per run-809 L1/L5) bit-exact GDN kernels/prep, GDN
   layers 0-2 being bit-exact is exactly what we see.
3. **The input to GDN layer 4 differs.** GDN 4's conv state after 3545 is
   `[old_row2, qkv_279, qkv_3545]` and it DIVERGES (2059/24576, maxabs
   1.562e-02). The old conv rows are shared, and `qkv_279` is the same (the
   logits after 279 are bit-exact ⇒ the hidden after 279 is bit-exact, lm_head
   is injective), so the divergence is in `qkv_3545` ⇒ the hidden state
   ENTERING GDN 4 for token 3545 differs between A and B.
4. **That hidden state is the output of attention layer 3 for token 3545**
   (residual + attn_out + MLP; the residual and MLP input are the same per (2),
   the MLP is deterministic). Therefore the attention output for token 3545
   differs between A and B.
5. **The attention inputs are identical.** Queries (same token, same RoPE offset
   88), keys/values (bit-exact per (1), 89 = 87 cached + 2 new in A; the B
   forwards see the same 88 then 89 keys — the 279 key written by B's first
   forward is bit-exact per (1)). The only difference: A computes token 3545 as
   row 1 of one 2-row causal-masked SDPA call (row 0 masked over key 88, row 1
   unmasked); B computes it as a 1-row unmasked SDPA call.
6. Token 279 is a control that PASSES: A's row 0 (2-row causal, 88 visible
   keys) vs B1 (1-row unmasked, 88 keys) — the logits after 279 are bit-exact,
   so the 2-row path is bit-exact for row 0. The difference appears for row 1
   (89 visible keys) — a query-row-count / mask-path dependent SDPA kernel
   behavior in the ROCm backend.

The MTP draft module is not the suspect: its own cache is not touched by either
variant, and the main-model state diverges before any draft-module involvement.
The L15 q_norm paradox (run-786) is not implicated — layer 15's divergence is
downstream of layer 4.

Magnitude: the input difference entering layer 4 is small (conv maxabs 1.6e-2 in
qkv space; ssm maxabs 7.2e-4 with 98% of the 524288 ssm elements differing by
tiny amounts — consistent with a rank-1 delta-rule update touching every
element). It grows with depth (conv maxabs → 1.25e-1 at layers 28/30) and ends
as final-logit divergence: 192967/248320 entries, maxabs 0.125.

## Verdict

- First diverging layer (state): **4 (gdn conv+ssm)**; full cascade 4→31.
- First diverging computation (root): **attention/SDPA output of layer 3 for the
  second token (3545) — 2-row causal SDPA (row 1) vs 1-row unmasked SDPA** —
  i.e. the attention-family hypothesis (a) from run-786, refined: not the KV page
  layout (k/v bit-exact), but the SDPA kernel's row-count/mask-path dependent
  computation for the second query row.
- GDN computation: exonerated again (layers 0-3 bit-exact; layer 4 divergence is
  input-driven).
- Consequence for m2-vs-m0: every T=2 verify in m2 that differs from the chained
  T=1 path does so because the second token's attention output (and hence every
  downstream layer state) is computed on a different SDPA path — the draft
  verification in m2 is systematically not the same computation m0 would run.

## Next step (m4)

Direct SDPA A/B at layer 3: from the saved captures, compare
`SDPA(Q_2row, K_89, V_89, causal)` row 1 vs `SDPA(Q_1row, K_89, V_89, none)`
(plus the intermediate hidden after layer 3 for token 3545) to confirm the
kernel-level difference and localize it (mask path vs tile/row-count path) in
the ROCm SDPA implementation.
