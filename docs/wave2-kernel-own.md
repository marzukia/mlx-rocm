# AUDIT WAVE 2/3 — P1-d option-B: the kernel-own error + the 4 root-cause fixes + F2 coverage

Date: 2026-11-27 (zero-GPU xhigh wave run; CPU-only verification)
Auditor: failure-wave wave-2 (fresh session, independent of the submitter)
Scope: the option-B leg-3 terminal claims on (1) the kernel-own error 2.722e-5 /
rel 6.592e-5 as a sound verification, (2) the 4 root-cause fixes, (3) F2 matrix
coverage. All evidence read from the record: findings card 4b0566b, kernel
commit 42dd9c6 (mirror branch p1d-forkb-optionb-kernel), kqfg deq_elem ground
truth, /tmp scripts on the box (read-only), worker session log, wheel md5.

## VERDICT: CONFIRMED (all three claims hold; three non-blocking notes below)

## 1. ISOLATION VALIDITY — the kernel-own error is a sound verification: CONFIRMED

Reference definition (reconstructed from /tmp/xq2.py + /tmp/g1v2b.py + mmq.hip):
the reference = the d2s6-quantized x, then an exact f32 numpy matmul.

- The numpy d2s6 replica was checked line-by-line against the in-tree
  `mmq_quant_in_q8_1_mmq` d2s6 branch: amax over the two 64-halves (exact,
  order-free), `d_inv = 127/amax` / `d = 1/d_inv` (same single f32 roundings),
  `t = rint(v*d_inv)` clamped to ±127 (RNE on both), the 6 per-16 qsums are
  INTEGER sums (max |qsum| = 127·16 = 2032 < 2^24: exact in f32 in ANY
  summation order — the numpy .sum order difference vs the GPU loop is exactly
  zero), f16 RNE storage of d0/d1 and d_g·qsum (numpy .astype(f16) = RNE =
  `__floats2half2_rn`), the per-group d mapping [d0×4, d1×2] matches the GPU
  `d_g = (16g < 64 ? d0 : d1)` for g = 0..5, min for i < 96 only. The replica
  is exact modulo f32→f16 boundary ties (measure zero; even a flip costs one
  f16 ulp of s_g/16 ≈ 5e-6·|x| — below the measured kernel-own).
- g1v2b.py and xq2.py use the SAME x (identical `RandomState(42)` + first
  `normal(0, 0.1, (2, K))` call sequence) and the SAME W-parsing/dequant code
  (byte-identical blocks; KQFG-CHECK M=2 = 2.45e-6 G1-validates that W
  dequant against the kqfg ground truth). So the three measurements (total,
  floor, kernel-own) share one x and one W.
- The decomposition is structurally exact, not an approximation:
  `total = (y1n − ref1) = (y1n − ref_xq) + (ref_xq − ref1)` = kernel-own +
  floor (elementwise telescoping, same x, same W). max|total| ≤ max|floor| +
  max|kernel| = 9.458e-2 + 2.722e-5; the measured total rel 2.290e-1 equals the
  floor rel 2.290e-1 to 4 digits — the required condition is kernel ≪ floor
  (2.722e-5 / 9.458e-2 = 0.029%), satisfied. The 4th digit cannot move; the
  5th could — the 4-digit agreement is exactly what the decomposition predicts.
- Magnitude re-derivation (why 2.7e-5 is consistent, not suspiciously good):
  every stored-value error is SHARED between kernel and reference — the f16
  W-side d/dmin (both read the same f16 file values; f16→f32 is exact), the
  int8 t codes, the f16 dy/my (the reference dequants the same f16 values).
  Kernel-own therefore = f32 arithmetic only: the w8 op order (the kernel's
  (q·s)·d − c2 vs the ground-truth (d·s)·q − c2 differ by ≤ 2 ulp per term —
  note the kqfg sub_exact order, the kernel's (q·s) is an exact integer product
  before one f32 multiply), v = dy·t + my (FMA vs 2 roundings, ≤ 1 ulp per
  term), and the 5120-term f32 accumulation order (systematic bound
  ε_f32·K·|partial| ≈ 1.2e-7·5120·0.07 ≈ 4.3e-5; random-walk scale
  √5120·1.2e-7·0.07 ≈ 6e-7; implied max|ref| = 2.722e-5/6.592e-5 = 0.413 is
  consistent with x ~ N(0, 0.1), K = 5120). Expected band ~1e-5..5e-5;
  measured 2.722e-5 is inside it.
- Corroborating before/after: the worker session log shows the pre-fix-4 build
  at kernel-own = 2.351e-3 (rel 5.694e-3) — the fragment-hole class — which
  collapsed to 2.722e-5 after the 4B-load fix (an 85× drop). The session log
  also shows the final floor 9.458e-02 / 2.290e-01 and kernel-vs-REF
  9.457e-02 / 2.290e-01 exactly as the card records.
- Evidence binding (all verified this run): the 3 kernel files at 42dd9c6 are
  byte-identical to the SoT worktree (work/3587 vendor tree, whose git log is
  42dd9c6); the deliverable wheel on the box has md5
  23baab1985fc94042f06d3a95834d947 = the card's recorded md5.

## 2. THE 4 FIXES — CONFIRMED (two notes)

Verified against the kqfg `deq_elem` ktype==10 (the bit-exact G1 ground truth,
2.45e-6-verified) and the numpy refs; additionally brute-forced in numpy over
ALL 16 window bases × 16 rows:

(i) **Warp-window base `b = (n0 + i0) % 256` — CORRECT.** n0 = it·I with I = 128
    (Q2_K rdna4 config: I = 128, J ∈ {16,32,64}; the M=1 gate picks J_best =
    16 → the x_file path; gates N%256 == 0, K%256 == 0, N%I == 0 all hold for
    the G1 battery N = 1024, K = 5120). The 16 rows e = b + r (r < 16, b
    16-aligned) stay in one 256-N superblock: max b%128 = 112, 112 + 15 < 128.
    The old `b = n0 % 256` would have made warps 1–7 read warp 0's window —
    the fix is necessary and correct.

(ii) **Code-byte base `16 + 32·(b/128) + 16·((b/16) & 1)`, shift
    `2·((b/32) & 3)` — CORRECT.** Brute-force: for all 16-aligned b, cbb + r
    and the shift equal the ground-truth per-element form
    (byte `16 + 32·(e/128) + (e%32)`, bit `2·((e/32) & 3)`) and the xq2/f2p
    numpy forms (which are the same formula with the qs[64]@16 view offset).
    The no-carry lemmas hold: e/128 = b/128 (checked), e%32 = b%32 + r
    (b%32 ∈ {0,16}, r ≤ 15), b%32 = 16·((b/16) & 1), e/16 = b/16 (scale byte
    `sci = b/16` = ground-truth `blk[e/16]`). The 16-group pattern is 0,16,0,16
    (computed). NOTE (non-blocking): the card's recorded old form
    `+ (b%32)` is mathematically identical to `+ 16·((b/16) & 1)` for
    16-aligned b, so fix (ii) is a defensive re-formulation (robust for any b,
    documents the pattern); the load-bearing window fix is (i). The "4 real
    bugs" characterization is slightly generous for (ii); the FINAL state is
    verified correct either way.

(iii) **dmin f16 @82 via `hp[1]` — CORRECT.** `dd = u32 at blk+80` covers bytes
    80–83; `hp[1]` = bytes 82–83 = the dmin f16 @82, matching the layout and
    the ground-truth `f16w(blk + 82)`. The old `&dd + 1` on the u32 read
    bytes 84–85 = the next block's sc[0] — a real bug (systematic wrong c2).
    Fix correct.

(iv) **Two aligned 4B loads replacing the unrolled 8 byte-loads —
    PROVABLY EQUIVALENT AT THE SOURCE LEVEL; the load-bearing fix is
    confirmed.** Alignment: block stride 84B = 4·21 → every block base is
    4-aligned; +80, +cbb (≡ 0 mod 16), +8h (≡ 0 mod 8), +4 are all 4-aligned →
    both u32 loads are aligned. Coverage: `b4a`/`b4b` = bytes
    cbb+8h .. cbb+8h+7 exactly; extraction `byte = (b4 >> (8·(l&3)))` is the
    correct little-endian byte l&3; rows r = 8h + l with h ∈ {0,1} (lane >> 4)
    cover 0..15 → the full 16 code bytes per window per k. The comment's
    "16B-aligned base" phrasing is inaccurate (only 4B alignment is guaranteed
    by the 84B stride) but the loads need only 4B alignment — cosmetic.
    Residual theoretical risk: the compiler re-merging the two adjacent aligned
    4B loads into one 8B vector load (the original bug trigger) — not provable
    from source, empirically excluded by F2 all-green on the shipped build and
    the kernel-own 2.35e-3 → 2.7e-5 collapse. Also verified: the y-side read
    (dy2 = half2 word 0, sg2 = words 1..3, t = word 4 + (k>>2) bit 8·(k&3),
    36-word/144B packed rows, words 36..39 zeroed and never read) matches the
    d2s6 layout and the host tile load; the x base
    `((kb0·256 + k00)·(N/256) + c)·84` matches the [K, N/256] file layout
    (stride_kbx = N/256 in the x_file branch) and the two 128-K halves
    (k00 = 0/128 vs the 36-word y rows k128 = kb0·2 + {0,1}); per-thread sum
    slots [J·I/(nwarps·warp_size)] with the 93-layout writeback
    `sum[(j0>>4)·8 + l]` (J = 16: 8 slots per thread, each (i,j) fragment
    touched by exactly one lane per k — no races); the E0 min-correction
    1.409e-2 = 0.0625·max|W[0,:]| (the d2s6 min term s_g/16, ≈ 1/16, to 3
    digits; the 15 sibling-group elements contribute a statistically ~0 term on
    this weight — the worker log's own arithmetic confirms 0.062435·0.2266 =
    0.014147 vs measured 0.01409).

## 3. F2 MATRIX COVERAGE — CONFIRMED AS END-TO-END (F2 + G1); the F2-only claim
    is slightly overstated; uncovered classes named

F2 instantiates (per /tmp/f2p.py, which writes via the ground-truth formula and
checks the target rows read exactly 3 — the printed 3.0 is the rounded int; the
unrounded 2.99982 from the earlier probe run is consistent with the v ≈ 1.0625
d2s6 min correction on the unit x): window bases b ∈ {0, 16, 32, 128, 192}
(cbb bases {16, 32, 48}; shifts {0, 2, 4}), superblock c = 1 (F2f), and K
rows {0, 128 (the k00 = 128 half), 255 (last k of the 2nd half), 5119 (last k
overall, kb0 = 19)}.

UNCOVERED CLASSES in the F2 matrix (named):
- **code-byte base 64** — b ∈ {144, 176, 208, 240} (the 16-group-with-offset in
  the 2nd 128 half: 16 + 32 + 16; the sum-of-two-terms combination is
  unprobed in F2);
- **shift 6** — b ∈ {96, 112, 224, 240} (the (b/32)&3 = 3 mask-wrap class);
- all F2 k-cases use window 0 (b = 0) only — no (k, b) combinations beyond
  k-stride × window-0;
- no negative-side check (non-target rows should read 0) and F2 exercises only
  s = 1 / mn = 0 scale bytes.

Why this is not a hole in the overall verification: the G1 battery (N = 1024 =
4 superblocks, I = 128 → it = 0..7) executes ALL 16 window bases per
superblock — b = 0..240 including {144, 176, 208, 224, 240} — on real data
with real non-zero scale/min bytes. A base-64 or shift-6 misread would
misread 16 whole N rows over all 5120 k (O(signal) error, ≳ 1e-2, orders of
magnitude above the 2.7e-5 kernel-own and the 9.5e-2 floor max), and the
measured total (2.290e-1, 4 digits = the floor) excludes it. Combined with the
independent brute-force verification that the formula in those classes equals
the ground truth, the claim "every 16-row window / boundary / shift group /
superblock / K-row carries data" holds end-to-end; strictly, the F2 matrix
ALONE does not instantiate base-64/shift-6.

## NAMED RECOMMENDATION (for the lane; the orchestrator + cloudy rule on it)

1. **Endorse the card's named next step unchanged**: dispatch M=1 to kqfg
   (the 2.45e-6 G1-verified path), and keep the option-B mmq kernel
   env-gated (MLX_KQFILE_MMQ_M1=1) pending the op/cloudy ruling on the
   2.29e-1 x-quantization floor. This audit found no kernel-correctness issue
   that should block that plan; if the floor is ruled unacceptable for serve,
   option B stays off-by-default (env-gated only) — the kernel is verified,
   the floor is the decision.
2. **Re-specify the gate for the mmq path** (either way): the literal
   M=1-total-rel ≤ 1e-3 gate is unreachable by any mmq path (architectural);
   a sound gate for the option-B (or option-A) path is kernel-own rel ≤ 1e-3
   (measured 6.6e-5 — passes ~15×) with the floor recorded as a property, not
   a tolerance.
3. **Optional hardening (cheap, before any serve use)**: add the 2 missing F2
   classes (a b = 224 window probe covers base-48+shift-6 jointly; a
   b = 144 probe covers base-64+shift-0) so the pre-clean discriminator
   matrix closes without relying on G1. Not a correctness requirement — G1
   already covers these classes end-to-end.

## Audit method (zero-GPU)

CPU-only: kernel/ground-truth/quantizer source reading (mirror 42dd9c6 +
kqfg deq_elem + mmq.hip), independent numpy re-derivation of the window
mapping (all 16 bases × 16 rows), the error-decomposition arithmetic, the
f32 error-budget re-derivation, the /tmp script cross-checks (read-only over
ssh), the worker session log cross-check, the SoT-worktree ↔ branch
byte-identity, and the wheel md5. No GPU touched.
