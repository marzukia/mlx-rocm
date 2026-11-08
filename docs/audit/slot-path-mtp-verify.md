# Stream 1 — slot-path MTP verify campaign (digest)

Consolidated narrative digest for the verify-side / SDPA / R5 / GDN cluster. The root REPORT-*/NOTES* files of this stream are consolidated into this digest (deleted by the PR); the reports/+reviews/ originals of this stream are moved verbatim under `docs/audit/`: gdn-tail-fusion.md, k3-gdn-crash- fix.md, mtp-s62-layerdiff.md, mtp-s62-sdpa.md, mtp-numpy-rca.md, k3-gdn- rereview.md (the reviews file).

Box: guest-a (RX 9070 XT / gfx1201, 32 CU, ROCm 10.1 guest userland). Model: qwen3-9b-mlx, 32 layers (24 GDN linear-attention + 8 full-attention, interval 4), 16 q-heads, 4 kv-heads, D=256, 4-bit affine quant. All benches: flock, 25 s idle, sclk locked high, medians of 3.

## 1. SDPA S4VA bandwidth probe — ADOPT

The fused S=4 verify wall was 44.85 ms, of which the 8 attention layers were 19.96 ms (44%), flat ~2.49 ms/layer: the stock `qL < 4` gate fell into the scalar flash fallback (qL=4) at the real decode shape (kL=768 causal, H=16/HKV=4/D=256). Micro timing at the real shape: qL=1 vector 0.045 ms; qL=4 scalar flash 2.959 ms (65.6x); qL=5 WMMA flash 0.959 ms; effective BW at qL=4 was 1.09 GB/s = 0.2% of the 546 GB/s floor. The S4VA vector kernel (env-gated `MLX_ROCM_SDPA_S4VA=1`, graft on the vendor-canon tree) cut attn to 0.287 ms/layer (-88.5%), the qL=4 micro to 0.054 ms (54.7x), effective BW 59.3 GB/s (10.9% of floor), and the fused verify wall to 22.09 ms (-50.7%). Same-wheel control A/B (3x3 interleaved, 322 anchor): A (base wheel, S4VA absent) 38.36 t/s, C (candidate wheel, S4VA off) 43.91 t/s, B (candidate + S4VA on) 51.92 t/s; B vs C +18.1% (gate >=+15% PASS; B vs A +35.2% includes wheel drift). Lossless: FIRST20 identical all arms; B-vs-C first token divergence at token 32 (1-ulp SDPA argmax flip, text coherent); SDPA isolated sanity maxdiff = 1 bf16 ulp. Verdict: **ADOPT S4VA** (both gate prongs PASS); branch sdpa-bandwidth-probe, PR #23.

## 2. R5 correction-fusion — net-negative, NOT adopted

R5 (the correction-fusion tape mode) landed at 38.36 t/s on the 322 anchor, in the same 38.3-38.6 band as every other same-wheel arm, versus the 43.50-class tape=1 anchor: a net **-11.9% loss** (best arm 38.62 still -11.4%); the fused verify wall under R5 was 44.85 ms (ticket 45.07). Round-1 fix candidates (A: in-place KVAppend, +239 lines fast.cpp; B: FUSED_VERIFY_S4VA, +155 lines gdn_fused_step.hip) cut attn per-op 19.96 -> 1.23 ms (-93.9%) / 0.85 ms (-95.7%) respectively, but end-to-end A 39.16 / B 38.43 t/s — ladder 1.021x / 1.002x vs the r4v anchor. Round-2: the PRE gate (locked regime, 3-run median) was 37.47 t/s, outside the 65-66 class, so the round-2 A/B was **STOPPED at the gate** and not run; the 39.16 run was identified and reproduced (059245e hybrid behavior; the H arm reproduced it at 38.88, 1657 vs 1657 tokens, D3 48.5 vs 48.5, REFWDS 395 vs 395). Informational only (uncontrolled, 1-run): hybrid 38.67-38.88 vs canonical 37.43 (~3.5% up, unvalidated); operator judgment on the VM-vs-bare-metal genealogy was required. The 144.3 ticket target was not met by any configuration in this family (best 39.16); the verdict was that this configuration class is not the path and R5 stays unadopted. R3 (REFWD_M1 lane): on the canonical 512 protocol the number is the 40.4-class (40.38), wheel-independent (A/B/C 9/9 bit-identical streams, 72.9/61.9/57.1% acceptance, REFWDS 395); the 65.68 (128-graft) vs 40.38 (512) gap is ms-per-step (49.0 -> 78.8), i.e. a protocol/stack property (KV growth), not acceptance (flat 3.16-3.22 tok/step) and not the wheel — with `QMV_ROWS=1` the `REFWD_M1` M=1 lx re-route is off by design, so the hunk is a no-op and was **NOT ported** to the canonical vendor tree (stop branch). R3 is the lane the int2 campaign anchors against.

## 3. Graph-capture probe — NO-GO (structural)

The AR-decode graph-capture probe was a structural NO-GO: the S=4 `big_shape` path plus 2 GB decode-arena exhaustion makes per-step re-capture non-viable in this wheel family. (The separate run-1059 graph-decode memory-fault workstream — second-capture deterministic fault, faulting kernel `gather_rows_kernel`, NO_LAUNCH clean / NO_COMMIT faults, suspected launch-time param copy or deferred-free — is recorded in the WORKSTATE digest in README; it never reached a fix.)

## 4. M3 verify-efficiency candidates — REJECTED; LX0 knob ADOPTED

The S=3 verify pass ran at 41.9% of the HBM floor (target 66%+); BASE (config-of-record) was 84.11 t/s, verify wall 22.30 ms. Phase-1 env sweep: the in-kernel lx hoist is **net-negative** on gfx1201 for M<=4 shapes (the per k-tile `__syncthreads` outweighs the saved lx re-reads): `MLX_QMV_MR_LX=0` gives +5.8% t/s / -10.2% verify wall, FIRST20 unchanged; MR8 is worse than MR16 (-2.5%). Phase-2 candidate A/B (bar: verify wall >=30% cut OR t/s >=+10% same-wheel, per candidate; FIRST20 silent gate, all 12 runs token-identical):
- **QMM nostage** (`MLX_QMV_MR_NOSTAGE`, removes the 32 KB-capped x-staging so
  the K>4096 `down_proj` (K=12288) stops re-reading the 25.2 MB weight tile M times): B vs A = +1.8% t/s / -3.8% wall — **REJECT** (the re-reads were L2-cheap at M=3). B vs BASE = 91.29 t/s (+8.5%), below the +10% bar.
- **GDN 2H** (2 heads per 512-thread block): -0.2% t/s / +0.6% wall — **REJECT**
  (the shipped 32-block launch already maps one v-head per CU).
- **LX0 env knob ADOPTED** as the config-of-record env change (arm A 89.70 t/s
  = +6.6% vs BASE, wall -11.7%, existing wheel, FIRST20 identical).

Also recorded: `MLX_QMV_MR_COLS=12` OOB-faults the GPU (mcols not a divisor of N=4096; the kernel's weight read is unguarded, only the store is) — flagged for the kernel-maintainers lane, not fixed in this run.

## 5. GDN cluster — crash, tail fusion, rereview

- **GDN crash (k3)**: the GDN fused-step crash under the k3 suite was **load-gated,
  not code-deterministic**: 16/16 under concurrent load, 0/50 serialized. The fix (run-761, wheel 75ca881, venv-kfix) was a launcher pointer fix; the k3 suite then passed serialized. Adversarial re-review: **PASS** (allocator no-destroy design bounded per copy; the 3 GDN port commits verified; WIP commit fine for squash-merge; size gate exempt for C++/HIP per policy).
- **GDN tail fusion**: G1 (bit-exact vs unfused, per-layer) and G2 (bit-exact
  full-stream, 3/3 x 24 state layers) **PASS**; the launcher pointer bug was found and fixed in run-761 (75ca881). **G3 (bench gate) / G4 (A/B gate) are PENDING — the bench gate has not been run**; any "103.8 t/s" claim for the tail-fusion kernel is unsupported by this repo. Do not cite a completion.

## 6. MTP s62 layer-diff + numpy RCA

- **s62 layerdiff**: with S4VA on, the 8 full-attention layers show an SDPA
  T-dependence (verify vs AR logits differ by construction at the 1-ulp level); first divergent layer is layer 4; max abs diff on the **final logits** is **1.250e-01** (not 2.86e-01 — corrected). The 24 GDN layers are T-invariant.
- **numpy RCA**: the numpy-vs-GPU MTP divergence was a **weight-file
  norm-convention bug** — 7 RMSNorm weights are stored as `w-1` (pre-subtract) in the weight file while the numpy path applied `w` directly. Not a kernel bug.

## Verdicts in one line

S4VA ADOPT (verify -50.7%, +18.1% same-wheel); R5 NOT adopted (net -11.9%); fix rounds 1-2 net-negative (round-2 A/B stopped at the PRE gate, 37.47 t/s); graph capture NO-GO; M3 candidates REJECTED (nostage +1.8% / 2H -0.2% vs the >=+10% bar), LX0 knob ADOPTED; GDN crash load-gated + fixed; GDN tail fusion G1/G2 PASS, **G3/G4 PENDING**; s62 maxabs 1.250e-01; numpy RCA = weight norm convention (7 RMSNorm weights).
