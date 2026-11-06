# REPORT-planes — GDN plane exactness (batched S=4 verify vs sequential S=1 re-forward)

Question: is the GDN recurrent+conv state at row k (k=1,2,3; the state after k of
the 4 verify rows) of a **batched S=4 verify** bit-identical to the state after k
**sequential S=1** forwards over the same tokens [t0,t1,t2,t3]? This decides whether
per-row state "planes" (O(1) slot-select, à la llama.cpp n_rs_seq) can replace the
126 S=1 trunk re-forwards (78.3% of steps, 3.6 GiB each) that the C control pays.

## VERDICT: RECURRENCE-ORDER BUG

The GDN state at row k of the batched S=4 verify is **NOT bit-identical** to the state after k sequential S=1 forwards. Row 1 (the state after a single token [t0]) mismatches in all 3 rounds, which is the decision rule's RECURRENCE-ORDER BUG branch. The batched tape row is NOT a bit-exact stand-in for the sequential re-forward state, so per-row planes are **not viable for bit-exactness**.

## Bit-comparison table (per row k × plane × round)

Batched arm = one S=4 verify over [t0,t1,t2,t3], reading the per-row state tape
(`last_tape`, TMAX=4). Sequential arm = 4× S=1 forwards over the same tokens
(TMAX=1), reading `last_tape` row 0 after each. Both arms build an identical base
state via the same S=322 prefill of the canonical 322-token anchor (27 prompt +
295 greedy-extended). Compare is all-in-mlx (native-dtype `!=` = bitwise; the
tape arrays are fresh allocations, stable under a held reference — verified by a
separate stability probe). 24 GDN layers; the "recurrent" n_diff is over
12,582,912 elements (32 heads × 128 × 128 × 24 layers), "conv" over 589,824
(3 × 8192 × 24). **Every value is bit-reproducible across the 3 rounds** (round 1
= round 2 = round 3, identical n_diff and maxdiff).

| round | row k | plane     | identical | n_diff  | maxdiff  | first-idx |
|:-----:|:-----:|:---------:|:---------:|:-------:|:--------:|:---------:|
| 1     | 1     | recurrent | NO        | 9307148 | 6.261e-03| 0         |
| 1     | 1     | conv      | NO        | 107110  | 1.250e-01| 16384     |
| 1     | 2     | recurrent | NO        | 11001423| 9.710e-03| 0         |
| 1     | 2     | conv      | NO        | 242093  | 1.250e-01| 8192      |
| 1     | 3     | recurrent | NO        | 11009098| 6.467e-03| 0         |
| 1     | 3     | conv      | NO        | 373038  | 1.250e-01| 0         |
| 1     | 4     | recurrent | NO        | 11009358| 8.083e-03| 0         |
| 1     | 4     | conv      | NO        | 402413  | 2.500e-01| 0         |
| 2     | 1     | recurrent | NO        | 9307148 | 6.261e-03| 0         |
| 2     | 1     | conv      | NO        | 107110  | 1.250e-01| 16384     |
| 2     | 2     | recurrent | NO        | 11001423| 9.710e-03| 0         |
| 2     | 2     | conv      | NO        | 242093  | 1.250e-01| 8192      |
| 2     | 3     | recurrent | NO        | 11009098| 6.467e-03| 0         |
| 2     | 3     | conv      | NO        | 373038  | 1.250e-01| 0         |
| 2     | 4     | recurrent | NO        | 11009358| 8.083e-03| 0         |
| 2     | 4     | conv      | NO        | 402413  | 2.500e-01| 0         |
| 3     | 1     | recurrent | NO        | 9307148 | 6.261e-03| 0         |
| 3     | 1     | conv      | NO        | 107110  | 1.250e-01| 16384     |
| 3     | 2     | recurrent | NO        | 11001423| 9.710e-03| 0         |
| 3     | 2     | conv      | NO        | 242093  | 1.250e-01| 8192      |
| 3     | 3     | recurrent | NO        | 11009098| 6.467e-03| 0         |
| 3     | 3     | conv      | NO        | 373038  | 1.250e-01| 0         |
| 3     | 4     | recurrent | NO        | 11009358| 8.083e-03| 0         |
| 3     | 4     | conv      | NO        | 402413  | 2.500e-01| 0         |

## Bug-guard and supporting evidence

- **Determinism self-check: PASS.** Two independent S=322 prefills into two fresh caches give a bit-identical base GDN state (NDIFF=0). The base state is therefore identical across the two arms.
- **Sequential-arm self-consistency: PASS.** The sequential tape row 0 equals the sequential cache leaf of the same forward (seq_rec_vs_leaf and seq_conv_vs_leaf = YES, n_diff=0, maxdiff=0, every boundary, every round). The sequential arm is internally consistent, so the batched-vs-sequential mismatch is a real difference between the two arms, not a probe artifact.
- **Secondary cross-check: consistent.** The batched tape row vs the sequential cache leaf (rec_vs_leaf, conv_vs_leaf) mismatches by exactly the same n_diff and maxdiff as the tape-vs-tape comparison, confirming the tape and the leaf track the same (divergent) states.
- **Bit-reproducible across 3 rounds.** Round 1 = round 2 = round 3 to the exact integer (n_diff) and exact maxdiff — deterministic, not noise.
- **Wheel untouched.** libmlx.so md5 = a0057c220d31a0ec87a8b373a97625e8 both before and after the run (the wheel-of-record).

## Magnitude and fingerprint

The mismatch is near-total in element count but tiny in magnitude: ~74–87% of recurrent elements differ and ~27–68% of conv elements differ, with maxdiff of ~6.3–9.7e-03 (recurrent) and 0.125–0.25 (conv). At the states' magnitudes (recurrent ~1–2, conv ~46), these are ~1–2 bf16 ULPs — the classic fingerprint of a **floating-point summation-order difference**: the fused TMAX=4 kernel computes the per-row state with a different accumulation order than the four separate TMAX=1 kernels. The recurrence, as implemented in the fused step, is therefore not order-independent and does not reproduce the sequential re-forward state bit-for-bit.

## Interpretation and connection to the prior F-REFWD result

The probe answers the bit-exactness question: the batched per-row tape is NOT bit-identical to the sequential re-forward state, at any row, in any round. Per-row "planes" (O(1) slot-select) would therefore produce a state that is NOT the one the C control's S=1 re-forward produces, so they are **not a bit-exact replacement** for the 126 re-forwards.

This is consistent with the prior F-REFWD result: F (tape + deferred commit) token-diverged from C (re-forward), while F-RF (re-forward state source) was bit-identical to C. This probe now shows the root: the tape row differs from the re-forward by ~1–2 ULP (recurrence-order), which is the source of the F-vs-C divergence. The downstream token-level impact of a ~1–2 ULP state difference is a separate question and is out of scope for this bit-exactness probe.

## Reproduction

- Wheel: venv-cf, libmlx a0057c220d31a0ec87a8b373a97625e8 (md5-verified pre/post).
- Config (13-var, config-of-record): GDN_TAPE=1, GDN_FUSE=1, GDN_T3=1,
  QMV_NO_MR=1, QMV_MR_LX=0, SDPA_S4VA=0; the 3 shape levers (QMV_ROWS,
  QMV_ROWS_MIN_N, QMM_M3) UNSET (presence-gated). No wheel change (code-first:
  both arms use the existing gdn_fused_step_tape op at TMAX=4 / TMAX=1).
- Env on guest-a (<tailnet-ip>, RX 9070 XT / gfx1201, ROCm 10.1):
  LD_PRELOAD=libhsa-runtime64.so.1, HCC_AMDGPU_TARGET=gfx1201,
  MODELS=<models-mount>, model=<models-mount>/qwen3-9b-mlx.
- Run: flock ~/gpu.lock, 25 s idle pre-check, probe bench/planes-exactness.py
  (3 rounds), nohup. SCLK sampling unavailable on box (no rocm-smi); idle/swap
  verified instead (swap 8191 MB stable, no non-monitor compute procs).
- Probe: bench/planes-exactness.py; as-run log: bench/plane-exactness-run.log.
