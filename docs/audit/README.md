# docs/audit/ — lane audit records (PR #26 checkpoint, 2026-11-07)

This directory is the single home for the scattered REPORT-*/NOTES-*.md lane records (root + reports/ + reviews/) and the run-state digest (WORKSTATE.md), consolidated by the checkpoint PR #26 (branch `checkpoint-20261107-92-23`). It holds, in production form:

1. **Four stream digests** — consolidated narratives, one per campaign, with the key numbers and verdicts (these are the entry points):
   - `slot-path-mtp-verify.md` — stream 1: SDPA S4VA probe (ADOPT), R5
     correction-fusion (net-negative, dropped), fix rounds 1-2, graph-capture NO-GO, M3 verify-efficiency (candidates REJECTED, LX0 knob ADOPTED), GDN crash/tail-fusion/rereview, s62 layerdiff + numpy RCA.
   - `int2-head-campaign.md` — stream 2: int2 scalar port (R4c) -> vectorized
     int2w -> VGPR v2 -> VGPR v3 (570.8 us, GATE PASS) -> step-level A/B (0% draft acceptance, NO ADOPT) -> cheap-head follow-up (4-bit MTP module ADOPTED; batched argmax INFEASIBLE).
   - `draft-acceptance-probes.md` — stream 3: stack assembly (49.07, R5
     dropped), TMAX=3 + postnorm (84.26, degenerate class), ngram fill (+34.2% DECLINED, remediation REJECTED), draft roofline (67.9% of floor, bandwidth-dominated), draft-gate probe (+15.7% mechanism, lossless gate FAILS), K2 verification + T-invariance, and the honest lossless hybrid fold (69.43 t/s, the stack number of record).
   - `qmm-verify-kernel.md` — stream 4: the `qmv_wide_mr3_kernel` shared-dequant
     M=3 lever (FMA:dequant 1:1 -> 3:1, no LDS, no x-staging; ADOPT `MLX_ROCM_QMM_M3=1`, +16.9% / -18.1% wall on the degenerate tape=1 trajectory). 2. **`pr22-consolidation.md`** — the prior (PR #22) consolidation record. 3. **The moved originals** — the reports/ + reviews/ audit records, moved verbatim (hard-wrap reflowed to one-paragraph-per-line; see the mapping table below). The four stream digests summarize the root REPORT-*/NOTES* files (deleted by this PR) plus these originals; the originals carry the tables, protocols, and evidence. Also moved: earlier lane records not covered by the four streams (gdn-t2-fast, k1-bench, k2-r2-suite-evidence, k2-r4-guards.log, k2-r4-suite-full.log, mtp-bench, mtp-findings, opt-gauntlet-areaB-backend, REPORT-phase2).

## Stack number of record

The evolution, with the degenerate-trajectory class flagged (the S=3 pure-tape `MTP6_TAPE=1` protocol locks into a single-token fixed point — 1919 attractor canonical, 883 on the anchor prompt; `hybrid` is non-degenerate):

| # | t/s | regime |
|---|---|---|
| 1 | 49.07 | stack-assembly (S=3 tape=1, R5 dropped) |
| 2 | 84.26 | TMAX=3 + postnorm (degenerate tape=1 class, 113.2% of AR 74.44) |
| 3 | 105.86 | + QMM_M3 (degenerate tape=1, the t2011 record) |
| 4 | **69.43** | **honest lossless** — hybrid trajectory + QMM_M3, 13-var config, 6/6 ALLTOKS-identical (the number to cite) |

Config of record (13 vars): `MLX_MTP=3 MTP6_TAPE=hybrid MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_T3=1 MTP6_HEAD_H=postnorm MTP6_DRAFTGATE=1 MTP6_DGATE_T0=16 MTP6_NGRAM=1 MTP6_NGRAM_N=4 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1`.

## Checkpoint pin (record 92.23)

The 92.23 t/s record (stamp commit 0cb21ec; venv-slot, libmlx md5 259f9aeefe3ec02f8993fe562f5ee62b) is pinned at `pins/venv-slot-259f9aee-20261107/`: the source tree exists only as a box-local worktree (HEAD 34bbe89 = "Phase B2: GdnFusedStepSlot op", parent a7ef8c1 = the JOB 2 af341e99 build state), absent from the fleet mirror and the GH fork. Pin contents: head tar (2046 files), tracked delta (1 file, +38: the uncommitted int2 bindings `lm_head_int2` + `lm_head_int2_wide` in python/src/fast.cpp), 0 untracked source files. Record identity: record stream a55aac5c21d2acf636d01c0f88b89856 (ALLTOKS md5), TPK 2.080, D1 355-class (69.3%), D2 55.8%, REFWDS=0; the 2026-11-06 20:34-20:43Z re-measure median was 92.71 (92.85/92.71/92.36, DPM auto). Bit-exact 3/3 across the 24 state layers; 6/6 one-stream deterministic; the slot lever is -0.78 ms/step. The 69.43 wheel is pinned at `pins/record-a0057c22-20261106/` (libmlx md5 a0057c22, 18 `qmv_wide_mr3_kernel` symbols, HEAD 75ca881).

## WORKSTATE digest (folded; file deleted by this consolidation)

`WORKSTATE.md` carried four run digests, folded here:

- **run-1045 (ent4 entropy-compress, final):** per-tensor E<3.9 gating (default
  `--gate 3.9`) on the final 9B model was a no-op (250/250 tensors E <= 3.754, all compressed; file 4,192,642,800 B; x1.0696 = run-1042). The E discrepancy vs run-1044 was two different model builds (the -clean build's E 3.9707 vs the final model's 3.713/3.7368 aggregate). Lossless gates green: CPU verifier 250/250 bad=0, GPU kernel gate 3/3 bad=0, full-stack A/B byte-identical. A/B bench: 64-token arm ~0.1% overhead (65.91 vs 65.85 median); 29k-ctx decode parity (LOSSLESS-GATE=PASS both).
- **run-1059 (HIP graph-capture of the AR decode step):** the decode-capture
  fault is deterministic on the SECOND capture/launch; NO_LAUNCH clean, NO_COMMIT faults => the fault requires hipGraphLaunch replay; the faulting kernel is `gather_rows_kernel`, at a computed (src + idx*row_size) VA far outside the 2 GB arena. Suspect: launch-time param copy from a dead VA or a stream-ordered (deferred) free of a still-referenced buffer. Never root- caused; the guest VM was rebooted mid-run and /models/tmp wiped at boot. Consistent with stream 1's structural NO-GO verdict.
- **t1226 (fused-T4 GDN verify, K=3 S=4):** T=4 kernel extension (TMAX
  1/2/4, generalized sliding conv, F-branch S<=4): probe 14/14 PASS (bit- exact + invariance), K=3 lossless A/B byte-identical, fused-AR 81.10 tok/s, GDN_FUSE_CALLS 2880 = 120 forwards x 24 GDN layers. K=3 re-bench 34.68 tok/s (+1.1% vs unfused): wall/verify 94.49 ms vs the 49.38/40.45 ms break-even — **correctness PASS, speed hypothesis FAIL** (the M=4 QMV generic path, not the GDN chain, is the T=4 bottleneck). Review-fix round 2 (t1240): 73 hard-wrapped prose lines reflowed, all 9 patch hunk headers re-verified.

## Mapping table (source -> destination, this consolidation)

| source | destination |
|---|---|
| `WORKSTATE.md` (root) | folded into this README (file deleted) |
| `REPORT-consolidate.md` (root) | `docs/audit/pr22-consolidation.md` |
| root `REPORT-*.md` (20 files) + `NOTES*.md` (5 files) | folded into the four stream digests (files deleted) |
| `reports/` (19 files: 17 .md, 1 .txt, 1 .log) | `docs/audit/` (verbatim; .md reflowed) |
| `reviews/` (4 files) | `docs/audit/` (verbatim, reflowed) |
| `patch/refwd-m1-port.patch` | `docs/audit/reference/refwd-m1-port.patch` (prepared but never applied, and not hunk-wellformed under the patches/ gate — kept as a record, not a live patch) |
| `notes/vendor-p3/` (8 files) | 1 byte-identical to vendor (`qmv_lx_kernel.hip`) deleted; 7 differing moved to `docs/audit/reference/vendor-p3/` |
| (new) | `docs/audit/{slot-path-mtp-verify,int2-head-campaign,draft-acceptance-probes,qmm-verify-kernel}.md` — the four stream digests |
| (new) | `docs/audit/README.md` (this file) |

## Check lines (run at consolidation)

- Persona grep over the full diff + commit messages: 0 substantive hits.
- Hard-wrap detector (line_wrap_check) over `docs/audit/`: 0 hits.
- Size gate (`scripts/size_ratchet.py --base HEAD`): .md files are not AST-gated (Python only); no >800-line Python file added.
- CI on the push head: lint-and-check + size green.
