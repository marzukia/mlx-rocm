# Record tree pin - wheel a0057c22 (venv-m3k), 2026-11-06

The source tree that produces the record wheel (libmlx md5 a0057c22, 69.43 tok/s config) exists ONLY as a box-local worktree on neon (/home/andryo/vendor-m3k). HEAD 75ca881b2e82f7b4069c8564a203792aa16d8a29 is absent from the fleet mirror and from the GH fork. This branch pins it:

- head-75ca881.tar.gz: git archive of HEAD (clean base tree)
- tracked-delta.patch: git diff HEAD (82 files, +6336/-694: the T3 route + the four levers QMM_M3/SDPA_S4VA/QMV_ROWS/REFWD_M1 + everything else perf-relevant)
- untracked.tar.gz: 16 untracked source files (incl. lm_head_int2.hip, lm_head_int2w.hip, qmv_wide_mr3_kernel.hip, gdn_fused_step*.hip, graph_decode_pilot_logic.h)
- HEAD-sha.txt: the base commit sha

Reconstruction: extract the head tar, apply the patch, extract the untracked tar over it. Snapshot taken 2026-11-06 04:45 UTC by cloudy (read-only on the box). Worker-section raw box paths in the t2223 report addendum need a persona scrub before this branch folds into #22.

---

LANE tree pin (vendor-r4v, wheel 8acf13fe), snapshot 2026-11-06 05:00 UTC:

- same HEAD 75ca881b2e82f7b4069c8564a203792aa16d8a29 (reuses head-75ca881.tar.gz)
- lane-delta-r4v.patch: git diff HEAD (82 files, +6329/-468)
- lane-untracked-r4v.tar.gz: 12 untracked source files

The record-vs-lane perf delta for the merge plan = tracked-delta.patch minus lane-delta-r4v.patch (plus the 4 m3k-only untracked files listed in the t2223 addendum).
