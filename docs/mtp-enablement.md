# MTP enablement recipe (as measured, 0.3.0-alpha)

MTP speculative decoding on the 9B work model is a **flag-class recipe** —
not a model-config switch. This is the recipe as measured on main
(`f5e7cd88`-class build), grounded in the recipes/protocols actually in
the tree.

## Enablement (serve)

- `--force-spec-decode` — force the spec-decode path.
- `--speculative-config method=mtp` — select the MTP method (the in-tree
  `models/qwen3_5.py` `MTP_INJECT_PATCH` builds the 1-layer MTP drafter
  module when the config asks; it reads no `MLX_MTP` — the spec-decode
  loop is driven by the env/bench class below).
- the **4-bit sidecar class**: the MTP sidecar next to the 9B weights —
  `model-mtp.safetensors` (the f32 sidecar; `MTP4_SIDECAR_F32`) or
  `model-mtp-4bit.safetensors` (`MTP_4BIT=1` / `CH_MTP_4BIT=1`; the
  `MTP4_SIDECAR` override). The in-tree conversion/quickstart
  (`docs/quickstart.md` §4) names `model-mtp-fixed.safetensors` (the
  fixed-W+1 RMSNorm build, `docs/audit/mtp-numpy-rca.md`).

The in-tree model (`models/qwen3_5.py`) carries the markers:
`MTP_INJECT_PATCH` (the drafter module build + the `mtp.*` key retention
in `sanitize`), `PRENORM_PATCH` (the pre-final-norm hidden exposure the
drafter consumes), and the `MTP_4BIT` quantized-module gate. The bench
`FUSE_BRANCH_PATCH` / model-patch lineage is `patches/qwen35_fuse_branch/`
(see `BUILD-RECIPE.md`).

## The measured record protocol (the MTP=3 + MTP6_PREFILL_FILL=1 env class)

- `MLX_MTP=3` — the spec-decode mode (in-tree mapping in
  `bench/k2arm-pinned.py`: mode 3 = the batched 3-token (S=3) verify
  forward; the K=3 record class per the 0.3.0 op record).
- `MTP6_PREFILL_FILL=1` — the prefill fill (the H4 decisive test).
- **Record: 111.01 t/s decode (MTP K=3), banked on main** (the 0.3.0
  standing; op record 2026-11-19). The in-tree record of the same
  protocol class is **111.84 t/s @1024** (`docs/w1/
  REPORT-W1-RECORD-11184-20261114.md`: the in_ba fusion ON
  (`MLX_ROCM_W1_INBA_FUSE=1`, env-gated, default off) + the zero-point
  fix, stamped 2026-11-13; supersedes the 107.56 t/s @1024 certified
  record, `docs/w1/REPORT-W1-PHASE2-REPRO-20261112.md`).
- The in-tree repro driver is `bench/gdn_seq1024_repro.sh` — it exports
  the arm env verbatim: `MLX_MTP=3 MTP6_TAPE=seq
  MTP6_DRAFT_HEAD=bf16 MTP6_HEAD_H=postnorm MTP6_MOD_BF16=1`, runs
  `bench/k2arm-pinned.py` on the in-tree model
  (`models/qwen3_5.py`, md5 `303b245b`), 1024 tokens x 3 rounds, DPM
  AUTO (unpinned), and records the sclk state per round.
- The `MTP6_TAPE=seq` arm routes the W1 sequential GDN kernel
  (`MLX_ROCM_GDN_SEQ=1` + `MLX_ROCM_GDN_T3=1` + `MLX_ROCM_GDN_FUSE=1`;
  all default-off, see `docs/env-flags.md`).
- Losslessness gates: the K=2/K=3 runs are token-identical to plain
  decode (the `ALLTOKS` prefix-match gates in
  `scripts/warm-mtp-run.sh`; #26: 92.23 t/s @1024, K=1/K=2
  token-identical).

## Pointer map

- the flag table (every env flag in the tree): `docs/env-flags.md`.
- the root-cluster reports (moved to `docs/notes/` by the 0.3.0 alpha
  prep): `docs/notes/REPORT.md`, `docs/notes/WALL-NOTE.md` (the lane
  state 2026-11-19), `docs/notes/REPORT-RELEASE020-20261112.md` (the
  0.2.0 release cut).
- the W1 record line: `docs/w1/REPORT-W1-RECORD-11184-20261114.md`,
  `docs/w1/REPORT-W1-PHASE2-REPRO-20261112.md`,
  `docs/w1/REPORT-W1-PHASE1-20261112.md`.
- the MTP audit trail: `docs/audit/mtp-*.md` (numpy RCA, r4b fused
  verify, s62 SDPA/layerdiff, the draft-gate probe report).
