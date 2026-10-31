# PR #6 adversarial re-review — round 2 (run-543 follow-up)

- **PR**: marzukia/mlx-rocm #6, head `k1-bench` @ `f5ba23a` (f5ba23a886a2f4480f8fc60645aa45e2f05a5988)
- **Reviewer**: submitter (round 2; round 1 was agent) — different reviewer from submitter-of-record and from round 1
- **Mode**: read-only + this artifact; host gates run locally
- **Date**: 2026-10-27

## Verdict: **PASS** — 0 BLOCKER / 0 MAJOR / 0 MINOR / 1 NIT

Round-1 findings MAJOR-1 and MINOR-1 are both verified fixed at f5ba23a.

## MAJOR-1 (vendor fork re-pin) — FIXED (verified)

Fix taken = option 2b (documented overlay). Verification:

**Documentation completeness (docs/vendor-overlay-k1.md + PR body "Vendor note")** — all four
required elements present and consistent across both:

1. Exact delta commit: `1051504` ("rocm/qmv: K1 v_dot4 int8-activation QMV + N=12288
   stale-output fix") — named in doc and PR body.
2. Exact guest location: `/tmp/mlx-k1-push` on guest-a, a `git worktree` of
   `/models/mlx-clone-k1` at the vendor ref — named in doc and PR body.
3. Exact ready-to-push command: `cd /tmp/mlx-k1-push && git push origin HEAD:rocm-current`
   (doc, fenced block; PR body states the commit is "ready to push").
4. Concrete build consequence: standard pinned build (`scripts/build-wheel.sh` cloning
   the vendor ref per `vendor.json`) does NOT contain the N=12288 fix or
   the K1 kernel; it needs the 2-file overlay — `qmm.hip` + `qmv_lx_kernel.hip` — applied
   before building (or use the guest `/models/venv-k1` build which already carries it).
   Re-pin procedure stated: set `vendor.json` primary.sha to the pushed sha in a PR.

Fork push itself: not verifiable from host (no push creds for <op-fork-user>; guest clone has
no credentials — 403 documented). Judged on documentation completeness per tasking: complete.

**Cross-check against the branch's committed vendor tree** — consistent:

- `vendor/mlx-rocm-upstream/mlx/backend/rocm/quantized/qmm.hip` and
  `qmv_lx_kernel.hip` both exist at f5ba23a.
- The repo-side delta is exactly the documented 2-file overlay: `b42efb6` (+210 lines,
  both files) + `99cddb5` (+34/−5, both files). No other vendor files touched.
- Content matches the description:
  - qmm.hip K1 gate admits f32 x (`x.dtype() == bfloat16 || float16 || float32`,
    `LAUNCH_LX` float32 instantiation) — as documented.
  - qmm.hip 4/8-bit tiled gate restricted to bf16/f16 x (`tiled_x_dtype_ok`,
    "The tiled kernel is only instantiated for bf16/f16 x") — the N=12288 fix, present.
  - qmv_lx_kernel.hip is a new file: sxq staged as [even x | odd x] per 8-value group for
    4-bit (comment at line ~108) and warp-reduce of acc before the out write (line ~167) —
    both 99cddb5 fixes present.

## MINOR-1 (speedup wording) — FIXED (verified)

- `reports/k1-bench.md` Verdict now reads: "1.05-1.33x at every N (cols=16 medians:
  1.33/1.05/1.14/1.21/1.27 for N=1024..12288), and the advantage does not monotonically
  grow with N — it dips to 1.05x at N=2048, then recovers (1.14x/1.21x/1.27x)".
- PR body: "1.05-1.33x at every N (cols=16 medians: 1.33/1.05/1.14/1.21/1.27 for
  N=1024..12288); the advantage does not monotonically grow with N (it dips at N=2048)".
- Grep across `scripts/`, `docs/`, `reports/`, `src/`, `tests/`, README, PLAN: no surviving
  "1.27-1.33x"; the only "grow with N" hit in the whole tree is the corrected negative
  phrasing above. The cols=16/cols=32 tables themselves were already correct.

## Gates (all green at f5ba23a)

| Gate | Result |
|------|--------|
| CI (gh, run <run-id> @ f5ba23a) | `CI` completed, conclusion **success** |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 15 files already formatted |
| `uv run python scripts/size_ratchet.py --base HEAD` | SIZE GATE: OK (0 ratcheted) |
| `uv run pytest -m "not gpu"` | 6 passed, 1 skipped |

## Findings

1. **NIT (new)**: commit message f5ba23a says "the fork"; vendor.json, the PR body,
   and docs/vendor-overlay-k1.md all correctly say "the vendor ref". Typo confined to the
   commit message; no action required before merge (or fix in a follow-up).

No other issues. The overlay documentation is accurate, self-contained, and actionable;
the build consequence is stated concretely with the exact 2-file list.

**PASS.** Reviewer (round 2, submitter) is distinct from the round-1 reviewer (agent) as
assigned by the operator. Merge per ground rules by a PEER (not the submitter),
`--squash`, delete branch.
