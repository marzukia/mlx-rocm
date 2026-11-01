# Adversarial review — PR #4 (pr3-hybrid-fused-wheel-durability, head 8ff0d39)

Fresh read-only session. Verified against the actual upstream state: the vendor fork
(lifeofpiboy/mlx, public), stock mlx-lm 0.32.2 (fresh pip download), repo main da9eeca.

## Verdict inputs (check lines)

| Check | Result |
|---|---|
| `uv sync --locked && uv run ruff check . && uv run ruff format --check .` | **PASS** (All checks passed; 79 files already formatted) |
| `uv run pytest -m "not gpu"` | **PASS** (21 passed, 1 skipped) |
| `uv run python scripts/size_ratchet.py --base origin/main` | **PASS** (SIZE GATE: OK, 2 ratcheted) |
| Persona-leak grep (diff + commit: marzuki-* hostnames, 100.64/10 IPs, andryo@, /home/andryo, neonvm@, 192.168.122.x, /mnt/models, *.junkyard.sh, phishasu) | **PASS** (0 hits; author line is the mandated fleet identity, excluded) |
| Placeholder-violation sweep (home paths, literal hostnames, tailnet IPs, passwords, ssh/scp channels in new files) | **PASS** (0 hits; all host paths are `<placeholders>`) |
| Line-wrap scan (PR diff + new commit, docs/report .md, commit messages, code comments) | **FAIL** — see F1 |

## Scope verification (the claims)

(a) Graft identity — VERIFIED in-repo:
- `patches/gdn_fused_step/gdn_fused_step.patch` is present on main (da9eeca), md5 `b9693851…`
  (matches the claimed prefix), byte-identical between main and head (not touched by the PR).
- Dry-run against the actual vendor base `75ca881b2e82f7b4069c8564a203792aa16d8a29`
  (fetched from the vendor fork): `patch -p1 --dry-run` = 6 files checked, 0 rejects,
  0 fuzz (rc 0); `git apply --check` also clean.
- Documented file list matches the patch exactly: 5 modified
  (CMakeLists.txt, fast.cpp, fast.h, fast_primitives.h, python/src/fast.cpp) + 1 new
  (mlx/backend/rocm/gdn_fused_step.hip).
- The patch's `gdn_fused_step.hip` md5 recomputed via `git apply` =
  `d8d4a30f37252991330e6517ece89123` — matches the claimed prefix `d8d4a30f…`.

(b) Model patch — VERIFIED:
- Applies cleanly to stock mlx-lm 0.32.2 `qwen3_5.py` (fresh `pip download
  mlx-lm==0.32.2`): `patch -p1 --dry-run` rc 0, 1 file, 0 rejects, 0 fuzz; real apply rc 0,
  no `.rej`/`.orig` left behind.
- Post-apply file is **25,912 bytes — exactly the claimed installed-file size** (strong
  corroboration of the byte-identical round-trip claim; the installed file itself is not
  in-repo, so full byte-identity is host-side evidence).
- Marker guard: `FUSE_BRANCH_PATCH` present after apply (grep guard short-circuits the
  re-apply = no-op); `MTP_INJECT_PATCH`/`PRENORM_PATCH` present; the fused branch is
  gated `MLX_ROCM_GDN_FUSE=1 and S == 1 and B == 1` with the `GDN_FUSE_CALLS` counter;
  the MTP module is only built when `mtp_num_hidden_layers > 0` and prenorm plumbing is
  only active when `return_prenorm=True` — inert under `MLX_MTP=0` as claimed.
- Patched file compiles (`py_compile` OK).

(c) Recipe — leak-clean (grep above), but the Step 1 command is defective, see F2.

(d) REPORT.md claims — the in-repo/verifiable claims (graft dry-run, md5s, file list,
  25,912-byte round-trip, SHA relationships: 75ca881 is the vendor-fork
  `tail-fusion-gdn` branch tip; 9320ee9 exists in this repo as the t1082 workline tip;
  main = da9eeca) all check out. Wheel-side claims (GdnFusedStep symbols, venv state,
  80.93/72.57 tok/s, r4d cosmetic-scrub delta) are host-side evidence, not
  re-verifiable here — accepted as recorded evidence, not asserted false.

## Findings

F1 (BLOCKER — line wrap): hard-wrapped prose in all four prose surfaces, at ~72-80
column widths, continuation lines mid-sentence:
- `BUILD-RECIPE.md`: 48 mid-sentence line ends (e.g. L3 "...fused-AR mlx wheel that" /
  L4 "measured **80.93 tok/s**..."; L5 "...All host-specific" / L6 "paths are
  placeholders:...") — the entire file is wrapped.
- `REPORT.md` PR3 section: 43 mid-sentence line ends (e.g. "no-fuse) existed only as a
  dirty vendor tree..." continuations).
- Commit 8ff0d39 message: 12 mid-sentence line ends wrapped at ~80 cols.
- `bench/mtp-bench3-side.py`: 54 mid-sentence comment/docstring line ends
  (e.g. header L11/L13-L18, docstring L49-L51).
The reader client wraps; this must be unwrapped (one-sentence-per-paragraph, or
formatter-owned wrapping only).

F2 (MAJOR — recipe Step 1 cannot fetch the pin): `git clone --depth 1
<vendor-fork-url>` checks out the fork's default branch, which is `d9add9d` (upstream
mlx, no ROCm port) — the recipe's own gate "must print
75ca881b2e82f7b4069c8564a203792aa16d8a29" fails as written. `75ca881` is the tip of
the `tail-fusion-gdn` branch (verified via `git ls-remote`); no other ref points at
it. Step 1 needs `git clone --branch tail-fusion-gdn --depth 1` (or an explicit
fetch+checkout of the pin) for the recipe to reproduce the base.

F3 (minor, observation): `REPORT.md` PR3 section names the vendor fork
(`lifeofpiboy/mlx`) while `BUILD-RECIPE.md` uses the `<vendor-fork-url>` placeholder —
inconsistent masking (the fork name is a personal account, though not a hostname/IP and
outside the mandated grep list).

F4 (minor, observation): recipe build step uses `CMAKE_PREFIX_PATH=/opt/rocm` while
`vendor.json` documents the 10.1 core at `/opt/rocm/core-10.1` — unresolvable
inconsistency, worth aligning.

## VERDICT

VERDICT: FAIL — F1 (hard-wrapped prose in BUILD-RECIPE.md, REPORT.md section, commit
message, bench comments) and F2 (recipe Step 1 cannot reproduce the pinned base).
All functional/graft/model-patch claims verified true; leak checks clean; hermetic CI
checks green.
