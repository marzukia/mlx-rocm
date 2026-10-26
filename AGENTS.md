# mlx-rocm

ROCm backend for MLX: adopt the ROCm port, build the `mlx` wheel for gfx1201
(RX 9070 XT on marzuki-neon, ROCm 10.1), validate, serve the qMLX stack.

Purpose in one line: get qmlx-serve running a 9B-class dense Qwen3 correctly on neon.

## Commands

- format: `uv run ruff format .`
- lint: `uv run ruff check .`
- test (hermetic, no GPU): `uv run pytest -m "not gpu"`
- test (GPU, ON marzuki-neon only): `uv run pytest -m gpu`
- size gate: `uv run python scripts/size_ratchet.py --base HEAD`
- build wheel (ON neon): `MLX_VENDOR_DIR=<vendor clone> bash scripts/build-wheel.sh`
- Phase 0 smoke (ON neon, after install): `uv run python scripts/smoke.py`
- Phase 1 serve smoke (ON neon): `bash scripts/smoke-serve.sh`

## Ground rules

- main is protected: no direct pushes, no force-pushes. Branch + push-first, PR to main,
  CI green, adversarial review by an agent DIFFERENT from the submitter, then a PEER
  (not the submitter) merges `--squash` and deletes the branch.
- Submitter of record for the MVP work: monky (builds on neon). Reviews: fresh session,
  read-only, verdict PASS/FAIL, reporter separate from submitter.
- GPU tests are marked `@pytest.mark.gpu` and skip off-GPU. CI runs the hermetic set only.
- GPU work happens ON marzuki-neon (100.91.94.56, RX 9070 XT / gfx1201, ROCm 10.1 guest userland).
  Do not assume a GPU on hydrogen.
- The vendored MLX source is NOT in this repo: `vendor.json` pins fork + ref;
  `scripts/build-wheel.sh` clones it. Re-pin in a PR when the base moves.
- File size is a load-bearing signal (size gate, CI + pre-commit): >800 non-blank lines
  must be in size-ratchet.json; ratcheted files may only shrink; `--update` after a split,
  never to raise a budget without a same-PR split or explicit operator OK.
- A change adding code to a >2,000-line file must name the target module, not the god file.
