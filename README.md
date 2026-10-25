# mlx-rocm

ROCm backend for MLX on **marzuki-neon** (AMD RX 9070 XT = gfx1201, ROCm 7.7): adopt the
ROCm port, build the `mlx` wheel, validate on gfx1201, serve the qMLX stack
(qmlx-serve, Qwen3).

Goal: qmlx-serve runs a 9B-class dense Qwen3 on neon, correctly, at usable decode speed.

## Layout

- `docs/plan.md` — the research report + phased plan (architecture, CUDA→ROCm map,
  prior art, gfx1201 findings, effort).
- `PLAN.md` — living plan (phases, current state, open questions).
- `vendor.json` — the pinned vendor fork + ref (primary/fallback) and the build flags.
- `scripts/build-wheel.sh` — build the `mlx` wheel with `-DMLX_BUILD_ROCM=ON` for gfx1201 (runs on neon).
- `scripts/smoke.py` — Phase 0 smoke: is_available, matmul/softmax numerics, 0.6B generate.
- `scripts/smoke-serve.sh` — Phase 1 smoke: qmlx-serve with a 9B-class dense Qwen3.
- `scripts/size_ratchet.py` + `size-ratchet.json` — deterministic size gate (CI).
- `tests/` — hermetic tests (CI) + `@pytest.mark.gpu` tests (neon only).

## Commands

```bash
uv sync
uv run ruff format . && uv run ruff check .
uv run pytest -m "not gpu"          # hermetic (CI)
uv run python scripts/size_ratchet.py --base HEAD
```

GPU commands run on marzuki-neon (see AGENTS.md).

## Rules

Branch + PR to main (protected: no direct pushes), CI green, adversarial review by a
different agent than the submitter, peer merges `--squash`. See AGENTS.md.
