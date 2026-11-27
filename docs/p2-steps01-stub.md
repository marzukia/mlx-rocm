# P2 pre-run record — steps 0+1 (locate + flag verification)

- **Vehicle (step 0):** the 11-26 path `/data/gguf/Qwen3-27B-q2k-gguf-26082001-5186620b.gguf`
  is gone (WAVE-17 finding). The c7flipab/t3592 run dir points at the salvage:
  `/home/andryo/models/Qwen3.8-27B-Q2_K-plain.gguf`, sha256
  `3b8d3c602f63414794d5f43ceb22bbb61eb586af2cddf07cbf10656860f5b80a` (== the C7 card's
  `3b8d3c60...`), 8,985,410,528 B (~8.99 GB; card class 8.97 GB). Verified on the box.
- **Flag verification (step 1):** build tree `/home/andryo/build-p2seq` @ `b9cb137` (==
  spec) has `MLX_ROCM_RELEASE_SHADOW` (allocator.cpp:1419/1426) + the flush paths.
  The spec-named venv's lib (`/home/andryo/venv-hbmfix` `libmlx.so` md5 `070122d8...`,
  dist-info `dev20261116+dcd1c69`, placed 11-27 18:00:38Z) has the env name count **0**
  (stale 11-16 build). The spec's wheel name `dev20261127+b9cb137` is installed in
  `/home/andryo/venv-ar` (lib md5 `cdb74a57...` == the 09:47Z b9cb137 build, env name
  count **1**); `c7run.py`/`run1.sh` use venv-ar. Resolution: both arms run the harness's
  flag-bearing lib `cdb74a57` (stock = env unset per the allocator's "default off = stock
  behavior" comment); no build, no lib swap.

Full card: `docs/p2-release-shadow-card.md` (per-arm table + gate verdicts + VERDICT).
