# P2 box verification card — MLX_ROCM_RELEASE_SHADOW arm vs stock control (27B Q2_K vehicle)

Session: 2026-11-27 ~22:21–22:49Z, GPU box dGPU (gfx1201 RX 9070 XT, 16GB / 17.096GB HBM total), one lock holder (`/home/guest/gpu.lock` flock) for the whole session. WAVE-17 3/3 terminal released the hold; this is the lane's pre-registered box verification. No-build session.

## Step 0 — vehicle located (the salvage)

- The 11-26 path `/data/gguf/Qwen3-27B-q2k-gguf-26082001-5186620b.gguf` is GONE (as the WAVE-17 strategy card found). `find /data /mnt /home/guest -maxdepth 4 -name "*Qwen3-27B*q2k*"` → nothing; the c7flipab (t3592) run dir points at the salvage: **`/home/guest/models/Qwen3.8-27B-Q2_K-plain.gguf`**.
- Identity verified: sha256 `3b8d3c602f63414794d5f43ceb22bbb61eb586af2cddf07cbf10656860f5b80a` (the C7 card's `3b8d3c60...`, == the lever-card record), 8,985,410,528 B (~8.99 GB; card class 8.97 GB). Both run logs re-verified the sha at run time. c7run.py's `GGUF` constant points at exactly this file.

## Step 1 — flag verification (CPU-only) — the venv provenance finding

- Build tree `/home/guest/build-p2seq` @ `b9cb137` (== spec): `MLX_ROCM_RELEASE_SHADOW` present (`mlx/backend/rocm/allocator.cpp:1419,1426`, the `g_release_shadow` getenv gate) + `flush_host_shadow` / `flush_all_dirty_shadows` present.
- **The spec-named venv's lib does NOT carry the flag.** `/home/guest/venv-hbmfix` `libmlx.so` md5 `070122d8b617093da104a1b5626cadbd` (dist-info `mlx-0.32.3.dev20261116+dcd1c69`, file placed 2026-11-27 18:00:38Z — after the C7 session): `strings`/`grep -a` count of `MLX_ROCM_RELEASE_SHADOW` = **0** (71 `MLX_ROCM_*` envs listed, none release-shadow; the `flush_host_shadow` symbols are present — the shadow staging machinery is in, the release gate is not).
- **The spec's wheel name (`mlx 0.32.3.dev20261127+b9cb137`) is installed in `/home/guest/venv-ar`** (dist-info `dev20261127+b9cb137`, placed 09:55Z, lib md5 `cdb74a5794fed2bdfec5103e085969c6` == the 09:47Z build in the b9cb137 tree): `strings` count = **1**, 6 flush symbols. `c7run.py` hard-inserts `VENV=/home/guest/venv-ar/...` and `run1.sh` invokes `venv-ar/bin/python` — i.e. the t3592/C7 harness runs the **flag-bearing** lib. The spec's md5 citation (`070122d8`) is the stale 11-16 venv-hbmfix build, not the b9cb137 wheel.
- Resolution (no-build session): both arms ran the harness's lib `cdb74a57` (the C7/t3592 lib, md5 re-verified at session start); stock arm = `MLX_ROCM_RELEASE_SHADOW` unset (`allocator.cpp`: "default off = stock behavior"); shadow arm = `=1`. No build, no save/swap/restore (box facts).

## Method

- Vehicle: `c7run.py` (the t3592 box harness) `--maxtok 64` (steady-state class; M=1 greedy, CH=256 prefill, warmup-100 exclusion, same model glue / kqfg / MTP-off-by-design), prompt = the d500x adversarial one-line (28574 B / 4096 tok, sha256 `be02c4f1...` both arms).
- (a) `rocm-smi --showmeminfo vram --json` (card0 "VRAM Total Used Memory (B)") at 3 s cadence. **Fixed sampler** (the 11-27 `grep "GPU[0] .* Used" | grep -oE "[0-9]+" | head -1` returned the device index — smi.log was all zeros, 0/6000). **Sampler validated on a known-nonzero baseline before the runs (WAVE-17 process fix): a 2 GiB `mx.ones` hold read 2,417,733,632 B on 3/3 samples vs 59,916,288 B baseline — SAMPLER-VAL-PASS.**
- (b) Host-side at 3 s: process `VmRSS`, O_TMPFILE(`memfd`) map count from `/proc/<pid>/maps`, system `Shmem` from `/proc/meminfo` (the 11-25 retained-class method). (c) `SharedSignalPool` exit warning grepped from the run logs.
- Inter-arm hygiene per box facts: `fuser /dev/kfd` + VRAM before the shadow arm — clean (59.9 MB baseline). One run-infrastructure note: the first shadow-arm launch died in 3 s on a session-script `env` options-ordering bug (rc=127, no GPU time consumed); re-run 22:43:17Z is the record. Post-state clean: kfd free, VRAM 59.9 MB, lock released.

## Per-arm results

| metric (3 s sampling; GB = 1e9 B, GiB = 2^30) | P2STOCK (env unset, rc=0) | P2SHADOW (env=1, rc=139) |
|---|---|---|
| HBM peak | **15.98 GB** (14.88 GiB) | **16.62 GB** (15.48 GiB) |
| HBM steady (late decode) | 15.96 GB (climbed 15.15→15.98 prefill-end→decode-end) | 16.40–16.53 GB (flat) |
| HBM delta vs stock (phase-aligned: prefill-end / decode+30s / decode-end) | — | **+1.46 / +0.87 / +0.44 GB** (real, not a phase artifact; unexplained — flagged) |
| host vRSS steady | 9.67 GB (max 10.39) | **1.21 GB** (max 8.94 = load phase, pre-release) |
| system Shmem steady | 10.12 GB (max 10.33; slow climb ~+0.8 MB/decode-step class) | **1.154 GB** (flat) |
| O_TMPFILE (memfd) maps | 0 for the whole run | 0 for the whole run (the map-count channel never fired; Shmem is the effective indicator) |
| tps class (64 tok) | 0.3678 t/s (174.0 s; ms/step p50 2706.8) | 0.6310 t/s (101.4 s; ms/step p50 1582.4) — **1.71× stock** (one run each; flagged for follow-up) |
| sampler-validation line | PASS (2 GiB hold → 2,417,733,632 B, 3/3 > baseline+1 GiB) | same validated sampler |
| leak-warning lines | `Warning: Resource leak detected by SharedSignalPool, 1384 Signals leaked.` (at exit) | **none recorded — the process segfaulted before the clean-exit diagnostic** |
| run integrity | rc=0, DONE-STATS complete; first20/text md5 `1a10eb9a...` (== C7 SC-C class) | decode + all metrics complete (MD5 first100 `387b252b...`, same first20 text); **segfault (rc=139, core dumped) in the post-decode artifact-save tail — after the MD5 line, before np.save/DONE-STATS**; new failure signature, flagged |
| model sha / prompt sha / mtp | `3b8d3c60...` / `be02c4f1...` / off | same |

## Gate verdicts (<14G ROUGH gate; the 14.53 GB HBM cap class = 85% of 17.096 GB)

- **P2STOCK: FAIL** — HBM peak 15.98 GB, steady 15.96 GB (1.0–2.0 GB over the gate; same class as the 11-26 vehicle record 15.84 GB, 1024 tok — the fixed sampler reproduces it).
- **P2SHADOW: FAIL** — HBM peak 16.62 GB, steady 16.41 GB (2.4–2.6 GB over the gate; also over the 14.53 GB cap class).

## Findings feeding the WAVE-17 host-side class

- The 64-step stock arm: steady vRSS 9.67 GB / Shmem 10.12 GB with a slow Shmem climb (~+0.8 MB/decode-step) — not the C1 14.2 GB spun state; the 1024-tok C1 class remains separate (not P2's question).
- The SharedSignalPool leak reproduces **identical at 1384 signals** for a 64-step run vs the C7 8-step SMOKE (1384) → the count is NOT step-proportional in this class; it is dominated by load/prefill (16 chunks), not decode steps.
- The shadow arm's segfault (rc=139, post-MD5/pre-np.save artifact tail) is a new signature for that class; post-crash box state was clean (no kfd residue, VRAM baseline).

## VERDICT

**FAIL (both arms vs the <14G HBM gate; the 27B Q2_K vehicle does NOT clear the gate under shadow mode).** MLX_ROCM_RELEASE_SHADOW=1 does not reduce HBM — it is a host-side lever: it frees the pinned host shadow at H2D completion, and that release is confirmed here (Shmem 10.12→1.15 GB, vRSS 9.67→1.21 GB, **−88%** steady; O_TMPFILE map channel read 0 in both arms, Shmem is the effective indicator). The HBM footprint (~15.9–16.6 GB = weights 8.99 GB + KV + kqfg HBM staging + allocator overhead) exceeds the 14 G gate in both arms, and the shadow arm actually read +1.4–1.5 GB HBM over the stock arm phase-aligned (unexplained; flagged for follow-up). Side findings: 1384 SharedSignalPool leak reproduced in the stock arm (== C7 SMOKE, load/prefill-dominated, not step-proportional); shadow decode 1.71× faster (0.631 vs 0.368 t/s, one run each); shadow arm segfaulted in the post-decode artifact-save tail (rc=139, new signature). The spec's installed-wheel citation (venv-hbmfix `070122d8`, 11-16 build) lacks the flag; the b9cb137 wheel (`dev20261127+b9cb137`, lib `cdb74a57`) with the flag lives in the harness venv (venv-ar) and is what both arms ran — flagged for the operator's venv SoT reconciliation.

## Artifacts (box)

`/home/guest/p2shadow/` (P2STOCK.gen / .json / .host.log, P2SHADOW.gen / .host.log, sampler-val.log, smi baseline/post/final jsons), `/home/guest/p2-session.sh`, `/home/guest/p2-shadow-rerun.sh`, session logs `p2shadow-session.log`, `p2shadow-rerun.log`. No build performed; no lib save/swap/restore; venv-hbmfix untouched.
