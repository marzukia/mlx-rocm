# S5-2 RUN 2 — STEP 3: the re-validation (the fixed wheel vs control, OOM gate + ladder)

**Wheels (md5 provenance):**

| wheel | venv | lib md5 | role |
|---|---|---|---|
| **fixed** (C1 + C2-fix) | `venv-s52c2` | `2c93341186eb24091c7e179d6996f382` | the fix under test |
| **control** (C1+C2−C3) | `venv-ar` | `cdb74a5794fed2bdfec5103e085969c6` | stable standing reference |
| **quarantined** (C1+C2) | `venv-s51` | `cf6bc664a9e6cc91d40dc3dd1fff1da8` | the S5-1 prefill-OOM wheel |
| **C1-only** (C2 reverted) | `venv-s51b1` | `e377b26f9740b3d368581c472f050858` | disambiguation baseline |

model sha `3b8d3c602f63414794d5f43ceb22bbb61eb586af2cddf07cbf10656860f5b80a`, prompt sha
`be02c4f1a46bef08bc94d59024f9337c2406ad2dee848f807f8318886424347d` (bytes 28574, tokens 4096).

**Method:** ONE detached flock-held GPU session (`setsid` + `flock`), samval-first, `$V/bin/python`
loads the arm's wheel (c7run.py's hardcoded `VENV` only supplies the `mlx_lm` models code,
identical everywhere). Per-leg `rc` + `Event::is_signaled` assertion check + `MD5` + tps +
phase-aligned prefill HBM census (`MLX_WORKER_DEBUG` on the prefill legs). Box verified clean
(kfd empty, VRAM 57 MiB) between sessions.

## samval

The 2 GiB GPU-hold check registered only **255 MiB** (the hold's `mx.ones` allocation completes
~4–5 s after import, after the 1.2 s sample window) — a sampler-**timing** artifact, not a
sampler defect. The HBM sampler the census reads (`mem_info_vram_used`) is validated by the real
prefill census: the `MLX_WORKER_DEBUG` `RECLAIM` lines report `active`/`cache`/`peak`/`hbm_total`
in the 9.6–15.9 GB range for the loaded 27 B model, i.e. the sampler tracks the GPU allocation
correctly. (Note: run-1's samval sampled `/proc/meminfo` for a GPU hold = the wrong signal; this
run samples the actual GPU VRAM the census uses.)

## THE OOM REPRO LEG — the prefill gate (GATE: PASSES)

Phase-aligned prefill HBM census (`MLX_WORKER_DEBUG`, the 4096-tok prompt, `maxtok=1`):

| arm | rc | prefill | active | cache | peak | reclaim `over` | outcome |
|---|---|---|---|---|---|---|---|
| **fixed** (`venv-s52c2`) | **0** | 196.3 s | **9.63–9.66 GB** | 4.88–4.91 GB | 14.49 GB | oscillates ±0 (reclaim fires, survives) | **completes** |
| quarantined (`venv-s51`) — step-1 probe | 134 | OOM | 11.84 GB | 2.69 GB | 14.53 GB | >0 (reclaim fires) | OOM crash |
| C1-only (`venv-s51b1`) — step-1 probe | 0 | ~201 s | 9.62 GB | 4.32 GB | — | <0 (no reclaim) | survives |
| control (`venv-ar`) — run-1 + FCM legs | 0 | ~201 s | ~10.4 GB class | — | — | <0 | survives |

**The excess is RELEASED:** the fixed wheel's prefill active envelope is **9.63 GB ≈ the C1-only
9.62 GB / control 10.4 GB class, NOT the 11.84 GB quarantined class.** Restoring the 2-alloc
cadence (the kept `set_data` reallocation) pulled the active envelope back down (11.84 → 9.63 GB)
and left the cache at 4.9 GB (vs the quarantined 2.69 GB), so when `active+cache` briefly
crosses the 85 % reclaim cap (14.53 GB) the reclaim frees cached buffers and the prefill
**completes (rc=0)** instead of OOMing. The 1.1–1.4 GB prefill excess the RCA named is released.
**Gate PASSES.**

## THE DECODE CRASH (the fix is a regression)

The fixed wheel survives **prefill** (maxtok=1, rc=0) but **crashes on decode (maxtok≥2)** with
the exact S5-1 `Event::is_signaled()` assertion (`event.hip:418`):

| arm | M64 | M2 | M128 |
|---|---|---|---|
| **fixed** (`venv-s52c2`) | **rc=134** (assert) | **rc=139** (assert) | **rc=139** (assert) |
| C1-only (`venv-s51b1`) | rc=0 | rc=0 | rc=0 |
| control (`venv-ar`) | rc=0 | rc=0 | rc=0 |

**Disambiguation (the A/B that decides it):** `venv-s51b1` (C1-only: `zeros()` + `set_data`, old
allocator) differs from `venv-s52c2` (my fix: `malloc_async` scratch + `set_data`, **same
allocator**, **same `set_data` reallocation**) by *exactly* the `:1175` `zeros()`→`malloc_async`
change. C1-only **survives all AR legs** (rc=0, md5s below); my fix **crashes all AR legs**
(`Event::is_signaled` assertion). Box was clean (57 MiB) between the runs; the stable control
runs fine on the same box. **Conclusion: the `zeros()` fill is load-bearing for decode
stability — it is not a dead temp. Replacing it with an uninitialized `malloc_async` scratch
(even with the `set_data` reallocation kept) causes a decode-phase `Event::is_signaled`
assertion.** The allocation-only fix that the RCA/step-2 premised on (2-alloc cadence =
OOM-safe) is refuted for decode.

## 6-leg ladder — tps + md5 (escape-hatch tps data, stable wheels)

C2 is quarantined, so the tps data comes from the stable wheels (C1-only vs control). All md5s
match the standing md5s (64:`387b252b`, 2:`ca803647`, 128:`23558e8f`) — md5 invariance holds
on every stable leg.

| depth | C1-only (`venv-s51b1`) tps | control (`venv-ar`) tps | md5 (first100) | standing |
|---|---|---|---|---|
| M64 | 0.3660 | 0.3675 | `387b252b` | ✓ `387b252b` |
| M2 | 0.6079 | 0.6044 | `ca803647` | ✓ `ca803647` |
| M128 | 0.2445 (all) / 0.1558 (warmup-excl) | 0.2338 (all) / 0.1486 (warmup-excl) | `23558e8f` | ✓ `23558e8f` |

## VERDICT

**C2 stays QUARANTINED.** The allocation-only fix (uninitialized `malloc_async` scratch, keeping
the `set_data` reallocation) **releases the prefill HBM excess** (9.63 GB vs the 11.84 GB
quarantined class — the prefill OOM gate **PASSES**) **but causes a decode-phase
`Event::is_signaled` assertion** (the S5-1 OOM signature) that the C1-only baseline does not
hit. The `zeros()` fill in the `KQuantFileGemm::eval_gpu` scratch is load-bearing for decode
stability, not dead; the "2-alloc cadence = OOM-safe" premise holds for prefill only. The fix
is rejected as a decode regression. (The prefill excess is real and was correctly sourced in the
RCA; the fix simply cannot remove the fill without breaking decode.)

## run-3 next-lane

**mmq M=1 depth-growth** (the next lane, since C2 is quarantined). The C2 lane is closed: the
fill is load-bearing for decode, and removing it (the whole point of C2's −404) is not
allocation-safe.

## box hygiene

Clean after all sessions: kfd empty, VRAM 57 MiB, no lingering `c7run.py` processes.
