# S5-2 RUN 2 — CARD: the 1.1–1.4 GB prefill excess, the C2 fix, and why C2 stays quarantined

**Bottom line:** the prefill HBM excess the S5-1 run quarantined is real and is sourced to the
**2→1 allocation-cadence change** in `KQuantFileGemm::eval_gpu` (the C2 change removed the
`set_data` reallocation). The fix — replace the `zeros()` fill with an uninitialized
`malloc_async` scratch **and keep** the `set_data` reallocation — **releases the prefill
excess (gate PASSES: 9.63 GB vs 11.84 GB)** but **crashes decode** with the S5-1
`Event::is_signaled` assertion. The `zeros()` fill is load-bearing for decode, not dead.
**C2 stays QUARANTINED.**

## RCA (the source of the excess)

The `RocmAllocator` **always recycles, never `hipFree`** (exact-size freelist,
min_utilization 1.0); the only HBM release (`hipFree` of cached buffers) is `malloc_async`'s
reclaim, which fires **only when `active + cache + need > 0.85 × total`** (14.53 GB cap on
15.93 GB). The C2 change's 2→1 cadence change (dropping the `set_data` reallocation) raised
the **active-buffer envelope** (9.62 → 11.84 GB) and shrank the cache (4.32 → 2.69 GB); with
too little cached memory to free, the reclaim can't pull `active+cache` back under the cap →
the prefill OOM. The "the freed `zeros` buffer was held, not released" hypothesis is refuted
(free=recycle caches it); the excess is in the **active trajectory crossing the cap**.

## the fix (diff, `kquant_file_gemm.hip:1173-1185`)

```diff
   if (!f32_target) {
     tmpf32 = std::make_unique<array>(
-        mlx::core::zeros(Shape{...}, float32, s));
+        rocm::malloc_async((int64_t)M * NBS * 256 * sizeof(float), encoder),
+        Shape{...}, float32);
+    // S5-2: uninitialized scratch (no zeros() fill dispatch = keeps C2's -404);
+    // the set_data reallocation below is KEPT (2-alloc cadence = OOM-safety per the RCA).
     encoder.add_temporary(*tmpf32);
     f32_target = tmpf32.get();
   }
   out.set_data(rocm::malloc_async(out.nbytes(), encoder));
   if (f32_target != &out) {
     f32_target->set_data(rocm::malloc_async(f32_target->nbytes(), encoder));   // KEPT
   }
```

Allocation-only (no semantics change): the scratch is uninitialized but the kept `set_data`
reallocation discards it (recycled to the cache) and writes the kept buffer fully — the C2
safety argument. Expected: −1 dispatch/gemm (the −404) + the restored 2-alloc cadence.

## build ledger (venv-s52c2)

- command: `pip wheel . --no-deps --no-cache-dir` (the first `pip wheel .` hit pip's wheel
  **cache** — lib md5 came back `e377b26f` = the C1 lib; redone with `--no-cache-dir` +
  `pip cache purge` + `touch` of the edited TU)
- `kquant_file_gemm.o` recompiled at 08:44 (confirmed); lib +160 B vs C1 (176,706,336 vs 176,706,176)
- wheel md5 `4ae7886b0266b6c175f5d7cfeaf9283d`, **lib md5 `2c93341186eb24091c7e179d6996f382`**
- base libs unchanged: control `cdb74a5794…`, C1 `e377b26f…`; import OK

## OOM gate line (prefill census, `MLX_WORKER_DEBUG`)

**GATE PASSES.** Fixed-wheel prefill: **rc=0, active 9.63 GB (≈ C1-only 9.62 GB / control
10.4 GB class, NOT the 11.84 GB quarantined class), cache 4.9 GB, peak 14.49 GB** — reclaim
fires (over oscillates ±0) but frees cached buffers and **completes**. The 1.1–1.4 GB excess
is **released**. Quarantined wheel (same census): rc=134, active 11.84 GB, OOM.

## the decode crash (why the fix is rejected)

| arm | prefill (maxtok=1) | M64 | M2 | M128 |
|---|---|---|---|---|
| **fixed** (`venv-s52c2`) | **rc=0** | **rc=134** assert | **rc=139** assert | **rc=139** assert |
| C1-only (`venv-s51b1`) | rc=0 | rc=0 | rc=0 | rc=0 |
| control (`venv-ar`) | rc=0 | rc=0 | rc=0 | rc=0 |

`venv-s51b1` (C1-only: `zeros`+`set_data`, old allocator) differs from `venv-s52c2` (my fix:
`malloc` scratch+`set_data`, same allocator) by **exactly** the `:1175` change. C1-only
survives all AR legs; my fix crashes all of them (`event.hip:418 Event::is_signaled
assertion`). **The `zeros()` fill is load-bearing for decode stability, not a dead temp.**

## 6-leg ladder (escape-hatch tps data, stable wheels; all md5s match standing)

| depth | C1-only tps | control tps | md5 first100 | standing |
|---|---|---|---|---|
| M64 | 0.3660 | 0.3675 | `387b252b` | ✓ |
| M2 | 0.6079 | 0.6044 | `ca803647` | ✓ |
| M128 | 0.2445 / 0.1558 | 0.2338 / 0.1486 | `23558e8f` | ✓ |

(model sha `3b8d3c60…`, prompt sha `be02c4f1…`, 4096 tok; samval: the 2 GiB GPU-hold
registered 255 MiB — a sampler-timing artifact, the hold completes ~4–5 s after import, after
the sample window; the HBM sampler is validated by the real prefill census 9.6–15.9 GB.)

## VERDICT

**C2 stays QUARANTINED.** The allocation-only fix releases the prefill HBM excess (gate
PASSES) but is a decode regression (`Event::is_signaled` assertion, maxtok≥2) that the C1-only
baseline does not hit. The `zeros()` fill cannot be removed (the whole point of C2's −404)
without breaking decode. The prefill excess was correctly sourced (RCA); the fix cannot remove
the fill allocation-safely.

## run-3 next-lane

**mmq M=1 depth-growth** (the next lane; the C2 lane is closed — the fill is load-bearing).

## box hygiene

Clean after all sessions: kfd empty, VRAM 57 MiB, no lingering `c7run.py` processes.
