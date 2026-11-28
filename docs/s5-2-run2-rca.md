# S5-2 RUN 2 — STEP 1: the C2 RCA (the 1.1–1.4 GB prefill HBM excess)

**Scope:** the S5-2 ruling item (2). The C2 change (uninitialized f32 `malloc_async`
scratch replacing the dead `zeros()` temp in `KQuantFileGemm::eval_gpu`, including the
removed `set_data` reallocation) OOM-crashed at prefill (99.6–99.8% HBM, +1.1–1.4 GB over
control at the same phase; 3 failure modes across retries). C1 is inactive at prefill M=256,
so the excess is 100% C2. This doc source-names the exact divergence, names the mechanism, and states the fix. No invented mechanism.

## Box-tree verification (ledger, before the read)

- Tree `/home/andryo/build-p2seq` @ `b9cb137` — verified C1 APPLIED, C2 REVERTED (the S5-1
  end state).
  - C1 present: `mmq/mmq.hip:318` (`int64_t real_total = -1`), `:321-322` (pad-block zero
    in-kernel), `qmm.hip:102-103` (the pad `hipMemsetAsync` removed).
  - C2 reverted: `kquant_file_gemm.hip:1175` = `mlx::core::zeros(Shape{...})` (the fill
    dispatch), `:1181` = `f32_target->set_data(rocm::malloc_async(f32_target->nbytes(), encoder))`
    (the reallocation) — both present = the old path.
- venv lib md5s (ledger, pre + post, byte-identical): `venv-s51` `cf6bc664…` (C1+C2, quarantined),
  `venv-s51b1` `e377b26f…` (C1-only = C2 reverted = OLD kqfg), `venv-ar` `cdb74a5794…` (control).
  Model sha `3b8d3c60…` matches. Box free pre-dispatch (kfd empty, VRAM 59.9 MB).

## Source read — the C2 path end to end

`KQuantFileGemm::eval_gpu` (`kquant_file_gemm.hip:1169-1187`). Non-f32-output path
(`f32_target == nullptr`):

```
1172  std::unique_ptr<array> tmpf32;
1173  if (!f32_target) {
1174    tmpf32 = std::make_unique<array>(
1175        mlx::core::zeros(Shape{M, NBS*256}, float32, s));   // OLD: alloc buffer A + FILL DISPATCH
1176    encoder.add_temporary(*tmpf32);
1177    f32_target = tmpf32.get();
1179  out.set_data(rocm::malloc_async(out.nbytes(), encoder));   // alloc buffer B (the out)
1180  if (f32_target != &out) {
1181    f32_target->set_data(rocm::malloc_async(f32_target->nbytes(), encoder));  // OLD: alloc B2, FREES A
1182  }
```

C2 removes the fill dispatch at `:1175` (replaced by an uninitialized `malloc_async` scratch) and
removes the reallocation at `:1181` (the scratch IS the buffer). Net per gemm call: **2 device
allocations in the OLD path (A then B2) vs 1 in the NEW path (C)**; both paths end with exactly
one live f32 temp + the out buffer.

## The `array`/`Data` deleter is IDENTICAL in both paths — this is NOT the divergence

- `array(allocator::Buffer, Shape, Dtype, Deleter = allocator::free)` (the C2 buffer constructor)
  and `array::set_data(allocator::Buffer, Deleter = allocator::free)` (the C2 reallocation)
  BOTH default the `Data.d` to `allocator::free` (`mlx/array.h:450`, the buffer-ctor default).
  `Data::~Data` calls `d(buffer)` (`mlx/array.h:253`). So the f32 scratch's deleter is
  `allocator::free` in the OLD path (A, via `zeros`) and in the NEW path (C, via `malloc_async`).
  `zeros()` and `malloc` route to the same pool (`RocmAllocator::malloc` = `malloc_async(size,-1,nullptr)`).
  **There is no deleter/temporary bookkeeping line where the NEW path retains what the OLD path
  drops.** The S5-1 partial-diagnosis hypothesis ("the new path's array ownership/lifetime keeps something resident the old path dropped") does not hold under this allocator (see next section).

## `free()` ALWAYS recycles — it NEVER hipFree. This refutes the "set_data freed the buffer" premise

`RocmAllocator::free` (`allocator.cpp:825-883`), the only free path, ends:

```
// Always recycle; never hipFree on the free() path.
// Bound HBM only from malloc_async (memory_limit_ pressure + max_pool_size_ trim there)...
w1probe::note_recycle(buf->size, buf->data);
buffer_cache_.recycle_to_cache(buf);
```

The `BufferCache` is an **exact-size freelist (min_utilization 1.0)** (`allocator.cpp:338-347`): a
freed slab stays **mapped in the freelist (HBM held)** until `malloc_async` reclaims it. The only
place cached HBM is actually released (`buffer_cache_.release_cached_buffers` → hipFree) is in
`malloc_async`'s reclaim (`allocator.cpp:629-667`), and it fires **only when
`active + cache + need > 0.85 * total_memory`** (the 14.53 GB cap on this 16 GB card; the
`hbm_cap` at `:633`).

Consequence: in the OLD path, the `:1181` `set_data` reallocation does **not** release buffer A's
HBM — it **recycles** A into the freelist (HBM held). The OLD path therefore holds an **extra
cached size-S slab (A)** that the NEW path never has. A naive read of the array/deleter path
predicts **OLD ≥ NEW** resident HBM — the **opposite** of the observed C2 excess. So the excess is NOT a cached-slab/deleter-residency effect.

## The probe (cheap, prefill-only, no full AR) — the decisive disambiguation

One flock-held detached session. A/B of **C2 alone at prefill** (`c7run.py --maxtok 1`, the 4096-tok
prompt = the prefill OOM condition; C1 is inactive at prefill M=256 so `venv-s51b1` (C2=OLD) vs
`venv-s51` (C2=NEW, quarantined) isolates C2). `MLX_WORKER_DEBUG=1` → the `RECLAIM` line prints
`active`/`cache`/`peak`/`over`; per-phase `rocm-smi` HBM. samval-first PASS (59.9 MB → 2.4177 GB ×3,
identical to S5-2 run-1). Provenance: lib md5s pre/post (`e377b26f…` / `cf6bc664…`), model sha
`3b8d3c60…`.

| | OLD (venv-s51b1, zeros+set_data) | NEW (venv-s51, malloc_async scratch) |
|---|---|---|
| rc | **0 — survives** (prefill 201.6 s) | **134 — OOM** (prefill ~150 s) |
| crash | — | `event.hip:418: Event::is_signaled() Assertion value()==1 failed` (= S5-1 failure mode #3, reproduced verbatim) |
| active | **9.62 GB** | **11.84 GB** (+2.2 GB) |
| cache | 4.32 GB | 2.69 GB (−1.6 GB) |
| `over` (reclaim) | **< 0 → no reclaim → survives** | **> 0 → reclaim fires → OOM** |
| peak (internal) | 10.31 GB | 13.51 GB |

**Reading.** The C2 excess is in the **ACTIVE-buffer trajectory crossing the 85 % reclaim cap
(14.53 GB)** — NOT a cached-slab residency. The NEW path's active envelope (11.84 GB) is +2.2 GB
higher and its cache (2.69 GB) is lower; together they hit the cap, fire reclaim, and OOM. The
OLD path's lower active envelope (9.62 GB) + higher cache (4.32 GB) = 13.94 GB stays **under** the
cap, so no HBM is ever exhausted and it survives. The OLD path caches **2 slabs per size class**
(A from `zeros` + B2 from `set_data`) → 4.32 GB cache; the NEW path caches **1** (C) → 2.69 GB
cache — consistent with the probe's cache split.

## Named divergence (file:line) — the mechanism

The excess is the **allocation-cadence change: 2 device allocations per gemm call (OLD) → 1 (NEW)**,
introduced by removing both `kquant_file_gemm.hip:1175` (the `zeros` alloc + fill) and
`:1181` (the `set_data` reallocation). Under this allocator's free=recycle + 85 % reclaim-cap
semantics, that cadence change (a) collapses the cache (4.32 → 2.69 GB) and (b) raises the
peak-active envelope (9.62 → 11.84 GB) enough to cross the 14.53 GB cap and OOM. **It is not a
single retained-buffer line** — there is no buffer the NEW path holds that the OLD path releases
(the deleter is identical and `free()` never hipFree's). The `:1181` reallocation is what
restores the 2-alloc cadence (cache headroom + lower active envelope); the `:1175` fill dispatch
is what the −404 removes.

## The fix (STEP 2) — keep the −404, restore the cadence

Minimal, allocation-only (not a semantics change): **replace the `:1175` `zeros()` with the
uninitialized `malloc_async` scratch (removes the fill dispatch = keeps C2's −404), but KEEP the
`:1181` `set_data(malloc_async(...))` reallocation.** Net: no fill dispatch (−404), 2 device
allocations per call (restores the OLD HBM profile: ~9.6 GB active + ~4.3 GB cache, under the cap).
This is an allocation change, so per the task C2 is expected to be **RELEASED** (the load-bearing
escape hatch is not triggered — the scratch is not semantically load-bearing; only the cadence is).
STEP 3's OOM-repro leg validates the fixed wheel directly vs the control.
