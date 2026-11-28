# S5-2 RUN 2 — STEP 2: the C2 fix build (venv-s52c2)

**Box tree:** `/home/andryo/build-p2seq` @ `b9cb137` + leg-4 + C1 applied (unchanged, the standing
S5 state) + the **C2 fix** (this step). Box tree state ledger: C1 present (`mmq.hip` `real_total`
count=2), the fix scratch present (`rocm::malloc_async((int64_t)M * NBS * 256 * sizeof(float), encoder)`
count=1), the `set_data` reallocation KEPT (count=1).

## The fix (verbatim, `kquant_file_gemm.hip:1173-1185`)

The `:1175` `zeros()` (fill dispatch + first allocation) is replaced by the **uninitialized
`malloc_async` scratch** (removes the fill dispatch = keeps C2's −404), and the `:1181`
`set_data(malloc_async(...))` reallocation is **KEPT** (restores the 2-allocation cadence = the
OOM-safety the RCA named). The full C2 change (which also removed the `:1181` reallocation and
caused the OOM) is NOT applied.

```
  if (!f32_target) {
    tmpf32 = std::make_unique<array>(
        rocm::malloc_async((int64_t)M * NBS * 256 * sizeof(float), encoder),
        Shape{(int)x.shape()[0], (int)((int64_t)NBS * 256)}, float32);
    // S5-2: uninitialized scratch (no zeros() fill dispatch = keeps C2's -404);
    // the set_data reallocation below is KEPT (2-alloc cadence = OOM-safety per the RCA).
    encoder.add_temporary(*tmpf32);
    f32_target = tmpf32.get();
  }
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  if (f32_target != &out) {
    f32_target->set_data(rocm::malloc_async(f32_target->nbytes(), encoder));   // KEPT
  }
```

Effect vs the control: −1 dispatch per gemm call (the removed fill) = the C2 −404; the 2-alloc
cadence (scratch + `set_data` reallocation) is preserved → the OLD HBM profile (cache headroom,
lower active envelope) → no prefill OOM. This is an **allocation-only change** (no semantics
change): the scratch is uninitialized but the `set_data` reallocation discards it (recycled to the
cache) and the kept buffer is written fully by the kqfg kernels (the S5-1 spec's C2 safety argument, unchanged).

## Build ledger (venv-s52c2)

| item | value |
|---|---|
| build command | `pip wheel . --no-deps --no-cache-dir` (the first `pip wheel .` hit pip's wheel **cache** and did NOT recompile — lib md5 came back `e377b26f` = the C1 lib; redone with `--no-cache-dir` + `pip cache purge` + a `touch` of the edited TU) |
| `kquant_file_gemm.o` recompile | **confirmed** — object timestamp `08:44` (during the build2 window `08:44:35`–`08:45:09`) |
| wheel | `mlx-0.32.3.dev20261128+b9cb137-cp314-cp314-linux_x86_64.whl`, md5 `4ae7886b0266b6c175f5d7cfeaf9283d` |
| **new lib `libmlx.so` md5** | **`2c93341186eb24091c7e179d6996f382`** (distinct from C1 `e377b26f…` and control `cdb74a5794…`) |
| new lib size | 176,706,336 B (C1 = 176,706,176 B; +160 B, consistent with the `zeros`→`malloc_async` call swap) |
| dist-info | `mlx-0.32.3.dev20261128+b9cb137.dist-info` |
| base libs (unchanged, verified pre/post) | control `cdb74a5794…`, C1 `e377b26f…` |
| venv base | cloned from `venv-s51b1` (C1-only), C1 `libmlx.so` removed before `--force-reinstall --upgrade` install of the new wheel |
| import | `IMPORT-OK 0.32.3.dev20261128+b9cb137` (CPU, no GPU) |

The M2 smoke (the `FTM2` ladder leg, `maxtok=2` AR at the 4096-tok prefill) is run first in the
STEP 3 session (it both verifies the lib end-to-end and is the M2 leg); the import check above
confirms the lib loads.
