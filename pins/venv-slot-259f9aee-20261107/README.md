# Record tree pin - wheel 259f9aee (venv-slot), 2026-11-07

The source tree that produces the 92.23 record wheel (venv-slot, libmlx md5 259f9aeefe3ec02f8993fe562f5ee62b) exists ONLY as a box-local worktree on neon (worktree 'vendor-merge' under the box home; path scrubbed per the standing persona rule). HEAD 34bbe89826237e21955583d40997fbb801b00cd1 ("Phase B2: GdnFusedStepSlot op", parent a7ef8c1 = B2 base merged-lane tree, the JOB 2 wheel af341e99 build state) is absent from the fleet mirror and from the GH fork. The record wheel was built from this HEAD plus the uncommitted tracked delta below (int2 bindings); nothing else was modified. This pin closes the t2348 finding that the "head tree = the source that produced the record build" claim was unsupported for the slot component (the in-tree record-a0057c22 pin covers the earlier 69.43 wheel, not venv-slot).

- head-34bbe89.tar.gz: git archive of HEAD (clean base tree, 2046 files)
- tracked-delta.patch: git diff HEAD (1 file, +38: python/src/fast.cpp — the int2 bindings `lm_head_int2` + `lm_head_int2_wide`, uncommitted at snapshot time)
- untracked.tar.gz: 0 untracked source files (the only untracked files are two evidence baks, qmv_wide_mr3_kernel.hip.vr1bak/.vr2bak, excluded per the pin spec)
- HEAD-sha.txt: the base commit sha

Reconstruction: extract the head tar, apply the patch, extract the untracked tar over it (no-op). Verified 2026-11-07: `git apply` clean on the extracted tree; post-patch fast.cpp carries 7 `lm_head_int2` refs; the slot kernel source is in-tree (python/src/fast.cpp dispatch, mlx/fast.h + mlx/fast_primitives.h declarations, mlx/backend/rocm/gdn_fused_step_tape.hip kernel, plus the tracked lm_head_int2.hip / lm_head_int2w.hip / gdn_fused_step*.hip kernels). Snapshot taken 2026-11-07 (2026-11-06 21:17 UTC) by jimmy, read-only on the box.

Identity block (record 92.23 t/s @1024 auto, stamp commit 0cb21ec):

- venv-slot libmlx md5 259f9aeefe3ec02f8993fe562f5ee62b (built from this tree, installed in the box-local venv 'venv-vr3'; wheel not in repo)
- bench/mtp-bench6-stepdec-slot.py md5 b00dec6b083428516b804b26f345d3de (the record producer, pinned in-tree, tree copy = as-run)
- sidecar model-mtp.safetensors md5 02a54359cda8b247f76b277dd9bc4d4c (box-local, not tracked)
- mlx_lm qwen3_5.py md5 ff7c8d0e47a606e55a7eb1266ba7f4b1 (box-local, not tracked)
- record stream a55aac5c21d2acf636d01c0f88b89856 (ALLTOKS md5; TOKS_S 92.85/92.71/92.36 median 92.71 re-measure 2026-11-06 20:34-20:43Z, DPM auto, TPK 2.080, D1 355/69.3% D2 198/55.8% REFWDS=0)
