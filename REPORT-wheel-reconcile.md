# REPORT-wheel-reconcile — REFWD-WHEEL-RECONCILE (diff, unified wheel, R0 gate)

## 1. Setup

- Branch `worker/wheel-reconcile`, base `b4ba2bd`.
- Trees compared (on the guest box): the **pinned wheel tree** (the canonical
  build tree — pristine 75ca881 + the 39-file canonical graft, i.e. the source
  of the baseline-of-record wheel, section 3 of `REPORT-rebaseline.md`) vs the
  **refwd wheel tree** (the build tree that produced the
  `0.32.3.dev20261102+75ca881b` wheel of `REPORT-refwd-ab2.md`).
- The `worker/refwd-fold @03c8b0b` committed vend tree is the 03c8b0b lineage
  (graph-decode fork base); the refwd wheel's actual build tree on the guest
  box is the authoritative source for the graft (verified: the grafted region of
  the wheel tree is byte-identical to the region of the built wheel's source;
  the committed 03c8b0b tree carries the same single-toggle M=1 re-route, the
  wheel tree's K1 section is the one that shipped in the wheel).

## 2. Task 1 — per-file delta (refwd wheel tree vs pinned wheel tree)

Full-tree `diff -rq` (excluding build dirs): **17 files differ** (+ 2
pinned-only files, + 1 junk artifact on the pinned side).

### Functional on the default path (all env unset, bf16 model)

| file | delta | default-path effect (B0, rows OFF) |
|---|---|---|
| `quantized/qmm.hip` | **t1251 layer missing in refwd**: (a) `qmv_wide_mr2_kernel` (M=2..4, default-on at K%1024==0, N>=16384, K=4096/8192/12288); (b) `mr_ok` N-floor `N >= mr2_min_n` (default 16384, small-N per-row); (c) f32 x in `wide_ok` + f32 wide/MR dispatch branches + MR f32 staging guard; (d) `MLX_T1251_DBG` (opt-in); (e) rows dispatch (opt-in, OFF in B0); (f) **REFWD_M1 toggle (opt-in, OFF in B0 — the graft)**; (g) `#include "qmv_rows_msg_kernel.hip"` | M=1 (decode + refwd correction) QMMs: **identical** (both per-row wide). **M=4 verify QMMs differ**: N>=16384 — pinned takes mr2 (K=4096: lds 16 KB; K=8192/12288: the kernel's raison d'être) vs refwd takes MR (K=4096) / per-row (K=8192/12288, lds > 32 KB cap); N<16384 K=4096 — pinned per-row (N-floor) vs refwd MR. f32 x (scale-dtype promotion): pinned wide/MR; refwd falls out of the wide path. |
| `quantized/qmv_wide_mr_kernel.hip` | pinned-only `qmv_wide_mr2_kernel` template (141 lines) | not launched at M=1; at M=4 see above |
| `quantized/qmv_rows_msg_kernel.hip` (+ a `.pre-wsfix` junk copy) | pinned-only file | rows path, opt-in, OFF in B0 — no B0 effect |
| `scaled_dot_product_attention.hip` | pinned = m8 single runtime-causal-mask kernel + `MLX_ROCM_SDPA_S4VA` (opt-in, default-off in both); refwd = 2 constexpr-causal templates | same math; codegen-level difference only (secondary) |

### Inert on the default path (graph-decode machinery, gated on
`graph_decode_mode()` = `MLX_GRAPH_DECODE != 0` AND runtime arm, or on
decode-arena slots which are null when the arena is inactive)

`device.cpp` (+503 lines), `device.h` (+38), `eval.cpp` (+416, G1 pilot),
`indexing.hip` (+103, G1K3 slots), `allocator.cpp`/`allocator.h` (+131/+32,
decode value/zero slots, `decode_arena_reserve_wrappers`), `copy.hip` (+66,
`g1_slot_write_kernel`, decode zero-slot / off-slot), `copy/copy_general_dynamic.hip`
(+25, shape data-size guard + G1Y diagnostic), `flash_attention.hip` (+29,
decode kL H2D slot), `graph_decode_pilot_logic.h` (pinned-only file).

### No effect

- `CMakeLists.txt`: 1 line (a doc URL string).
- `README.md`: graft notes (40 lines).
- `gpu/fp16_math.cuh`: a scrub artifact on the pinned side
  (`__bfloarun-162float`), clean on the refwd side — cuda files are not
  compiled in the ROCm build.
- GDN fused/tape files, `fast.cpp`, `fast.h`, `fast_primitives.h`,
  `gated_delta_update.cpp`, python `src/fast.cpp`: **byte-identical** between
  the trees (absent from the diff) — the tape frame is the same object in
  both wheels.

### Candidate causes of the ~6% B0 delta (55.92/55.93 refwd vs 59.17/59.25 pinned)

1. **Primary — the t1251 verify-dispatch layer (mr2 + N-floor + f32 wide
   support) is present in the canonical graft but absent from the refwd wheel
   tree.** In B0 (rows OFF) the M=1 path is bit-identical, but the M=4 verify
   QMMs take different kernels: the in-tree lab notes put mr2 at 6-9% faster
   than full-staging MR at K=4096 N=248320 (2.59 vs 2.79 ms; per-row 4.00 ms),
   and verify is the dominant step (the rebaseline documents the entire
   wheel-vs-anchor delta sitting in verify). The candidate wheel's R0 verify
   (33.07/33.12 ms, this report) vs the refwd wheel's B0 verify (35.95 ms,
   `REPORT-refwd-ab2.md`) is a ~3 ms / ~8% verify gap in exactly the
   verify-QMM direction the missing layer predicts.
2. **Secondary — SDPA codegen** (m8 single runtime-mask kernel vs 2
   constexpr templates); same math, different instantiation.
3. **Ruled out**: rows path (off in both), GDN tape frame (byte-identical
   files; both wheels 128/128 n_ok, n_fb=0), graph-decode machinery (dormant
   in both), REFWD_M1 toggle (unset in B0), the cuda scrub artifact (not
   compiled).

## 3. Task 2 — candidate unified wheel (pinned tree + minimal REFWD_M1 graft)

**The graft** (single file, single section — `quantized/qmm.hip` K1 block):
the K1 gate `lx_env != nullptr` → `lx_engage = (lx_env != nullptr) ||
(refwd_m1 && M == 1)` with the `MLX_ROCM_REFWD_M1` static and the guarded
`lx_cols` read (32 cols when engaged via the toggle; `MLX_QMV_LX=16` still
selects 16). Committed as `patches/graft-refwd-m1-wheel.diff`. Default OFF —
unset, `lx_engage` reduces to the original gate, zero behavior change.

**Build** (guest box, scratch dir, canonical recipe
`CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
-DBLA_VENDOR=OpenBLAS -Drocwmma_DIR=/opt/rocm/lib/cmake/rocwmma"
pip wheel . --no-deps`):

| item | value |
|---|---|
| base tree | the canonical build tree (pristine 75ca881 + 39-file graft), copied wholesale (build dir excluded) |
| graft applied | `patches/graft-refwd-m1-wheel.diff` — `patch -p1` clean |
| tree manifest | 1910 files; md5 manifest vs the canonical tree: **exactly 1 file differs** (`qmm.hip`) |
| grafted region | K1 section, 45 lines from the `K1:` marker — **md5 `57fcb8f14ab57069165ea5167ec5d524`, byte-identical** to the same region of the refwd wheel's build tree (source of the built refwd wheel) |
| residual diff of `qmm.hip` vs canonical | exactly the graft (K1 gate + `lx_cols` guard), nothing else |
| wheel | `mlx-0.32.3.dev20261102+75ca881-cp314-cp314-linux_x86_64.whl` (22,479,022 B) |
| wheel sha256 | `f6873c65201ef5b2235be0c02b2b0c8b77ed1e5c10cf1b2bf9920c8019ac532a` |
| libmlx.so md5 | `5bf011c3cc3eb6502475fae61e4ef3af` (distinct from the canonical `b1de59fd…`; the graft changes code, expected) |
| venv | NEW scratch venv (CPython 3.14.8, same layout as the canonical venv; created by read-only copy of the canonical venv — the canonical venv untouched — then the mlx package replaced by zipfile extraction of this wheel) |

**Gate (PASS)**:
- capability probe (the t1375 script, run against the scratch venv):
  **4/4 ops ok** — `gdn_fused_step_tape`, `gdn_fused_step`, `gdn_prep`,
  `gdn_norm_gated`;
- `strings` on the scratch venv `libmlx.so`: `MLX_ROCM_REFWD_M1` present
  (1; **0 in the canonical libmlx.so**), `MLX_ROCM_QMV_ROWS` (3),
  `MLX_QMV_LX` (1), `MLX_ROCM_GDN_TAPE` (3), `MLX_ROCM_GDN_FUSE` (2).

Note: the version string matches the canonical wheel's (same base); the two
wheels are distinguished by sha256 (`f6873c65…` candidate vs
`5bc7ca20…` canonical, `REPORT-rebaseline.md` section 3) and libmlx md5.

## 4. Task 3 — GPU gate (2 R0 runs, exclusive window)

Lock: free at start (first 15 s poll, no bounded wait needed); held across both
runs; released on exit (verified free afterward; no orphan GPU processes).
Env per run (canonical R0 protocol, `run-anchors.sh` form): `MODELS=<model
mount>`, `LD_PRELOAD=libhsa-runtime64`, `HCC_AMDGPU_TARGET=gfx1201`,
`MLX_MTP=4`, `MLX_ROCM_GDN_TAPE=1`, `MLX_ROCM_GDN_FUSE=1`; rows OFF,
`MLX_ROCM_REFWD_M1` unset. Bench `mtp-bench-k3-tape.py` md5 `2c2d6c90…`
verified before use; guard `gdn_tape_commit.py` md5 `4a262a0d…` (canonical
pin).

| run | tok/s | TOKS | D-ACC | single ms | verify ms | refwd ms | n_refwd | TAPE n_ok/n_fb | first20 md5 |
|---|---|---|---|---|---|---|---|---|---|
| R0-1 | 60.97 | 408 | 93/59/27 | 9.74 | 33.07 | 9.25 | 101 | 128/0 | `c5c5d688…` (both runs identical) |
| R0-2 | 60.92 | 408 | 93/59/27 | 9.72 | 33.12 | 9.26 | 101 | 128/0 | |
| **median** | **60.945** | — | — | 9.73 | 33.10 | 9.255 | 101 | 128/128 | in-session spread 0.08% |

**Gate verdict: PASS (borderline, at the top of the band).** Window
`[59.1×0.97, 59.2×1.03] = [57.33, 60.98]`: both runs inside (R0-1 at
60.97, 0.006 tok/s under the upper edge). Vs the baseline-of-record pair
(59.17/59.25): R0-1 +2.90%/+3.04%, R0-2 +2.82%/+2.96% — the median sits at
+3.0% of 59.21, within measurement noise for this box (the box has shown
multi-percent wheel-state drift between rounds; the signature is otherwise
identical to the canonical B0: D-ACC 93/59/27, n_refwd 101, single 9.73 vs
9.75, TAPE 128/128 n_fb=0, TAPE_CALLS=5496, FUSE_CALLS=0). The default path
is unperturbed by the graft; the candidate wheel in fact runs ~3% faster than
the baseline-of-record in this window — consistent with the graft being
default-off and the box being in a slightly better state than the baseline
window.

## 5. Bottom line

- The ~6% B0 delta is wheel-induced and localized to the **M=4 verify QMM
  dispatch layer** (t1251 mr2 + N-floor + f32 wide support, in the canonical
  graft, absent from the refwd wheel tree). The M=1 path is bit-identical
  between the wheels; the tape frame is the same object in both.
- The candidate unified wheel (canonical tree + the minimal default-off
  REFWD_M1 graft) builds cleanly, passes the capability gate (4/4 probe,
  `MLX_ROCM_REFWD_M1` present, all other gates present), and its R0 runs
  pass the 59.1–59.2 ±3% gate (borderline at the top of the band). It is the
  vehicle for the next refwd A/B round: the refwd toggle now coexists with
  the full canonical graft, so an A/B on one wheel isolates the M=1 re-route
  from base-tree drift.

## Persona scrub

Scrubbed the diff + commit message for the standard persona patterns (foreign
hostnames, tailnet/LAN IP ranges, foreign user + home paths, the model-mount
path, codenames): **0 hits**.
