# MTP R4c — int2 g128-asymmetric draft head: port, wheel, A/B

**Verdict (up front): the int2 draft head is PORTED, LOSSLESS, and WHEEL-CLEAN, but it LOSES the speed gate: 27.3 t/s vs 48.9 t/s (0.56x) on the 322-iter anchor and 0.60x on the 512 protocol. Root cause (measured, not prose): the ported `lm_head_int2` kernel is compute-bound in a scalar dequant regime — 18.7 ms per row vs 0.95 ms for the shared 4-bit `lm_head` (19.75x) in a tight warm loop — so the 2-bit weight-traffic advantage never engages. NO ADOPT for MTP draft; the bench instrumentation (`MTP6_DRAFT_HEAD=int2`, probe, xcheck, runner) ships on the branch for a vectorized (WMMA/quant-dequant) int2 kernel follow-up. The int2 lever is OFF by default in the shipped harnesses: with `MTP6_DRAFT_HEAD` / `MTP6_TAPE` unset, the canonical 4-bit path runs.**

## TL;DR (anchor, 322-iter, 3-run medians, config of record)

| arm | venv | head | t/s (3 runs) | median | FIRST20 |
|---|---|---|---|---|---|
| OFF | canon `b1de59fd` | 4-bit qmm | 48.99 / 48.89 / 48.93 | **48.93** | OK (3/3) |
| ON | r4c `03bcf1f4` | int2 + top-32 re-rank | 27.37 / 27.32 / 19.92 | **27.32** | OK (3/3) |
| CTRL | r4c `03bcf1f4` | 4-bit qmm (int2 off) | 48.94 / 48.93 / 47.79 | **48.93** | OK (3/3) |
| CTRL2 | r4c `03bcf1f4` | int2, `QMV_ROWS*` env off | 26.21 / 26.13 / 20.53 | **26.13** | OK (3/3) |

Config of record for all arms: `MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1 REFWD_M1=1` (REFWD_M1 is a no-op on both wheels — the knob is not in the vendor tree, same as the R4b runs). The arms select the head explicitly per arm (OFF/CTRL = the 4-bit qmm head, ON/CTRL2 = int2), which is what makes the A/B robust to the harness defaults.

## Lever and wheels

- **Build base = the guest `vendor-canon` worktree** (byte-verified source of the canonical wheel: repo clone vs worktree differ only in the 7 lever files + `REPORT.md` + 88 build-artifact files) + the 7 R4c lever files from this branch (`lm_head_int2.hip` new; `CMakeLists.txt`, `qmm.hip` R4b `wide_ok` fix, `fast.cpp`/`fast.h`/`fast_primitives.h` C++ op, `python/src/fast.cpp` binding). The GitHub vendor snapshot is partial (no `mlx.pc.in`); the full tree lives only on the guest.
- Wheel `mlx-0.32.3.dev20261103+75ca881` md5 `328dc04ad585d87897f166c10e237323`; venv-r4c `libmlx.so` md5 `03bcf1f4a4235f58a3a715afadfcbeae` (venv-canon untouched, `b1de59fd…` pre/post).
- Symbol delta vs the canonical wheel is exactly the int2 set (`LMHeadInt2::*`, `lm_head_int2`, `launch_lm_head_int2`, `lm_head_int2_kernel`, pybind `name`/`typeinfo`/`vtable`); GdnFusedStepTape and RMSNormResidual retained. Quick single-file `hipcc --offload-arch=gfx1201 -c lm_head_int2.hip` rc=0 before the full build.
- Sanity on venv-r4c: `INT2_OP True GDN_FUSE True FUSE_PATCH True MTP_INJECT True`.

## AR spot check (wheel gate)

Plain-greedy `mtp-bench3.py` (`MLX_MTP=0`, config of record), 1024 tokens:

| arm | venv | t/s | FIRST20 |
|---|---|---|---|
| AR OFF | canon | 64.42 | `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279` |
| AR ON | r4c | 64.56 | identical (bit-for-bit, incl. `ALLTOKS`) |

ON vs OFF = +0.2%. **Note:** the task's "74.4-class" expectation does not reproduce in this session — the *canonical* wheel ARs at 64.4 here, so the wheel-consistency gate (ON ≈ OFF) is the meaningful one and it passes; absolute AR level is a guest-state drift, recorded, not a wheel defect.

## Phase table (25 s idle before/after, lock held)

| phase | arm | t/s (3 runs) | median | tokens/verifies | FIRST20 |
|---|---|---|---|---|---|
| 64v (128-iter) | OFF | 50.01 (1) | 50.01 | 215/64 | OK |
| 64v (128-iter) | ON | 27.64 (1) | 27.64 | 207/64 | OK |
| 512v (1024-iter) | OFF | 40.36 / 40.29 / 40.31 | 40.31 | 1628/512 | OK (3/3) |
| 512v (1024-iter) | ON | 24.23 / 24.24 / 21.29 | 24.23 | 1648/512 | OK (3/3) |

The 512v OFF median (40.31) reproduces the R3 40.4-class 512 protocol (40.38) — instrument and box are consistent with the prior lane. **PENDING re-measure:** the ON rows of this table predate the correction of the phase bench's reference quantizer to the kernel's dequant coefficients (see Quantization semantics below), so the 64v/512v ON rows — and the 0.60x protocol number they carry — are PENDING re-measure on the next GPU slot (this round is code-only; the GPU is held by the AR R4 run). The NO-ADOPT verdict is unaffected: the A/B arms set the draft head explicitly per arm (anchor arm table: OFF = 4-bit qmm on the canon wheel, ON = int2 + top-32 re-rank, CTRL = 4-bit qmm with int2 off, CTRL2 = int2 with the `QMV_ROWS*` env off), and the anchor rows come from the canonical bench, whose reference quantizer was already the kernel-correct form — so the 0.56x anchor gate and the verdict stand on rows this fix does not touch.

## Head-traffic probe evidence (the diagnosis)

Standalone probe on the model's own `lm_head` (N=248320, K=4096; the op is M=1-only; bench6 calls it per draft row):

| measurement | result |
|---|---|
| 4-bit shared head, 1 row | **963 µs** (probe) / 949 µs (xcheck warm loop) |
| int2 coarse op, 1 row | **19200 µs** (probe) / 18743 µs (xcheck warm loop) |
| int2 coarse + top-32 re-rank, 1 row | 19236 µs (re-rank ≈ 36 µs — negligible) |
| ratio (xcheck, warm) | **19.75x slower** |
| argmax agreement (256 seeded rows, re-rank vs shared) | 201/256 (78.5%) |
| max abs logit diff, coarse int2 vs shared | 1.935 |

Consistency with the full runs: each 322-iter step runs 4 draft rows through int2 (verify/correction rows use the shared head); 161 steps × (4×18.7 ms − 1×~3 ms) ≈ 11–12 s of extra head time vs the observed 8.4 s ON−OFF GEN delta — consistent within acceptance variance (the int2 arm also shifts D2/D3 acceptance: 62.0/45.3 vs 62.7/47.3).

Why the kernel loses: `lm_head_int2.hip` is grid=(N/256)×256 threads, one column per thread, inner loop `q×32 groups×128` scalar dequant-MAC (`li2_bf2f16` bit-pattern decode + FMA per element) plus the 8-round shared-memory top-8 selection. No WMMA/LDS tiling of the MAC. 248 K threads × 4096 MACs in 18.7 ms ≈ 54 GMAC/s effective — the 242 MB packed weight read would take ~0.5 ms at HBM speed, so the kernel is compute-bound, not bandwidth-bound: the 2-bit traffic saving never engages at M≤4, and the scalar dequant costs ~20x the 4-bit quantized GEMM path. The radiance 1.2–1.4x expectation presupposes a vectorized int2-dequant GEMM; this port does not have one. (For the record: no prior lane ever measured int2 on a built wheel — R3's wheel did not carry the op, so the 1.2–1.4x is an expectation, not a baseline.)

## Quantization semantics (the kernel is the truth)

The dequant algebra the kernel executes is, per 128-group (kernel refs: `lm_head_int2.hip:48` `Q = K>>2`, `:49` `GQ = Q>>7`, `:64` per-group line, header contract at `:5`):

    out += dot_g * scale_g − sx_g * zs_g        (scale_g = s,  zs_g = z·s)

with `dot_g = Σ_k x_k·q_k`, `sx_g = Σ_k x_k` (host-precomputed), and the quantizer `s = max((hi−lo)/3, 1e-8)`, `z = clamp(round(−lo/s), 0, 3)`, `q = clamp(round(w/s + z), 0, 3)`. Substituting, `s·dot_g − sx_g·(z·s) = Σ_k x_k·s·(q_k − z)` — exactly the dequant of `ŵ_k = s·(q_k − z)`, i.e. the levels map `q=0 → −z·s` (≈ lo) through `q=3 → (3−z)·s` (≈ hi). That is the correct form, and the lossless bench proves the kernel's semantics end-to-end: **scale = s, zs = z·s**.

One revision of the phase bench's reference quantizer wrote `scale = 4·s`, `zs = (4+z)·s`; its per-group contribution is then `Σ_k x_k·s·(4·q_k − 4 − z_k)` — every group rescaled by 4× plus a per-element offset `−4·s·x_k`, a non-constant transform that distorts the logit ranking and had perturbed the 64v/512v phase rows behind the 0.60x protocol number. The reference now writes `s` and `z·s`, matching the kernel and the canonical bench's reference (which always had the correct form).

## Host-side guards (fast.cpp)

The kernel's real shape requirements, traced from `lm_head_int2.hip`:

- **N % 256 == 0.** The grid is `ceil(N/256)` blocks of 256 threads; threads with `col >= N` early-return before writing their shared entries, but the 8-round top-8 selection still runs over the whole `sh[256]` and the per-sub-block candidate writes still execute. A partial block (N%256≠0) therefore (a) runs the selection over dead shared columns (stale/unwritten entries can win a top-8 slot and point at an out-of-range column) and (b) has dead sub-blocks write past the `[N/64, 8]` candidate arrays (e.g. N=257 → block 1 sub-blocks 1–3 write cand indices 32–48 of a 32-element array). The old `N % 64` guard did not cover either hazard.
- **K % 512 == 0** (equivalently `(K/4) % 128 == 0`, i.e. `GQ = (K/4)>>7` does not round). `GQ` (line 49) truncates: each quarter's loop covers only `GQ·128` elements, so a quarter whose length is not a multiple of 128 — i.e. K%512≠0 — silently drops the tail 128-group from the dot product.

Verified-safe target (qwen3-9b, the only shape this branch runs): **M = 1** (the bench reshapes `x` to `[1, K]`; the host `M==1` guard checks it), **K = 4096 = 8×512** → `GQ = (4096/4)>>7 = 8 = K/512` exact, all 4×8 = 32 = `K/128` groups covered, `scale`/`sx`/`zs` indices `G ∈ [0, 32)` all in-bounds; **N = 248320 = 970×256** → grid exactly 970 blocks, no partial block (no dead shared columns, no OOB candidate writes; the last write `(969·4+3)·8+r < (N/64)·8`). The target shape reads no dead columns and drops no tail group — the earlier `N%64`-only guard was nevertheless weaker than the kernel requires for other shapes. `fast.cpp` now throws for `N%256≠0` and `K%512≠0` (on top of the pre-existing M==1 / `wq [N,K/4]` / `scale,zs [N,K/128]` / `sx [K/128]` checks), so a wrong shape fails loudly instead of reading dead columns or silently dropping a tail group.

## Acceptance

| gate | criterion | result |
|---|---|---|
| Lossless | FIRST20 identical across all 12 MTP=4 runs + 2 AR runs | **PASS** (all `271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279`) |
| Wheel sanity | AR ON ≈ AR OFF on venv-r4c; FIRST20 canonical | **PASS** (64.56 vs 64.42; 74.4-class not reproducible by canon either — session drift) |
| Control | CTRL (r4c, int2 off) ≈ OFF | **PASS** (48.93 vs 48.93) |
| Env isolation | CTRL2 (rows env off) vs ON | 26.13 vs 27.32 — same regime; the `QMV_ROWS*` gate does not touch the int2 path, as expected |
| **Speed** | **ON > OFF on anchor + draft-phase head traffic collapse** | **FAIL** — 27.32 vs 48.93 (0.56x) on the anchor, 24.23 vs 40.31 (0.60x on 512v — ON row PENDING re-measure, phase table note) |

## Protocol notes (two incidents, both resolved inside the session)

1. The task's AR spot-check script `mtp-bench3-side.py` renames the other mtp sidecars to `.disabled` at start and restores them only on a clean `main()` return; when its `load()` crashed (sidecar layout), the canonical `model-mtp.safetensors` was left `.disabled` and broke every later arm ("Missing 15 parameters"). Fixed by (a) restoring the sidecar (md5 `02a54359…` re-verified pre/post every subsequent session) and (b) running the AR spot on plain `mtp-bench3.py` (no sidecar surgery). The bench6 arms were never affected (they glob the sidecar, no rename).
2. The first guest build attempt from the GitHub vendor snapshot failed (partial tree, no `mlx.pc.in`); the wheel was then built from the guest `vendor-canon` worktree + the 7 lever files (above).

## Checks

```
uv run ruff format --check .                          # pass (all files)
uv run ruff check .                                  # pass (0 errors; the bench harnesses are ruff-excluded as as-run evidence)
uv run pytest -m "not gpu"                           # hermetic set: 25 passed, 1 skipped

uv run python scripts/size_ratchet.py --base HEAD    # SIZE GATE: OK (2 ratcheted)
CI on the head of this report (gh API):             # lint-and-check success, size success

flock -x ~/gpu-bench.lock                           # held for every GPU run
SBCAR md5 02a54359cda8b247f76b277dd9bc4d4c         # pre/post all sessions

venv-canon libmlx b1de59fda0e4fc607d620e287deeaaf8 # untouched pre/post
```

Persona gate: grep over the full branch diff + all commit messages (the run brief's pattern set: fleet-IP block, guest-identity prefixes, home/model mount paths, junkyard scripts, mrzk.io mail): **0 hits**.
Line-wrap audit: hard-wrapped lines in the report and in the branch commit bodies: **0**.
Authorship: author AND committer = `jimmyateworld <jimmy@ag.mrzk.io>` on every branch commit, **0 fallbacks**.

**CI note:** on the head of this report both CI jobs (`lint-and-check`, `size`) pass. The failures seen on earlier heads were inherited base state (verified on a `72e33a9` worktree: 982 ruff errors in vendor-python and the same two NEW BLOAT test files, `test_array.py` 2472 / `test_ops.py` 3618 > 2400, not ratcheted); the R4b lane's branch rewrite (`09ddc98`, not a descendant of `72e33a9`) is the clean line. This diff adds no gate-relevant content.

Build log tail (`~/r4c-build2.log`):

```
grafted: int2=1 r4b=1 cmlist=1
quickcheck rc=0
=== full build start Tue Nov  3 10:34:00 AM UTC 2026
=== full build rc=0 done Tue Nov  3 10:37:02 AM UTC 2026
  Created wheel for mlx: filename=mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl size=22490943
328dc04ad585d87897f166c10e237323  ~/wheelout-r4c/mlx-0.32.3.dev20261103+75ca881-cp314-cp314-linux_x86_64.whl
```

**Recommendation:** keep the int2 lever OFF (default) in the served wheel — with `MTP6_DRAFT_HEAD` and `MTP6_TAPE` unset, both bench harnesses now run the canonical 4-bit path by default. The follow-up that makes this lever adoptable is a vectorized int2-dequant GEMM (WMMA/quant-dequant, or a 2-bit path in the quantized GEMM) — the A/B harness for it is ready on this branch (`MTP6_DRAFT_HEAD=int2`, `r4c_head_probe.py`, `r4c_xcheck.py`, `r4c-runner.sh`).
