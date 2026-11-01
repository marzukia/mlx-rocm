# m2 draft-collapse RCA — neon (RX 9070 XT / gfx1201)

Branch: `rca/m2-draft-collapse` (base `da9eeca`). Question: `MLX_MTP=2` (fp32 MTP sidecar) acceptance fell from 388/512 to 0/512 between two wheels. Same venv, same sidecar, same bench — allegedly. AR healthy on both. Is it the graph-wave deltas, the gdn graft, or something else?

## 1. Verdict

**It is neither.** The MTP sidecar file was swapped between the two runs. The good run used `model-mtp.safetensors`; at 07:51:59 the bench script was edited to default to `model-mtp-fixed.safetensors`, whose seven RMSNorm weights are shifted by exactly **−1.0** relative to the original (vLLM-style `(1+w)` norm weights vs. the module's `x·w` application). That makes the drafter output systematically wrong tokens → 0/512. The MTP sidecar is loaded into the drafter only (draft-side): the target/AR path never reads it, which is why the generated (target) text is unaffected by the swap and AR stays healthy.

Decisive A/B (same wheel, only the sidecar differs):

| wheel | sidecar | m2 result |
|---|---|---|
| 1c-class (B3-era, .so `8679f3`) | `model-mtp-fixed` (default) | **0/512** @ 42.58 tok/s |
| same wheel | `model-mtp` (original, renamed in) | **388/512** @ 69.13 tok/s |

388/512 is reproduced exactly with the original sidecar — on a wheel that the allegation says was "broken". The wheel is exonerated; the sidecar is the culprit.

## 2. The "wheel-only" premise was false

Forensic timeline (neon, 2026-11-01, UTC; `btime`/`mtime` + log evidence):

| time | event |
|---|---|
| 04:21 | venv `venv-fused2` created (mlx_lm 0.32.2) |
| 04:42 | "good" wheel built (`/home/<host-user>/wheelout`) |
| 06:41 | 1093-vendor wheel built |
| 07:24:23 | `mtp_inject_patch.py` written: injects `MTPModule` into `mlx_lm/models/qwen3_5.py` |
| ~07:25 | patch v1 applied (fp32 MTPModule, no quant path) |
| 07:24:34 | main model safetensors updated |
| **07:26:01** | **`bench/mtp-bench3-side.py` created** (copy of mtp-probe2 + sidecar loader + m2 mode) |
| **07:26:34** | **GOOD run: `/tmp/bm2.log` — `MTP=2 ... 62.73 TOKS_S ... ACCEPTED=388/512`** |
| 07:37–07:43 | 4bit sidecar swapped (old → `.bak`, new written) |
| 07:43:27 | `qwen3_5.py` re-patched (v2: adds `quant=` path, `MTP_QBITS`, TextModelArgs MTP fields) — functionally identical to v1 on the fp32 path |
| **07:51:59** | **`mtp-bench3-side.py` edited** — among other things the default sidecar became `model-mtp-fixed.safetensors` |
| ~08:00+ | broken runs begin: `bench1c.log` 09:21 `MTP=2 ... 42.69 TOKS_S ... ACCEPTED=0/512` |

Consequences for the RCA:

1. **The bench changed mid-experiment.** Every "broken" bench (and m1, m0 probes) used the post-07:51 script; the good run used the 07:26 version. "Same bench claimed" is false.
2. **The sidecar changed mid-experiment.** The 07:51 edit made `model-mtp-fixed.safetensors` the default. "Same sidecar claimed" is false — and this is the load-bearing change (Section 4).
3. **The venv changed mid-experiment.** `qwen3_5.py` was patched 07:24/07:43 (MTP_INJECT_PATCH). Verified v1→v2 is functionally identical on the fp32 path (same module construction; `quant=False` default; the TextModelArgs fields only enable `lm.mtp` auto-construction). Not the culprit, but it belongs in the delta.
4. **Every wheel tested reproduced 0/512 under the fixed sidecar** — the 04:42 wheel, the 1093 wheel, the B1/B2c/B3 bisect builds, and the 1c wheel. Under the original sidecar the same wheel family gives 388/512. The wheel-vs-wheel comparison the task set up was measuring the sidecar all along.

## 3. Isolating the broken component (draft vs. verify vs. target)

All probes on the broken default (fixed sidecar), same wheel:

- **m2 (batched verify): 0/512** (repro of the incident).
- **m1 (two 1-token verifies, GDN save/restore): 0/512 @ 49.71 tok/s** → the break is not m2-batch-specific; the *draft* side is broken in both modes.
- **Target-stream byte-identity, scoped.** The verified byte-identity covers the same-wheel comparison: the 1c-class wheel (`.so` `8679f3`) run with the fixed sidecar vs. the same wheel run with the original sidecar (Section 1 A/B) — the target streams match token-for-token, and AR tok/s stays in the healthy 72–80 band → the target model and all its forwards are fine. Cross-wheel streams legitimately differ: `bm2` on the 04:42 "original" wheel vs. `bench1c` on the 1c wheel first diverge at **token 203** — different kernel sources round differently and a greedy argmax eventually flips. That divergence is expected cross-wheel drift, not a sidecar effect: the sidecar is draft-side only (loaded into the MTP drafter; the target/AR path never reads it), so a sidecar swap cannot move the target stream.
- `mtp_wcheck` (fp32 sidecar vs. module state): **15/15 tensors exact (max diff 0.0)** → the sidecar-loading path (`_setp` loop) is not corrupting weights.
- `mtp_ref` (hand-written fp64/numpy reference of the 1-layer MTP block, weights read straight from the sidecar): module output matches the reference to **rel 1.7e-07 (MODULE-OK)** → the MTPModule math (incl. the `v·sigmoid(gate)` gated attention) is correct.
- `diag1` (fresh per-step MTP cache, step-0 draft): **draft argmax = 51167 vs. ar_next = 1919; 0/16 hits with prenorm hidden AND 0/16 with postnorm** → the draft is wrong from the very first step, independent of cache state, GDN save/restore, or the hidden-flavour choice. The only remaining input is the sidecar itself.

## 4. The mechanism: `model-mtp-fixed` is a −1.0-corrupted sidecar

`sidecar_diff` (`model-mtp` "original" vs. `model-mtp-fixed`), 15 tensors:

| tensor | status |
|---|---|
| mtp.fc.weight, mtp.layers.0.mlp.{down,gate,up}_proj.weight, mtp.layers.0.self_attn.{k,o,q,v}_proj.weight | **identical** |
| mtp.layers.0.input_layernorm.weight | diff max 1.0 (mean 1.0991 → 0.0991) |
| mtp.layers.0.post_attention_layernorm.weight | diff max 1.0 (mean 1.2413 → 0.2413) |
| mtp.layers.0.self_attn.q_norm.weight | diff max 1.0 (mean 1.7573 → 0.7573) |
| mtp.layers.0.self_attn.k_norm.weight | diff max 1.0 (mean 1.7445 → 0.7445) |
| mtp.norm.weight | diff max 1.0 (mean 2.4258 → 1.4258) |
| mtp.pre_fc_norm_embedding.weight | diff max 1.0 (mean 0.5203 → −0.4797) |
| mtp.pre_fc_norm_hidden.weight | diff max 1.0 (mean 0.7725 → −0.2275) |

Exactly the seven RMSNorm weights — `mtp.layers.0.input_layernorm.weight`, `mtp.layers.0.post_attention_layernorm.weight`, `mtp.layers.0.self_attn.q_norm.weight`, `mtp.layers.0.self_attn.k_norm.weight`, `mtp.norm.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight` — each shifted by a constant −1.0. The `MTPModule` RMSNorms apply `x̂·w` with `x̂` unit-RMS, i.e. they consume the weights *as stored*. The original file stores vLLM/transformers-style `(1+w)` weights, so the module effectively applies `x̂·(1+w)` — the correct vLLM MTP dataflow. The "fix" that produced `model-mtp-fixed` subtracted 1.0 to convert to the `w` convention, but the module in this build does **no** `+1` compensation, so the "fixed" file applies `x̂·w` — every norm channel scaled by `w` instead of `1+w`. With mean norm weights ≈ 0.1–1.4, that is a gross per-channel distortion of every normalized vector in the drafter: fc output, attention input/outputs, MLP, final norm. The draft logits come out of the right ballpark of being garbage and argmax never matches the target's greedy token: **0/512**, while the target stream (untouched; the sidecar is draft-side only) is identical on the same wheel.

The 4-bit and 8-bit sidecars are quantizations of the *broken* file (sibling parity probe: "8bit" rel 0.0068 vs. fixed-fp32 ≈ fp8 noise; the old "4bit" (`.bak`) is a true 4-bit of fixed, rel 0.13). All three non-original sidecars are therefore equally broken; only `model-mtp.safetensors` works.

## 5. The wheel bisect, re-interpreted

The graft/wave split the task set up is real but not the cause:

- Full graft revert breaks the **build**: `KVAppend` is declared in the grafted `fast_primitives.h` but implemented in the wave's `copy.hip` — the graft is load-bearing, not a free revert.
- Keep graft, revert only the 4 `r4d_gdn_*` kernels → **0/512**.
- B1 (flash_attention/rope/indexing → 75ca881): **0/512**.
- B2c (copy_general_dynamic → 75ca881; wave `copy.hip` + allocator kept): **0/512**.
- B3 (≈ good reference: 75ca881 + wave qmm only, untracked qmv kernels): **0/512**.
- 04:42 "good" wheel: **0/512**.
- Same wheel family + original sidecar: **388/512**.

Every "0/512" cell above was running the post-07:51 bench with the fixed sidecar. The bisect was correctly showing that the *wheel* is not the variable; it was the bench environment. (The B3 "≈ good reference" was ≈ good *wheel*, not ≈ good *run*.)

## 6. Secondary findings (bench hygiene, worth fixing)

1. **Guard leak on crash.** The sidecar guard renames other sidecars to `.disabled`; a crashed run leaves them parked, and the *next* run's cleanup `os.rename` then raises `FileNotFoundError` (observed once during this RCA; recovered). Make the cleanup `try/finally`-idempotent (skip missing files).
2. **Unguarded `load()` crashes.** With an extra `model*.safetensors` sidecar present, `mlx_lm` merges its keys and then trips `ValueError: [quantize] bits 32 not supported` (config.json `quantization` re-quantization path). The guard exists for this; any new probe must park non-active sidecars before `load()`.
3. **Shared-venv race.** The `<host-user>` account is shared across sessions; during this RCA the venv's `core...so` was swapped by a concurrent session (1c → 1093, md5 `8679f3…`), and the 1c wheel file (a pip ephemeral build) was since deleted. The 1c wheel was therefore **rebuilt from its documented source state** (graft kept + the 4 `r4d_gdn_*` kernels reverted to 75ca881, clean build, `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS"`), installed back into the venv, and verified healthy (m0 AR 73.58 tok/s; `/tmp/benchAR_1c.out`). The rebuild's `.so` md5 is `8679f3…` — byte-identical to the pr3/1093 family, i.e. the 4 reverted kernels do not change this build's `.so`; the 1c-era bisect cells are therefore all the same wheel family, which the verdict in Section 1 already covers. The original `6c125d…` file cannot be recovered (ephemeral cache deleted); the rebuild is the functional 1c. Pin wheel identity by md5 in any long-running bench session and re-verify before trusting a run.
4. **`mtp_ref.py` had a syntax error** (unclosed paren) and a wrong `mx.dequantize` positional-arg call — both fixed to run the probes this report relies on.

## 7. Fix

1. **Bench:** default the active sidecar back to `model-mtp.safetensors` (the working file). If a true mlx-conversion is ever wanted, the correct conversion for this build is *none* — the module consumes `(1+w)` as-is. `model-mtp-fixed` (and the 4-bit/8-bit variants derived from it) should be re-derived from `model-mtp.safetensors`.
2. **Bench:** hash-pin the active sidecar at startup (print the md5) so a future swap is visible in the log.
3. **Bench:** make the guard cleanup crash-safe (Section 6.1).
4. No wheel change is required for the acceptance fix; the 1c wheel (or any wheel of this family) with the original sidecar gives 388/512.

## 8. Evidence index (neon)

- Good run: `/tmp/bm2.log` (07:26:34) — `62.73 tok/s, 388/512`.
- Broken run: `/tmp/bench1c.log` (09:21) — `42.69 tok/s, 0/512`; target stream diverges from the good run at token 203 — expected cross-wheel drift (different kernel sources round differently; a greedy argmax flips), not a sidecar effect.
- m1 broken: `/tmp/benchM1.log` — `0/512`.
- A/B: `/tmp/benchM2_orig3.log` — original sidecar on the 1c-class wheel: `69.13 tok/s, 388/512`.
- Weight load check: `/tmp/wcheck.out` (15/15 exact).
- Module-vs-reference: `/tmp/mtp_ref.out` (rel 1.7e-07, MODULE-OK; step-0 draft 51167 vs. ar_next 1919).
- Hidden-flavour probe: `/tmp/diag1.out` (0/16 prenorm, 0/16 postnorm).
- Sidecar diff: `/tmp/sidecar_diff.out` (7 RMSNorm weights, −1.0 constant shift each; keys listed in Section 4).
- Script mtimes: `bench/mtp-bench3-side.py` created 07:26:01, edited 07:51:59; `mlx_lm/models/qwen3_5.py` 07:43:27; `model-mtp-4bit.safetensors` swapped 07:37–07:43.


## PR3-HYBRID fused wheel — durability restoration (the 80.93 tok/s wheel, from git)

The 80.93 tok/s fused-AR wheel (gdn_fused_step live, +11.5% vs 72.57 tok/s no-fuse) existed only as a dirty vendor tree + installed venv on the GPU host. This section records the tree↔repo relationship, the established pristine graft state, and the extraction evidence. Deliverables: `BUILD-RECIPE.md` (the exact proven build + install + bench commands), `patches/qwen35_fuse_branch/` (the model-side patch), `bench/mtp-bench3-side.py` (the as-run side bench; the `mtp-bench3.py` no-fuse arm was already in the repo).

### Vendor tree ↔ repo relationship (first step)

- The GPU host's vendor tree (`<guest>/vendor/mlx`) at the time was a FRESH shallow clone (2026-11-01 06:41 UTC, done by the concurrent t1196 bisection run) of the private ROCm-port fork (`<vendor-fork>`), detached at `FETCH_HEAD` = `75ca881` (full sha `75ca881b2e82f7b4069c8564a203792aa16d8a29`). `75ca881` is a SINGLE-commit snapshot of the entire mlx source with the ROCm port (GDN tail fusion + the `gdn_prep` launcher pointer fix in the base; the reflog shows only `clone` + `checkout`).
- This repo is the public wrapper: `pyproject`/`scripts`/`docs`/`patches`/`reports` plus a committed, pruned copy of the vendor tree at `vendor/mlx-rocm-upstream` (STALE: it predates the graft, lacks root files like `CMakeLists.txt`/`docs`/`examples`, and is NOT byte-identical to `75ca881` despite the `vendor.json` comment). `vendor.json` pins `sha 75ca881` with the `fork` field pointing at THIS repo — that fork field is stale/inaccurate: `75ca881` is a commit of the vendor fork, not of this repo (this repo's main = `da9eeca`).
- SHA comparison: `75ca881` = vendor base (the wheel's `+75ca881` suffix); `9320ee9` = the `t1082-exec-replay-fix` branch tip in this repo (a different workline — the exec-cache replay fix; force-updated `255d441` → `9320ee9` around the same date, NOT the PR3-HYBRID graft); main `da9eeca` = the graph decode 2nd-capture fault fix (#2). None of `9320ee9`/`da9eeca` are vendor-base commits; the graft applies to `75ca881` only.

### Pristine graft state — established unambiguously (despite the t1196 race)

A concurrent run (t1196) was actively bisecting the vendor tree during this restoration (it later DELETED both backup dirs and the side bench copy mid-run; all evidence was pulled before the deletions). The graft is nonetheless unambiguous:

- **Graft = base `75ca881` + `patches/gdn_fused_step/gdn_fused_step.patch`** (the patch file is already in main, md5 `b9693851…`, byte-identical between the GPU-host repo copy and this repo).
- Evidence:
  1. The patch dry-runs cleanly against `75ca881` (6 files, 0 rejects, 0 fuzz).
  2. The patch's `gdn_fused_step.hip` is byte-identical (md5 `d8d4a30f…`) to the working-tree copy in the pre-build vendor snapshot (taken 2026-10-31, before the 2026-11-01 04:42 wheel build), whose `fast.cpp.orig`/`fast.h.orig`/`fast_primitives.h.orig` are the pre-patch backups.
  3. The built wheel's `libmlx.so` contains `GdnFusedStep` (34 symbol hits) and `gdn_fused` in the python extension; it LACKS `RMSNormResidual` (0 hits) — the K2 `rms_norm_residual` graft present in the pre-build working-tree snapshot was NOT in the built wheel. The proven build tree was base + GDN patch only.
  4. The installed venv (`venv-fused2`, CPython 3.14.8) reports `mx.fast.gdn_fused_step = True`, `rms_norm_residual = False`.
  5. The r4d GDN kernel files (`r4d_gdn_*.hip`, `r4d_gdn_wmma.h`, `gated_delta_update.cpp`) differ from base `75ca881` in the pre-build snapshot by a COSMETIC comment scrub only (a provenance attribution line, in 4 files) — no functional r4d kernel change is part of the graft; the r4d GDN kernels in the base are already GDN-ready.

### Model-side patch

Extracted by diffing the installed `<venv>/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py` (25,912 bytes, mtime 2026-11-01 07:43) against stock `mlx-lm==0.32.2` (fresh `pip download --no-deps`): `patches/qwen35_fuse_branch/qwen35_fuse_branch.diff`. Verified: the diff applies cleanly to stock and round-trips to a BYTE-IDENTICAL copy of the installed file. Markers: `FUSE_BRANCH_PATCH` (the S==1 && B==1 branch dispatching to `mx.fast.gdn_fused_step`, gated on `MLX_ROCM_GDN_FUSE=1`, with the `GDN_FUSE_CALLS` dispatch counter; idempotent apply = grep the marker first), plus `MTP_INJECT_PATCH` (MTP drafter module) and `PRENORM_PATCH` (pre-norm hidden plumbing) — both inert under `MLX_MTP=0`.

### Result

Fused AR (S==1, `MLX_ROCM_GDN_FUSE=1`): **80.93 tok/s**; no-fuse AR (`MLX_MTP=0`, no env gate): **72.57 tok/s** — **+11.5%**, lossless (byte-identical token streams, per the as-run bench pair `bench/mtp-bench3.py` / `bench/mtp-bench3-side.py`). The exact proven build + install + bench commands are in `BUILD-RECIPE.md`.
