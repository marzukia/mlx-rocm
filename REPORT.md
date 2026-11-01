# m2 draft-collapse RCA — neon (RX 9070 XT / gfx1201)

Branch: `rca/m2-draft-collapse` (base `da9eeca`).
Question: `MLX_MTP=2` (fp32 MTP sidecar) acceptance fell from 388/512 to 0/512
between two wheels. Same venv, same sidecar, same bench — allegedly. AR healthy
on both. Is it the graph-wave deltas, the gdn graft, or something else?

## 1. Verdict

**It is neither.** The MTP sidecar file was swapped between the two runs. The
good run used `model-mtp.safetensors`; at 07:51:59 the bench script was edited
to default to `model-mtp-fixed.safetensors`, whose six RMSNorm weights are
shifted by exactly **−1.0** relative to the original (vLLM-style `(1+w)` norm
weights vs. the module's `x·w` application). That makes the drafter output
systematically wrong tokens → 0/512. The target/AR path never touches the MTP
sidecar, which is why the generated text stays byte-identical and AR stays
healthy.

Decisive A/B (same wheel, only the sidecar differs):

| wheel | sidecar | m2 result |
|---|---|---|
| 1c-class (B3-era, .so `8679f3`) | `model-mtp-fixed` (default) | **0/512** @ 42.58 tok/s |
| same wheel | `model-mtp` (original, renamed in) | **388/512** @ 69.13 tok/s |

388/512 is reproduced exactly with the original sidecar — on a wheel that the
allegation says was "broken". The wheel is exonerated; the sidecar is the
culprit.

## 2. The "wheel-only" premise was false

Forensic timeline (neon, 2026-11-01, UTC; `btime`/`mtime` + log evidence):

| time | event |
|---|---|
| 04:21 | venv `venv-fused2` created (mlx_lm 0.32.2) |
| 04:42 | "good" wheel built (`/home/andryo/wheelout`) |
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

1. **The bench changed mid-experiment.** Every "broken" bench (and m1, m0
   probes) used the post-07:51 script; the good run used the 07:26 version.
   "Same bench claimed" is false.
2. **The sidecar changed mid-experiment.** The 07:51 edit made
   `model-mtp-fixed.safetensors` the default. "Same sidecar claimed" is false —
   and this is the load-bearing change (Section 4).
3. **The venv changed mid-experiment.** `qwen3_5.py` was patched 07:24/07:43
   (MTP_INJECT_PATCH). Verified v1→v2 is functionally identical on the fp32
   path (same module construction; `quant=False` default; the TextModelArgs
   fields only enable `lm.mtp` auto-construction). Not the culprit, but it
   belongs in the delta.
4. **Every wheel tested reproduced 0/512 under the fixed sidecar** — the 04:42
   wheel, the 1093 wheel, the B1/B2c/B3 bisect builds, and the 1c wheel.
   Under the original sidecar the same wheel family gives 388/512. The
   wheel-vs-wheel comparison the task set up was measuring the sidecar all
   along.

## 3. Isolating the broken component (draft vs. verify vs. target)

All probes on the broken default (fixed sidecar), same wheel:

- **m2 (batched verify): 0/512** (repro of the incident).
- **m1 (two 1-token verifies, GDN save/restore): 0/512 @ 49.71 tok/s** → the
  break is not m2-batch-specific; the *draft* side is broken in both modes.
- **AR (m0) text is byte-identical** between good and broken runs
  (`FIRST16` and full 1024-token streams match `bm2.log`), and AR tok/s is in
  the healthy 72–80 band → the target model and all its forwards are fine.
- `mtp_wcheck` (fp32 sidecar vs. module state): **15/15 tensors exact
  (max diff 0.0)** → the sidecar-loading path (`_setp` loop) is not corrupting
  weights.
- `mtp_ref` (hand-written fp64/numpy reference of the 1-layer MTP block,
  weights read straight from the sidecar): module output matches the reference
  to **rel 1.7e-07 (MODULE-OK)** → the MTPModule math (incl. the
  `v·sigmoid(gate)` gated attention) is correct.
- `diag1` (fresh per-step MTP cache, step-0 draft): **draft argmax = 51167 vs
  ar_next = 1919; 0/16 hits with prenorm hidden AND 0/16 with postnorm** → the
  draft is wrong from the very first step, independent of cache state, GDN
  save/restore, or the hidden-flavour choice. The only remaining input is the
  sidecar itself.

## 4. The mechanism: `model-mtp-fixed` is a −1.0-corrupted sidecar

`sidecar_diff` (`model-mtp` "original" vs. `model-mtp-fixed`), 15 tensors:

| tensor | status |
|---|---|
| fc.weight, down/gate/up_proj, k/o/q/v_proj | **identical** |
| input_layernorm.weight | diff max 1.0 (mean 1.0991 → 0.0991) |
| post_attention_layernorm.weight | diff max 1.0 (mean 1.2413 → 0.2413) |
| self_attn.q_norm.weight | diff max 1.0 (mean 1.7573 → 0.7573) |
| self_attn.k_norm.weight | diff max 1.0 (mean 1.7445 → 0.7445) |
| norm.weight | diff max 1.0 (mean 2.4258 → 1.4258) |
| pre_fc_norm_embedding.weight | diff max 1.0 (mean 0.5203 → −0.4797) |
| pre_fc_norm_hidden.weight | diff max 1.0 (mean 0.7725 → −0.2275) |

Exactly the six RMSNorm weights, each shifted by a constant −1.0. The
`MTPModule` RMSNorms apply `x̂·w` with `x̂` unit-RMS, i.e. they consume the
weights *as stored*. The original file stores vLLM/transformers-style `(1+w)`
weights, so the module effectively applies `x̂·(1+w)` — the correct vLLM MTP
dataflow. The "fix" that produced `model-mtp-fixed` subtracted 1.0 to convert
to the `w` convention, but the module in this build does **no** `+1`
compensation, so the "fixed" file applies `x̂·w` — every norm channel scaled by
`w` instead of `1+w`. With mean norm weights ≈ 0.1–1.4, that is a gross
per-channel distortion of every normalized vector in the drafter: fc output,
attention input/outputs, MLP, final norm. The draft logits come out of the
right ballpark of being garbage and argmax never matches the target's greedy
token: **0/512**, while the target stream (untouched) is identical.

The 4-bit and 8-bit sidecars are quantizations of the *broken* file (sibling
parity probe: "8bit" rel 0.0068 vs. fixed-fp32 ≈ fp8 noise; the old "4bit"
(`.bak`) is a true 4-bit of fixed, rel 0.13). All three non-original sidecars
are therefore equally broken; only `model-mtp.safetensors` works.

## 5. The wheel bisect, re-interpreted

The graft/wave split the task set up is real but not the cause:

- Full graft revert breaks the **build**: `KVAppend` is declared in the grafted
  `fast_primitives.h` but implemented in the wave's `copy.hip` — the graft is
  load-bearing, not a free revert.
- Keep graft, revert only the 4 `r4d_gdn_*` kernels → **0/512**.
- B1 (flash_attention/rope/indexing → 75ca881): **0/512**.
- B2c (copy_general_dynamic → 75ca881; wave `copy.hip` + allocator kept):
  **0/512**.
- B3 (≈ good reference: 75ca881 + wave qmm only, untracked qmv kernels):
  **0/512**.
- 04:42 "good" wheel: **0/512**.
- Same wheel family + original sidecar: **388/512**.

Every "0/512" cell above was running the post-07:51 bench with the fixed
sidecar. The bisect was correctly showing that the *wheel* is not the variable;
it was the bench environment. (The B3 "≈ good reference" was ≈ good *wheel*,
not ≈ good *run*.)

## 6. Secondary findings (bench hygiene, worth fixing)

1. **Guard leak on crash.** The sidecar guard renames other sidecars to
   `.disabled`; a crashed run leaves them parked, and the *next* run's cleanup
   `os.rename` then raises `FileNotFoundError` (observed once during this RCA;
   recovered). Make the cleanup `try/finally`-idempotent (skip missing files).
2. **Unguarded `load()` crashes.** With an extra `model*.safetensors` sidecar
   present, `mlx_lm` merges its keys and then trips
   `ValueError: [quantize] bits 32 not supported` (config.json
   `quantization` re-quantization path). The guard exists for this; any new
   probe must park non-active sidecars before `load()`.
3. **Shared-venv race.** The `andryo` account is shared across sessions; during
   this RCA the venv's `core...so` was swapped by a concurrent session
   (1c → 1093, md5 `8679f3…`), and the 1c wheel file (a pip ephemeral build)
   was since deleted. Pin wheel identity by md5 in any long-running bench
   session and re-verify before trusting a run.
4. **`mtp_ref.py` had a syntax error** (unclosed paren) and a wrong
   `mx.dequantize` positional-arg call — both fixed to run the probes this
   report relies on.

## 7. Fix

1. **Bench:** default the active sidecar back to `model-mtp.safetensors`
   (the working file). If a true mlx-conversion is ever wanted, the correct
   conversion for this build is *none* — the module consumes `(1+w)` as-is.
   `model-mtp-fixed` (and the 4-bit/8-bit variants derived from it) should be
   re-derived from `model-mtp.safetensors`.
2. **Bench:** hash-pin the active sidecar at startup (print the md5) so a
   future swap is visible in the log.
3. **Bench:** make the guard cleanup crash-safe (Section 6.1).
4. No wheel change is required for the acceptance fix; the 1c wheel (or any
   wheel of this family) with the original sidecar gives 388/512.

## 8. Evidence index (neon)

- Good run: `/tmp/bm2.log` (07:26:34) — `62.73 tok/s, 388/512`.
- Broken run: `/tmp/bench1c.log` (09:21) — `42.69 tok/s, 0/512`; identical
  token stream to the good run.
- m1 broken: `/tmp/benchM1.log` — `0/512`.
- A/B: `/tmp/benchM2_orig3.log` — original sidecar on the 1c-class wheel:
  `69.13 tok/s, 388/512`.
- Weight load check: `/tmp/wcheck.out` (15/15 exact).
- Module-vs-reference: `/tmp/mtp_ref.out` (rel 1.7e-07, MODULE-OK; step-0
  draft 51167 vs. ar_next 1919).
- Hidden-flavour probe: `/tmp/diag1.out` (0/16 prenorm, 0/16 postnorm).
- Sidecar diff: `/tmp/sidecar_diff.out` (6 norm weights, −1.0 constant shift).
- Script mtimes: `bench/mtp-bench3-side.py` created 07:26:01, edited
  07:51:59; `mlx_lm/models/qwen3_5.py` 07:43:27; `model-mtp-4bit.safetensors`
  swapped 07:37–07:43.
