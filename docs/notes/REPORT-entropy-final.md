# REPORT — ent4 final (run-1045): run-1044 fixes ported + per-tensor E<3.9 gating

Continuation of run-1042 (ent4 lever 1: lossless 4-bit weight compression, 6.5 %
memory, ~0.1 % overhead) and run-1044 (3 compressor fixes + re-measure).
Branch `phase3-entropy-compress`, commits 500eb37 / 306b746 / dc47877
on top of run-1042's 7f9d1bd.

## 1. The E discrepancy: two model builds, not two code paths

run-1044's re-measure (aggregate E 3.9707; classes el 3.46 / attn 4.21 / mlp
4.08) and run-1042's (3.713; range 3.62–3.75) both measured the *code* correctly
but different *inputs*:

| build | shards | tensors | U32 | naming |
|---|---|---|---|---|
| /models/qwen3-9b-mlx (served; run-1042's input) | 2 | 927 | 250 | `language_model.*` |
| /models/qwen3-9b-mlx-clean (run-1044's input) | 1 (5 GB) | 927 | 250 | `language_model.*` |

Running run-1044's exact script (md5-verified 8ec0580) on the -clean build
reproduces its log values exactly (lm_head 2.0498, l0.down 4.8877, l1.down
2.0000, embed 4.8780). The -clean build contains degenerate (1–2 distinct
symbol) and near-uniform tensors; the served build's 250 U32 tensors all have
16 distinct symbols and per-tensor E in [3.537, 3.754]. The MTP shards
(both builds) have 0 U32 tensors.

**Per-class E, final served model** (compressor's own path, this work):

| class | tensors | E_huff (bits/sym) |
|---|---|---|
| el (embed + lm_head) | 2 | 3.7472 |
| attn | 152 | 3.7331 |
| mlp | 96 | 3.7392 |
| aggregate | 250 | 3.7368 (run-1042's independent histogram: same) |

## 2. Step 1 — run-1044's 3 fixes ported onto run-1042's final compressor

- **764fc08** (timing/RSS/traceback): per-tensor `t=… rss=… total_t=…`
  diagnostics, per-tensor `del` of the big arrays, `main → _main` with
  traceback guard.
- **4f4a9e1** (int32 cumsum): per-chunk code-length cumsum in int32 (max
  4096×12288×15 = 7.5e8 < 2^31; the global row-start cum stays int64).
- **8ec0580** (stream-end OOB): the 15-bit window decoder reads a 4-byte
  window; the last code's window can extend up to 3 bytes past the content
  end. run-1044 fixed it in its `--verify` clamp; the equivalent fix in this
  stack is a +3 zero pad on the consumer side: `ent4_verify.py` (C
  verifier), `ent4_json.h` (GPU payload `stream.resize(comp_bytes + 3)`),
  and the `load.cpp` patch (weight buffer allocation + `hipMemcpy`, with an
  idempotent re-apply edit). Documented in `docs/entropy4.md`.

Verification: the ported compressor's output is **byte-identical** to
7f9d1bd's on synthetic data (uniform E=4.0, skewed E=3.15, N>4096 chunk
split); the C verifier (with the +3 pad) passes 7/7 OK, bad=0.

## 3. Step 2 — per-tensor E<3.9 entropy gate (decision C)

A tensor with measured `E >= gate` (default 3.9; `--gate 0` = compress-all
legacy) is written **plain**: a canonical 4-bit encoding (identity symbols,
lengths all 4, stream = raw packed bytes high-nibble-first,
`row_start[r] = r·k·4`). The decoder works unchanged — a length-4 canonical
code is symbol identity, the 11-bit window always contains the 4 code bits,
and the LUT covers every 11-bit suffix. The per-tensor header gains a flag
byte (`<IIII B>`); consumers were updated in lockstep. The compressor now
reports per-tensor PLAIN/COMP plus a per-class (el/attn/mlp) aggregation,
and `index.json` carries `gate` + `tensor_info` diagnostics.

**On the final 9B model the gate is a no-op**: `GATE 3.9: plain=0
compressed=250` (per-tensor E ≤ 3.754). Size cross-check: gated file
4,192,642,800 B = run-1042's 4,192,642,550 B + 250 flag bytes; the compressed
streams are otherwise identical. The gate is the safety mechanism for models
where a class runs above 3.9 (e.g. the -clean build's attn 4.21 / mlp 4.08
tensors would go plain, recovering exactly the 4-bit size on those tensors
while keeping the win on the rest).

## 4. Step 3 — lossless gates (the build VM, RX 9070 XT / gfx1201, ROCm 10.1)

Wheel rebuilt in the existing `mlx-m8r2` tree (updated `ent4.h` /
`ent4_json.h`, `load.cpp` +3 allocation + memcpy; `pip wheel` OK; installed
into `venv-m8`).

| gate | result |
|---|---|
| CPU verifier, gated .ent4, 250 tensors | **250/250 OK, bad=0** |
| GPU kernel gate (standalone HIP, ktest updated for flag byte + 3-byte pad; tensors 0/1/249: embed 248320×4096, small attn 32×4096, mlp up_proj 12288×4096) | **3/3 bad=0** |
| Full-stack A/B (fresh wheel, gated file; forward logits + 32-token generation, baseline vs ent4) | **logs byte-identical** |

## 5. Step 4 — A/B bench (flock `/models/gpu-bench.lock -w 1800`)

**64-token arm** (the run-1042 65.77 setup; 3 samples each, warm-up discarded):

| arm | samples (tok/s) | median |
|---|---|---|
| baseline AR | 66.20 / 65.91 / 65.83 | **65.91** |
| ent4-GATED | 65.85 / 65.90 / 65.84 | **65.85** |

Overhead ≈ 0.1 % (speed-neutral, as run-1042); generated text identical in all
6 runs.

**29k-ctx arm** (29,000-token prompt; A = small-context AR cross-check,
B = warm + measured, B2 = fresh-process measured-only):

| arm | baseline | ent4-GATED | delta |
|---|---|---|---|
| A (AR, tok/s) | 72.29 | 73.16 | +1.2 % (noise) |
| B (decode, win, tok/s) | 61.86 | 61.60 | −0.4 % (parity) |
| B (decode, all, tok/s) | 62.09 | 61.83 | −0.4 % |
| B2 (fresh, decode win, tok/s) | 60.85 | 57.25 | −5.9 % |

Both B arms hit `LOSSLESS-GATE=PASS` (warm vs measured stream identity, with
the ent4 decode path active). The B2 gap is a **measurement artifact**:
B2 has no warm pass, so the ent4 arm's first-touch decode (kernel init +
first decode of each registered tensor) falls inside the measured window —
the same warm-up that the 64-token arm discards. Steady-state (warmed)
decode is at parity; the 29k run is dominated by prefill (≈58 s) anyway.
Load time: +1.2 s for the ent4 file (4.18 GB read + per-tensor LUT build).

## 6. Bottom line

- run-1044's 3 fixes are ported and re-verified (byte-identical encoding,
  lossless verifier).
- Decision C's E<3.9 gate is implemented end-to-end (compressor, flag byte,
  verifier, GPU payload, docs) and is a **no-op on the final 9B dataset** —
  all 250 tensors stay compressed, so run-1042's 6.5 % memory saving stands on
  this model; the gate protects future models with less compressible
  classes.
- Lossless gates green (CPU 250/250, GPU kernel 3/3, full-stack A/B
  byte-identical); A/B bench: 0.1 % overhead (64-token arm), parity on
  warmed 29k decode.
- The E "disagreement" between run-1042 and run-1044 is fully explained by the two
  different model builds (see §1).

## Artifacts (the build VM)

- gated .ent4 + logs: `/home/guest/ent4-final/` (compress.log, qwen3-9b.ent4
  + index.json, fs_base.log / fs_ent4.log, run-1045-29k*.log, gt/)
- rebuilt wheel: `/models/tmp/dist-run-1045/mlx-0.32.3.dev20261030+75ca881b-…whl`
  (installed in `/models/tmp/run-1012-29k/venv-m8`)
- updated ktest: `/models/tmp/ent4_kernel_test(.cpp)`
