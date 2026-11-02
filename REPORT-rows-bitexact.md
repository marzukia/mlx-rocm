# QMV-ROWS bit-exact restructure — guest (RX 9070 XT / gfx1201)

Branch: `worker/rows-bitexact` (base `2d65738`, the ws[4] GPU verify). Question: the ws[4]
dodge made the multi-row QMV kernel 3.14x faster but **not bit-exact** (maxdiff 9.54e-07,
md5 mismatch, K=3 A/B diverges at token 172). Can we restructure the rows kernel so the
per-row math is *identical* to the one-row reference (`qmv_wide_kernel`), keeping the
multi-row win?

## 1. Verdict

**Yes — bit-exact, and lossless at the model level.** The restructured
`qmv_rows_msg_kernel` produces **maxdiff 0.0 (raw bit-identical) on every shape/M tested**
vs the wide reference, and the K=3 MTP A/B at the 9B model is a **byte-identical 1668-token
stream** (no token-172 divergence). The ws[4] wheel is still the "do not serve" verdict;
this wheel is serve-eligible on the bit-exact axis.

| gate | result |
|---|---|
| disassembly gate (dataflow order, 6 critical f32/4-bit cases) | **PASS** |
| probe 2: bit-exact, both shape configs, M=1..4, 16 checks | **PASS** (maxdiff 0.0 everywhere) |
| probe 3: K=3 MTP A/B lossless (rows @ MIN_N=8192) | **PASS** (1668 tokens byte-identical) |
| battery: A3 vs A3+rows token streams | **PASS** (byte-identical, md5 `a5bb39a0…`) |
| battery: A3 vs A3+rows tok/s | 31.14 (31.17/31.11) vs 30.81 (30.81/30.81) — **−1.05%** |
| A3+rows+fold | **BLOCKED** — `mtp-bench7.py` (the `MLX_MTP_FOLD` bench-side harness) does not exist on the guest or in the repo |

## 2. Root cause of the ws[4] mismatch

The ws[4] kernel staged x in 16-byte chunks (workgroup staging) and computed 2D
`acc[4][32]` context. Under ROCm 10.0 / clang 23 `-O3` that context changes FP contraction
ordering: the inner 32-element dot is still an FMA chain, but the compiler re-schedules the
accumulation relative to the per-row decode, producing 1–2 ULP deltas vs the one-row kernel
whose single-accumulator 32-element chain is the reference.

The restructure (`patches/qmv-rows-bitexact-restructure.diff`) removes the 16B staging:
scalar per-element x-loads, one `la`/`lx` accumulator per row, the identical
shift+mask+cvt decode chain, and the identical q-major/j-minor FMA order — i.e. the rows
kernel is now M copies of the wide kernel's per-row body, with the weight decoded once per
(column, k-tile) and MAC'd across the up-to-4 rows.

## 3. Disassembly gate (why a raw opcode diff is the wrong check)

The ROCm compiler software-pipelines the inner body (hoisted decode, delayed FMAs), so
two kernels with *identical* FP semantics have different raw opcode sequences. The gate
(`bench/qmv_rows_bitexact_disasm.sh` + `bench/qmv_rows_bitexact_cmp.py`) therefore checks
**dataflow-tracked order** instead:

1. FMA chain structure (single-accumulator `v_fma_f32` init + 31× `v_fmac_f32_e32`), with
   dual-instruction lines (`… :: v_dual_fmac_f32 …`) split so a FMA in *either* segment is
   detected (this fixed the 31-vs-32 chain count).
2. Per-chain weight shift-amount sequence via backward dataflow trace.
3. x-operand index sequence from load offsets + register slots: ascending b128 pattern
   `(16·(i//4), i%4)`.
4. Decode multiset (`v_bfe_u32` fused shift+and, `v_cvt_f32_ubyte0_e32`, masks).
5. Per-run identity: rows = 4 m-unroll × wide's per-tile FMA run (132 = 4×33 for 4-bit;
   8-bit k-unrolls ×2: 4×(2×16+1)).
6. Per-chain element-count gate: `chain_r % chain_w == 0`, 1 ≤ ratio ≤ 8.

AMDGPU ELF (machine 224) carved from the guest-compiled `.hip_fatbin`; 18 instantiations of
each kernel parsed. The six critical-path cases (T=float, BITS=4, GROUP_SIZE 32/64/128)
fully PASS: chain 32/32, xasc OK, runid OK, close OK. f16/bf16/8-bit instantiations are
informational (not on the 9B dense path: f32 activations, 4-bit weights).

## 4. Probe 2 — bit-exact, both required shape configs

Wheel: rebuilt on the guest from the patched header (header md5 `7abcacc`, stale `qmm.o`
deleted, fresh `qmm.o` → wheel sha256 `b5a2708…`, `libmlx.so` md5 `e2445f1e…`), installed
into the guest venv via `--target`. `bench/qmv_rows_bitexact_probe2.py`:
N=248320 (default band floor 16384) and N=12288 (floor lowered to 8192), K=4096,
GS=64/BITS=4, bf16 x/scale, M=1..4, seed-42 deterministic inputs, raw-view bit comparison,
two dispatch configs:

| config | rows active on | M=2..4 speedup | bit-exact |
|---|---|---|---|
| default band (MIN_N=16384) | N=248320 only (N=12288 stays wide — control) | 1.26–1.30x (N=248320), ~1.0x (control) | **16/16 maxdiff 0.0** |
| MIN_N=8192 | both shapes | 1.26–1.30x / ~1.0x | **16/16 maxdiff 0.0** |

M=1 stays on the wide path by dispatch in both configs (sanity control). Note the speedup
moved from ws[4]'s 3.14x to ~1.27x at N=248320: the 16B staging that bought the extra
speed is exactly what broke bit-exactness; correctness first.

## 5. Probe 3 + battery — K=3 A/B lossless at the 9B model

`bench/mtp-bench6.py` (`MLX_MTP=4` = the K=3 arm), 9B dense, f32 path, seed-deterministic
prompt. Arm A (rows off) vs arm B (`MLX_ROCM_QMV_ROWS=1`, `MLX_ROCM_QMV_ROWS_MIN_N=8192`
so the N=12288 FFN path exercises the rows kernel):

- A: `TOKS=1668 VERIFIES=512 D1ACC=397 D2ACC=247 D3ACC=135`
- B: `TOKS=1668 VERIFIES=512 D1ACC=397 D2ACC=247 D3ACC=135`
- ALLTOKS: **byte-identical** (md5 `a5bb39a0dc3d639d336f26189d99eeb5`). No token-172
  divergence. The battery's two arms reproduce the same md5 (run-to-run reproducible).

Battery (2 runs per arm, `MLX_MTP=4`):

| arm | run 1 | run 2 | mean |
|---|---|---|---|
| A3 (rows off) | 31.17 | 31.11 | 31.14 tok/s |
| A3+rows (MIN_N=8192) | 30.81 | 30.81 | 30.81 tok/s |

Interpretation: on 9B decode (N ≤ 12288) the rows kernel is **neutral-to-slightly-negative
(−1.05%)** — its win lives in the N ≥ 16384 band (probe 2: 1.26–1.30x at N=248320). The
A3+rows+fold arm is blocked: `mtp-bench7.py` / `MLX_MTP_FOLD` exist neither on the guest
nor in the repo.

**RETIRED-FRAME (11-02):** this battery ran `mtp-bench6.py`, a self-carried script
frame — its 31.14/30.81 tok/s is *not* on the 69.48-tok/s canonical frame, so no
absolute speed claim may be read off it. The relative −1.05% stands as same-session
A/B data (method-consistent within the frame); the canonical-frame re-bench below
supersedes it for any speed verdict.

## 6. Canonical battery (11-02 re-bench, op order: no method variation)

Same-session B0 vs R1 on the **canonical script** `mtp-bench-k3-tape.py` (md5
`2c2d6c90c2349ae087e53e5d6c60018e`, verbatim, arm differences env-only) with the
`gdn_tape_commit.py` helper (md5 `4a262a0d…`), in a fresh guest dir, 2 runs per arm,
flock per arm released after each. Wheel: the bit-exact wheel — `libmlx.so` md5
`e2445f1e…` verified in the guest venv-v2 before running (the `b5a2708…` wheel file
was no longer on the guest; the .so pin is the load-bearing verification and matches).
Sidecar hash-pin: only `model-mtp.safetensors` active. Arm B0 = A3 config
(`MLX_MTP=4 DRAFT=3 MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1 MLX_ROCM_SDPA_S4VA=1`);
R1 = B0 + `MLX_ROCM_QMV_ROWS=1 MLX_ROCM_QMV_ROWS_MIN_N=1`.

**Baseline gate (run before trusting R1): PASS.** B0 reproduces the 69.48 baseline
in-session: 70.20 tok/s (+1.04%), pair 44.74 ms vs 45.20, n_refwd 100 vs 101,
TAPE n_ok=128 n_fb=0, ALLTOKS bit-identical to the t1292/t1307 reference streams
(md5 `f3b09680…`).

| | B0 run 1 | B0 run 2 | R1 run 1 | R1 run 2 |
|---|---|---|---|---|
| TOKS | 402 | 402 | 402 | 402 |
| VERIFIES | 128 | 128 | 128 | 128 |
| TOKS_S | 70.20 | 70.20 | 68.98 | 68.95 |
| D1/D2/D3 PCT | 70.3/62.2/50.0 | 70.3/62.2/50.0 | 70.3/62.2/50.0 | 70.3/62.2/50.0 |
| WALL_MS single | 9.73 | 9.71 | 9.74 | 9.75 |
| WALL_MS verify | 25.60 | 25.63 | 26.43 | 26.46 |
| WALL_MS refwd | 9.17 | 9.16 | 9.14 | 9.11 |
| n_refwd (of 128) | 100 | 100 | 100 | 100 |
| PER_PAIR_MS wall | 44.74 | 44.74 | 45.53 | 45.55 |
| TAPE | 128 ok / 0 fb | 128 ok / 0 fb | 128 ok / 0 fb | 128 ok / 0 fb |

Deltas (R1 vs B0, run 1 / run 2): tok/s **−1.7% / −1.8%**; verify +0.83 ms / +0.83 ms
(+3.2%); pair +0.79 ms / +0.81 ms (+1.8%); single +0.01 / +0.04 ms; refwd −0.03 / −0.05 ms.
Noise floor (run 1 vs run 2, per arm): 0.00% tok/s, 0.01–0.04 ms wall, 0 pp D — the R1
deltas are above noise. Acceptance identical (bit-exact wheel: the tape-committed stream
cannot change; ALLTOKS md5 `f3b09680…` on all four runs).

**Serve decision re-derived from the canonical frame.** Bit-exact rows lands at
**68.95–68.98 tok/s**: −0.3%/−0.8% vs the 69.48 baseline, and at **92% of the sibling
branch's 75.00-tok/s statistical bar** (gap ~6.0 tok/s) — the statistical branch's +7.2%
win does not carry over to the bit-exact restructure: on the canonical frame the bit-exact
rows kernel is −1.8% (same-sign as the retired-frame −1.05%), with the regression entirely
in the verify step (+0.83 ms, +3.2%); single-step and re-forward walls are unchanged. The
kernel-level bit-exactness verdicts (probe 2/3) stand; the rows kernel's speed win remains
restricted to the N ≥ 16384 band (probe 2: 1.26–1.30x at N=248320), and no speed-based
serve argument exists for the 9B-decode shape on this wheel.

## 7. Serve decision

The restructured wheel is **serve-eligible on correctness**: bit-exact at the kernel level
and lossless at the model level. The rows kernel should be gated to the large-N band
(N ≥ 16384, e.g. lm_head / prefill-class shapes) where it wins; on 9B decode shapes it
buys nothing and costs ~1–2% (canonical-frame: −1.8%).

## 8. Artifacts

## 7. Artifacts

- patch: `patches/qmv-rows-bitexact-restructure.diff`
- gate: `bench/qmv_rows_bitexact_disasm.sh` + `bench/qmv_rows_bitexact_cmp.py` (as-run on guest)
- probe: `bench/qmv_rows_bitexact_probe2.py` (as-run on guest; ref/check, per-shape timing)
- wheel: sha256 `b5a2708…` (guest build, patched header md5 `7abcacc`, provenance chain
  header 02:05 → qmm.o 02:10 → wheel 02:11 → install 02:30)
- logs: `/tmp/be2-ref/`, `/tmp/k3A-restructure.log`, `/tmp/k3B-restructure.log`,
  `/tmp/beat-A3-*.log`, `/tmp/beat-AROWS-*.log` (guest)
- canonical battery (11-02): `B0.log`, `B0b.log`, `R1.log`, `R1b.log` in the guest
  `t1325-canon` dir (fresh), env-only arm differences, flock per arm
