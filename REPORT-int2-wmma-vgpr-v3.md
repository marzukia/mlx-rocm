# REPORT: int2 WMMA VGPR v3 — 570.8 us/row (0.60× the 4-bit anchor) — GATE PASS

**Per-row (3 runs, same session, guest-a, gfx1201, RX 9070 XT, ROCm 10.1, GPU clock locked high):**

| loop | v2 (192 VGPR, 1 block) | v3c (74 VGPR, 2 blocks) | 4-bit anchor |
|---|---|---|---|
| `wide_loop` (int2 wide) | 1365.4 us | **570.8 / 570.4 / 571.8 us** | 953 us |
| `int2_loop` (scalar) | 19248 us | 18922–19015 us | — |
| `head_loop` | 953 us | 950–954 us | 953 us |

Gate: per-row ≤ 0.953 ms (4-bit anchor). **570.8 us = 0.60× the anchor. PASS.** v3c is 2.39× faster than v2 (1365.4 → 570.8) and 2.01× faster than the first 2-block v3 build (v3b, 1149.7 us).

**Verdict: the int2 draft head runs 2 blocks/CU (64 waves) with bit-identical logits and a register-resident FMA at 0.60× the 4-bit anchor. qmlx-serve's 9B-class dense Qwen3 head now has a 2.4× faster int2 draft path than the v2 register-resident kernel.**

## The v3c kernel (what shipped)

`lm_head_int2_wide` (block (32, 32) = 1024 threads, 16 KB x staging + selection shared, 2 blocks/CU = the full 2048-thread CU budget):

- **x**: staged once per block (f32) in a bank-spread layout `k2 = p*1024 + 32j + 4gl + l3` for `k = p*1024 + 128gl + 32l3 + j`; streamed from LDS in 4-j chunks (8 c-chunks × 4 planes × 4 scalar loads = 128 LDS reads per column, peak live x window = 16 floats). A 32-lane load at fixed (c, b) reads words `4gl + l3` — 32 distinct banks, one LDS cycle (the v2-style 32-word stride is an 8-way conflict; the first permutation, k2 = p*1024 + 128gl + 4j + l3, was still 8-way: all 8 group bases 128gl map to the same 4 banks).
- **weights**: 32-byte window preloaded once per column (2 × 16B, non-volatile, L1-hot), 7-ternary select per 4-j chunk. Exactly 32 bytes per column — the v3i per-chunk u32 load made the compiler re-issue the whole window twice per chunk as 2 × b128 (8× redundant traffic, 9664 us/row); a per-chunk volatile 4B load (v3b) measured 1149.7 us — the volatile flat load (SCOPE_SYS path) is 2× slower than the preload.
- **FMA order**: per column, c 0..7 × b 0..3 × p 0..3 = 128 FMA; per accumulator d[p], j = 4c + b ascending 0..31 — byte-identical to the v2 128-value preload (it 0..3 × b 0..7 × q 0..3). Plane p of byte j pairs with `x[p*1024 + j]` exactly as v2; the 4-plane reduction + scale run once per column after all 128 FMA. Bit-identity by construction; **16-row probe max_abs_diff = 0.000000e+00** (re-verified on v3b and v3c).
- **Registers**: 74 VGPR, 22 SGPR, 0 AGPR, 0 scratch (occ probe, .s metadata). The v2 preload (192 VGPR) is gone; peak live x window is 16 floats.

## The occupancy inversion (and why it didn't matter)

The simple "VGPR ≤ 96 AND SGPR ≤ 24/wave" model is **wrong** on gfx1201. Empirical caps (512-thread blocks, LDS = 0, 5/5 deterministic):

| variant | VGPR | SGPR | max waves | blocks @1024 |
|---|---|---|---|---|
| 4-bit anchor | 51 | 24 | 64 | 2 |
| v2 | 192 | 24 | 32 | 1 |
| v3i | 89 | 26 | **64** | **2** |
| v3t | 86 | 24 | **48** | **1** |
| v3x | 97 | 26 | 48 | 1 |
| v3b | 62 | 22 | 64 | 2 |
| v3c | 74 | 22 | 64 | 2 |

v3t uses strictly fewer of every observable static resource than v3i (86 < 89 VGPR, 24 < 26 SGPR, kernarg 320 < 336 B, identical instruction multiset, 0 scratch both) yet caps at 48 waves. No single-resource cap (VGPR/SGPR, with or without 8/16/32 rounding, per-CU or per-SIMD) is consistent with both rows; adding the 4 args back to v3t (v3u, 28 SGPR) keeps it at 48. The 24-SGPR budget is not binding (v3i exceeds it and fits). The constraint the driver computes is not exposed in the .s metadata.

The practical resolution: **occupancy on this driver is structure-sensitive, not a function of the metadata counts** — so every candidate was verified with the occ probe (or 512-thread sweep for the wave cap) before a wheel build. v3c (74 VGPR / 22 SGPR) fits 64 waves cleanly, and the 570.8 us/row measurement (0.60× the anchor, 2.39× v2) confirms the 2-block occupancy in the real wheel, not just the probe.

## Path that got here (abridged; full log in NOTES-int2vgpr-v3.md)

- v3 (5370be7) 16-j × 2-sub float4 streaming: 118 VGPR → 1 block; the per-chunk u32 weight load (v3i) hit 2 blocks but at 9664 us/row (8× redundant weight traffic).
- v3j–v3o: preload + select-chain variants: 26–30 SGPR (scalar ALU branch scratch / fresh s-pair for the candidate stores), all 1 block. `__noinline__` is a no-op in ROCm; `#pragma unroll` is ignored on the runtime-r output loop; volatile 64-bit locals make SGPR worse.
- v3x: volatile 4B weight (kills the b128 vectorization): 97 VGPR → 48 waves. v3y: + first x permutation: 62 VGPR / 22 SGPR, 2 blocks, but 2675.9 us/row — the permutation still conflicted (8-way).
- v3b: bank-spread layout: 1149.7 us/row (1.21× anchor) — GATE FAIL; bit-identity re-confirmed.
- v3c: volatile → 32B preload + select: **570.8 us/row — GATE PASS.**

## sel_match = 0/256 is a structural artifact, not a bug

The xcheck compares the scalar int2 op's candidate pool (top-8 per 64-column subblock: **16 candidates per 128 columns**) against the wide op's pool (top-8 per 32-column tile: **32 candidates per 128 columns**). The top-32 sets differ by construction (a 16-element set can never equal a 32-element set), so sel_match = 0/256 on every variant including the bit-identical ones. The load-bearing correctness check is the 16-row logit bit-identity probe: **max_abs_diff = 0.0** on v3c.

## Reproduction

- wheel: `~/vendor-int2wbuild` graft of `lm_head_int2w.hip` (this commit), `pip wheel . --no-build-isolation --no-deps` with `HCC_AMDGPU_TARGET=gfx1201 LD_PRELOAD=libhsa-runtime64`, install into `~/venv-int2w` (zipfile extract; mlx_lm 0.32.2 re-extracted from `~/wheels/`).
- xcheck: `~/bench-int2w/r4c_xcheck.py` under `flock ~/gpu.lock`, `MTP_MODEL=<models-mount>/qwen3-9b-mlx` (N = 248320, K = 4096).
- probe: `~/bench-int2w/int2w_probe.py` (16 random rows, per-column ulp-level vs the scalar op).
