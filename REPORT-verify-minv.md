# REPORT-verify-minv: single-basis verify class table + M-invariance code-first

Ticket: verify-class basis fix + M-invariance code-first (continuation of the verify-efficiency workstream). Base: verify-eff-probe @ 27bef8a (REPORT-verify-eff.md accepted, t2199). All work code-first / CPU-only; NO GPU arms were run (no `~/gpu.lock` taken; the on-box toolchain was used for host-side codegen metadata probes only, no GPU use).

## 0. Anchors (as given, not re-derived)

- Verified canon: 69.43 t/s / 31.59 ms / 2.193 tok/step (13-var record, lossless).
- In-session streaming floor: 631.8 GB/s (t2199, var 0.13%).
- Record floor band: 595-624 GB/s (t2048); the record's class fractions were computed at 595 GB/s (verified: 2.531 GiB / 8.447 ms = 321.8 GB/s = 54.08% of 595).
- Verify wall: 21.16 ms (13-var COR record).

## 1. HEADLINE: one-basis class table (per verify step)

One basis for everything: **per verify step = 1x M=3 + 4x M=1 forwards, single weight pass per GEMM, per-matmul bytes = 1.125 x weight bytes** (weight + bf16
scale + bf16 bias, group 64; aux streams 12.5% combined, verify-eff section 1.1).
Bytes per verify step = 5 x per-forward class bytes. ms = sum of the isolated per-GEMM times (1x M=3 + 4x M=1 per GEMM), which is exactly how the in-session record was already summed.

| class | bytes / verify step | in-session ms | in-session GB/s | in-session % of 631.8 | record ms | record GB/s | record % of 595 |
|---|---|---|---|---|---|---|---|
| mlp (gate+up+down) | 12,960.0 MiB (12.655 GiB) | 32.36 | 420.0 | 66.5% | see below | see below | see below |
| qmv (qkv+z+b+a+o) | 5,787.5 MiB (5.652 GiB) | 23.49 | 258.3 | 40.8% | 24.46 | 248.1 | 41.7% |
| all GEMMs | 18,747.5 MiB (18.29 GiB) | 55.85 | 352.0 | 55.7% | 55.01 (est) | 357.4 | 60.1% (est) |

In-session column: re-derived from the per-GEMM table in REPORT-verify-eff.md section 3 (M=1: 5.526 ms mlp / 4.096 ms qmv per forward from the 32-layer sums;
M=3: 10.25 ms / 7.11 ms class totals): 10.25 + 4x5.526 = 32.36 ms, 7.11 +
4x4.096 = 23.49 ms — matches the report's 32.34 / 23.50 within rounding.

Record column: the record (t2048) does NOT give per-verify-step times directly;
the fractions are the record's own numbers, re-placed:
- qmv 41.7%: placed on the per-verify-step basis (verify-eff section 3
cross-check: 41.7% matches the per-verify-step qmv 40.8% within 1 pt).
Time = 5,787.5 MiB / (0.417 x 595 GB/s) = 24.46 ms (derived from the recorded fraction, not an independently recorded time).
- mlp 54.1%: **cannot be placed on the one basis.** It back-computes to the per-M=3-forward class (2.531 GiB / 8.447 ms = 321.8 GB/s = 54.08% of 595), i.e. it uses the 8.447 ms recorded M=3-forward class time. The record has NO
M=1 per-GEMM numbers, so the 4x M=1 part of the per-verify-step basis is missing from the record. Mixed-basis estimate (record M=3 + in-session M=1, labeled, not a record value): 8.447 + 4x5.526 = 30.55 ms = 444.8 GB/s = 74.8%
of 595.

### Re-ranked class order (per verify step)

| rank | class | basis: cost share | basis: bytes share | basis: efficiency (% of floor) |
|---|---|---|---|---|
| 1 (fattest) | mlp | 57.9% in-session / 55.5% record(est) | 12,960 MiB (2.24x qmv) | 66.5% in-session / 74.8% record(est) |
| 2 | qmv | 42.1% / 44.5% | 5,787.5 MiB | 40.8% / 41.7% (the LESS efficient class) |

**"mlp is the fattest class" survives the re-rank, on the cost/bytes axis:**
mlp is 2.24x the bytes and 57.9-55.5% of the per-verify-step GEMM time on both bases; the rank order is identical on the in-session and record bases (no reordering occurred). On the *efficiency* axis the ordering does NOT change either (mlp 66.5/74.8% > qmv 40.8/41.7% on both bases), so "fattest" in the sense of *least efficient* is qmv, not mlp — stated explicitly to keep the two readings apart. The record mlp row is the only non-record value in the table (the mixed-basis estimate above); everything else is record or in-session as labeled.

Per-GEMM per-verify-step detail (in-session, 1x M=3 + 4x M=1 per GEMM, 32
layers, bytes = 5 x per-forward per-GEMM bytes at 1.125x weight):

| GEMM (N x K) | per-step ms | GB/s | % of 631.8 | kernel path (M=3) |
|---|---|---|---|---|
| mlp gate (12288 x 4096) | 11.19 | 404.8 | 64.1% | qmv_wide_mr (single pass) |
| mlp up (12288 x 4096) | 11.19 | 404.8 | 64.1% | qmv_wide_mr (single pass) |
| mlp down (4096 x 12288) | 9.97 | 273.6 | 43.3% | qmv_wide per-m (MR-LDS gate: 72 KiB > 32 KiB) |
| qmv in_proj_qkv (8192 x 4096) | 8.41 | 359.1 | 56.8% | qmv_wide_mr (single pass) |
| qmv in_proj_z (4096 x 4096) | 5.46 | 276.6 | 43.8% | qmv_wide_mr (single pass) |
| qmv out_proj (4096 x 4096) | 5.46 | 276.5 | 43.8% | qmv_wide_mr (single pass) |
| qmv in_proj_b / in_proj_a (32 x 4096) | 2.09 (both) | 5.7 (both) | 0.9% (both) | latency-bound, negligible bytes |

Telemetry caveat (modeled vs measured): the per-verify-step class totals are sums of isolated per-GEMM wall measurements (100-chain amortization, measured wall). The 21.16 ms verify wall anchor (fused 13-var COR run) does not reconcile with the 55.85 ms GEMM sum on this basis; the fused run may overlap GEMMs with non-GEMM work and/or run a wheel path the pinned tree does not contain (the in-session measurement ran with `MLX_ROCM_QMM_M3=1`, a wheel-side knob absent from the pinned tree @ 75ca881 — byte accounting unaffected, it only selects among the single-pass M<=4 variants). The single-basis table is the per-GEMM accounting view; the wall anchor stands as the fused number.
Per-round variance on the underlying in-session arms (recorded in REPORT-verify-eff.md section 3): 13.2% M1-gate, 6.3% M1-down, 6.2% M3-qkv, <1% on all other arms; floor var 0.13%. P-state caveat: idle snapshot sclk 41 MHz, no SCLK sampling tool on the box, boost clock under load unmeasured — all GB/s are measured wall, no modeled clocks were used for any class number (section 3 occupancy arithmetic is separately labeled modeled).

## 2. LDS GATE (code-first, no GPU)

**The 32 KiB budget constant: `vendor/mlx-rocm-upstream/mlx/backend/rocm/quantized/qmm.hip:4129`**
(inside `QuantizedMatmul::eval_gpu`, function at :3567):

```
4128:      size_t mr_lds = (size_t)std::min(4, M) * (size_t)K * 2;
4129:      bool mr_ok = use_mr_qmv && (M <= 4) && (mr_lds <= 32768);
```

M=3, K=12288 (down_proj): `mr_lds` = 3 x 12288 x 2 = 73,728 B = exactly 72 KiB
(plus the 1 KB `s_lx` = 73 KiB block total) > 32,768 → `mr_ok` false → the per-m `qmv_wide_kernel` path (`:4132` fall-through) instead of `qmv_wide_mr`.
The gate was introduced by vendor-tree commit cc73083 (the only commit in the tree history carrying the constant; the visible tree history restarts at the 0.1.2 migration 1cf0e08, so earlier provenance is not in-tree). No occupancy justification is recorded anywhere in the tree for the 32 KiB bound: the kernel header comment (`qmv_wide_mr_kernel.hip:55-59`) says only "The 32 KB gate bounds the x-staging only; the 1 KB s_lx sits on top (33 KB total at the max shape, well within the 64 KB per-block limit)", where the 64 KB per-block default is `kMaxSharedMemoryPerBlock = 65536`
(`device/config.h:66`, `device.h:452`). The runtime-queried per-block limit exists and is available but UNUSED by the gate: `hw.shared_mem_per_cu =
props.sharedMemPerBlock` (`qmm.hip:700`, HWInfo fields `device/config.h:109-110`, `hw_info` already in scope at `:3980`; the Rdna4 tier detection for gfx1201 is at `:715`).

**Answer: the 32 KiB budget is inherited from the RDNA3-era 32 KiB LDS/CU, and there is no occupancy constraint that justifies it on gfx1201 (128 KiB LDS/CU).**
Reasoning (spec-labeled, code-first):

1. The bound equals the RDNA3 per-CU LDS capacity (32 KiB, public vendor spec)
exactly — a block larger than the whole per-CU LDS cannot run on RDNA3.
It is a plain constant; there is no arch branch around it.
2. On gfx1201 (RDNA4, 128 KiB LDS per CU — ticket-given and public spec) the occupancy is wave-capped, not LDS-capped: RDNA4 max 4 wavefronts per CU
(public vendor spec; the tree's HWInfo sets `simds_per_cu = 2` for the Rdna4 tier, `qmm.hip:716`). The MR block is 512 threads = 16 wavefronts (`qmv_wide_mr_kernel.hip` block `(32, 16)`), so at most ONE block is resident per CU whether the block takes 25.5 KiB (K=4096) or 73 KiB
(K=12288) of LDS, and the 4/16 resident waves are the same in both cases.
The 32 KiB gate therefore changes nothing about occupancy on gfx1201 — it protects the RDNA3 budget (and the 64 KB per-block default), not gfx1201
occupancy.
3. Toolchain-verified host-side (measured, no GPU, on-box ROCm 7.14 / LLVM 23 —
deviation: the pinned tree builds under ROCm 10.1; these metadata limits are stable across that range): max VGPRs per lane on gfx1201 = 192
(llc inline-asm binary search: 192 registers accepted, 193 rejected with "inline assembly requires more registers than available"). The LDS limit is NOT enforced at llc level (a 1 MiB local segment compiles fine); it is launch-time, via the `sharedMemPerBlock` prop, whose gfx1201 value was not queried here (runtime prop = GPU; deferred, one-line follow-up). So a 73 KiB
block is launchable iff the driver reports `sharedMemPerBlock >= 73,728`;
the tree already queries that prop (`:700`), it is just not wired into the gate.

**The exact change that puts K=12288 down_proj on the single-pass path**
(one line, `qmm.hip:4129`):

```
bool mr_ok = use_mr_qmv && (M <= 4) &&
(mr_lds <= ((hw_info.shared_mem_per_cu > 0)
? (size_t)hw_info.shared_mem_per_cu : 32768u));
```

Byte/occupancy consequence: block LDS 73 KiB (72 KiB x-staging + 1 KB s_lx)
instead of 25.5 KiB at K=4096; still exactly 1 block per CU (73 <= 128 KiB), 4/16 waves resident — occupancy IDENTICAL to the K=4096 MR case, no loss;
weight HBM passes 3 -> 1 (the per-m path's 2 extra passes are L2-mitigated, which is why it already holds 51.0%); LDS+L1 share the 128 KiB per-CU SRAM on RDNA4 (public spec), so L1 data shrinks ~128 -> ~55 KiB while the 73 KiB block is resident — secondary, since the weight stream is fed from the 32 MB L2;
the 64 KB per-block default (`config.h:66`) must yield to the queried prop (`device.cpp:418`). Expected gain is MODEST (capped, modeled): the K=12288 MR
kernel does the same per-code ops as the K=4096 MR kernel that measures 38.6-51%, so the ceiling is ~51-55% vs the current 51.0% — a few points at most. The change also unblocks M=4 K=12288 (96 + 1 = 97 KiB <= 128 KiB) and M=3 8-bit shapes at the same budget.

## 3. DEQUANT-ALU OCCUPANCY: M=3 K=4096 (measured 38.6% of floor, in-session)

Shape: gate/up 12288 x 4096, `qmv_wide_mr_kernel`, measured 116.1 us / 243.8
GB/s / 38.6% of the 631.8 floor (in-session, recorded; per-round var for this arm < 1%).

- **Tile shape** (`qmv_wide_mr_kernel.hip`): block `(32, 16)` = 512 threads =
16 wavefronts; grid `(N/16, M/4)` = (768, 1) for N=12288, M=3; per lane per k-tile: one `uint4` (16 B = 32 four-bit codes) global weight load, 32-code dequant (shift+mask+cvt, hoisted OUT of the row loop by the run-825 weight reuse and reused across all M rows), then the per-row m-loop (32 FMA per row for 3 rows = 96 FMA + 3 FMA into `acc`) with the run-852 lx hoist (32 FADD
per (row, k-tile) + one `__syncthreads` per k-iteration, 4 k-iterations at K=4096); final per-row 16-step `__shfl_down` reduction (`:260`).
- **LDS usage**: `min(4,M) x K x 2` = 24,576 B = 24.5 KiB x-staging + 1 KB
double-buffered `s_lx` = 25.5 KiB per block — NOT binding on gfx1201
(128 KiB per CU; 1 block/CU either way because the 16-wave block exceeds the 4-wave cap).
- **VGPR usage**: code-level ESTIMATE, not measured (deviation recorded: the on-box clang-23 hip frontend segfaults in host compilation on all 3 attempts — the default toolchain cannot produce GPU asm on this box; the 7.2 install has no compiler; no GPU used to work around it, per ticket). From the unrolled source: `wn[32]` + `acc[4]` + `la`/`lx` + `s`/`b` + `k`/`g`/`p` +
~5 pointers + shfl chain => roughly 55-70 live VGPRs, far under the 192
ceiling — register pressure is not the limiter (estimate, labeled).
- **Occupancy, gfx1201**: achievable = 4 wavefronts/CU (public vendor spec, 2 SIMD x 32 lanes); the block supplies 16 waves, so 4/16 of a block is resident = 1 block/CU, 64 CUs. Achieved occupancy: NOT measurable without a GPU profiler (deferred); inferred (modeled) from the measured GB/s plateau.
In-flight bytes: 1 outstanding 16 B load per lane => 4 waves x 32 lanes x 16 B
= 2 KB/CU, vs the 9.87 GB/s/CU the 631.8 floor demands at 64 CUs.
- **Per-code op budget (modeled, from the source unrolls)**: M=1 `qmv_wide`
does ~3 ALU (dequant) + 1 FMA per code; M=3 MR does ~3 ALU + 3 FMA + ~1 FADD
per code — 2.4x the per-code ops, matching the 76.8 -> 38.6 collapse ratio (2.0x) in magnitude. The per-row m-loop is three 32-deep dependent FMA chains (independent across rows, serial within a row). Note for honesty: the M=1
kernel has the same 2 KB/CU in-flight count and reaches 76.8%, so in-flight load count alone does not explain the collapse; the per-code op growth (shared dequant ALU + 3x FMA + the dependent chains) is the modeled primary suspect. The precise stall attribution (ALU issue vs load latency vs barrier)
needs a GPU profiler — out of scope for this code-first ticket.

**Ranked levers to lift M=3 efficiency (no implementation in this ticket):**

1. **Software-pipelined / prefetched weight loads in `qmv_wide_mr`** (2-4
stages of the 16 B `uint4` load): cost +4-12 VGPR (fits the 192 ceiling), zero change to the FMA order => bit-identical-safe (same tile order, same per-row FMA order; only load timing moves). Expected ceiling: ~60-75%, capped by the M=1 plateau of 76.8-80.2% (modeled ceiling; the M=1 numbers are the measured cap).
2. **K=12288 MR single pass (the section-2 gate change)**: expected ceiling ~51-55% on the down shape (current 51.0% is already L2-mitigated) — capped and modest; also unblocks M=4 K=12288.
3. **Dequant ALU reduction**: already hoisted (run-825 weight reuse: 1 dequant per k-tile reused across all rows); the remaining ~3 ALU/code is the group-64
scale-density floor — the 12.5% aux-stream overhead is second-order and the layout lever was closed in verify-eff (1.125x, not 3x).
4. **WMMA / matrix-pipe path for M=2..4** (RDNA4 native WMMA; `kWmmaArches`
includes gfx1201, `qmm.hip:729-738`): the HIGHEST ceiling (matrix pipes beat scalar FMA), but it changes the FMA order (ULP drift; the bit-identical property is lost and the lossless token gate becomes the practical gate) and needs a bf16 LDS staging redesign — a project, not a lever.
5. **K-split / more blocks per GEMM**: MEASURED falsified precedent on the C
class (t2050: -4.05%, 4.596 vs 4.418 ms, both bars miss; t2048 wash confirmed K in 4.4-4.7); also changes FMA order. Do not pursue.
6. **Batching the 4 M=1 forwards into one M=4**: M=4 K=4096 MR fits the gate (32 KiB), but M=4 MR efficiency would be ~M=3 (35-39%, modeled) << 4x M=1
(76-80%, measured) — counterproductive. The 1x M=3 + 4x M=1 split is the right shape; do not merge it.
- **"Raise occupancy" is NOT an available lever**: the 4-wave/CU cap is hardware; LDS is not binding at either 25.5 or 73 KiB.

**Expected ceilings, class level (upper-bound arithmetic, modeled; the M=3
kernels would have to reach the measured M=1 plateau, which has never been observed at M=3):** M=3 mlp 10.25 ms @ 265.3 GB/s -> ~5.6 ms at 485 GB/s (-4.6 ms); M=3 qmv 7.11 ms @ 170.5 GB/s -> ~3.6 ms (-3.5 ms). Per verify step:
55.85 ms -> ~27.4 ms, i.e. roughly half the GEMM portion — the M=3 collapse is the dominant cost on this basis (see the honest notes below).

## 4. Two honest notes (record kept straight)

1. The M=3 collapse (76.8 -> 38.6% gate, 66.6 -> 35.8% qkv, 51.0 -> 28.0%
z/o, in-session measured) is a MEASURED finding that supersedes the stream-layout theory: the 1.125x byte accounting closed the layout as a cause, and the measured per-M collapse is where the cost actually sits.
2. The earlier ULP / M-invariance suspicions from the attention side (the planes/t2177 line) point the same direction: **M-invariance of the kernels (M=1 vs M>=2 behavior) is the systemic issue on this backend, not one bug.** The QMV class collapse here and the attention-side ULP drift are two faces of the same property: the kernels are not M-invariant.

## 5. Deviations and stop-lines observed

- No GPU arms run; `~/gpu.lock` not taken; no runtime device-prop queries (sharedMemPerBlock for gfx1201 deferred as a one-line follow-up, section 2).
- Box toolchain: default ROCm 7.14 / LLVM 23 (not the pinned 10.1); used for host-side codegen metadata only (VGPR ceiling probe). The on-box clang-23
hip frontend segfaults in host compilation (3 attempts, crash backtrace class, not retried past the stop line); the 7.2 install has no compiler.
Consequence: the real kernel's VGPR count is not obtainable on this box without a working toolchain or GPU — section 3 VGPR is a code-level estimate, labeled.

## 6. Check lines

- ruff format: see commit-time check (section 7).
- persona/secret leak scan (report + diff): run at commit time (section 7).
- line wrap: FLEET detector over this file (section 7).
- size gate: no code added to any >2,000-line file (this ticket is a report only; the vendor tree is untouched).
- GPU gate: N/A — no GPU arms by design.

## 7. Check-line results

- ruff format check: `89 files already formatted` (pass); ruff check: `All checks passed!` (this ticket adds a report only, no code changes).
- persona/secret leak scan over the report (persona prefixes, the fleet /10 tailnet address range, operator handles, fleet hostnames, fleet identity in tree content): CLEAN (zero hits).
- line wrap: FLEET detector (`line_wrap_check.py`) over this file: exit 0 (clean) after the one-paragraph-per-line reflow; the pre-reflow draft failed (22 hard-wrapped pairs) and was reflowed, not shipped wrapped.
- size gate: N/A — no code added to any file (report only; the vendor tree is untouched).
- GPU gate: N/A — no GPU arms by design (section 5).
