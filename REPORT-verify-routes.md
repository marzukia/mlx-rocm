# REPORT — QMV route env-sweep (no code change), gfx1201

**Question answered:** Is the M-invariance collapse (small-M GEMM at ~27–42% of the streaming floor) a **THRESHOLD (policy)** or **KERNEL (implementation)** issue?

**Verdict:** Primarily **KERNEL (implementation / occupancy)**, with a real secondary **THRESHOLD (policy)** component. Detail in §7.

- Wheel: `mlx 0.32.3.dev20261104+75ca881` (libmlx md5 `8acf13fe`, sidecar md5 `02a54359` — verified, per ticket)
- GPU: RX 9070 XT, gfx1201, 32 CUs, warp 32, native_wmma=1 (bound device 0)
- Model: `/mnt/models/qwen3-9b-mlx` (vocab 248320 → lm_head 248320×4096)
- In-session streaming floor (both ends): **631.5 / 631.7 GB/s** (var 0.13–0.14%), consistent with the t2199 anchor of 631.8.

---

## 1. What was swept

7 GEMM shapes (N×K, 4-bit affine, group 64, `quantized_matmul(..., transpose=True)`): gate/up 12288×4096, down 4096×12288, qkv 8192×4096, z 4096×4096, o 4096×4096, lmh 248320×4096. M ∈ {3, 4}. 3 interleaved rounds, round-robin over shape×M×arm.

**Arms.** The ticket named 5 arms: default, LX, ROWS, ROWS+HOIST, NO_MR. A byte-grep of the wheel shows which env knobs are actually present:

| arm | knob | in wheel `8acf13fe`? |
|---|---|---|
| default | — | yes (baseline) |
| LX | `MLX_QMV_LX` | **yes** |
| NO_MR | `MLX_QMV_NO_MR` | **yes** |
| ROWS | `MLX_ROCM_QMV_ROWS` | **no** |
| ROWS+HOIST | `MLX_ROCM_QMV_ROWS`+`MLX_ROCM_QMV_HOIST` | **no** |

So only **default / LX / NO_MR** could be measured on this wheel; ROWS and ROWS+HOIST are **not in the wheel** (reported as such, not silently dropped). This wheel is older than the pinned source `@ 75ca881`; the new-generation knobs (`MLX_ROCM_QMV_ROWS`, `_MIN_N`, `_HOIST`, `MLX_ROCM_QMM_M3`, `MLX_ROCM_QMV_MR2_MIN_N`) are all **absent** (QMM_M3: `None`). Per the prior-probe note, the new kernel generations (mlp-gen2 t2031, qmv gen2 t2040) are closed, but their env knobs are not in this wheel — so this sweep is **env-route selection on the existing kernels only**, a different axis from the closed generation work.

**Why one process per (shape,M,arm).** All QMV dispatch knobs are function-local `static`, read once on first dispatch per process. An `os.environ` change mid-process has no effect, so each (shape,M,arm,round) is a fresh subprocess with the arm set in its inherited environment.

**Method / P-state control.** Master loads the model once, extracts the 7 packed tensors (weight `uint32 [N, K/8]`, scales, biases), saves them per-file (`mx.save` → `.npy`, confirmed to round-trip both value *and* dtype: bf16→bf16, uint32→uint32, ndiff=0), reloads, and verifies the round-trip op is bit-exact (ndiff=0). After a 25 s idle (GPU settles to deep idle: sclk 78 MHz / mclk 96 MHz), the in-session floor is taken, then the matrix.

Each time worker does three things on the same tensors: (1) **burst** — the reference 100-chain (one ~12 ms eval), after a 1 s prime; (2) **ext** — one continuous eval of ~0.7 s (no gaps), `N_ext` sized so the eval ≈ 0.7 s; (3) a per-op P-state snapshot. The burst-vs-ext pair is the **discriminator for the clock-priming hypothesis** (see §4).

---

## 2. P-state is not the confounder (the key control)

The pre-matrix snapshot shows the GPU at **deep idle** (sclk 78 MHz / mclk 96 MHz) after the 25 s idle. Per-op snapshots taken after a burst show mclk at level 0 (96 MHz) — but that is the **idle-gap reading**: a 200+ GB/s op cannot run at 96 MHz mclk (that would be a few GB/s), so the memory clock ramps *during* the burst and the snapshot catches the gap after it drops back.

The decisive control: **ext (0.7 s continuous) ≈ burst (12 ms) for every cell.**

```
ext/burst ratio (default arm, median of 3 rounds):
  gate 0.985 / 0.991   up 0.987 / 0.990   down 0.996 / 0.997
  qkv  0.980 / 0.987   z 0.978 / 0.980    o 0.978 / 0.983   lmh 0.993 / 0.993
```

A sustained load does **not** raise the small-M GB/s. So the small-M throughput is **not** an under-measured clock-priming artifact — it is a real, stable limit. (Clock *locking* was not an option: `rocm-smi --setclock/--setperflevel` require sudo, which needs an interactive terminal unavailable in this non-interactive ssh; the sysfs `pp_dpm_*` files are "busy".)

---

## 3. Results — median GB/s over 3 rounds (weight bytes `N*K*9/16`, per-op wall)

`%floor` is against 631.7 GB/s. Best achievable shown in the last column.

```
shape  M  default  lx      nomr    def%flr  best(arm)%flr
gate   3  249.4   205.8   307.1   39.5%     nomr 48.6%
gate   4  190.6   158.1   233.9   30.2%     nomr 37.0%
up     3  249.0   205.3   307.2   39.4%     nomr 48.6%
up     4  190.6   158.2   233.1   30.2%     nomr 36.9%
down   3  313.0   236.6   313.1   49.5%     nomr 49.6%
down   4  240.4   180.5   239.9   38.1%     nomr 38.0%
qkv    3  239.8   196.7   293.4   38.0%     nomr 46.4%
qkv    4  182.5   151.8   227.8   28.9%     nomr 36.1%
z      3  211.7   176.8   264.9   33.5%     nomr 42.0%
z      4  167.1   139.3   212.1   26.5%     nomr 33.6%
o      3  211.3   176.6   265.2   33.4%     nomr 41.9%
o      4  167.0   139.3   213.3   26.4%     nomr 33.8%
lmh    3  268.7   209.0   210.1   42.5%     default 42.5%
lmh    4  200.1   157.3   157.9   31.7%     default 31.7%
```

Range vs the reference "27–42% of floor": measured M=3 spans **39.4–49.5%**, M=4 spans **26.4–38.1%** — bracketing the reference figure (which sits in the M=4 / lower-M=3 band).

---

## 4. M-invariance: violated, and it is real

Default arm, median over 3 rounds:

```
shape  M3      M4      M4/M3
gate   249.4   190.6   0.764
up     249.0   190.6   0.765
down   313.0   240.4   0.768
qkv    239.8   182.5   0.761
z      211.7   167.1   0.789
o      211.3   167.0   0.790
lmh    268.7   200.1   0.745
```

M4/M3 clusters tightly at **≈0.75 = 3/4**. Round-to-round spread per cell is ~0.5–1%, so this is systematic, not noise. **M=3 is ~26% faster than M=4** — the op is **not** M-invariant.

To rule out a cross-subprocess P-state ordering artifact, I ran a clean same-process test (M-independent sustained prime, then M=3/M=4 interleaved back-to-back, 3× each):

```
gate:  M3 249.6 / 248.9 / 250.0   M4 190.0 / 190.0 / 190.4   (M4/M3 0.761)
down:  M3 307.0 / 306.6 / 306.9   M4 236.0 / 237.6 / 237.4   (M4/M3 0.769)
lmh:   M3 266.5 / 266.4 / 266.4   M4 199.2 / 199.3 / 199.3   (M4/M3 0.747)
```

Rock-stable. The M-dependence is real.

**Interpretation.** For a weight-bound GEMM the weight read `N*K*1.125` is M-independent, so M=3 and M=4 should be nearly equal. They are not: per-op time scales ~linearly with M (M4 = 4/3 × M3's time). The base M=3 time (e.g. gate 113 µs) is close to the pure weight-read time (~89 µs at 631 GB/s), but the *marginal* row adds ~36 µs at only ~2.8 TFLOP/s — i.e. the few rows cannot fill the CUs. **Small-M (3–4) GEMMs run occupancy-limited, not in the ideal weight-streaming regime.** That is the mechanism behind the collapse.

---

## 5. Arm (dispatch) effect — the policy signal

At M=3, median over rounds:

```
shape  default  lx      nomr
gate   249.4   205.8   307.1
up     249.0   205.3   307.2
down   313.0   236.6   313.1
qkv    239.8   196.7   293.4
z      211.7   176.8   264.9
o      211.3   176.6   265.2
lmh    268.7   209.0   210.1
```

- **NO_MR is 22–25% faster than default on the medium-N shapes** (gate/up/qkv/z/o), but **slower on the huge-N lmh** (210 vs 269, −22%). Disabling the MR code path helps medium-N small-M GEMMs and hurts the very-large-N GEMM — a dispatch-threshold signature (MR is good for large N, bad for medium N).
- **LX is 15–20% slower than default** on every shape.
- `down` (N-major, K=12288) is the one shape where NO_MR ≈ default (313.1 vs 313.0).

Because a dispatch env toggle moves the number by 22–25%, the default dispatch is selecting a suboptimal path for these shapes — that is the **policy (threshold)** component.

---

## 6. Bit-exact (rows-exact) M-invariance check

Across all 3 arms × 7 shapes × M{3,4} (42 rows), the batched output row `M` is compared to the single-row output for each row:

```
42 / 42  identical=YES  ndiff=0
```

**Bit-identical on every cell, every arm.** The collapse is a **pure throughput/timing** effect — M=3 vs M=4 (and the arm variants) produce *exactly* the same numerics. It rules out "M=3 uses a numerically-different, slower kernel by design"; the math is identical, only the timing differs.

---

## 7. Verdict — THRESHOLD (policy) vs KERNEL (implementation)

Decomposing the ~60% floor-gap at default M=3 (e.g. gate 249.4 → 39.5% of floor, gap 60.5%):

- **Policy (threshold) component — secondary, ≈15% of the gap.** NO_MR (a dispatch path toggle) closes ~9 floor-points on the medium-N shapes (gate 249→307 GB/s). The default dispatch applies the MR path to medium-N small-M GEMMs where it hurts, and reverses on huge-N (lmh). A dispatch fix (reserve MR for large N) is worth ~15% of the gap on those shapes — a concrete, low-cost win.
- **Kernel (occupancy) component — primary, ≈85% of the gap.** Even the *best* dispatch (NO_MR) reaches only **~42–50% of the streaming floor** for small-M GEMMs. That residual gap is the occupancy limit: 3–4 rows cannot fill the 32 CUs, per-row compute runs at ~2.8 TFLOP/s. Closing it requires a **better small-M kernel** (more work per CU / higher-occupancy tiling), not a dispatch change. The M=3→M=4 threshold (M4 ~26% slower) is part of this same occupancy/tiling picture.

**Bottom line: the M-invariance collapse is dominated by a KERNEL (implementation / occupancy) limit** — small-M (3–4) quantized GEMMs cannot saturate the CUs, so even the best available dispatch only reaches ~42–50% of the streaming floor. There is a real, secondary **THRESHOLD (policy)** component: the default dispatch's MR path is suboptimal for medium-N small-M GEMMs (NO_MR recovers 22–25%), and the M≥4 threshold selects a lower-occupancy config than M=3.

The new-generation kernel work (mlp-gen2 t2031, qmv gen2 t2040) is closed; the knobs that would implement a better small-M route (`MLX_ROCM_QMV_ROWS`, `_HOIST`, `_MIN_N`, `MLX_ROCM_QMM_M3`, `_MR2_MIN_N`) are **not in this wheel** (`8acf13fe`), so they could not be exercised here. Arms ROWS and ROWS+HOIST are therefore reported as **not in the wheel**.

---

## 8. Caveats

- Wheel `8acf13fe` is older than pinned source `@ 75ca881`; only default/LX/NO_MR are present.
- Clock locking unavailable (no sudo / interactive terminal); P-state controlled via sustained prime + the burst-vs-ext control (§2), which shows the small-M GB/s is not a clock artifact.
- `MLX_ROCM_QMV_ROWS`/`_HOIST`/`_MIN_N`/`_MR2_MIN_N`/`MLX_ROCM_QMM_M3` absent → arms (c) and (d) not measurable on this wheel.
- GPU has 2 devices; the bound compute device is 0 (gfx1201).
