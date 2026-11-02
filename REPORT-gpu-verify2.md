# GPU verify2 — report

Marzuki-neon (100.78.68.72), RX 9070 XT / gfx1201, ROCm 10.1 guest userland (Ubuntu 26.10).
All GPU runs under `flock /home/andryo/gpu-bench.lock`, back-to-back A/B in the same session.

| Item | State |
|---|---|
| `model-mtp.safetensors` sha256 | `edef9ec6f48a31f40659ec7937bae93b4a1c501c4cd23959933c992df31d1df8` — **matches pin** |
| Sidecars | `model-mtp.safetensors` active; `model-mtp-4bit.safetensors.disabled`, `model-mtp-fixed.safetensors.disabled` — **as expected** |
| GPU lane at handoff | 9 W, 0 % util, 45 °C — clean |
| venvs | `venv-verify` = V2 wheel (baseline, patched qwen3_5); `venv-attn` = v3 wheel (S=4 SDPA graft, patched qwen3_5) |
| v3 wheel legitimacy | `nm` on `libmlx.so`: `kernel_sdpav_1pass<T,D>` symbols with **no causal bool template tag** (single instantiation) — the graft's 1-pass kernel, not the old 2-variant split |

---

## Job 1 — GDN tape commit: falls back every round; root-caused and fix-verified

Protocol: K=3 MTP bench (128 verify, 400-class token budget), same prompt/model as the t1265
run, `MLX_ROCM_GDN_FUSE=1` required (the bench's `set_gdn_fuse()` alone does not enable the
F-branch; both env vars are checked at import). venv-verify (V2 wheel, which carries the tape
op).

| arm | config | toks | tok/s | acc d1/d2/d3 | n_refwd | verify ms/pair | refwd ms/pair |
|---|---|---|---|---|---|---|---|
| B1 (B0 control) | TAPE=0 | 400 | 45.56 | 87/57/32 | 272 | 33.89 | 24.83 |
| t1265 `k3-v2-b0` (reference) | TAPE=0 | 400 | 45.77 | 87/57/32 | 272 | 33.61 | 24.78 |
| T1b (tape, stock guard) | TAPE=1 | 395 | 38.15 | 85/54/27 | **395** | 34.48 | 36.53 |
| T1f (tape, fixed guard) | TAPE=1 | 422 | **61.10** | 100/66/30 | **98** | 34.94 | **9.03** |

B1 reproduces t1265 within 0.5 % on every column (first20 identical) — cross-validation of the
environment.

### Finding 1 — the commit falls back on every verify round (t1265 and T1b)

`TAPE n_ok=0 n_fb=128`, `TAPE_CALLS=12552` (the op is called on every S≤4 forward — 128 verify
+ 128 correction + 384 … × 24 layers), so the tape machinery engages; the **commit guard** then
fails all 128 rounds. Failing check, first one in the all-or-nothing sweep:

```
FB_FAIL[128]: layer 0 state_tape (4, 32, 128, 128)/mlx.core.float32
             != expected (4, 1) + (128, 128) f32
```

The tape op itself is correct: it returns per-row tapes `state_tape (T, 32, 128, 128) f32` and
`conv_state_tape (T, 3, 8192) bf16` (probe-confirmed, T=1 and T=4). The guard has an axis
confusion in three places — the live SSM state is **batch-led `[1, 32, 128, 128]`** but the
guard reads it as if it were `[Hv, 128, 128]`:

1. `hv = c[1].shape[0]` → reads the batch dim (1), not the head count (32) → expected shape
   `(4, 1, 128, 128)` ≠ actual `(4, 32, 128, 128)` — the check that fires;
2. `c[1].shape[1:] != (128, 128)` → `(32, 128, 128)` ≠ `(128, 128)` — fails next (confirmed by
   re-run after fixing #1: `live SSM shape (1, 32, 128, 128) tail != (128, 128)`);
3. rebind `c[1] = st[row]` drops the batch axis (`(32,128,128)` into a `(1,32,128,128)` slot).

Note: with the guard broken, the fallback arm is **worse than B0** (n_refwd 395 > 272): the
guard runs before the `keep==4` shortcut, so it re-forwards even on full accepts.

### Finding 2 — with the guard fixed, the commit engages and does its job

| | T1f (fixed) | vs B1 |
|---|---|---|
| n_ok / n_fb | 128 / 0 | |
| n_refwd | 98 = 128 − 30 (d3acc) | **−64 %** |
| refwd ms/pair | 9.03 | −64 % |
| tok/s | 61.10 | **+34 %** |
| wall ms/pair | 53.8 | −21 % |
| verify ms/pair | 34.94 | +1.05 (tape-recording cost, ~2.4 %) |

n_refwd = 98 exactly matches the number of partial-accept rounds (one correction forward each);
full-accept rounds do zero re-forwards. The re-run reproduces T1f (61.10 vs 61.22 tok/s).

### Finding 3 — the tape arm is not a bit-exact drop-in (conv tape rows 0/1)

Row-by-row probe (prefill 16, then one S=4 tape step vs four S=1 fused steps, layer-0 states):

| row | SSM md5 (tape vs chain) | conv md5 (tape vs chain) |
|---|---|---|
| 0 | `24298c097c756c31` = `24298c097c756c31` | `c5b2d85c…` ≠ `db43fe0b…` |
| 1 | `796d8ea9a6e61861` = `796d8ea9a6e61861` | `2e8aff51…` ≠ `77208049…` |
| 2 | `e9350d9fb9dc89c8` = `e9350d9fb9dc89c8` | `4f9bb9f9…` = `4f9bb9f9…` |
| 3 | `05d27b274dcd6509` = `05d27b274dcd6509` | `b6bea526…` = `b6bea526…` |

**All four SSM rows are bit-exact; conv tape rows 0 and 1 are not** (rows 2–3 are). Steady-state
S=1 runs, by contrast, are bit-exact between the tape and fused ops (32-token argmax streams
identical, final SSM md5 identical). Consequence: a partial-accept commit (keep=1 → row 0,
keep=2 → row 1) rebinds a conv state that differs from what the re-forward path would produce,
so the tape arm's stream diverges from B0 (T1f first20 ≠ B1 first20). **The tape commit must
not be enabled in serving until the conv row-tape for rows 0/1 is fixed to match the S=1 chain.**
The SSM side needs nothing.

### Finding 4 — tape op crashes on `state=None`

`gdn_fused_step_tape`'s pybind declares `state: array` (non-optional). The model's tape branch
fires for any S≤4 forward when both env vars are set, including the first step of a fresh cache
(state is `None`) → `TypeError: py::argument … did not (cannot) convert`. Any raw `TAPE=1`
generation from a cold cache crashes. The bench masks it (its prompt forward is S>4). The model
branch needs a `state is not None` condition (or the pybind an `Optional` state).

---

## Job 2 — S=4 verify attention graft (scalar flash → 1-pass vector kernel)

v3 wheel built from the grafted vendor copy (4 min, 16 cores); graft markers present in both
`mlsda` and `mlsda_fa`; symbol table confirms the single-instantiation 1-pass kernel.

### G2 — S=4 causal rows vs AR S=1: **PASS**

- G2.1: **320/320** bf16 bit-exact (20 positions × 8 seeds, causal S=4 row i vs S=1 step i);
- G2.2: bf16 S=4 batch == bf16 S=1 chain — PASS;
- G2.3: f32 reference max abs diff `3.75e-3` (< 2e-2 gate) — the vector kernel agrees with the
  f32 reference in both arms;
- G2.4: overhead `0.38 µs`/call (negligible).

### M=1 invariant — **PASS**

Greedy AR, 300 tokens, same prompt, back-to-back under flock:

| | venv-verify (V2) | venv-attn (v3) |
|---|---|---|
| tokens | 300 | 300 |
| tok/s | 84.97 | 84.79 (−0.2 %) |
| ALLTOKS md5 | `fe73cf3bea85a67aca711e75f6400ad7` | `fe73cf3bea85a67aca711e75f6400ad7` |

Bit-exact token streams; no AR regression from the v3 wheel.

### G4 — lossless MTP (K=3 stream vs AR stream, same wheel)

300-token window, same prompt:

| wheel | K=3 vs AR | first divergence |
|---|---|---|
| venv-attn (v3, graft) | **bit-exact — PASS** | — |
| venv-verify (V2, old path) | diverges | token 299 (ar=13081, k3=974) |

The grafted wheel is the first one in which greedy MTP K=3 is a bit-exact lossless transform of
greedy AR; the old wheel breaks at the end of the window.

### Per-pair wall A/B — **verify −53 %, tok/s +44 %**

K=3 bench, 1024-token budget, 512 verify, back-to-back:

| | V2 (attn old) | V3 (graft) | Δ |
|---|---|---|---|
| tok/s | 35.75 | 51.41 | **+43.8 %** |
| wall ms/pair | 88.7 | 61.3 | −30.8 % |
| verify ms/pair | 54.29 | 25.64 | **−52.8 %** |
| single ms/pair | 9.67 | 9.64 | −0.3 % |
| refwd ms/pair | 24.54 | 25.90 | +5.5 % (stream-dependent, n_refwd 1071→1131) |
| acc d1/d2/d3 | 372/227/138 | 365/226/121 | streams branch early; d1pct 72.7→71.3, d3pct 60.8→53.5 |
| SUB draft / dlmh | 5.40 / 4.20 | 5.38 / 4.19 | flat |

The verify-class collapse (54.3 → 25.6 ms/pair) is the graft; draft/dlmh/single are flat, i.e.
the MTP module and lm_head are untouched, as expected.

---

## G6 — cleanup

- Pin and sidecars verified (table at top); no files created in `/mnt/models`.
- Guest scratch: `vendor-attn/` (grafted source), `wheelout-attn/` (v3 wheel), `t1280/` (bench,
  tests, fixed+stock `gdn_tape_commit.py`), `venv-attn/`, `/tmp/{ar,k3,tinv,biteq,j1}_*.log`.
- Lane left clean; lock held for the entire session, no contention observed.

## Environment notes (no impact on any A/B)

- Early in the session the AR baseline ran at ~42 tok/s instead of ~85; the cause was the wheel
  build (16-core cmake) contending for CPU/IO plus first-run GPU warm, not a stuck clock —
  runs 30 min later hit 84.97. All comparative numbers above are back-to-back and unaffected.
- `rocm-smi` on this userland prints `0Mhz/96Mhz` for SCLK even while delivering full
  throughput (driver/rocm-smi quirk on Ubuntu 26.10); throughput, not the clock readout, is the
  valid signal. `pp_dpm_sclk` similarly shows `0: 600Mhz *` statically.

## Bottom line

1. **Tape commit**: engaged op, dead guard. Root cause = 3 batch-axis confusion bugs in the
   guard (hv from `shape[0]`, SSM tail check, rebind dropping the batch axis) — not the op.
   Fixed guard: n_refwd 272→98, refwd 24.8→9.0 ms/pair, tok/s 45.6→61.1. **But the conv
   tape rows 0/1 are not bit-exact** (SSM rows are), so the tape arm is not servable yet;
   additionally the tape op crashes on `state=None` (fresh cache).
2. **S=4 verify graft**: G2 320/320 bit-exact, M=1 AR bit-exact with 0.2 % parity, K=3 lossless
   **PASS** (first wheel where it holds), verify −52.8 %, tok/s +43.8 %, draft path flat.
   The v3 wheel is ready to carry the S=4 verify attention in serving.
3. Recommended follow-ups (not done — outside this brief): fix the conv row-tape (rows 0/1) in
   the tape op or the commit protocol; add the `state is not None` guard to the model tape
   branch; fix the three guard axis bugs upstream (patch on guest: `t1280/gdn_tape_commit.py`,
   stock copy preserved as `.bak`).
