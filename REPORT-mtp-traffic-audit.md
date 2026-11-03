# MTP traffic audit (t1582)

Per-phase weight-traffic re-measurement on the current best stack (R4b fused
multi-row QMV verify, 48.93-class), answering ONE question: does the
commit/correction phase re-read the full target weights per step, or do
verify+commit share one target read?

## TL;DR

**Direct answer: commit/correction re-reads target: PARTIAL.** Commit does
NOT re-read (tape rebind = 0 GiB); the S=1 correction forward re-reads the
full target **once per rejected step** (126/161 = 78.3% of steps at the
anchor) = **3.26 GiB/step**. verify+commit share one target read (4.16
GiB); correction is a second, separate read.

Per-phase bytes (anchor mix: 161 verifies, 126 corrections, measured
tensor bytes, median of 3 per run):

| phase | target reads/step | target GiB/step | other GiB/step | total GiB/step |
|---|---|---|---|---|
| verify (S=4 batched, rows-fused M=4) | 1.000 | 4.16 | 0 | 4.16 |
| commit (tape rebind + KV trim) | 0 | 0 | 0 | 0.00 |
| correction (S=1, per rejected step) | 0.783 | 3.26 | 0 | 3.26 |
| draft (3x sidecar + 3x 4-bit head) | 0 | 0 | 1.97 | 1.97 |
| **total** | **1.783** | **7.42** | **1.97** | **9.39** |

(+ ~0.07 GiB KV reads on the 8 full-attention layers at ~510 tokens -
negligible). **Total ~9.4 GiB/step vs the ~11 GiB requirement: the stack
is already at/below the anchor traffic budget.** The remaining gap to
144.3 t/s is therefore NOT weight-traffic-bound; it is
compute/schedule-side (GDN ops on 32 CUs, 4-bit draft head - the R4b wheel
has no int2 op, kernel launch/eval bookkeeping). Per the brief's decision
rule, correction re-reading the target keeps **R5 (single-pass fused
step)** as the endgame lever: removing the correction forward takes
9.39 -> 6.13 GiB/step; but since we are already <= 11 GiB, the compute
ladders (int2 draft head port: -0.71 GiB; GDN op efficiency) cover the
rest.

## Setup

- Stack: branch mtp-r4b-fused-verify @ 808d4a4, R4b wheel
  `mlx-0.32.3.dev20261103+6f410ed` (libmlx md5 3d19e4f3420b3b067abaeec6ae971dac,
  verified on the guest) in its venv; standing env
  `MLX_MTP=4 MTP6_TAPE=hybrid GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1
  MLX_ROCM_QMV_ROWS_MIN_N=1 REFWD_M1=1`; the R4b venv's mlx_lm carries the
  FUSE + MTP_INJECT patches. Guest: RX 9070 XT / gfx1201 (32 CUs),
  <models-mount>/qwen3-9b-mlx. SBCAR sidecar
  `02a54359cda8b247f76b277dd9bc4d4c` recorded pre and post - unchanged.
- GPU protocol: `fuser` first (an in-lane job held the lock; waited, never
  benched contended), `flock -x` for the whole probe, 25 s idle before the
  first run and between runs.
- Bench md5s verified on the guest: `bench/mtp-bench6.py`
  700f1810207ccf8837be58f81c5eb944, `bench/mtp-bench6-phase.py`
  608a7770ef29fee23c33be057e7114a1.
- Regime sanity: anchor reproduced in-session - median 49.00 t/s
  (49.03/49.00/48.99, 161 verifies, FIRST20 identical to the R4b report's
  `271 1919 369 264 15352 ...`), 64v median 50.16 t/s (50.17/50.16/50.15,
  within 0.3% of the R4b report's 50.02).

## Method

Source (a) analytic: per-phase kernel inventory from the bench code +
measured tensor bytes, with one weight pass per op per forward (S=4 verify
rows run as one M=4 QMM pass with the rows fusion; S=1 correction/draft
forwards as one M=1 QMV pass; the R4b probe established the fused M=4 pass
is one weight pass, 1.4-2.4x vs per-row, maxdiff 1.2e-4 bf16).
Source (b) measured HSA byte counters: **not available** - the guest has no
rocprof and no amdgpu debugfs counters (non-root). Source (a) is the sole
basis.

`bench/mtp-traffic-audit.py` (new, ruff-formatted): loads the model and
enumerates weight-tensor bytes per module (packed 4-bit + scales + biases,
recursive walk), builds the int2 head only when the op exists, then runs a
short in-situ mode-4 hybrid generation on the exact stack env and counts
target forwards (S=4 verify, S=1 correction), sidecar forwards, and draft
head calls.

### Measured tensor bytes (probe, 64-token run)

| module | bytes | notes |
|---|---|---|
| target trunk (32 layers, 248 QLinears) | 3.63 GiB | 24 GDN @ 117.98 MB (7 QL) + 8 full-attn @ 122.91 MB (8 QL) |
| target lm_head (4-bit) | 0.533 GiB | 248320 x 4096, also the draft head as-run |
| **target per forward** | **4.16 GiB** | brief's ~5.3 GiB assumed a bf16 head; this target head is 4-bit |
| sidecar mtp module | 0.125 GiB | 1 GDN layer + norm |
| int2 draft head (if op present) | 0.296 GiB | ABSENT in the R4b wheel - INT2=off on all runs |

### Dispatch counters (in-situ)

`GDN_TAPE_CALLS` = 24 GDN layers x (verifies + corrections) on both runs:
short 1392 = 24 x (32+26), anchor 6888 = 24 x (161+126) - confirms the
fused/tape op path runs in both verify and correction. `GDN_FUSE_CALLS`
does not exist in this wheel (n/a). REFWDS matches the rejection counts
(126/161 anchor, 51/64 at 64v, 26/32 short).

## Per-phase bytes (derived, anchor acceptance mix)

Per step (1 verify + 1 draft chain), target read cost 4.16 GiB:

- verify: 1.000 read (S=4 batched, one pass per tensor) = 4.16 GiB.
- commit: 0 reads (tape state rebind + KV trim, no forward) = 0.00 GiB.
- correction: 0.783 reads (one S=1 forward per rejected step, 126/161)
  = 3.26 GiB.
- draft: 3 sidecar (3 x 0.125 = 0.38) + 3 x 4-bit lm_head (3 x 0.533 =
  1.60) = 1.97 GiB. Embedding gathers are single-row (negligible).
- total: 9.39 GiB/step (~9.5 with KV).

64v mix (51/64 rejects): 9.50 GiB/step (probe's 9.51 at 26/32).

## Per-phase wall (phase harness 64v, median of 3)

| phase | ms/step | notes |
|---|---|---|
| draft (3 sidecar + 3 head; also forces the lazily-evaluated correction forward) | 39.3 | 2516.0 / 64 |
| verify (lazy graph build only) | 0.5 | the S=4 GEMM is forced by the first argmax |
| commit (S=4 GEMM + tape rebind) | 26.7 | 1709.2-1710.6 / 64 |
| correction (bookkeeping; GEMM billed to next draft) | 0.4 | 25.0-25.3 / 64 |

Anchor: 10.49 s / 161 steps = 65.2 ms/step at 9.39 GiB -> ~146 GB/s
effective; 64v: ~66.9 ms/step -> ~140 GB/s. FIRST20 and accepts are
bit-stable across all runs (anchor D1/D2/D3 = 73.3/62.7/47.3, identical to
the R4b report; FIRST20 identical to the R4b report).

## t1508 comparison (34 GiB -> 9.4 GiB)

Pre-tape reconstruction with the same acceptance mix (verify per-row QMV,
savestate commit re-forward of the accepted prefix, 4-bit draft head):
verify 4 x 4.16 = 16.64 + commit E[re-forward] = (0.267 x 1 + 0.273 x 2
+ 0.243 x 3) = 1.543 reads = 6.42 + correction 3.26 + draft 1.97 = ~28.3
GiB (the t1508 battery's 34 also carried longer-KV and older-head
regime overhead). The two levers removed in bytes: rows fusion
(verify 4 -> 1 reads) **-12.48 GiB/step**, tape commit (re-forward -> 0)
**-6.42 GiB/step** - i.e. the whole 34 -> 9.4 delta is the two levers.

## Endgame verdict

1. Commit shares the verify read (0 GiB) - the tape lever fully removed
   the commit re-read.
2. Correction re-reads the full target once per rejected step (78.3% of
   anchor steps, 3.26 GiB/step) - this is the remaining structural
   re-read; R5 (single-pass fused step) removes it: 9.39 -> 6.13 GiB/step.
3. But the total (9.4 GiB) is already at/below the ~11 GiB anchor budget,
   so weight traffic is no longer the binding constraint: the 48.93 ->
   144.3 t/s gap is compute/schedule-side. Immediate levers: int2 draft
   head port (absent in this wheel; -0.71 GiB/step and its wall time is
   in the 39.3 ms draft phase) and GDN op efficiency on 32 CUs.

## Check lines

- Format check (`uv run ruff format --check .`): pass (90 files already
  formatted).
- Persona-leak grep over the diff (the named marzuki-* hostnames, 100.64/10
  addresses, the guest ssh username, *.junkyard.sh, neonvm@, 192.168.122.x,
  /mnt/models): 2 hits, both the self-referential search terms in this
  check line (the guest models path is written as `<models-mount>` in the
  report body).
- Line-wrap hit count (diff lines > 120 chars): 1 (the canonical-bench
  PROMPT string, verbatim from mtp-bench6.py; no hard-wrapped prose).
