# SALVAGE-NOTE t1339 (cancelled 08:17Z by cancel80, wall-less 4h fallback)

## Goal
WMMA (RDNA4 bf16/fp16 matrix-unit) M<=4 QMV kernel for the K=3 MTP verify GEMMs.
New kernel qmv_wmma_m4_kernel + dispatch in vend qmm.hip, env MLX_QMV_WMMA=1
(default off, zero behavior change when unset). Statistical gate: maxdiff <= 1e-2
vs the rows reference (NOT row-exact). Bench targets: verify <= 16 ms, pair
<= 35 ms on the canonical mtp-bench-k3-tape.py (B0/R1, reproduce-first gate).
Design source: REPORT-gemms-m4.md section 4 (branch worker/gemms-m4 @cad05da).

## Where we stopped
- Kernel + dispatch: patches/qmv-wmma-m4/ (qmv_wmma_m4_kernel.hip +
  qmv-wmma-m4-dispatch.diff against vend qmm.hip). Guest build tree:
  ~/t1339-wmma/vend (dirty: modified qmm.hip + kernel in place),
  build logs build.log .. build17.log.
- History: builds 1-13 failed (compile). Build 14 = first clean wheel (07:48Z).
  Then the A-fragment was reworked bf16 -> fp16. The unit reference matrix
  (rows_* / wmma_* .npy, 6 shapes x M=1..4) was captured under the bf16 build
  at 06:34 in ~/t1339-wmma/unit/.
- After the fp16 rework: "Rows >=1 all garbage" - unit outputs wrong.
  Debugging in progress at kill time: (a) dispatch macro argument-order check,
  (b) kernel-side debug prints comparing A row-1 contents vs global x row-1,
  (c) an off-by-half tail-load fix identified in two spots (word offset
  formula), in flight.
- Probes: dbg2.py, dbg3.py (small quantized_matmul repros; dbg3 needs K
  divisible by the group size 64).
- Guest lane: the 48h placeholder flock was reaped by the orchestrator at
  08:19Z (lane free). An orphaned build-17 (pip wheel + hipcc) was still
  compiling at kill time and may have finished on the guest; check
  build17.log + wheelout/ and reap leftover build processes before reusing
  the tree.

## Next (continuation, in order)
1. Read this note + the tail of session.jsonl (last ~40 tool calls) to load
   the debug state; check the guest (build17.log, wheelout/, vend state).
2. Fix the Rows>=1 garbage bug (candidates in play: dispatch arg order,
   A-fragment layout, off-by-half tail load).
3. Unit gate: maxdiff <= 1e-2 vs the rows reference, all 6 shapes x M=1..4
   (references cached in ~/t1339-wmma/unit/).
4. K=3 A/B spot check (report first-divergence token).
5. Canonical B0/R1: B0 (rows OFF) must reproduce ~69.48 tok/s +/-3% in-session
   before R1 (rows ON + WMMA ON); two runs per arm; report the noise floor.
6. Deliver REPORT-wmma-m4.md + the patches/ files, commit EARLY per verified
   step (a second cancel80 at 4h is live for wall-less runs), persona scrub,
   push, release the GPU lock, reap all background processes before exit.

## Canonical refs
Bench: /srv/fleetwork/work/1292/mtp-bench-k3-tape.py (md5
2c2d6c90c2349ae087e53e5d6c60018e) + gdn_tape_commit.py helper. Wheel base:
vendor-rs @75ca881 (t1334 tree, rows + hoist gated off). venv-rs. Model
qwen3-9b-mlx (guest model mount), sidecar hash-pin ONLY model-mtp.safetensors.
