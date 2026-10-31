# bench/

Guest-side MTP spec-decode bench harness, promoted from guest `/tmp` into the repo
(run-638 audit: the as-run lossless gate was vacuous — `cut -d=` against `ALLTOKS:`
lines with no `=`).

- `mtp-bench3.py` + `bench4.sh` — run-612 K=1 (S=2) harness + runner.
- `mtp-bench5.py` + `bench5.sh` — run-631 K=2 (S=3) harness + runner.
- `g1-t3-parts.py` — un-sandwiched T=1 vs S=3 forward + 2-draft timing probe.
- `step0_run.sh` + `step0_streaming_read.py` + `step0_parse.py` + `step0_hip_cu.cu`
  — step 0 (method-correction doc 2026-10-28, section 4): in-guest streaming-read
  ceiling (mx.sum over a >= 2 GiB f32 buffer, default 4 GiB; 3 warmup + 10
  measured sums) timed by rocprofv3 kernel time only (doc rule 2), plus runtime
  CU count (rocminfo + amd-smi + KFD topology + direct HIP query — the value
  mlx uses). Prints/asserts effective config (doc rule 6).
  `bash step0_run.sh` (env: STEP0_NBYTES / STEP0_WARMUP / STEP0_ITERS / STEP0_OUT).

Run on guest-a (all GPU work under `flock /models/gpu-bench.lock`):
`bash bench5.sh` (full bench + lossless gate) or
`BENCH5_OUT=/models/tmp/bench5 bash bench5.sh check` (no-GPU gate on saved
streams; same `check` mode for `bench4.sh`). Vacuity guard: any empty compared
stream -> `STATUS=VACUOUS` + exit 1; `MATCH` only on a non-empty full-length compare.
