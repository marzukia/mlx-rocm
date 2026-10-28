# bench/

Guest-side MTP spec-decode bench harness, promoted from guest `/tmp` into the repo
(t638 audit: the as-run lossless gate was vacuous — `cut -d=` against `ALLTOKS:`
lines with no `=`).

- `mtp-bench3.py` + `bench4.sh` — t612 K=1 (S=2) harness + runner.
- `mtp-bench5.py` + `bench5.sh` — t631 K=2 (S=3) harness + runner.
- `g1-t3-parts.py` — un-sandwiched T=1 vs S=3 forward + 2-draft timing probe.

Run on marzuki-neon (all GPU work under `flock /mnt/models/gpu-bench.lock`):
`bash bench5.sh` (full bench + lossless gate) or
`BENCH5_OUT=/mnt/models/tmp/bench5 bash bench5.sh check` (no-GPU gate on saved
streams; same `check` mode for `bench4.sh`). Vacuity guard: any empty compared
stream -> `STATUS=VACUOUS` + exit 1; `MATCH` only on a non-empty full-length compare.
