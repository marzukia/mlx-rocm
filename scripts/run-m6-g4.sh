#!/bin/bash
# m6: G4 lossless gate (2-row causal verify vs 1-row AR) on venv-d.
#
# Adapted from bench/bench4.sh (same gate logic and prompts): venv-d +
# the repo's mtp-bench3.py (uploaded to $OUT), bounded flock -w 900 with
# retries (an earlier run may re-take the lock; the m5 retry pattern).
#
# Usage:
#   bash run-m6-g4.sh run     # run all arms + gate (under the lock)
#   bash run-m6-g4.sh check   # re-evaluate the gate from existing outputs
#
# Arms: m0 = AR (MLX_MTP=0); m1 = MTP K=0 (control); m2a/b/c = MTP K=1
# (MLX_MTP=2). 1024 tokens each (mtp-bench3.py default).
#
# The m6 gate (the task's "G4 lossless gate"): M2-M0-LOSSLESS must be
# MATCH (bit-exact). Pre-patch it FAILs at token 149 (first_div=149).
# The other gates are the m4/m5 controls (they must stay MATCH - they
# passed pre-patch; a regression here would implicate the patch).
#
# Also extracts TOKS_S (AR from m0, MTP K=1 from m2a) for the bench
# regression check (baselines: AR 69.65 / MTP K=1 64.34 tok/s).

export PATH=/opt/rocm/bin:/opt/rocm/core-10.1/bin:/opt/rocm/lib:/opt/rocm/core-10.1/lib:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:/opt/rocm/lib

MODELS="${MODELS:-/models}"
OUT=${M6_OUT:-$MODELS/tmp/an earlier run}
mkdir -p $OUT
LOCK=$MODELS/gpu-bench.lock
PY=$MODELS/venv-d/bin/python
BENCH=$OUT/mtp-bench3.py

get_stream() {
  # mtp-bench3.py prints "ALLTOKS: <toks>" (no '='; the an earlier run audit fixed
  # the vacuous ^STREAM= grep - an empty match would pass empty==empty)
  grep "^ALLTOKS:" "$1" 2>/dev/null | cut -d' ' -f2-
}

cmp_streams() {  # $1=name  $2=stream_a  $3=stream_b
  python3 - "$1" "$2" "$3" <<'EOF'
import sys
name, a, b = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
n = min(len(a), len(b))
first = -1
for i in range(n):
    if a[i] != b[i]:
        first = i
        break
if first >= 0:
    print(f"{name}: FAIL first_div={first} a={a[first]} b={b[first]}")
elif len(a) != len(b):
    print(f"{name}: FAIL len_a={len(a)} len_b={len(b)}")
else:
    print(f"{name}: MATCH n={n}")
EOF
}

run_gate() {
  s0=$(get_stream $OUT/m0.out)
  s1=$(get_stream $OUT/m1.out)
  s2a=$(get_stream $OUT/m2a.out)
  s2b=$(get_stream $OUT/m2b.out)
  s2c=$(get_stream $OUT/m2c.out)
  [ -n "$s0" ] || { echo "GATE VACUOUS: m0.out has no STREAM"; return 1; }
  [ -n "$s1" ] || { echo "GATE VACUOUS: m1.out has no STREAM"; return 1; }
  [ -n "$s2a" ] || { echo "GATE VACUOUS: m2a.out has no STREAM"; return 1; }
  [ -n "$s2b" ] || { echo "GATE VACUOUS: m2b.out has no STREAM"; return 1; }
  [ -n "$s2c" ] || { echo "GATE VACUOUS: m2c.out has no STREAM"; return 1; }
  cmp_streams "M2-M0-LOSSLESS (the m6 gate)" "$s2a" "$s0"
  cmp_streams "M1-M0-CONTROL" "$s1" "$s0"
  cmp_streams "M1-M2A" "$s1" "$s2a"
  cmp_streams "M2-A-B" "$s2a" "$s2b"
  cmp_streams "M2-B-C" "$s2b" "$s2c"
  grep -h "TOKS_S=" $OUT/m0.out $OUT/m2a.out | sed 's/^/bench: /'
}

# $1=env (e.g. MLX_MTP=0)  $2=outfile
run_one() {
  local envv=$1 out=$2
  for attempt in 1 2 3 4 5 6 7 8; do
    flock -w 900 $LOCK env $envv $PY $BENCH > $out 2>&1
    rc=$?
    if [ $rc -eq 1 ] && ! grep -q "TOKS_S=" $out 2>/dev/null; then
      echo "FLOCK-FAIL $envv attempt=$attempt $(date -u +%T)" >> $OUT/run.log
      sleep 60
      continue
    fi
    echo "done $envv pyrc=$rc attempt=$attempt $(date -u +%T)" >> $OUT/run.log
    return 0
  done
  echo "FLOCK-FAIL $envv after 8 attempts" >> $OUT/run.log
  return 1
}

if [ "$1" = "check" ]; then
  run_gate
  exit 0
fi

[ -f $BENCH ] || { echo "missing $BENCH (upload bench/mtp-bench3.py first)"; exit 3; }
: > $OUT/run.log
run_one "MLX_MTP=0" $OUT/m0.out
run_one "MLX_MTP=1" $OUT/m1.out
run_one "MLX_MTP=2" $OUT/m2a.out
run_one "MLX_MTP=2" $OUT/m2b.out
run_one "MLX_MTP=2" $OUT/m2c.out
run_gate
