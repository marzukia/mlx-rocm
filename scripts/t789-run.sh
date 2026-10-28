#!/bin/bash
# t789 gate runner (guest, under flock). Runs the from-ref-wheel gates:
#   config (rule 6), kernels (rocprof dispatch), G1 (fused serve/generation,
#   no-fault), G2 (fused vs m0 reference + fused vs unfused, maxdiff 0.0),
#   plus the t761 primitive gates (prep/norm/fwd) on the same wheel.
# Vacuity-guarded stream compare (bench4 pattern): a compare passes only on a
# non-empty full-length match.
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
OUT=${T789_OUT:-/mnt/models/tmp/bench-t789-out}
mkdir -p $OUT
LOCK=/mnt/models/gpu-bench.lock
PY=/mnt/models/venv-t789/bin/python
D=/mnt/models/tmp/bench-t789

get_stream() { grep "^ALLTOKS:" "$1" 2>/dev/null | cut -d' ' -f2-; }

cmp_streams() {
  local label=$1 fa=$2 fb=$3
  local A B
  A=$(get_stream "$fa"); B=$(get_stream "$fb")
  if [ -z "$A" ] || [ -z "$B" ]; then
    echo "$label: STATUS=VACUOUS (empty stream: $(basename "$fa")=$( [ -n "$A" ] && echo set || echo EMPTY ) $(basename "$fb")=$( [ -n "$B" ] && echo set || echo EMPTY ))"
    return 2
  fi
  local AA BB NA NB N i first=-1 match=0 diff=0
  read -ra AA <<< "$A"; read -ra BB <<< "$B"
  NA=${#AA[@]}; NB=${#BB[@]}
  if [ "$NA" -ne "$NB" ]; then
    N=$NB; [ "$NA" -lt "$NB" ] && N=$NA
  else
    N=$NA
  fi
  for ((i=0; i<N; i++)); do
    if [ "${AA[$i]}" = "${BB[$i]}" ]; then
      match=$((match+1))
    else
      if [ $first -lt 0 ]; then first=$((i+1)); fi
      diff=$((diff+1))
    fi
  done
  if [ "$diff" -eq 0 ]; then
    echo "$label: STATUS=MATCH (maxdiff 0.0) first_div=- match=$match/$N"
    return 0
  fi
  echo "$label: STATUS=FAIL first_div=$first match=$match diff=$diff of $N"
  return 1
}

run() { # run <name> <env...> <cmd...>
  local name=$1; shift
  echo "RUN $name"
  flock -w 300 $LOCK env "$@" $PY $D/t789_$name.py > $OUT/$name.out 2>&1
  local rc=$?
  echo "DONE $name rc=$rc"
  return $rc
}

fail=0
run config || fail=1
grep -q "t789-config: OK" $OUT/config.out && echo "CONFIG-OK" || { echo "CONFIG-FAIL"; fail=1; }

run kernels || fail=1
grep -q "KERNEL-DISPATCHED: gdn_prep" $OUT/kernels.out && grep -q "KERNEL-DISPATCHED: gdn_qk_norm" $OUT/kernels.out && grep -q "KERNEL-DISPATCHED: gdn_norm_gated" $OUT/kernels.out && echo "KERNELS-OK (gdn_prep, gdn_qk_norm, gdn_norm_gated)" || { echo "KERNELS-FAIL"; fail=1; }

# G1: fused serve/generation, no fault
run g1-fused MLX_GDN_TAIL_FUSION=1 || fail=1
grep -q "TOKS=1024" $OUT/g1-fused.out && ! grep -qE "HSA|fault|Aborted|core dumped|Segmentation|HIP" $OUT/g1-fused.out && echo "G1-PASS: no-fault serve/generation (1024 tokens, TAIL_FUSION=1)" || { echo "G1-FAIL"; fail=1; }

# G2: fused vs m0 reference + fused vs unfused (same wheel)
run g2-unfused MLX_GDN_TAIL_FUSION=0 || fail=1
cmp_streams G2-FUSED-VS-M0-REF $OUT/g1-fused.out /mnt/models/tmp/bench4/m0.out || fail=1
cmp_streams G2-FUSED-VS-UNFUSED $OUT/g1-fused.out $OUT/g2-unfused.out || fail=1

# t761 primitive gates on the same wheel
run prim-t724c_prep MLX_GDN_TAIL_FUSION=1 || fail=1
run prim-t724c_norm MLX_GDN_TAIL_FUSION=1 || fail=1
run prim-t724c_fwd MLX_GDN_TAIL_FUSION=1 || fail=1
if grep -q "t724c-prep: PASS" $OUT/prim-t724c_prep.out && grep -q "t724c-norm: PASS" $OUT/prim-t724c_norm.out && grep -q "t724c-fwd: PASS" $OUT/prim-t724c_fwd.out; then
  echo "PRIM-GATES-PASS"
  grep -E "maxdiff" $OUT/prim-t724c_prep.out $OUT/prim-t724c_norm.out $OUT/prim-t724c_fwd.out | sed 's/^.*: //'
else
  echo "PRIM-GATES-FAIL"
  fail=1
fi

if [ $fail -eq 0 ]; then echo "T789-GATES-DONE"; else echo "T789-GATES-FAILED"; fi
exit $fail
