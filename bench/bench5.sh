#!/bin/bash
# K=2 MTP spec-decode bench: m0 (AR) + m3 (K=2, S=3) x3 + m2 sanity anchor.
# 1024 tokens, greedy, fresh process per run, flock.
#
# Lossless gate (t638 audit B-1 fix): mtp-bench5.py prints "ALLTOKS: <toks>"
# (no '='), so the stream is extracted with cut -d' '. The executed guest copy
# (md5 42efb930) used grep '^ALLTOKS=' | cut -d= -> every stream var empty ->
# every compare passed on empty==empty (vacuous). Every compare now fails LOUD
# when a compared stream is empty: STATUS=VACUOUS, exit 1. MATCH may only be
# printed on a non-empty full-length compare (for the prefix check: the full
# length of the shorter stream, i.e. all of m0's 1024 tokens).
#
# Usage:
#   bash bench5.sh                          # full bench (GPU, under flock) + lossless gate
#   BENCH5_OUT=/mnt/models/tmp/bench5 bash bench5.sh check
#                                          # no-GPU: lossless gate on saved streams only
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
PY=/mnt/models/venv-g1/bin/python
OUT=${BENCH5_OUT:-/mnt/models/tmp/bench5}
mkdir -p $OUT
LOCK=/mnt/models/gpu-bench.lock

# --- lossless gate (vacuity-guarded) ------------------------------------------
# mtp-bench5.py prints exactly:  MTP=...  /  FIRST16: ...  /  ALLTOKS: <toks>  /  TEXT: ...
get_stream() { grep "^ALLTOKS:" "$1" 2>/dev/null | cut -d' ' -f2-; }

# cmp_streams <label> <fileA> <fileB> [prefix]
#   exact  : A and B non-empty, equal length, all positions equal -> MATCH
#   prefix : A non-empty, B at least as long, all of A's positions equal -> MATCH
# Prints: <label>: STATUS=<MATCH|FAIL|VACUOUS> first_div=<1-based tok|-> match=<m> diff=<d> of <n>
# Returns: 0=MATCH 1=FAIL 2=VACUOUS
cmp_streams() {
  local label=$1 fa=$2 fb=$3 mode=${4:-exact}
  local A B
  A=$(get_stream "$fa"); B=$(get_stream "$fb")
  if [ -z "$A" ] || [ -z "$B" ]; then
    echo "$label: STATUS=VACUOUS (empty stream: $(basename "$fa")=$( [ -n "$A" ] && echo set || echo EMPTY ) $(basename "$fb")=$( [ -n "$B" ] && echo set || echo EMPTY ))"
    return 2
  fi
  local AA BB NA NB N i first=-1 match=0
  read -ra AA <<< "$A"; read -ra BB <<< "$B"
  NA=${#AA[@]}; NB=${#BB[@]}; N=$NA
  [ "$NB" -lt "$N" ] && N=$NB
  for ((i = 0; i < N; i++)); do
    if [ "${AA[$i]}" = "${BB[$i]}" ]; then
      match=$((match + 1))
    elif [ "$first" -lt 0 ]; then
      first=$((i + 1))
    fi
  done
  local diff=$((N - match))
  if [ "$mode" = prefix ]; then
    if [ "$NB" -lt "$NA" ]; then
      echo "$label: STATUS=FAIL (B=$NB shorter than A=$NA) first_div=$first match=$match diff=$diff of $N"
      return 1
    fi
    if [ "$diff" -eq 0 ]; then
      echo "$label: STATUS=MATCH (A=$NA is prefix of B=$NB) first_div=- match=$match/$N"
      return 0
    fi
    echo "$label: STATUS=FAIL first_div=$first match=$match diff=$diff of $N (A=$NA B=$NB)"
    return 1
  fi
  if [ "$NA" -ne "$NB" ]; then
    echo "$label: STATUS=FAIL (length A=$NA B=$NB) first_div=$first match=$match diff=$diff of $N"
    return 1
  fi
  if [ "$diff" -eq 0 ]; then
    echo "$label: STATUS=MATCH first_div=- match=$match/$N"
    return 0
  fi
  echo "$label: STATUS=FAIL first_div=$first match=$match diff=$diff of $N"
  return 1
}

VACUOUS=0
gate() {  # gate <label> <fileA> <fileB> [prefix]; missing file -> SKIP (not vacuous)
  local label=$1 fa=$2 fb=$3
  if [ ! -f "$fa" ] || [ ! -f "$fb" ]; then
    echo "$label: STATUS=SKIP (missing: $( [ -f "$fa" ] || echo "$(basename "$fa") " )$( [ -f "$fb" ] || echo "$(basename "$fb")" ))"
    return 0
  fi
  cmp_streams "$@"
  local rc=$?
  [ "$rc" -eq 2 ] && VACUOUS=1
  return 0
}

run_gate() {
  # K=2 lossless claim (the check the executed guest copy passed vacuously): m0 prefix of m3a
  gate M3-M0-LOSSLESS-PREFIX "$OUT/m0.out" "$OUT/m3a.out" prefix
  # m2 sanity vs m0 (K=1 emits 1 token/verify step -> equal length, exact match)
  gate M2-M0-STREAMS "$OUT/m0.out" "$OUT/m2s.out"
  # control: T=1 re-forward machinery (m1s) must be byte-identical to m0
  gate M1S-M0-CONTROL "$OUT/m0.out" "$OUT/m1s.out"
  # m3 determinism (a==b==c, full streams)
  gate M3-A-B "$OUT/m3a.out" "$OUT/m3b.out"
  gate M3-B-C "$OUT/m3b.out" "$OUT/m3c.out"
  # out-of-protocol diagnostic, if present (third fp variant)
  gate M2S-FALLBACK-M0 "$OUT/m0.out" "$OUT/m2s_fallback.out"
  if [ "$VACUOUS" -eq 1 ]; then
    echo "STATUS=VACUOUS"
    exit 1
  fi
  echo "GATE-LIVE (all compared streams non-empty)"
}

if [ "$1" = "check" ]; then
  run_gate
  exit 0
fi

# S=3 vs T=1 forward timing (un-sandwiched, like g1-t2-parts.py)
flock -w 900 $LOCK $PY /tmp/g1-t3-parts.py > $OUT/t3parts.out 2>&1

# m0 AR baseline
flock -w 900 $LOCK env MLX_MTP=0 $PY /tmp/mtp-bench5.py > $OUT/m0.out 2>&1

# m2 sanity anchor (reproduce the t612 63.8 reference; validates wheel+harness)
flock -w 900 $LOCK env MLX_MTP=2 $PY /tmp/mtp-bench5.py > $OUT/m2s.out 2>&1

# m3 K=2 x3 (stability, per t612 protocol)
flock -w 900 $LOCK env MLX_MTP=3 $PY /tmp/mtp-bench5.py > $OUT/m3a.out 2>&1
flock -w 900 $LOCK env MLX_MTP=3 $PY /tmp/mtp-bench5.py > $OUT/m3b.out 2>&1
flock -w 900 $LOCK env MLX_MTP=3 $PY /tmp/mtp-bench5.py > $OUT/m3c.out 2>&1

echo "=== SUMMARY ==="
for f in t3parts m0 m2s m3a m3b m3c; do
  echo "-- $f:"
  grep -E "T1_FWD_MS=|S3_FWD_MS=|S3_OVER_T1=|DRAFT2_MS=|MTP=|TOKS_S=" $OUT/$f.out 2>/dev/null | head -3
done

run_gate
echo BENCH5-DONE
