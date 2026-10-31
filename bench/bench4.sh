#!/bin/bash
# K=1 MTP spec-decode bench (run-612): m0 (AR) + m1 (2x T=1 verify) + m2 (S=2) x3.
# 1024 tokens, greedy, fresh process per run, flock.
#
# Lossless gate (run-638 audit n-1 fix): mtp-bench3.py prints "ALLTOKS: <toks>"
# (no '='), but the executed bench4.sh grepped '^STREAM=' (0 matches) and cut
# on '=' -> A=B="" -> STREAMS-MATCH passed on empty==empty (vacuous). The
# grep now targets the real line (cut -d' '); every compare fails LOUD when a
# compared stream is empty: STATUS=VACUOUS, exit 1. MATCH may only be printed
# on a non-empty full-length compare.
#
# Usage:
#   bash bench4.sh                          # full bench (GPU, under flock) + lossless gate
#   BENCH4_OUT=$MODELS/tmp/bench4 bash bench4.sh check
#                                          # no-GPU: lossless gate on saved streams only
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
MODELS="${MODELS:-/models}"
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
OUT=${BENCH4_OUT:-$MODELS/tmp/bench4}
mkdir -p $OUT
LOCK=$MODELS/gpu-bench.lock

# --- lossless gate (vacuity-guarded; same helper as bench5.sh) -----------------
# mtp-bench3.py prints exactly:  MTP=...  /  FIRST16: ...  /  ALLTOKS: <toks>  /  TEXT: ...
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
  # K=1 lossless claim (run-612 report: "m2 token stream byte-identical to m0")
  gate M2-M0-LOSSLESS "$OUT/m0.out" "$OUT/m2a.out"
  # control: 2x T=1 verify machinery (m1) must be byte-identical to m0
  gate M1-M0-CONTROL "$OUT/m0.out" "$OUT/m1.out"
  # the pair the executed bench4.sh compared (m1 vs m2a)
  gate M1-M2A "$OUT/m1.out" "$OUT/m2a.out"
  # m2 determinism (a==b==c, full streams)
  gate M2-A-B "$OUT/m2a.out" "$OUT/m2b.out"
  gate M2-B-C "$OUT/m2b.out" "$OUT/m2c.out"
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

flock -w 900 $LOCK env MLX_MTP=0 $MODELS/venv-a/bin/python /tmp/mtp-bench3.py > $OUT/m0.out 2>&1
flock -w 900 $LOCK env MLX_MTP=1 $MODELS/venv-a/bin/python /tmp/mtp-bench3.py > $OUT/m1.out 2>&1
flock -w 900 $LOCK env MLX_MTP=2 $MODELS/venv-a/bin/python /tmp/mtp-bench3.py > $OUT/m2a.out 2>&1
flock -w 900 $LOCK env MLX_MTP=2 $MODELS/venv-a/bin/python /tmp/mtp-bench3.py > $OUT/m2b.out 2>&1
flock -w 900 $LOCK env MLX_MTP=2 $MODELS/venv-a/bin/python /tmp/mtp-bench3.py > $OUT/m2c.out 2>&1

run_gate

for f in m0 m1 m2a m2b m2c; do echo "-- $f:"; grep -E "TOKS=|ACCEPTED=|ALLTOKS:" $OUT/$f.out 2>/dev/null | head -2; done
echo BENCH4-DONE
