#!/bin/bash
# dist-decode speed bench (run-761 fixed wheel, clone @75ca881):
#   (a) AR baseline      = mtp-bench3.py MLX_MTP=0  x3
#   (b) MTP K=1 M=2      = mtp-bench3.py MLX_MTP=2  x3
#   (c) MTP K=2 (S=3)    = mtp-bench5.py MLX_MTP=3  x1  (only if K2=1)
# 1024 tokens, greedy, fresh process per run, all GPU work under flock.
# 3 repeats per config, median reported.
#
# G4 lossless gate: same vacuity-guarded helper as bench4.sh / bench5.sh (PR #8).
# Any empty compared stream -> STATUS=VACUOUS + exit 1; MATCH only on a
# non-empty full-length compare.
#
# Usage (on guest-a):
#   bash bench-decode.sh                 # AR x3 + MTP K=1 x3 + gate
#   K2=1 bash bench-decode.sh           # + MTP K=2 x1 (run after (b) >= AR)
#
# Env overrides: BENCH_PY (venv python), BENCH_HDIR (harness dir), BENCH_OUT.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
MODELS="${MODELS:-/models}"
PY=${BENCH_PY:-$MODELS/venv-d/bin/python}
H=${BENCH_HDIR:-$MODELS/tmp/bench-decode}
OUT=${BENCH_OUT:-$MODELS/tmp/bench-decode-out}
mkdir -p $OUT
LOCK=$MODELS/gpu-bench.lock

flock -w 10 $LOCK -c true || { echo "LOCK-BUSY"; exit 3; }

run() { # run <name> <harness> <mode>
  echo "RUN $1 harness=$2 MLX_MTP=$3"
  flock -w 300 $LOCK env MLX_MTP=$3 $PY $H/$2 > $OUT/$1.out 2>&1
  echo "DONE $1 rc=$?"
}

run m0a mtp-bench3.py 0
run m0b mtp-bench3.py 0
run m0c mtp-bench3.py 0
run m2a mtp-bench3.py 2
run m2b mtp-bench3.py 2
run m2c mtp-bench3.py 2
if [ "${K2:-0}" = 1 ]; then
  run m3a mtp-bench5.py 3
fi

# --- lossless gate (vacuity-guarded; same helper as bench4.sh / bench5.sh) -----
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

gate G4-M2-M0 "$OUT/m2a.out" "$OUT/m0a.out"
gate M2-A-B   "$OUT/m2a.out" "$OUT/m2b.out"
gate M2-B-C   "$OUT/m2b.out" "$OUT/m2c.out"
gate M0-A-B   "$OUT/m0a.out" "$OUT/m0b.out"
gate M0-B-C   "$OUT/m0b.out" "$OUT/m0c.out"
if [ -f "$OUT/m3a.out" ]; then
  gate M3-M0   "$OUT/m3a.out" "$OUT/m0a.out"
fi
if [ "$VACUOUS" -eq 1 ]; then
  echo "STATUS=VACUOUS"
  exit 1
fi
echo "GATE-LIVE (all compared streams non-empty)"

# --- summary: median of TOKS_S per config (3 repeats -> middle) ---------------
summarize() { # summarize <cfg-prefix>
  local vals
  vals=$(grep -h "TOKS_S=" $OUT/$1?.out 2>/dev/null | sed 's/.*TOKS_S=\([0-9.]*\).*/\1/' | sort -n)
  [ -z "$vals" ] && { echo "$1: NO-DATA"; return 0; }
  echo "$vals" | awk -v c="$1" '
    { v[NR] = $1 }
    END {
      n = NR; med = v[(n + 1) / 2]
      printf "%s: n=%d min=%.2f median=%.2f max=%.2f\n", c, n, v[1], med, v[n]
    }'
  grep -h "ACCEPTED=" $OUT/$1?.out 2>/dev/null | sort | uniq -c | sed 's/^/  /'
}
summarize m0
summarize m2
[ -f "$OUT/m3a.out" ] && summarize m3
echo BENCH-G4D-DONE
