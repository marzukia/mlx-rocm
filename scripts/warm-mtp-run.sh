#!/bin/bash
# run-1014 K2 k_proj T-invariance: probes + gates + bench, lane wheel
# (run-1012-29k/venv-m8, libmlx.so sha256[:12]=de6de5f154c0).
#
# 1. gate (a): direct k_proj T-invariance probe (D1 model QMM k_proj,
#    D1b q/v control, D2 plain matmul (P3 audit direct test), D3 determinism)
# 2. in-flow audit probe (m8-kproj-flowprobe.py, fork @ verify idx 47)
# 3. gate (b): G4 5-arm (mtp-bench3.py, 1024 tokens, 14-token prompt)
# 4. K=2 byte-identical @ 14-token prompt (mtp-bench5.py: AR 2048 vs K=2)
# 5. bench: 14-token prompt (run-1014 warm harness, 1024 tokens, warmup-first)
# 6. bench: 29K prompt (run-1012-29k.py AR unmodified; run-1014 warm harness K=1/K=2)
# 7. K=2 byte-identical @ 29K (512 tokens, warmoff)
set -u
MODELS="${MODELS:-/models}"
PY=$MODELS/tmp/run-1012-29k/venv-m8/bin/python
D=$MODELS/tmp/run-1014
LOCK=$MODELS/gpu-bench.lock
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:/opt/rocm/lib
LDP="env LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
run() { # $1=tag $2...=cmd
  local tag=$1; shift
  flock -w 3600 $LOCK $LDP "$@" > $D/$tag.out 2>&1
  echo "rc=$? $tag $(date -u +%T)" >> $D/run.log
}

echo "START $(date -u +%FT%TZ)" >> $D/run.log
$PY -c "import mlx.core as mx; print('IMPORT-OK', mx.__file__)" >> $D/import.log 2>&1
sha256sum $MODELS/tmp/run-1012-29k/venv-m8/lib/python3.12/site-packages/mlx/lib/libmlx.so >> $D/import.log
free -g >> $D/import.log

# 1) gate (a)
run kproj-probe $PY $D/k2-kproj-tinvariance-probe.py

# 2) in-flow audit probe
run flowprobe $PY $D/m8-kproj-flowprobe.py

# 3) gate (b): G4 5-arm
for arm in "m0 MLX_MTP=0" "m1 MLX_MTP=1" "m2a MLX_MTP=2" "m2b MLX_MTP=2" "m2c MLX_MTP=2"; do
  set -- $arm
  flock -w 3600 $LOCK env $2 $PY $D/mtp-bench3.py > $D/$1.out 2>&1
  echo "rc=$? $1 $(date -u +%T)" >> $D/run.log
done
python3 - <<'EOF' >> $D/gate.log 2>&1
import os
MODELS = os.environ.get("MODELS", "/models")
import subprocess

def get_stream(p):
    try:
        with open(p) as f:
            for line in f:
                if line.startswith("ALLTOKS:"):
                    return line.split(None, 1)[1].split()
    except OSError:
        pass
    return []

def cmp(name, a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is not None:
        print(f"{name}: FAIL first_div={first} a={a[first]} b={b[first]}")
    elif len(a) != len(b):
        print(f"{name}: FAIL len_a={len(a)} len_b={len(b)}")
    else:
        print(f"{name}: MATCH n={n}")

D = f"{MODELS}/tmp/run-1014"
s0 = get_stream(f"{D}/m0.out")
s1 = get_stream(f"{D}/m1.out")
sa = get_stream(f"{D}/m2a.out")
sb = get_stream(f"{D}/m2b.out")
sc = get_stream(f"{D}/m2c.out")
assert s0, "m0.out has no stream"
assert s1, "m1.out has no stream"
assert sa, "m2a.out has no stream"
cmp("M2-M0-LOSSLESS (the m6 gate)", sa, s0)
cmp("M1-M0-CONTROL", s1, s0)
cmp("M1-M2A", s1, sa)
cmp("M2-A-B", sa, sb)
cmp("M2-B-C", sb, sc)
EOF

# 4) K=2 byte-identical @ 14-token prompt
flock -w 3600 $LOCK env MLX_MTP=0 MTP_MAX_TOKENS=2048 $PY $D/mtp-bench5.py > $D/ar2048.out 2>&1
echo "rc=$? ar2048 $(date -u +%T)" >> $D/run.log
flock -w 3600 $LOCK env MLX_MTP=3 MTP_MAX_TOKENS=1024 $PY $D/mtp-bench5.py > $D/m3-14tok.out 2>&1
echo "rc=$? m3-14tok $(date -u +%T)" >> $D/run.log
python3 - <<'EOF' >> $D/k2-14tok-gate.log 2>&1
import os
MODELS = os.environ.get("MODELS", "/models")
def toks(p):
    with open(p) as f:
        for line in f:
            if line.startswith("ALLTOKS:"):
                return line.split(None, 1)[1].split()
    return []

ar = toks(f"{MODELS}/tmp/run-1014/ar2048.out")
m3 = toks(f"{MODELS}/tmp/run-1014/m3-14tok.out")
n = len(m3)
ok = ar[:n] == m3
first = next((i for i in range(min(len(ar), n)) if ar[i] != m3[i]), None)
print(f"K2-14TOK: ar_len={len(ar)} m3_len={n} prefix_match={ok} first_diff={first}")
EOF
grep -h "MTP=3\|MTP=0" $D/ar2048.out $D/m3-14tok.out | sed 's/^/bench: /' >> $D/k2-14tok-gate.log

# 5) bench @ 14-token prompt (warmup-first)
for m in 0 2 3; do
  flock -w 3600 $LOCK env MLX_MTP=$m T14_MAX_TOKENS=1024 $PY $D/run-1014-warm-mtp-bench.py > $D/bench14-m$m.out 2>&1
  echo "rc=$? bench14-m$m $(date -u +%T)" >> $D/run.log
done

# 6) bench @ 29K (AR = run-1012-29k.py UNMODIFIED; K=1/K=2 = run-1014 warm harness)
flock -w 3600 $LOCK $LDP $PY $MODELS/tmp/run-1012-29k/run-1012-29k.py > $D/ar29k-run-1012.out 2>&1
echo "rc=$? ar29k-run-1012 $(date -u +%T)" >> $D/run.log
for m in 2 3; do
  flock -w 3600 $LOCK env MLX_MTP=$m T29K_TARGET=29000 T14_MAX_TOKENS=256 $PY $D/run-1014-warm-mtp-bench.py > $D/bench29k-m$m.out 2>&1
  echo "rc=$? bench29k-m$m $(date -u +%T)" >> $D/run.log
done

# 7) K=2 byte-identical @ 29K (512 tokens, warmoff)
flock -w 3600 $LOCK env MLX_MTP=0 T29K_TARGET=29000 T14_MAX_TOKENS=512 T14_WARM=0 $PY $D/run-1014-warm-mtp-bench.py > $D/ar29k512.out 2>&1
echo "rc=$? ar29k512 $(date -u +%T)" >> $D/run.log
flock -w 3600 $LOCK env MLX_MTP=3 T29K_TARGET=29000 T14_MAX_TOKENS=512 T14_WARM=0 $PY $D/run-1014-warm-mtp-bench.py > $D/m3-29k512.out 2>&1
echo "rc=$? m3-29k512 $(date -u +%T)" >> $D/run.log
python3 - <<'EOF' >> $D/k2-29k-gate.log 2>&1
import os
MODELS = os.environ.get("MODELS", "/models")
def toks(p):
    with open(p) as f:
        for line in f:
            if line.startswith("ALLTOKS:"):
                return line.split(None, 1)[1].split()
    return []

ar = toks(f"{MODELS}/tmp/run-1014/ar29k512.out")
m3 = toks(f"{MODELS}/tmp/run-1014/m3-29k512.out")
n = min(len(ar), len(m3))
ok = ar[:n] == m3[:n]
first = next((i for i in range(n) if ar[i] != m3[i]), None)
print(f"K2-29K: ar_len={len(ar)} m3_len={len(m3)} prefix_match={ok} first_diff={first}")
EOF
echo "DONE $(date -u +%FT%TZ)" >> $D/run.log
