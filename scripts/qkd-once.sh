#!/bin/bash
# t2050 qkd follow-up one-off (on the box, after the main session):
# model-tensor bit-identity (fixed discovery path) + single-op path-proof
# timing (K = 8192 N = 4096, stock vs MLX_ROCM_QGD_KSPLIT=1, same venv).
# flock ~/gpu.lock, bounded retry 20x30s; 25 s idle before the measures.
set -u
H=$HOME
MNT=/mnt
export MODELS="$MNT/models"
HSALIB=$(ldconfig -p 2>/dev/null | grep 'hsa-runtime64' | head -1 | awk '{print $NF}')
export LD_PRELOAD=$HSALIB
export HCC_AMDGPU_TARGET=gfx1201
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib
D=$H/qkd
L=$D/once.log
PY=$H/venv-ksplit/bin/python
mkdir -p $D
> $L
echo "=== QKD ONCE START $(date -u) ===" | tee -a $L
sclk() { m=$(grep "*" /sys/class/drm/card1/device/pp_dpm_sclk 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1); if [ -z "$m" ]; then m=$(timeout 15 rocm-smi --showclocks 2>/dev/null | grep -i "sclk\|GPU Clock" | head -1); fi; echo "$m"; }
exec 9>$H/gpu.lock
LOCKED=0
for i in $(seq 1 20); do
  if flock -n 9; then LOCKED=1; break; fi
  echo "lock-wait $i/20 $(date -u +%H:%M:%SZ)" | tee -a $L
  sleep 30
done
[ "$LOCKED" = "1" ] || { echo "PENDING-LOCK" | tee -a $L; exit 3; }
echo "=== LOCK ACQUIRED $(date -u +%H:%M:%SZ) ===" | tee -a $L
sleep 25
echo "pre-once sclk=[$(sclk)]" | tee -a $L
CEBC="MTP_MODEL=$MODELS/qwen3-9b-mlx"
timeout 1800 env MODE=stock $CEBC $PY $D/qkd-ksplit-bitcheck.py > $D/once-stock.jsonl 2> $D/once-stock.err
echo "ONCE-STOCK-rc=$? lines=$(grep -c '^{' $D/once-stock.jsonl)" | tee -a $L
timeout 1800 env MODE=ksplit MLX_ROCM_QGD_KSPLIT=1 REF_FILE=$D/once-stock.jsonl $CEBC $PY $D/qkd-ksplit-bitcheck.py > $D/once-ksplit.jsonl 2> $D/once-ksplit.err
echo "ONCE-KSPLIT-rc=$? lines=$(grep -c '^{' $D/once-ksplit.jsonl)" | tee -a $L
timeout 600 env MODE=compare REF_FILE=$D/once-stock.jsonl NEW_FILE=$D/once-ksplit.jsonl $PY $D/qkd-ksplit-bitcheck.py > $D/once-cmp.log 2>&1
echo "ONCE-CMP-rc=$?" | tee -a $L
grep -E '^(COMPARE|FALLBACK|COMPARE_DONE|OPS_DONE)' $D/once-cmp.log >> $L 2>&1
grep -E '^(OPS_DONE|WARN)' $D/once-*.err $D/once-*.jsonl >> $L 2>&1
# Path proof: one op, K = 8192 N = 4096 (ksplit gate passes: K/4 = 2048),
# stock vs ksplit, 20 warmup + 100 timed reps, same venv, back to back.
cat > $D/once-time.py <<'EOF'
import time
import mlx.core as mx

mx.random.seed(42)
K, N = 8192, 4096
x = mx.random.uniform(-1.0, 1.0, (1, K)).astype(mx.bfloat16)
w = mx.random.uniform(-1.0, 1.0, (N, K))
wq, s, b = mx.quantize(w, 64, 4)
mx.eval(wq, s, b)


def f():
    return mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4)


for _ in range(20):
    mx.eval(f())
ts = []
for _ in range(100):
    t0 = time.perf_counter()
    mx.eval(f())
    ts.append((time.perf_counter() - t0) * 1000.0)
ts.sort()
print("TIME_MS mean=%.4f p50=%.4f min=%.4f" % (sum(ts) / len(ts), ts[len(ts) // 2], ts[0]))
EOF
timeout 600 env $CEBC $PY $D/once-time.py > $D/once-time-stock.log 2>&1
echo "ONCE-TIME-STOCK-rc=$? $(cat $D/once-time-stock.log)" | tee -a $L
sleep 25
timeout 600 env $CEBC MLX_ROCM_QGD_KSPLIT=1 $PY $D/once-time.py > $D/once-time-ksplit.log 2>&1
echo "ONCE-TIME-KSPLIT-rc=$? $(cat $D/once-time-ksplit.log)" | tee -a $L
echo "=== QKD ONCE DONE $(date -u) ===" | tee -a $L
echo DONE > $D/.once-done
exit 0
