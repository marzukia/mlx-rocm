#!/bin/bash
# S4(1) 3s host sampler (the 11-25 retained class, t3418 method): vRSS +
# O_TMPFILE map count + shmem + VRAM. 11-27 sampler-grep-bug fix: the VRAM
# line is read by extracting the number AFTER "Memory (B):" (the old grep
# returned the device index "0"). O_TMPFILE maps appear as /dev/zero
# (deleted) (t3418 pool-grep fix); memfd names also counted.
# Usage: s41-hostsample.sh <pgrep-pattern> <outlog> <max-samples>
# Runs while the target process is alive (and up to <max-samples> samples).
set -u
PAT=$(cat "$1")
OUT=$2; MAX=${3:-90}
SMI=/opt/rocm/bin/rocm-smi
echo "SAMPLE-START $(date -u +%FT%TZ) pat=$PAT max=$MAX" > "$OUT"
n=0
while [ "$n" -lt "$MAX" ]; do
  n=$((n + 1))
  t=$(date -u +%s)
  pid=$(pgrep -f "$PAT" | head -1)
  rsskb=0; maps=0
  if [ -n "$pid" ] && [ -r "/proc/$pid/maps" ]; then
    rsskb=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    maps=$(grep -cE "/dev/zero \(deleted\)|memfd" "/proc/$pid/maps" 2>/dev/null || true)
  fi
  shmem=$(awk '/^Shmem:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
  # 11-27 fix: extract the number after "Memory (B):" (do not grep the
  # device-index column).
  vram=$("$SMI" --showmeminfo vram 2>/dev/null | grep "Used Memory (B)" \
    | sed -n 's/.*Used Memory (B): \([0-9][0-9]*\).*/\1/p' | head -1)
  echo "$t vRSS_KB=${rsskb:-0} otmpfile_maps=${maps:-0} shmem_KB=${shmem:-0} vram_B=${vram:-0}" >> "$OUT"
  sleep 3
done
echo "SAMPLE-END $(date -u +%FT%TZ) n=$n" >> "$OUT"
