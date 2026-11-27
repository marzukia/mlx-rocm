#!/bin/bash
# P2 session (ruling 2026-11-27 14:29Z): their Q4_K dp4a dot (lib-ab3 4738da13)
# in isolation at the SIX trio classes (M=512, the SoT shapes) + corr vs
# refgen2 6/6 BEFORE any timing. Same save/swap/restore + flock pattern as
# p1-session.sh. corr before timing per the ruling.
set -uo pipefail
D=/home/andryo/gemm1-leg2
exec 9>/home/andryo/gpu.lock
flock -w 120 9 || { echo "P2-LOCK-FAIL"; exit 1; }
echo "P2-START $(date -u +%FT%TZ)"
fuser /dev/kfd 2>/dev/null && { echo "P2-KFD-BUSY"; exit 1; }
VP=/home/andryo/venv-hbmfix
SITE=$VP/lib/python3.14/site-packages
LIB=$SITE/mlx/lib/libmlx.so
CORE=$SITE/mlx/core.cpython-314-x86_64-linux-gnu.so
N=/home/andryo/rcp/new724/tree/opt/rocm-7.2.4
cp "$LIB" $D/save-lib.so; cp "$CORE" $D/save-core.so
echo "SAVED lib=$(md5sum $D/save-lib.so | cut -d" " -f1)"
cp "$D/lib-ab3.so" "$LIB"; cp "$D/core-ab3.so" "$CORE"
L=$(md5sum "$LIB" | cut -d" " -f1)
echo "SWAPPED lib=$L"; [ "$L" = "4738da138f1e0d8613fa90da06a73355" ] || { echo "P2-LIB-MD5-FAIL"; exit 1; }
run() {
  local CLS=$1 NN=$2 KK=$3
  echo "P2-RUN-$CLS $(date -u +%FT%TZ)"
  env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/andryo \
    HCC_AMDGPU_TARGET=gfx1201 HIP_VISIBLE_DEVICES=0 MLX_MMQ_CFG=1 \
    P2_CLASS=$CLS P2_M=512 P2_N=$NN P2_K=$KK \
    LD_LIBRARY_PATH=$N/lib:/home/andryo/t2246/libs \
    timeout -k 30 900 $VP/bin/python3 $D/p2-probe.py > $D/p2-$CLS.log 2>&1
  echo "P2-$CLS rc=$? $(date -u +%FT%TZ)"
  grep -aE "P2-$CLS-|abq4k|Error|error" $D/p2-$CLS.log | tail -16
}
run n1024k4096 1024 4096
run n4096k4096 4096 4096
run n4096k12288 4096 12288
run n8192k4096 8192 4096
run n12288k4096 12288 4096
run n248320k4096 248320 4096
cp $D/save-lib.so "$LIB"; cp $D/save-core.so "$CORE"
echo "RESTORED lib=$(md5sum "$LIB" | cut -d" " -f1)"
fuser /dev/kfd 2>/dev/null && echo "P2-KFD-RESIDUE"
flock -u 9
echo "P2-END $(date -u +%FT%TZ)"
