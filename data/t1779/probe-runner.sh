#!/bin/bash
set -u
D=<guest-home>/t1779
V=<guest-home>/venv-r4v/bin/python
mkdir -p $D
md5sum <models-mount>/qwen3-9b-mlx/model-mtp.safetensors > $D/sidecar.log 2>&1
run() { # $1 tag, $2 gd, $3 script, $4 extra env (empty ok)
  local tag=$1 gd=$2 sc=$3 extra=$4
  echo "[$(date -u +%H:%M:%S)] start $tag" >> $D/identity.txt
  sleep 25
  env MODELS=<models-mount> HCC_AMDGPU_TARGET=gfx1201 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
    MLX_ROCM_QMV_ROWS=1 MLX_ROCM_REFWD_M1=1 MLX_ROCM_GDN_FUSE=1 \
    MLX_MTP=0 PROBE_N=24 MLX_ROCM_GRAPH_DECODE=$gd $extra \
    $V $sc > $D/$tag.log 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag rc=$?" >> $D/identity.txt
}
run PA 0 <guest-home>/t1779-probe-ssm.py ""
run PG 1 <guest-home>/t1779-probe-ssm.py ""
run PB 1 <guest-home>/t1779-probe-idx.py "MLX_G1Y_DEBUG=1"
md5sum <models-mount>/qwen3-9b-mlx/model-mtp.safetensors >> $D/sidecar.log 2>&1
echo "ALLDONE $(date -u +%H:%M:%S)" >> $D/identity.txt
