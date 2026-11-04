#!/bin/bash
# t1858 kfix wheel build: graft the patched qmv_rows_msg_kernel.hip onto the
# r4v-source vendor tree (vendor dir = $1, default /tmp/vendor-kfix6) and build
# the wheel into venv-kfix.
set -uo pipefail
VENDOR=${1:-/tmp/vendor-kfix6}
export TMPDIR=/tmp
export LD_PRELOAD=libhsa-runtime64.so.1
export HCC_AMDGPU_TARGET=gfx1201
echo "BUILD_START vendor=$VENDOR $(date -u)"
rm -rf ~/venv-kfix
cp -al ~/venv-r4v ~/venv-kfix
echo "VENV_COPY_DONE $(date -u)"
mkdir -p /tmp/kfix-wheel
cd "$VENDOR"
rm -rf build dist
CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201"
CMAKE_ARGS="$CMAKE_ARGS -DBLA_VENDOR=OpenBLAS"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0"
export CMAKE_ARGS
~/venv-kfix/bin/python -m pip wheel . --no-deps \
  -w /tmp/kfix-wheel/ > /tmp/kfix-wheel/build.log 2>&1
echo "WHEEL_EXIT=$? $(date -u)"
ls -la /tmp/kfix-wheel/
if ls /tmp/kfix-wheel/mlx-*.whl 2>/dev/null; then
  ~/venv-kfix/bin/python -m pip install \
    --force-reinstall --no-deps /tmp/kfix-wheel/mlx-*.whl 2>&1 | tail -3
  echo "WHEEL_INSTALLED $(date -u)"
  echo "--- md5 (kfix must differ; r4v must be 8acf13fec6fc1636026d8e1dc69cd497):"
  md5sum ~/venv-kfix/lib/python3.14/site-packages/mlx/lib/libmlx.so
  md5sum ~/venv-r4v/lib/python3.14/site-packages/mlx/lib/libmlx.so
  ~/venv-kfix/bin/pip show mlx 2>/dev/null | head -2
  echo "BUILD_DONE $(date -u)"
else
  echo "NO_WHEEL"
fi
