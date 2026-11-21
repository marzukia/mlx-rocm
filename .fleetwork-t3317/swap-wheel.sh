#!/bin/bash
# swap-wheel.sh <whl-path> <backup-label>
set -eu
SP=/home/andryo/gauntlet-vehicle-venv/lib/python3.14/site-packages
mkdir -p /home/andryo/venv-backups
cp -r "$SP/mlx" "/home/andryo/venv-backups/mlx-$2-$(date -u +%Y%m%dT%H%M)"
/home/andryo/gauntlet-vehicle-venv/bin/pip install -q --force-reinstall "$1"
md5sum "$SP/mlx/lib/libmlx.so"
