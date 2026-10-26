#!/usr/bin/env bash
# Build the mlx wheel with the ROCm backend for gfx1201 (RX 9070 XT).
# Runs ON marzuki-neon (ROCm 10.1 guest userland; add -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 — see docs/quickstart.md). Clones the pinned vendor (vendor.json) if not present.
set -euo pipefail

VENDOR_DIR="${MLX_VENDOR_DIR:-$HOME/vendor/mlx}"
VENDOR_JSON="$(cd "$(dirname "$0")/.." && pwd)/vendor.json"

# vendor pin: primary first, fallback second (re-pin in a PR when the base moves)
read -r FORK REF <<<"$(python3 - "$VENDOR_JSON" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
print(v["primary"]["fork"], v["primary"]["ref"])
EOF
)"
CMAKE_ARGS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"]["cmake_args"])' "$VENDOR_JSON")"

if [ ! -d "$VENDOR_DIR/.git" ]; then
  git clone --depth 1 --branch "$REF" "$FORK" "$VENDOR_DIR"
fi
cd "$VENDOR_DIR"
git fetch --depth 1 origin "$REF" && git checkout -q FETCH_HEAD

echo "Building wheel from $FORK @ $REF (branch pinned by vendor.json)"
CMAKE_ARGS="$CMAKE_ARGS" python -m pip wheel . --no-deps -w dist/
ls -la dist/
