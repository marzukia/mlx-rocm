#!/usr/bin/env bash
# Build the mlx wheel with the ROCm backend for gfx1201 (RX 9070 XT).
# Runs ON the build VM (ROCm 10.1 guest userland; add -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 — see docs/quickstart.md). Clones the pinned vendor (vendor.json) if not present.
set -euo pipefail

VENDOR_DIR="${MLX_VENDOR_DIR:-$HOME/vendor/mlx}"
VENDOR_JSON="$(cd "$(dirname "$0")/.." && pwd)/vendor.json"

# vendor pin: primary first, fallback second (re-pin in a PR when the base moves)
read -r FORK REF PIN_SHA <<<"$(python3 - "$VENDOR_JSON" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
print(v["primary"]["fork"], v["primary"]["ref"], v["primary"].get("sha", ""))
EOF
)"
CMAKE_ARGS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"]["cmake_args"])' "$VENDOR_JSON")"

if [ ! -d "$VENDOR_DIR/.git" ]; then
  git clone --depth 1 --branch "$REF" "$FORK" "$VENDOR_DIR"
fi
cd "$VENDOR_DIR"
git fetch --depth 1 origin "$REF" && git checkout -q FETCH_HEAD

# Fail loudly if the checked-out commit is not the vendor.json pin (an earlier run: the pin's
# ref and sha can disagree when the pinned sha was never pushed to the fork).
if [ -n "$PIN_SHA" ]; then
  CHECKED_SHA="$(git rev-parse HEAD)"
  PIN_FULL_SHA="$(git rev-parse --verify --quiet "${PIN_SHA}^{commit}" || true)"
  if [ -z "$PIN_FULL_SHA" ]; then
    echo "ERROR: vendor.json pin sha=$PIN_SHA not resolvable in $FORK (fetched ref $REF)" >&2
    exit 1
  fi
  if [ "$CHECKED_SHA" != "$PIN_FULL_SHA" ]; then
    echo "ERROR: vendor pin mismatch: vendor.json sha=$PIN_SHA ($PIN_FULL_SHA) but checkout is $CHECKED_SHA (ref $REF)" >&2
    exit 1
  fi
  echo "Vendor pin verified: $FORK @ $REF = $PIN_FULL_SHA (vendor.json sha=$PIN_SHA)"
fi

echo "Building wheel from $FORK @ $REF (branch pinned by vendor.json, sha verified)"
CMAKE_ARGS="$CMAKE_ARGS" python -m pip wheel . --no-deps -w dist/
ls -la dist/
