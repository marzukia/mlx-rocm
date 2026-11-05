#!/bin/bash
# Post-report checks for REPORT-verify-eff.md (ticket step 5).
set -u
cd "$(git rev-parse --show-toplevel)"
echo "== ruff format =="
uv run ruff format --check . && echo OK
echo "== ruff check =="
uv run ruff check . && echo OK
echo "== line wrap (longest non-table line) =="
awk 'length > 76 && $0 !~ /^\|/ && $0 !~ /^```/' REPORT-verify-eff.md | awk '{print length, $0}' | sort -rn | head -3
echo "== persona/secret leak scan =="
grep -nE "jimmy|flemming|Tailscale|tailnet|100\.78\.|10\.129\.|flemming@" REPORT-verify-eff.md bench/probe_verify_eff.py && echo "LEAK FOUND" || echo "CLEAN"
echo "== box IP/hostname scan in report =="
grep -nE "100\.|neon|guest-a" REPORT-verify-eff.md || echo "CLEAN"
echo "== done =="
