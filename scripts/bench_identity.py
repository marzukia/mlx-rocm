#!/usr/bin/env python3
"""Emit the canonical identity block for a benchmark number.

A benchmark number without this block is a claim, not a measurement. A wheel that
cannot be rebuilt, a sidecar whose bytes changed, or an environment that differed by
one dispatch flag has already invalidated comparisons in this repo; the identity block
is the cure. It is deterministic, needs no GPU, and never prints credentials.

Usage:
  uv run python scripts/bench_identity.py --out identity.txt
  uv run python scripts/bench_identity.py --venv .venv --models ./models --bench bench/mtp-bench6.py

Exit codes: 0 = block written, 2 = a required input was missing (the block is still
written, with the missing field marked, so a report can show what was not proven).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ENV_PREFIXES = ("MLX_", "MTP_", "MTP6_", "ROCM_", "HCC_", "HIP_")

# Filesystem identity is captured by path + digest; the path is data, not a constant.
MODEL_SUFFIXES = (".safetensors", ".json", ".npz")


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    """md5 as well as sha256: existing reports in this repo quote md5 prefixes."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def find_libmlx(venv: Path | None) -> Path | None:
    """Only the venv under test counts. A system-wide wheel is deliberately not a fallback:
    a forgotten --venv must produce a MISSING field, never a confidently pinned wrong wheel."""
    if venv is None:
        return None
    candidates = sorted(venv.glob("lib/python*/site-packages/mlx/lib/libmlx*.so"))
    return candidates[-1] if candidates else None


def gpu_identity() -> tuple[list[str], list[str]]:
    """Read the PCI identity of every DRM card. No ROCm tooling required.

    Card indices are probe-order dependent and have already caused a report to
    describe the wrong device, so the identity is keyed on PCI address and device id.
    """
    lines: list[str] = []
    missing: list[str] = []
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return ["gpu: MISSING (no /sys/class/drm)"], ["gpu"]
    for card in sorted(drm.glob("card[0-9]*")):
        if not card.name[4:].isdigit():
            continue
        device = card / "device"
        try:
            pci = device.resolve().as_posix()
            dev_id = (device / "device").read_text().strip()
            vendor = (device / "vendor").read_text().strip()
        except OSError as exc:
            # An unreadable node is a hard failure, not absent data: a device held
            # exclusively returns empty output rather than raising, and a silent skip
            # is how a harness ends up describing the wrong card.
            lines.append(f"gpu: MISSING ({card.name} unreadable: {exc.__class__.__name__})")
            missing.append(f"gpu:{card.name}")
            continue
        lines.append(f"gpu: {card.name} pci={pci} vendor={vendor} device={dev_id}")
    if not lines:
        lines.append("gpu: MISSING (no readable card nodes)")
        missing.append("gpu")
    return lines, missing


def rocm_product_names() -> list[str]:
    candidates = ["/opt/rocm/bin/rocm-smi", "/opt/rocm-7.2.0/bin/rocm-smi"]
    binary = next((c for c in candidates if Path(c).is_file()), None)
    if binary is None:
        return []
    try:
        out = subprocess.run(
            [binary, "--showproductname"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if "Card" in line and "GPU[" in line]


def env_block() -> list[str]:
    keys = sorted(k for k in os.environ if k.startswith(ENV_PREFIXES))
    if not keys:
        return ["env: none set"]
    return [f"env: {k}={os.environ[k]}" for k in keys]


def venv_python(venv: Path | None) -> Path | None:
    if venv is None:
        return None
    candidate = venv / "bin" / "python"
    return candidate if candidate.is_file() else None


def mlx_version_from(venv: Path | None) -> str:
    """Ask the venv's own interpreter, so a host python cannot answer for the wheel."""
    interpreter = venv_python(venv)
    if interpreter is None:
        return "unavailable (no venv interpreter)"
    try:
        out = subprocess.run(
            [str(interpreter), "-c", "import mlx.core as mx; print(getattr(mx, '__version__', 'unknown'))"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable (interpreter failed)"
    version = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    return version or f"unavailable (rc={out.returncode})"


def digest_body(lines: list[str]) -> str:
    body = "\n".join(lines).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def build_block(
    venv: Path | None, models: Path | None, bench: list[Path], include_host: bool = False
) -> tuple[list[str], list[str]]:
    lines = ["# bench identity", "# generated by scripts/bench_identity.py"]
    missing: list[str] = []

    # Reports in this repo are public. A hostname is operational information, so it is
    # redacted unless the caller explicitly opts in for a private record.
    lines.append(f"host: {platform.node()}" if include_host else "host: redacted (--include-host to record)")
    lines.append(f"machine: {platform.machine()} {platform.system()} {platform.release()}")
    lines.append(f"python: {platform.python_version()}")

    libmlx = find_libmlx(venv)
    if libmlx is not None:
        lines.append(f"libmlx: path={libmlx} md5={md5_file(libmlx)} sha256={sha256_file(libmlx)}")
    else:
        lines.append("libmlx: MISSING (no venv given and no system mlx found)")
        missing.append("libmlx")

    lines.append(f"mlx_version: {mlx_version_from(venv)}")

    gpu_lines, gpu_missing = gpu_identity()
    lines.extend(gpu_lines)
    missing.extend(gpu_missing)
    lines.extend(rocm_product_names())

    if models is None:
        lines.append("model: NOT REQUESTED (no --models; provenance incomplete)")
        missing.append("models")
    elif models.is_dir():
        digested = sorted(p for p in models.rglob("*") if p.suffix in MODEL_SUFFIXES)
        for path in digested:
            lines.append(f"model: path={path} md5={md5_file(path)} sha256={sha256_file(path)}")
        if not digested:
            # A directory with nothing digest-able is as unproven as a missing one.
            lines.append(f"model: MISSING (no {'/'.join(MODEL_SUFFIXES)} files under {models})")
            missing.append("models")
    else:
        lines.append(f"model: MISSING ({models})")
        missing.append("models")

    for path in bench:
        if path.is_file():
            lines.append(f"bench: path={path} md5={md5_file(path)} sha256={sha256_file(path)}")
        else:
            lines.append(f"bench: MISSING ({path})")
            missing.append(f"bench:{path}")

    lines.extend(env_block())
    lines.append(f"identity_digest: {digest_body(lines)}")
    return lines, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the canonical bench identity block.")
    parser.add_argument("--venv", type=Path, default=None, help="venv root holding the mlx wheel")
    parser.add_argument("--models", type=Path, default=None, help="model directory to digest")
    parser.add_argument("--bench", type=Path, nargs="*", default=[], help="bench file(s) to digest")
    parser.add_argument("--out", type=Path, default=None, help="write the block here instead of stdout")
    parser.add_argument(
        "--include-host",
        action="store_true",
        help="record the hostname; redacted by default because reports in this repo are public",
    )
    args = parser.parse_args(argv)

    lines, missing = build_block(args.venv, args.models, args.bench, include_host=args.include_host)
    text = "\n".join(lines) + "\n"

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(lines)} lines)")
    else:
        sys.stdout.write(text)

    for field in missing:
        print(f"identity-missing: {field}", file=sys.stderr)
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
