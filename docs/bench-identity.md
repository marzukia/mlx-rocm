# Bench identity — every number carries its provenance

A benchmark number on its own is a claim. This repo has already paid for that lesson: a headline result was quoted against a wheel that later could not be found on the machine, a sidecar format differed between two arms compared as if equal, and a clock sampler in the harness was reading a different device than the one doing the work. None of those were measurement errors in the arithmetic. All three were missing provenance.

`scripts/bench_identity.py` emits the block that closes the gap. It is deterministic, needs no GPU, and prints no credentials.

## Usage

Run it beside every benchmark and store the block next to the number it belongs to.

```
uv run python scripts/bench_identity.py --venv .venv --models ./models --bench bench/mtp-bench6.py --out identity.txt
```

Fields:

- `--venv`: the venv root whose wheel is under test. The venv's own interpreter is asked for the MLX version, so a host interpreter cannot answer on the wheel's behalf.
- `--models`: a model directory. Every `.safetensors`, `.json` and `.npz` file under it is digested, so a sidecar that changed format between two runs cannot pass as the same input.
- `--bench`: one or more bench files to digest. A bench edited between arms is a different experiment.
- `--out`: write to a file. Without it the block goes to stdout.

Exit code 0 means the block is complete. Exit code 2 means a requested input was missing; the block is still written, with each missing field marked `MISSING`, and the missing names are listed on stderr. A report may legitimately quote a number with a missing field, but it must say which field was not proven.

## What the block contains

- Host and machine, so an architecture difference cannot hide in a comparison.
- The wheel: path, md5 and sha256. Both digests are emitted because existing reports in this repo quote md5 prefixes; sha256 is there for anything that needs collision resistance.
- The MLX version as reported by the venv's interpreter.
- GPU identity by PCI address and vendor/device id, plus the marketing names when ROCm tooling is present.
- Model and bench file digests.
- Every set environment variable whose name starts with `MLX_`, `MTP_`, `MTP6_`, `ROCM_`, `HCC_` or `HIP_`.
- A digest over the block itself, so two runs can be compared with one equality test.

## Why PCI identity and not a DRM card index

DRM card indices follow probe order and are not stable across boots or display-configuration changes. A harness that hardcodes an index can silently sample a different device; that has already happened here, and the read is quiet rather than loud because an unreadable device returns empty output instead of failing.

So the block records where each card sits on the bus. When a device must be named to a tool, name it by PCI address or by the vendor/device pair, and treat an unreadable node as a hard failure rather than as absent data.

## What the block does not prove

It pins inputs, not conditions. It cannot show that two arms ran at the same temperature, the same memory clock, or without contention from another process. Those still need to be controlled in the harness, and stated in the report.

It also cannot show that a number is reproducible. It shows only that a later run used the same inputs, which is the precondition for asking whether the number reproduces.