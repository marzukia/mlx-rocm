# PLAN — mlx-rocm

Living plan. Full research report: [docs/plan.md](docs/plan.md) (an earlier research pass, 2026-10-25).

## Definition of minimum

qmlx-serve runs a 9B-class dense Qwen3 on the build VM (RX 9070 XT / gfx1201, ROCm 10.1),
correctly, at usable decode speed. MTP, disk KV, 27B+/122B-hybrid = Phase 2.

**ROCm version (single source of truth, 2026-10-26):** the verified guest toolchain on
the build VM is **ROCm 10.1** (core at `/opt/rocm/core-10.1`,
`CMAKE_PREFIX_PATH=/opt/rocm/core-10.1`, `amdrocm-*-dev10.1` packages; see
docs/quickstart.md). Nothing else is verified: 6.4 was the early probe validation and
7.7 the original target — both are historical, not the verified toolchain.

## Current state (2026-10-25)

- Research done (an earlier research pass): the backend is ~90% done in upstream PR an open community MLX ROCm backend PR
  (head the vendor ref); newer forward-merge the vendor ref.
  Effort: 2.5-5 person-weeks for the minimum (Phase 0+1).
- the build VM is UP (RX 9070 XT; guest userland ROCm 10.1 — the only verified toolchain).
  submitter is building the MVP on the the build VM guest VM (smoke unblocked, relayed to submitter).
- Repo bootstrapped: hermetic CI (lint/format/pytest/size gate), vendor pin, build +
  smoke scripts. GPU validation still to be done on the build VM.

## Phases

### Phase 0 — Adopt + build + smoke (0.5-1 person-week)
1. Clone the pinned primary (vendor.json), build the wheel:
   `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS"`.
2. On the build VM: `mx.rocm.is_available()` true, matmul/softmax correct, Qwen3-0.6B (bf16 + 8bit)
   generates sane text. (`scripts/smoke.py`)
- **Exit criteria:** is_available()==True; numerics correct; 0.6B sane.
- **Gate:** if primary (the vendor ref) does not build clean on ROCm 10.1, fall back to
  the vendor ref and record why (docs/plan.md open question 1).

### Phase 1 — Validate gfx1201 + qmlx-serve minimum (1-2 person-weeks)
1. QMM 4bit on gfx1201 (Rdna4 tier + WMMA dense qmm) vs CPU numerics.
2. SDPA decode (vector) + prefill (WMMA flash); verify prefill uses the WMMA path.
3. fp8 GEMM: test the known gfx1201 fp8 prefill bug; ship `MLX_ROCM_GEMM_BF16=1` as the
   default until proven clean (docs/plan.md).
4. Allocator on a discrete GPU: host_shadow / pinned host_stage path (not APU fine-grained).
5. Run qmlx-serve with a 9B-class dense Qwen3; check the `is_metal` error-classification
   path in qmlx `vllm_mlx/engine_core.py:675` handles ROCm/hip runtime errors.
- **Exit criteria:** 9B Qwen3 served on gfx1201; no NaNs; no OOM→SEGV single-process.
  (`scripts/smoke-serve.sh`)

### Phase 2 — Full qMLX stack (deferred; ~2-4 person-weeks)
GatedDeltaNet (122B hybrid), MoE (A10B/27B+), MTP spec decode (blocked on the M3
stream-TLS fix), disk KV cache validation. Details in docs/plan.md §Phase 2.

## Open questions (operator's call)

1. Base: the vendor ref (preferred) vs the vendor ref (fallback) — resolved by Phase 0 build.
2. Wheel vs source: source-on-the build VM for now; PyPI wheel (`mlx-rocm` name slot already held
   by us) is a Phase 2+ concern.
3. fp8 default: `MLX_ROCM_GEMM_BF16=1` until the gfx1201 fp8 prefill is proven clean on 10.1.
4. MTP: Phase 2, strictly (M3 stream-TLS fix blocks it either way).
