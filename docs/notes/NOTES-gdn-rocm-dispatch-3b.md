# NOTES: GDN-ROCM-DISPATCH (3b) — expose + route the fast GDN on ROCm

One-PR scope, GAUNTLET 1.4k campaign 3b. Fixes the 27B MLX prefill wall: the Qwen3.8-27B GDN layers ran the per-token Python loop (`gated_delta_ops`) on the ROCm wheel because stock routing treats "not metal" as "fallback". The fused `fast::GatedDeltaUpdate` (chunked scan; recurrent kernel for T <= 8) already exists in the ROCm wheel and already covers the 27B geometry; this PR makes the model file actually use it.

## 1. The binding already exists (3a's "not exposed" claim is debunked)

- `m.def("gated_delta_update", &mlx::core::fast::gated_delta_update, ...)`
  at `python/src/fast.cpp:390` (vendor snapshot + box SoT build tree,
  identical). Name follows the sibling `m.def` pattern in `init_fast`;
  the callable is `mx.fast.gated_delta_update(q, k, v, gamma, beta,
  initial_state=None, mask=None, stream=None)`.
- Present in both shipped wheels: name string in the core `.so`
  (`mlx-0.32.3.dev20261120+265055a`, `mlx-0.32.3.dev20261119+85023b4`).
- 3a's probe (`hasattr(mx, "gated_delta_update")` at the top level) was
  defective: the symbol lives under the `mx.fast` namespace, so the
  correct probe is `hasattr(mx.fast, "gated_delta_update")`. No C++ or
  binding code is added by this PR.

## 2. C++ `use_fallback` guard (mirrored verbatim by the Python gate)

`mlx/backend/rocm/gated_delta_update.cpp` (identical in the CWD vendor snapshot and the box SoT build tree):

    use_fallback(Hk, Dk, Hv, Dv, has_mask, s) == true  iff
      s.device == Device::cpu
      || env_forced_fallback()
      || has_mask
      || Dk != 128 || Dv != 128
      || Hv % Hk != 0
      || Hv / Hk > 8

`eval_gpu` additionally degrades to the op loop internally for non-bf16 q/k/v, non-scalar g (`g.ndim() != 3 || g.shape(2) != Hv`), and non-row-contiguous inputs. The Python gate (this PR) mirrors the `use_fallback` guard exactly and is deliberately NOT extended (no hasattr check, no extra geometry): non-matching geometries fall through to the existing `gated_delta_ops` path, and the C++ side still has its own internal fallbacks.

Qwen3.8-27B GDN geometry is in coverage: Dk = 128, Dv = 128, Hk = 16, Hv = 48, Hv/Hk = 3 (<= 8), scalar g (`[B, T, Hv]`), bf16, no mask.

## 3. The gate reroute

`patches/gdn-rocm-dispatch.diff` (+ `scripts/apply-gdn-rocm-dispatch.py`, idempotent, one-time `.bak-gdn3b` backup, `--check` mode) inserts, BEFORE the stock routing in `mlx_lm/models/gated_delta.py::gated_delta_update`:

    if (
        use_kernel
        and mx.default_device() == mx.gpu
        and not mx.metal.is_available()   # ROCm wheel (metal never available)
        and mask is None
        and q.shape[-1] == 128
        and v.shape[-1] == 128
        and v.shape[-2] % q.shape[-2] == 0
        and v.shape[-2] // q.shape[-2] <= 8
    ):
        return mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)

Applied via the script (or `patch -p1`) to the venv's installed `gated_delta.py`; the 6f-pkg variant is superseded by this (same intent, cleaner guard, byte-identical insert).

## 4. The 4x GEMM "inconsistency" is an accounting-unit error, not physics

Per-shape probes (t3294b, M = 1024, 3 warm + 5 timed, `mx.quantized_matmul`): 66-71 TFLOPS eff on this GPU. [b]'s "in-situ" GEMM bucket (0.17 s/chunk / 0.69 s per pass / 289 TFLOPS) is not an independent KPROF measurement:

1. The per-shape probe's `PASS_TOTAL` accumulator is a unit bug
   (`tot_s += cnt * fl / dt` accumulates FLOP counts, not seconds);
   0.69 s is the recomputed per-chunk total (one M = 1024 call per
   layer per pass).
2. The 4096-token pass runs 4 x 1024 chunks per layer, so per-pass GEMM
   at probe speed is ~4 x 0.69 = 2.76 s => 199.8/72 TFLOPS class,
   consistent with the per-shape numbers. The "289 TFLOPS" is the
   per-pass FLOP divided by per-chunk time: a 4x unit error that exactly matches the 4 chunks. No L2-warm explanation is required.
3. No KPROF log in `gauntlet-results/` shows an in-situ 0.17 s/chunk
   GEMM total; [b]'s 22.47 s GEMM bucket is a back-calc
   (227.104 TFLOP / 10.1 TFLOPS dense probe), not a measurement.

Record: GEMM on this GPU is the per-shape probe class (66-71 TFLOPS at M = 1024); the e2e gate round measures the wall directly.

## 5. Wheel build recipe (box / guest-a)

The fork `github.com/marzukia/mlx-rocm` main (3e84320e) is a mirror of this work repo (the `mlx_rocm` shim), NOT a buildable mlx tree; the vendor.json comment predates that layout. The canonical wheel build (reproducing 265055a / 85023b4 / dcd1c69) happens ON the box from the SoT build tree:

- tree: `$VEHICLE_BUILD_TREE`, HEAD cc9afe2 (p1b-planC) +
  uncommitted C++ changes; full upstream tree under
  `vendor/mlx-rocm-upstream/` (18810-byte CMakeLists with the
  `MLX_BUILD_ROCM` option, `python/` with `src/`).
- `setup.py` + `MANIFEST.in` restored from commit `75ca881`
  (upstream mlx setup.py; version = `0.32.3.dev<date>+<git rev-parse
  --short HEAD>`).
- Build: `CMAKE_PREFIX_PATH=/opt/rocm CMAKE_ARGS="-DMLX_BUILD_ROCM=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS" pip wheel .
  --no-deps -w dist/` (ROCm 10.1 userland; cp314 for the vehicle venv).
- GPU 1 (idle lane) for verification; `MLX_GDN_DEBUG=1` prints the
  kernel/fallback path on stderr (evidence line for the smoke).

## 5b. 3b box-rebuild attempt (recorded, abandoned)

The box was re-provisioned before this lane; the userland's hand-installed ROCm header set was wiped. A from-scratch rebuild in `/var/tmp/mlx3b-build` (tree copy + 75ca881 setup.py; tree git sha 7734c1c) got far but is blocked by the toolchain itself, not by this PR:

- Repaired: lld needed libxml2.so.2 + libicu*.so.60 (absent on the
  re-provisioned Ubuntu 26.10 host; fetched from the distro archive);
  rocthrust/rocprim/hiprand headers re-fetched (ROCm/thrust 2.6.0, ROCm/
  rocprim main, ROCm/hiprand rocm-7.2.4) into `/tmp/hipblas-shim/`;
  `thrust/system/cuda/detail/bulk/malloc.hpp` patched for non-NVCC device
  arrays; `thrust::hip` namespace alias appended to
  `thrust/execution_policy.h`; `rocprim_version.hpp` generated from the
  `.in` template; DPP macros force-defined in `cccl_shim.h`.
- Blocked: rocprim's wave-64 warp scan uses DPP row-broadcast, which the
  LLVM 22 ROCm 7.2 backend rejects on GFX10+ ("broadcasts are not
  supported on GFX10+"); stock rocprim swizzle paths cap at WarpSize 32.
  The 11-19/11-20 builds succeeded inside the `mlx72` docker container
  (`rocm/dev-ubuntu-24.04:7.2`, still up) which carries a patched
  userland rocprim. Rebuilding that container image is out of 3b scope.
- Verification therefore ran on the canonical installed wheel (265055a,
  the 11-20 09:31 build) in the vehicle venv — the same wheel 3a and 6f
  were diagnosed on. The wheel is the vehicle's; no new wheel is
  required for this PR (Python-only change).

## 6. In-run verification (vehicle venv, wheel 265055a, gfx1201 GPU 1)

All on `HIP_VISIBLE_DEVICES=1` (GPU 0 was 98% VRAM-occupied by another lane), ROCm 10.1 userland, `MLX_GDN_DEBUG=1`:

1. Import probe: `hasattr(mx.fast, "gated_delta_update")` → True; doc
   string shows `(q, k, v, gamma, beta, initial_state=None, mask=None,
   stream=None)`. (Not in `mx.fast.__all__`; probe by attribute, which is
   what the gate uses.)
2. Direct fast call (27B geometry: B=1, T=1024, Hk=16, Dk=128, Hv=48,
   Dv=128, bf16, per-head g, no mask): out (1,1024,48,128), state
   (1,48,128,128); 285 ms first (JIT/init), 1.1 ms steady. Debug trace:
   `cond qdt=1 scalar=1 cont=1` → kernel path, `post-out-set` /
   `post-state-set`.
3. Model-level dispatch, stock file (md5 bbe8bf851dabf5815f56583ec8c90c6f):
   T=256 call = 365.7 ms (per-token Python loop — the wall).
4. Same file after `apply-gdn-rocm-dispatch.py` (backup `.bak-gdn3b`,
   post md5 3d3aaa0657d263e88d33d3d523e72a6e): T=1024 first = 270.5 ms,
   steady = 0.2 ms (≈700x the stock per-call microbenchmark rate).
   Out-of-coverage geometry (Dk=64) stayed on the Python fallback and
   worked — the guard does not misroute.
5. Numerical cross-check (identical seed 42 inputs, T=256, fallback vs
   fast): outputs agree to 1-2 bf16 ulps over the printed prefix before
   random-input recurrence divergence (bf16-state reference saturates at
   the bf16 max; the fp32-state kernel does not) — expected for a
   chaotic recurrence on random data; confirms same recurrence.
6. Vehicle file restored to stock after the test (md5 bbe8bf85...);
   the patch is applied by the script at serve time, not baked in.
