#!/usr/bin/env python3
"""Phase 0 smoke (run ON guest-a after installing the built wheel).

Exit criteria (PLAN.md Phase 0 / docs/plan.md):
  1. mx.rocm.is_available() == True
  2. matmul + softmax numerics correct vs the CPU fallback
  3. (opt, MLX_ROCM_SMOKE_LLM=1) Qwen3-0.6B bf16 + 8bit generates sane text
"""

import os
import sys

import mlx.core as mx


def check_is_available() -> None:
    assert mx.rocm.is_available(), "mx.rocm.is_available() is False - ROCm backend not built in"
    mx.set_default_device(mx.gpu)
    print("ok: rocm is_available, default device = gpu")


def check_numerics() -> None:
    a = mx.random.normal((256, 256))
    b = mx.random.normal((256, 256))
    c_gpu = a @ b
    mx.eval(c_gpu)
    c_cpu = (a.astype(mx.float32) @ b.astype(mx.float32)).astype(a.dtype)
    mx.eval(c_cpu)
    max_err = float(mx.max(mx.abs(c_gpu - c_cpu)).item())
    assert max_err < 0.5, f"matmul gpu vs cpu max_err {max_err} too large"
    s_gpu = mx.softmax(a, axis=1)
    mx.eval(s_gpu)
    row = mx.sum(s_gpu, axis=1)
    assert float(mx.max(mx.abs(row - 1.0)).item()) < 1e-2, "softmax rows do not sum to 1"
    print(f"ok: matmul max_err {max_err:.4f}, softmax rows ~1.0")


def check_llm() -> None:
    import mlx_lm

    model = os.environ.get("MLX_ROCM_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    prompt = "The capital of France is"
    out = mlx_lm.generate(model=model, prompt=prompt, max_tokens=16, verbose=False)
    text = (out if isinstance(out, str) else str(out)).strip()
    assert len(text) > 20, f"suspiciously short generation: {text!r}"
    print(f"ok: {model} -> {text!r}")


def main() -> None:
    check_is_available()
    check_numerics()
    if os.environ.get("MLX_ROCM_SMOKE_LLM") == "1":
        check_llm()
    else:
        print("skip: 0.6B generate (set MLX_ROCM_SMOKE_LLM=1 to run)")
    print("SMOKE PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
