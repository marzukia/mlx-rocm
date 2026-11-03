#!/usr/bin/env python3
"""GDN per-op cost probe (GDN launch-merge probe, t1585 follow-up).

Measures the per-op cost of the single-launch GDN ops (gdn_fused_step,
gdn_fused_step_tape) at the per-layer shapes of the qwen3-9b hybrid
(key_dim 2048, value_dim 4096, conv_dim 8192, Hv 32, Dk=Dv=128), for
T=1 (draft/correction), T=2, T=4 (verify), with per-layer synthetic
inputs (random bf16/f32, correct shapes) in a fresh process:

  HOST    - N async calls, no sync: host-side launch + arg staging per op
            (the cost that launch merging targets; the kernel runs
            out-of-band).
  WALL    - N eval-sandwiched calls: host launch + kernel execution
            (serial cost, upper bound of the op's in-pipeline cost).
  PIPE24  - 24 calls queued then ONE eval: the actual per-op cost when
            24 layers are pipelined in one forward (host launch overlaps
            the previous op's kernel).

All three, for both T=1 and T=4 (T=2 for completeness), tape + non-tape
op, fresh process, after a warmup burn-in.
"""

import os
import time

import mlx.core as mx

bf = mx.bfloat16
f32 = mx.float32

KEY_DIM = 2048  # 16 k heads x 128
VAL_DIM = 4096  # 32 v heads x 128
CONV_DIM = 8192
HV = 32


def mk(T):
    g = mx.random.state(1234)
    return dict(
        q=mx.random.uniform(-1.0, 1.0, (1, T, KEY_DIM), g).astype(bf),
        k=mx.random.uniform(-1.0, 1.0, (1, T, KEY_DIM), g).astype(bf),
        v=mx.random.uniform(-1.0, 1.0, (1, T, VAL_DIM), g).astype(bf),
        b=mx.random.uniform(-1.0, 1.0, (1, T, HV), g).astype(bf),
        a=mx.random.uniform(-1.0, 1.0, (1, T, HV), g).astype(bf),
        A_log=mx.random.uniform(-3.0, 0.0, (HV,)),
        dt_bias=mx.random.uniform(-3.0, 0.0, (HV,)),
        conv_state=mx.random.uniform(-1.0, 1.0, (3, CONV_DIM), g).astype(bf),
        conv_weight=mx.random.uniform(-0.1, 0.1, (CONV_DIM, 4), g).astype(bf),
        z=mx.random.uniform(-1.0, 1.0, (1, T, HV, 128), g).astype(bf),
        state=mx.random.uniform(-0.01, 0.01, (HV, 128, 128)),
        nw=mx.random.uniform(-1.0, 1.0, (128,), g).astype(bf),
    )


def call(op, x, T):
    return op(
        x["q"], x["k"], x["v"], x["b"], x["a"], x["A_log"], x["dt_bias"],
        x["conv_state"], x["conv_weight"], x["z"], x["state"], x["nw"],
        inv_scale=128**-0.5, prep_eps=1e-6, norm_eps=1e-6,
    )


def bench(tag, op, T, nhost=50, nwall=25):
    x = mk(T)
    call(op, x, T)  # warmup (allocation, codegen)
    mx.eval()
    for _ in range(5):
        call(op, x, T)
    mx.eval()

    t0 = time.perf_counter()
    for _ in range(nhost):
        call(op, x, T)
    host = (time.perf_counter() - t0) / nhost * 1000.0
    mx.eval()

    t0 = time.perf_counter()
    for _ in range(nwall):
        mx.eval()
        t1 = time.perf_counter()
        call(op, x, T)
        mx.eval()
        t2 = time.perf_counter()
        del t1, t2
    wall = (time.perf_counter() - t0) / nwall * 1000.0

    t0 = time.perf_counter()
    for _ in range(24):
        call(op, x, T)
    mx.eval()
    pipe24 = (time.perf_counter() - t0) / 24 * 1000.0

    print(
        "OP %s T=%d: HOST=%.3fms WALL=%.3fms PIPE24=%.3fms" % (tag, T, host, wall, pipe24),
        flush=True,
    )


def main():
    tape = getattr(mx.fast, "gdn_fused_step_tape", None)
    step = getattr(mx.fast, "gdn_fused_step", None)
    print("HAVE tape=%s step=%s" % (tape is not None, step is not None), flush=True)
    if step is not None:
        for T in (1, 2, 4):
            bench("gdn_fused_step", step, T)
    if tape is not None:
        for T in (1, 2, 4):
            bench("gdn_fused_step_tape", tape, T)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
