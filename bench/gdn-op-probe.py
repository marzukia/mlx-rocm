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
  WALL    - N sandwiched calls: host launch + kernel execution (serial
            cost, upper bound of the op's in-pipeline cost).
  PIPE24  - 24 calls queued then ONE force: the per-op cost when 24
            layers are pipelined in one forward (host launch overlaps
            the previous op's kernel).

Force semantics: venv-canon wheel quirk — mx.eval() does NOT force the
ROCm queue; the first .item()/argmax does (the r4b finding "mx.eval is
a side effect"). force() sums the outputs and reads them (sum+item is
~0.5 ms, dwarfed by the 25-sample average and by the kernel time
being measured; the input random fills are pre-evaluated in the
warmup, so the timed sections only carry the op + its outputs).
"""

import time

import mlx.core as mx

bf = mx.bfloat16

KEY_DIM = 2048  # 16 k heads x 128
VAL_DIM = 4096  # 32 v heads x 128
CONV_DIM = 8192
HV = 32


def mk(T):
    # timing probe: global-PRNG values are fine (no correctness
    # requirement); the shapes/dtypes are the per-layer shapes of the
    # qwen3-9b GDN layers.
    return dict(
        q=mx.random.uniform(-1.0, 1.0, (1, T, KEY_DIM)).astype(bf),
        k=mx.random.uniform(-1.0, 1.0, (1, T, KEY_DIM)).astype(bf),
        v=mx.random.uniform(-1.0, 1.0, (1, T, VAL_DIM)).astype(bf),
        b=mx.random.uniform(-1.0, 1.0, (1, T, HV)).astype(bf),
        a=mx.random.uniform(-1.0, 1.0, (1, T, HV)).astype(bf),
        A_log=mx.random.uniform(-3.0, 0.0, (HV,)),
        dt_bias=mx.random.uniform(-3.0, 0.0, (HV,)),
        conv_state=mx.random.uniform(-1.0, 1.0, (3, CONV_DIM)).astype(bf),
        conv_weight=mx.random.uniform(-0.1, 0.1, (CONV_DIM, 4)).astype(bf),
        z=mx.random.uniform(-1.0, 1.0, (1, T, HV, 128)).astype(bf),
        state=mx.random.uniform(-0.01, 0.01, (HV, 128, 128)),
        nw=mx.random.uniform(-1.0, 1.0, (128,)).astype(bf),
    )


def call(op, x):
    return op(
        x["q"], x["k"], x["v"], x["b"], x["a"], x["A_log"], x["dt_bias"],
        x["conv_state"], x["conv_weight"], x["z"], x["state"], x["nw"],
        inv_scale=128**-0.5, prep_eps=1e-6, norm_eps=1e-6,
    )


def force(r):
    # Read the outputs (sum -> item) so the stream-ordered queue drains
    # (forcing the last submitted output completes everything before it
    # on the in-order stream).
    if isinstance(r, (list, tuple)):
        for a in r:
            mx.sum(a).item()
    elif r is not None:
        mx.sum(r).item()


def bench(tag, op, T, nhost=50, nwall=25):
    x = mk(T)
    force(call(op, x))  # warmup (allocation, codegen)
    for _ in range(5):
        force(call(op, x))

    t0 = time.perf_counter()
    last = None
    for _ in range(nhost):
        last = call(op, x)
    host = (time.perf_counter() - t0) / nhost * 1000.0
    force(last)

    t0 = time.perf_counter()
    for _ in range(nwall):
        force(call(op, x))
    wall = (time.perf_counter() - t0) / nwall * 1000.0

    t0 = time.perf_counter()
    last = None
    for _ in range(24):
        last = call(op, x)
    force(last)
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
