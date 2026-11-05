#!/usr/bin/env python3
"""verify-eff class probe (t2199): per-GEMM wall via chained-eval amortization,
streaming floor (t2048-style read kernel), on the COR wheel (venv-verify,
libmlx 8acf13fe). COR env: MLX_ROCM_QMM_M3=1 (wheel-side knob).

No rocprofv3 on this box: all GB/s are wall-clock-measured with the per-op
host overhead amortized by chaining N ops into one graph (measured(wall) label).
25 s idle before each arm; rocm-smi P-state snapshot around each arm.
"""

import os
import subprocess
import time

import mlx.core as mx
import mlx_lm as lm

assert mx.rocm.is_available(), "rocm not available"
MODEL = os.environ.get("MODEL", "/mnt/models/qwen3-9b-mlx")
ROCM_SMI = "/opt/rocm/bin/rocm-smi"
IDLE_S = int(os.environ.get("PROBE_IDLE", "25"))


def pstate(tag):
    try:
        out = subprocess.run(
            [ROCM_SMI, "--showuse", "--showclocks"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        print(f"PS:{tag}:{' | '.join(lines[2:6])}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PS:{tag}:UNAVAILABLE({type(e).__name__})", flush=True)


def idle(tag):
    print(f"IDLE:{tag}:{IDLE_S}s", flush=True)
    time.sleep(IDLE_S)
    pstate(f"{tag}:pre")


def chain_eval(build, iters):
    mx.eval(build())  # warmup inside caller
    t0 = time.perf_counter()
    mx.eval(build())
    return (time.perf_counter() - t0) / iters


def streaming_floor(rounds=3, chunks=50, gib=4.0):
    n = int(gib * (1 << 30)) // 4
    x = mx.ones(n, dtype=mx.float32)
    print(f"FLOOR:buf={n * 4}B chunks={chunks} rounds={rounds}", flush=True)

    def build():
        acc = None
        for _ in range(chunks):
            s = mx.sum(x)
            acc = s if acc is None else acc + s
        return acc

    for _ in range(2):
        mx.eval(build())
    ts = []
    for r in range(rounds):
        t = chain_eval(build, chunks)
        ts.append(t)
        print(f"FLOOR:round{r}={t * 1e3:.3f}ms", flush=True)
    best = min(ts)
    gbs = chunks * gib * (1 << 30) / best / 1e9
    var = max(ts) / min(ts) - 1.0
    print(f"FLOOR:GBS={gbs:.1f} VAR={var * 100:.2f}%", flush=True)
    return gbs, var


def time_gemm(x, lin, rounds=2, iters=100):
    def build():
        acc = None
        for _ in range(iters):
            y = mx.quantized_matmul(
                x,
                lin["weight"],
                scales=lin["scales"],
                biases=lin.get("biases"),
                transpose=True,
                group_size=lin.group_size,
                bits=lin.bits,
                mode=lin.mode,
            )
            acc = y if acc is None else acc + y
        return acc

    for _ in range(2):
        mx.eval(build())
    ts = []
    for _ in range(rounds):
        ts.append(chain_eval(build, iters))
    return ts


def main():
    print(f"CONFIG: model={MODEL} qmm_m3={os.environ.get('MLX_ROCM_QMM_M3')} wheel={mx.__version__}", flush=True)
    model, _ = lm.load(MODEL)
    print("LOAD:OK", flush=True)
    layer = model.model.layers[0]
    gl = layer["mlp"]["gate_proj"]
    dl = layer["mlp"]["down_proj"]
    la = layer["linear_attn"]
    rng = mx.random
    b = mx.bfloat16

    def mkx(k):
        return (rng.uniform(-0.5, 0.5, (3, k))).astype(b)

    x4 = mkx(4096)
    x12 = mkx(12288)

    # Qwen3-Next (GDN linear attention) model: 32 layers.
    shapes = {
        "qkv": (la["in_proj_qkv"], x4),  # 8192 x 4096
        "z": (la["in_proj_z"], x4),  # 4096 x 4096
        "b": (la["in_proj_b"], x4),  # 32 x 4096
        "a": (la["in_proj_a"], x4),  # 32 x 4096
        "o": (la["out_proj"], x4),  # 4096 x 4096
        "gate": (gl, x4),  # 12288 x 4096
        "down": (dl, x12),  # 4096 x 12288
    }

    # Per-matmul bytes (4-bit, group 64, bf16 scale + bias):
    # weight N*K/2 + scale N*K/32 + bias N*K/32 = 9/16*N*K
    def nbytes(n, k):
        return n * k * 9 // 16

    results = {}
    for M in (1, 3):
        for name, (lin, x) in shapes.items():
            arm = f"M{M}-{name}"
            idle(arm)
            xs = x[:M]
            n = lin["weight"].shape[0]
            k = xs.shape[1]
            ts = time_gemm(xs, lin)
            per = sum(ts) / len(ts)
            gb = nbytes(n, k) / per / 1e9
            results[arm] = (per, gb)
            var = max(ts) / min(ts) - 1.0
            print(f"GEMM:{arm} t={per * 1e3:.4f}ms gbs={gb:.1f} var={var * 100:.2f}%", flush=True)
            pstate(f"{arm}:post")

    idle("floor")
    gbs_f, var_f = streaming_floor()

    # class totals per forward (32 layers) and per verify step
    # (1x M=3 + 4x M=1, single weight pass per GEMM)
    L = 32
    nb = {
        "qkv": (8192, 4096),
        "z": (4096, 4096),
        "b": (32, 4096),
        "a": (32, 4096),
        "o": (4096, 4096),
        "gate": (12288, 4096),
        "down": (4096, 12288),
    }
    for M in (1, 3):
        mlp_t = 2 * results[f"M{M}-gate"][0] + results[f"M{M}-down"][0]
        qmv_t = (
            results[f"M{M}-qkv"][0]
            + results[f"M{M}-z"][0]
            + results[f"M{M}-b"][0]
            + results[f"M{M}-a"][0]
            + results[f"M{M}-o"][0]
        )
        mlp_b = L * (2 * nbytes(*nb["gate"]) + nbytes(*nb["down"]))
        qmv_b = L * sum(nbytes(*nb[s]) for s in ("qkv", "z", "b", "a", "o"))
        print(f"CLASS:M{M} mlp: fwd_t={mlp_t * L * 1e3:.2f}ms fwd_gbs={mlp_b / (mlp_t * L) / 1e9:.1f}", flush=True)
        print(f"CLASS:M{M} qmv: fwd_t={qmv_t * L * 1e3:.2f}ms fwd_gbs={qmv_b / (qmv_t * L) / 1e9:.1f}", flush=True)
    # verify step (1x M3 + 4x M1)
    mlp_step = (
        2 * results["M3-gate"][0] + results["M3-down"][0] + 4 * (2 * results["M1-gate"][0] + results["M1-down"][0])
    ) * L
    qmv_step = (
        (results["M3-qkv"][0] + results["M3-z"][0] + results["M3-b"][0] + results["M3-a"][0] + results["M3-o"][0])
        + 4 * (results["M1-qkv"][0] + results["M1-z"][0] + results["M1-b"][0] + results["M1-a"][0] + results["M1-o"][0])
    ) * L
    mlp_b = L * (2 * nbytes(*nb["gate"]) + nbytes(*nb["down"]))
    qmv_b = L * sum(nbytes(*nb[s]) for s in ("qkv", "z", "b", "a", "o"))
    print(
        f"STEP: mlp t={mlp_step * 1e3:.3f}ms gbs={5 * mlp_b / mlp_step / 1e9:.1f} "
        f"pct_floor={5 * mlp_b / mlp_step / 1e9 / gbs_f * 100:.1f}%",
        flush=True,
    )
    print(
        f"STEP: qmv t={qmv_step * 1e3:.3f}ms gbs={5 * qmv_b / qmv_step / 1e9:.1f} "
        f"pct_floor={5 * qmv_b / qmv_step / 1e9 / gbs_f * 100:.1f}%",
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
