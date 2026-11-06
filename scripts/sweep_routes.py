#!/usr/bin/env python3
"""QMV route env-sweep (verify M=3/M=4) — no code change, COR wheel 8acf13fe.

Arms (wheel-in): default, lx (MLX_QMV_LX=1), nomr (MLX_QMV_NO_MR=1).
Arms (NOT in wheel 8acf13fe — byte-grep 0 hits for the string): rows
(MLX_ROCM_QMV_ROWS / _MIN_N / _HOIST) -> reported 'not in wheel'.

Because the QMV dispatch knobs are function-local `static` (read once on the
first QuantizedMatmul dispatch per process), each (shape, M, arm, round) is a
FRESH subprocess with its arm env set in the process environment. The master
loads the model ONCE, extracts + saves the 7 GEMM tensors via mx.save
(round-trip verified, bf16/uint32 preserved), then 25 s idle, floor#1,
round-robin-over-arms matrix (3 rounds, arms cycled per shape — not arm-blocks),
bit-exact (rows-exact) check, floor#2.

GB/s = (N*K*9//16 bytes = 1.125 x weight) / per-op wall (100-chain, one mx.eval).
measured(wall). P-state: rocm-smi snapshot per arm (no SCLK sampler on box).
"""

import os
import subprocess
import sys
import time

MODEL = "<models-mount>/qwen3-9b-mlx"
WEIGHTS = "/tmp/sw"  # directory of per-tensor .mx files
ROCM_SMI = "/opt/rocm/bin/rocm-smi"
SELF = os.path.abspath(__file__)
GS, BITS, MODE = 64, 4, "affine"  # uniform across all shapes (config.json)

ARM_ENV = {
    "default": {},
    "lx": {"MLX_QMV_LX": "1"},
    "nomr": {"MLX_QMV_NO_MR": "1"},
}
ARMS = ["default", "lx", "nomr"]

# N x K (stored weight shape, transpose=True)
SHAPE_NK = {
    "gate": (12288, 4096),
    "up": (12288, 4096),
    "down": (4096, 12288),
    "qkv": (8192, 4096),
    "z": (4096, 4096),
    "o": (4096, 4096),
    "lmh": (248320, 4096),
}
SHAPES = list(SHAPE_NK)

EXTRACT = {
    "gate": lambda m: m.model.layers[0]["mlp"]["gate_proj"],
    "up": lambda m: m.model.layers[0]["mlp"]["up_proj"],
    "down": lambda m: m.model.layers[0]["mlp"]["down_proj"],
    "qkv": lambda m: m.model.layers[0]["linear_attn"]["in_proj_qkv"],
    "z": lambda m: m.model.layers[0]["linear_attn"]["in_proj_z"],
    "o": lambda m: m.model.layers[0]["linear_attn"]["out_proj"],
    "lmh": lambda m: m["language_model"]["lm_head"],
}


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


def streaming_floor(rounds=3, chunks=50, gib=4.0):
    import mlx.core as mx

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
        t0 = time.perf_counter()
        mx.eval(build())
        ts.append(time.perf_counter() - t0)
        print(f"FLOOR:round{r}={ts[-1] * 1e3:.3f}ms", flush=True)
    best = min(ts)
    gbs = chunks * gib * (1 << 30) / best / 1e9
    var = max(ts) / min(ts) - 1.0
    print(f"FLOOR:GBS={gbs:.1f} VAR={var * 100:.2f}%", flush=True)
    return gbs, var


def nbytes(n, k):
    return n * k * 9 // 16


def _load_tensors(shape):
    import mlx.core as mx

    w = mx.load(f"{WEIGHTS}/{shape}_w.npy")
    s = mx.load(f"{WEIGHTS}/{shape}_s.npy")
    bi = mx.load(f"{WEIGHTS}/{shape}_b.npy")
    biases = None if bi.shape[0] == 0 else bi
    return w, s, biases


def _op(x, w, s, biases):
    import mlx.core as mx

    return mx.quantized_matmul(
        x,
        w,
        scales=s,
        biases=biases,
        transpose=True,
        group_size=GS,
        bits=BITS,
        mode=MODE,
    )


def worker_time(shape, M, arm, rnd):
    import mlx.core as mx

    w, s, biases = _load_tensors(shape)
    n, k = SHAPE_NK[shape]
    x = mx.random.uniform(-0.5, 0.5, (M, k)).astype(mx.bfloat16)

    def build(n):
        acc = None
        for _ in range(n):
            y = _op(x, w, s, biases)
            acc = y if acc is None else acc + y
        return acc

    # ---- burst: the reference 100-chain (one ~12ms eval) ----
    mx.eval(build(100))  # warmup 1
    mx.eval(build(100))  # warmup 2
    prime_s = float(os.environ.get("SWEEP_PRIME", "1.0"))
    tp = time.perf_counter()
    while time.perf_counter() - tp < prime_s:
        mx.eval(build(100))
    t0 = time.perf_counter()
    mx.eval(build(100))  # 100-chain, one eval
    per_b = (time.perf_counter() - t0) / 100
    gb_b = nbytes(n, k) / per_b / 1e9

    # ---- extended: ONE continuous eval of ~ext_s (no gaps) to test whether a
    # sustained load ramps the memory clock above the burst value. This is the
    # discriminator: if gb_ext >> gb_burst, the small-M 'collapse' is largely a
    # P-state (clock) artifact; if gb_ext ~ gb_burst, it is a real kernel limit
    # (few rows cannot saturate HBM regardless of clock).
    ext_s = float(os.environ.get("SWEEP_EXT", "0.7"))
    n_ext = max(200, min(5000, int(ext_s / per_b)))
    mx.eval(build(n_ext))  # warmup (ramps the clock for the sustained read)
    pstate(f"ext:{arm}:{shape}:M{M}")
    t1 = time.perf_counter()
    mx.eval(build(n_ext))  # single continuous eval, ~ext_s
    per_e = (time.perf_counter() - t1) / n_ext
    gb_e = nbytes(n, k) / per_e / 1e9
    pstate(f"time:{arm}:{shape}:M{M}:r{rnd}")
    print(
        f"GEMM:arm={arm} shape={shape} M={M} round={rnd} N={n} K={k} "
        f"tb={per_b * 1e6:.1f}us gb_burst={gb_b:.1f} ne={n_ext} "
        f"te={per_e * 1e6:.1f}us gb_ext={gb_e:.1f}",
        flush=True,
    )


def worker_bitexact(shape, arm):
    import mlx.core as mx
    import numpy as np

    w, s, biases = _load_tensors(shape)
    n, k = SHAPE_NK[shape]
    x4 = mx.random.uniform(-0.5, 0.5, (4, k)).astype(mx.bfloat16)
    mx.eval(x4)
    for M in (3, 4):
        x = x4[:M]
        batched = _op(x, w, s, biases)
        mx.eval(batched)  # [M, N]
        tot_diff = 0
        for r in range(M):
            single = _op(x[r : r + 1], w, s, biases)
            mx.eval(single)  # [1, N]
            cnt = mx.sum(mx.not_equal(batched[r], single[0]))
            tot_diff += int(np.asarray(cnt))
        same = tot_diff == 0
        print(
            f"BITEQ:arm={arm} shape={shape} M={M} N={n} K={k} identical={'YES' if same else 'NO'} ndiff={tot_diff}",
            flush=True,
        )


def run_worker(argv, arm):
    base = os.environ.copy()
    base.update(ARM_ENV[arm])
    subprocess.run([sys.executable, SELF, "--worker"] + argv, env=base, check=True)


def main_master():
    import mlx.core as mx
    import mlx_lm as lm
    import numpy as np

    assert mx.rocm.is_available(), "rocm not available"
    rounds = int(os.environ.get("SWEEP_ROUNDS", "3"))
    idle_s = int(os.environ.get("SWEEP_IDLE", "25"))
    sel_shapes = [s for s in (os.environ.get("SWEEP_SHAPES") or ",".join(SHAPES)).split(",") if s in SHAPE_NK]
    sel_arms = [a for a in (os.environ.get("SWEEP_ARMS") or ",".join(ARMS)).split(",") if a in ARM_ENV]
    print(
        f"WHEEL: mlx={mx.__version__} qmm_m3={os.environ.get('MLX_ROCM_QMM_M3')} "
        f"rounds={rounds} shapes={sel_shapes} arms={sel_arms}",
        flush=True,
    )

    # ---- post-launch verify: process up + model loaded before any arm ----
    model, _ = lm.load(MODEL)
    print("MODEL_LOADED:OK", flush=True)

    # ---- extract + save tensors (per-file mx.save preserves bf16/uint32) ----
    import os as _os

    _os.makedirs(WEIGHTS, exist_ok=True)
    for name in SHAPES:
        lin = EXTRACT[name](model)
        mx.save(f"{WEIGHTS}/{name}_w.npy", lin["weight"])
        mx.save(f"{WEIGHTS}/{name}_s.npy", lin["scales"])
        bi = lin.get("biases")
        mx.save(f"{WEIGHTS}/{name}_b.npy", bi if bi is not None else mx.zeros((0,)))
        print(
            f"EXTRACT:{name} wshape={tuple(lin['weight'].shape)} "
            f"gs={lin.group_size} bits={lin.bits} mode={lin.mode} "
            f"bias={'Y' if bi is not None else 'N'}",
            flush=True,
        )
    print(f"SAVED:{WEIGHTS}", flush=True)

    # ---- round-trip verify (reload, op, compare to source, bit-exact) ----
    lin = EXTRACT["gate"](model)
    x0 = mx.random.uniform(-0.5, 0.5, (3, 4096)).astype(mx.bfloat16)
    mx.eval(x0)
    ref = _op(x0, lin["weight"], lin["scales"], lin.get("biases"))
    mx.eval(ref)
    rw = mx.load(f"{WEIGHTS}/gate_w.npy")
    rs = mx.load(f"{WEIGHTS}/gate_s.npy")
    rb = mx.load(f"{WEIGHTS}/gate_b.npy")
    rbi = None if rb.shape[0] == 0 else rb
    re = _op(x0, rw, rs, rbi)
    mx.eval(re)
    nd = int(np.asarray(mx.sum(mx.not_equal(ref, re))))
    print(f"ROUNDTRIP:ndiff={nd} {'OK' if nd == 0 else 'FAIL'}", flush=True)
    if nd != 0:
        print("ROUNDTRIP:FAIL - abort (fall back to per-arm model load)", flush=True)
        sys.exit(3)

    # ---- 25 s idle before first arm ----
    print(f"IDLE:{idle_s}s", flush=True)
    time.sleep(idle_s)
    pstate("pre-matrix")

    # ---- floor #1 (start of window) ----
    print("=== FLOOR#1 (start) ===", flush=True)
    streaming_floor(rounds=3)

    # ---- timing matrix: rounds, round-robin over arms (not arm-blocks) ----
    for rnd in range(rounds):
        for shape in sel_shapes:
            for M in (3, 4):
                for arm in sel_arms:
                    run_worker(
                        ["--role", "time", "--shape", shape, "--m", str(M), "--arm", arm, "--round", str(rnd)],
                        arm,
                    )

    # ---- bit-exact (rows-exact) check, wheel-in arms ----
    for arm in sel_arms:
        for shape in sel_shapes:
            run_worker(
                ["--role", "biteq", "--shape", shape, "--arm", arm],
                arm,
            )

    # ---- floor #2 (end of window) ----
    print("=== FLOOR#2 (end) ===", flush=True)
    streaming_floor(rounds=3)
    print("DONE", flush=True)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        role = sys.argv[sys.argv.index("--role") + 1]
        shape = sys.argv[sys.argv.index("--shape") + 1]
        arm = sys.argv[sys.argv.index("--arm") + 1]
        if role == "time":
            m = int(sys.argv[sys.argv.index("--m") + 1])
            rnd = int(sys.argv[sys.argv.index("--round") + 1])
            worker_time(shape, m, arm, rnd)
        elif role == "biteq":
            worker_bitexact(shape, arm)
    else:
        main_master()


if __name__ == "__main__":
    main()
