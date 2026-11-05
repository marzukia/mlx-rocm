#!/usr/bin/env python3
"""Ceiling report, item 2: verify the mlp 'dequant/latency-bound' diagnosis.

One process, one model load, three same-session arms on the mlp class
(24 layers x {gate,up: N=12288 K=4096; down: K=12288 N=4096}, M=3):

  A  stock 4-bit mr3 (MLX_ROCM_QMM_M3=1)
  B  same GEMMs, weights pre-dequantized to bf16 in HBM (4x weight
     traffic, no in-loop dequant, no scale/bias gather)
  C  stock 4-bit mr3 pointed at ONE shared ~80 MB weight set (layer 0's
     buffers), L2-hot after a warm pass (isolates ALU+gather: weights
     hit L2; scales+biases ~0.75 MB also L2-hot)
  C2 stock (RESTORED) weights, constant scale/bias (isolates the
     per-group scale-gather ALU+traffic from the per-weight dequant)

Plus context classes from the t2031 probe: qmv_gdn (5 GEMMs) and the
pure gdn_fused tape (in_proj GEMMs precomputed and reused), to split the
4.33 ms class into GEMM vs tape components.

Run under flock, venv-m3k. 25 s idle before each measurement section; an
external SCLK sampler runs in the background. Per-layer = median of 5
(2 warm); class = sum of 24 per-layer medians.
"""

import os
import time

import mlx.core as mx
import mlx.nn as mxnn
import mlx_lm as mlxlm

AR = int(os.environ.get("CEIL_AR", "64"))


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2]


def pause(tag):
    print(f"PAUSE {tag} 25s idle", flush=True)
    time.sleep(25)
    print(f"RESUME {tag}", flush=True)


print("CONFIG mlp-3arm", flush=True)
model, tokenizer = mlxlm.load(os.environ["CEIL_MODEL"], adapter=None)
hf_config = model.config
assert hf_config.n_layers == 24
assert hf_config.n_heads == 32

prompt = "Write a function that multiplies all the numbers in a list."
tokens = tokenizer.encode(prompt, add_bos=True)
model.tokens = mx.array(tokens)
for _ in range(AR):
    y, _ = model.generate(model.tokens, max_tokens=1, verbose=False)
    model.tokens = y
print(f"AR_DONE ar={AR}", flush=True)

layers = [ly for ly in model.layers if isinstance(ly, mxnn.TransformerBlock)]
assert len(layers) == 24
print("LAYERS_OK 24", flush=True)


def get_hidden():
    mx.random.seed(42)
    h3 = mx.random.normal((3, 4096))
    h4 = mx.random.normal((3, 4096))
    mx.eval(h3, h4)
    return h3, h4


# ---- arm B: pre-dequantized bf16 weights -------------------------------
print("DEQUANT start", flush=True)
b16 = []
for ly in layers:
    b16.append(
        [
            mx.dequantize(
                ly.mlp.gate.weight,
                ly.mlp.gate.scales,
                ly.mlp.gate.biases,
                bits=4,
                group_size=64,
            ),
            mx.dequantize(
                ly.mlp.up.weight,
                ly.mlp.up.scales,
                ly.mlp.up.biases,
                bits=4,
                group_size=64,
            ),
            mx.dequantize(
                ly.mlp.down.weight,
                ly.mlp.down.scales,
                ly.mlp.down.biases,
                bits=4,
                group_size=64,
            ),
        ]
    )
    mx.eval(b16[-1])
print("DEQUANT done", flush=True)


def mlp_layer_a(ly, xa):
    return ly.mlp(xa)


def mlp_layer_b(ly, xa, wb):
    g = mx.matmul(xa, wb[0].T)
    u = mx.matmul(xa, wb[1].T)
    su = g * mx.sigmoid(g) * u
    return mx.matmul(su, wb[2].T)


# ---- arm C setup (applied AFTER arm A is measured) ----------------------
P2I = {"gate": 0, "up": 1, "down": 2}
shared = []
stock = []
for p in ("gate", "up", "down"):
    q = getattr(layers[0].mlp, p)
    shared.append([q.weight, q.scales, q.biases])
    for ly in layers:
        qq = getattr(ly.mlp, p)
        stock.append([qq.weight, qq.scales, qq.biases])


def swap_c():
    for ly in layers:
        for p in ("gate", "up", "down"):
            q = getattr(ly.mlp, p)
            s = shared[P2I[p]]
            q.weight = s[0]
            q.scales = s[1]
            if s[2] is not None:
                q.biases = s[2]


def swap_c2():
    # restore the STOCK weights first (C left them shared), then
    # constant scale/bias: isolates the per-group scale-gather
    # ALU+traffic from the per-weight dequant.
    i = 0
    for p in ("gate", "up", "down"):
        q0 = getattr(layers[0].mlp, p)
        ones = mx.ones_like(q0.scales)
        mx.eval(ones)
        z = mx.zeros_like(q0.biases) if q0.biases is not None else None
        if z is not None:
            mx.eval(z)
        for ly in layers:
            q = getattr(ly.mlp, p)
            o = stock[i]
            i += 1
            q.weight = o[0]
            q.scales = ones
            q.biases = z


def mlp_layer_c(ly, xa):
    return ly.mlp(xa)


# ---- per-layer class timing ---------------------------------------------
def time_class(fn, tag):
    per_layer = []
    for ly in layers:
        xa, _ = get_hidden()
        for _ in range(2):
            mx.eval(fn(ly, xa))
        walls = []
        for _ in range(5):
            t0 = time.perf_counter()
            o = fn(ly, xa)
            mx.eval(o)
            walls.append(time.perf_counter() - t0)
        per_layer.append(median(walls))
    total = sum(per_layer)
    print(
        f"CLASS {tag} total={total * 1e3:.4f} ms per_layer_ms={[f'{w * 1e3:.4f}' for w in per_layer]}",
        flush=True,
    )
    return total


# ---- context classes (probe definitions) --------------------------------
def qmv_gdn_layer(ly, xb, x):
    g = ly.gated_delta_net
    mx.eval(g.in_proj_qkv(x))
    mx.eval(g.in_proj_z(x))
    mx.eval(g.in_proj_b(xb))
    mx.eval(g.in_proj_a(xb))
    mx.eval(g.out_proj(xb))


def gdn_fused_layer(ly, xa):
    xb = mxnn.rms_norm(xa, m=ly.norm_eps)
    mx.eval(xb)
    g = ly.gated_delta_net
    q, k, v = mx.split(mx.eval(g.in_proj_qkv(xb)), 3)
    b = mx.eval(g.in_proj_b(xb))
    a = mx.eval(g.in_proj_a(xb))
    z = mx.eval(g.in_proj_z(xb))
    cs, hs = g.conv1d.state, g.gated_deltanet.state
    cs = mx.take_along_axis(cs, (0 + mx.arange(3) + 1) % 3, axis=0)
    c = mx.eval(g.conv1d(mx.concat([cs, xb])))
    c = mx.take(c, 3)
    return mx.fast.gdn_fused_step_tape(q, k, v, b, a, z, cs, hs, g.conv1d.weight, g.conv1d.bias, 3)


print("=== ARM A: stock 4-bit mr3 ===", flush=True)
pause("A")
tA = time_class(lambda ly, x: mlp_layer_a(ly, x), "mlp_A")

# per-GEMM medians (one layer), same x for all three
pause("A-gemm")
xA0, _ = get_hidden()
ga = []
for nm in ("gate", "up", "down"):
    ws = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(getattr(layers[0].mlp, nm)(xA0))
        ws.append(time.perf_counter() - t0)
    ga.append(median(ws))
print(
    f"MLP_A_GEMM_ms gate={ga[0] * 1e3:.4f} up={ga[1] * 1e3:.4f} down={ga[2] * 1e3:.4f}",
    flush=True,
)

print("=== ARM B: bf16 pre-dequantized ===", flush=True)
pause("B")
per_layer_B = []
for i, ly in enumerate(layers):
    xa, _ = get_hidden()
    for _ in range(2):
        mx.eval(mlp_layer_b(ly, xa, b16[i]))
    walls = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(mlp_layer_b(ly, xa, b16[i]))
        walls.append(time.perf_counter() - t0)
    per_layer_B.append(median(walls))
tB = sum(per_layer_B)
print(
    f"CLASS mlp_B total={tB * 1e3:.4f} ms per_layer_ms={[f'{w * 1e3:.4f}' for w in per_layer_B]}",
    flush=True,
)

print("=== ARM C: stock mr3, shared 80 MB weight set ===", flush=True)
swap_c()
print("SWAP_C_DONE", flush=True)
# warm the shared buffers (one full pass)
xa, _ = get_hidden()
for ly in layers:
    mx.eval(mlp_layer_c(ly, xa))
print("WARM_C done", flush=True)
pause("C")
tC = time_class(lambda ly, x: mlp_layer_c(ly, x), "mlp_C")

print("=== ARM C2: stock weights, constant scale/bias ===", flush=True)
swap_c2()
print("SWAP_C2_DONE", flush=True)
# re-warm (scale values changed)
for ly in layers:
    mx.eval(mlp_layer_c(ly, xa))
pause("C2")
tC2 = time_class(lambda ly, x: mlp_layer_c(ly, x), "mlp_C2")

print("=== context: qmv_gdn + gdn_fused ===", flush=True)
pause("ctx")
gdn_walls = []
for ly in layers:
    xb, xa = get_hidden()
    for _ in range(2):
        qmv_gdn_layer(ly, xb, xa)
    ws = []
    for _ in range(5):
        t0 = time.perf_counter()
        qmv_gdn_layer(ly, xb, xa)
        ws.append(time.perf_counter() - t0)
    gdn_walls.append(median(ws))
tG = sum(gdn_walls)
print(f"CLASS qmv_gdn total={tG * 1e3:.4f} ms", flush=True)

gf_walls = []
for ly in layers:
    xa, _ = get_hidden()
    for _ in range(2):
        gdn_fused_layer(ly, xa)
    ws = []
    for _ in range(5):
        t0 = time.perf_counter()
        gdn_fused_layer(ly, xa)
        ws.append(time.perf_counter() - t0)
    gf_walls.append(median(ws))
print(f"CLASS gdn_fused_full total={sum(gf_walls) * 1e3:.4f} ms", flush=True)

# pure fused tape: precompute qkv/b/a/z/conv inputs, time only the tape
tape_walls = []
for ly in layers:
    x = mx.random.normal((3, 4096))
    mx.eval(x)
    g = ly.gated_delta_net
    q, k, v = mx.split(mx.eval(g.in_proj_qkv(x)), 3)
    b = mx.eval(g.in_proj_b(x))
    a = mx.eval(g.in_proj_a(x))
    z = mx.eval(g.in_proj_z(x))
    cs = g.conv1d.state
    cs = mx.take_along_axis(cs, (0 + mx.arange(3) + 1) % 3, axis=0)
    c = mx.eval(g.conv1d(mx.concat([cs, x])))
    c = mx.take(c, 3)
    hs = g.gated_deltanet.state
    for _ in range(2):
        mx.fast.gdn_fused_step_tape(q, k, v, b, a, z, c, hs, g.conv1d.weight, g.conv1d.bias, 3)
    ws = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.fast.gdn_fused_step_tape(q, k, v, b, a, z, c, hs, g.conv1d.weight, g.conv1d.bias, 3)
        ws.append(time.perf_counter() - t0)
    tape_walls.append(median(ws))
print(
    f"CLASS gdn_fused_tape_only total={sum(tape_walls) * 1e3:.4f} ms "
    f"per_layer_ms={[f'{w * 1e3:.4f}' for w in tape_walls]}",
    flush=True,
)

print(
    f"SUMMARY mlp_A={tA * 1e3:.4f} mlp_B={tB * 1e3:.4f} "
    f"mlp_C={tC * 1e3:.4f} mlp_C2={tC2 * 1e3:.4f} "
    f"qmv_gdn={tG * 1e3:.4f} "
    f"gf_full={sum(gf_walls) * 1e3:.4f} gf_tape={sum(tape_walls) * 1e3:.4f}",
    flush=True,
)
print("DONE", flush=True)
