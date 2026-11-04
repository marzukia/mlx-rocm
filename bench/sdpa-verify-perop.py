#!/usr/bin/env python3
"""SDPA/bandwidth probe (t1858 ticket #2): one S=4 verify forward, per-op class
timing + isolated SDPA microbench at the model's real attention shapes, on the
wheel under test (venv chosen at runtime).

Extends bench/roofline-verify-perop.py (t1858 salvage) with:
  - per-attention-layer wall (median already in base)
  - isolated SDPA microbench: qL=4 (verify), qL=1 (decode), qL=5 (prefill/WMMA)
    at the real (H=16, Hkv=4, D=256, kL=ctx) with the cached KV, plus the
    same calls with MLX_ROCM_SDPA_S4VA toggled (env read per process, so the
    toggled variant is a separate process run: see run-probe.sh)
  - effective bandwidth vs the 546 GB/s HBM floor

Env: MTP_MODEL (model dir), MTP_AR_N, MTP6_REPS. The wheel's own env gates
(MLX_ROCM_GDN_TAPE etc.) are set as defaults below, matching the bench env.
"""

import json
import os
import time

import mlx.core as mx
from mlx_lm import load

for k, v in [
    ("MLX_ROCM_GDN_TAPE", "1"),
    ("MLX_ROCM_GDN_FUSE", "1"),
    ("MLX_ROCM_QMV_ROWS", "1"),
    ("MLX_ROCM_QMV_ROWS_MIN_N", "1"),
    ("MLX_ROCM_REFWD_M1", "1"),
]:
    os.environ.setdefault(k, v)

MODEL = os.environ.get("MTP_MODEL", "model dir on box")
AR_N = int(os.environ.get("MTP_AR_N", "512"))
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
EOS = 248044
REPS = int(os.environ.get("MTP6_REPS", "5"))
HBM_FLOOR_GBPS = 546.0


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2]


def save_state(cache):
    snap = []
    for c in cache:
        if "advance" in dir(c) and not hasattr(c, "keys"):
            snap.append((c, "gdn", c[0], c[1], c.lengths))
        else:
            snap.append((c, "kv", c.keys, c.values, c.offset))
    return snap


def restore_state(cache, snap):
    for c, kind, a, b, off in snap:
        if kind == "gdn":
            c[0] = a
            c[1] = b
            c.lengths = off
        else:
            c.keys = a
            c.values = b
            c.offset = off


def main():
    import mlx_lm.models.qwen3_5 as _m5

    t0 = time.time()
    model, tok = load(MODEL)
    lm = model.language_model
    t_load = time.time() - t0
    layers = lm.model.layers
    gdn_layers = [ly for ly in layers if ly.is_linear]
    attn_layers = [ly for ly in layers if not ly.is_linear]
    n_l, n_g, n_a = len(layers), len(gdn_layers), len(attn_layers)
    print(f"LOAD_S={t_load:.2f} NL={n_l} NG={n_g} NA={n_a} S4VA_ENV={os.environ.get('MLX_ROCM_SDPA_S4VA', 'unset')}")

    ga0 = gdn_layers[0].linear_attn
    p0 = ga0.in_proj_qkv.weight
    print("SHAPE_CHECK qkv.weight", tuple(p0.shape), "scales", tuple(ga0.in_proj_qkv.scales.shape))

    ids = tok.encode(PROMPT)
    cache = lm.make_cache()
    _, h = lm(mx.array([ids]), cache=cache, return_hidden=True, return_prenorm=True)
    mx.eval(h)
    ctx = len(ids)

    _last = ids[-1]
    for _ in range(AR_N):
        logits, h = lm(
            mx.array([[_last]]),
            cache=cache,
            return_hidden=True,
            return_prenorm=True,
        )
        _last = int(mx.argmax(logits[0, -1]))
        if _last == EOS:
            break
    mx.eval(h)
    ctx += AR_N
    print(f"CTX={ctx} LAST={_last} PRE_S={time.time() - t0:.1f}")

    # (1) Fused S=4 verify wall.
    def verify4_wall():
        toks = [_last, 11, 13, 29]
        snap = save_state(cache)
        t0 = time.perf_counter()
        logits, _ = lm(mx.array([toks]), cache=cache, return_hidden=True)
        mx.eval(logits)
        dt = time.perf_counter() - t0
        restore_state(cache, snap)
        return dt

    for _ in range(2):  # warm
        verify4_wall()
    fused = [verify4_wall() * 1e3 for _ in range(REPS)]
    fused_ms = median(fused)
    print(f"FUSED_VERIFY_MS={fused_ms:.2f} all={[f'{x:.2f}' for x in fused]}")
    tape0 = getattr(_m5, "GDN_TAPE_CALLS", -1)

    _, h4 = lm(
        mx.array([[_last, 11, 13, 29]]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    mx.eval(h4)

    classes = {}

    def class_wall(name, fn, reps=3):
        vals = []
        for _ in range(reps):
            mx.eval(h4)
            t0 = time.perf_counter()
            fn()
            mx.eval()
            vals.append(time.perf_counter() - t0)
        classes[name] = median(vals) * 1e3

    def one_gdn_projs(i):
        ly = gdn_layers[i]
        xa = ly.input_layernorm(h4)
        ga = ly.linear_attn
        outs = (
            ga.in_proj_qkv(xa),
            ga.in_proj_z(xa),
            ga.in_proj_b(xa),
            ga.in_proj_a(xa),
        )
        mx.eval(*outs)
        out = ga.out_proj(outs[1].reshape(1, 4, -1))
        mx.eval(out)

    class_wall("qmv_gdn", lambda: one_gdn_projs(0))

    def gdn_fused_layer(i):
        ly = gdn_layers[i]
        xa = ly.input_layernorm(h4)
        ga = ly.linear_attn
        c = cache[[j for j, ll in enumerate(layers) if ll is ly][0]]
        qkv = ga.in_proj_qkv(xa)
        b = ga.in_proj_b(xa)
        a = ga.in_proj_a(xa)
        z = ga.in_proj_z(xa).reshape(1, 4, ga.num_v_heads, ga.head_v_dim)
        conv_state = (
            c[0]
            if c[0] is not None
            else mx.zeros(
                (1, ga.conv_kernel_size - 1, ga.conv_dim),
                dtype=h4.dtype,
            )
        )
        t0 = time.perf_counter()
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if c.lengths is not None:
            ends = mx.clip(c.lengths, 0, 4)
            positions = (ends[:, None] + mx.arange(ga.conv_kernel_size - 1))[..., None]
            c[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            c[0] = mx.contiguous(conv_input[:, -(ga.conv_kernel_size - 1) :, :])
        out, state, st_tape, cs_tape = mx.fast.gdn_fused_step_tape(
            qkv[:, :, : ga.key_dim],
            qkv[:, :, ga.key_dim : 2 * ga.key_dim],
            qkv[:, :, 2 * ga.key_dim :],
            b,
            a,
            ga.A_log,
            ga.dt_bias,
            conv_state.reshape(ga.conv_kernel_size - 1, ga.conv_dim),
            ga.conv1d.weight.reshape(ga.conv_dim, ga.conv_kernel_size),
            z,
            c[1],
            ga.norm.weight,
            inv_scale=ga.head_k_dim**-0.5,
            prep_eps=1e-6,
            norm_eps=ga.layer_norm_epsilon,
        )
        c[1] = state
        c.advance(4)
        mx.eval(out, state)
        return time.perf_counter() - t0

    gdn_fused_ms_list = []
    for _ in range(3):
        vals = []
        snap = save_state(cache)
        for i in range(len(gdn_layers)):
            mx.eval(h4)
            dt = gdn_fused_layer(i)
            vals.append(dt)
        restore_state(cache, snap)
        gdn_fused_ms_list.append(median(vals) * 1e3)
    classes["gdn_fused"] = median(gdn_fused_ms_list)

    # (c) Attention sub-block per layer, with per-layer walls recorded.
    attn_per_layer_ms = []

    def attn_layer(i, record=False):
        ly = attn_layers[i]
        xa = ly.input_layernorm(h4)
        c = cache[[j for j, ll in enumerate(layers) if ll is ly][0]]
        t0 = time.perf_counter()
        o = ly.self_attn(xa, None, c)
        mx.eval(o)
        dt = time.perf_counter() - t0
        if record:
            attn_per_layer_ms.append(dt * 1e3)
        return dt

    attn_ms_list = []
    for _ in range(3):
        vals = []
        snap = save_state(cache)
        for i in range(len(attn_layers)):
            mx.eval(h4)
            vals.append(attn_layer(i, record=(_ == 2)))
        restore_state(cache, snap)
        attn_ms_list.append(median(vals) * 1e3)
    classes["attn"] = median(attn_ms_list)

    # (d) MLP per layer (norm outside).
    def mlp_layer(i):
        ly = layers[i]
        xa = ly.post_attention_layernorm(h4)
        t0 = time.perf_counter()
        o = ly.mlp(xa)
        mx.eval(o)
        return time.perf_counter() - t0

    mlp_ms_list = []
    for _ in range(3):
        vals = []
        for i in range(len(layers)):
            mx.eval(h4)
            vals.append(mlp_layer(i))
        mlp_ms_list.append(median(vals) * 1e3)
    classes["mlp"] = median(mlp_ms_list)

    # (e) Norms per layer.
    def norm_layer(i):
        ly = layers[i]
        t0 = time.perf_counter()
        mx.eval(ly.input_layernorm(h4), ly.post_attention_layernorm(h4))
        return time.perf_counter() - t0

    norm_ms_list = []
    for _ in range(3):
        vals = []
        for i in range(len(layers)):
            mx.eval(h4)
            vals.append(norm_layer(i))
        norm_ms_list.append(median(vals) * 1e3)
    classes["norms"] = median(norm_ms_list)

    # (f) lm_head.
    class_wall("lm_head", lambda: lm.lm_head(h4))
    tape1 = getattr(_m5, "GDN_TAPE_CALLS", -1)
    print(f"TAPE_DELTA={tape1 - tape0}")

    # (g) Isolated SDPA microbench at the real attention shapes.
    # Pull the real K/V out of one attention-layer cache and run the SDPA
    # primitive directly at qL=4 (verify), qL=1 (decode), qL=5 (WMMA path).
    a0 = attn_layers[0]
    nq, nkv, dhead = 16, 4, 256
    c0 = cache[[j for j, ll in enumerate(layers) if ll is a0][0]]
    kL = c0.keys.shape[2]

    def sdpa_wall(qL, reps=5, do_causal=True):
        q = mx.zeros((1, nq, qL, dhead), dtype=mx.bfloat16)
        k = c0.keys  # [1, nkv, kL, dhead] bf16, last-dim stride 1
        v = c0.values
        mask = "causal" if do_causal else None
        # warm
        for _ in range(2):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (dhead**0.5), mask=mask)
        mx.eval()
        vals = []
        for _ in range(reps):
            t0 = time.perf_counter()
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (dhead**0.5), mask=mask)
            mx.eval(o)
            vals.append((time.perf_counter() - t0) * 1e3)
        return median(vals)

    s4 = sdpa_wall(4)
    s1 = sdpa_wall(1)
    s5 = sdpa_wall(5)
    # Bytes moved per call (per attention layer): K+V read (bf16), Q read,
    # O write. GQA: each of the 16 q-heads reads its KV head (4x redundancy).
    kv_b = 2 * kL * dhead * 2 * nkv
    act_b = (nq * 4 * dhead * 2) + (nq * 4 * dhead * 2)  # q read + o write (qL=4)
    bytes_sdpa4 = kv_b + act_b
    bw4 = bytes_sdpa4 / (s4 * 1e-3) / 1e9
    print(
        f"SDPA_MICRO H={nq} HKV={nkv} D={dhead} kL={kL} "
        f"ms_qL1={s1:.3f} ms_qL4={s4:.3f} ms_qL5={s5:.3f} "
        f"bytes_qL4={bytes_sdpa4} eff_bw_qL4={bw4:.1f} GB/s "
        f"pct_floor_qL4={100 * bw4 / HBM_FLOOR_GBPS:.1f}%"
    )
    print(f"ATTN_PER_LAYER_MS={json.dumps([f'{x:.3f}' for x in attn_per_layer_ms])}")

    out = {
        "ctx_tokens": ctx,
        "kL": kL,
        "s4va_env": os.environ.get("MLX_ROCM_SDPA_S4VA", "unset"),
        "fused_verify_ms": fused_ms,
        "classes_total_ms": {
            "qmv_gdn": classes["qmv_gdn"] * len(gdn_layers),
            "gdn_fused": classes["gdn_fused"] * len(gdn_layers),
            "attn": classes["attn"] * len(attn_layers),
            "mlp": classes["mlp"] * len(layers),
            "norms": classes["norms"] * len(layers),
            "lm_head": classes["lm_head"],
        },
        "sdpa_micro_ms": {"qL1": s1, "qL4": s4, "qL5": s5},
        "sdpa4_eff_bw_gbps": bw4,
    }
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
