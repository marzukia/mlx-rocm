#!/usr/bin/env python3
"""ROOFLINE PROBE: one S=4 verify forward, per-op class timing.

Bench-only (no model/weight changes). Loads the quantized model, prefills,
generates N AR tokens, then:
  (1) fused S=4 verify forward wall (median of reps, state restored between),
  (2) per-op-class isolated timing at the same shapes, replaying the same
      4-token forward piece by piece on the real modules:
        - qmv_gdn: GDN 4 in-proj + out_proj QMMs (24 layers)
        - gdn_fused: conv state update + gdn_fused_step_tape (24 layers, T=4)
        - attn: full attention sub-block (q/k/v/o QMMs + RoPE + SDPA, 8 layers)
        - mlp: gate/up/down QMMs (32 layers, silu included)
        - norms: input/post RMSNorms
        - lm_head: shared 4-bit head QMM (M=4)
  (3) per-class bytes moved (weights 4-bit packed + f32 scales; activations bf16).
Sum-of-parts vs fused wall = launch/serialization overhead.

Env: MLX_ROCM_GDN_TAPE=1 MLX_ROCM_GDN_FUSE=1 MLX_ROCM_QMV_ROWS=1
MLX_ROCM_QMV_ROWS_MIN_N=1 MLX_ROCM_REFWD_M1=1 (set as defaults below).
MTP_MODEL / MTP_AR_N / MTP6_REPS envs for model path, AR warmup, reps.
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
    print(f"LOAD_S={t_load:.2f} N_LAYERS={len(layers)} N_GDN={len(gdn_layers)} N_ATTN={len(attn_layers)}")

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
        logits, h = lm(mx.array([[_last]]), cache=cache, return_hidden=True, return_prenorm=True)
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

    # h4 = pre-final-norm hidden for the same 4 tokens (via tape path).
    _, h4 = lm(mx.array([[_last, 11, 13, 29]]), cache=cache, return_hidden=True, return_prenorm=True)
    mx.eval(h4)
    h4 = h4  # [1, 4, 4096]; model modules expect [B, S, H]

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

    # (a) GDN projections per layer: 4 in-proj + out_proj (no summing;
    # in_proj_qkv output is the q|k|v concat).
    def one_gdn_projs(i):
        ly = gdn_layers[i]
        xa = ly.input_layernorm(h4)
        ga = ly.linear_attn
        outs = (ga.in_proj_qkv(xa), ga.in_proj_z(xa), ga.in_proj_b(xa), ga.in_proj_a(xa))
        mx.eval(*outs)
        out = ga.out_proj(outs[1].reshape(1, 4, -1))
        mx.eval(out)

    class_wall("qmv_gdn", lambda: one_gdn_projs(0))

    # (b) GDN conv update + fused tape op per layer (model-faithful flow).
    def gdn_fused_layer(i):
        ly = gdn_layers[i]
        xa = ly.input_layernorm(h4)
        ga = ly.linear_attn
        c = cache[[j for j, ll in enumerate(layers) if ll is ly][0]]
        qkv = ga.in_proj_qkv(xa)
        b = ga.in_proj_b(xa)
        a = ga.in_proj_a(xa)
        z = ga.in_proj_z(xa).reshape(1, 4, ga.num_v_heads, ga.head_v_dim)
        conv_state = c[0] if c[0] is not None else mx.zeros((1, ga.conv_kernel_size - 1, ga.conv_dim), dtype=h4.dtype)
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
    for _ in range(3):  # reps: full 24-layer sweep each rep
        vals = []
        snap = save_state(cache)
        for i in range(len(gdn_layers)):
            mx.eval(h4)
            dt = gdn_fused_layer(i)
            vals.append(dt)
        restore_state(cache, snap)
        gdn_fused_ms_list.append(median(vals) * 1e3)
    classes["gdn_fused"] = median(gdn_fused_ms_list)

    # (c) Attention sub-block per layer (norm outside; q/k/v/o QMM + RoPE + SDPA).
    def attn_layer(i):
        ly = attn_layers[i]
        xa = ly.input_layernorm(h4)
        c = cache[[j for j, ll in enumerate(layers) if ll is ly][0]]
        t0 = time.perf_counter()
        o = ly.self_attn(xa, None, c)
        mx.eval(o)
        return time.perf_counter() - t0

    attn_ms_list = []
    for _ in range(3):
        vals = []
        snap = save_state(cache)
        for i in range(len(attn_layers)):
            mx.eval(h4)
            vals.append(attn_layer(i))
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

    # (e) Norms per layer (input + post attention).
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

    # (f) lm_head (single call, M=4).
    class_wall("lm_head", lambda: lm.lm_head(h4))

    tape1 = getattr(_m5, "GDN_TAPE_CALLS", -1)
    print(f"TAPE_DELTA={tape1 - tape0}")

    # Bytes moved per class (4-bit: packed container nbytes + f32 scales;
    # activations bf16 at M=4: in 4*K*2 + out 4*N*2 per QMM).
    def qmm_bytes(linear):
        w = linear.weight
        s = linear.scales
        K = w.shape[1] * 8
        N = w.shape[0]
        act = 4 * K * 2 + 4 * N * 2
        return w.nbytes + s.nbytes + act

    gdn_proj = 0
    for n in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]:
        gdn_proj += qmm_bytes(getattr(ga0, n))
    ga0a = attn_layers[0].self_attn
    attn_gemm = 0
    for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        attn_gemm += qmm_bytes(getattr(ga0a, n))
    m0 = layers[0].mlp
    mlp_b = 0
    for n in ["gate_proj", "up_proj", "down_proj"]:
        mlp_b += qmm_bytes(getattr(m0, n))
    head_b = qmm_bytes(lm.lm_head)
    # Attention also moves the KV cache: 2 (k+v) x ctx x d_head x 2 bytes
    # read per layer at M=4 (writes are 4 new rows, negligible vs reads).
    ka = ga0a.k_proj
    nkv = 4
    for obj in (getattr(lm, "args", None), lm):
        nkv = getattr(obj, "num_key_value_heads", nkv)
        break
    d_head = ka.weight.shape[0] // nkv  # k out = n_kv_heads * d_head
    kv_bytes = 2 * ctx * d_head * 2 * len(attn_layers)
    # GDN fused op: ssm state read+write per layer.
    c0 = cache[[j for j, ll in enumerate(layers) if ll is gdn_layers[0]][0]]
    ss_b = (c0[1].nbytes * 2) * len(gdn_layers)
    conv_b = (c0[0].nbytes + 4 * c0[0].shape[2] * 2) * len(gdn_layers)
    norm_b = 32 * (4 * 4096 * 2) * 2  # 2 norms x 32 layers x in+out bf16
    attn_b = attn_gemm + kv_bytes
    gdn_fused_b = ss_b + conv_b + 4 * 4096 * 2 * 24  # out read + write

    out = {
        "ctx_tokens": ctx,
        "fused_verify_ms": fused_ms,
        "fused_all_ms": fused,
        "classes_ms_per_layer": {
            "qmv_gdn": classes["qmv_gdn"],
            "gdn_fused": classes["gdn_fused"],
            "attn": classes["attn"],
            "mlp": classes["mlp"],
            "norms": classes["norms"],
        },
        "lm_head_ms": classes["lm_head"],
        "classes_total_ms": {
            "qmv_gdn": classes["qmv_gdn"] * len(gdn_layers),
            "gdn_fused": classes["gdn_fused"] * len(gdn_layers),
            "attn": classes["attn"] * len(attn_layers),
            "mlp": classes["mlp"] * len(layers),
            "norms": classes["norms"] * len(layers),
            "lm_head": classes["lm_head"],
        },
        "bytes": {
            "qmv_gdn": gdn_proj * len(gdn_layers),
            "gdn_fused": gdn_fused_b,
            "attn": attn_b,
            "mlp": mlp_b * len(layers),
            "norms": norm_b,
            "lm_head": head_b,
        },
    }
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
