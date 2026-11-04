#!/usr/bin/env python3
"""ROOFLINE PROBE: one K=3 MTP draft step (B config, MTP6_TAPE=hybrid),
per-op class timing.

The draft step of the K=3 MTP loop in the B config = one S=1 trunk
forward (the hybrid correction forward that produces the next draft
chain's pre-norm h + l) + the K=3 draft chain (3x MTP module calls +
3x shared 4-bit lm_head argmax + 3x embed_tokens; the MTP module's own
KV cache advances by 3). The trunk forward re-reads the full ~3.6 GiB
of 4-bit trunk weights at M=1 - unlike llama's MTP draft (a small head
that never touches the trunk).

Bench-only (no model/weight changes). Loads the quantized model with the
MTP sidecar, prefills, generates N AR tokens to reach the same CTX
regime as the verify probe, then:
  (1) fused draft-step wall (S=1 trunk forward + K=3 chain; median of
      reps, state restored between), plus the two sub-walls;
  (2) per-op-class isolated timing at the same shapes, replaying the same
      1-token forward piece by piece on the real modules:
        - qmv_gdn: GDN 4 in-proj + out_proj QMMs (24 layers, M=1)
        - gdn_fused: conv state update + gdn_fused_step_tape (24 layers, T=1)
        - attn: full attention sub-block (q/k/v/o QMM + RoPE + SDPA, 8 layers, M=1)
        - mlp: gate/up/down QMMs (32 layers, silu, M=1)
        - norms: input/post RMSNorms (32 layers)
        - lm_head: shared 4-bit head QMM (M=1; x1 in the correction forward)
        - mtp_head: one MTP module call (fc + 1 bf16 decoder layer + KV)
        - embed: one embed_tokens row gather
  (3) per-class bytes moved (trunk 4-bit packed + f32 scales; MTP bf16;
      activations bf16 at M=1).
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
from mlx_lm.models.base import create_attention_mask as _cam
from mlx_lm.models.cache import KVCache

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
IDLE_S = int(os.environ.get("MTP6_IDLE_S", "25"))


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2]


def save_states(cache, mtp_cache):
    snap = []
    for c in list(cache) + list(mtp_cache):
        if hasattr(c, "keys"):
            snap.append((c, "kv", c.keys, c.values, c.offset))
        else:
            snap.append((c, "gdn", c[0], c[1], c.lengths))
    return snap


def restore_states(snap):
    for c, kind, a, b, off in snap:
        if kind == "gdn":
            c[0] = a
            c[1] = b
            c.lengths = off
        else:
            c.keys = a
            c.values = b
            c.offset = off


def quantize_int2(w, chunk=8192):
    """Int2 g128-group asymmetric quantization of the bf16 lm_head weights
    (quarter-split 2-bit packing, radiance port; copied from mtp-bench6)."""
    N, K = w.shape
    NG, Q = K // 128, K // 4
    packed = mx.zeros((N, Q), dtype=mx.int32)
    scale = mx.zeros((N, NG), dtype=mx.float32)
    zs = mx.zeros((N, NG), dtype=mx.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        wf = w[i:j].astype(mx.float32).reshape(j - i, NG, 128)
        lo = wf.min(axis=2)
        hi = wf.max(axis=2)
        s = mx.maximum((hi - lo) / 3.0, 1e-8)
        z = mx.minimum(mx.maximum(mx.round(-lo / s), 0.0), 3.0)
        q = (
            mx.minimum(
                mx.maximum(mx.round(wf / s[..., None] + z[..., None]), 0.0),
                3.0,
            )
            .astype(mx.int32)
            .reshape(j - i, K)
        )
        packed[i:j] = q[:, :Q] | (q[:, Q : 2 * Q] << 2) | (q[:, 2 * Q : 3 * Q] << 4) | (q[:, 3 * Q :] << 6)
        scale[i:j] = s
        zs[i:j] = z * s
    return packed.astype(mx.uint8), scale, zs


def int2_argmax(x, qh):
    """Int2 draft head: int2 g128-asym dequant GEMV (mx.fast.lm_head_int2[_wide])
    + exact bf16 re-rank of the global top-32 candidates. Returns the argmax token id."""
    x = x.reshape(1, -1)
    K = x.shape[1]
    NG = K // 128
    xf = x.reshape(NG, 128).astype(mx.float32)
    sx = mx.sum(xf, axis=1)
    _op = mx.fast.lm_head_int2_wide if os.environ.get("MTP6_DRAFT_HEAD", "bf16") == "int2w" else mx.fast.lm_head_int2
    logits, cval, ccol = _op(x, qh["packed"], qh["scale"], qh["zs"], sx)
    perm = mx.argsort(cval.reshape(-1))
    idx = mx.take(ccol.reshape(-1), perm[-32:])
    if "w" in qh:
        cols = mx.take(qh["w"], idx, 0)
    else:
        cols = mx.dequantize(
            mx.take(qh["wq"], idx, 0),
            mx.take(qh["s4"], idx, 0),
            mx.take(qh["b4"], idx, 0),
            group_size=qh["gs"],
            bits=qh["bits"],
        )
    exact = mx.sum(x.astype(mx.float32) * cols.astype(mx.float32), axis=1)
    return int(mx.take(idx, mx.argmax(exact)))


def main():
    import mlx_lm.models.qwen3_5 as _m5

    t0 = time.time()
    model, tok = load(MODEL)
    lm = model.language_model
    t_load = time.time() - t0
    layers = lm.model.layers
    gdn_layers = [lyr for lyr in layers if lyr.is_linear]
    attn_layers = [lyr for lyr in layers if not lyr.is_linear]
    assert lm.mtp is not None, "no mtp module (config.mtp_num_hidden_layers?)"
    mtp_cache = [KVCache()]
    # Arms (each independently switchable via env; B = the default, no env).
    # CH_MTP_4BIT (NOT MTP_4BIT: that one is read by the wheel at model-init
    # time, before load() - this switch rebuilds the module after load, with
    # no sidecar surgery on the box).
    _arm4 = os.environ.get("CH_MTP_4BIT", "0") == "1"
    _arm2 = os.environ.get("MTP6_DRAFT_HEAD", "bf16") == "int2w"
    print(f"ARMS 4BIT_MTP={_arm4} INT2W={_arm2}", flush=True)
    if _arm4:
        # 4-bit MTP module: rebuild the quant module, load the 4-bit sidecar
        # (bench3-side pattern; the file is NOT renamed, path overridable).
        import safetensors.numpy as _sfn
        from mlx_lm.models.qwen3_5 import MTPModule

        _side = os.environ.get("MTP4_SIDECAR") or os.path.join(MODEL, "model-mtp-4bit.safetensors")
        assert os.path.exists(_side), _side
        lm.mtp = MTPModule(lm.args, 1, quant=True)

        def _setp(mod, path, val):
            cur = mod
            for p in path.split(".")[:-1]:
                cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
            setattr(cur, path.split(".")[-1], val)

        for _k, _v in _sfn.load_file(_side).items():
            if not _k.startswith("mtp."):
                continue
            _kk = _k[4:]
            if _kk.endswith(".qweight"):
                _base = _kk[: -len(".qweight")] + ".weight"
            elif _kk.endswith(".bias"):
                _base = _kk[: -len(".bias")] + ".biases"
            else:
                _base = _kk
            _setp(lm.mtp, _base, mx.array(_v))
        mx.eval(lm.mtp.parameters())
        print(f"MTP4_SIDECAR: {os.path.basename(_side)}", flush=True)
    qh = None
    if _arm2:
        # int2w draft head: build the g128-asymmetric int2 qh from the loaded
        # 4-bit lm_head (mtp-bench6 code; no sidecar file involved).
        assert hasattr(mx.fast, "lm_head_int2_wide"), "int2w op not in this wheel"
        tq0 = time.time()
        _h = lm.lm_head
        w_bf16 = mx.dequantize(_h.weight, _h.scales, _h.biases, group_size=_h.group_size, bits=_h.bits)
        packed, scale, zs = quantize_int2(w_bf16)
        mx.eval(packed, scale, zs)
        del w_bf16
        mx.clear_cache()
        qh = {
            "packed": packed,
            "scale": scale,
            "zs": zs,
            "wq": _h.weight,
            "s4": _h.scales,
            "b4": _h.biases,
            "bits": _h.bits,
            "gs": _h.group_size,
        }
        print(f"INT2_BUILD {time.time() - tq0:.1f}s", flush=True)

    def head_tok(x):
        if qh is not None:
            return int2_argmax(x, qh)
        return int(mx.argmax(lm.lm_head(x.reshape(1, -1))))

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
    print(f"IDLE_S={IDLE_S}", flush=True)
    time.sleep(IDLE_S)

    # Setup: one untimed S=1 correction forward to get (l, h) for the chain.
    _l, h = lm(mx.array([[_last]]), cache=cache, return_hidden=True, return_prenorm=True)
    TOK = int(mx.argmax(_l[0, -1]))
    mx.eval(h)
    h1 = h[0, -1][None, None, :]  # [1, 1, 4096] pre-norm hidden, as chain input
    print(f"TOK={TOK}")

    # (1a) Sub-wall: the S=1 correction trunk forward (state restored between reps).
    def correction_wall():
        snap = save_states(cache, mtp_cache)
        t0 = time.perf_counter()
        logits, h = lm(mx.array([[TOK]]), cache=cache, return_hidden=True, return_prenorm=True)
        mx.eval(logits, h)
        dt = time.perf_counter() - t0
        restore_states(snap)
        return dt

    # (1b) Sub-wall: the K=3 draft chain (3x mtp + 3x head argmax + 3x embed).
    def chain_wall():
        snap = save_states(cache, mtp_cache)
        t0 = time.perf_counter()
        emb1 = lm.model.embed_tokens(mx.array([[TOK]]))
        dh1 = lm.mtp(h[0, -1], emb1, mtp_cache)
        d1 = head_tok(dh1[0, -1])
        emb2 = lm.model.embed_tokens(mx.array([[d1]]))
        dh2 = lm.mtp(dh1[0, -1], emb2, mtp_cache)
        d2 = head_tok(dh2[0, -1])
        emb3 = lm.model.embed_tokens(mx.array([[d2]]))
        dh3 = lm.mtp(dh2[0, -1], emb3, mtp_cache)
        head_tok(dh3[0, -1])
        mx.eval(dh3)
        dt = time.perf_counter() - t0
        restore_states(snap)
        return dt

    # (1c) Fused draft-step wall (correction forward + chain).
    def draft_wall():
        snap = save_states(cache, mtp_cache)
        t0 = time.perf_counter()
        logits, h = lm(mx.array([[TOK]]), cache=cache, return_hidden=True, return_prenorm=True)
        emb1 = lm.model.embed_tokens(mx.array([[TOK]]))
        dh1 = lm.mtp(h[0, -1], emb1, mtp_cache)
        d1 = head_tok(dh1[0, -1])
        emb2 = lm.model.embed_tokens(mx.array([[d1]]))
        dh2 = lm.mtp(dh1[0, -1], emb2, mtp_cache)
        d2 = head_tok(dh2[0, -1])
        emb3 = lm.model.embed_tokens(mx.array([[d2]]))
        dh3 = lm.mtp(dh2[0, -1], emb3, mtp_cache)
        head_tok(dh3[0, -1])
        mx.eval(logits, dh3)
        dt = time.perf_counter() - t0
        restore_states(snap)
        return dt

    for _ in range(2):  # warm
        draft_wall()
    corr = [correction_wall() * 1e3 for _ in range(REPS)]
    chain = [chain_wall() * 1e3 for _ in range(REPS)]
    fused = [draft_wall() * 1e3 for _ in range(REPS)]
    print(f"CORR_MS={median(corr):.2f} all={[f'{x:.2f}' for x in corr]}")
    print(f"CHAIN_MS={median(chain):.2f} all={[f'{x:.2f}' for x in chain]}")
    print(f"FUSED_DRAFT_MS={median(fused):.2f} all={[f'{x:.2f}' for x in fused]}")
    tape0 = getattr(_m5, "GDN_TAPE_CALLS", -1)
    fuse0 = getattr(_m5, "GDN_FUSE_CALLS", -1)

    classes = {}

    def class_wall(name, fn, reps=3):
        vals = []
        for _ in range(reps):
            mx.eval(h1)
            t0 = time.perf_counter()
            outs = fn()
            if isinstance(outs, (list, tuple)):
                mx.eval(*outs)
            else:
                mx.eval(outs)
            vals.append(time.perf_counter() - t0)
        classes[name] = median(vals) * 1e3

    # (a) GDN projections per layer at M=1 (4 in-proj + out_proj; no summing;
    # in_proj_qkv output is the q|k|v concat). Norm pre-evaluated outside.
    xa0 = gdn_layers[0].input_layernorm(h1)
    mx.eval(xa0)

    def one_gdn_projs():
        ga = gdn_layers[0].linear_attn
        outs = (ga.in_proj_qkv(xa0), ga.in_proj_z(xa0), ga.in_proj_b(xa0), ga.in_proj_a(xa0))
        out = ga.out_proj(outs[1].reshape(1, 1, -1))
        return (*outs, out)

    class_wall("qmv_gdn", one_gdn_projs)

    # (b) GDN conv update + fused tape op per layer at T=1 (model-faithful:
    # the hybrid correction forward goes through the tape branch at S=1).
    def gdn_fused_layer(i):
        lyr = gdn_layers[i]
        xa = lyr.input_layernorm(h1)
        ga = lyr.linear_attn
        c = cache[[j for j, ll in enumerate(layers) if ll is lyr][0]]
        qkv = ga.in_proj_qkv(xa)
        b = ga.in_proj_b(xa)
        a = ga.in_proj_a(xa)
        z = ga.in_proj_z(xa).reshape(1, 1, ga.num_v_heads, ga.head_v_dim)
        conv_state = c[0] if c[0] is not None else mx.zeros((1, ga.conv_kernel_size - 1, ga.conv_dim), dtype=h1.dtype)
        mx.eval(qkv, b, a, z)  # projections belong to the qmv_gdn class
        t0 = time.perf_counter()
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if c.lengths is not None:
            ends = mx.clip(c.lengths, 0, 1)
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
        c.advance(1)
        mx.eval(out, state)
        return time.perf_counter() - t0

    gdn_fused_ms_list = []
    for _ in range(3):  # reps: full 24-layer sweep each rep
        vals = []
        snap = save_states(cache, mtp_cache)
        for i in range(len(gdn_layers)):
            mx.eval(h1)
            vals.append(gdn_fused_layer(i))
        restore_states(snap)
        gdn_fused_ms_list.append(median(vals) * 1e3)
    classes["gdn_fused"] = median(gdn_fused_ms_list)

    # (c) Attention sub-block per layer at M=1 (norm outside; q/k/v/o QMM + RoPE + SDPA).
    # Use the real per-forward fa_mask (created once, shared by all 8 layers).
    _fa_idx = [j for j, ll in enumerate(layers) if ll is attn_layers[0]][0]
    fa_mask = _cam(h1, cache[_fa_idx])
    mx.eval(fa_mask)

    def attn_layer(i):
        lyr = attn_layers[i]
        xa = lyr.input_layernorm(h1)
        mx.eval(xa)
        c = cache[[j for j, ll in enumerate(layers) if ll is lyr][0]]
        t0 = time.perf_counter()
        o = lyr.self_attn(xa, fa_mask, c)
        mx.eval(o)
        return time.perf_counter() - t0

    attn_ms_list = []
    for _ in range(3):
        vals = []
        snap = save_states(cache, mtp_cache)
        for i in range(len(attn_layers)):
            mx.eval(h1)
            vals.append(attn_layer(i))
        restore_states(snap)
        attn_ms_list.append(median(vals) * 1e3)
    classes["attn"] = median(attn_ms_list)

    # (d) MLP per layer at M=1 (norm outside).
    def mlp_layer(i):
        lyr = layers[i]
        xa = lyr.post_attention_layernorm(h1)
        mx.eval(xa)
        t0 = time.perf_counter()
        o = lyr.mlp(xa)
        mx.eval(o)
        return time.perf_counter() - t0

    mlp_ms_list = []
    for _ in range(3):
        vals = []
        for i in range(len(layers)):
            mx.eval(h1)
            vals.append(mlp_layer(i))
        mlp_ms_list.append(median(vals) * 1e3)
    classes["mlp"] = median(mlp_ms_list)

    # (e) Norms per layer at M=1 (input + post attention).
    def norm_layer(i):
        lyr = layers[i]
        t0 = time.perf_counter()
        mx.eval(lyr.input_layernorm(h1), lyr.post_attention_layernorm(h1))
        return time.perf_counter() - t0

    norm_ms_list = []
    for _ in range(3):
        vals = []
        for i in range(len(layers)):
            mx.eval(h1)
            vals.append(norm_layer(i))
        norm_ms_list.append(median(vals) * 1e3)
    classes["norms"] = median(norm_ms_list)

    # (f) Shared 4-bit lm_head at M=1 (the correction forward computes logits);
    # the int2w arm times the int2w op instead (same slot in the chain).
    if qh is not None:
        class_wall("lm_head", lambda: mx.array([int2_argmax(h1[0, -1], qh)]))
    else:
        class_wall("lm_head", lambda: lm.lm_head(h1))
    # (f2) Head at M=3 / M=4 (probe for the batched-argmax lever; the lever
    # itself is infeasible under the serial MTP dataflow - see the report).
    if qh is None:
        _h3 = mx.stack([h1[0, -1]] * 3, axis=0)[None]
        _h4 = mx.stack([h1[0, -1]] * 4, axis=0)[None]
        class_wall("lm_head_m3", lambda: lm.lm_head(_h3))
        class_wall("lm_head_m4", lambda: lm.lm_head(_h4))
        print(f"HEAD_M3_MS={classes['lm_head_m3']:.2f} HEAD_M4_MS={classes['lm_head_m4']:.2f}", flush=True)

    # (g) MTP module (one call; x3 per draft chain), mtp KV cache advances 1 row.
    emb_in = lm.model.embed_tokens(mx.array([[TOK]]))
    mx.eval(emb_in)

    def one_mtp():
        snap = save_states([], mtp_cache)
        t0 = time.perf_counter()
        out = lm.mtp(h1[0, -1], emb_in, mtp_cache)
        mx.eval(out)
        restore_states(snap)
        return time.perf_counter() - t0

    classes["mtp_head"] = median([one_mtp() for _ in range(3)]) * 1e3

    # (h) Embedding row gather (x3 per draft chain).
    class_wall("embed", lambda: lm.model.embed_tokens(mx.array([[TOK]])))

    tape1 = getattr(_m5, "GDN_TAPE_CALLS", -1)
    fuse1 = getattr(_m5, "GDN_FUSE_CALLS", -1)
    print(f"TAPE_DELTA={tape1 - tape0} FUSE_DELTA={fuse1 - fuse0}")

    # Bytes moved per class (trunk 4-bit: packed container nbytes + f32 scales;
    # MTP bf16: weight nbytes; activations bf16 at M=1: in K*2 + out N*2 per QMM).
    def qmm_bytes(linear, m=1):
        w = linear.weight
        act = m * w.shape[1] * 2 + m * w.shape[0] * 2
        if hasattr(linear, "scales"):
            return w.nbytes + linear.scales.nbytes + act
        return w.nbytes + act

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
    # Attention also moves the KV cache: 2 (k+v) x ctx x (nkv x d_head) x 2
    # bytes read per layer at M=1, plus 1 new row written (negligible vs the
    # read). (The verify probe's table under-counted this by the nkv factor;
    # corrected here.)
    ka = ga0a.k_proj
    nkv = getattr(getattr(lm, "args", None), "num_key_value_heads", 8)
    d_head = ka.weight.shape[0] // nkv
    kv_bytes = 2 * ctx * nkv * d_head * 2 * len(attn_layers) + 2 * 1 * nkv * d_head * 2 * len(attn_layers)
    # GDN fused op: ssm state read+write per layer, conv state, small T=1 in/out.
    c0 = cache[[j for j, ll in enumerate(layers) if ll is gdn_layers[0]][0]]
    ss_b = (c0[1].nbytes * 2) * len(gdn_layers)
    conv_b = (c0[0].nbytes + c0[0].nbytes) * len(gdn_layers)
    gdn_fused_b = ss_b + conv_b + (1 * 8192 * 2 + 1 * 64 * 2 + 1 * 64 * 2 + 1 * 8192 * 2) * len(gdn_layers)
    norm_b = len(layers) * (1 * 4096 * 2) * 2  # 2 norms x 32 layers x in+out bf16

    # MTP module (bf16): fc + 1 decoder layer (attn + mlp) + its KV (a few rows,
    # untrimmed in the canonical protocol) + 1-row activations.
    mtp_b = qmm_bytes(lm.mtp.fc)
    sa_m = lm.mtp.layers[0].self_attn
    for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        mtp_b += qmm_bytes(getattr(sa_m, n))
    m_m = lm.mtp.layers[0].mlp
    for n in ["gate_proj", "up_proj", "down_proj"]:
        mtp_b += qmm_bytes(getattr(m_m, n))
    mtp_b += 2 * 4 * nkv * d_head * 2 + 3 * 4096 * 2  # mtp KV read (few rows) + 1-row act traffic
    embed_b = 1 * 4096 * 2
    attn_b = attn_gemm + kv_bytes

    # Trunk 4-bit weight totals (the 3.6 GiB cross-check).
    gdn_names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
    attn_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_names = ["gate_proj", "up_proj", "down_proj"]
    packed = 0
    scales = 0
    for lyr in layers:
        if lyr.is_linear:
            for n in gdn_names:
                packed += getattr(lyr.linear_attn, n).weight.nbytes
                scales += getattr(lyr.linear_attn, n).scales.nbytes
        else:
            for n in attn_names:
                packed += getattr(lyr.self_attn, n).weight.nbytes
                scales += getattr(lyr.self_attn, n).scales.nbytes
        for n in mlp_names:
            packed += getattr(lyr.mlp, n).weight.nbytes
            scales += getattr(lyr.mlp, n).scales.nbytes
    head_packed = lm.lm_head.weight.nbytes
    head_scales = lm.lm_head.scales.nbytes

    out = {
        "ctx_tokens": ctx,
        "int2_head_bytes": (
            qh["packed"].nbytes + qh["scale"].nbytes + qh["zs"].nbytes + 1 * 4096 * 2 + 1 * (4096 // 128) * 4
        )
        if qh is not None
        else None,
        "fused_draft_ms": median(fused),
        "fused_all_ms": fused,
        "correction_ms": median(corr),
        "chain_ms": median(chain),
        "classes_ms_per_layer": {
            "qmv_gdn": classes["qmv_gdn"],
            "gdn_fused": classes["gdn_fused"],
            "attn": classes["attn"],
            "mlp": classes["mlp"],
            "norms": classes["norms"],
        },
        "lm_head_ms": classes["lm_head"],
        "lm_head_m3_ms": classes.get("lm_head_m3"),
        "lm_head_m4_ms": classes.get("lm_head_m4"),
        "mtp_head_ms": classes["mtp_head"],
        "embed_ms": classes["embed"],
        "arm": {
            "mtp_4bit": _arm4,
            "int2w": _arm2,
        },
        "classes_total_ms": {
            "qmv_gdn": classes["qmv_gdn"] * len(gdn_layers),
            "gdn_fused": classes["gdn_fused"] * len(gdn_layers),
            "attn": classes["attn"] * len(attn_layers),
            "mlp": classes["mlp"] * len(layers),
            "norms": classes["norms"] * len(layers),
            "lm_head": classes["lm_head"] * 4,  # 1 correction + 3 draft argmax, M=1
            "mtp_head": classes["mtp_head"] * 3,
            "embed": classes["embed"] * 3,
        },
        "bytes": {
            "qmv_gdn": gdn_proj * len(gdn_layers),
            "gdn_fused": gdn_fused_b,
            "attn": attn_b,
            "mlp": mlp_b * len(layers),
            "norms": norm_b,
            "lm_head": head_b * 4,
            "mtp_head": mtp_b * 3,
            "embed": embed_b * 3,
        },
        "trunk_4bit_bytes": {
            "layers_packed": packed,
            "layers_scales": scales,
            "head_packed": head_packed,
            "head_scales": head_scales,
        },
    }
    print("RESULT_JSON " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
