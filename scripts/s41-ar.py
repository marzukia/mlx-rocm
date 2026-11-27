#!/usr/bin/env python3
# S4(1) step-2: 64-token AR, both arms, the C7 harness class (vehicle model
# glue identical to c7run.py; CH=256 prefill, greedy, maxtok=64,
# warmup-excl class = all 64 steps at this length). Per-step GPU time =
# trace-attributed per-kernel sum (the memory-255 SDK-injected rocprof;
# this wheel has NO mx.Event API); the host/sync split (submit wall vs
# eval wall) is logged as the in-process cross-check, so the per-step
# GPU-time share is the step-1 SPLIT analog in-context.
#
# Arm C = wheel default (mr_gate=1 -> M=1 routes to mmq); arm B =
# MLX_KQUANT_FILE_MR=0 (kqfg single-row). Separate process runs; the sampler
# (s41-hostsample.sh) anchors on this file name via pgrep.
import argparse
import hashlib
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VENV = "/home/andryo/venv-ar/lib/python3.14/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs  # noqa: E402
from mlx_lm.models.qwen3_next import Qwen3NextAttention  # noqa: E402


class FullLayer64(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.is_linear = False
        self.self_attn = Qwen3NextAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = nn.Sequential() if False else None
        from mlx_lm.models.qwen3_next import Qwen3NextMLP  # noqa: E402

        self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


GGUF = "/home/andryo/models/Qwen3.8-27B-Q2_K-plain.gguf"
TOK = "/home/andryo/w1-gpu3/base-baked-2c/tokenizer.json"
PROMPT = "/home/andryo/d500x-20261115/prompt.oneline"
LIB = "/home/andryo/venv-ar/lib/python3.14/site-packages/mlx/lib/libmlx.so"
KTYPE, MODE, AXIS, GROUP, KE, PR = 10, 0, 0, 32, 1, 15
BB = 84
EOS = 248044
WARMUP = 0  # maxtok=64: the warmup-excl class = all 64 steps (SC-B/C method)


def log(m):
    print(m, flush=True)


def kq_gemm(x, w):
    shp = x.shape
    y = mx.kquant_file.kquant_file_gemm(x.reshape(-1, shp[-1]), w, KTYPE, MODE, AXIS, GROUP, KE, PR)
    return y.reshape(*shp[:-1], w.shape[1] * 256)


class KQLinear(nn.Module):
    def __init__(self, raw, K, N):
        super().__init__()
        self.w = raw.reshape(K, N // 256, BB)
        self.K, self.N = K, N

    def __call__(self, x):
        return kq_gemm(x, self.w)


def small_linear(raw, K, N):
    d = mx.rocm.kq_dequant(raw, 10).reshape(K, N)
    lin = nn.Linear(K, N, bias=False)
    lin.weight = d.T.astype(mx.bfloat16)
    return lin


def get_linear(raw, K, N):
    if N >= 256:
        return KQLinear(raw, K, N)
    return small_linear(raw, K, N)


class KQEmbed(nn.Module):
    def __init__(self, raw, K, N):
        super().__init__()
        self.w = raw.reshape(K, N // 256, BB)
        self.K = K

    def __call__(self, idx):
        flat = idx.reshape(-1).astype(mx.int32)
        y = mx.kquant_file.kquant_file_gather(flat, self.w, KTYPE, MODE, AXIS, GROUP, KE, PR, "bf16")
        return y.reshape(*idx.shape, self.K)


def build(weights, mtp_on):
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=65,
        num_attention_heads=24,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        num_key_value_heads=4,
        max_position_embeddings=262144,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=256,
        full_attention_interval=4,
        mtp_num_hidden_layers=1 if mtp_on else 0,
        rope_parameters={
            "type": "default",
            "rope_type": "default",
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
        },
    )
    model = TextModel(args)
    tmod = model.model
    tmod.layers[64] = FullLayer64(args)
    tmod.embed_tokens = KQEmbed(weights["token_embd.weight"], 5120, 248320)
    model.lm_head = KQLinear(weights["output.weight"], 5120, 248320)
    tmod.norm.weight = weights["output_norm.weight"].astype(mx.bfloat16)
    for N in range(65):
        layer = tmod.layers[N]
        p = f"blk.{N}."
        layer.input_layernorm.weight = weights[p + "attn_norm.weight"].astype(mx.bfloat16)
        layer.post_attention_layernorm.weight = weights[p + "post_attention_norm.weight"].astype(mx.bfloat16)
        mlp = layer.mlp
        mlp.gate_proj = KQLinear(weights[p + "ffn_gate.weight"], 5120, 17408)
        mlp.up_proj = KQLinear(weights[p + "ffn_up.weight"], 5120, 17408)
        mlp.down_proj = KQLinear(weights[p + "ffn_down.weight"], 17408, 5120)
        if layer.is_linear:
            la = layer.linear_attn
            la.in_proj_qkv = KQLinear(weights[p + "attn_qkv.weight"], 5120, 10240)
            la.in_proj_z = KQLinear(weights[p + "attn_gate.weight"], 5120, 6144)
            la.in_proj_a = get_linear(weights[p + "ssm_alpha.weight"], 5120, 48)
            la.in_proj_b = get_linear(weights[p + "ssm_beta.weight"], 5120, 48)
            la.out_proj = KQLinear(weights[p + "ssm_out.weight"], 6144, 5120)
            la.conv1d.weight = weights[p + "ssm_conv1d.weight"].T.reshape(10240, 4, 1).astype(mx.bfloat16)
            la.dt_bias = weights[p + "ssm_dt.bias"].astype(mx.float32)
            la.A_log = weights[p + "ssm_a"].astype(mx.float32)
            la.norm.weight = weights[p + "ssm_norm.weight"].astype(mx.bfloat16)
        else:
            sa = layer.self_attn
            sa.q_proj = KQLinear(weights[p + "attn_q.weight"], 5120, 12288)
            sa.k_proj = KQLinear(weights[p + "attn_k.weight"], 5120, 1024)
            sa.v_proj = KQLinear(weights[p + "attn_v.weight"], 5120, 1024)
            sa.o_proj = KQLinear(weights[p + "attn_output.weight"], 6144, 5120)
            sa.q_norm.weight = weights[p + "attn_q_norm.weight"].astype(mx.bfloat16)
            sa.k_norm.weight = weights[p + "attn_k_norm.weight"].astype(mx.bfloat16)
    if mtp_on:
        m = model.mtp
        m.fc = KQLinear(weights["blk.64.nextn.eh_proj.weight"], 10240, 5120)
        m.pre_fc_norm_hidden.weight = weights["blk.64.nextn.hnorm.weight"].astype(mx.bfloat16)
        m.pre_fc_norm_embedding.weight = weights["blk.64.nextn.enorm.weight"].astype(mx.bfloat16)
        m.norm.weight = weights["blk.64.nextn.shared_head_norm.weight"].astype(mx.bfloat16)
        raise KeyError(
            "MTP-ON LOAD-FAULT (shared-head design): MTPDecoderLayer needs "
            "standalone self_attn/ mlp weights; file has only nextn."
            "{eh_proj,enorm,hnorm,shared_head_norm} (shared with blk.64 by design)"
        )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--maxtok", type=int, default=64)
    ap.add_argument("--outdir", default="/home/andryo/s41")
    a = ap.parse_args()
    MAXTOK = a.maxtok
    OUT = a.outdir
    t0 = time.time()
    log(f"AR-START t0={t0:.2f} tag={a.tag} pid={os.getpid()}")
    env = {
        k: os.environ.get(k)
        for k in ("MLX_KQFILE_MMQ_M1", "MLX_KQUANT_FILE_MR", "MLX_MMQ_J", "HIP_VISIBLE_DEVICES", "HCC_AMDGPU_TARGET")
    }
    log(f"ENV {json.dumps(env)}")
    log(f"WHEEL {mx.__version__}")
    libmd5 = hashlib.md5()
    with open(LIB, "rb") as f:
        libmd5.update(f.read())
    log(f"LIB-MD5 {libmd5.hexdigest()}")
    model_sha = hashlib.sha256()
    with open(GGUF, "rb") as f:
        model_sha.update(f.read())
    model_sha256s = model_sha.hexdigest()
    log(f"MODEL sha256={model_sha256s}")
    weights = mx.load(GGUF)
    mx.eval([v for v in weights.values()])
    t_load = time.time()
    log(f"LOAD-OK n_tensors={len(weights)} load_wall={t_load - t0:.1f}s")

    with open(PROMPT) as f:
        tok_text = f.read().replace("\n", " ")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOK)
    ids = tok.encode(tok_text).ids
    prompt_sha = hashlib.sha256(tok_text.encode("utf-8")).hexdigest()
    log(f"PROMPT bytes={len(tok_text)} tokens={len(ids)} sha256={prompt_sha}")

    mx.random.seed(42)
    try:
        model = build(weights, mtp_on=True)
        log("MTP-ON build ok (unexpected)")
        mtp_on = True
    except KeyError as e:
        log(f"MTP-ON {e}")
        model = build(weights, mtp_on=False)
        mtp_on = False
        log("MTP-OFF build ok")
    mx.eval([p for p in model.parameters()])
    t_build = time.time()
    log(f"BUILD-OK mtp_on={mtp_on} build_wall={t_build - t_load:.1f}s")

    cache = model.make_cache()
    CH = 256
    logits = None
    for i in range(0, len(ids), CH):
        chunk = ids[i : i + CH]
        out = model(mx.array([chunk]), cache=cache)
        lr = out[0, -1]
        mx.eval(lr)
        logits = lr
        del out
    for c in cache:
        if hasattr(c, "to_quantized"):
            c.to_quantized(group_size=64, bits=8)
    mx.eval([c for c in cache if hasattr(c, "to_quantized")])
    t_pref = time.time()
    log(f"PREFILL-OK prefill_wall={t_pref - t_build:.1f}s")

    toks = list(ids)
    step_times = []  # wall ms per step
    steps_win = []  # [t0_ns, t_submit_ns, t1_ns] for the attributor
    margins = []
    maxlogs = []
    eos_at = None
    for i in range(MAXTOK):
        lrow = logits if logits.ndim == 1 else logits[0, -1]
        mx.eval(lrow)
        ln = np.asarray(mx.astype(lrow, mx.float32))
        nt = int(np.argmax(ln))
        toks.append(nt)
        part = np.partition(ln, -2)
        margins.append(float(part[-1] - part[-2]))
        maxlogs.append(float(np.max(np.abs(ln))))
        if nt == EOS and i > 8:
            eos_at = i
            log(f"EOS at step {i}")
            break
        # per-step timing: wall-ns windows (the trace attributor joins the
        # kernel rows; the wheel has no event API) + host/sync split
        t0c = time.perf_counter_ns()
        t = mx.array([[nt]])
        logits = model(t, cache=cache)
        ts_c = time.perf_counter_ns()
        mx.eval(logits)
        t1c = time.perf_counter_ns()
        step_times.append((t1c - t0c) / 1e6)
        steps_win.append([t0c, ts_c, t1c])
    wall = float(sum(step_times))
    n_out = len(toks) - len(ids)
    st = np.array(step_times) if step_times else np.array([0.0])
    st_w = st[WARMUP:] if len(st) > WARMUP else st
    log(
        f"AR-OK out_tokens={n_out} decode_wall={wall:.2f}s "
        f"tok/s(all)={n_out / wall if wall > 0 else 0.0:.4f} "
        f"tok/s(warmup-excl)={len(st_w) / float(sum(st_w)) if len(st_w) and sum(st_w) > 0 else 0.0:.4f}"
    )
    log(
        f"MS-STEP p50(warmup-excl)={1000 * float(np.percentile(st_w, 50)):.3f} "
        f"p95={1000 * float(np.percentile(st_w, 95)):.3f} "
        f"min={1000 * float(st_w.min()):.3f} max={1000 * float(st_w.max()):.3f}"
    )
    outt = toks[len(ids) :]
    first20 = tok.decode(outt[:20])
    log("FIRST20: " + first20.replace("\n", " "))
    text = tok.decode(outt)
    log("TEXT-HEAD: " + text[:300].replace("\n", " "))
    m = np.array(margins)
    ml = np.array(maxlogs)
    log(
        f"MARGIN n={len(m)} median={float(np.median(m)) if len(m) else -1.0:.4f} "
        f"count(<0.1)={int(np.sum(m < 0.1))} maxabslogit={float(ml.max()) if len(ml) else -1.0:.3f}"
    )
    first100 = hashlib.md5(",".join(str(t) for t in outt[:100]).encode()).hexdigest()
    full = hashlib.md5(",".join(str(t) for t in outt).encode()).hexdigest()
    log(f"MD5 first100={first100} full={full}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, a.tag + ".ids.txt"), "w") as f:
        f.write(" ".join(str(t) for t in outt))
    np.save(os.path.join(OUT, a.tag + ".steppwall.npy"), st)
    res = {
        "tag": a.tag,
        "env": env,
        "wheel": mx.__version__,
        "model_sha256": model_sha256s,
        "prompt_tokens": len(ids),
        "prompt_sha256": prompt_sha,
        "mtp_on": mtp_on,
        "n_out": n_out,
        "eos_at": eos_at,
        "decode_wall_s": wall,
        "tps_all": n_out / wall if wall > 0 else 0.0,
        "ms_step_p50": 1000 * float(np.percentile(st_w, 50)),
        "ms_step_p95": 1000 * float(np.percentile(st_w, 95)),
        "first20": first20,
        "margin_median": float(np.median(m)) if len(m) else None,
        "md5_first100": first100,
        "md5_full": full,
        "pid": os.getpid(),
        "steps": steps_win,
        "load_wall_s": t_load - t0,
        "build_wall_s": t_build - t_load,
        "prefill_wall_s": t_pref - t_build,
        "wall_exit_s": time.time() - t0,
    }
    with open(os.path.join(OUT, a.tag + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"AR-DONE-STATS {json.dumps({k: v for k, v in res.items() if not isinstance(v, str)})}")


if __name__ == "__main__":
    main()
