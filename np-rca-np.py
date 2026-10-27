"""Pure-numpy reference of Qwen3.5-9B text trunk + MTP (bf16 HF weights, f32 math).

Ported from mlx_lm/models/qwen3_5.py + gated_delta.py + qwen3_next.py + rope_utils.py.
All math in float32. Weights read directly from the HF safetensors shards.

Modes:
  e1   numpy trunk on the fruits prompt -> t1, t2_ref, h1_ref, e1_ref, prompt logits
  e3   numpy MTP (true HF mtp weights) fed (h1_ref, e1_ref) -> rank of t2_ref, max logit
  e4   numpy MTP fed (h1_mlx, e1_mlx) -> rank of t2_mlx, max logit
  mtpcheck  sanity: run MTP on a synthetic input, print intermediate stats
"""
import os
import sys
import struct
import json
import time

import numpy as np

HF = "/mnt/models/hf-qwen3-9b"
IDX = HF + "/model.safetensors.index.json"

# ---------------- config (from text_config) ----------------
HID = 4096
NQ = 16
NKV = 4
HD = 256
INTER = 12288
VOCAB = 248320
EPS = 1e-6
LKD = 128   # linear_key_head_dim
LNK = 16    # linear_num_key_heads
LNV = 32    # linear_num_value_heads
LVD = 128   # linear_value_head_dim
LCK = 4     # linear_conv_kernel_dim
ROPE_THETA = 1e7
PRF = 0.25
N_LAYERS = 32
FA_INT = 4

PROMPT = "What color are the fruits watermelon, strawberry, orange and banana?"
PROMPT_IDS = [3710, 1829, 513, 279, 24517, 2919, 68705, 11, 70135, 11, 18016, 321, 41674, 30]


# ---------------- safetensors reader (pure numpy) ----------------
_ELEMSIZE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
            "I8": 1, "U8": 1, "U32": 4, "BOOL": 1}
_NPDT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
         "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
         "U8": np.uint8, "U32": np.uint32, "BOOL": np.bool_}


class ShardReader:
    def __init__(self, index_path=IDX):
        idx = json.load(open(index_path))
        self.wmap = idx["weight_map"]
        self.files = sorted(set(self.wmap.values()))
        self.hdrs = {}
        self.hdrn = {}
        self.paths = {}
        for f in self.files:
            p = os.path.join(HF, f)
            self.paths[f] = p
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.hdrs[f] = json.loads(fh.read(n))
                self.hdrn[f] = n

    def tensor(self, key, out=np.float32):
        f = self.wmap[key]
        info = self.hdrs[f][key]
        shape = info["shape"]
        dt = info["dtype"]
        if "data_offsets" in info:
            off = info["data_offsets"][0]
        else:
            off = info["data_offset"]
        n = 1
        for s in shape:
            n *= s
        with open(self.paths[f], "rb") as fh:
            fh.seek(8 + self.hdrn[f] + off)
            raw = fh.read(n * _ELEMSIZE[dt])
        arr = np.frombuffer(raw, dtype=_NPDT[dt])
        if dt == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = arr.astype(np.float32)
        return arr.reshape(shape).astype(out, copy=False)


# ---------------- math helpers ----------------
def rms_norm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(x)))


def linear(x, W):
    return x @ W.T


def apply_rope(x, pos, theta=ROPE_THETA, prf=PRF):
    """x: [B, H, L, D]; rotate first int(D*prf) dims (half-split, non-traditional)."""
    D = x.shape[-1]
    dims = int(D * prf)
    half = dims // 2
    inv_freq = theta ** (-np.arange(half, dtype=np.float32) / half)
    pos = np.asarray(pos, dtype=np.float32)
    angles = pos[:, None] * inv_freq[None, :]
    cos = np.cos(angles)[None, None, :, :]
    sin = np.sin(angles)[None, None, :, :]
    out = x.copy()
    x1 = out[..., :half]
    x2 = out[..., half:2 * half]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:2 * half] = x1 * sin + x2 * cos
    return out


# ---------------- GatedDeltaNet ----------------
class GatedDeltaNet:
    def __init__(self, W):
        self.W = W

    def _causal_conv(self, conv_input, w):
        B, T_in, C = conv_input.shape
        K = w.shape[2]
        n_out = T_in - K + 1
        out = np.zeros((B, n_out, C), dtype=np.float32)
        for i in range(K):
            out[:, :, :] += w[:, 0, i][None, None, :] * conv_input[:, i:i + n_out, :]
        return out

    def _normalize_qk(self, q, k):
        inv_scale = LKD ** -0.5
        rms_eps = EPS * inv_scale ** 2
        qn = (inv_scale ** 2) * (q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + rms_eps))
        kn = inv_scale * (k / np.sqrt(np.mean(k * k, axis=-1, keepdims=True) + rms_eps))
        return qn, kn

    def __call__(self, x, conv_state, ssm_state):
        B, S, _ = x.shape
        W = self.W
        qkv = linear(x, W["in_proj_qkv"])
        z = linear(x, W["in_proj_z"]).reshape(B, S, LNV, LVD)
        b = linear(x, W["in_proj_b"])
        a = linear(x, W["in_proj_a"])
        _dbg = os.environ.get("NP_RCA_GDN_DBG")
        if _dbg:
            print(f"      [gdn xnorm_in={np.linalg.norm(x):.3f} znorm={np.linalg.norm(z):.3f} zmean={z.mean():.4f} zmax={z.max():.4f} bnorm={np.linalg.norm(b):.3f} anorm={np.linalg.norm(a):.3f}]")
        conv_input = np.concatenate([conv_state, qkv], axis=1)
        conv_out = silu(self._causal_conv(conv_input, W["conv1d"]))
        new_conv_state = conv_input[:, -3:, :].copy()
        key_dim = LKD * LNK
        value_dim = LVD * LNV
        q = conv_out[..., :key_dim].reshape(B, S, LNK, LKD)
        k = conv_out[..., key_dim:2 * key_dim].reshape(B, S, LNK, LKD)
        v = conv_out[..., 2 * key_dim:2 * key_dim + value_dim].reshape(B, S, LNV, LVD)
        q, k = self._normalize_qk(q, k)
        rep = LNV // LNK
        q = np.repeat(q, rep, axis=2)
        k = np.repeat(k, rep, axis=2)
        beta = sigmoid(b)
        A = np.exp(W["A_log"])
        g = np.exp(-A[None, None, :] * softplus(a + W["dt_bias"][None, None, :]))
        state = ssm_state
        ys = np.zeros((B, S, LNV, LVD), dtype=np.float32)
        for t in range(S):
            state = state * g[:, t, :, None, None]
            kv_mem = np.sum(state * k[:, t, :, None, :], axis=-1)
            delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
            state = state + k[:, t, :, None, :] * delta[:, :, None, :]
            ys[:, t] = np.sum(state * q[:, t, :, None, :], axis=-1)
        xnorm = rms_norm(ys, W["gdn_norm"], EPS)
        out = silu(z) * xnorm
        if _dbg:
            print(f"      [gdn ysnorm={np.linalg.norm(ys):.3f} xnormnorm={np.linalg.norm(xnorm):.3f} siluzmean={silu(z).mean():.4f} outnorm={np.linalg.norm(out):.3f}]")
        out = linear(out.reshape(B, S, -1), W["out_proj"])
        if _dbg:
            print(f"      [gdn rnorm={np.linalg.norm(out):.3f}]")
        return out, new_conv_state, state


# ---------------- full attention ----------------
def full_attention(x, W, kv_cache, pos_offset):
    B, L, _ = x.shape
    qpo = linear(x, W["q_proj"]).reshape(B, L, NQ, 2 * HD)
    queries = qpo[..., :HD]
    gate = qpo[..., HD:]
    keys = linear(x, W["k_proj"]).reshape(B, L, NKV, HD)
    values = linear(x, W["v_proj"]).reshape(B, L, NKV, HD)
    queries = rms_norm(queries, W["q_norm"], EPS).transpose(0, 2, 1, 3)
    keys = rms_norm(keys, W["k_norm"], EPS).transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    pos = pos_offset + np.arange(L)
    queries = apply_rope(queries, pos)
    keys = apply_rope(keys, pos)
    if kv_cache is not None:
        ck, cv = kv_cache
        keys_all = np.concatenate([ck, keys], axis=2)
        values_all = np.concatenate([cv, values], axis=2)
    else:
        keys_all = keys
        values_all = values
    n_rep = NQ // NKV
    k_rep = np.repeat(keys_all, n_rep, axis=1)
    v_rep = np.repeat(values_all, n_rep, axis=1)
    scale = HD ** -0.5
    scores = np.einsum("bnlh,bnlh->bnl", (queries * scale), k_rep)
    # scores: [B, NQ, L, Lt]
    Lt = keys_all.shape[2]
    qpos = pos_offset + np.arange(L)
    kpos = np.arange(Lt)
    mask = kpos[None, :] <= qpos[:, None]
    scores = np.where(mask[None, None, :, :], scores, -1e30)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / e.sum(axis=-1, keepdims=True)
    out = np.einsum("bnlj,bnlj->bnl", weights, v_rep)  # [B, NQ, L, HD]
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    out = out * sigmoid(gate.reshape(B, L, -1))
    out = linear(out, W["o_proj"])
    return out


def mlp(x, W):
    gate = linear(x, W["gate_proj"])
    up = linear(x, W["up_proj"])
    h = silu(gate) * up
    return linear(h, W["down_proj"])


# ---------------- trunk ----------------
class Trunk:
    def __init__(self, reader):
        self.reader = reader
        self.embed = reader.tensor("model.language_model.embed_tokens.weight")
        self.lm_head = reader.tensor("lm_head.weight")
        self.final_norm = reader.tensor("model.language_model.norm.weight") + 1.0  # HF stores (w-1)
        self.gdn_state = {}
        self.kv_cache = {}
        self._layer_w = {}

    def _load_layer(self, i):
        if i in self._layer_w:
            return self._layer_w[i]
        p = f"model.language_model.layers.{i}."
        W = {
            "input_layernorm": self.reader.tensor(p + "input_layernorm.weight") + 1.0,  # (w-1)
            "post_attention_layernorm": self.reader.tensor(p + "post_attention_layernorm.weight") + 1.0,  # (w-1)
            "gate_proj": self.reader.tensor(p + "mlp.gate_proj.weight"),
            "up_proj": self.reader.tensor(p + "mlp.up_proj.weight"),
            "down_proj": self.reader.tensor(p + "mlp.down_proj.weight"),
        }
        if (i + 1) % FA_INT != 0:
            W["in_proj_qkv"] = self.reader.tensor(p + "linear_attn.in_proj_qkv.weight")
            W["in_proj_z"] = self.reader.tensor(p + "linear_attn.in_proj_z.weight")
            W["in_proj_b"] = self.reader.tensor(p + "linear_attn.in_proj_b.weight")
            W["in_proj_a"] = self.reader.tensor(p + "linear_attn.in_proj_a.weight")
            W["A_log"] = self.reader.tensor(p + "linear_attn.A_log")
            W["dt_bias"] = self.reader.tensor(p + "linear_attn.dt_bias")
            W["conv1d"] = self.reader.tensor(p + "linear_attn.conv1d.weight")
            W["gdn_norm"] = self.reader.tensor(p + "linear_attn.norm.weight")
            W["out_proj"] = self.reader.tensor(p + "linear_attn.out_proj.weight")
            W["gdn"] = GatedDeltaNet(W)
        else:
            W["q_proj"] = self.reader.tensor(p + "self_attn.q_proj.weight")
            W["k_proj"] = self.reader.tensor(p + "self_attn.k_proj.weight")
            W["v_proj"] = self.reader.tensor(p + "self_attn.v_proj.weight")
            W["o_proj"] = self.reader.tensor(p + "self_attn.o_proj.weight")
            W["q_norm"] = self.reader.tensor(p + "self_attn.q_norm.weight") + 1.0  # (w-1)
            W["k_norm"] = self.reader.tensor(p + "self_attn.k_norm.weight") + 1.0  # (w-1)
        self._layer_w[i] = W
        return W

    def reset_state(self):
        self.gdn_state = {}
        self.kv_cache = {}

    def forward_with_cache(self, token_ids, pos_offset=0, track=False):
        """Like forward but maintains/returns kv cache properly."""
        B = 1
        L = len(token_ids)
        x = self.embed[np.asarray(token_ids)].reshape(B, L, HID).astype(np.float32)
        if track:
            print(f"    [in] norm={np.linalg.norm(x):.3f}", flush=True)
        for i in range(N_LAYERS):
            W = self._load_layer(i)
            h1 = rms_norm(x, W["input_layernorm"], EPS)
            if (i + 1) % FA_INT != 0:
                cs = self.gdn_state.get(i)
                if cs is None:
                    cs = (np.zeros((B, 3, LKD * LNK * 2 + LVD * LNV), dtype=np.float32),
                         np.zeros((B, LNV, LVD, LKD), dtype=np.float32))
                r, cs, ss = W["gdn"](h1, cs[0], cs[1])
                self.gdn_state[i] = (cs, ss)
            else:
                kc = self.kv_cache.get(i)
                r, new_kc = full_attention_cached(h1, W, kc, pos_offset)
                self.kv_cache[i] = new_kc
            h = x + r
            h2 = rms_norm(h, W["post_attention_layernorm"], EPS)
            x = h + mlp(h2, W)
            if track:
                kind = "GDN" if (i + 1) % FA_INT != 0 else "FA "
                print(f"    [L{i:2d} {kind}] xnorm={np.linalg.norm(x):.3f} rnorm={np.linalg.norm(r):.3f}", flush=True)
        hpre = x
        hnorm = rms_norm(hpre, self.final_norm, EPS)
        logits = linear(hnorm, self.lm_head)
        return logits, hpre


def full_attention_cached(x, W, kv_cache, pos_offset):
    """Returns (attn_out, new_kv_cache)."""
    B, L, _ = x.shape
    qpo = linear(x, W["q_proj"]).reshape(B, L, NQ, 2 * HD)
    queries = qpo[..., :HD]
    gate = qpo[..., HD:]
    keys = linear(x, W["k_proj"]).reshape(B, L, NKV, HD)
    values = linear(x, W["v_proj"]).reshape(B, L, NKV, HD)
    queries = rms_norm(queries, W["q_norm"], EPS).transpose(0, 2, 1, 3)
    keys = rms_norm(keys, W["k_norm"], EPS).transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    pos = pos_offset + np.arange(L)
    queries = apply_rope(queries, pos)
    keys = apply_rope(keys, pos)
    if kv_cache is not None:
        ck, cv = kv_cache
        keys_all = np.concatenate([ck, keys], axis=2)
        values_all = np.concatenate([cv, values], axis=2)
    else:
        keys_all = keys
        values_all = values
    n_rep = NQ // NKV
    k_rep = np.repeat(keys_all, n_rep, axis=1)
    v_rep = np.repeat(values_all, n_rep, axis=1)
    scale = HD ** -0.5
    scores = np.einsum("bnlh,bnjh->bnlj", (queries * scale), k_rep)
    Lt = keys_all.shape[2]
    qpos = pos_offset + np.arange(L)
    kpos = np.arange(Lt)
    mask = kpos[None, :] <= qpos[:, None]
    scores = np.where(mask[None, None, :, :], scores, -1e30)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / e.sum(axis=-1, keepdims=True)
    out = np.einsum("bnlj,bnjh->bnlh", weights, v_rep)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    out = out * sigmoid(gate.reshape(B, L, -1))
    out = linear(out, W["o_proj"])
    return out, (keys_all, values_all)


# ---------------- MTP module ----------------
class MTP:
    def __init__(self, reader):
        p = "mtp."
        self.pre_fc_norm_hidden = reader.tensor(p + "pre_fc_norm_hidden.weight") + 1.0  # (w-1)
        self.pre_fc_norm_embedding = reader.tensor(p + "pre_fc_norm_embedding.weight") + 1.0  # (w-1)
        self.fc = reader.tensor(p + "fc.weight")
        self.norm = reader.tensor(p + "norm.weight") + 1.0  # (w-1)
        q = "mtp.layers.0."
        self.W = {
            "input_layernorm": reader.tensor(q + "input_layernorm.weight") + 1.0,  # (w-1)
            "post_attention_layernorm": reader.tensor(q + "post_attention_layernorm.weight") + 1.0,  # (w-1)
            "gate_proj": reader.tensor(q + "mlp.gate_proj.weight"),
            "up_proj": reader.tensor(q + "mlp.up_proj.weight"),
            "down_proj": reader.tensor(q + "mlp.down_proj.weight"),
            "q_proj": reader.tensor(q + "self_attn.q_proj.weight"),
            "k_proj": reader.tensor(q + "self_attn.k_proj.weight"),
            "v_proj": reader.tensor(q + "self_attn.v_proj.weight"),
            "o_proj": reader.tensor(q + "self_attn.o_proj.weight"),
            "q_norm": reader.tensor(q + "self_attn.q_norm.weight") + 1.0,  # (w-1)
            "k_norm": reader.tensor(q + "self_attn.k_norm.weight") + 1.0,  # (w-1)
        }

    def __call__(self, hidden, token_embed):
        # hidden, token_embed: [H] or [1,1,H]
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
        token_embed = np.asarray(token_embed, dtype=np.float32).reshape(-1)
        e = rms_norm(token_embed[None, None], self.pre_fc_norm_embedding, EPS)
        h = rms_norm(hidden[None, None], self.pre_fc_norm_hidden, EPS)
        x = linear(np.concatenate([e, h], axis=-1), self.fc)  # [1,1,H]
        # full-attention decoder layer, fresh KV, position 0
        W = self.W
        h1 = rms_norm(x, W["input_layernorm"], EPS)
        r, _ = full_attention_cached(h1, W, None, 0)
        h = x + r
        h2 = rms_norm(h, W["post_attention_layernorm"], EPS)
        x = h + mlp(h2, W)
        x = rms_norm(x, self.norm, EPS)
        return x  # [1,1,H]


def rank_of(logits, target):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    order = np.argsort(-logits)
    return int(np.where(order == target)[0][0])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "e1"
    t0 = time.time()
    reader = ShardReader()
    print(f"[{time.time()-t0:.1f}s] index loaded", flush=True)

    if mode == "track":
        trunk = Trunk(reader)
        trunk.reset_state()
        print("=== numpy per-layer norms (prompt) ===", flush=True)
        logits, _ = trunk.forward_with_cache(PROMPT_IDS, pos_offset=0, track=True)
        print("track-done", flush=True)

    elif mode == "l0":
        trunk = Trunk(reader)
        W = trunk._load_layer(0)
        x = trunk.embed[np.asarray(PROMPT_IDS)].reshape(1, len(PROMPT_IDS), HID).astype(np.float32)
        emb = x
        h1 = rms_norm(x, W["input_layernorm"], EPS)
        cs = (np.zeros((1, 3, LKD * LNK * 2 + LVD * LNV), dtype=np.float32),
             np.zeros((1, LNV, LVD, LKD), dtype=np.float32))
        r, cs, ss = W["gdn"](h1, cs[0], cs[1])
        z = linear(h1, W["in_proj_z"]).reshape(1, len(PROMPT_IDS), LNV, LVD)
        np.save("/tmp/np-rca-np-r.npy", r)
        np.save("/tmp/np-rca-np-z.npy", z)
        np.save("/tmp/np-rca-np-h1.npy", h1)
        np.save("/tmp/np-rca-np-ipz.npy", W["in_proj_z"])
        np.save("/tmp/np-rca-np-iln.npy", W["input_layernorm"])
        np.save("/tmp/np-rca-np-emb.npy", emb)
        print(f"l0: rnorm={np.linalg.norm(r):.3f} znorm={np.linalg.norm(z):.3f} h1norm={np.linalg.norm(h1):.3f}", flush=True)
        print("l0-done", flush=True)

    elif mode == "e1":
        trunk = Trunk(reader)
        print(f"[{time.time()-t0:.1f}s] embed+lm_head loaded", flush=True)
        trunk.reset_state()
        logits_p, _ = trunk.forward_with_cache(PROMPT_IDS, pos_offset=0)
        t1 = int(np.argmax(logits_p[0, -1]))
        print(f"[{time.time()-t0:.1f}s] prompt done, t1={t1}", flush=True)
        logits, h1 = trunk.forward_with_cache([t1], pos_offset=len(PROMPT_IDS))
        t2 = int(np.argmax(logits[0, -1]))
        e1 = trunk.embed[t1].astype(np.float32)
        print(f"[{time.time()-t0:.1f}s] decode done, t2={t2}", flush=True)
        print("numpy prompt top8:", [(int(i), round(float(logits_p[0, -1][i]), 2))
              for i in np.argsort(-logits_p[0, -1])[:8]], flush=True)
        np.savez("/tmp/np-rca-e1.npz",
                t1=np.int64(t1), t2_ref=np.int64(t2),
                h1_ref=h1[0, -1], e1_ref=e1,
                prompt_logits_last=logits_p[0, -1], prompt_logits=logits_p[0],
                decode_logits=logits[0, -1])
        print("E1-DONE", flush=True)

    elif mode == "mtp":
        mtp = MTP(reader)
        print(f"[{time.time()-t0:.1f}s] mtp loaded", flush=True)
        np.save("/tmp/np-rca-mtp-ready.npy", np.ones(1))
        # keep mtp for e3/e4 by re-loading in those modes instead

    elif mode in ("e3", "e4"):
        mtp = MTP(reader)
        d = np.load("/tmp/np-rca-e1.npz") if mode == "e3" else np.load("/tmp/np-rca-e2.npz")
        h1 = d["h1_ref" if mode == "e3" else "h1_mlx"]
        e1 = d["e1_ref" if mode == "e3" else "e1_mlx"]
        t2 = int(d["t2_ref" if mode == "e3" else "t2_mlx"])
        xn = mtp(h1, e1)
        logits = linear(xn[0, 0], reader.tensor("lm_head.weight"))
        rk = rank_of(logits, t2)
        print(f"{mode}: draft max={logits.max():.3f} t2={t2} rank={rk} "
              f"top1={int(np.argmax(logits))}", flush=True)
        # also correlation with target logits
        tgt = d["decode_logits"]
        corr = np.corrcoef(logits, tgt)[0, 1]
        print(f"{mode}: corr(draft,target)={corr:.4f}", flush=True)
        np.savez(f"/tmp/np-rca-{mode}.npz", draft_logits=logits, target_logits=tgt,
                t2=np.int64(t2), rank=np.int64(rk))
        print(f"{mode.upper()}-DONE", flush=True)


if __name__ == "__main__":
    main()
