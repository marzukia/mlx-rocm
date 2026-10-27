import os
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import json, struct, glob
import numpy as np
import mlx.core as mx
from mlx_lm import load

D = "/mnt/models/qwen3-9b-mlx/"
HF = "/mnt/models/hf-qwen3-9b/"

def read_st(path, keys):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    out = {}
    for k in keys:
        if k in hdr:
            t = hdr[k]; a, b = t["data_offsets"]
            fh = open(path, "rb"); fh.seek(8 + n + a); raw = fh.read(b - a)
            out[k] = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return out

NORM_KEYS = [
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]

# HF raw (w-1)
import importlib.util
spec = importlib.util.spec_from_file_location("nprca", "/tmp/np-rca-np.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r = m.ShardReader()
hf_raw = {k: r.tensor(k) for k in NORM_KEYS}

# loaded model
model, tok = load(D)
mtp = model.language_model.mtp
loaded = {
    "mtp.layers.0.input_layernorm.weight": np.array(mtp.layers[0].input_layernorm.weight.astype(mx.float32)),
    "mtp.layers.0.post_attention_layernorm.weight": np.array(mtp.layers[0].post_attention_layernorm.weight.astype(mx.float32)),
    "mtp.layers.0.self_attn.q_norm.weight": np.array(mtp.layers[0].self_attn.q_norm.weight.astype(mx.float32)),
    "mtp.layers.0.self_attn.k_norm.weight": np.array(mtp.layers[0].self_attn.k_norm.weight.astype(mx.float32)),
    "mtp.norm.weight": np.array(mtp.norm.weight.astype(mx.float32)),
    "mtp.pre_fc_norm_embedding.weight": np.array(mtp.pre_fc_norm_embedding.weight.astype(mx.float32)),
    "mtp.pre_fc_norm_hidden.weight": np.array(mtp.pre_fc_norm_hidden.weight.astype(mx.float32)),
}

# model-mtp.safetensors (last loaded by glob)
mtpfile = read_st(D + "model-mtp.safetensors", NORM_KEYS)

print("%-42s %10s %10s %10s %10s" % ("key", "HF_raw", "HF+1", "loaded", "mtpfile"))
for k in NORM_KEYS:
    hf = hf_raw[k].mean()
    ld = loaded[k].mean()
    mf = mtpfile[k].mean() if k in mtpfile else float("nan")
    print("%-42s %10.4f %10.4f %10.4f %10.4f" % (k, hf, hf + 1, ld, mf))
print()
print("Conclusion: loaded MTP norms match HF_raw (w-1) => MTP using (w-1), NOT true weight (w=HF+1)")
print("  => true MTP weight = HF + 1; loaded model is missing +1 on MTP norms" if abs(loaded[NORM_KEYS[0]].mean() - hf_raw[NORM_KEYS[0]].mean()) < 0.05 else "  => check")
