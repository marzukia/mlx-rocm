import os
MODELS = os.environ.get("MODELS", "/models")
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import numpy as np
import mlx.core as mx
from mlx_lm import load
import importlib.util
spec = importlib.util.spec_from_file_location("nprca", "/tmp/np-rca-np.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r = m.ShardReader()
model, tok = load(f"{MODELS}/qwen3-9b-mlx/")

def cmp(name, hf, mlxw):
    mlxw = np.array(mlxw.astype(mx.float32))
    d = mlxw - hf
    print("%-30s HFmean=%8.4f mlxmean=%8.4f  HF+1==mlx? %s  maxdiff=%.4f" % (
        name, hf.mean(), mlxw.mean(), "YES" if abs(hf.mean()+1-mlxw.mean())<0.02 else "no", np.abs(d).max()))

# GDN norm (layer 0)
cmp("gdn_norm L0", r.tensor("model.language_model.layers.0.linear_attn.norm.weight"),
    model.language_model.layers[0].linear_attn.norm.weight)
# attention q/k norm (layer 3, full attention)
cmp("attn q_norm L3", r.tensor("model.language_model.layers.3.self_attn.q_norm.weight"),
    model.language_model.layers[3].self_attn.q_norm.weight)
cmp("attn k_norm L3", r.tensor("model.language_model.layers.3.self_attn.k_norm.weight"),
    model.language_model.layers[3].self_attn.k_norm.weight)
# trunk layernorm (layer 0) - known (w-1)
cmp("trunk input_ln L0", r.tensor("model.language_model.layers.0.input_layernorm.weight"),
    model.language_model.layers[0].input_layernorm.weight)
