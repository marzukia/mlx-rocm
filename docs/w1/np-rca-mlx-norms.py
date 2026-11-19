import os
MODELS = os.environ.get("MODELS", "/models")
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib"
import numpy as np
import mlx.core as mx
from mlx_lm import load
D = f"{MODELS}/qwen3-9b-mlx"
model, tok = load(D)
tm = model.language_model.model
mx.eval(model.parameters())
layer0 = tm.layers[0]
layer5 = tm.layers[5]
final = model.language_model.final_layernorm if hasattr(model.language_model, "final_layernorm") else None
def dump(name, w):
    a = np.array(w.astype(mx.float32))
    print(f"{name}: mean={a.mean():.4f} min={a.min():.4f} max={a.max():.4f} first8={np.round(a[:8],4)}", flush=True)
dump("mlx layer0.input_layernorm", layer0.input_layernorm.weight)
dump("mlx layer0.post_attn_layernorm", layer0.post_attention_layernorm.weight)
dump("mlx layer5.input_layernorm", layer5.input_layernorm.weight)
# final norm
if hasattr(tm, "norm"):
    dump(f"mlx tm.norm (final)", tm.norm.weight)
# mtp norms
mtp = model.language_model.mtp
dump("mlx mtp.pre_fc_norm_hidden", mtp.pre_fc_norm_hidden.weight)
dump("mlx mtp.pre_fc_norm_embedding", mtp.pre_fc_norm_embedding.weight)
dump("mlx mtp.norm", mtp.norm.weight)
dump("mlx mtp.layer0.input_layernorm", mtp.layers[0].input_layernorm.weight)
dump("mlx mtp.layer0.post_attn_layernorm", mtp.layers[0].post_attention_layernorm.weight)
np.save("/tmp/np-rca-mlx-post0.npy", np.array(layer0.post_attention_layernorm.weight.astype(mx.float32)))
np.save("/tmp/np-rca-mlx-final.npy", np.array(tm.norm.weight.astype(mx.float32)))
print("mlx-norms-done", flush=True)
