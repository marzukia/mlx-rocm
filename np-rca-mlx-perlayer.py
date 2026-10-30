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
PROMPT_IDS = [3710, 1829, 513, 279, 24517, 2919, 68705, 11, 70135, 11, 18016, 321, 41674, 30]

def n(t):
    return float(np.linalg.norm(np.array(t.astype(mx.float32))))

x = tm.embed_tokens(mx.array([PROMPT_IDS], dtype=mx.int32))
print("[in] norm=%.3f" % n(x))
h = x
for i, layer in enumerate(tm.layers):
    x_in = h
    h = layer(h)
    print("  [L%2d %s] xnorm=%.3f rnorm=%.3f" % (i, "GDN" if (i+1)%4!=0 else "FA ", n(x_in), n(h - x_in)))
hpre = h
hnorm = tm.norm(hpre)
print("hpre norm=%.3f  hnorm(final) norm=%.3f" % (n(hpre), n(hnorm)))
logits = tm.lm_head(hnorm)
print("logits max=%.3f" % float(logits.max()))
t1 = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
print("t1=%d" % t1)
np.savez("/tmp/mlx-perlayer.npz",
        h1=hpre[0, -1].astype(mx.float32),
        e1=tm.embed_tokens(mx.array([[t1]], dtype=mx.int32)).astype(mx.float32)[0,0],
        prompt_logits=logits[0].astype(mx.float32))
print("MLX-PERLAYER-DONE")
