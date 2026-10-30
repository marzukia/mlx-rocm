import os
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = "The capital of France is"

def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None

model, tok = load(MODEL)
lm = model.language_model
mx.eval(model.parameters())
mtp_cache = [KVCache()]
prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
cache = lm.make_cache()
logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
lg = logits[0, -1]
hidd = hidden[0, -1]

# run the MTP loop, capturing the MTP module's hidden state + KV cache at step 100
out = []
for step in range(110):
    t = int(mx.argmax(lg))
    out.append(t)
    emb = lm.model.embed_tokens(mx.array([[t]]))
    # capture the MTP module's input (hidden state) + KV cache before the MTP call
    if step == 100:
        hidd_in = npf(hidd)
        emb_in = npf(emb[0, 0])
        mtp_keys = npf(mtp_cache[0].keys)
        mtp_values = npf(mtp_cache[0].values)
        print(f"step {step}: t={t}")
        print(f"MTP input hidden: shape={hidd_in.shape} maxabs={np.abs(hidd_in).max():.3e}")
        print(f"MTP input emb: shape={emb_in.shape} maxabs={np.abs(emb_in).max():.3e}")
        print(f"MTP KV cache keys: shape={mtp_keys.shape} maxabs={np.abs(mtp_keys).max():.3e}")
        print(f"MTP KV cache values: shape={mtp_values.shape} maxabs={np.abs(mtp_values).max():.3e}")
        # run the MTP module and capture the draft
        dh = lm.mtp(hidd, emb, mtp_cache)
        draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
        print(f"draft={draft}")
        # now run the MTP module again from the SAME input (to check determinism)
        mtp_cache2 = [KVCache()]
        # restore the MTP cache to the snapshot
        mtp_cache[0].keys = mx.array(mtp_keys)
        mtp_cache[0].values = mx.array(mtp_values)
        dh2 = lm.mtp(hidd, emb, mtp_cache)
        draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
        print(f"draft2 (re-run)={draft2} same={draft==draft2}")
        # check the MTP module's output (dh) determinism
        dh_np = npf(dh[0, -1])
        dh2_np = npf(dh2[0, -1])
        d = dh_np - dh2_np
        print(f"MTP output (dh): n={np.count_nonzero(d!=0)}/{d.size} maxabs={np.abs(d).max():.3e}")
        break
    dh = lm.mtp(hidd, emb, mtp_cache)
    draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
    prev = t
    logits2, hidden2 = lm(mx.array([[prev, draft]]), cache=cache, return_hidden=True, return_prenorm=True)
    t2 = int(mx.argmax(logits2[0, 0]))
    if t2 == draft:
        out.append(t2)
        lg, hidd = logits2[0, 1], hidden2[0, 1]
    else:
        saved = [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if not isinstance(c, KVCache)]
        for i, conv, ssm, lengths in saved:
            cache[i][0] = conv
            cache[i][1] = ssm
            cache[i].lengths = lengths
        for c in cache:
            if isinstance(c, KVCache):
                c.trim(2)
        out.append(t2)
        logits2, hidden2 = lm(mx.array([[prev]]), cache=cache, return_hidden=True, return_prenorm=True)
        logits2, hidden2 = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
        lg, hidd = logits2[0, -1], hidden2[0, -1]
print(f"done: n_out={len(out)} last={out[-8:]}")
