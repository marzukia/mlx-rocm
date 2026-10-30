import os
import numpy as np
import pickle
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache

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

def snap_cache(cache):
    snap = []
    for c in cache:
        if isinstance(c, KVCache):
            snap.append((c, npf(c.keys), npf(c.values), c.offset))
        else:
            snap.append((c, [npf(x) for x in c.cache], c.left_padding, c.lengths))
    return snap

def restore_cache(cache, snap):
    for c, data, extra1, extra2 in snap:
        if isinstance(c, KVCache):
            # the verify forward appends 2 entries in-place; trim to restore
            c.trim(2)
        else:
            c.cache = [mx.array(x) for x in data]
            c.left_padding = extra1
            c.lengths = extra2

out = []
target_step = None
for step in range(130):
    t = int(mx.argmax(lg))
    out.append(t)
    emb = lm.model.embed_tokens(mx.array([[t]]))
    dh = lm.mtp(hidd, emb, mtp_cache)
    draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
    prev = t
    if step == 100:
        target_step = (prev, draft)
        # snapshot
        snap = snap_cache(cache)
        # Fork A: S=2 [prev, draft]
        logitsA, _ = lm(mx.array([[prev, draft]]), cache=cache, return_hidden=True, return_prenorm=True)
        mx.eval([logitsA])
        stateA = [(i, [npf(x) for x in c.cache]) for i, c in enumerate(cache) if not isinstance(c, KVCache)]
        kvA = [(i, npf(c.keys), npf(c.values)) for i, c in enumerate(cache) if isinstance(c, KVCache)]
        # restore
        restore_cache(cache, snap)
        # Fork B: S=1 [prev] then [draft]
        logitsB1, _ = lm(mx.array([[prev]]), cache=cache, return_hidden=True, return_prenorm=True)
        mx.eval([logitsB1])
        logitsB2, _ = lm(mx.array([[draft]]), cache=cache, return_hidden=True, return_prenorm=True)
        mx.eval([logitsB2])
        stateB = [(i, [npf(x) for x in c.cache]) for i, c in enumerate(cache) if not isinstance(c, KVCache)]
        kvB = [(i, npf(c.keys), npf(c.values)) for i, c in enumerate(cache) if isinstance(c, KVCache)]
        # compare
        print(f"step {step}: prev={prev} draft={draft}")
        print("=== GDN state (conv, ssm) S=2 vs two S=1 ===")
        for (iA, dataA), (iB, dataB) in zip(stateA, stateB):
            for j, (a, b) in enumerate(zip(dataA, dataB)):
                d = a - b
                name = "conv" if j == 0 else "ssm"
                print(f"  layer {iA} {name}: n={np.count_nonzero(d!=0)}/{d.size} maxabs={np.abs(d).max():.3e}")
        print("=== KV cache S=2 vs two S=1 ===")
        for (iA, kA, vA), (iB, kB, vB) in zip(kvA, kvB):
            dk = kA - kB
            dv = vA - vB
            print(f"  layer {iA} keys: n={np.count_nonzero(dk!=0)}/{dk.size} maxabs={np.abs(dk).max():.3e}")
            print(f"  layer {iA} values: n={np.count_nonzero(dv!=0)}/{dv.size} maxabs={np.abs(dv).max():.3e}")
        # logits
        dA = npf(logitsA[0, 1]) - npf(logitsB2[0, 0])
        print(f"logits after draft (A[0,1] vs B2[0,0]): n={np.count_nonzero(dA!=0)}/{dA.size} maxabs={np.abs(dA).max():.3e}")
        break
    # continue the MTP loop
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
