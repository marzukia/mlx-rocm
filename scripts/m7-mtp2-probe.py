"""m7: localize the MTP drafter non-reproducibility (t14 resume-2).

t14's m7-mtp-probe found dh differs 4096/4096 (maxabs 5.452e-01) when the
MTP module is re-run from a restored KV snapshot. This probe extends it to
restore the FULL KVCache state (keys, values, AND offset) and to test the
stateless variant (fresh cache per draft, candidate fix).

Findings established by inspection (this probe confirms on-GPU):
  - The MTP block (MultiTokenPredictor) is a single full-attention
    DecoderLayer (layer_idx = full_attention_interval - 1 = 3;
    (3+1) % 4 == 0 -> is_linear False). It has NO GDN/conv/ssm state -
    only its own KV cache (mtp_cache).
  - Qwen3NextAttention uses cache.offset for the RoPE position and for the
    k/v write slot; KVCache.state = (keys, values, offset). t14's probe
    restored keys/values but NOT offset, so the second lm.mtp call ran at
    RoPE position 101 (not 100) and attended over a zero slot.

Probe A (reproducibility from input + FULL state):
  snapshot (keys, values, offset) -> mtp -> restore (keys, values, offset)
  -> mtp -> dh must be bit-exact (n=0/4096).

Probe B (stateless variant, candidate fix):
  two lm.mtp(hidd, emb, None) calls -> dh bit-exact (pure function of input).

Probe C (stateless vs cached, numerical-equivalence check):
  compares the stateless dh to the cached dh (documents the difference;
  the MTP draft only affects the accept rate, not the emitted stream).
"""
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
logits, hidden = lm(
    mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True
)
lg = logits[0, -1]
hidd = hidden[0, -1]

# run the MTP loop to step 100
out = []
for step in range(110):
    t = int(mx.argmax(lg))
    out.append(t)
    emb = lm.model.embed_tokens(mx.array([[t]]))
    if step == 100:
        # ---- snapshot the MTP module's FULL state (keys, values, offset) ----
        mtp_keys = npf(mtp_cache[0].keys)
        mtp_values = npf(mtp_cache[0].values)
        mtp_offset = mtp_cache[0].offset
        print(f"step {step}: t={t} mtp_offset={mtp_offset}")
        print(f"MTP input hidden: shape={npf(hidd).shape} maxabs={np.abs(npf(hidd)).max():.3e}")
        print(f"MTP KV keys: shape={mtp_keys.shape} maxabs={np.abs(mtp_keys).max():.3e}")

        # ---- Probe A: reproducibility from input + FULL state ----
        dh = lm.mtp(hidd, emb, mtp_cache)
        mx.eval(dh)
        draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
        print(f"[A] first draft={draft}")
        # restore the FULL state (keys, values, AND offset)
        mtp_cache[0].keys = mx.array(mtp_keys)
        mtp_cache[0].values = mx.array(mtp_values)
        mtp_cache[0].offset = mtp_offset
        dh2 = lm.mtp(hidd, emb, mtp_cache)
        mx.eval(dh2)
        draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
        d = npf(dh[0, -1]) - npf(dh2[0, -1])
        print(f"[A] dh (full-state restore): n={np.count_nonzero(d != 0)}/{d.size} "
              f"maxabs={np.abs(d).max():.3e} draft2={draft2} same={draft == draft2}")

        # ---- Probe B: stateless variant (fresh cache per draft) ----
        dh_s1 = lm.mtp(hidd, emb, None)
        mx.eval(dh_s1)
        draft_s1 = int(mx.argmax(lm.lm_head(dh_s1)[0, -1]))
        dh_s2 = lm.mtp(hidd, emb, None)
        mx.eval(dh_s2)
        draft_s2 = int(mx.argmax(lm.lm_head(dh_s2)[0, -1]))
        d_s = npf(dh_s1[0, -1]) - npf(dh_s2[0, -1])
        print(f"[B] stateless dh: n={np.count_nonzero(d_s != 0)}/{d_s.size} "
              f"maxabs={np.abs(d_s).max():.3e} draft_s1={draft_s1} draft_s2={draft_s2}")

        # ---- Probe C: stateless vs cached (numerical-equivalence check) ----
        d_c = npf(dh_s1[0, -1]) - npf(dh[0, -1])
        print(f"[C] stateless vs cached dh: n={np.count_nonzero(d_c != 0)}/{d_c.size} "
              f"maxabs={np.abs(d_c).max():.3e}")
        break
    # continue the MTP loop (same as m7-mtp-probe)
    dh = lm.mtp(hidd, emb, mtp_cache)
    draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
    prev = t
    logits2, hidden2 = lm(
        mx.array([[prev, draft]]), cache=cache, return_hidden=True, return_prenorm=True
    )
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
        logits2, hidden2 = lm(
            mx.array([[prev]]), cache=cache, return_hidden=True, return_prenorm=True
        )
        logits2, hidden2 = lm(
            mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True
        )
        lg, hidd = logits2[0, -1], hidden2[0, -1]
print(f"done: n_out={len(out)} last={out[-8:]}")
