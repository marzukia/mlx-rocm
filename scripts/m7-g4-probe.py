"""m7: G4 lossless diagnostic - compare MTP-path state vs AR-path state.

Runs the MTP spec-decode flow (mode 2, batched verify) and at each verify
step checks whether the target model's state (GDN + KV) after the verify
step matches the state that two separate AR forwards ([t] then [t2]) would
produce. Reports the first step where the GDN or KV state diverges.

If the MTP-path state always matches the AR-path state, the emitted tokens
must be bit-exact with AR (lossless). A divergence here localizes the G4
first-divergence (token 148) to a specific step + component.
"""
import os
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = ("Explain in great detail how a modern smartphone works, from the "
          "hardware components to the operating system, covering every "
          "subsystem. Be exhaustive.")
MAX_TOKENS = 200
EOS = 248044


def npf(a):
    return np.asarray(mx.array(a).astype(mx.float32)).copy() if a is not None else None


def save_gdn(cache):
    return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]


def restore_gdn(cache, saved):
    for i, conv, ssm, lengths in saved:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths


def snap_all(cache):
    """Snapshot the FULL target state: GDN (conv, ssm, lengths) + KV (keys, values, offset)."""
    gdn = [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]
    kv = [(i, npf(c.keys), npf(c.values), c.offset) for i, c in enumerate(cache) if isinstance(c, KVCache)]
    return gdn, kv


def restore_all(cache, snap):
    gdn, kv = snap
    for i, conv, ssm, lengths in gdn:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths
    for i, keys, values, offset in kv:
        cache[i].keys = mx.array(keys)
        cache[i].values = mx.array(values)
        cache[i].offset = offset


def cmp_state(snap_a, snap_b, tag, report):
    """Compare two full-state snapshots; append any divergence to report."""
    gdn_a, kv_a = snap_a
    gdn_b, kv_b = snap_b
    for (ia, ca, sa, la), (ib, cb, sb, lb) in zip(gdn_a, gdn_b):
        for j, (a, b, name) in enumerate([(ca, cb, "conv"), (sa, sb, "ssm")]):
            if a is None or b is None:
                continue
            d = npf(a) - npf(b)
            n = np.count_nonzero(d != 0)
            if n > 0 and tag not in report:
                report[tag] = (n, d.size, float(np.abs(d).max()), name)
    for (ia, ka, va, oa), (ib, kb, vb, ob) in zip(kv_a, kv_b):
        for a, b, name in [(ka, kb, "keys"), (va, vb, "values")]:
            d = npf(a) - npf(b)
            n = np.count_nonzero(d != 0)
            if n > 0 and tag not in report:
                report[tag] = (n, d.size, float(np.abs(d).max()), name)
        if oa != ob and tag not in report:
            report[tag] = (1, 1, float(abs(oa - ob)), "offset")


model, tok = load(MODEL)
lm = model.language_model
mx.eval(model.parameters())
assert lm.mtp is not None
mtp_cache = [KVCache()]
prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
cache = lm.make_cache()
logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
l = logits[0, -1]
h = hidden[0, -1]
draft = None
prev_token = None
out = []
accepted = 0
verifies = 0
report = {}
first_div_step = None

for _ in range(MAX_TOKENS):
    if draft is None:
        t = int(mx.argmax(l))
        if t == EOS:
            break
        out.append(t)
        emb = lm.model.embed_tokens(mx.array([[t]]))
        dh = lm.mtp(h, emb, mtp_cache)
        draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
        prev_token = t
    else:
        d = draft
        draft = None
        verifies += 1
        saved = save_gdn(cache)
        snap_before = snap_all(cache)
        logits, hidden = lm(mx.array([[prev_token, d]]), cache=cache, return_hidden=True, return_prenorm=True)
        l0, h0 = logits[0, 0], hidden[0, 0]
        l1, h1 = logits[0, 1], hidden[0, 1]
        t2 = int(mx.argmax(l0))
        if t2 == EOS:
            break
        # MTP-path state after the verify step (accept: S_{k+2}; reject: will re-forward)
        if t2 == d:
            accepted += 1
            out.append(t2)
            snap_mtp = snap_all(cache)
            l, h = l1, h1
        else:
            restore_gdn(cache, saved)
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(2)
            out.append(t2)
            logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
            logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
            snap_mtp = snap_all(cache)
            l, h = logits[0, -1], hidden[0, -1]
        # AR-path state: roll back to before the verify step, then [prev_token] then [t2]
        restore_all(cache, snap_before)
        logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
        logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
        snap_ar = snap_all(cache)
        # compare
        cmp_state(snap_mtp, snap_ar, f"v{verifies}", report)
        if f"v{verifies}" in report and first_div_step is None:
            first_div_step = verifies
            print(f"FIRST-DIV verify-step={verifies} (out_len={len(out)}) "
                  f"n={report[f'v{verifies}'][0]}/{report[f'v{verifies}'][1]} "
                  f"maxabs={report[f'v{verifies}'][2]:.3e} comp={report[f'v{verifies}'][3]}")
        # restore the MTP-path state so the flow can continue
        restore_all(cache, snap_mtp)
        # re-eval l, h from the MTP-path state (re-forward to get fresh l, h)
        # (l, h were captured from the MTP-path forward above; they are still valid)
    if len(out) >= MAX_TOKENS:
        break

print(f"done: n_out={len(out)} accepted={accepted}/{verifies} first_div_step={first_div_step}")
print(f"last16: {out[-16:]}")
if report:
    print(f"diverging steps: {sorted(report.keys(), key=lambda x: int(x[1:]))}")
