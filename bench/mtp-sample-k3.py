#!/usr/bin/env python3
"""K=3 sampling gate (t1307, statistical bar / MTPLX-style distribution
parity). Same loop as mtp-bench-k3.py (target S=4 verify — the path where
the rows kernel engages at M=4), but the target token at each position is
SAMPLED instead of argmax: temperature 1.0, top-p 0.95, one shared
mx.random stream per run (seeded).

Env: MX_SEED (required), MTP_MODEL, MODELS. Arm env (wrapper-set):
MLX_ROCM_QMV_ROWS / MLX_ROCM_QMV_ROWS_MIN_N, MLX_ROCM_GDN_TAPE=1,
MLX_ROCM_GDN_FUSE=1, MLX_ROCM_SDPA_S4VA=1.

Output: SAMPSTATS line + ALLTOKS (token ids, for overlap metrics).
"""

import os
import time

import mlx.core as mx

from mlx_lm import load
from mlx_lm.cache_types import KVCache

MODEL = os.environ.get("MTP_MODEL", f"{os.environ.get('MODELS', '/mnt/models')}/qwen3-9b-mlx")
PROMPT = "Explain in great detail how a modern smartphone works, from the hardware components to the operating system, covering every subsystem. Be exhaustive."
MAX_TOKENS = int(os.environ.get("MTP_MAX_TOKENS", "256"))
TEMP = 1.0
TOP_P = 0.95
EOS = 248044
SEED = int(os.environ["MX_SEED"])


def sample(logits):
    """One target sample: temperature 1.0 + top-p 0.95, shared stream."""
    p = mx.softmax(logits / TEMP)
    # top-p mask: keep the smallest set whose cumulative mass >= TOP_P
    sorted_p, idx = mx.sort(p, descending=True)
    csum = mx.cumsum(sorted_p)
    cutoff = csum - sorted_p + TOP_P  # keep where csum - p < top_p
    keep = (csum <= cutoff) | (csum - sorted_p < TOP_P)
    masked = mx.where(keep, sorted_p, 0.0)
    masked = masked / mx.sum(masked)
    i = mx.random.categorical(masked)
    return int(idx[i])


def save_gdn(cache):
    return [
        (i, c[0], c[1], c.lengths)
        for i, c in enumerate(cache)
        if hasattr(c, "cache")
    ]


def restore_gdn(cache, saved):
    for i, conv, ssm, lengths in saved:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths


def main():
    os.environ["MLX_MTP"] = "4"
    mx.random.seed(SEED)
    t0 = time.time()
    model, tok = load(MODEL)
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())
    assert lm.mtp is not None, "no mtp module"
    mtp_cache = [KVCache()]

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache,
                       return_hidden=True, return_prenorm=True)
    mx.eval(logits, hidden)
    l = logits[0, -1]
    h = hidden[0, -1]
    prev_token = sample(l)
    out = [prev_token]
    verifies = 0
    d1_acc = d2_acc = d3_acc = 0
    t1 = time.perf_counter()

    for _ in range(MAX_TOKENS):
        if len(out) >= MAX_TOKENS:
            break
        # Single-token step: target sample at the carried position, then
        # the MTP drafter chain (S=1 — rows never engages here by design).
        t = sample(l)
        if t == EOS:
            break
        out.append(t)
        prev_token = t
        emb1 = lm.model.embed_tokens(mx.array([[t]]))
        dh1 = lm.mtp(h, emb1, mtp_cache)
        mx.eval(dh1)
        d1 = int(mx.argmax(lm.lm_head(dh1)[0, -1]))
        emb2 = lm.model.embed_tokens(mx.array([[d1]]))
        dh2 = lm.mtp(dh1, emb2, mtp_cache)
        mx.eval(dh2)
        d2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
        emb3 = lm.model.embed_tokens(mx.array([[d2]]))
        dh3 = lm.mtp(dh2, emb3, mtp_cache)
        mx.eval(dh3)
        d3 = int(mx.argmax(lm.lm_head(dh3)[0, -1]))
        # Verify (S=4 target — rows engages here when ON)
        verifies += 1
        saved = save_gdn(cache)
        vlogits, vhidden = lm(mx.array([[prev_token, d1, d2, d3]]),
                            cache=cache, return_hidden=True, return_prenorm=True)
        mx.eval(vlogits, vhidden)
        l0, h0 = vlogits[0, 0], vhidden[0, 0]
        l1v, h1v = vlogits[0, 1], vhidden[0, 1]
        l2v, h2v = vlogits[0, 2], vhidden[0, 2]
        l3v, h3v = vlogits[0, 3], vhidden[0, 3]
        t2 = sample(l0)
        if t2 == EOS:
            break
        if t2 == d1:
            d1_acc += 1
            out.append(t2)
            t3 = sample(l1v)
            if t3 == EOS:
                break
            if t3 == d2:
                d2_acc += 1
                out.append(t3)
                t4 = sample(l2v)
                if t4 == EOS:
                    break
                if t4 == d3:
                    d3_acc += 1
                    out.append(t4)
                    l, h = l3v, h3v
                else:
                    out.append(t4)
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(4)
                    for tok_ in (prev_token, d1, d2, t4):
                        lg, hh = lm(mx.array([[tok_]]), cache=cache,
                                   return_hidden=True, return_prenorm=True)
                        mx.eval(lg, hh)
                    l, h = lg[0, -1], hh[0, -1]
            else:
                out.append(t3)
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(4)
                for tok_ in (prev_token, d1, t3):
                    lg, hh = lm(mx.array([[tok_]]), cache=cache,
                               return_hidden=True, return_prenorm=True)
                    mx.eval(lg, hh)
                l, h = lg[0, -1], hh[0, -1]
        else:
            out.append(t2)
            restore_gdn(cache, saved)
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(4)
            for tok_ in (prev_token, t2):
                lg, hh = lm(mx.array([[tok_]]), cache=cache,
                           return_hidden=True, return_prenorm=True)
                mx.eval(lg, hh)
            l, h = lg[0, -1], hh[0, -1]
        prev_token = out[-1]

    wall = time.perf_counter() - t1
    n = len(out)
    d1pct = 100.0 * d1_acc / verifies if verifies else 0.0
    d2pct = 100.0 * d2_acc / d1_acc if d1_acc else 0.0
    d3pct = 100.0 * d3_acc / d2_acc if d2_acc else 0.0
    print(
        f"SAMPSTATS: seed={SEED} load={t_load:.1f} gen={wall:.2f} toks={n} "
        f"toks_s={n / wall:.2f} verifies={verifies} "
        f"D1PCT={d1pct:.1f} D2PCT={d2pct:.1f} D3PCT={d3pct:.1f}",
        flush=True,
    )
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)


if __name__ == "__main__":
    main()
