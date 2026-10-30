#!/usr/bin/env python3
"""K=2 MTP spec-decode BENCHMARK probe (9B work model, smartphone prompt).

Extends /tmp/mtp-bench3.py (K=1) with a fourth mode:

MLX_MTP=0 -> plain greedy (byte-identical path to the baseline).
MLX_MTP=1 -> spec-decode K=1, verify = two 1-token (T=1) target forwards.
MLX_MTP=2 -> spec-decode K=1, verify = ONE batched 2-token (S=2) forward.
MLX_MTP=3 -> spec-decode K=2, verify = ONE batched 3-token (S=3) forward.

K=2 draft chain (MTP module runs twice, own KV cache advances by 2):
    emb1 = embed(prev_token)
    dh1  = mtp(h, emb1, mtp_cache)          # h = target pre-final-norm hidden
    draft1 = argmax(lm_head(dh1))
    emb2 = embed(draft1)
    dh2  = mtp(dh1, emb2, mtp_cache)        # feed MTP's post-norm output back
    draft2 = argmax(lm_head(dh2))
(vLLM Qwen3-Next MTP dataflow: the module's returned hidden is the
hidden_states for the next draft step; pre_fc_norm_hidden is applied on top.)

K=2 verify (from S_k, drafts D1, D2): ONE batched [1,3] forward of
[prev_token, D1, D2]; S=3, cache advances 3.
    l0 (after prev_token), l1 (after D1), l2 (after D2)
    t2 = argmax(l0)  (true greedy token after prev_token)
    if t2 == EOS: break (lossless: EOS not appended, mirror plain mode)
    if t2 == D1 (draft1 accepted):
        emit D1; t3 = argmax(l1) (true greedy token after D1)
        if t3 == EOS: break (emit D1, stop, EOS not appended)
        if t3 == D2 (draft2 accepted): emit D2; state = after [prev,D1,D2]
        else (draft2 rejected): emit t3; rollback S=3 (restore GDN to S_k,
            trim KV by 3), re-forward [prev_token],[D1],[t3] (S=1 each)
    else (draft1 rejected): emit t2; rollback S=3 (restore GDN to S_k,
        trim KV by 3), re-forward [prev_token],[t2] (S=1 each)
S=3 = T=3 <= 8 -> recurrent GDN kernel + QMV M=3 path (no C++ change).

GDN state: save/restore exactly like mtp-bench3.py (functional-new-arrays,
zero-copy save valid). For MLX_MTP=3 the save point is BEFORE the batched
3-token forward.

Prints: TOKS, TOKS_S, ACCEPTED (draft1/draft2), verifies, D1ACC, D2ACC,
TOK_PER_STEP, stream.
"""
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
MODELS = os.environ.get("MODELS", "/models")

MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = "Explain in great detail how a modern smartphone works, from the hardware components to the operating system, covering every subsystem. Be exhaustive."
MAX_TOKENS = int(os.environ.get("MTP_MAX_TOKENS", "1024"))
EOS = 248044


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
    mode = os.environ.get("MLX_MTP", "0")
    assert mode in ("0", "1", "2", "3"), mode
    mtp_on = mode in ("1", "2", "3")
    t0 = time.time()
    model, tok = load(MODEL)
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())
    if mtp_on:
        assert lm.mtp is not None, "no mtp module (config.mtp_num_hidden_layers?)"
        mtp_cache = [KVCache()]

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()

    logits, hidden = lm(
        mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True
    )
    l = logits[0, -1]
    h = hidden[0, -1]
    draft1 = None
    draft2 = None
    prev_token = None
    out = []
    verifies = 0
    d1_acc = 0
    d2_acc = 0
    t1 = time.time()
    for _ in range(MAX_TOKENS):
        if draft1 is None:
            # Single-token step (from S_k)
            t = int(mx.argmax(l))
            if t == EOS:
                break
            out.append(t)
            if mtp_on:
                prev_token = t
                # Draft token 1 (from AR state)
                emb1 = lm.model.embed_tokens(mx.array([[t]]))
                dh1 = lm.mtp(h, emb1, mtp_cache)
                draft1 = int(mx.argmax(lm.lm_head(dh1)[0, -1]))
                if mode == "3":
                    # Draft token 2 (from MTP module given token1)
                    emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
                    dh2 = lm.mtp(dh1, emb2, mtp_cache)
                    draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
                # State stays S_k (no target forward)
            else:
                # baseline: advance the state with t
                logits, hidden = lm(
                    mx.array([[t]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                l = logits[0, -1]
                h = hidden[0, -1]
        elif mode == "1":
            # Verify step (from S_k, draft D1). Two 1-token forwards.
            d = draft1
            draft1 = draft2 = None
            verifies += 1
            logits, hidden = lm(
                mx.array([[prev_token]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, -1], hidden[0, -1]
            saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[d]]), cache=cache, return_hidden=True, return_prenorm=True
            )
            l1, h1 = logits[0, -1], hidden[0, -1]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d:
                d1_acc += 1
                out.append(t2)
                l, h = l1, h1
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(1)
                out.append(t2)
                logits, hidden = lm(
                    mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True
                )
                l, h = logits[0, -1], hidden[0, -1]
        elif mode == "2":
            # Verify step (from S_k, draft D1). BATCHED S=2.
            d = draft1
            draft1 = draft2 = None
            verifies += 1
            saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d:
                d1_acc += 1
                out.append(t2)
                l, h = l1, h1
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)
                out.append(t2)
                logits, hidden = lm(
                    mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True
                )
                logits, hidden = lm(
                    mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True
                )
                l, h = logits[0, -1], hidden[0, -1]
        else:
            # Verify step (from S_k, drafts D1, D2). BATCHED S=3.
            d1, d2 = draft1, draft2
            draft1 = draft2 = None
            verifies += 1
            saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            l2, h2 = logits[0, 2], hidden[0, 2]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d1:
                d1_acc += 1
                out.append(t2)
                t3 = int(mx.argmax(l1))
                if t3 == EOS:
                    break
                if t3 == d2:
                    d2_acc += 1
                    out.append(t3)
                    l, h = l2, h2
                else:
                    out.append(t3)
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(3)
                    logits, hidden = lm(
                        mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    logits, hidden = lm(
                        mx.array([[d1]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    logits, hidden = lm(
                        mx.array([[t3]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    l, h = logits[0, -1], hidden[0, -1]
            else:
                out.append(t2)
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(3)
                logits, hidden = lm(
                    mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True
                )
                logits, hidden = lm(
                    mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True
                )
                l, h = logits[0, -1], hidden[0, -1]
    dt = time.time() - t1
    n = len(out)
    d1pct = 100.0 * d1_acc / verifies if verifies else 0.0
    d2pct = 100.0 * d2_acc / d1_acc if d1_acc else 0.0
    tok_step = 1.0 + (d1_acc / verifies) if verifies else 0.0
    print(
        "MTP=%s LOAD=%.2f GEN=%.2f TOKS=%d TOKS_S=%.2f VERIFIES=%d D1ACC=%d D2ACC=%d D1PCT=%.1f D2PCT=%.1f TOK_PER_STEP=%.3f"
        % (mode, t_load, dt, n, n / max(dt, 1e-9), verifies, d1_acc, d2_acc, d1pct, d2pct, tok_step),
        flush=True,
    )
    print("FIRST16:", " ".join(map(str, out[:16])), flush=True)
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)
    print("TEXT:", tok.decode(out), flush=True)


if __name__ == "__main__":
    main()
