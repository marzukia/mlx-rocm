#!/usr/bin/env python3
"""K=1 MTP spec-decode BENCHMARK probe (9B work model, stdref fruits prompt).

Copy of /tmp/mtp-probe2.py with MAX_TOKENS=1024 (stop at EOS) and a third
mode:

MLX_MTP=0 -> plain greedy (byte-identical path to the baseline).
MLX_MTP=1 -> spec-decode, verify = two 1-token target forwards with GDN
  save/restore mid-verify (the existing gate-probe form).
MLX_MTP=2 -> spec-decode, verify = BATCHED: one target forward of
  [prev_token, draft] (shape [1,2], S=2, cache advances 2) giving l0 (after
  prev_token) and l1 (after draft) in a single call.
    ACCEPT (t2==draft): state stays (S=2 forward already applied both),
      emit 2 tokens (prev_token was emitted in the single-token step; emit
      t2=draft).
    REJECT: t2=argmax(l0); restore GDN state saved BEFORE the batched
      forward; trim KV by 2 (removes prev_token AND draft); re-forward
      [prev_token] then [t2] (S=1 each) to land the correct state - same
      math as the mode-1 reject path.
Draft step unchanged (MTP 1-layer, argmax). Greedy, deterministic.

GDN state: the trunk GDN (gated delta net) recurrent state is saved/
restored exactly like mtp-probe2.py (functional-new-arrays per forward,
zero-copy save valid). For MLX_MTP=2 the save point is BEFORE the batched
2-token forward.

MTP input hidden = target PRE-final-norm hidden (vLLM Qwen3-Next MTP
dataflow: pre_fc_norm_hidden runs on it; feeding post-norm
double-normalizes).

Prints: TOKS, TOKS_S, ACCEPTED, verifies, stream.
"""
import os
import time

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
MODELS = os.environ.get("MODELS", "/models")

MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = "Explain in great detail how a modern smartphone works, from the hardware components to the operating system, covering every subsystem. Be exhaustive."
MAX_TOKENS = 1024
EOS = 248044


def save_gdn(cache):
    """Reference-save GDN states. gated_delta_update is functional (new
    arrays), so the pre-forward states stay alive: zero copies. lengths
    is per-layer ArraysCache state (advance(S) per forward); the conv
    update positions off it, so it must be saved/restored too."""
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


def save_mtp(mtp_cache):
    """m7: snapshot the MTP module's FULL state (keys, values, AND
    offset). The MTP block is full-attention (no GDN state), so its
    entire state is the KV cache. The offset is load-bearing: the
    attention uses cache.offset for the RoPE position and the k/v
    write slot. Restoring keys/values without the offset makes the
    next MTP call run at the wrong RoPE position and attend over a
    stale/zero slot (t14: 4096/4096 drift, maxabs 5.452e-01;
    a5af599: full-state restore is bit-exact, n=0/4096)."""
    if mtp_cache is None or len(mtp_cache) == 0:
        return None
    c = mtp_cache[0]
    if c.keys is None:
        return (None, None, c.offset)
    return (
        np.asarray(mx.array(c.keys).astype(mx.float32)).copy(),
        np.asarray(mx.array(c.values).astype(mx.float32)).copy(),
        c.offset,
    )


def restore_mtp(mtp_cache, saved):
    if saved is None or mtp_cache is None or len(mtp_cache) == 0:
        return
    keys, values, offset = saved
    c = mtp_cache[0]
    if keys is not None:
        c.keys = mx.array(keys)
        c.values = mx.array(values)
    c.offset = offset


def main():
    mode = os.environ.get("MLX_MTP", "0")
    assert mode in ("0", "1", "2"), mode
    mtp_on = mode in ("1", "2")
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
    draft = None
    prev_token = None  # token to process in the verify step
    out = []
    accepted = 0
    verifies = 0
    t1 = time.time()
    mtp_saved = None  # m7: FULL MTP state snapshot (keys+values+offset), taken before the draft step
    for _ in range(MAX_TOKENS):
        if draft is None:
            # Single-token step (from S_k)
            t = int(mx.argmax(l))
            if t == EOS:
                break
            out.append(t)
            if mtp_on:
                mtp_saved = save_mtp(mtp_cache)  # m7: snapshot BEFORE the draft entry is written
                emb = lm.model.embed_tokens(mx.array([[t]]))
                dh = lm.mtp(h, emb, mtp_cache)
                draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
                prev_token = t
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
            # Verify step (from S_k, draft D). Two 1-token forwards so the
            # GDN recurrent state can be saved after prev_token (mid-verify).
            d = draft
            draft = None
            verifies += 1
            # Forward 1: target(prev_token) -> S_{k+1}
            logits, hidden = lm(
                mx.array([[prev_token]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, -1], hidden[0, -1]   # after prev_token
            saved = save_gdn(cache)                 # GDN state after prev_token
            # Forward 2: target(d) -> S'
            logits, hidden = lm(
                mx.array([[d]]), cache=cache, return_hidden=True, return_prenorm=True
            )
            l1, h1 = logits[0, -1], hidden[0, -1]   # after d
            t2 = int(mx.argmax(l0))                 # true greedy token after prev_token
            if t2 == EOS:                          # lossless: EOS not appended (mirror plain mode)
                break
            if t2 == d:                          # accept
                accepted += 1
                out.append(t2)
                l, h = l1, h1                   # state S' (after [prev_token, d]); no rollback
            else:                                 # reject: target's own token proceeds
                restore_gdn(cache, saved)         # GDN state -> after prev_token
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(1)                # remove d from attention
                out.append(t2)
                logits, hidden = lm(
                    mx.array([[t2]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                l, h = logits[0, -1], hidden[0, -1]
            draft = None
        else:
            # Verify step (from S_k, draft D). BATCHED: one [1,2] forward of
            # [prev_token, d]; S=2, cache advances 2.
            d = draft
            draft = None
            verifies += 1
            saved = save_gdn(cache)                 # GDN state BEFORE batched forward (S_k)
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, 0], hidden[0, 0]     # after prev_token
            l1, h1 = logits[0, 1], hidden[0, 1]     # after d
            t2 = int(mx.argmax(l0))                 # true greedy token after prev_token
            if t2 == EOS:                          # lossless: EOS not appended (mirror plain mode)
                break
            if t2 == d:                          # accept
                accepted += 1
                out.append(t2)
                l, h = l1, h1                   # state S' (after [prev_token, d]); no rollback
            else:                                 # reject: roll the S=2 forward back,
                restore_gdn(cache, saved)         # GDN state -> S_k (before batched forward)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)                # remove prev_token AND d from attention
                restore_mtp(mtp_cache, mtp_saved)  # m7: restore FULL MTP state (keys+values+offset)
                out.append(t2)
                # Re-forward [prev_token] (S=1) -> S_{k+1}, then [t2] (S=1) -> S'';
                # same math as the mode-1 reject path.
                logits, hidden = lm(
                    mx.array([[prev_token]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                logits, hidden = lm(
                    mx.array([[t2]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True,
                )
                l, h = logits[0, -1], hidden[0, -1]
            draft = None
    dt = time.time() - t1
    n = len(out)
    print(
        "MTP=%s LOAD=%.2f GEN=%.2f TOKS=%d TOKS_S=%.2f ACCEPTED=%d/%d"
        % (mode, t_load, dt, n, n / max(dt, 1e-9), accepted, verifies),
        flush=True,
    )
    print("FIRST16:", " ".join(map(str, out[:16])), flush=True)
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)
    print("TEXT:", tok.decode(out), flush=True)


if __name__ == "__main__":
    main()
