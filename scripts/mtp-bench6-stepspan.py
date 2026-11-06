#!/usr/bin/env python3
"""K=3 MTP spec-decode BENCHMARK probe (9B work model, smartphone prompt).

Extends mtp-bench5.py (K=2 arm) with a fifth mode. mtp-bench5.py is NOT
K-generic (each K is a hard-coded branch: draft-chain length, verify row
count S=K+1, rejection-cascade depth, rollback trim), so the K=3 arm is a
new file on the same structure - one added mode "4" branch.

MLX_MTP=0 -> plain greedy (byte-identical path to the baseline).
MLX_MTP=1 -> spec-decode K=1, verify = two 1-token (T=1) target forwards.
MLX_MTP=2 -> spec-decode K=1, verify = ONE batched 2-token (S=2) forward.
MLX_MTP=3 -> spec-decode K=2, verify = ONE batched 3-token (S=3) forward.
MLX_MTP=4 -> spec-decode K=3, verify = ONE batched 4-token (S=4) forward.

Env-mode mapping: the MLX_MTP values live in the BENCH scripts, not the
model. venv-fused2's mlx_lm (qwen3_5.py MTP_INJECT_PATCH) only builds the
1-layer MTP drafter module (config mtp_num_hidden_layers: 1); it reads no
MLX_MTP. The spec-decode loop (draft/verify/reject) is driven here.

K=3 draft chain (MTP module runs three times, own KV cache advances by 3):
    emb1 = embed(prev_token)
    dh1  = mtp(h, emb1, mtp_cache)          # h = target pre-final-norm hidden
    draft1 = argmax(lm_head(dh1))
    emb2 = embed(draft1)
    dh2  = mtp(dh1, emb2, mtp_cache)        # feed MTP's post-norm output back
    draft2 = argmax(lm_head(dh2))
    emb3 = embed(draft2)
    dh3  = mtp(dh2, emb3, mtp_cache)
    draft3 = argmax(lm_head(dh3))
(vLLM Qwen3-Next MTP dataflow: the module's returned hidden is the
hidden_states for the next draft step; pre_fc_norm_hidden is applied on top.)

K=3 verify (from S_k, drafts D1, D2, D3): ONE batched [1,4] forward of
[prev_token, D1, D2, D3]; S=4, cache advances 4.
    l0 (after prev_token), l1 (after D1), l2 (after D2), l3 (after D3)
    t2 = argmax(l0)  (true greedy token after prev_token)
    if t2 == EOS: break (lossless: EOS not appended, mirror plain mode)
    if t2 == D1 (draft1 accepted):
        emit D1; t3 = argmax(l1) (true greedy token after D1)
        if t3 == EOS: break (emit D1, stop, EOS not appended)
        if t3 == D2 (draft2 accepted):
            emit D2; t4 = argmax(l2) (true greedy token after D2)
            if t4 == EOS: break (emit D2, stop, EOS not appended)
            if t4 == D3 (draft3 accepted): emit D3; state = after [prev,D1,D2,D3]
            else (draft3 rejected): emit t4; rollback S=4 (restore GDN to S_k,
                trim KV by 4), re-forward [prev_token],[D1],[D2],[t4] (S=1 each)
        else (draft2 rejected): emit t3; rollback S=4 (restore GDN to S_k,
            trim KV by 4), re-forward [prev_token],[D1],[t3] (S=1 each)
    else (draft1 rejected): emit t2; rollback S=4 (restore GDN to S_k,
        trim KV by 4), re-forward [prev_token],[t2] (S=1 each)
S=4 = T=4 <= 8 -> recurrent GDN kernel path (no C++ change); QMV M=4 has
no fused M=4 specialization (the K=2 arm used the QMV M=3 path) - expect
a generic fallback for the verify step; the full-run numbers show the cost.

GDN state: save/restore exactly like mtp-bench3.py (functional-new-arrays,
zero-copy save valid). For MLX_MTP=4 the save point is BEFORE the batched
4-token forward. The MTP module's own KV cache is NOT rolled back on
rejection (same protocol as the K=2 arm; mtp-bench3-side.py is the
full-state-restore K=1 variant).

Prints: TOKS, TOKS_S, verifies, D1ACC/D2ACC/D3ACC, D1PCT/D2PCT/D3PCT,
TOK_PER_STEP, FIRST16, FIRST20, ALLTOKS, TEXT.
TOK_PER_STEP = 1 + (D1ACC+D2ACC+D3ACC)/VERIFIES (the K=2 arm's printed
formula omitted the D2 term; corrected here).
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

# t2246 seam: per-step GPU span via HIP events on the legacy default
# stream (span.so). SPAN=0 disables. span_end() syncs the GPU; the
# bench already syncs each step via int(mx.argmax(...)), so the
# extra cost is ~0.
import ctypes as _ct
_SP = None
if os.environ.get("SPAN", "1") == "1":
    _SP = _ct.CDLL(os.environ.get("SPAN_SO", os.path.expanduser("~/t2246/span.so")))
    _SP.span_init.argtypes = []
    _SP.span_init.restype = _ct.c_int
    _SP.span_begin.argtypes = []
    _SP.span_end.argtypes = []
    _SP.span_end.restype = _ct.c_double
    print("SPAN_INIT_OK", flush=True)
else:
    print("SPAN_DISABLED", flush=True)


def _span_begin():
    if _SP is not None:
        _SP.span_begin()


def _span_end():
    return _SP.span_end() if _SP is not None else -1.0


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
    assert mode in ("0", "1", "2", "3", "4"), mode
    mtp_on = mode in ("1", "2", "3", "4")
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
    draft3 = None
    prev_token = None
    out = []
    verifies = 0
    d1_acc = 0
    d2_acc = 0
    d3_acc = 0
    if _SP is not None:
        assert _SP.span_init() == 0, "span_init failed"

    t1 = time.time()
    for it in range(MAX_TOKENS):
        _ws = time.perf_counter()
        _kind = "ar" if (draft1 is None) else "verify"
        _span_begin()
        try:
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
                    if mode in ("3", "4"):
                        # Draft token 2 (from MTP module given token1)
                        emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
                        dh2 = lm.mtp(dh1, emb2, mtp_cache)
                        draft2 = int(mx.argmax(lm.lm_head(dh2)[0, -1]))
                    if mode == "4":
                        # Draft token 3 (from MTP module given token2)
                        emb3 = lm.model.embed_tokens(mx.array([[draft2]]))
                        dh3 = lm.mtp(dh2, emb3, mtp_cache)
                        draft3 = int(mx.argmax(lm.lm_head(dh3)[0, -1]))
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
                draft1 = draft2 = draft3 = None
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
                draft1 = draft2 = draft3 = None
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
            elif mode == "3":
                # Verify step (from S_k, drafts D1, D2). BATCHED S=3.
                d1, d2 = draft1, draft2
                draft1 = draft2 = draft3 = None
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
            else:
                # Verify step (from S_k, drafts D1, D2, D3). BATCHED S=4.
                d1, d2, d3 = draft1, draft2, draft3
                draft1 = draft2 = draft3 = None
                verifies += 1
                saved = save_gdn(cache)
                logits, hidden = lm(
                    mx.array([[prev_token, d1, d2, d3]]),
                    cache=cache,
                    return_hidden=True,
                    return_prenorm=True
                )
                l0, h0 = logits[0, 0], hidden[0, 0]
                l1, h1 = logits[0, 1], hidden[0, 1]
                l2, h2 = logits[0, 2], hidden[0, 2]
                l3, h3 = logits[0, 3], hidden[0, 3]
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
                        t4 = int(mx.argmax(l2))
                        if t4 == EOS:
                            break
                        if t4 == d3:
                            d3_acc += 1
                            out.append(t4)
                            l, h = l3, h3
                        else:
                            out.append(t4)
                            restore_gdn(cache, saved)
                            for c in cache:
                                if isinstance(c, KVCache):
                                    c.trim(4)
                            logits, hidden = lm(
                                mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True
                            )
                            logits, hidden = lm(
                                mx.array([[d1]]), cache=cache, return_hidden=True, return_prenorm=True
                            )
                            logits, hidden = lm(
                                mx.array([[d2]]), cache=cache, return_hidden=True, return_prenorm=True
                            )
                            logits, hidden = lm(
                                mx.array([[t4]]), cache=cache, return_hidden=True, return_prenorm=True
                            )
                            l, h = logits[0, -1], hidden[0, -1]
                    else:
                        out.append(t3)
                        restore_gdn(cache, saved)
                        for c in cache:
                            if isinstance(c, KVCache):
                                c.trim(4)
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
                            c.trim(4)
                    logits, hidden = lm(
                        mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    logits, hidden = lm(
                        mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True
                    )
                    l, h = logits[0, -1], hidden[0, -1]
        finally:
            _we = time.perf_counter()
            _span_ms = _SP.span_end() if _SP is not None else -1.0
            _wall_ms = (_we - _ws) * 1000.0
            _seam_ms = (_wall_ms - _span_ms) if _span_ms >= 0 else -1.0
            print(f"SPANSTEP {it} {_kind} wall={_wall_ms:.3f} span={_span_ms:.3f} seam={_seam_ms:.3f}", flush=True)
    dt = time.time() - t1
    n = len(out)
    d1pct = 100.0 * d1_acc / verifies if verifies else 0.0
    d2pct = 100.0 * d2_acc / d1_acc if d1_acc else 0.0
    d3pct = 100.0 * d3_acc / d2_acc if d2_acc else 0.0
    tok_step = 1.0 + (d1_acc + d2_acc + d3_acc) / verifies if verifies else 0.0
    print(
        "MTP=%s LOAD=%.2f GEN=%.2f TOKS=%d TOKS_S=%.2f VERIFIES=%d D1ACC=%d D2ACC=%d D3ACC=%d D1PCT=%.1f D2PCT=%.1f D3PCT=%.1f TOK_PER_STEP=%.3f"
        % (mode, t_load, dt, n, n / max(dt, 1e-9), verifies, d1_acc, d2_acc, d3_acc, d1pct, d2pct, d3pct, tok_step),
        flush=True,
    )
    print("FIRST16:", " ".join(map(str, out[:16])), flush=True)
    print("FIRST20:", " ".join(map(str, out[:20])), flush=True)
    print("ALLTOKS:", " ".join(map(str, out)), flush=True)
    print("TEXT:", tok.decode(out), flush=True)


if __name__ == "__main__":
    main()
