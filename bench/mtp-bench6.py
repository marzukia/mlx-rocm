#!/usr/bin/env python3
"""K=3 MTP spec-decode BENCHMARK probe (9B work model, smartphone prompt).

Extends mtp-bench5.py (K=2 arm) with a fifth mode. mtp-bench5.py is NOT K-generic (each K is a hard-coded branch: draft-chain length, verify row count S=K+1, rejection-cascade depth, rollback trim), so the K=3 arm is a new file on the same structure - one added mode "4" branch.

MLX_MTP=0 -> plain greedy (byte-identical path to the baseline).
MLX_MTP=1 -> spec-decode K=1, verify = two 1-token (T=1) target forwards.
MLX_MTP=2 -> spec-decode K=1, verify = ONE batched 2-token (S=2) forward.
MLX_MTP=3 -> spec-decode K=2, verify = ONE batched 3-token (S=3) forward.
MLX_MTP=4 -> spec-decode K=3, verify = ONE batched 4-token (S=4) forward.

Env-mode mapping: the MLX_MTP values live in the BENCH scripts, not the model. venv-fused2's mlx_lm (qwen3_5.py MTP_INJECT_PATCH) only builds the 1-layer MTP drafter module (config mtp_num_hidden_layers: 1); it reads no MLX_MTP. The spec-decode loop (draft/verify/reject) is driven here.

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
(vLLM Qwen3-Next MTP dataflow: the module's returned hidden is the hidden_states for the next draft step; pre_fc_norm_hidden is applied on top.)

K=3 verify (from S_k, drafts D1, D2, D3): ONE batched [1,4] forward of [prev_token, D1, D2, D3]; S=4, cache advances 4.
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
            else (draft3 rejected): emit t4; rollback S=4 (restore GDN to S_k, trim KV by 4), re-forward [prev_token],[D1],[D2],[t4] (S=1 each)
        else (draft2 rejected): emit t3; rollback S=4 (restore GDN to S_k, trim KV by 4), re-forward [prev_token],[D1],[t3] (S=1 each)
    else (draft1 rejected): emit t2; rollback S=4 (restore GDN to S_k, trim KV by 4), re-forward [prev_token],[t2] (S=1 each)
S=4 = T=4 <= 8 -> recurrent GDN kernel path (no C++ change); QMV M=4 has no fused M=4 specialization (the K=2 arm used the QMV M=3 path) - expect a generic fallback for the verify step; the full-run numbers show the cost.

GDN state: save/restore exactly like mtp-bench3.py (functional-new-arrays, zero-copy save valid). For MLX_MTP=4 the save point is BEFORE the batched 4-token forward. The MTP module's own KV cache is NOT rolled back on rejection (same protocol as the K=2 arm; mtp-bench3-side.py is the full-state-restore K=1 variant).

Prints: TOKS, TOKS_S, verifies, D1ACC/D2ACC/D3ACC, D1PCT/D2PCT/D3PCT, TOK_PER_STEP, FIRST16, FIRST20, ALLTOKS, TEXT.
TOK_PER_STEP = 1 + (D1ACC+D2ACC+D3ACC)/VERIFIES (the K=2 arm's printed formula omitted the D2 term; corrected here).
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


def quantize_int2(w, chunk=8192):
    """Int2 g128-group asymmetric quantization of the bf16 lm_head weights
    (quarter-split 2-bit packing, radiance port): per 128-group, s =
    max((hi-lo)/3, 1e-8), z = clamp(round(-lo/s), 0, 3), q =
    clamp(round(w/s + z), 0, 3); byte j holds v0..v3 for x indices j,
    j+Q, j+2Q, j+3Q (Q = K/4). Returns (packed u8 [N, K/4], scale f32
    [N, K/128] = s, zs f32 [N, K/128] = z*s)."""
    N, K = w.shape
    NG, Q = K // 128, K // 4
    packed = mx.zeros((N, Q), dtype=mx.int32)
    scale = mx.zeros((N, NG), dtype=mx.float32)
    zs = mx.zeros((N, NG), dtype=mx.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        wf = w[i:j].astype(mx.float32).reshape(j - i, NG, 128)
        lo = wf.min(axis=2)
        hi = wf.max(axis=2)
        s = mx.maximum((hi - lo) / 3.0, 1e-8)
        z = mx.minimum(mx.maximum(mx.round(-lo / s), 0.0), 3.0)
        q = (
            mx.minimum(
                mx.maximum(
                    mx.round(wf / s[..., None] + z[..., None]), 0.0
                ),
                3.0,
            )
            .astype(mx.int32)
            .reshape(j - i, K)
        )
        packed[i:j] = q[:, :Q] | (q[:, Q:2*Q] << 2) | (q[:, 2*Q:3*Q] << 4) | (q[:, 3*Q:] << 6)
        scale[i:j] = s
        zs[i:j] = z * s
    return packed.astype(mx.uint8), scale, zs


def int2_argmax(x, qh):
    """Int2 draft head: int2 g128-asym dequant GEMV (mx.fast.lm_head_int2)
    + exact bf16 re-rank of the global top-32 candidates (in-kernel per-64-column top-8 emission). Returns the argmax token id."""
    x = x.reshape(1, -1)
    K = x.shape[1]
    NG = K // 128
    xf = x.reshape(NG, 128).astype(mx.float32)
    sx = mx.sum(xf, axis=1)
    logits, cval, ccol = mx.fast.lm_head_int2(
        x, qh["packed"], qh["scale"], qh["zs"], sx
    )
    perm = mx.argsort(cval.reshape(-1))
    idx = mx.take(ccol.reshape(-1), perm[-32:])
    if "w" in qh:
        cols = mx.take(qh["w"], idx, 0)
    else:
        # Re-dequantize exactly the 32 candidate rows (bit-exact vs the target's 4-bit head; avoids keeping a 2 GB bf16 copy).
        cols = mx.dequantize(
            mx.take(qh["wq"], idx, 0),
            mx.take(qh["s4"], idx, 0),
            mx.take(qh["b4"], idx, 0),
            group_size=qh["gs"],
            bits=qh["bits"],
        )
    exact = mx.sum(x.astype(mx.float32) * cols.astype(mx.float32), axis=1)
    return int(mx.take(idx, mx.argmax(exact)))


def main():
    mode = os.environ.get("MLX_MTP", "0")
    assert mode in ("0", "1", "2", "3", "4"), mode
    mtp_on = mode in ("1", "2", "3", "4")
    if mode == "4":
        # FIX#1: GDN state-tape commit (the model's TAPE branch is gated on these; S==4 forwards go through gdn_fused_step_tape in the wheel model file; S=1 correction forwards use the fused op). MTP6_TAPE=0      -> savestate: canonical save/restore GDN + re-forward of the accepted prefix (baseline). MTP6_TAPE=1      -> pure tape (round-1 measured, 39.16 class): tape commit + ONE S=1 correction forward (state advance), but l = the batched row logits and h = the batched S=4 row h (the next draft chain reuses the batched-row h; the correction token is emitted in the branch and re-emitted by the next argmax(l)). MTP6_TAPE=hybrid -> round-2 hybrid: tape commit + ONE S=1 correction forward; l and h are the correction forward's own (S=1 sequential pre-norm h for the next draft chain).
        # MTP6_TAPE=r5b -> R5b: r5 + the commit rebind MATERIALIZES the selected tape row (a fresh contiguous array) instead of rebinding the strided row-slice view of the tape batch.
        # MTP6_TAPE=r5b -> R5b: r5 + the commit rebind MATERIALIZES the selected tape row (a fresh contiguous array) instead of rebinding the strided row-slice view of the tape batch.
        # MTP6_TAPE=r5e -> R5e: r5 + the S=4 verify GEMM is forced in the verify window (mx.eval after the verify forward) instead of by the first argmax in the commit window (the deferred-eval-order diagnostic/fix).
        tape_mode = os.environ.get("MTP6_TAPE", "1")
        assert tape_mode in ("0", "1", "hybrid", "r5", "r5b", "r5e"), tape_mode
        if tape_mode in ("1", "hybrid", "r5", "r5b", "r5e"):
            os.environ["MLX_ROCM_GDN_TAPE"] = "1"
        os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    t0 = time.time()
    model, tok = load(MODEL)
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())
    if mtp_on:
        assert lm.mtp is not None, "no mtp module (config.mtp_num_hidden_layers?)"
        mtp_cache = [KVCache()]
    qh = None
    if mode == "4":
        # FIX#2: int2 g128-asymmetric draft head (replaces the full bf16 lm_head per draft; MTP6_DRAFT_HEAD=bf16 to A/B the old head).
        if os.environ.get("MTP6_DRAFT_HEAD", "int2") == "int2" and hasattr(
            mx.fast, "lm_head_int2"
        ):
            tq0 = time.time()
            _h = lm.lm_head
            if type(_h).__name__ == "QuantizedLinear":
                w_bf16 = mx.dequantize(
                    _h.weight, _h.scales, _h.biases, group_size=_h.group_size, bits=_h.bits
                )
            else:
                w_bf16 = _h.weight
            packed, scale, zs = quantize_int2(w_bf16)
            mx.eval(packed, scale, zs)
            n_rows, n_cols = w_bf16.shape
            if type(_h).__name__ == "QuantizedLinear":
                # 16 GB card: do NOT keep the 2 GB bf16 copy; re-rank by re-dequantizing the 32 candidate rows.
                del w_bf16
                mx.clear_cache()
                qh = {
                    "packed": packed, "scale": scale, "zs": zs,
                    "wq": _h.weight, "s4": _h.scales, "b4": _h.biases,
                    "bits": _h.bits, "gs": _h.group_size,
                }
            else:
                qh = {"packed": packed, "scale": scale, "zs": zs, "w": w_bf16}
            print(
                "INT2_BUILD %.1fs N=%d K=%d (packed %d MB)" % (
                    time.time() - tq0,
                    n_rows,
                    n_cols,
                    packed.nbytes / (1024 * 1024),
                ),
                flush=True,
            )

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
    n_refwd = 0
    n_cdeferred = 0
    pending_c = None
    h_r = None

    def draft_tok(dh):
        return int2_argmax(dh, qh) if qh is not None else int(
            mx.argmax(lm.lm_head(dh)[0, -1])
        )
    t1 = time.time()
    for _ in range(MAX_TOKENS):
        if draft1 is None:
            # Single-token step (from S_k)
            if pending_c is not None:
                # R5: the correction is deferred; ctoken is row 0 of the next verify (already emitted at the reject). The draft chain re-anchors on the rejected row's pre-norm hidden (in the S=4 graph; zero extra target work).
                t = pending_c
                pending_c = None
                h = h_r
                h_r = None
            else:
                t = int(mx.argmax(l))
                if t == EOS:
                    break
                out.append(t)
            if mtp_on:
                prev_token = t
                # Draft token 1 (from AR state)
                emb1 = lm.model.embed_tokens(mx.array([[t]]))
                dh1 = lm.mtp(h, emb1, mtp_cache)
                draft1 = draft_tok(dh1)
                if mode in ("3", "4"):
                    # Draft token 2 (from MTP module given token1)
                    emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
                    dh2 = lm.mtp(dh1, emb2, mtp_cache)
                    draft2 = draft_tok(dh2)
                if mode == "4":
                    # Draft token 3 (from MTP module given token2)
                    emb3 = lm.model.embed_tokens(mx.array([[draft2]]))
                    dh3 = lm.mtp(dh2, emb3, mtp_cache)
                    draft3 = draft_tok(dh3)
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
            # Verify step (from S_k, drafts D1, D2, D3). BATCHED S=4 (gdn_fused_step_tape: per-row state tapes). FIX#1: commit = index select on the state tape + trims (lazy cache-leaf rebind; NO save/restore, NO re-forward of accepted rows). MTP6_TAPE=0 -> savestate (canonical restore + re-forward of the accepted prefix).
            d1, d2, d3 = draft1, draft2, draft3
            draft1 = draft2 = draft3 = None
            verifies += 1
            if tape_mode == "0":
                saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2, d3]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True
            )
            if tape_mode == "r5e":
                # R5e: force the S=4 GEMM in the verify window; the commit window then runs argmax on already-evaluated logits (cheap sync) + the rebind bookkeeping.
                mx.eval(logits, hidden)
            l0, l1, l2, l3 = logits[0, 0], logits[0, 1], logits[0, 2], logits[0, 3]
            # batched S=4 row pre-norm hiddens (pure tape reuses these for the next draft chain on rejected steps)
            hb0, hb1, hb2, hb3 = hidden[0, 0], hidden[0, 1], hidden[0, 2], hidden[0, 3]

            def commit(krow, prefix):
                # Select the per-row GDN state-tape row (the state AFTER row krow) and rebind the cache leaves to the tape slice (lazy; no copies, no sync). lengths is None in this model (advance() is a no-op), so no lengths fix-up.
                tape_avail = tape_mode in ("1", "hybrid", "r5", "r5b", "r5e")
                if tape_avail:
                    try:
                        st0 = lm.model.layers[0].linear_attn.last_tape[0]
                        tape_avail = st0 is not None
                    except Exception:
                        tape_avail = False
                if tape_avail:
                    for i, c in enumerate(cache):
                        if hasattr(c, "cache"):
                            st, cstab = lm.model.layers[i].linear_attn.last_tape
                            # tape rows are B-squeezed: restore the B dim
                            if tape_mode == "r5b":
                                # R5b: materialize the selected row - the r5 rebind is a strided row-slice view of the S=4 tape batch; the next step's fused GDN step / S=4 GEMM window should read contiguous state (the bounded copy is ~1 MB/layer, sub-ms; it also isolates the strided-view hypothesis from the deferred-eval-order hypothesis, since r5b differs from r5 by exactly this copy).
                                c[1] = st[krow].reshape((1,) + st[krow].shape).copy()
                                c[0] = cstab[krow].reshape((1,) + cstab[krow].shape).copy()
                            else:
                                c[1] = st[krow].reshape((1,) + st[krow].shape)
                                c[0] = cstab[krow].reshape((1,) + cstab[krow].shape)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(3 - krow)
                    return True
                else:
                    # savestate A/B (MTP6_TAPE=0): canonical restore + trim ALL verify rows + re-forward of the accepted prefix (the GDN restore makes the re-forward mandatory, so the kept rows must be dropped).
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(4)
                    for pt in prefix:
                        lm(mx.array([[pt]]), cache=cache)
                    return False

            def reject(krow, lk, hb, ktrim, prefix, ctoken):
                nonlocal l, h, n_refwd, pending_c, h_r, n_cdeferred
                # Rejected step: commit the tape row (or canonical restore + re-forward), trim mtp_cache to the accepted prefix. r5: NO S=1 correction re-forward - ctoken becomes row 0 of the next verify and the draft chain re-anchors on the rejected row's pre-norm h (the llama.cpp/radiance structure); fallback (no tape) runs the legacy correction forward. Then choose (l, h) for the next draft chain by mode: pure (MTP6_TAPE=1): l = the batched row logits, h = the batched S=4 row h. hybrid: l and h are the correction forward's own.
                committed = commit(krow, prefix)
                if os.environ.get("MTP6_MTPCACHE_TRIM", "1") == "1":
                    mtp_cache[0].trim(ktrim)
                if tape_mode in ("r5", "r5b", "r5e") and committed:
                    # R5/R5b/R5e: no S=1 correction re-forward; ctoken = row 0 of the next verify, draft re-anchors on the rejected row's h
                    pending_c = ctoken
                    h_r = hb
                    out.append(ctoken)
                    n_cdeferred += 1
                    return
                logits, hidden = lm(
                    mx.array([[ctoken]]), cache=cache,
                    return_hidden=True, return_prenorm=True
                )
                if tape_mode == "1":
                    l, h = lk, hb
                else:
                    l, h = logits[0, 0], hidden[0, 0]
                out.append(ctoken)
                n_refwd += 1

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
                        # Full accept: model's state (after row 3) is already in the leaves; no rebind, no trims.
                        l, h = l3, hb3
                    else:
                        # draft3 rejected: commit row 2 (after D2), then ONE S=1 forward of the correction (the accepted prefix costs zero - tape commit).
                        reject(2, l2, hb2, 1, [prev_token, d1, d2], t4)
                else:
                    # draft2 rejected: commit row 1 (after D1), forward t3.
                    reject(1, l1, hb1, 2, [prev_token, d1], t3)
            else:
                # draft1 rejected: commit row 0 (after prev_token), forward t2.
                reject(0, l0, hb0, 3, [prev_token], t2)
    dt = time.time() - t1
    n = len(out)
    d1pct = 100.0 * d1_acc / verifies if verifies else 0.0
    d2pct = 100.0 * d2_acc / d1_acc if d1_acc else 0.0
    d3pct = 100.0 * d3_acc / d2_acc if d2_acc else 0.0
    tok_step = 1.0 + (d1_acc + d2_acc + d3_acc) / verifies if verifies else 0.0
    if mode == "4":
        import mlx_lm.models.qwen3_5 as _m5
        print(
            "INFO TAPEMODE=%s REFWDS=%d CDEFERRED=%d GDN_TAPE_CALLS=%d INT2=%s" % (
                os.environ.get("MTP6_TAPE", "1"),
                n_refwd,
                n_cdeferred,
                getattr(_m5, "GDN_TAPE_CALLS", -1),
                "on" if qh is not None else "off",
            ),
            flush=True,
        )
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
