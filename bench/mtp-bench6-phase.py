#!/usr/bin/env python3
"""INSTRUMENTED copy of mtp-bench6.py: phase timing + mode-4 path switches.

Diagnostic only - NOT the official protocol (mtp-bench6.py is the
byte-identical canonical bench; this file exists to split the per-step
cost: prefill / draft (sidecar + head) / verify (batched S=4) / commit
(restore+re-forward vs tape rebind) / single-token step B).

Switches (mode 4 only):
    MTP6_ACCEPT=savestate|tape   - savestate = the canonical path
        (save/restore GDN + 2-4 re-forwards on rejection; the default
        when nothing is set); tape = the gdn_fused_step_tape per-row
        state-tape commit (lazy cache-leaf rebind, no re-forward).
        tape sets MLX_ROCM_GDN_TAPE=1.
    MTP6_DRAFT_HEAD=int2|bf16   - int2 = mx.fast.lm_head_int2 dequant
        GEMV + exact top-32 re-rank (requires the patched wheel; an
        explicit opt-in); bf16 = the canonical full lm_head (the
        default: the canonical 4-bit path runs when unset).
    MTP6_MTPTRIM=1              - also trim mtp_cache to the accepted
        prefix (off by default: the canonical path never rolls the sidecar back).

All other modes (0-3) run the canonical path unchanged; the phase
timers still accumulate (step B / AR).

Prints the mtp-bench6 protocol lines + PHASE ms (cumulative, % of the generate window) + INFO (REFWDS / GDN_TAPE_CALLS / INT2 / MTPTRIM).
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

PT = {}


def acc(name, sec):
    PT[name] = PT.get(name, 0.0) + sec


def save_gdn(cache):
    return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if hasattr(c, "cache")]


def restore_gdn(cache, saved):
    for i, conv, ssm, lengths in saved:
        cache[i][0] = conv
        cache[i][1] = ssm
        cache[i].lengths = lengths


def quantize_int2(w, chunk=8192):
    """Int2 g128-group asymmetric quantization (quarter-split 2-bit
    packing, radiance port): per 128-group, s = max((hi-lo)/3, 1e-8),
    z = clamp(round(-lo/s), 0, 3), q = clamp(round(w/s + z), 0, 3);
    byte j holds v0..v3 for x indices j, j+Q, j+2Q, j+3Q (Q = K/4).
    Returns (packed u8 [N, K/4], scale f32 [N, K/128] = s, zs f32
    [N, K/128] = z*s)."""
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
                mx.maximum(mx.round(wf / s[..., None] + z[..., None]), 0.0),
                3.0,
            )
            .astype(mx.int32)
            .reshape(j - i, K)
        )
        packed[i:j] = q[:, :Q] | (q[:, Q : 2 * Q] << 2) | (q[:, 2 * Q : 3 * Q] << 4) | (q[:, 3 * Q :] << 6)
        # Dequant coefficients exactly as the kernel takes them (lm_head_int2.hip: the header contract and the per-group algebra `acc += d*s - sx*(z*s)`, i.e. the dequant of w_hat = s*(q - z) for q = clamp(round(w/s + z), 0, 3)). An earlier revision of this reference wrote 4.0*s / (4.0+z)*s, which rescales each group by 4x plus a per-group offset and corrupts the argmax ranking.
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
    _op = (
        mx.fast.lm_head_int2_wide
        if os.environ.get("MTP6_DRAFT_HEAD", "bf16") == "int2w"
        else mx.fast.lm_head_int2
    )
    logits, cval, ccol = _op(x, qh["packed"], qh["scale"], qh["zs"], sx)
    perm = mx.argsort(cval.reshape(-1))
    idx = mx.take(ccol.reshape(-1), perm[-32:])
    if "w" in qh:
        cols = mx.take(qh["w"], idx, 0)
    else:
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
    # MTP6_TAPE: 0=savestate (canonical), 1=pure tape (round-1 measured), hybrid (round-2: tape commit + S=1 correction forward, h from the correction forward). Legacy MTP6_ACCEPT alias: tape -> hybrid.
    tape_mode = os.environ.get("MTP6_TAPE", "")
    if not tape_mode:
        legacy = os.environ.get("MTP6_ACCEPT", "savestate")
        assert legacy in ("savestate", "tape"), legacy
        tape_mode = "0" if legacy == "savestate" else "hybrid"
    assert tape_mode in ("0", "1", "hybrid"), tape_mode
    mtptrim = os.environ.get("MTP6_MTPTRIM", "0") == "1"
    if mode == "4" and tape_mode in ("1", "hybrid"):
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
        _dh = os.environ.get("MTP6_DRAFT_HEAD", "bf16")
        if _dh in ("int2", "int2w") and hasattr(mx.fast, "lm_head_int2_wide" if _dh == "int2w" else "lm_head_int2"):
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
                "INT2_BUILD %.1fs N=%d K=%d (packed %d MB)"
                % (
                    time.time() - tq0,
                    n_rows,
                    n_cols,
                    packed.nbytes / (1024 * 1024),
                ),
                flush=True,
            )

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()

    tp0 = time.perf_counter()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    acc("prefill", time.perf_counter() - tp0)
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
    n_tape = 0

    def draft_tok(dh):
        return int2_argmax(dh, qh) if qh is not None else int(mx.argmax(lm.lm_head(dh)[0, -1]))

    t1 = time.time()
    for _ in range(MAX_TOKENS):
        if draft1 is None:
            # Single-token step (from S_k)
            tb0 = time.perf_counter()
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
                acc("draft", time.perf_counter() - tb0)
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
                acc("ar", time.perf_counter() - tb0)
        elif mode == "1":
            # Verify step (from S_k, draft D1). Two 1-token forwards.
            d = draft1
            draft1 = draft2 = draft3 = None
            verifies += 1
            tv0 = time.perf_counter()
            logits, hidden = lm(
                mx.array([[prev_token]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, -1], hidden[0, -1]
            saved = save_gdn(cache)
            logits, hidden = lm(mx.array([[d]]), cache=cache, return_hidden=True, return_prenorm=True)
            l1, h1 = logits[0, -1], hidden[0, -1]
            acc("verify", time.perf_counter() - tv0)
            tc0 = time.perf_counter()
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
                logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
                l, h = logits[0, -1], hidden[0, -1]
                n_refwd += 1
            acc("commit", time.perf_counter() - tc0)
        elif mode == "2":
            # Verify step (from S_k, draft D1). BATCHED S=2.
            d = draft1
            draft1 = draft2 = draft3 = None
            verifies += 1
            tv0 = time.perf_counter()
            saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            acc("verify", time.perf_counter() - tv0)
            tc0 = time.perf_counter()
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
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
                l, h = logits[0, -1], hidden[0, -1]
                n_refwd += 2
            acc("commit", time.perf_counter() - tc0)
        elif mode == "3":
            # Verify step (from S_k, drafts D1, D2). BATCHED S=3.
            d1, d2 = draft1, draft2
            draft1 = draft2 = draft3 = None
            verifies += 1
            tv0 = time.perf_counter()
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
            acc("verify", time.perf_counter() - tv0)
            tc0 = time.perf_counter()
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
                    logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                    logits, hidden = lm(mx.array([[d1]]), cache=cache, return_hidden=True, return_prenorm=True)
                    logits, hidden = lm(mx.array([[t3]]), cache=cache, return_hidden=True, return_prenorm=True)
                    l, h = logits[0, -1], hidden[0, -1]
                    n_refwd += 3
            else:
                out.append(t2)
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(3)
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache, return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
                l, h = logits[0, -1], hidden[0, -1]
                n_refwd += 2
            acc("commit", time.perf_counter() - tc0)
        else:
            # Verify step (from S_k, drafts D1, D2, D3). BATCHED S=4. MTP6_TAPE: 0=savestate (canonical), 1=pure tape (round-1 measured: h from the batched S=4 row), hybrid (round-2: h from the S=1 correction forward).
            d1, d2, d3 = draft1, draft2, draft3
            draft1 = draft2 = draft3 = None
            verifies += 1
            tv0 = time.perf_counter()
            if tape_mode == "0":
                saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2, d3]]),
                cache=cache, return_hidden=True, return_prenorm=True,
            )
            acc("verify", time.perf_counter() - tv0)
            tc0 = time.perf_counter()
            l0, l1, l2, l3 = logits[0, 0], logits[0, 1], logits[0, 2], logits[0, 3]
            hb0, hb1, hb2, hb3 = hidden[0, 0], hidden[0, 1], hidden[0, 2], hidden[0, 3]
            did_commit = False

            def commit(krow, prefix):
                # tape: select the per-row GDN state-tape row and rebind the cache leaves (lazy). lengths is None in this model.
                nonlocal n_tape
                tape_avail = tape_mode in ("1", "hybrid")
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
                            c[1] = st[krow].reshape((1,) + st[krow].shape)
                            c[0] = cstab[krow].reshape((1,) + cstab[krow].shape)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(3 - krow)
                    n_tape += 1
                else:
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(4)
                    for pt in prefix:
                        lm(mx.array([[pt]]), cache=cache)

            def reject(krow, lk, hb, ktrim, prefix, ctoken):
                # Rejected step: commit the tape row (or canonical restore
                nonlocal l, h, n_refwd, did_commit
                commit(krow, prefix)
                mtp_cache[0].trim(ktrim)
                acc("commit", time.perf_counter() - tc0)
                tc1 = time.perf_counter()
                logits, hidden = lm(
                    mx.array([[ctoken]]), cache=cache,
                    return_hidden=True, return_prenorm=True,
                )
                acc("correction", time.perf_counter() - tc1)
                if tape_mode == "1":
                    l, h = lk, hb
                else:
                    l, h = logits[0, 0], hidden[0, 0]
                out.append(ctoken)
                n_refwd += 1
                did_commit = True

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
                        l, h = l3, hb3
                    else:
                        reject(2, l2, hb2, 1, [prev_token, d1, d2], t4)
                else:
                    reject(1, l1, hb1, 2, [prev_token, d1], t3)
            else:
                reject(0, l0, hb0, 3, [prev_token], t2)
            if not did_commit:
                acc("commit", time.perf_counter() - tc0)

    t_gen = time.time() - t1
    t_elapsed = time.time() - t0
    toks = len(out)
    print(
        "TOKS=%d TIME=%.1fs GEN=%.1fs TOKS_S=%.2f LOAD=%.1fs" % (toks, t_elapsed, t_gen, toks / t_gen, t_load),
        flush=True,
    )
    print(
        "MTP=%s VERIFIES=%d D1ACC=%d D2ACC=%d D3ACC=%d D1PCT=%.1f D2PCT=%.1f D3PCT=%.1f TOK_PER_STEP=%.2f"
        % (
            mode,
            verifies,
            d1_acc,
            d2_acc,
            d3_acc,
            100.0 * d1_acc / verifies if verifies else 0.0,
            100.0 * d2_acc / verifies if verifies else 0.0,
            100.0 * d3_acc / verifies if verifies else 0.0,
            1.0 + (d1_acc + d2_acc + d3_acc) / verifies if verifies else 0.0,
        ),
        flush=True,
    )
    import mlx_lm.models.qwen3_5 as _m5

    print(
        "INFO TAPEMODE=%s REFWDS=%d TAPE_COMMITS=%d GDN_TAPE_CALLS=%d INT2=%s MTPTRIM=%s"
        % (
            tape_mode,
            n_refwd,
            n_tape,
            getattr(_m5, "GDN_TAPE_CALLS", -1),
            "on" if qh is not None else "off",
            mtptrim,
        ),
        flush=True,
    )
    print(
        "PHASE prefill=%.1fms draft=%.1fms verify=%.1fms commit=%.1fms correction=%.1fms ar=%.1fms"
        % (
            1000.0 * PT.get("prefill", 0.0),
            1000.0 * PT.get("draft", 0.0),
            1000.0 * PT.get("verify", 0.0),
            1000.0 * PT.get("commit", 0.0),
            1000.0 * PT.get("correction", 0.0),
            1000.0 * PT.get("ar", 0.0),
        ),
        flush=True,
    )
    print(
        "PHASE_PCT prefill=%.1f draft=%.1f verify=%.1f commit=%.1f correction=%.1f ar=%.1f"
        % tuple(
            100.0 * PT.get(k, 0.0) / t_gen if t_gen else 0.0
            for k in ("prefill", "draft", "verify", "commit", "correction", "ar")
        ),
        flush=True,
    )
    print("FIRST16: " + " ".join(str(x) for x in out[:16]), flush=True)
    print("FIRST20: " + " ".join(str(x) for x in out[:20]), flush=True)
    print("ALLTOKS: " + " ".join(str(x) for x in out), flush=True)
    print("TEXT: " + tok.decode(out), flush=True)


if __name__ == "__main__":
    main()
