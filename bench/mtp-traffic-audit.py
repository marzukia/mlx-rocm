"""MTP traffic audit probe (measurement only - no kernel changes).

Answers: does the commit/correction phase re-read the full target
weights per step, or do verify+commit share one target read?

Method (source (a), analytic - the guest has no rocprof / amdgpu
debug counters, so no HSA byte counters are available):
1. INVENTORY: load the model, enumerate the weight-tensor bytes
   (packed 4-bit + scales + biases) per module: target trunk
   (layers), target lm_head (4-bit = also the draft head in the
   as-run R4b stack), sidecar mtp module. GEMM ops per target
   forward = number of QuantizedLinear in the trunk; S=4 verify
   rows run as one M=4 QMM pass (fused, MLX_ROCM_QMV_ROWS), S=1
   correction/draft forwards as one M=1 QMV pass - every weight
   tensor is read exactly once per op per forward.
2. SHORT RUN: one in-situ mode-4 hybrid generation (the exact stack
   env: tape commit + one S=1 correction forward; draft head = the
   4-bit target lm_head, since the R4b wheel has no int2 op),
   counting target forwards (S=4 verify, S=1 correction), sidecar
   forwards, and draft-head calls; prints the wheel's GDN
   counters. The short run is a regime sanity check; the per-step
   bytes use the anchor bench's acceptance mix.
3. PER-PHASE BYTES: derived from the counts.
"""

import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

MODELS = os.environ.get("MODELS", "/models")
MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
PROMPT = "Explain in great detail how a modern smartphone works, from the hardware components to the operating system, covering every subsystem. Be exhaustive."
MAX_TOKENS = int(os.environ.get("TRAUDIT_MAX_TOKENS", "64"))
EOS = 248044


def walk_bytes(mod):
    """Recursive weight bytes of a module tree (every param array
    incl. 4-bit scales/biases). Returns (total_bytes,
    n_quantized_linears)."""
    if isinstance(mod, mx.array):
        return mod.nbytes, 0
    items = list(mod.items()) if hasattr(mod, "items") else []
    total = 0
    n_lin = 0
    for _name, v in items:
        if isinstance(v, mx.array):
            total += v.nbytes
        elif hasattr(v, "items"):
            t, n = walk_bytes(v)
            total += t
            n_lin += n
    n_lin += sum(1 for _, v in items if type(v).__name__ == "QuantizedLinear")
    return total, n_lin


def quantize_int2(w, chunk=8192):
    """Int2 g128-group asymmetric quantization (same as mtp-bench6)."""
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
        scale[i:j] = 4.0 * s
        zs[i:j] = (4.0 + z) * s
    return packed.astype(mx.uint8), scale, zs


def int2_argmax(x, qh):
    x = x.reshape(1, -1)
    K = x.shape[1]
    NG = K // 128
    xf = x.reshape(NG, 128).astype(mx.float32)
    sx = mx.sum(xf, axis=1)
    logits, cval, ccol = mx.fast.lm_head_int2(x, qh["packed"], qh["scale"], qh["zs"], sx)
    perm = mx.argsort(cval.reshape(-1))
    idx = mx.take(ccol.reshape(-1), perm[-32:])
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
    os.environ["MLX_ROCM_GDN_TAPE"] = "1"
    os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    t0 = time.time()
    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())

    trunk_bytes, trunk_nlin = 0, 0
    per_layer = []
    for layer in lm.model.layers:
        b, n = walk_bytes(layer)
        trunk_bytes += b
        trunk_nlin += n
        per_layer.append((b, n))
    head_bytes, head_nlin = walk_bytes(lm.lm_head)
    side_bytes, side_nlin = walk_bytes(lm.mtp)
    n_layers = len(lm.model.layers)
    uniq = sorted(set(per_layer))
    print(
        "INV T_TRUNK=%.2f GiB (%d layers, %d QLinears) | T_HEAD=%.3f GiB (%d) | "
        "SIDE=%.4f GiB (%d QLinears) | per-layer uniq: %s"
        % (
            trunk_bytes / 2**30,
            n_layers,
            trunk_nlin,
            head_bytes / 2**30,
            head_nlin,
            side_bytes / 2**30,
            side_nlin,
            uniq,
        ),
        flush=True,
    )
    t_target = trunk_bytes + head_bytes

    int2_on = hasattr(mx.fast, "lm_head_int2")
    h_bytes = head_bytes  # 4-bit draft head (the as-run stack of record)
    print(
        "DRAFT_HEAD %.3f GiB (%s)"
        % (
            h_bytes / 2**30,
            "int2 op present" if int2_on else "int2 op ABSENT -> 4-bit head as-run",
        ),
        flush=True,
    )

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    mtp_cache = [KVCache()]

    def draft_tok(dh):
        nonlocal n_dhead
        n_dhead += 1
        if int2_on:
            return int2_argmax(dh, qh)
        return int(mx.argmax(lm.lm_head(dh)[0, -1]))

    def save_gdn(c):
        return [(i, x[0], x[1], x.lengths) for i, x in enumerate(c) if hasattr(x, "cache")]

    def commit_row(krow, saved):
        st0 = lm.model.layers[0].linear_attn.last_tape[0]
        if st0 is not None:
            for i, c in enumerate(cache):
                if hasattr(c, "cache"):
                    st, cstab = lm.model.layers[i].linear_attn.last_tape
                    c[1] = st[krow].reshape((1,) + st[krow].shape)
                    c[0] = cstab[krow].reshape((1,) + cstab[krow].shape)
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(3 - krow)
        else:
            for i, x0, x1, xl in saved:
                cache[i][0] = x0
                cache[i][1] = x1
                cache[i].lengths = xl
            for c in cache:
                if isinstance(c, KVCache):
                    c.trim(4)

    n_side = 0
    n_dhead = 0
    qh = None
    verifies = 0
    d1_acc = d2_acc = d3_acc = 0
    n_refwd = 0
    draft1 = draft2 = draft3 = None
    prev_token = None
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache, return_hidden=True, return_prenorm=True)
    l = logits[0, -1]
    h = hidden[0, -1]

    t1 = time.time()
    for _ in range(MAX_TOKENS):
        if draft1 is None:
            t = int(mx.argmax(l))
            if t == EOS:
                break
            prev_token = t
            e1 = lm.model.embed_tokens(mx.array([[t]]))
            dh1 = lm.mtp(h, e1, mtp_cache)
            n_side += 1
            draft1 = draft_tok(dh1)
            e2 = lm.model.embed_tokens(mx.array([[draft1]]))
            dh2 = lm.mtp(dh1, e2, mtp_cache)
            n_side += 1
            draft2 = draft_tok(dh2)
            e3 = lm.model.embed_tokens(mx.array([[draft2]]))
            dh3 = lm.mtp(dh2, e3, mtp_cache)
            n_side += 1
            draft3 = draft_tok(dh3)
        else:
            d1, d2, d3 = draft1, draft2, draft3
            draft1 = draft2 = draft3 = None
            verifies += 1
            saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2, d3]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, l1, l2, l3 = logits[0, 0], logits[0, 1], logits[0, 2], logits[0, 3]
            hb0, hb1, hb2, hb3 = hidden[0, 0], hidden[0, 1], hidden[0, 2], hidden[0, 3]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 != d1:
                commit_row(0, saved)
                mtp_cache[0].trim(3)
                logits, hidden = lm(mx.array([[t2]]), cache=cache, return_hidden=True, return_prenorm=True)
                l, h = logits[0, 0], hidden[0, 0]
                n_refwd += 1
            else:
                d1_acc += 1
                t3 = int(mx.argmax(l1))
                if t3 == EOS:
                    break
                if t3 != d2:
                    commit_row(1, saved)
                    mtp_cache[0].trim(2)
                    logits, hidden = lm(mx.array([[t3]]), cache=cache, return_hidden=True, return_prenorm=True)
                    l, h = logits[0, 0], hidden[0, 0]
                    n_refwd += 1
                else:
                    d2_acc += 1
                    t4 = int(mx.argmax(l2))
                    if t4 == EOS:
                        break
                    if t4 != d3:
                        commit_row(2, saved)
                        mtp_cache[0].trim(1)
                        logits, hidden = lm(mx.array([[t4]]), cache=cache, return_hidden=True, return_prenorm=True)
                        l, h = logits[0, 0], hidden[0, 0]
                        n_refwd += 1
                    else:
                        d3_acc += 1
                        l, h = l3, hb3

    dt = time.time() - t1
    import mlx_lm.models.qwen3_5 as _m5

    print(
        "RUN GEN=%.2fs VERIFIES=%d D1=%d D2=%d D3=%d REFWDS=%d SIDE_FWD=%d DHEAD=%d "
        "GDN_TAPE_CALLS=%d GDN_FUSE_CALLS=%s"
        % (
            dt,
            verifies,
            d1_acc,
            d2_acc,
            d3_acc,
            n_refwd,
            n_side,
            n_dhead,
            getattr(_m5, "GDN_TAPE_CALLS", -1),
            getattr(_m5, "GDN_FUSE_CALLS", "n/a"),
        ),
        flush=True,
    )
    steps = max(verifies, 1)
    t_target_gib = t_target / 2**30
    p_rej = n_refwd / steps
    t_verify = t_target_gib
    t_commit = 0.0
    t_corr = p_rej * t_target_gib
    t_draft = (n_side / steps) * (side_bytes / 2**30) + (n_dhead / steps) * (h_bytes / 2**30)
    total = t_verify + t_commit + t_corr + t_draft
    print(
        "BYTES_PER_STEP (short-run mix) verify=%.2f commit=%.2f correction=%.2f draft=%.2f total=%.2f GiB"
        % (t_verify, t_commit, t_corr, t_draft, total),
        flush=True,
    )
    print("LOAD=%.1f" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
