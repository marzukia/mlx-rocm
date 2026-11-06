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

T1917 acceptance probe (env-gated, this file):
  - mode 3 (S=3, K=2) TAPES: MTP6_TAPE in (1, hybrid, r5) routes the S=3 verify
    through gdn_fused_step_tape (the TMAX=3 instantiation; MLX_ROCM_GDN_T3=1 is
    set here - the unpatched wheel rejects T=3 in use_fallback = clean error).
    Commit = index-select on the state tape + trims (mirror of the mode-4
    commit/reject, K=2: commit row 1 after D1 / row 0 after prev_token, KV trim
    S-1-krow, mtp_cache trim K-krow, r5 deferred correction). MTP6_TAPE=0 =
    the legacy save/restore + re-forward path (unchanged).
  - MTP6_DRAFT_HEAD: int2 (default: int2 GEMV + 4-bit re-rank) | bf16 (the
    model's 4-bit QuantizedLinear lm_head direct) | bf16full (bf16 dequant of
    the 4-bit lm_head - the quantization bit-width probe, no new weights).
  - MTP6_HEAD_H: prenorm (default: the trunk pre-final-norm hidden) | postnorm
    (the trunk's final RMSNorm applied in-bench before the MTP module - the
    head input-feature probe, no retraining).
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
import hashlib
import json
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
MODELS = os.environ.get("MODELS", "/models")

MODEL = os.environ.get("MTP_MODEL", f"{MODELS}/qwen3-9b-mlx")
# t1996: MTP6_PROMPT env override (default = the canonical smartphone prompt, verbatim).
# The t1436 llama.cpp anchor protocol used the 1-token prompt "Hello" on a diverse
# non-degenerate trajectory (2.15/3 acceptance, no loop); matching it is the most
# comparable non-degenerate trajectory for the S=3 re-measurement.
PROMPT = os.environ.get(
    "MTP6_PROMPT",
    "Explain in great detail how a modern smartphone works, from the hardware components to the operating system, covering every subsystem. Be exhaustive.",
)
MAX_TOKENS = int(os.environ.get("MTP_MAX_TOKENS", "1024"))
EOS = 248044


class NgramFill:
    """Zero-cost n-gram continuation fill (MTP6_NGRAM_S3, mode 3 S=3 arm; the
    same class as the mode-4 NgramFill in mtp-bench6-dg.py). Mirrors
    common_ngram_map_draft semantics (llama.cpp ngram-map.cpp:222-400):
    key = trailing (n-1) confirmed context tokens + candidate prefix (the
    'sampled' role); the adopted draft = the most frequent token FOLLOWING
    the key in the context (min_hits = 1, llama.cpp:333). On a hit NO draft
    forward runs; the MTP module KV cache (mtp_cache) then lags by the
    skipped rows (stale-attention approximation, same class as the known
    un-trimmed mtp_cache divergence). Context = prompt +
    confirmed generated tokens (candidates never enter the context)."""

    def __init__(self, ctx, n):
        self.n = n
        self.seq = list(ctx)
        self.m = {}
        for i in range(len(self.seq) - self.n):
            self._add(i)

    def _add(self, i):
        key = tuple(self.seq[i:i + self.n])
        v = self.m.get(key)
        if v is None:
            self.m[key] = {self.seq[i + self.n]: 1}
        else:
            v[self.seq[i + self.n]] = v.get(self.seq[i + self.n], 0) + 1

    def update(self, t):
        self.seq.append(t)
        if len(self.seq) >= self.n + 1:
            self._add(len(self.seq) - self.n - 1)

    def lookup(self, i, cands):
        # i = 1-based draft position; cands = [c1..c_{i-1}]
        if len(self.seq) < self.n:
            return None
        pat = self.seq[-(self.n - i + 1):] + list(cands)
        v = self.m.get(tuple(pat))
        if not v:
            return None
        # most frequent continuation; deterministic tie-break = smallest id
        return min(v.items(), key=lambda kv: (-kv[1], kv[0]))[0]


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
    _op = (
        mx.fast.lm_head_int2_wide
        if os.environ.get("MTP6_DRAFT_HEAD", "bf16") == "int2w"
        else mx.fast.lm_head_int2
    )
    logits, cval, ccol = _op(
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
    tape_mode = "0"
    if mode in ("3", "4"):
        # T1917: the mode-3 (S=3, K=2) tape arm mirrors the mode-4 tape setup; MLX_ROCM_GDN_T3=1 enables the TMAX=3 instantiation (new wheel; the unpatched wheel's T=3 is rejected in use_fallback = clean error, no OOB).
        tape_mode = os.environ.get("MTP6_TAPE", "1")
        assert tape_mode in ("0", "1", "hybrid", "r5", "slot"), tape_mode
        if tape_mode in ("1", "hybrid", "r5", "slot"):
            if tape_mode == "slot":
                # SLOT (Phase B): slot op snapshots commit rows; tape env OFF
                # so the model dispatch (SLOT branch first) selects gdn_fused_step_slot.
                os.environ["MLX_ROCM_GDN_SLOT"] = "1"
                os.environ["MLX_ROCM_GDN_TAPE"] = "0"
            else:
                os.environ["MLX_ROCM_GDN_TAPE"] = "1"
            os.environ["MLX_ROCM_GDN_T3"] = "1"
        os.environ["MLX_ROCM_GDN_FUSE"] = "1"
    # T1917 head probe env gates (all no-retraining; model weights unchanged):
    # MTP6_DRAFT_HEAD: int2 (default) | bf16 (model 4-bit QuantizedLinear direct) | bf16full (bf16 dequant of the 4-bit lm_head; bit-width probe).
    # MTP6_HEAD_H: prenorm (default) | postnorm (trunk final RMSNorm in-bench; input-feature probe).
    head = os.environ.get("MTP6_DRAFT_HEAD", "int2")
    assert head in ("int2", "int2w", "bf16", "bf16full"), head
    head_h = os.environ.get("MTP6_HEAD_H", "prenorm")
    assert head_h in ("prenorm", "postnorm"), head_h
    t0 = time.time()
    model, tok = load(MODEL)
    t_load = time.time() - t0
    lm = model.language_model
    mx.eval(model.parameters())
    if mtp_on:
        assert lm.mtp is not None, "no mtp module (config.mtp_num_hidden_layers?)"
        mtp_cache = [KVCache()]
        if os.environ.get("CH_MTP_4BIT", "0") == "1":
            # 4-bit MTP module arm: rebuild the quant module, load the 4-bit
            # sidecar (bench3-side pattern; file NOT renamed, path overridable.
            # CH_MTP_4BIT, not MTP_4BIT - that one is read by the wheel at
            # model-init, before load(), which would mismatch the globbed
            # bf16 sidecar).
            import safetensors.numpy as _sfn
            from mlx_lm.models.qwen3_5 import MTPModule

            _side = os.environ.get("MTP4_SIDECAR") or os.path.join(MODEL, "model-mtp-4bit.safetensors")
            assert os.path.exists(_side), _side
            lm.mtp = MTPModule(lm.args, 1, quant=True)

            def _setp(mod, path, val):
                cur = mod
                for p in path.split(".")[:-1]:
                    cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
                setattr(cur, path.split(".")[-1], val)

            for _k, _v in _sfn.load_file(_side).items():
                if not _k.startswith("mtp."):
                    continue
                _kk = _k[4:]
                if _kk.endswith(".qweight"):
                    _base = _kk[: -len(".qweight")] + ".weight"
                elif _kk.endswith(".bias"):
                    _base = _kk[: -len(".bias")] + ".biases"
                else:
                    _base = _kk
                _setp(lm.mtp, _base, mx.array(_v))
            mx.eval(lm.mtp.parameters())
            print(f"MTP4_ARM {os.path.basename(_side)}", flush=True)
    qh = None
    if mode in ("3", "4"):
        # T1917: the draft-head probe arms (MTP6_DRAFT_HEAD) are available to mode 3 (S=3, K=2) as well as mode 4 (S=4, K=3); int2 build (FIX#2, int2 g128-asymmetric draft head, MTP6_DRAFT_HEAD=bf16 to A/B the model 4-bit head) and bf16full (bf16 dequant, no new weights). int2w = the vectorized int2 kernel (base opt-in, same qh build, wide op).
        if head in ("int2", "int2w") and hasattr(
            mx.fast, "lm_head_int2_wide" if head == "int2w" else "lm_head_int2"
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
        elif head == "bf16full":
            # T1917 bit-width probe arm: the full-precision (bf16 dequant) draft head; NO new weights (dequant of the model's 4-bit lm_head), no retraining.
            tq0 = time.time()
            _h = lm.lm_head
            if type(_h).__name__ == "QuantizedLinear":
                w_bf16 = mx.dequantize(
                    _h.weight, _h.scales, _h.biases, group_size=_h.group_size, bits=_h.bits
                )
            else:
                w_bf16 = _h.weight
            qh = {"wbf16": w_bf16}
            mx.eval(qh["wbf16"])
            print(
                "BF16FULL_BUILD %.1fs N=%d K=%d (%d MB)" % (
                    time.time() - tq0,
                    w_bf16.shape[0],
                    w_bf16.shape[1],
                    w_bf16.nbytes / (1024 * 1024),
                ),
                flush=True,
            )
        # head == "bf16": qh stays None; draft_tok uses the model's (4-bit quantized) lm_head directly.

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

    # t1986 ngram-S3 probe: MTP6_NGRAM_S3=1 wires the mode-4 n-gram fill
    # (NgramFill) into the S=3 (mode 3) draft chain. Default off = the
    # canonical S=3 path runs verbatim (nfill stays None; fw1/fw2 stay True;
    # the reject trims below reduce to the fixed 1/2 constants). A fill hit
    # at draft position i skips i MTP forwards + head argmaxes; the verify
    # pass is unchanged (all emitted tokens remain target-greedy, lossless).
    ngram_s3 = mode == "3" and mtp_on and os.environ.get(
        "MTP6_NGRAM_S3", "0"
    ) == "1"
    ng_n = int(os.environ.get("MTP6_NGRAM_N", "4"))
    nfill = NgramFill(prompt_ids, ng_n) if ngram_s3 else None
    nfill_len = 0
    n_ng = 0
    fw1 = True
    fw2 = True
    if ngram_s3:
        print(
            "NGRAM_S3 ON n=%d" % ng_n,
            flush=True,
        )

    def draft_tok(dh):
        if qh is not None and "wbf16" in qh:
            return int(mx.argmax(mx.matmul(dh, qh["wbf16"].T)[0, -1]))
        return int2_argmax(dh, qh) if qh is not None else int(
            mx.argmax(lm.lm_head(dh)[0, -1])
        )

    def hhead(x):
        # T1917 head input-feature probe (no retraining): postnorm feeds the MTP module the trunk's post-final-norm hidden (the final RMSNorm applied in-bench) instead of the pre-final-norm hidden.
        return lm.model.norm(x) if head_h == "postnorm" else x
    t1 = time.time()
    # t2241 step-decomposition instrumentation (worktree-local copy; record
    # bench 54ce16ac untouched): per-phase wall (perf_counter + mx.eval at
    # phase boundaries) + wall_ns stdout markers for kernel-trace correlation.
    # STEPDEC_OUT = JSON output path; STEPDEC_WARM/STEPDEC_WIN = 16/32 default.
    # Timing is only recorded for mode 3; token semantics are unchanged (the
    # added mx.eval calls sit at points where the queue is already drained by
    # the int() syncs of the record bench).
    stepdec_on = bool(os.environ.get("STEPDEC_OUT")) and mode == "3"
    ssteps = []

    def _mk(tag, i):
        if stepdec_on:
            print("PHASE %d S%d %s" % (time.time_ns(), i, tag), flush=True)

    def _steprec(i, kind, **ph):
        rec = {"i": i, "kind": kind}
        for k, v in ph.items():
            rec[k + "_ms"] = v * 1e3
        rec["wall_ms"] = ph["wall"] * 1e3
        ssteps.append(rec)

    for _ in range(MAX_TOKENS):
        if nfill is not None:
            # Catch up the n-gram context over tokens emitted since the last
            # update (the verify-step emissions, incl. the r5 deferred
            # correction); the single-token step's t is updated at append
            # time below (before the lookups that consume it).
            for _tt in out[nfill_len:]:
                nfill.update(_tt)
            nfill_len = len(out)
        if draft1 is None:
            # Single-token step (from S_k)
            _i0 = len(out)
            _t_a = time.perf_counter() if stepdec_on else 0.0
            _mk("step_start", _i0)
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
                if nfill is not None:
                    nfill.update(t)
                    nfill_len += 1
            if mtp_on:
                prev_token = t
                _mk("post_sample", _i0)
                fw1 = True
                fw2 = True
                # Draft token 1 (n-gram fill, else from AR state; the trunk hidden input goes through the MTP6_HEAD_H probe gate). A fill hit skips the MTP forward (fw1 = False -> the mtp_cache row accounting in the mode-3 reject trims).
                c1 = nfill.lookup(1, []) if ngram_s3 else None
                if c1 is not None:
                    n_ng += 1
                    draft1 = c1
                    fw1 = False
                else:
                    emb1 = lm.model.embed_tokens(mx.array([[t]]))
                    dh1 = lm.mtp(hhead(h), emb1, mtp_cache)
                    draft1 = draft_tok(dh1)
                _t_b = time.perf_counter() if stepdec_on else 0.0
                _mk("post_d1", _i0)
                if mode in ("3", "4"):
                    # Draft token 2 (n-gram fill, else from MTP module given token1); a fill hit skips the forward (fw2 = False).
                    c2 = nfill.lookup(2, [draft1]) if ngram_s3 else None
                    if c2 is not None:
                        n_ng += 1
                        draft2 = c2
                        fw2 = False
                    else:
                        emb2 = lm.model.embed_tokens(mx.array([[draft1]]))
                        dh2 = lm.mtp(dh1, emb2, mtp_cache)
                        draft2 = draft_tok(dh2)
                _t_c = time.perf_counter() if stepdec_on else 0.0
                _mk("post_d2", _i0)
                if mode == "4":
                    # Draft token 3 (from MTP module given token2; the ngram_s3 gate is mode-3-only, so this path is untouched)
                    emb3 = lm.model.embed_tokens(mx.array([[draft2]]))
                    dh3 = lm.mtp(dh2, emb3, mtp_cache)
                    draft3 = draft_tok(dh3)
                # State stays S_k (no target forward)
                if stepdec_on:
                    mx.eval()
                    _t_e = time.perf_counter()
                    _steprec(
                        _i0, "draft",
                        sample=_t_b - _t_a, draft1=_t_c - _t_b, draft2=_t_e - _t_c,
                        wall=_t_e - _t_a,
                    )
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
            # Verify step (from S_k, drafts D1, D2). BATCHED S=3. T1917: tape arm (MTP6_TAPE in 1/hybrid/r5; MLX_ROCM_GDN_T3=1, the TMAX=3 instantiation) mirrors the mode-4 commit/reject with K=2: commit = index-select on the per-row state tape + trims (KV trim S-1-krow, mtp_cache trim K-krow); r5 = deferred correction (ctoken becomes row 0 of the next verify, the draft chain re-anchors on the rejected row's pre-norm h); hybrid/1 = ONE S=1 correction forward; MTP6_TAPE=0 = the legacy save/restore + re-forward (unchanged).
            _i0 = len(out)
            _t_a = time.perf_counter() if stepdec_on else 0.0
            _mk("vstart", _i0)
            d1, d2 = draft1, draft2
            draft1 = draft2 = draft3 = None
            verifies += 1
            if tape_mode == "0":
                saved = save_gdn(cache)
            logits, hidden = lm(
                mx.array([[prev_token, d1, d2]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True
            )
            if stepdec_on:
                mx.eval(logits, hidden)
                _t_v = time.perf_counter()
            l0, l1, l2 = logits[0, 0], logits[0, 1], logits[0, 2]
            hb0, hb1, hb2 = hidden[0, 0], hidden[0, 1], hidden[0, 2]

            def commit(krow, prefix):
                # Select the per-row GDN state-tape row (the state AFTER row krow) and rebind the cache leaves to the tape slice (lazy; no copies, no sync). Same structure as the mode-4 commit; the S=3 tape rows are indices 0, 1.
                tape_avail = tape_mode in ("1", "hybrid", "r5", "slot")
                if tape_avail:
                    try:
                        st0 = lm.model.layers[0].linear_attn.last_tape[0]
                        tape_avail = st0 is not None
                    except Exception:
                        tape_avail = False
                if tape_avail:
                    _cc = os.environ.get("MTP6_SLOT_COPY", "0")
                    _copy = (tape_mode == "slot" and _cc in ("1", "2"))
                    for i, c in enumerate(cache):
                        if hasattr(c, "cache"):
                            st, cstab = lm.model.layers[i].linear_attn.last_tape
                            # tape rows are B-squeezed: restore the B dim
                            _s1 = st[krow:krow+1]
                            _s0 = cstab[krow:krow+1]
                            if _copy:
                                c[1] = mx.astype(_s1, _s1.dtype)
                                c[0] = mx.astype(_s0, _s0.dtype)
                            else:
                                c[1] = _s1
                                c[0] = _s0
                    if _copy and _cc == "1":
                        mx.eval(*[c[1] for c in cache if hasattr(c, "cache")])
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(2 - krow)
                    return True
                else:
                    # legacy (MTP6_TAPE=0): canonical restore + trim ALL verify rows + re-forward of the accepted prefix.
                    restore_gdn(cache, saved)
                    for c in cache:
                        if isinstance(c, KVCache):
                            c.trim(3)
                    for pt in prefix:
                        lm(mx.array([[pt]]), cache=cache)
                    return False

            def reject(krow, lk, hb, ktrim, prefix, ctoken):
                nonlocal l, h, n_refwd, pending_c, h_r, n_cdeferred
                committed = commit(krow, prefix)
                if os.environ.get("MTP6_MTPCACHE_TRIM", "1") == "1":
                    mtp_cache[0].trim(ktrim)
                if tape_mode in ("r5", "slot") and not (tape_mode == "slot" and os.environ.get("MTP6_SLOT_IMM", "0") == "1") and committed:
                    # R5: no S=1 correction re-forward; ctoken = row 0 of the next verify, draft re-anchors on the rejected row's h
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

            # t1986: mtp_cache row accounting - n_fwd rows were pushed this step (one per real draft forward; an n-gram fill hit skips the push); a rejection keeps the accepted rows -> trim n_fwd - n_accepted. With fw1=fw2=True (ngram off) this reduces to the canonical 1 (d2 reject) / 2 (d1 reject) constants.
            n_fwd = int(fw1) + int(fw2)
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
                    # Full accept: the model's state (after row 1) is already in the leaves; no rebind, no trims.
                    l, h = l2, hb2
                else:
                    # draft2 rejected: commit row 1 (after D1), correction token t3; mtp_cache trim = n_fwd - 1 accepted row.
                    reject(1, l1, hb1, max(0, n_fwd - int(fw1)), [prev_token, d1], t3)
            else:
                # draft1 rejected: commit row 0 (after prev_token), correction token t2; mtp_cache trim = all n_fwd rows.
                reject(0, l0, hb0, n_fwd, [prev_token], t2)
            if stepdec_on:
                mx.eval()
                _t_e = time.perf_counter()
                _mk("post_commit", _i0)
                _steprec(
                    _i0, "verify",
                    sample=0.0, draft1=0.0, draft2=0.0,
                    verify=_t_v - _t_a, commit=_t_e - _t_v, wall=_t_e - _t_a,
                )
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
            l0, l1, l2, l3 = logits[0, 0], logits[0, 1], logits[0, 2], logits[0, 3]
            # batched S=4 row pre-norm hiddens (pure tape reuses these for the next draft chain on rejected steps)
            hb0, hb1, hb2, hb3 = hidden[0, 0], hidden[0, 1], hidden[0, 2], hidden[0, 3]

            def commit(krow, prefix):
                # Select the per-row GDN state-tape row (the state AFTER row krow) and rebind the cache leaves to the tape slice (lazy; no copies, no sync). lengths is None in this model (advance() is a no-op), so no lengths fix-up.
                tape_avail = tape_mode in ("1", "hybrid", "r5", "slot")
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
                if tape_mode in ("r5", "slot") and committed:
                    # R5: no S=1 correction re-forward; ctoken = row 0 of the next verify, draft re-anchors on the rejected row's h
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
    if stepdec_on:
        _payload = {
            "steps": ssteps, "n": n, "gen_s": dt, "verifies": verifies,
            "d1_acc": d1_acc, "d2_acc": d2_acc, "d3_acc": d3_acc,
            "n_refwd": n_refwd, "n_cdeferred": n_cdeferred,
            "tok_step": tok_step, "tok_s": n / max(dt, 1e-9),
            "alltoks_md5": hashlib.md5(
                " ".join(map(str, out)).encode()
            ).hexdigest(),
        }
        with open(os.environ["STEPDEC_OUT"], "w") as f:
            json.dump(_payload, f)
        print("STEPDEC_JSON %s" % os.environ["STEPDEC_OUT"], flush=True)
    if ngram_s3:
        print(
            "INFO NGRAM_S3 n=%d NNG=%d" % (ng_n, n_ng),
            flush=True,
        )
    if mode in ("3", "4"):
        import mlx_lm.models.qwen3_5 as _m5
        print(
            "INFO TAPEMODE=%s REFWDS=%d CDEFERRED=%d GDN_TAPE_CALLS=%d HEAD=%s HEAD_H=%s" % (
                tape_mode,
                n_refwd,
                n_cdeferred,
                getattr(_m5, "GDN_TAPE_CALLS", -1),
                head,
                head_h,
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
