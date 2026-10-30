#!/usr/bin/env python3
"""m8 P3: MTP-flow bisection at the G4 first_div state (token index 148).

Reproduces the mtp-bench3 mode-2 flow EXACTLY (same prompt, same
accept/reject logic, same f32 mtp_cache restore on reject) and forks at
the verify step that appends token index 148 (m2a first_div=148,
a=33443 b=24296):

  A  = S=2 [prev, d]   (rows 0 + 1)
  B1 = S=1 [prev]      (row 0)  from the pre-verify state
  B2 = S=1 [d]         (row 0)  from the state after B1
  B3 = S=1 [t2]        (row 0)  from the state after B2 (reject path)

Row-0 invariance: A row 0 vs B1 row 0 (the l0 source).
Row-1 invariance: A row 1 vs B2 row 0 (the l1 source on ACCEPT - if
t2==d, the flow's l = l1; a row-1 diff explains the token flip at 148).
Dynamic proof: A post-state vs B1+B2 post-state (GDN + KV, every buffer).

Captured per forward (row 0 for B arms; rows 0+1 for A):
  - mx.fast.gdn_prep inputs (q,k,v,conv_state) + outputs (qn,kn,vc,beta,g)
    per GDN-layer call order
  - mx.fast.gated_delta_update output (out, state)
  - mx.fast.gdn_norm_gated output
  - decoder layer outputs (all 32)
  - FA-layer components (input_layernorm, q/k/v proj, norms, sdpa q/k/v/out)
All numpy forced-eval; restore is DTYPE-PRESERVING (no f32 artifact).
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
TARGET_IDX = 48
EOS = 248044
MAX_TOKENS = 170


def npf(a):
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def snap_all(cache, mtp_cache):
    """dtype-preserving numpy snapshot of the full state."""
    gdn, kv, mtp = [], [], None
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            kv.append((i, c.keys.dtype, npf(c.keys), c.values.dtype,
                      npf(c.values), c.offset))
        else:
            ln = c.lengths
            gdn.append((i, c[0].dtype, npf(c[0]), c[1].dtype, npf(c[1]),
                       npf(ln) if ln is not None else None))
    if mtp_cache:
        c = mtp_cache[0]
        if c.keys is not None:
            mtp = (c.keys.dtype, npf(c.keys), c.values.dtype,
                   npf(c.values), c.offset)
    return gdn, kv, mtp


def restore_all(cache, mtp_cache, snap):
    gdn, kv, mtp = snap
    for i, dt0, cc, dt1, ss, ln in gdn:
        cache[i][0] = mx.array(cc, dtype=dt0)
        cache[i][1] = mx.array(ss, dtype=dt1)
        cache[i].lengths = mx.array(ln,
                                   dtype=cache[i].lengths.dtype) if ln is not None else None
    for i, dtk, kk, dtv, vv, off in kv:
        cache[i].keys = mx.array(kk, dtype=dtk)
        cache[i].values = mx.array(vv, dtype=dtv)
        cache[i].offset = off
    if mtp is not None and mtp_cache:
        dtk, kk, dtv, vv, off = mtp
        mtp_cache[0].keys = mx.array(kk, dtype=dtk)
        mtp_cache[0].values = mx.array(vv, dtype=dtv)
        mtp_cache[0].offset = off


def cmp(name, a, b):
    if a is None or b is None:
        return f"{name}: SKIP"
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return f"{name}: SHAPE {a.shape} vs {b.shape}"
    d = a - b
    n = int(np.count_nonzero(d != 0))
    ma = float(np.abs(d).max()) if d.size else 0.0
    tag = "" if n == 0 else "  <-- T-DEP"
    return f"{name}: n={n}/{d.size} maxabs={ma:.3e}{tag}"


def main():
    from mlx_lm import load
    import mlx_lm.models.qwen3_next as qn

    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    mtp_cache = [KVCache()]
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    logits, hidden = lm(mx.array([prompt_ids]), cache=cache,
                       return_hidden=True, return_prenorm=True)
    l = logits[0, -1]
    h = hidden[0, -1]

    # ---------- capture machinery ----------
    CAP = {"mode": None, "seq": {}, "gdn_call": 0}

    def put(key, arr):
        if CAP["mode"] is None:
            return
        e = CAP["seq"].setdefault(key, {})
        if CAP["mode"] not in e:
            e[CAP["mode"]] = npf(arr)

    # mx.fast GDN hooks (fused path is always active in venv-d)
    orig_gdn_prep = mx.fast.gdn_prep

    def gdn_prep_wrap(*a, **k):
        out = orig_gdn_prep(*a, **k)
        if CAP["mode"] is not None:
            i = CAP["gdn_call"]
            CAP["gdn_call"] += 1
            put(f"gp{i}_q", a[0])
            put(f"gp{i}_k", a[1])
            put(f"gp{i}_v", a[2])
            put(f"gp{i}_cs", a[7])
            qn_, kn_, vc_, beta_, g_ = out
            put(f"gp{i}_qn", qn_)
            put(f"gp{i}_kn", kn_)
            put(f"gp{i}_vc", vc_)
            put(f"gp{i}_beta", beta_)
            put(f"gp{i}_g", g_)
        return out

    orig_gdu = mx.fast.gated_delta_update

    def gdu_wrap(*a, **k):
        out = orig_gdu(*a, **k)
        if CAP["mode"] is not None:
            i = CAP["gdn_call"] - 1
            put(f"gdu{i}_out", out[0])
            put(f"gdu{i}_state", out[1])
        return out

    orig_ng = mx.fast.gdn_norm_gated

    def ng_wrap(*a, **k):
        out = orig_ng(*a, **k)
        if CAP["mode"] is not None:
            i = CAP["gdn_call"] - 1
            put(f"ng{i}_out", out)
        return out

    mx.fast.gdn_prep = gdn_prep_wrap
    mx.fast.gated_delta_update = gdu_wrap
    mx.fast.gdn_norm_gated = ng_wrap

    # decoder-layer hooks (all 32): rows 0+1 for A, row 0 for B arms
    layers = lm.model.layers
    id_to_idx = {id(layers[i]): i for i in range(len(layers))}
    gdn_layer_ids = [i for i, ly in enumerate(layers)
                    if hasattr(ly, "linear_attn")
                    and isinstance(ly.linear_attn, qn.Qwen3NextGatedDeltaNet)]
    layer_cls = type(layers[0])
    orig_layer = layer_cls.__call__

    def hooked_layer(self, *args, **kwargs):
        out = orig_layer(self, *args, **kwargs)
        i = id_to_idx.get(id(self))
        if i is not None and CAP["mode"] is not None:
            o = out[0] if isinstance(out, tuple) else out
            put(f"L{i}_out", o)
            x0 = args[0]
            put(f"L{i}_x", x0 if isinstance(x0, mx.array)
                else mx.array(x0))
        return out

    layer_cls.__call__ = hooked_layer

    def wrap_mod(mod, key):
        orig = mod.__call__

        def w(*a, **k):
            o = orig(*a, **k)
            if CAP["mode"] is not None:
                put(key, o)
            return o
        mod.__call__ = w
        return orig

    # per-decoder-layer norm hooks (both GDN and FA layers)
    for i, ly in enumerate(layers):
        wrap_mod(ly.input_layernorm, f"L{i}_input_ln")
        wrap_mod(ly.post_attention_layernorm, f"L{i}_post_ln")

    # FA-layer component hooks (per-instance module __call__, per-layer keys)
    fa_layers = [i for i, ly in enumerate(layers)
                if hasattr(ly, "self_attn")
                and isinstance(ly.self_attn, qn.Qwen3NextAttention)]

    fa_id_to_idx = {id(layers[i].self_attn): i for i in fa_layers}
    orig_fa = []
    for i in fa_layers:
        at = layers[i].self_attn
        for nm in ("q_proj", "q_norm", "k_proj", "k_norm", "v_proj",
                   "o_proj"):
            orig_fa.append(wrap_mod(getattr(at, nm), f"FA{i}_{nm}"))
        rope_seq = {"n": 0}
        orig_rope = at.rope.__call__

        def make_rope_w(_o, _s, _i=i):
            def rw(*a, **k):
                o = _o(*a, **k)
                if CAP["mode"] is not None:
                    put(f"FA{_i}_rope_{chr(97 + min(_s['n'], 1))}", o)
                    _s["n"] += 1
                return o
            return rw
        at.rope.__call__ = make_rope_w(orig_rope, rope_seq)

    # per-layer sdpa capture (class-level attention __call__ sets current idx)
    CUR = {"fa": None}
    attn_cls = qn.Qwen3NextAttention
    orig_attn = attn_cls.__call__

    def attn_hook(self, *a, **k):
        CUR["fa"] = fa_id_to_idx.get(id(self))
        if CAP["mode"] is not None and CUR["fa"] == 15 and len(a) >= 1:
            x = a[0]
            try:
                put("FA15_kprojT2", self.k_proj(x))
                put("FA15_kprojT1", self.k_proj(x[:, 1:2, :]))
                put("FA15_kprojT1b", self.k_proj(x[:, 0:1, :]))
            except Exception as ex:
                print(f"  kprojT hook failed: {ex}")
        return orig_attn(self, *a, **k)

    attn_cls.__call__ = attn_hook

    orig_sdpa = qn.scaled_dot_product_attention

    def sdpa_wrap(queries, keys, values, cache=None, scale=1.0, mask=None):
        out = orig_sdpa(queries, keys, values, cache=cache, scale=scale,
                       mask=mask)
        out2 = orig_sdpa(queries, keys, values, cache=cache, scale=scale,
                        mask=mask)
        if CAP["mode"] is not None and CUR["fa"] is not None:
            fi = CUR["fa"]
            put(f"FA{fi}_sdpa_q", queries)
            put(f"FA{fi}_sdpa_k", keys)
            put(f"FA{fi}_sdpa_v", values)
            put(f"FA{fi}_scale", float(scale))
            put(f"FA{fi}_sdpa_out", out)
            put(f"FA{fi}_sdpa_out2", out2)
        return out

    qn.scaled_dot_product_attention = sdpa_wrap

    # ---------- mode-2 flow (exact mtp-bench3) ----------
    def save_gdn(cache):
        return [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache)
                if hasattr(c, "cache")]

    def restore_gdn(cache, saved):
        for i, conv, ssm, lengths in saved:
            cache[i][0] = conv
            cache[i][1] = ssm
            cache[i].lengths = lengths

    def save_mtp(mtp_cache):
        c = mtp_cache[0]
        if c.keys is None:
            return (None, None, c.offset)
        return (npf(c.keys), npf(c.values), c.offset)

    def restore_mtp(mtp_cache, saved):
        keys, values, offset = saved
        c = mtp_cache[0]
        if keys is not None:
            c.keys = mx.array(keys)
            c.values = mx.array(values)
        c.offset = offset

    out = []
    accepted = 0
    verifies = 0
    draft = None
    prev_token = None
    mtp_saved = None
    fork_done = False
    fork_lh = None

    def fork_bisect(d, prev_token):
        """Fork at the verify step appending token index TARGET_IDX-1
        (index TARGET_IDX is a single-token step: t = argmax(l), where l
        = l1 (A row 1) on accept or the [t2] re-forward on reject)."""
        nonlocal fork_done
        snap_pre = snap_all(cache, mtp_cache)
        off0 = [c.offset for c in cache if isinstance(c, KVCache)]
        print(f"DEBUG offsets pre-verify (first 8 KV): {off0[:8]}", flush=True)
        # --- Arm A: S=2 [prev, d] ---
        CAP["mode"] = "A"
        CAP["gdn_call"] = 0
        logitsA, hiddenA = lm(mx.array([[prev_token, d]]), cache=cache,
                             return_hidden=True, return_prenorm=True)
        l0A, h0A = logitsA[0, 0], hiddenA[0, 0]
        l1A, h1A = logitsA[0, 1], hiddenA[0, 1]
        t2 = int(mx.argmax(l0A))
        print(f"DEBUG offsets post-A: {[c.offset for c in cache if isinstance(c, KVCache)][:8]}", flush=True)
        snap_Apost = snap_all(cache, mtp_cache)
        # --- restore pre-verify; Arm B1: S=1 [prev] ---
        restore_all(cache, mtp_cache, snap_pre)
        print(f"DEBUG offsets post-restore: {[c.offset for c in cache if isinstance(c, KVCache)][:8]}", flush=True)
        CAP["mode"] = "B1"
        CAP["gdn_call"] = 0
        logitsB1, hiddenB1 = lm(mx.array([[prev_token]]), cache=cache,
                               return_hidden=True, return_prenorm=True)
        print(f"DEBUG offsets post-B1: {[c.offset for c in cache if isinstance(c, KVCache)][:8]}", flush=True)
        snap_B1post = snap_all(cache, mtp_cache)
        # --- Arm B2: S=1 [d] ---
        CAP["mode"] = "B2"
        CAP["gdn_call"] = 0
        logitsB2, hiddenB2 = lm(mx.array([[d]]), cache=cache,
                               return_hidden=True, return_prenorm=True)
        print(f"DEBUG offsets post-B2: {[c.offset for c in cache if isinstance(c, KVCache)][:8]}", flush=True)
        for nm in ("FA3_sdpa_k", "FA15_sdpa_k", "FA3_sdpa_q"):
            e = CAP["seq"].get(nm, {})
            for arm in ("A", "B1", "B2", "B3"):
                if arm in e:
                    print(f"DEBUG shape {arm} {nm}: {e[arm].shape}", flush=True)
        ea = CAP["seq"].get("FA3_sdpa_k", {}).get("A")
        eb = CAP["seq"].get("FA3_sdpa_k", {}).get("B2")
        if ea is not None and eb is not None:
            for i in range(min(ea.shape[2], eb.shape[2])):
                dd = ea[0, 0, i] - eb[0, 0, i]
                n = int(np.count_nonzero(dd != 0))
                if n > 0:
                    print(f"DEBUG FA3_k row {i}: n={n}/{dd.size} "
                          f"maxabs={float(np.abs(dd).max()):.3e} "
                          f"a[:4]={ea[0, 0, i, :4].tolist()} "
                          f"b[:4]={eb[0, 0, i, :4].tolist()}")
            print(f"DEBUG FA3_k prefix rows 0..72: n_diff_rows="
                  f"{sum(1 for i in range(min(73, ea.shape[2])) if np.any(ea[0, 0, i] != eb[0, 0, i]))}")
        if ea is not None and eb is not None:
            dd = ea[0, 0] - eb[0, 0]
            print(f"DEBUG FA3_k fullslice diff: n={int(np.count_nonzero(dd != 0))}/{dd.size} maxabs={float(np.abs(dd).max()):.3e} sum_a={float(ea[0, 0].sum()):.3f} sum_b={float(eb[0, 0].sum()):.3f} shape={ea[0, 0].shape}")
        snap_B2post = snap_all(cache, mtp_cache)
        # --- Arm B3: S=1 [t2] (reject path) ---
        CAP["mode"] = "B3"
        CAP["gdn_call"] = 0
        logitsB3, hiddenB3 = lm(mx.array([[t2]]), cache=cache,
                               return_hidden=True, return_prenorm=True)
        CAP["mode"] = None

        # ---------- reports ----------
        S = CAP["seq"]
        print(f"=== FORK at verify step appending token index {len(out)} "
              f"(t2={t2}) ===")
        print(f"PREDICTED token@{TARGET_IDX}: mtp(argmax l1A)="
              f"{int(np.argmax(npf(l1A)))} ar(argmax l_B2)="
              f"{int(np.argmax(npf(logitsB2[0, 0])))}")
        print(f"fork step: d={d} t2={t2} accept={t2 == d}")
        d = npf(l0A) - npf(logitsB1[0, 0])
        print(f"l0: A vs B1 n={np.count_nonzero(d != 0)}/{d.size} "
              f"maxabs={np.abs(d).max():.3e} t2(A)={t2} "
              f"t2(B1)={int(np.argmax(npf(logitsB1[0, 0])))}")
        d = npf(l1A) - npf(logitsB2[0, 0])
        print(f"l1(row1): A vs B2 n={np.count_nonzero(d != 0)}/{d.size} "
              f"maxabs={np.abs(d).max():.3e}")
        d = npf(logitsB3[0, 0]) - npf(logitsB2[0, 0])
        print(f"(ref) B3[t2] vs B2[d] row0 n={np.count_nonzero(d != 0)}/{d.size} "
              f"maxabs={np.abs(d).max():.3e}")
        # ---------- row-r extraction (layout-aware) ----------
        def rowx(a, r, key):
            a = np.asarray(a, dtype=np.float32)
            if "_state" in key:
                return a[0].reshape(-1) if a.ndim >= 1 else a.reshape(-1)
            if "_sdpa_q" in key or "_sdpa_out" in key or "_rope_" in key:
                # (B, H, L, D) - L is axis 2
                aa = a[0] if a.ndim == 4 else a
                if aa.ndim == 3:
                    return (aa[:, r, :].reshape(-1)
                           if aa.shape[1] > r else None)
                if aa.ndim == 2:
                    return aa[r].reshape(-1) if aa.shape[0] > r else None
                return aa.reshape(-1)
            if "_sdpa_k" in key or "_sdpa_v" in key:
                # (B, kvL, H_kv, D) or (kvL, H_kv, D) - prefix comparison
                # (row 0 sees only the prefix + its own key; the extra
                # row in A is unseen)
                aa = a
                while aa.ndim > 2:
                    aa = aa[0]
                return aa if aa.ndim == 2 else aa.reshape(aa.shape[0], -1)
            if a.ndim >= 3:
                # (B, S, ...) - S is axis 1
                return a[0, r].reshape(-1) if a.shape[1] > r else None
            if a.ndim == 2:
                return a[r].reshape(-1) if a.shape[0] > r else None
            return a.reshape(-1)

        # row-0 bisection: A row0 vs B1 row0
        full = []
        gdn_i = 0
        for i in range(len(layers)):
            if i in gdn_layer_ids:
                gi = gdn_i
                full += [f"L{i}_input_ln", f"gp{gi}_q", f"gp{gi}_k", f"gp{gi}_v", f"gp{gi}_cs",
                         f"gp{gi}_qn", f"gp{gi}_kn", f"gp{gi}_vc",
                         f"gp{gi}_beta", f"gp{gi}_g", f"gdu{gi}_out",
                         f"gdu{gi}_state", f"ng{gi}_out"]
                gdn_i += 1
            else:
                full += [f"L{i}_input_ln", f"FA{i}_q_proj", f"FA{i}_q_norm",
                         f"FA{i}_k_proj", f"FA{i}_k_norm", f"FA{i}_v_proj",
                         f"FA{i}_rope_a", f"FA{i}_rope_b", f"FA{i}_sdpa_q",
                         f"FA{i}_sdpa_k", f"FA{i}_sdpa_v", f"FA{i}_sdpa_out",
                         f"FA{i}_o_proj"]
            full.append(f"L{i}_out")
            full.append(f"L{i}_post_ln")
        full.append("LOGITS_r0")
        first = None
        for key in full:
            e = S.get(key)
            if e is None or "A" not in e or "B1" not in e:
                continue
            a0 = rowx(e["A"], 0, key)
            b0 = rowx(e["B1"], 0, key)
            if "_sdpa_k" in key or "_sdpa_v" in key:
                m = min(a0.shape[0], b0.shape[0])
                a0, b0 = a0[:m].reshape(-1), b0[:m].reshape(-1)
            if a0 is None or b0 is None or a0.shape != b0.shape:
                sa = None if a0 is None else a0.shape
                sb = None if b0 is None else b0.shape
                print(f"  {key}: SHAPE {sa} vs {sb} (skipped)")
                continue
            d = a0 - b0
            n = int(np.count_nonzero(d != 0))
            ma = float(np.abs(d).max()) if d.size else 0.0
            tag = "" if n == 0 else "  <-- T-DEP"
            print(f"  {key}: n={n}/{d.size} maxabs={ma:.3e}{tag}")
            if n > 0 and first is None:
                first = key
        print(f"ROW-0 FIRST DIVERGING COMPONENT: {first}")
        # row-1 bisection: A row1 vs B2 row0 (accept path l source)
        first1 = None
        for key in full:
            e = S.get(key)
            if e is None or "A" not in e or "B2" not in e:
                continue
            a1 = rowx(e["A"], 1, key)
            b0 = rowx(e["B2"], 0, key)
            if "_sdpa_k" in key or "_sdpa_v" in key:
                m = min(a1.shape[0], b0.shape[0])
                a1, b0 = a1[:m].reshape(-1), b0[:m].reshape(-1)
            if a1 is None or b0 is None or a1.shape != b0.shape:
                continue
            d = a1 - b0
            if key == "FA3_sdpa_k":
                print(f"DEBUG rowx {key}: sum_a={float(a1.sum()):.3f} sum_b={float(b0.sum()):.3f} a.shape={a1.shape} b.shape={b0.shape}", flush=True)
            n = int(np.count_nonzero(d != 0))
            if n > 0 and first1 is None:
                ma = float(np.abs(d).max())
                print(f"  [row1] {key}: n={n}/{d.size} "
                      f"maxabs={ma:.3e}  <-- FIRST ROW-1 T-DEP")
                first1 = key
        print(f"ROW-1 FIRST DIVERGING COMPONENT: {first1}")
        # dynamic proof: A post-state vs B1+B2 post-state
        print("=== DYNAMIC PROOF: A post-state vs B1+B2 post-state ===")
        leak = False
        for tag, sa, sb in (("Apost-vs-B2post", snap_Apost, snap_B2post),):
            for gA, gB in zip(sa[0], sb[0]):
                for j, (x, y, nm) in enumerate([(gA[2], gB[2], "conv"),
                                               (gA[4], gB[4], "ssm")]):
                    if x is None or y is None:
                        continue
                    dd = x - y
                    n = int(np.count_nonzero(dd != 0))
                    if n > 0:
                        leak = True
                        print(f"  GDN layer {gA[0]} {nm}: n={n}/{dd.size} "
                              f"maxabs={np.abs(dd).max():.3e} (LEAK)")
            for kA, kB in zip(sa[1], sb[1]):
                for x, y, nm in [(kA[2], kB[2], "keys"),
                                (kA[4], kB[4], "values")]:
                    dd = x - y
                    n = int(np.count_nonzero(dd != 0))
                    if n > 0:
                        leak = True
                        print(f"  KV layer {kA[0]} {nm}: n={n}/{dd.size} "
                              f"maxabs={np.abs(dd).max():.3e} (LEAK)")
        print(f"DYNAMIC PROOF: {'LEAK FOUND' if leak else 'NO LEAK (state exact)'}")
        # --- P3 probe: fresh SDPA recompute from the captured f32 tensors ---
        print("=== P3 FRESH-RECOMPUTE PROBE (from captured tensors) ===")
        import mlx.core as _mx
        try:
            scale = float(lm.layers[15].self_attn.scale)
        except Exception as ex:  # pragma: no cover
            scale = 0.125
            print(f"  scale lookup failed ({ex}); using {scale}")
        print(f"  model scale={scale}")
        for L in (3, 7, 11, 15, 19, 23, 27, 31):
            sc = CAP["seq"].get(f"FA{L}_scale", {}).get("A")
            print(f"  FA{L} hook scale: A={float(sc) if sc is not None else 'NA'} B2={float(CAP['seq'].get(f'FA{L}_scale', {}).get('B2', float('nan')))}")
            def g(arm, nm, L=L):
                return CAP["seq"].get(f"FA{L}_{nm}", {}).get(arm)
            qa, ka, va, oa = g("A", "sdpa_q"), g("A", "sdpa_k"), g("A", "sdpa_v"), g("A", "sdpa_out")
            qb, kb, vb, ob = g("B2", "sdpa_q"), g("B2", "sdpa_k"), g("B2", "sdpa_v"), g("B2", "sdpa_out")
            if any(x is None for x in (qa, ka, va, qb, kb, vb, oa, ob)):
                print(f"  FA{L}: capture missing")
                continue
            qA = _mx.array(qa).astype(_mx.bfloat16)
            kA = _mx.array(ka).astype(_mx.bfloat16)
            vA = _mx.array(va).astype(_mx.bfloat16)
            qB = _mx.array(qb).astype(_mx.bfloat16)
            kB = _mx.array(kb).astype(_mx.bfloat16)
            vB = _mx.array(vb).astype(_mx.bfloat16)
            oA_f = _mx.fast.scaled_dot_product_attention(qA, kA, vA, scale=scale, mask="causal")
            oB_f = _mx.fast.scaled_dot_product_attention(qB, kB, vB, scale=scale)
            oA_f2 = _mx.fast.scaled_product_attention if False else _mx.fast.scaled_dot_product_attention(qA, kA, vA, scale=scale, mask="causal")
            aA2 = np.asarray(oA_f2[:, :, 1, :].astype(_mx.float32))[0]
            aA = np.asarray(oA_f[:, :, 1, :].astype(_mx.float32))[0]
            aB = np.asarray(oB_f[:, :, 0, :].astype(_mx.float32))[0]
            fA = oa[0, :, 1, :] if oa.ndim == 4 else oa[1]
            fB = ob[0, :, 0, :] if ob.ndim == 4 else ob[0]
            print(f"  FA{L} shapes: qa={qa.shape} ka={ka.shape} oa={oa.shape} oA_f={tuple(oA_f.shape)} aA={aA.shape} fA={fA.shape}")
            d1, d2, d3 = aA - fA, aB - fB, aA - aB
            d4 = aA - aA2
            # strided-layout recompute (mimic in-flow transposed q / buffer-slice k,v)
            bigA = _mx.zeros((1, 2, 16, 512), dtype=_mx.bfloat16)
            bigA[:, :, :, :256] = _mx.array(qa).transpose(0, 2, 1, 3).astype(_mx.bfloat16)
            qA_s = bigA.transpose(0, 2, 1, 3)[:, :, :, :256]
            bigB = _mx.zeros((1, 1, 16, 512), dtype=_mx.bfloat16)
            bigB[:, :, :, :256] = _mx.array(g("B2", "sdpa_q")).transpose(0, 2, 1, 3).astype(_mx.bfloat16)
            qB_s = bigB.transpose(0, 2, 1, 3)[:, :, :, :256]
            kA_s = _mx.zeros((1, 4, 256, 256), dtype=_mx.bfloat16); kA_s[:, :, :ka.shape[2], :] = kA
            vA_s = _mx.zeros((1, 4, 256, 256), dtype=_mx.bfloat16); vA_s[:, :, :va.shape[2], :] = vA
            kB_s = _mx.zeros((1, 4, 256, 256), dtype=_mx.bfloat16); kB_s[:, :, :kb.shape[2], :] = kB
            vB_s = _mx.zeros((1, 4, 256, 256), dtype=_mx.bfloat16); vB_s[:, :, :vb.shape[2], :] = vB
            oA_ls = _mx.fast.scaled_dot_product_attention(qA_s, kA_s, vA_s, scale=scale, mask="causal")
            aA_ls = np.asarray(oA_ls[:, :, 1, :].astype(_mx.float32))[0]
            d6 = aA_ls - fA
            # A-row1 values, S=1 no-mask call vs A-row1 values, S=2 causal row 1
            qAr1 = _mx.array(qa)[:, :, 1:2, :].astype(_mx.bfloat16)
            oBstyle = _mx.fast.scaled_dot_product_attention(qAr1, kA, vA, scale=scale)
            aBstyle = np.asarray(oBstyle[:, :, 0, :].astype(_mx.float32))[0]
            d7 = aBstyle - aA
            d8 = aBstyle - aB
            qAr1c = _mx.array(qa)[:, :, 1:2, :].astype(_mx.bfloat16)
            kAc = _mx.array(ka).astype(_mx.bfloat16)
            vAc = _mx.array(va).astype(_mx.bfloat16)
            oBstyle2 = _mx.fast.scaled_dot_product_attention(qAr1c, kAc, vAc, scale=scale)
            aBstyle2 = np.asarray(oBstyle2[:, :, 0, :].astype(_mx.float32))[0]
            d9 = aBstyle2 - aBstyle
            oX = _mx.fast.scaled_dot_product_attention(qAr1, kAc, vAc, scale=scale)
            aX = np.asarray(oX[:, :, 0, :].astype(_mx.float32))[0]
            d10 = aX - aBstyle
            oY = _mx.fast.scaled_dot_product_attention(qAr1c, kA, vA, scale=scale)
            aY = np.asarray(oY[:, :, 0, :].astype(_mx.float32))[0]
            d11 = aY - aBstyle
            print(f"  FA{L} valcheck: q n={int(np.count_nonzero(qa != qb))}/{qa.size} max={np.abs(qa-qb).max():.3e} | q-r1-vs-b0 n={int(np.count_nonzero(qa[:, :, 1, :] != qb[:, :, 0, :]))}/{qa[:, :, 1, :].size} | k n={int(np.count_nonzero(ka != kb))}/{ka.size} | v n={int(np.count_nonzero(va != vb))}/{va.size}")
            rb = lambda x: np.asarray(_mx.array(x).astype(_mx.bfloat16).astype(_mx.float32))
            print(f"  FA{L} roundtrip: q n={int(np.count_nonzero(rb(qa) != rb(qb)))}/{qa.size} | k n={int(np.count_nonzero(rb(ka) != rb(kb)))}/{ka.size} | v n={int(np.count_nonzero(rb(va) != rb(vb)))}/{va.size}")
            idx = np.argwhere(ka != kb)
            if idx.size:
                t0 = tuple(idx[0])
                print(f"  FA{L} k diff at {t0}: a={ka[t0]:.6f} b={kb[t0]:.6f} | n_diff={len(idx)}")
            # rope_b (pre-cache) vs cached k row 74, per arm
            rba = CAP["seq"].get(f"FA{L}_rope_b", {}).get("A")
            rbb = CAP["seq"].get(f"FA{L}_rope_b", {}).get("B2")
            if rba is not None:
                dA = rba[0, :, 1, :] - ka[0, :, 74, :]
                print(f"  FA{L} ropeA-r1 vs cachedA-row74: n={int((dA!=0).sum())}/{dA.size} maxabs={np.abs(dA).max():.3e}")
            if rbb is not None:
                dB = rbb[0, :, 0, :] - kb[0, :, 74, :]
                print(f"  FA{L} ropeB-r0 vs cachedB-row74: n={int((dB!=0).sum())}/{dB.size} maxabs={np.abs(dB).max():.3e}")
            if rba is not None and rbb is not None:
                dR = rba[0, :, 1, :] - rbb[0, :, 0, :]
                print(f"  FA{L} ropeA-r1 vs ropeB-r0: n={int((dR!=0).sum())}/{dR.size} maxabs={np.abs(dR).max():.3e}")
            if L == 15:
                k2 = CAP["seq"].get("FA15_kprojT2", {}).get("A")
                k1 = CAP["seq"].get("FA15_kprojT1", {}).get("A")
                k1b = CAP["seq"].get("FA15_kprojT1b", {}).get("A")
                if k2 is not None and k1 is not None:
                    dK = k2[0, 1] - k1[0, 0]
                    print(f"  kprojT A: T2-row1 vs T1(row1): n={int((dK!=0).sum())}/{dK.size} maxabs={np.abs(dK).max():.3e}")
                if k2 is not None and k1b is not None:
                    dK0 = k2[0, 0] - k1b[0, 0]
                    print(f"  kprojT A: T2-row0 vs T1(row0): n={int((dK0!=0).sum())}/{dK0.size} maxabs={np.abs(dK0).max():.3e}")
            o2 = CAP["seq"].get(f"FA{L}_sdpa_out2", {}).get("A")
            if o2 is not None:
                fA2 = o2[0, :, 1, :] if o2.ndim == 4 else o2[1]
                d5 = fA - fA2
            else:
                fA2, d5 = fA, fA
            print(f"  FA{L}: freshA-r1 vs flowA-r1: n={int((d1!=0).sum())}/{d1.size} maxabs={np.abs(d1).max():.3e} | "
                  f"freshB vs flowB: n={int((d2!=0).sum())}/{d2.size} maxabs={np.abs(d2).max():.3e} | "
                  f"freshA-r1 vs freshB: n={int((d3!=0).sum())}/{d3.size} maxabs={np.abs(d3).max():.3e} | "
                  f"freshA-r1 rerun: n={int((d4!=0).sum())}/{d4.size} maxabs={np.abs(d4).max():.3e} | "
                  f"flowA-r1 vs flowA-r1rerun: n={int((d5!=0).sum())}/{d5.size} maxabs={np.abs(d5).max():.3e} | "
                  f"stridedA-r1 vs flowA-r1: n={int((d6!=0).sum())}/{d6.size} maxabs={np.abs(d6).max():.3e} | "
                  f"stridedA-r1 vs freshA-r1: n={int(((aA_ls-aA)!=0).sum())}/{aA.size} maxabs={np.abs(aA_ls-aA).max():.3e} | "
                  f"A-r1values-S1nomask vs A-r1values-S2causal: n={int((d7!=0).sum())}/{d7.size} maxabs={np.abs(d7).max():.3e} | "
                  f"S1(A-r1vals) vs S1(B2-vals): n={int((d8!=0).sum())}/{d8.size} maxabs={np.abs(d8).max():.3e} | "
                  f"S1(copy1) vs S1(copy2 same vals): n={int((d9!=0).sum())}/{d9.size} maxabs={np.abs(d9).max():.3e} | "
                  f"S1(q1,kv2) vs S1(q1,kv1): n={int((d10!=0).sum())}/{d10.size} maxabs={np.abs(d10).max():.3e} | "
                  f"S1(q2,kv1) vs S1(q1,kv1): n={int((d11!=0).sum())}/{d11.size} maxabs={np.abs(d11).max():.3e}\n")
        # continue the flow from the A post-state (accept or reject as usual)
        restore_all(cache, mtp_cache, snap_Apost)
        fork_done = True
        return t2, l1A, h1A

    for _ in range(MAX_TOKENS):
        if draft is None:
            t = int(mx.argmax(l))
            if t == EOS:
                break
            out.append(t)
            mtp_saved = save_mtp(mtp_cache)
            emb = lm.model.embed_tokens(mx.array([[t]]))
            dh = lm.mtp(h, emb, mtp_cache)
            draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
            prev_token = t
        else:
            d = draft
            draft = None
            verifies += 1
            saved = save_gdn(cache)
            if len(out) == TARGET_IDX - 1 and not fork_done:
                t2, l1, h1 = fork_bisect(d, prev_token)
                fork_lh = (l1, h1)
            else:
                logits, hidden = lm(mx.array([[prev_token, d]]), cache=cache,
                                   return_hidden=True, return_prenorm=True)
                l0, h0 = logits[0, 0], hidden[0, 0]
                l1, h1 = logits[0, 1], hidden[0, 1]
                t2 = int(mx.argmax(l0))
                if t2 == EOS:
                    break
            if t2 == EOS:
                break
            if t2 == d:
                accepted += 1
                out.append(t2)
                l, h = l1, h1
            else:
                restore_gdn(cache, saved)
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)
                restore_mtp(mtp_cache, mtp_saved)
                out.append(t2)
                logits, hidden = lm(mx.array([[prev_token]]), cache=cache,
                                   return_hidden=True, return_prenorm=True)
                logits, hidden = lm(mx.array([[t2]]), cache=cache,
                                   return_hidden=True, return_prenorm=True)
                l, h = logits[0, -1], hidden[0, -1]
            draft = None
    print(f"done: n_out={len(out)} accepted={accepted}/{verifies} "
          f"fork_done={fork_done}")
    print(f"token@{TARGET_IDX}={out[TARGET_IDX] if len(out) > TARGET_IDX else 'n/a'}")
    print(f"first16: {out[:16]}")


if __name__ == "__main__":
    main()
