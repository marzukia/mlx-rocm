#!/usr/bin/env python3
"""m3: per-layer FULL-STATE DIFF at the S62 inputs (the m2-vs-m0 divergence).

Goal: the first model layer whose state differs between one T=2 verify
forward and two chained T=1 forwards, both started from the identical
pre-verify state (the S62 arm: prev_token=279, draft=3545, n_out==61).

Setup: the m2 draft-verify flow (same prompt/model as the an earlier run S62 diag;
deterministic - it reaches the S62 arm at n_out==61, act=ACC). At the
arm the ENTIRE main-model cache is snapshotted, then the two variants:

  variant A: one T=2 verify forward   lm([[279, 3545]])
  restore:   the snapshot is written back (deep-copied arrays; the
            restore is verified bit-exact against the snapshot)
  variant B: two T=1 forwards        lm([[279]]) then lm([[3545]])

Per-layer state harvested after each variant (every layer, in layer
order): GDN layers -> conv + ssm arrays; attention layers -> K + V page
arrays (sliced to offset) + offset. The diff reports EVERY layer with a
non-zero diff (not stop-at-first): which sub-state, n_diff/total,
maxabs, first index - one table, so the map shows whether it is one
layer or a cascade.

Cache entry types (inspected from mlx_lm/models/cache.py +
qwen3_5.TextModel.make_cache):
  - ArraysCache (GDN/linear layers): .cache=[conv, ssm], .left_padding,
    .lengths
  - KVCache (attention layers): .keys, .values, .offset
The snapshot records every array as f32 numpy + the original dtype
(bf16/f32/int round-trips through f32 are bit-exact). The MTP draft
module's own KV cache (mtp_cache) is not touched by the variants (the
draft step is not re-run) and is therefore not part of the diff.

Also compared: logits after 279 (A[0,0] vs B1[0,0]) and after 3545
(A[0,1] vs B2[0,0]) - i.e. the next t token under each variant.

Exit codes: 0 = arm reached + diff printed; 2 = arm mismatch (evidence
printed; the arm is NOT forced); 3 = unexpected failure.
"""

import os
import pickle
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache

MODEL = os.environ.get("MTP_MODEL", "/mnt/models/qwen3-9b-mlx")
PROMPT = (
    "Explain in great detail how a modern smartphone works, from the "
    "hardware components to the operating system, covering every "
    "subsystem. Be exhaustive."
)
MAX_TOKENS = int(os.environ.get("DIAG_MAX", "160"))
EOS = 248044
MODE = os.environ.get("MODE", "m3")
D = os.environ.get("CAP_DIR", "/mnt/models/tmp/an earlier run")
S62_PREV = 279
S62_DRAFT = 3545
EXPECT_N_OUT = 61


def npf(a):
    """mx array -> f32 numpy (avoids the bf16 PEP 3118 buffer issue)."""
    if a is None:
        return None
    return np.asarray(mx.array(a).astype(mx.float32)).copy()


def cmp_report(name, a, b):
    """Bit-exact comparison of two f32 arrays; prints MATCH / DIFF n maxabs."""
    if a is None or b is None:
        print(f"  {name:<30} SKIP (None)", flush=True)
        return
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        print(f"  {name:<30} SHAPE {a.shape} vs {b.shape}", flush=True)
        return
    diff = int(np.count_nonzero(a != b))
    if diff == 0:
        print(f"  {name:<30} MATCH  (n={a.size})", flush=True)
    else:
        ad = np.abs(a - b)
        fi = int(np.argmax(a != b))
        print(
            f"  {name:<30} DIFF   n={diff}/{a.size} maxabs={float(ad.max()):.3e} first={fi}",
            flush=True,
        )


def cmp_arr(a, b):
    """Bit-exact compare; returns (status, n_diff, total, maxabs, first)."""
    if a is None and b is None:
        return ("NONE", 0, 0, 0.0, -1)
    if a is None or b is None:
        return ("NONE-vs-ARR", 0, 0, 0.0, -1)
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return (f"SHAPE {a.shape} vs {b.shape}", 0, 0, 0.0, -1)
    d = a != b
    n = int(np.count_nonzero(d))
    if n == 0:
        return ("MATCH", 0, a.size, 0.0, -1)
    return ("DIFF", n, a.size, float(np.abs(a - b).max()), int(np.argmax(d)))


# ---------------------------------------------------------------------------
# full-cache snapshot / restore / verify
# ---------------------------------------------------------------------------


def snap_arr(a):
    """Deep-copy one mx array: (f32 numpy, original dtype) or (None, None)."""
    if a is None:
        return (None, None)
    return (npf(a), a.dtype)


def restore_arr(v):
    """(f32 numpy, dtype) -> fresh mx array (a copy; f32 round-trip is exact)."""
    npv, dt = v
    if npv is None:
        return None
    return mx.array(npv).astype(dt)


def snap_cache(cache):
    """Snapshot EVERY cache entry (arrays deep-copied, dtype recorded)."""
    snap = []
    for c in cache:
        if isinstance(c, KVCache):
            snap.append({"kind": "kv", "keys": snap_arr(c.keys), "values": snap_arr(c.values), "offset": c.offset})
        elif isinstance(c, ArraysCache):
            snap.append(
                {
                    "kind": "ac",
                    "cache": [snap_arr(x) for x in c.cache],
                    "left_padding": snap_arr(c.left_padding),
                    "lengths": snap_arr(c.lengths),
                }
            )
        else:
            raise ValueError(f"unknown cache entry type {type(c)!r}")
    return snap


def restore_cache(cache, snap):
    """Write the snapshot back (fresh arrays; the live arrays are replaced)."""
    for c, s in zip(cache, snap, strict=True):
        if s["kind"] == "kv":
            c.keys = restore_arr(s["keys"])
            c.values = restore_arr(s["values"])
            c.offset = s["offset"]
        else:
            c.cache = [restore_arr(x) for x in s["cache"]]
            c.left_padding = restore_arr(s["left_padding"])
            c.lengths = restore_arr(s["lengths"])


def verify_restore(cache, snap):
    """Bit-exact check that the restore returned the snapshot state."""
    nbad = 0
    for i, (c, s) in enumerate(zip(cache, snap, strict=True)):
        if s["kind"] == "kv":
            for nm in ("keys", "values"):
                live = c.keys if nm == "keys" else c.values
                st, n, tot, _, _ = cmp_arr(npf(live), s[nm][0])
                if st not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} {nm}: {st} n={n}/{tot}", flush=True)
            if c.offset != s["offset"]:
                nbad += 1
                print(f"  RESTORE-DIFF entry={i} offset {c.offset} vs {s['offset']}", flush=True)
        else:
            for j, live in enumerate(c.cache):
                st, n, tot, _, _ = cmp_arr(npf(live), s["cache"][j][0])
                if st not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} arr{j}: {st} n={n}/{tot}", flush=True)
            for nm in ("left_padding", "lengths"):
                st, n, tot, _, _ = cmp_arr(npf(getattr(c, nm)), s[nm][0])
                if st not in ("MATCH", "NONE"):
                    nbad += 1
                    print(f"  RESTORE-DIFF entry={i} {nm}: {st} n={n}/{tot}", flush=True)
    return nbad


# ---------------------------------------------------------------------------
# per-layer state harvest + diff
# ---------------------------------------------------------------------------


def harvest(cache):
    """Per-layer state (f32 numpy): gdn -> (conv, ssm); attn -> (k, v, off)."""
    out = []
    for c in cache:
        if isinstance(c, KVCache):
            off = c.offset
            k = c.keys[..., :off, :] if c.keys is not None else None
            v = c.values[..., :off, :] if c.values is not None else None
            out.append(("attn", off, npf(k), npf(v)))
        elif isinstance(c, ArraysCache):
            out.append(("gdn", None, npf(c[0]), npf(c[1])))
        else:
            raise ValueError(f"unknown cache entry type {type(c)!r}")
    return out


def diff_table(ha, hb):
    """One table, every layer in order, every sub-state (not stop-at-first)."""
    print("== m3 per-layer diff: variant A (T=2) vs variant B (2xT=1) ==", flush=True)
    print(f"{'layer':<6} {'type':<5} {'sub':<6} {'n_diff/total':<16} {'maxabs':<12} first_idx", flush=True)
    diverging = []
    for i, (a, b) in enumerate(zip(ha, hb, strict=True)):
        kind = a[0]
        pairs = []
        if kind == "attn":
            if a[1] != b[1]:
                pairs.append(("offset", (a[1], b[1])))
            pairs.append(("k", (a[2], b[2])))
            pairs.append(("v", (a[3], b[3])))
        else:
            pairs.append(("conv", (a[2], b[2])))
            pairs.append(("ssm", (a[3], b[3])))
        for nm, (x, y) in pairs:
            if nm == "offset":
                print(f"{i:<6} {kind:<5} {nm:<6} {f'{x} vs {y}':<16} {'-':<12} -", flush=True)
                diverging.append((i, kind, nm))
                continue
            st, n, tot, mx_, fi = cmp_arr(x, y)
            if st == "MATCH":
                print(f"{i:<6} {kind:<5} {nm:<6} {f'0/{tot}':<16} {'-':<12} -", flush=True)
            elif st == "NONE":
                print(f"{i:<6} {kind:<5} {nm:<6} {'none':<16} {'-':<12} -", flush=True)
            else:
                print(f"{i:<6} {kind:<5} {nm:<6} {f'{n}/{tot}':<16} {mx_:<12.3e} {fi}", flush=True)
                diverging.append((i, kind, nm))
    print("== summary ==", flush=True)
    if not diverging:
        print("ALL LAYERS BIT-EXACT (A == B)", flush=True)
    else:
        first = diverging[0]
        print(f"first diverging layer: {first[0]} ({first[1]}, {first[2]})", flush=True)
        print("diverging: " + ", ".join(f"{i}:{k}:{s}" for i, k, s in diverging), flush=True)
    return diverging


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    if MODE != "m3":
        print(f"this script is m3-only (MODE={MODE})", flush=True)
        sys.exit(3)

    model, tok = load(MODEL)
    lm = model.language_model
    mx.eval(model.parameters())
    mtp_cache = [KVCache()]
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = lm.make_cache()
    lmap = ["attn" if isinstance(c, KVCache) else "gdn" for c in cache]
    print(f"layer map ({len(lmap)} layers): {lmap}", flush=True)

    logits, hidden = lm(
        mx.array([prompt_ids]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    lg = logits[0, -1]
    hidd = hidden[0, -1]
    draft = None
    prev_token = None
    out = []
    armed = False

    for step in range(MAX_TOKENS):
        if draft is None:
            t = int(mx.argmax(lg))
            if t == EOS:
                break
            out.append(t)
            emb = lm.model.embed_tokens(mx.array([[t]]))
            dh = lm.mtp(hidd, emb, mtp_cache)
            draft = int(mx.argmax(lm.lm_head(dh)[0, -1]))
            prev_token = t
            if prev_token == S62_PREV and not armed:
                armed = True
                run_m3(lm, cache, out, prev_token, draft, step)
        else:
            d = draft
            draft = None
            # the m2 REJ rollback (exactly as the an earlier run flow): ArraysCache
            # entries restored, KVCache trimmed by 2 (page layout is NOT
            # rolled back - that is the an earlier run suspect, preserved here).
            saved = [(i, c[0], c[1], c.lengths) for i, c in enumerate(cache) if isinstance(c, ArraysCache)]
            logits, hidden = lm(
                mx.array([[prev_token, d]]),
                cache=cache,
                return_hidden=True,
                return_prenorm=True,
            )
            l0, _h0 = logits[0, 0], hidden[0, 0]
            l1, h1 = logits[0, 1], hidden[0, 1]
            t2 = int(mx.argmax(l0))
            if t2 == EOS:
                break
            if t2 == d:
                out.append(t2)
                lg, hidd = l1, h1
            else:
                for i, conv, ssm, lengths in saved:
                    cache[i][0] = conv
                    cache[i][1] = ssm
                    cache[i].lengths = lengths
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(2)
                out.append(t2)
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
                lg, hidd = logits[0, -1], hidden[0, -1]

    if not armed:
        print(
            f"ARM NOT REACHED: n_out={len(out)} last_out={out[-8:]} "
            f"(flow did not reach prev_token={S62_PREV}; "
            "not forcing the arm)",
            flush=True,
        )
        sys.exit(2)


def run_m3(lm, cache, out, prev_token, draft, step):
    n_out = len(out)
    print(f"ARM m3: step={step} prev={prev_token} draft={draft} n_out={n_out} out[-8:]={out[-8:]}", flush=True)
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            print(f"ARM attn offset layer={i}: {c.offset}", flush=True)
    if n_out != EXPECT_N_OUT:
        print(f"ARM MISMATCH: expected n_out=={EXPECT_N_OUT}, got {n_out}", flush=True)
        sys.exit(2)
    if draft != S62_DRAFT:
        print(f"ARM MISMATCH: expected draft=={S62_DRAFT}, got {draft}", flush=True)
        sys.exit(2)

    # materialize + snapshot the ENTIRE cache
    for c in cache:
        if isinstance(c, KVCache):
            mx.eval([c.keys, c.values])
        else:
            mx.eval([x for x in c.cache if x is not None])
    snap = snap_cache(cache)
    print(f"snapshot taken: {len(snap)} entries", flush=True)

    # ---- variant A: one T=2 verify forward -------------------------------
    print(f"== variant A: T=2 verify forward [[{S62_PREV}, {S62_DRAFT}]] ==", flush=True)
    logits_a, hidden_a = lm(
        mx.array([[S62_PREV, S62_DRAFT]]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    mx.eval([logits_a, hidden_a])
    t2 = int(mx.argmax(logits_a[0, 0]))
    act = "ACC" if t2 == S62_DRAFT else "REJ"
    print(f"variant A act={act} (t2={t2} draft={S62_DRAFT})", flush=True)
    ha = harvest(cache)

    # ---- restore + verify the fork ---------------------------------------
    restore_cache(cache, snap)
    nbad = verify_restore(cache, snap)
    print(f"restore verified: {len(snap) - nbad}/{len(snap)} entries bit-exact", flush=True)
    if nbad:
        print("RESTORE NOT BIT-EXACT - the A/B diff is suspect", flush=True)

    # ---- variant B: two T=1 forwards --------------------------------------
    print(f"== variant B: T=1 forward [[{S62_PREV}]] ==", flush=True)
    logits_b1, hidden_b1 = lm(
        mx.array([[S62_PREV]]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    mx.eval([logits_b1, hidden_b1])
    print(f"== variant B: T=1 forward [[{S62_DRAFT}]] ==", flush=True)
    logits_b2, hidden_b2 = lm(
        mx.array([[S62_DRAFT]]),
        cache=cache,
        return_hidden=True,
        return_prenorm=True,
    )
    mx.eval([logits_b2, hidden_b2])
    hb = harvest(cache)

    # ---- per-layer diff ---------------------------------------------------
    diff_table(ha, hb)

    # ---- next-t-token comparisons -------------------------------------------
    cmp_report("logits after 279  (A[0,0] vs B1[0,0])", npf(logits_a[0, 0]), npf(logits_b1[0, 0]))
    cmp_report("logits after 3545 (A[0,1] vs B2[0,0])", npf(logits_a[0, 1]), npf(logits_b2[0, 0]))

    with open(os.path.join(D, "cap-m3-s62.pkl"), "wb") as f:
        pickle.dump({"ha": ha, "hb": hb, "act": act, "n_out": n_out, "out_tail": out[-8:]}, f)
    print(f"m3 captures saved -> {os.path.join(D, 'cap-m3-s62.pkl')}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
