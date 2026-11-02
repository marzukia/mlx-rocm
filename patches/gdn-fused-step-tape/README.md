# gdn-fused-step-tape — S=4 GDN step kernel + conv-window row fix

This directory holds the S=4 tape graft and its verification. The tape kernel
itself (`gdn_fused_step_tape.hip`, pre-fix) ships in
`patches/rocm-vendor-delta/new_files/` (it is part of the cumulative vendor
delta that reproduces the v2 wheel tree). This directory carries:

- **`conv-row-fix.diff`** — the rows 0/1 bit-exactness fix (Job 1 of this
  branch).
- **`state-None-guard.diff`** — the fresh-cache crash guard for the
  `qwen3_5.py` model file (Job 2 of this branch; applies to the vendored
  mlx-lm `mlx_lm/models/qwen3_5.py` in the guest venv).

## The rows 0/1 bug (root cause)

The tape kernel emits a per-row conv_state tape (`last_tape`), one
3-row window per output row. `gdnt_cs_tap()` selects the conv-window taps:
taps idx 0–2 read the pre-window conv_state rows, taps idx 3+ read raw
in-window rows. The window math is:

```
row t window = [cs0, cs1, cs2, raw0 .. raw3]   (rawi = row t+i, clamped)
```

The stock `gdnt_cs_tap` took **one** channel argument for both the cs and the
raw index:

```cpp
template <typename T>
__device__ __forceinline__ T gdnt_cs_tap(
    const T* cs, const T* raw, int raw_dim, int ch, int idx, int conv_dim) {
  return (idx <= 2) ? cs[idx * conv_dim + ch]
                    : raw[(idx - 3) * raw_dim + ch];
}
```

The q branch (conv channel == raw channel) is correct under this. The k and v
branches pass the **raw-side** channel offset (`kch = k`, `vch = key_dim + k`)
while the conv_state is laid out in **conv channels** (k block offset 0, v
block offset `2*key_dim`), i.e. the conv-side offsets are `ck = 2*key_dim + k`
and `cv = 4*key_dim + key_dim + k`. The m1005 rowprobe signature: rows 0/1
diverge in exactly the k+v channels (raw 2048–8191 of the 8192-wide conv
state) on cs-tap reads, while rows 2/3 are bit-exact (their per-row windows'
cs taps do not read the mis-indexed k/v conv blocks).

## The fix

Separate the two channel bases:

```cpp
template <typename T>
__device__ __forceinline__ T gdnt_cs_tap(
    const T* cs, const T* raw, int raw_dim,
    int raw_ch, int cs_ch, int idx, int conv_dim) {
  return (idx <= 2) ? cs[idx * conv_dim + cs_ch]
                    : raw[(idx - 3) * raw_dim + raw_ch];
}
```

and pass `(kch, ck)`, `(vch, cv)`, `(qch, cq)` from the three branches
(`conv_channel = raw_channel + block_conv_offset`).

## Verification (GPU, guest-a / RX 9070 XT / gfx1201)

`rowprobe` (prefill 16, then 1×S=4 tape vs 4×S=1 fused, per-row conv md5):

| row | pre-fix (m1005) | post-fix (this branch) |
|---|---|---|
| 0 | differ (k+v channels) | **md5-identical** |
| 1 | differ (k+v channels) | **md5-identical** |
| 2 | md5-identical | md5-identical |
| 3 | md5-identical | md5-identical |

Full-state parity (1×S=4 tape vs 4×S=1 fused, 4 tokens after prefill 16):
conv state and recurrent state md5-identical on all four rows. The end-to-end
logits of an S=4 step vs four S=1 steps on the *no-graft* wheel still differ
beyond ulp — **control-proven to be the attention (prefill kernel) path, not
the GDN ops** (GDN-ops-OFF control shows the same-magnitude diff; see
REPORT.md Job 1). With the attention graft engaged (env ON, wheel v4) the K=3
stream bit-matches AR (battery arms A1/A3).

## The fresh-cache state-None crash (Job 2)

Both op pybinds (`gdn_fused_step`, `gdn_fused_step_tape`) declare `state` as a
non-optional `array` argument. On the first S<=4 forward of a fresh cache
(`cache[1]` is `None`) the patched `qwen3_5.py` branches selected the op and
the pybind conversion raised. The guard (`state-None-guard.diff`) computes
`_state = cache[1] if cache is not None else None` once and adds
`and _state is not None` to **both** branch conditions, so a fresh-cache
S<=4 forward falls to the default 4-op chain (which handles a `None` state).
Verified: fresh-cache S=4 4-token first forward and fresh-cache S=1 first
forward both complete without the guard error (GATE2A/GATE2B, REPORT.md Job 2).
