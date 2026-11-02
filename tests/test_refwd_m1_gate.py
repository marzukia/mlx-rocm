"""Hermetic gate test for the REFWD-M1 toggle (MLX_ROCM_REFWD_M1). No GPU needed.

The toggle re-routes ONLY the M=1 (decode / correction re-forward) QMMs to the
existing int8-dot4 LX path (an already-compiled, already-correct kernel) so the
next GPU round can A/B the M=1 scalar-FMA wide default vs the integer-dot
path. This test pins the routing decision (a faithful mirror of the gate logic
in qmm.hip) and the shape-validity of the re-route over all 249 refwd GEMMs.

It does NOT run the model or the GPU; it validates the dispatch gate — the
only thing the change touches. The correctness of the LX kernel itself is
inherited (it is the pre-existing MLX_QMV_LX path).
"""

# qwen3-9b-mlx config (from the pinned model config.json, text_config).
HIDDEN = 4096
INTER = 12288
LAYERS = 32
N_GDN = 24
N_ATTN = 8
VH, KVH = 16, 4
HEAD_DIM = 256
V_HEADS, K_HEADS = 32, 16
K_HEAD_DIM, V_HEAD_DIM = 128, 128
VOCAB = 248320
GROUP = 64  # 4-bit group 64
BITS = 4


def key_dim():
    return K_HEADS * K_HEAD_DIM


def value_dim():
    return V_HEADS * V_HEAD_DIM


def refwd_shapes():
    """Enumerate the (name, N, K) of every quantized GEMM in ONE M=1 (refwd)
    forward of the 9B target model. The GDN/attention interleaving is
    irrelevant to the gate — only N/K matter, so the layer ordering is
    nominal."""
    shapes = []
    kd, vd = key_dim(), value_dim()
    for i in range(LAYERS):
        shapes += [
            (f"l{i}/mlp/gate", INTER, HIDDEN),
            (f"l{i}/mlp/up", INTER, HIDDEN),
            (f"l{i}/mlp/down", HIDDEN, INTER),
        ]
        if i < N_GDN:  # 24 GDN layers
            shapes += [
                (f"l{i}/gdn/in_proj_qkv", 2 * kd + vd, HIDDEN),
                (f"l{i}/gdn/in_proj_z", vd, HIDDEN),
                (f"l{i}/gdn/in_proj_b", V_HEADS, HIDDEN),
                (f"l{i}/gdn/in_proj_a", V_HEADS, HIDDEN),
                (f"l{i}/gdn/out_proj", HIDDEN, vd),
            ]
        else:  # 8 full-attention layers
            shapes += [
                (f"l{i}/attn/q", VH * HEAD_DIM, HIDDEN),
                (f"l{i}/attn/k", KVH * HEAD_DIM, HIDDEN),
                (f"l{i}/attn/v", KVH * HEAD_DIM, HIDDEN),
                (f"l{i}/attn/o", HIDDEN, VH * HEAD_DIM),
            ]
    shapes.append(("lm_head", VOCAB, HIDDEN))
    return shapes


def _lx_gate(m, k, group, lx_env_set, refwd_m1_set):
    """Mirror of the (modified) LX gate in qmm.hip. The C++ reads
    `lx_env[0]=='1' && lx_env[1]=='6'` on a null-terminated string, so a
    1-char value is safe there; the shape-validity terms are the rest."""
    lx_engage = lx_env_set or (refwd_m1_set and m == 1)
    return (
        lx_engage and (BITS in (4, 8)) and 1 <= m <= 8 and (k % 32 == 0) and (k % group == 0) and group in (32, 64, 128)
    )


def _wide_gate(m, k, group, no_wide=False):
    """Mirror of the wide (default-on) gate in qmm.hip (4-bit affine bf16,
    non-batched, transpose, host_warp=32)."""
    return (not no_wide) and (BITS in (4, 8)) and 1 <= m <= 8 and (k % 32 == 0) and group in (32, 64, 128)


def _dispatch_path(m, k, group, lx_env_set, refwd_m1_set, no_wide=False):
    """Mirror of the dispatch order (6-bit -> LX -> wide -> ...) restricted to
    the 4-bit affine bf16 non-batched transpose decode case. 6-bit is n/a
    (BITS=4), so the decision is LX-then-wide."""
    if _lx_gate(m, k, group, lx_env_set, refwd_m1_set):
        return "lx"
    if _wide_gate(m, k, group, no_wide=no_wide):
        return "wide_mr" if m <= 4 else "wide"
    return "tiled_or_warp"


# --- the 249-shape enumeration ---------------------------------------------


def test_refwd_shape_count_is_249():
    # the task's "M=1 qmv over all 249 quantized GEMMs": 32*3 (MLP) +
    # 24*5 (GDN) + 8*4 (attn) + 1 (lm_head) = 96 + 120 + 32 + 1 = 249.
    assert len(refwd_shapes()) == 249


def test_refwd_shapes_are_lx_shape_valid():
    # the LX re-route gate requires (K % 32 == 0) and (K % group == 0);
    # every refwd shape must satisfy it so the re-route engages uniformly.
    for name, _n, k in refwd_shapes():
        assert k % 32 == 0, f"{name}: K={k} % 32 != 0"
        assert k % GROUP == 0, f"{name}: K={k} % {GROUP} != 0"


# --- the toggle decision ---------------------------------------------------


def test_default_off_is_a_noop():
    # with MLX_ROCM_REFWD_M1 unset (and MLX_QMV_LX unset), the gate reduces to
    # the original (lx_env != nullptr): M=1 and M=4 both fall through to the
    # wide default. Zero behavior change.
    for m in (1, 2, 4):
        assert _dispatch_path(m, 4096, GROUP, lx_env_set=False, refwd_m1_set=False) == ("wide_mr" if m <= 4 else "wide")


def test_refwd_m1_reroutes_only_m1_to_lx():
    # MLX_ROCM_REFWD_M1 set: M=1 -> the LX path (re-route), M=2 and M=4 stay
    # on the wide default (the M=4 verify / rows path is untouched).
    assert _dispatch_path(1, 4096, GROUP, lx_env_set=False, refwd_m1_set=True) == "lx"
    assert _dispatch_path(2, 4096, GROUP, lx_env_set=False, refwd_m1_set=True) == "wide_mr"
    assert _dispatch_path(4, 4096, GROUP, lx_env_set=False, refwd_m1_set=True) == "wide_mr"


def test_refwd_m1_engages_on_all_refwd_shapes():
    # the re-route must engage (M=1, refwd_m1 set) for EVERY refwd shape.
    for name, n, k in refwd_shapes():
        assert _dispatch_path(1, k, GROUP, lx_env_set=False, refwd_m1_set=True) == "lx", (
            f"{name} (N={n}, K={k}) did not re-route to LX"
        )


def test_refwd_m1_does_not_touch_m4_verify():
    # the M=4 verify shape (K=4096 MLP) stays on wide_mr with the toggle set —
    # the rows / verify win is protected.
    assert _dispatch_path(4, 4096, GROUP, lx_env_set=False, refwd_m1_set=True) == "wide_mr"


def test_lx_cols_guard_is_backward_compatible():
    # the C++ cols read is `lx_env[0]=='1' && lx_env[1]=='6'` on a
    # null-terminated string (a 1-char value is safe there). Mirror it with a
    # length guard so the Python model matches: "16" -> 16, anything else
    # (incl. the refwd_m1 engagement where lx_env is null) -> 32.
    def cols(lx_env):
        c = 32
        if lx_env is not None and len(lx_env) >= 2 and lx_env[0] == "1" and lx_env[1] == "6":
            c = 16
        return c

    assert cols("16") == 16
    assert cols("1") == 32
    assert cols(None) == 32  # the refwd_m1 engagement (lx_env null) -> 32
