"""Hermetic Phase 2 gate: the fused GDN decode-step kernel is a verbatim replication of the 4-op chain's arithmetic.
No GPU needed.

These pin the lossless contract: the exact per-phase op sequence (conv 4-tap sequential sum, the 256-thread rms_norm reduction, the r4d state looperational order, the f32-g/low-u16 read, the f2bf rounding, the bf16 pre-rounding of the qk scales). If any of these strings move, the bit-exactness argument moves with it - review the test in the same PR.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIP = ROOT / "patches" / "gdn_fused_step" / "gdn_fused_step.hip"
PATCH = ROOT / "patches" / "gdn_fused_step" / "gdn_fused_step.patch"


def _src() -> str:
    return HIP.read_text()


def test_kernel_file_and_patch_exist():
    assert HIP.exists(), "patches/gdn_fused_step/gdn_fused_step.hip missing"
    assert PATCH.exists(), "patches/gdn_fused_step/gdn_fused_step.patch missing"
    assert "gdn_fused_step.hip" in PATCH.read_text()


def test_env_gate_is_the_only_switch():
    t = PATCH.read_text()
    assert 'getenv("MLX_ROCM_GDN_FUSE")' in t
    assert 'getenv("MLX_ROCM_GDN_FALLBACK")' in t
    # Default path: env unset -> use_fallback true -> 4-op chain.
    assert "e == nullptr || e[0] != '1'" in t
    # T in {1, 2, 4} is the always-fused range; T=3 (S=3, K=2 spec-decode) is admitted only under MLX_ROCM_GDN_T3=1 (the TMAX=3 instantiation is new codegen, default OFF -> T=3 degrades to the clear error, no OOB fall-through to the TMAX=4 kernel); T>4 is rejected.
    assert 'getenv("MLX_ROCM_GDN_T3")' in t
    assert "if (B != 1 || T > 4) {" in t
    assert "if (T == 3) {" in t
    # The eval_gpu comment stays consistent with the admission gate.
    assert "T is 1, 2, 3 or 4" in t
    # The T=3 instantiation exists and is dispatched before the TMAX=4 fall-through (the tmax == 3 branch precedes the final <4>).
    src = _src()
    assert "launch_gdn_fused_step<3>" in src
    assert src.index("(tmax == 3)") < src.index("launch_gdn_fused_step<4>")


def test_conv_phase_reuses_the_4tap_sequential_sum():
    src = _src()
    # Sliding window (E = [cs0, cs1, cs2, raw rows t-2..t]; row t = sum_j E[t+j]*w[j], tap j sources conv_state[j-t] while j-t in [0,2) else raw row t+1-j) keeps gdn_prep's 4-tap k-order sequential f32 sum. Written generalized over t in [0, TMAX): for TMAX=2/1 the loop unrolls to constants and every ternary folds EXACTLY to the original T=2/T=1 straight-line windows (bit-identical codegen).
    # Tap 0/1/2 (v, k, q channels = 3 each):
    assert src.count("(t <= 2) ? conv_state[t * conv_dim + ") == 3
    assert src.count("(t <= 1) ? conv_state[(t + 1) * conv_dim + ") == 3
    assert src.count("(t == 0) ? conv_state[2 * conv_dim + ") == 3
    # Raw inputs read as packed t-indexed rows (the op force-copies the model's row-strided q3/k3/v3 slices for T>1); tap3 = raw row t.
    for raw, dim, ch in (
        ("v_raw", "value_dim", "vch"),
        ("k_raw", "key_dim", "kch"),
        ("q_raw", "key_dim", "qch"),
    ):
        assert f"{raw}[(t - 3) * {dim} + {ch}]" in src
        assert f"{raw}[(t - 2) * {dim} + {ch}]" in src
        assert f"{raw}[(t - 1) * {dim} + {ch}]" in src
        assert f"static_cast<float>({raw}[t * {dim} + {ch}])" in src
    # silu via the exact op structs (not a re-derivation)
    assert "Multiply{}(conv_bf, Sigmoid{}(conv_bf))" in src
    # TMAX is a template constant: T=1 unrolls to the straight-line kernel; T=2 to the T=2 windows;
    # T=4 instantiates the same structure over 4 rows.
    assert "template <typename T, int TMAX>" in src
    assert "for (int t = 0; t < TMAX; ++t)" in src
    assert "launch_gdn_fused_step<1>" in src
    assert "launch_gdn_fused_step<2>" in src
    assert "launch_gdn_fused_step<3>" in src
    assert "launch_gdn_fused_step<4>" in src
    # s_beta/s_g shared buffers are sized for the T=4 row count.
    assert "__shared__ T s_beta[4];" in src
    assert "__shared__ float s_g[4];" in src


def test_qk_norm_phase_reuses_the_block_reduction():
    src = _src()
    # The 256-thread reduction, verbatim structure from gdn_prep.hip.
    assert "gdnf_warp_reduce_sum_rms" in src
    assert "__shfl_xor(val, offset)" in src
    assert "shared_sum[warp_id]" in src
    assert "1.0f / sqrtf(shared_sum[0] / axis_size + eps)" in src
    # The qk scale is pre-rounded to bf16 (inv_scale is not exact in bf16) -- q uses inv_scale**2, k uses inv_scale.
    assert "float scale = inv_scale * inv_scale;" in src
    assert "float scale = inv_scale;" in src
    assert src.count("static_cast<float>(static_cast<T>(scale))") == 2


def test_state_update_phase_matches_r4d_order():
    src = _src()
    # r4d per-thread looperational order: decay+dot, shfl(1), u=(v-dot)*beta, outer-product+on, shfl(1), f2bf store by half==0 only.
    m = re.search(
        r"for \(int j = 0; j < 64; \+\+j\) \{\s*h\[j\] \*= eg;\s*dot \+= "
        r"h\[j\] \* kk\[j\];\s*\}\s*dot \+= __shfl_xor\(dot, 1, 32\);",
        src,
    )
    assert m, "r4d decay/dot phase order moved"
    m = re.search(
        r"for \(int j = 0; j < 64; \+\+j\) \{\s*h\[j\] \+= u \* kk\[j\];\s*on "
        r"\+= h\[j\] \* qq\[j\];\s*\}\s*on \+= __shfl_xor\(on, 1, 32\);",
        src,
    )
    assert m, "r4d outer-product/on phase order moved"
    assert "if (half == 0)" in src
    # g is computed f32 but the chain's r4d kernel receives it as BF16 (fast.cpp astypes g at the op boundary to q's dtype); replicate by rounding s_g to bf16 before the u16 read.
    assert "gdnf_bf2f16(gdnf_f2bf_rne(s_g[t]))" in src
    assert "gdnf_bf2f_low16f32" not in src
    # the s_g->bf16 cast is RNE, matching MLX's astype (u += (u>>16&1)+0x7FFF).
    assert "u >> 16 & 1u) + 0x7FFFu" in src
    assert "__expf(" in src and "fmaxf(" in src and "-80.0f" in src
    # f2bf round-half-away, not RTNE; the u16->T store is a bit-cast (an integer-to-float static_cast would re-interpret the bits as a value).
    assert "+ 0x8000u) >> 16" in src
    assert "*reinterpret_cast<T*>(&out_bf)" in src


def test_gate_phase_matches_gdn_prep_g_chain():
    src = _src()
    # compute_g chain: bf16(a+dt_bias) -> logaddexp via max/log1p(exp(min-max)) -> bf16 round of softplus -> expf(A_log) -> g = expf(-e_al*sp) in f32.
    for s in (
        "T sum_bf = static_cast<T>(static_cast<float>(a[t * num_v_heads + hv]) +",
        "float mxv = fmaxf(fx, 0.0f);",
        "float sp_f = mxv + log1pf(expf(mnv - mxv));",
        "float sp = static_cast<float>(static_cast<T>(sp_f));",
        "float e_al = expf(A_log[hv]);",
        "s_g[t] = expf(arg);",
    ):
        assert s in src


def test_norm_gated_phase_matches_gdn_norm_gated():
    src = _src()
    for s in (
        "float sig = Sigmoid{}(z_f);",
        "float gate = z_f * sig;  // silu in f32",
        "float r = gate * x_f;    // f32 multiply",
        "result[(t * num_v_heads + hv) * head_dim + i] = static_cast<T>(r);",
    ):
        assert s in src


def test_one_block_per_v_head_one_launch():
    src = _src()
    assert "const int hv = blockIdx.x;" in src
    assert "dim3 grid(num_v_heads);" in src
    assert "dim3 block(256);" in src
    # No second launch inside eval_gpu.
    assert src.count("encoder.launch_kernel") == 1


def test_patch_touches_only_the_expected_files():
    t = PATCH.read_text()
    files = set(re.findall(r"^diff --git a/(\S+) b/", t, re.M))
    expected = {
        "mlx/fast.h",
        "mlx/fast.cpp",
        "mlx/fast_primitives.h",
        "python/src/fast.cpp",
        "mlx/backend/rocm/CMakeLists.txt",
        "mlx/backend/rocm/gdn_fused_step.hip",
    }
    assert files == expected, f"unexpected patch file set: {files}"


def test_patch_hunk_counts_are_wellformed():
    # A hunk whose claimed line counts do not match its body makes the patch unapplicable (git apply: "corrupt Patch", GNU patch: "malformed patch"); the base patch of record was well-formed, so every hunk in every .patch/.diff under patches/ must be.
    for patch in sorted(ROOT.glob("patches/**/*")):
        if patch.suffix not in (".patch", ".diff"):
            continue
        text = patch.read_text().split("\n")
        i = 0
        while i < len(text):
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", text[i])
            if not m:
                i += 1
                continue
            old_n = int(m.group(2) or 1)
            new_n = int(m.group(4) or 1)
            j = i + 1
            n_old = n_new = 0
            while j < len(text):
                ln = text[j]
                if (
                    ln.startswith("@@ ")
                    or ln.startswith("--- ")
                    or ln.startswith("+++ ")
                    or ln.startswith("diff --git")
                ):
                    break
                if ln.startswith("+"):
                    n_new += 1
                elif ln.startswith("-"):
                    n_old += 1
                elif ln.startswith(" "):
                    n_old += 1
                    n_new += 1
                j += 1
            assert n_old == old_n, (
                f"{patch.name}: hunk at line {i + 1} ({m.group(0)}): {n_old} old lines != {old_n} claimed"
            )
            assert n_new == new_n, (
                f"{patch.name}: hunk at line {i + 1} ({m.group(0)}): {n_new} new lines != {new_n} claimed"
            )
            i = j
