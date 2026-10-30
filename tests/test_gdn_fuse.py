"""Hermetic Phase 2 gate: the fused GDN decode-step kernel is a verbatim
replication of the deployed 4-op chain's arithmetic. No GPU needed.

These pin the lossless contract: the exact per-phase op sequence (conv 4-tap
sequential sum, the 256-thread rms_norm reduction, the r4d per-thread state
looperational order, the f32-g/low-u16 read, the f2bf rounding, the bf16 pre-rounding
of the qk scales). If any of these strings move, the bit-exactness argument
moves with it and the test must be reviewed in the same PR.
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


def test_conv_phase_reuses_the_4tap_sequential_sum():
    src = _src()
    # gdn_prep's conv body: 4 sequential acc += f32(x)*f32(w), tap 3 = raw input.
    assert src.count("static_cast<float>(conv_state[0 * conv_dim + ") >= 3
    assert src.count("static_cast<float>(conv_state[1 * conv_dim + ") >= 3
    assert src.count("static_cast<float>(conv_state[2 * conv_dim + ") >= 3
    assert src.count("static_cast<float>(conv_weight[cv * kernel_size + 3])") == 1
    assert src.count("static_cast<float>(conv_weight[ck * kernel_size + 3])") == 1
    assert src.count("static_cast<float>(conv_weight[cq * kernel_size + 3])") == 1
    # tap 3 sources: v_raw / k_raw / q_raw, in the chain's channel order
    # (raw inputs are flat [value_dim]/[key_dim]; the head offset is part of
    # the channel index -- a missing offset reads the wrong head's rows).
    assert "static_cast<float>(v_raw[hv * head_dim + tid])" in src
    assert "static_cast<float>(k_raw[hk * head_dim + tid])" in src
    assert "static_cast<float>(q_raw[hk * head_dim + qi])" in src
    # silu via the exact op structs (not a re-derivation)
    assert "Multiply{}(conv_bf, Sigmoid{}(conv_bf))" in src


def test_qk_norm_phase_reuses_the_block_reduction():
    src = _src()
    # The 256-thread reduction, verbatim structure from gdn_prep.hip.
    assert "gdnf_warp_reduce_sum_rms" in src
    assert "__shfl_xor(val, offset)" in src
    assert "shared_sum[warp_id]" in src
    assert "1.0f / sqrtf(shared_sum[0] / axis_size + eps)" in src
    # The qk scale is pre-rounded to bf16 (inv_scale is not exact in bf16) --
    # q uses inv_scale**2, k uses inv_scale.
    assert "float scale = inv_scale * inv_scale;" in src
    assert "float scale = inv_scale;" in src
    assert src.count("static_cast<float>(static_cast<T>(scale))") == 2


def test_state_update_phase_matches_r4d_order():
    src = _src()
    # r4d per-thread looperational order: decay+dot, shfl(1), u=(v-dot)*beta,
    # outer-product+on, shfl(1), f2bf store by half==0 only.
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
    # g is computed f32 but the chain's r4d kernel receives it as BF16
    # (fast.cpp astypes g at the op boundary to q's dtype); replicate by
    # rounding s_g to bf16 before the u16 read.
    assert "gdnf_bf2f16(gdnf_f2bf_rne(s_g))" in src
    assert "gdnf_bf2f_low16f32" not in src
    # the s_g->bf16 cast is RNE, matching MLX's astype (u += (u>>16&1)+0x7FFF).
    assert "u >> 16 & 1u) + 0x7FFFu" in src
    assert "__expf(" in src and "fmaxf(" in src and "-80.0f" in src
    # f2bf round-half-away, not RTNE; the u16->T store is a bit-cast (an
    # integer-to-float static_cast would re-interpret the bits as a value).
    assert "+ 0x8000u) >> 16" in src
    assert "*reinterpret_cast<T*>(&out_bf)" in src


def test_gate_phase_matches_gdn_prep_g_chain():
    src = _src()
    # compute_g chain: bf16(a+dt_bias) -> logaddexp via max/log1p(exp(min-max))
    # -> bf16 round of softplus -> expf(A_log) -> g = expf(-e_al*sp) in f32.
    for s in (
        "T sum_bf = static_cast<T>(static_cast<float>(a[hv]) +",
        "float mxv = fmaxf(fx, 0.0f);",
        "float sp_f = mxv + log1pf(expf(mnv - mxv));",
        "float sp = static_cast<float>(static_cast<T>(sp_f));",
        "float e_al = expf(A_log[hv]);",
        "s_g = expf(arg);",
    ):
        assert s in src


def test_norm_gated_phase_matches_gdn_norm_gated():
    src = _src()
    for s in (
        "float sig = Sigmoid{}(z_f);",
        "float gate = z_f * sig;  // silu in f32",
        "float r = gate * x_f;    // f32 multiply",
        "result[hv * head_dim + i] = static_cast<T>(r);",
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
