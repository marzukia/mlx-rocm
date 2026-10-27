// Copyright © 2024 Apple Inc.
//
// ROCm GatedDeltaUpdate: fused gated-delta-net kernels ported from R4D
// (libr4d e8de4bc / vllm-radiance 1d2e248; see reports/k3-gdn-port.md).
// Decode (T == 1) runs the recurrent state update; longer sequences run the
// chunked scan (gate prep + kkt_solve + chunk_scan, with the extreme-decay
// repair inside the scan entry). Geometry outside the compiled kernels
// (Dk/Dv != 128, Hv % Hk != 0, ratio > 8, masks, non-bf16 output, vector
// gates) falls back to the per-token op-loop graph. Non-row-contiguous
// inputs (e.g. v as a raw split+reshape view in the GDN hybrid model) get
// a row-contiguous copy before the fused kernels.

#include <cstdlib>
#include <sstream>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

// Defined in r4d_gdn_kkt_solve_k128_c64_bf16.hip
extern "C" int r4d_gdn_kkt_solve_k128_c64_bf16(
    const void* k,
    const void* beta,
    const void* g,
    void* A,
    int B,
    int T,
    int H,
    int Hg,
    int K,
    int bt,
    void* stream);

// Defined in r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip
extern "C" int r4d_gdn_gate_prep_c64_bf16(
    const void* g,
    const void* beta,
    void* g_cum,
    void* beta_f32,
    int B,
    int T,
    int H,
    void* stream);
extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* A,
    const void* g,
    const void* beta,
    const void* h0,
    void* o,
    void* ht,
    int B,
    int T,
    int H,
    int Hg,
    int K,
    int V,
    int bt,
    float scale,
    void* stream);

// Defined in r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip
extern "C" int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* h0,
    void* o,
    float* ht,
    int B,
    int T,
    int H,
    int Hg,
    int K,
    int V,
    void* stream);

namespace {

// MLX_ROCM_GDN_FALLBACK=1 forces the per-token op-loop graph (A/B testing).
bool env_forced_fallback() {
  static const bool forced = [] {
    const char* e = std::getenv("MLX_ROCM_GDN_FALLBACK");
    return e != nullptr && e[0] == '1';
  }();
  return forced;
}

// The fused kernels read raw buffer + offset (no strides): they require
// row-contiguous inputs. In the GDN hybrid model v is a raw split+reshape
// view (a slice of a wider row); q/k pass through rms_norm (fresh
// contiguous buffers). Make a row-contiguous copy only for inputs that are
// NOT row-contiguous; already-contiguous inputs pass through unchanged
// (zero behavior change).
array make_row_contiguous(const array& a, const Stream& s) {
  if (a.flags().row_contiguous) {
    return a;
  }
  array out(a.shape(), a.dtype(), nullptr, {});
  copy_gpu(a, out, CopyType::General, s);
  return out;
}

}  // namespace

bool GatedDeltaUpdate::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    const bool has_mask,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (env_forced_fallback()) {
    return true;
  }
  if (has_mask) {
    return true;
  }
  // Compiled geometry: head dims 128/128, chunk 64.
  if (Dk != 128 || Dv != 128) {
    return true;
  }
  if (Hv % Hk != 0) {
    return true;
  }
  // The kkt_solve preamble is templated up to 8 v heads per k head.
  if (Hv / Hk > 8) {
    return true;
  }
  return false;
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& g = inputs[3];
  auto& beta = inputs[4];
  auto& h0 = inputs[5];
  auto& out = outputs[0];
  auto& state = outputs[1];

  const int B = q.shape(0);
  const int T = q.shape(1);
  const int Hk = q.shape(2);
  const int Dk = q.shape(3);
  const int Hv = v.shape(2);
  const int Dv = v.shape(3);
  if (std::getenv("MLX_GDN_DEBUG")) {
    for (size_t i = 0; i < inputs.size(); i++) {
      fprintf(stderr, "GDN-DBG in[%zu] status=%d dptr=%p nbytes=%zu\n",
              i, (int)inputs[i].status(),
              (void*)inputs[i].data_shared_ptr().get(),
              inputs[i].nbytes());
    }
  }

  // The fused kernels are compiled for bf16 q/k/v/g/beta, fp32 state, scalar
  // gates and no mask. Anything else runs the op-loop fallback the graph
  // would have used.
  const bool scalar_gates = (g.ndim() == 3 && g.shape(2) == Hv);
  const bool contiguous =
      q.flags().row_contiguous && k.flags().row_contiguous &&
      v.flags().row_contiguous && g.flags().row_contiguous &&
      beta.flags().row_contiguous && h0.flags().row_contiguous;
  if (std::getenv("MLX_GDN_DEBUG")) {
    fprintf(stderr, "GDN-DBG flags q=%d k=%d v=%d g=%d beta=%d h0=%d\n",
            (int)q.flags().row_contiguous, (int)k.flags().row_contiguous,
            (int)v.flags().row_contiguous, (int)g.flags().row_contiguous,
            (int)beta.flags().row_contiguous, (int)h0.flags().row_contiguous);
    fprintf(stderr, "GDN-DBG cond qdt=%d scalar=%d cont=%d T=%d B=%d\n",
            q.dtype() == bfloat16 ? 1 : 0, (int)scalar_gates, (int)contiguous, T, B);
  }
  if (q.dtype() != bfloat16 || !scalar_gates) {
    if (std::getenv("MLX_GDN_DEBUG")) fprintf(stderr, "GDN-DBG FALLBACK\n");
    auto r = fallback_(inputs);
    outputs[0] = r[0];
    outputs[1] = r[1];
    return;
  }

  // Row-contiguous copies for any non-row-contiguous input (v in the GDN
  // hybrid model); already-contiguous inputs pass through unchanged.
  array qc = make_row_contiguous(q, s);
  array kc = make_row_contiguous(k, s);
  array vc = make_row_contiguous(v, s);
  array gc = make_row_contiguous(g, s);
  array bc = make_row_contiguous(beta, s);
  array hc = make_row_contiguous(h0, s);

  encoder.set_input_array(qc);
  encoder.set_input_array(kc);
  encoder.set_input_array(vc);
  encoder.set_input_array(gc);
  encoder.set_input_array(bc);
  encoder.set_input_array(hc);
  if (std::getenv("MLX_GDN_DEBUG")) fprintf(stderr, "GDN-DBG pre-out-set\n");
  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  if (std::getenv("MLX_GDN_DEBUG")) fprintf(stderr, "GDN-DBG post-out-set\n");
  state.set_data(mlx::core::rocm::malloc_async(state.nbytes(), encoder));
  if (std::getenv("MLX_GDN_DEBUG")) fprintf(stderr, "GDN-DBG post-state-set\n");
  if (std::getenv("MLX_GDN_DEBUG")) {
    fprintf(stderr,
            "GDN-DBG post-set out.id=%p out.data=%p out.buf=%p "
            "state.id=%p state.data=%p state.buf=%p\n",
            (void*)out.id(),
            (void*)out.data_shared_ptr().get(),
            out.data_shared_ptr() ? (void*)out.buffer().ptr() : nullptr,
            (void*)state.id(),
            (void*)state.data_shared_ptr().get(),
            state.data_shared_ptr() ? (void*)state.buffer().ptr()
                : nullptr);
  }

  // Small T (decode + spec-decode verify, T <= 8) runs the recurrent
  // kernel: it loops sequentially over T, so T=2 is bit-identical to two
  // T=1 steps. The chunked scan below is a different reduction order
  // (not bit-identical) and carries chunk machinery that is slower than a
  // few recurrent steps for small T. T > 8 (prefill) keeps the scan.
  if (T <= 8) {
    // Decode: one recurrent state update per token. The recurrent kernel
    // reads the multiplicative gates directly, so no temp buffers are needed.
    encoder.launch_kernel([&](hipStream_t stream) {
      if (std::getenv("MLX_GDN_DEBUG")) {
        fprintf(stderr,
                "GDN-DBG lambda out.data=%p out.buf=%p state.data=%p "
                "state.buf=%p\n",
                (void*)out.data_shared_ptr().get(),
                out.data_shared_ptr() ? (void*)out.buffer().ptr() : nullptr,
                (void*)state.data_shared_ptr().get(),
                state.data_shared_ptr() ? (void*)state.buffer().ptr()
                    : nullptr);
      }
      const int rc = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
          gpu_ptr<unsigned short>(qc),
          gpu_ptr<unsigned short>(kc),
          gpu_ptr<unsigned short>(vc),
          gpu_ptr<unsigned short>(gc),
          gpu_ptr<unsigned short>(bc),
          gpu_ptr<float>(hc),
          gpu_ptr<unsigned short>(out),
          gpu_ptr<float>(state),
          B,
          T,
          Hv,
          Hk,
          Dk,
          Dv,
          stream);
      if (rc != 0) {
        std::ostringstream msg;
        msg << "GatedDeltaUpdate: recurrent update launch failed (" << rc
            << ").";
        throw std::runtime_error(msg.str());
      }
    });
  } else {
    // Prefill: gate prep -> kkt_solve -> chunk_scan (the scan entry also
    // launches the extreme-decay repair).
    array g_cum({B * T, Hv}, float32, nullptr, {});
    array beta_f32({B * T, Hv}, float32, nullptr, {});
    g_cum.set_data(mlx::core::rocm::malloc_async(g_cum.nbytes(), encoder));
    beta_f32.set_data(mlx::core::rocm::malloc_async(beta_f32.nbytes(), encoder));
    encoder.add_temporary(g_cum);
    encoder.add_temporary(beta_f32);
    array A({B * T, Hv, 64}, bfloat16, nullptr, {});
    A.set_data(mlx::core::rocm::malloc_async(A.nbytes(), encoder));
    encoder.add_temporary(A);

    encoder.launch_kernel([&](hipStream_t stream) {
      int rc = r4d_gdn_gate_prep_c64_bf16(
          gpu_ptr<unsigned short>(gc),
          gpu_ptr<unsigned short>(bc),
          gpu_ptr<float>(g_cum),
          gpu_ptr<float>(beta_f32),
          B,
          T,
          Hv,
          stream);
      if (rc != 0) {
        std::ostringstream msg;
        msg << "GatedDeltaUpdate: gate prep launch failed (" << rc << ").";
        throw std::runtime_error(msg.str());
      }
      rc = r4d_gdn_kkt_solve_k128_c64_bf16(
          gpu_ptr<unsigned short>(kc),
          gpu_ptr<float>(beta_f32),
          gpu_ptr<float>(g_cum),
          gpu_ptr<unsigned short>(A),
          B,
          T,
          Hv,
          Hk,
          Dk,
          64,
          stream);
      if (rc != 0) {
        std::ostringstream msg;
        msg << "GatedDeltaUpdate: kkt solve launch failed (" << rc << ").";
        throw std::runtime_error(msg.str());
      }
      rc = r4d_gdn_chunk_scan_k128_v128_c64_bf16(
          gpu_ptr<unsigned short>(qc),
          gpu_ptr<unsigned short>(kc),
          gpu_ptr<unsigned short>(vc),
          gpu_ptr<unsigned short>(A),
          gpu_ptr<float>(g_cum),
          gpu_ptr<float>(beta_f32),
          gpu_ptr<float>(hc),
          gpu_ptr<unsigned short>(out),
          gpu_ptr<float>(state),
          B,
          T,
          Hv,
          Hk,
          Dk,
          Dv,
          64,
          1.0f,
          stream);
      if (rc != 0) {
        std::ostringstream msg;
        msg << "GatedDeltaUpdate: chunk scan launch failed (" << rc << ").";
        throw std::runtime_error(msg.str());
      }
    });
  }
}

}  // namespace mlx::core::fast
