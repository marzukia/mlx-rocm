// Ported fattn kernel — the attention port (v3 card 35ce5495 sec 4).
//
// Ports the RDNA WMMA mma-f16 path of ggml flash-attn (llama.cpp b12062 @
// 29c720eaf: fattn-mma-f16.cuh + fattn-common.cuh + the fattn.cu dispatch
// gate) onto mlx-rocm, on the production instance <256,256,16,4>
// (ncols1=16 query rows/block, ncols2=4 GQA heads, D=256 one pass).
//
// v1 scope:
//   - bf16 in / bf16 out; rocwmma 16x16x16 bf16->f32 (the arm-A idiom; the
//     ref's f16 Q-fragment-load NaN class is bypassed, stage 0 verifies)
//   - causal on-the-fly (card default (a)): visibility in registers
//     (kv_global > q_global -> -inf); no mask tensor, no KV_max scan
//   - 1 block per (seq, kv-head, q-tile): grid = tile count, so the stream-K
//     fixup path is provably unneeded in v1 (no dst_meta, no fixup kernel);
//     stream-K load balancing = v2 lever
//   - no LSE (Q-C4 deferred), no alibi / softcap / sinks / sparse
//
// Isolation (stage 1): env MLX_SDPA_FATTN=1 routes prefill SDPA here; unset
// leaves dispatch bit-identical to the pre-port wheel.
//
// Device code: flash_attention_fattn_device.cuh (same PR).

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace mlx::core {
namespace rocm {

// Element strides for the (B, H, L, D) mlx layout: sb = batch, sh = head,
// sl = row (last dim is contiguous by the dispatch's prepare_sdpa_input).
struct FaFftnParams {
  int B;
  int KVH;  // kv heads (= 4, v1)
  int H;    // q heads (= 4*KVH)
  int D;    // head dim (= 256, v1)
  int qL;
  int kL;
  int kv_offset;  // kL - qL: absolute kv row of q row 0
  int iter_j;     // ceil(qL / ncols1)
  float scale;
  int64_t Q_sb, Q_sh, Q_sl;
  int64_t K_sb, K_sh, K_sl;
  int64_t V_sb, V_sh, V_sl;
  int64_t O_sb, O_sh, O_sl;
};

} // namespace rocm
} // namespace mlx::core

#include "flash_attention_fattn_device.cuh"

namespace mlx::core {

namespace {
constexpr int FATTN_D = 256;
constexpr int FATTN_KVH = 4;   // v1 instantiates GQA4 only (kv_heads = 4)
constexpr int FATTN_NC1 = 16;  // query rows per block (ncols1)
constexpr int FATTN_NC2 = 4;   // gqa heads per block (ncols2)
constexpr int FATTN_NC = FATTN_NC1 * FATTN_NC2;
constexpr int FATTN_NTHREADS = 64;  // ref config (256,256,16): 64 threads
// LDS: Q (2 d-halves, [64][128 bf16]) + K + V ([32][256 bf16] each),
// 65,536 B = EXACTLY the gfx1201 per-block cap (verified 2026-11-18 via
// hipGetDeviceProperties, RX 9070 XT; dispatch gate is <=). Zero-pad layout;
// fallback if exactly-max launch fails = Q-in-regs (32,768 B).
constexpr int FATTN_LDS =
    (2 * FATTN_NC * (FATTN_D / 2) + 2 * 32 * FATTN_D) * 2;  // 65,536 B
} // namespace

// v1 gate: D=256 bf16, gqa_ratio exactly 4 (kv_heads=4), no arr-mask, no LSE,
// prefill kvL >= qL. Stride alignment is structural for the mlx layout
// (bf16, D=256 -> 512 B rows, 16 B aligned); the ref's use_gqa_opt 16 B
// stride check holds for all in-model + stage-0 shapes.
bool supports_sdpa_flash_fattn(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool output_logsumexp) {
  if (has_arr_mask) return false;
  if (output_logsumexp) return false;
  if (q.dtype() != bfloat16 || k.dtype() != bfloat16 || v.dtype() != bfloat16)
    return false;
  const int D = q.shape(-1);
  if (D != FATTN_D || D != k.shape(-1) || D != v.shape(-1)) return false;
  if (k.shape(1) != FATTN_KVH) return false;
  if (q.shape(1) != 4 * k.shape(1)) return false;  // gqa_ratio == 4
  if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0)) return false;
  if (k.shape(2) < q.shape(2)) return false;       // kvL >= qL (prefill)
  return true;
}

// LDS bytes the ported kernel needs for a given head dim (v1: D=256 only).
int sdpa_flash_fattn_smem(int D) {
  return D == FATTN_D ? FATTN_LDS : 0;
}

void sdpa_flash_fattn(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    Stream s,
    array* lse /* v1: must be null */) {
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int KVH = k.shape(1);
  const int qL = q.shape(2);
  const int kL = k.shape(2);
  const int D = q.shape(3);

  o.set_data(rocm::malloc_async(o.nbytes(), enc));
  (void)lse;

  rocm::FaFftnParams p;
  p.B = B;
  p.KVH = KVH;
  p.H = H;
  p.D = D;
  p.qL = qL;
  p.kL = kL;
  p.kv_offset = kL - qL;
  p.iter_j = (qL + FATTN_NC1 - 1) / FATTN_NC1;
  p.scale = scale;
  p.Q_sb = q.strides(0);
  p.Q_sh = q.strides(1);
  p.Q_sl = q.strides(2);
  p.K_sb = k.strides(0);
  p.K_sh = k.strides(1);
  p.K_sl = k.strides(2);
  p.V_sb = v.strides(0);
  p.V_sh = v.strides(1);
  p.V_sl = v.strides(2);
  p.O_sb = o.strides(0);
  p.O_sh = o.strides(1);
  p.O_sl = o.strides(2);

  // 1 block per (seq, kv-head, q-tile): the ref's single-block-tile path
  // (kb0_start == 0, kb0_stop == iter_k -> needs_fixup == false).
  const int tiles = B * KVH * p.iter_j;
  dim3 grid(tiles, 1, 1);
  dim3 block(FATTN_NTHREADS, 1, 1);
  const uint32_t smem = FATTN_LDS;

  enc.set_input_array(q);
  enc.set_input_array(k);
  enc.set_input_array(v);
  enc.set_output_array(o);

  const hip_bfloat16* qp = gpu_ptr<const hip_bfloat16>(q);
  const hip_bfloat16* kp = gpu_ptr<const hip_bfloat16>(k);
  const hip_bfloat16* vp = gpu_ptr<const hip_bfloat16>(v);
  hip_bfloat16* op = gpu_ptr<hip_bfloat16>(o);

  if (do_causal) {
    enc.add_kernel_node_ex(
        &rocm::kernel_sdpa_flash_fattn<true>, grid, block, smem,
        qp, kp, vp, op, p);
  } else {
    enc.add_kernel_node_ex(
        &rocm::kernel_sdpa_flash_fattn<false>, grid, block, smem,
        qp, kp, vp, op, p);
  }
}

} // namespace mlx::core
