// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/fast_primitives.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace mlx::core {

namespace rocm {
extern void g1_check_hip(
    const char* call, hipError_t err, const char* phase = nullptr);
} // namespace rocm

namespace {
// Env-gated (MLX_COUNT_COPIES) attribution of general copies by shape, to find
// which model ops generate the copy_gg/copy_g kernels. Prints a histogram at exit.
struct CopyStats {
  std::mutex mu;
  std::map<std::string, long> hist;
  bool on = std::getenv("MLX_COUNT_COPIES") != nullptr;
  void record(CopyType ct, const Shape& sh) {
    if (!on) return;
    std::string key = (ct == CopyType::GeneralGeneral) ? "GG " : "G  ";
    key += "[";
    for (size_t i = 0; i < sh.size(); ++i) {
      key += std::to_string(sh[i]);
      if (i + 1 < sh.size()) key += ",";
    }
    key += "]";
    std::lock_guard<std::mutex> g(mu);
    hist[key]++;
  }
  ~CopyStats() {
    if (!on) return;
    long tot = 0;
    for (auto& kv : hist) tot += kv.second;
    std::vector<std::pair<std::string, long>> v(hist.begin(), hist.end());
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    fprintf(stderr, "\n=== MLX_COUNT_COPIES: %ld general copies, top shapes ===\n",
            tot);
    for (size_t i = 0; i < v.size() && i < 40; ++i)
      fprintf(stderr, "%8ld  %s\n", v[i].second, v[i].first.c_str());
  }
};
CopyStats g_copy_stats;
} // namespace

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  auto& encoder = rocm::get_command_encoder(s);
  bool donated = set_copy_output_data(
      in, out, ctype, [&](auto n) { return mlx::core::rocm::malloc_async(n, encoder); });
  if (donated && in.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int64_t offset_in,
    int64_t offset_out,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_offset_in,
    std::optional<array> dynamic_offset_out) {
  if (out.size() == 0) {
    return;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Handle dynamic offsets
  if (dynamic_offset_in.has_value() || dynamic_offset_out.has_value()) {
    auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
        shape, std::vector{strides_in, strides_out}, INT32_MAX);
    
    // Create zero offset arrays for missing dynamic offsets
    // We need to allocate and initialize on GPU to avoid hipDeviceSynchronize
    if (!dynamic_offset_in) {
      dynamic_offset_in = array({1}, int64, nullptr, {});
      dynamic_offset_in->set_data(mlx::core::rocm::malloc_async(sizeof(int64_t), encoder));
      encoder.add_temporary(*dynamic_offset_in);
      // Initialize to zero on GPU using hipMemset
      int64_t* ptr = gpu_ptr<int64_t>(*dynamic_offset_in);
      encoder.launch_kernel([ptr](hipStream_t stream) {
        hipError_t ms = hipMemsetAsync(ptr, 0, sizeof(int64_t), stream);
        rocm::g1_check_hip("hipMemsetAsync (copy dynamic_offset_in)", ms);
        (void)ms;
      });
    }
    if (!dynamic_offset_out) {
      dynamic_offset_out = array({1}, int64, nullptr, {});
      dynamic_offset_out->set_data(mlx::core::rocm::malloc_async(sizeof(int64_t), encoder));
      encoder.add_temporary(*dynamic_offset_out);
      // Initialize to zero on GPU using hipMemset
      int64_t* ptr = gpu_ptr<int64_t>(*dynamic_offset_out);
      encoder.launch_kernel([ptr](hipStream_t stream) {
        hipError_t ms = hipMemsetAsync(ptr, 0, sizeof(int64_t), stream);
        rocm::g1_check_hip("hipMemsetAsync (copy dynamic_offset_out)", ms);
        (void)ms;
      });
    }
    encoder.set_input_array(*dynamic_offset_in);
    encoder.set_input_array(*dynamic_offset_out);
    
    copy_general_dynamic(
        encoder,
        ctype,
        in,
        out,
        offset_in,
        offset_out,
        shape_collapsed,
        strides_vec[0],
        strides_vec[1],
        *dynamic_offset_in,
        *dynamic_offset_out);
    return;
  }
  
  if (ctype == CopyType::Scalar || ctype == CopyType::Vector) {
    copy_contiguous(encoder, ctype, in, out, offset_in, offset_out);
    return;
  }

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    g_copy_stats.record(ctype, shape);
    auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
        shape, std::vector{strides_in, strides_out}, INT32_MAX);
    if (ctype == CopyType::General) {
      copy_general_input(
          encoder,
          ctype,
          in,
          out,
          offset_in,
          offset_out,
          shape_collapsed,
          strides_vec[0]);
    } else {
      copy_general(
          encoder,
          ctype,
          in,
          out,
          offset_in,
          offset_out,
          shape_collapsed,
          strides_vec[0],
          strides_vec[1]);
    }
    return;
  }
}

void fill_gpu(const array& in, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  auto& encoder = rocm::get_command_encoder(s);
  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  copy_contiguous(encoder, CopyType::Scalar, in, out, 0, 0);
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    auto& encoder = rocm::get_command_encoder(s);
    out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

namespace rocm {

// K2: batched KV append — writes the new keys AND values into the
// pre-allocated cache buffers at `offset` in a single launch.
// Keys/values are [B, H, T, D] with T=1 (decode); treated as 2D
// [B*H, D] with per-row strides. The dst base is advanced by
// offset * cache_t_stride.
template <typename T, int VEC>
__global__ void kv_append_kernel(
    const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ dk, T* __restrict__ dv,
    int64_t k_row_stride, int64_t v_row_stride,
    int64_t dk_row_stride, int64_t dv_row_stride,
    int64_t n_rows, int64_t d) {
  int64_t nvec = n_rows * (d / VEC);
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * nvec) {
    return;
  }
  bool is_v = (i >= nvec);
  int64_t j = is_v ? (i - nvec) : i;
  int64_t row = j / (d / VEC);
  int64_t col = (j % (d / VEC)) * VEC;
  const T* src = is_v ? (v + row * v_row_stride + col) : (k + row * k_row_stride + col);
  T* dst = is_v ? (dv + row * dv_row_stride + col) : (dk + row * dk_row_stride + col);
  if constexpr (VEC == 8) {
    // 8 x 16-bit = 16 bytes
    uint4 val = *reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst) = val;
  } else {
    // 4 x 32-bit = 16 bytes
    uint4 val = *reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst) = val;
  }
}

// Scalar fallback for head dims not divisible by the vector width.
template <typename T>
__global__ void kv_append_scalar_kernel(
    const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ dk, T* __restrict__ dv,
    int64_t k_row_stride, int64_t v_row_stride,
    int64_t dk_row_stride, int64_t dv_row_stride,
    int64_t n_rows, int64_t d) {
  int64_t total = 2 * n_rows * d;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) {
    return;
  }
  bool is_v = (i >= n_rows * d);
  int64_t j = is_v ? (i - n_rows * d) : i;
  int64_t row = j / d;
  int64_t col = j % d;
  const T* src = is_v ? (v + row * v_row_stride + col) : (k + row * k_row_stride + col);
  T* dst = is_v ? (dv + row * dv_row_stride + col) : (dk + row * dk_row_stride + col);
  *dst = *src;
}

} // namespace rocm

namespace fast {

bool KVAppend::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void KVAppend::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  const array& k = inputs[0];
  const array& v = inputs[1];
  const array& ck = inputs[2];
  const array& cv = inputs[3];
  array& ok = outputs[0];
  array& ov = outputs[1];

  // In-place: the outputs share the cache buffers.
  ok.copy_shared_buffer(ck);
  ov.copy_shared_buffer(cv);

  if (k.size() == 0) {
    return;
  }

  int64_t B = k.shape(0);
  int64_t H = k.shape(1);
  int64_t D = k.shape(3);
  int64_t n_rows = B * H;

  // Decode: T=1, so the [B*H, D] rows are addressed via the H stride.
  int64_t k_row_stride = k.strides()[1];
  int64_t v_row_stride = v.strides()[1];
  int64_t dk_row_stride = ck.strides()[1];
  int64_t dv_row_stride = cv.strides()[1];
  // Advance the dst base to the write offset (T stride * offset).
  int64_t dk_base = offset_ * ck.strides()[2];
  int64_t dv_base = offset_ * cv.strides()[2];

  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_output_array(ok);
  encoder.set_output_array(ov);

  constexpr int BLOCK = 256;
  auto launch = [&](auto tag) {
    using DT = decltype(tag);
    constexpr int VEC = sizeof(DT) == 2 ? 8 : 4;
    if (D % VEC == 0) {
      int64_t nvec = n_rows * (D / VEC);
      int64_t total = 2 * nvec;
      int nblocks = (total + BLOCK - 1) / BLOCK;
      if constexpr (sizeof(DT) == 2) {
        encoder.add_kernel_node(
            &rocm::kv_append_kernel<DT, 8>,
            dim3(nblocks), dim3(BLOCK), 0,
            gpu_ptr<DT>(k), gpu_ptr<DT>(v),
            gpu_ptr<DT>(ok) + dk_base, gpu_ptr<DT>(ov) + dv_base,
            k_row_stride, v_row_stride, dk_row_stride, dv_row_stride,
            n_rows, D);
      } else {
        encoder.add_kernel_node(
            &rocm::kv_append_kernel<DT, 4>,
            dim3(nblocks), dim3(BLOCK), 0,
            gpu_ptr<DT>(k), gpu_ptr<DT>(v),
            gpu_ptr<DT>(ok) + dk_base, gpu_ptr<DT>(ov) + dv_base,
            k_row_stride, v_row_stride, dk_row_stride, dv_row_stride,
            n_rows, D);
      }
    } else {
      int64_t total = 2 * n_rows * D;
      int nblocks = (total + BLOCK - 1) / BLOCK;
      encoder.add_kernel_node(
          &rocm::kv_append_scalar_kernel<DT>,
          dim3(nblocks), dim3(BLOCK), 0,
          gpu_ptr<DT>(k), gpu_ptr<DT>(v),
          gpu_ptr<DT>(ok) + dk_base, gpu_ptr<DT>(ov) + dv_base,
          k_row_stride, v_row_stride, dk_row_stride, dv_row_stride,
          n_rows, D);
    }
  };
  switch (k.dtype()) {
    case float32: launch(float{}); break;
    case float16: launch(__half{}); break;
    case bfloat16: launch(hip_bfloat16{}); break;
    default: throw std::runtime_error("Unsupported type for kv_append");
  }
}

} // namespace fast

} // namespace mlx::core
