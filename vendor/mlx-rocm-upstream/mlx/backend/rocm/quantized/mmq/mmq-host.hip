// W1 GDN prefill fix (2026-11-14): host shims for the llama mmq port.
// The port (mmq.hip) links against the llama ggml-cuda host runtime, but the
// ported tree only carries the device + template headers. This file defines
// the missing host symbols so the mmq host path resolves:
//   - ggml_cuda_info / get_device / set_device / error
//   - the ggml_backend_cuda_context destructor + pool factory
//   - the explicit mul_mat_q_case instantiations (mmq.cuh declares them
//     `extern template`; the port's mmq.cu subset does not define them)
//   - the MoE-only quantize/ids launchers (stubs - the mlx path is dense)
//   - the small ggml C API subset the ported host code references
// Only the DENSE path is exercised from mlx (rocm::mmq_q8_0_g64 in mmq.hip).
#pragma once
#define GGML_USE_HIP 1
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>

// ------------------------------------------------------------ device runtime
int ggml_cuda_get_device() {
  return 0;
}

void ggml_cuda_set_device(int device) {
  hipSetDevice(device);
}

const ggml_cuda_device_info& ggml_cuda_info() {
  static ggml_cuda_device_info info = [] {
    ggml_cuda_device_info i{};
    int count = 0;
    hipGetDeviceCount(&count);
    if (count < 1) count = 1;
    i.device_count = count;
    i.physical_device_count = count;
    for (int d = 0; d < count; ++d) {
      hipDeviceProp_t prop{};
      hipGetDeviceProperties(&prop, d);
      auto& dev = i.devices[d];
      int major = 0;
      int minor = 0;
      // ggml cc encoding: AMD = GGML_CUDA_CC_OFFSET_AMD + gfx hex
      // (gfx1201 -> 0x1001201)
      if (sscanf(prop.gcnArchName, "gfx%x", &major) == 1) {
        dev.cc = GGML_CUDA_CC_OFFSET_AMD + major;
      } else {
        dev.cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
      }
      dev.nsm = (int)prop.multiProcessorCount;
      dev.smpb = prop.sharedMemPerBlock;
      dev.smpbo = prop.sharedMemPerBlockOptin;
      dev.integrated = prop.integrated;
      dev.vmm = true;
      dev.vmm_granularity = 2 * 1024 * 1024;
      dev.total_vram = prop.totalGlobalMem;
      dev.warp_size = 32;  // RDNA
      dev.supports_cooperative_launch = true;
      dev.physical_device = d;
      dev.physical_share_count = 1;
      dev.virtual_index = 0;
    }
    return i;
  }();
  return info;
}

void ggml_cuda_error(const char* stmt, const char* func,
                     const char* file, int line,
                     const char* msg) {
  fprintf(stderr, "mmq-shim: CUDA error in %s(%d): %s: %s: %s\n", file, line,
          func, stmt, msg ? msg : "?");
  if (std::getenv("MLX_MMQ_NONFATAL")) { hipGetLastError(); return; }
  abort();
}

// ------------------------------------------------------------ context pool
namespace {
struct mmq_shim_pool : ggml_cuda_pool {
  void* alloc(size_t size, size_t* actual_size) override {
    void* p = nullptr;
    CUDA_CHECK(hipMalloc(&p, size));
    if (p == nullptr) {
      fprintf(stderr, "mmq-shim: hipMalloc(%zu) failed\n", size);
      abort();
    }
    *actual_size = size;
    return p;
  }
  void free(void* ptr, size_t) override {
    CUDA_CHECK(hipFree(ptr));
  }
};
}  // namespace

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
  // pools: unique_ptr members (auto-destroyed). streams/cublas: never created
  // on the mlx path (the stream is passed in by the caller).
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(
    ggml_backend_cuda_context&, int, int) {
  return std::make_unique<mmq_shim_pool>();
}

// ------------------------------------------------------------ explicit
// instantiations (mirror the extern list in mmq.cuh, minus the extern)
template void mul_mat_q_case<GGML_TYPE_Q1_0, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q2_0, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_1, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q2_K, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q3_K, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ1_S, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_XS, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_XXS, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_NL, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_XS, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q8>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t);

// ------------------------------------------------------------ MoE-only
// launchers (stubs - abort if reached; the mlx path is dense)
namespace {
[[noreturn]] void mmq_shim_unreachable(const char* name) {
  fprintf(stderr, "mmq-shim: %s reached (MoE-only path, not ported)\n", name);
  abort();
}
}  // namespace

void quantize_mmq_q8_1_cuda(const float*, const int32_t*, void*, ggml_type,
                            int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t,
                            cudaStream_t) {
  mmq_shim_unreachable("quantize_mmq_q8_1_cuda");
}
void quantize_mmq_fp4_cuda(const float*, const int32_t*, void*, float*,
                           ggml_type, bool, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t,
                           cudaStream_t) {
  // G1 build-hygiene canary (host odr-use: volatile array, non-foldable load;
  // proves the b97-t2fix tree compiled into this lib)
  volatile const char b97t2fix_canary_ref = B97T2FIX_CANARY[0];
  (void)b97t2fix_canary_ref;
  // G1.5 probe build-hygiene canary (same volatile-array + host odr-use pattern)
  volatile const char b97probe_canary_ref = B97PROBE_CANARY[0];
  (void)b97probe_canary_ref;
  mmq_shim_unreachable("quantize_mmq_fp4_cuda");
}
void quantize_scatter_mmq_q8_1_cuda(const float*, const int32_t*, void*,
                                    ggml_type, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, int, cudaStream_t) {
  mmq_shim_unreachable("quantize_scatter_mmq_q8_1_cuda");
}
void quantize_scatter_mmq_fp4_cuda(const float*, const int32_t*, void*,
                                   float*, ggml_type, bool, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int,
                                   cudaStream_t) {
  mmq_shim_unreachable("quantize_scatter_mmq_fp4_cuda");
}
void ggml_cuda_launch_mm_ids_helper(const int32_t*, int32_t*, int32_t*,
                                    int32_t*, int, int, int, int, int, int,
                                    bool, cudaStream_t) {
  mmq_shim_unreachable("ggml_cuda_launch_mm_ids_helper");
}

// ------------------------------------------------------------ ggml C API
// (the subset the ported host code references; declarations from ggml.h carry
// the C linkage)
namespace {
int64_t mmq_shim_blck_size(enum ggml_type t) {
  switch (t) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
      return 1;
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
      return 256;
    case GGML_TYPE_IQ4_XS:
      return 128;
    case GGML_TYPE_NVFP4:
      return 16;
    default:
      return 32;
  }
}
}  // namespace

size_t ggml_nbytes(const struct ggml_tensor* t) {
  size_t n = 1;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    n *= (size_t)(t->ne[i] > 0 ? t->ne[i] : 1);
  }
  return n;
}

size_t ggml_type_size(enum ggml_type type) {
  switch (type) {
    case GGML_TYPE_F32:
      return sizeof(float);
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
      return sizeof(uint16_t);
    case GGML_TYPE_Q1_0:
      return sizeof(block_q1_0);
    case GGML_TYPE_Q2_0:
      return sizeof(block_q2_0);
    case GGML_TYPE_Q4_0:
      return sizeof(block_q4_0);
    case GGML_TYPE_Q4_1:
      return sizeof(block_q4_1);
    case GGML_TYPE_Q5_0:
      return sizeof(block_q5_0);
    case GGML_TYPE_Q5_1:
      return sizeof(block_q5_1);
    case GGML_TYPE_Q8_0:
      return sizeof(block_q8_0);
    case GGML_TYPE_Q8_1:
      return sizeof(block_q8_1);
    case GGML_TYPE_Q2_K:
      return sizeof(block_q2_K);
    case GGML_TYPE_Q3_K:
      return sizeof(block_q3_K);
    case GGML_TYPE_Q4_K:
      return sizeof(block_q4_K);
    case GGML_TYPE_Q5_K:
      return sizeof(block_q5_K);
    case GGML_TYPE_Q6_K:
      return sizeof(block_q6_K);
    case GGML_TYPE_IQ1_S:
      return sizeof(block_iq1_s);
    case GGML_TYPE_IQ2_XXS:
      return sizeof(block_iq2_xxs);
    case GGML_TYPE_IQ2_XS:
      return sizeof(block_iq2_xs);
    case GGML_TYPE_IQ2_S:
      return sizeof(block_iq2_s);
    case GGML_TYPE_IQ3_XXS:
      return sizeof(block_iq3_xxs);
    case GGML_TYPE_IQ3_S:
      return sizeof(block_iq3_s);
    case GGML_TYPE_IQ4_XS:
      return sizeof(block_iq4_xs);
    case GGML_TYPE_IQ4_NL:
      return sizeof(block_iq4_nl);
    default:
      return sizeof(float);
  }
}

size_t ggml_element_size(const struct ggml_tensor* t) {
  return ggml_type_size(t->type) / (size_t)mmq_shim_blck_size(t->type);
}

bool ggml_is_contiguously_allocated(const struct ggml_tensor* t) {
  size_t n = 1;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (t->ne[i] <= 0) break;
    if ((size_t)t->nb[i] != n * ggml_element_size(t)) return false;
    n *= (size_t)t->ne[i];
  }
  return true;
}

bool ggml_are_same_shape(const struct ggml_tensor* t0,
                         const struct ggml_tensor* t1) {
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (t0->ne[i] != t1->ne[i]) return false;
  }
  return true;
}

bool ggml_are_same_stride(const struct ggml_tensor* t0,
                          const struct ggml_tensor* t1) {
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (t0->nb[i] != t1->nb[i]) return false;
  }
  return true;
}

void ggml_abort(const char* file, int line, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s(%d): ", file, line);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  abort();
}

void ggml_log_internal(enum ggml_log_level level, const char* format, ...) {
  (void)level;
  va_list ap;
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer,
                                         const struct ggml_tensor* tensor) {
  (void)buffer;
  return ggml_nbytes(tensor);
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(
    ggml_backend_buffer_t buffer) {
  (void)buffer;
  return GGML_BACKEND_BUFFER_USAGE_ANY;
}
