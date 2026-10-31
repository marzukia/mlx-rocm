// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include <algorithm>
#include <atomic>
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/backend/rocm/g1_hiplog.h"
#include "mlx/utils.h"

#include <chrono>
#include <sys/user.h>
#include <sys/syscall.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <cxxabi.h>
#include <dlfcn.h>
#include <vector>

namespace mlx::core::rocm {
__global__ void g1b_noop();  // G1B (defined in copy.hip)


// G1 root-cause instrumentation (defined below).
extern long g1_trace_n;
// G1 first-error tracking (defined below, gated by MLX_G1_TRACE /
// MLX_G1_FIRST_ERR). Forward-declared for use in the capture-region call
// sites above the definition.
bool g1_fe_on();
void g1_check_hip(const char* call, hipError_t err, const char* phase = nullptr);
void g1_check_rocblas(const char* call, rocblas_status st);
// G1f: GEMM shape prewarm (defined in qmm.hip, same namespace).
void g1_prewarm_gemm_shapes(rocblas_handle handle, hipStream_t stream);
void g1_check_status(const char* call, int code, const char* msg,
                    const char* phase = nullptr);
void g1_reset_first_error();
void g1_first_error_summary(const char* phase);

namespace {

// Can be tuned with MLX_MAX_OPS_PER_BUFFER
constexpr int default_max_ops_per_buffer = 2000;

inline bool is_empty_dim(dim3 dim) {
  return (dim.x == 0 && dim.y == 0 && dim.z == 0) ||
      (dim.x == 1 && dim.y == 1 && dim.z == 1);
}

} // namespace

// True while the engine is in a single-token decode step; false during prefill
// (multi-token) and outside generation. Set by set_graph_decode_mode().
static std::atomic<bool> g_graph_decode_mode{false};

bool use_hip_graphs() {
  // The rebuild-every-eval graph-build path is a net loss vs eager (and the
  // prefill variant segfaults on RDNA3.5). Decode uses build-once
  // capture/replay (decode_capture_*) and prefill uses the WMMA GEMM — neither
  // goes through here — so this path is permanently off.
  //
  // Train graphs: also hard-off until a TrainArena (see RocmAllocator
  // train_arena_*) wraps the step tape with stable addresses. Flipping this
  // without an arena reintroduces aperture violations / wrong loss.
  return false;
}

// Count of inline (graph-splitting) launches — library GEMM / JIT / memset run
// outside the graph via launch_kernel. Declared extern in device.h.
std::atomic<long> g_inline_launches_{0};
// True while a full decode-step stream capture is in progress. The worker's
// completion signaling uses hipLaunchHostFunc, which is NOT stream-capturable
// (and a captured host node would re-fire on every replay). While capturing we
// skip the host callback and signal completion inline instead (see Worker).
std::atomic<bool> g_decode_capturing{false};
std::atomic<int> g_decode_capture_slot{0};

// G1X_OFF: env-gate the G1X deep copy (isolation test).
static bool g1x_off() {
  static const bool o = std::getenv("MLX_G1X_OFF") != nullptr;
  return o;
}
// G1X: guarded 8-byte read of a pointer slot that may be unmapped
// (stack garbage beyond the real arg count). Uses mincore to check
// page presence without faulting (the sigaction/sigsetjmp guard was
// not reliably catching the fault on ROCm 10.1 worker threads).
static inline uint64_t g1x_safe_read8(void* v) {
  if (!v)
    return 0;
  uintptr_t u = reinterpret_cast<uintptr_t>(v);
  if (u < 0x10000000 || u >= 0x10000000000 || (u & 7))
    return 0;
  uintptr_t base = u & ~4095ULL;
  bool second = (u - base) + 8 > 4095;
  unsigned char pres[2] = {1, 1};
  if (syscall(SYS_mincore, reinterpret_cast<void*>(base), second ? 8192 : 4096, pres) != 0)
    return 0;
  if (pres[0] == 0 || (second && pres[1] == 0))
    return 0;
  return *reinterpret_cast<uint64_t*>(v);
}

// G1KSL: tiny kernel that writes one int64 (kernel launches are
// reliably delivered on this vGPU; H2D memcpys apparently are not).
__global__ void g1_slot_write_kernel(int64_t* dst, uint64_t val);

// G1E (v2): dump the SOURCE graph's kernel arg values for the first N
// kernel nodes. ROCm 10.1's HIP has no exec GetParams/GetNodes (only
// hipGraphKernelNodeGetParams + hipGraphExecKernelNodeSetParams), so the
// exec's own staged values are probed with the G1S sentinel instead.
static void g1e_source_dump(hipGraph_t g, const char* tag) {
  if (std::getenv("MLX_G1_EXECDUMP") == nullptr &&
      std::getenv("MLX_G1_NODEDUMP") == nullptr)
    return;
  int nmax = 8;
  if (const char* e = std::getenv("MLX_G1_EXECDUMP_N"))
    nmax = std::atoi(e);
  std::vector<hipGraphNode_t> nodes(1024);
  size_t qn = nodes.size();
  if (!g || hipGraphGetNodes(g, nodes.data(), &qn) != hipSuccess) {
    (void)hipGetLastError();
    qn = 0;
  }
  for (size_t i = 0; i < qn && i < (size_t)nmax; ++i) {
    hipGraphNode_t nd = nodes[i];
    hipGraphNodeType ty = (hipGraphNodeType)0;
    if (hipGraphNodeGetType(nd, &ty) != hipSuccess ||
        ty != hipGraphNodeTypeKernel)
      continue;
    hipKernelNodeParams kp;
    memset(&kp, 0, sizeof(kp));
    if (hipGraphKernelNodeGetParams(nd, &kp) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    char buf[512];
    int off = 0;
    auto** ka = static_cast<void**>(kp.kernelParams);
    if (ka) {
      for (int a = 0; a < 6; ++a) {
        void* v = ka[a];
        uint64_t val = v ? g1x_safe_read8(v) : 0;
        off += snprintf(buf + off, sizeof(buf) - off, " a%d=%016llx", a,
                        (unsigned long long)val);
      }
    }
    fprintf(stderr, "[g1s] %s n%zu grid=%u,%u,%u%s\n",
            tag, i, kp.gridDim.x, kp.gridDim.y, kp.gridDim.z, buf);
    fflush(stderr);
  }
}

// G1S: sentinel — does the exec's packet pick up a SetParams change made
// at (re)capture time? Re-points the first kernel node's out arg (arg2
// of the 6-arg gather_rows signature) at a zeroed scratch buffer, then
// after the launch checks who was written: scratch written -> the exec
// staged the NEW value (ExecUpdate/SetParams propagated, the values are
// not stale); scratch untouched -> the exec still has the OLD out
// pointer (stale ExecUpdate packet, rocm-systems#10714 class). Armed
// only on the pre-existing-exec (ExecUpdate) path.
struct g1s_state_t {
  void* scratch = nullptr;
  size_t n = 512;
  bool armed = false;
  void* orig_v2 = nullptr;
};
static g1s_state_t g1s_state;

static void g1s_arm(hipGraphExec_t exec, hipGraph_t g) {
  if (std::getenv("MLX_G1_SENTINEL") == nullptr)
    return;
  std::vector<hipGraphNode_t> nodes(16);
  size_t qn = nodes.size();
  if (hipGraphGetNodes(g, nodes.data(), &qn) != hipSuccess) {
    (void)hipGetLastError();
    qn = 0;
  }
  hipGraphNode_t k0 = nullptr;
  for (size_t i = 0; i < qn; ++i) {
    hipGraphNodeType ty = (hipGraphNodeType)0;
    if (hipGraphNodeGetType(nodes[i], &ty) == hipSuccess &&
        ty == hipGraphNodeTypeKernel) {
      k0 = nodes[i];
      break;
    }
  }
  if (!k0)
    return;
  hipKernelNodeParams kp;
  memset(&kp, 0, sizeof(kp));
  if (hipGraphKernelNodeGetParams(k0, &kp) != hipSuccess) {
    (void)hipGetLastError();
    return;
  }
  if (!kp.kernelParams || !kp.kernelParams[2])
    return;
  if (!g1s_state.scratch) {
    if (hipMalloc(&g1s_state.scratch, g1s_state.n * 8) != hipSuccess ||
        hipMemset(g1s_state.scratch, 0, g1s_state.n * 8) != hipSuccess) {
      (void)hipGetLastError();
      return;
    }
  }
  uint64_t vals[8];
  void* newkp[8];
  int n = 0;
  for (int i = 0; i < 8 && kp.kernelParams[i]; ++i) {
    vals[i] = g1x_safe_read8(kp.kernelParams[i]);
    newkp[i] = &vals[i];
  }
  if (n > 2) {
    g1s_state.orig_v2 = (void*)vals[2];
    vals[2] = (uint64_t)g1s_state.scratch;
    newkp[2] = &vals[2];
  }
  newkp[n] = nullptr; // hipKernelNodeParams has no numParams; the array is
  // null-terminated.
  kp.kernelParams = newkp;
  hipError_t r = hipGraphExecKernelNodeSetParams(exec, k0, &kp);
  g1s_state.armed = (r == hipSuccess);
  fprintf(stderr, "[g1s] armed=%d (SetParams rc=%d) scratch=%p\n",
          (int)g1s_state.armed, (int)r, g1s_state.scratch);
  fflush(stderr);
}

static void g1s_check(hipStream_t st) {
  if (!g1s_state.armed || !g1s_state.scratch)
    return;
  g1s_state.armed = false;
  (void)hipStreamSynchronize(st);
  std::vector<uint64_t> h(g1s_state.n, 0);
  (void)hipMemcpy(h.data(), g1s_state.scratch, g1s_state.n * 8,
                  hipMemcpyDeviceToHost);
  size_t written = 0;
  for (uint64_t v : h)
    if (v != 0)
      written++;
  int64_t s0 = -1, s1 = -1, s2 = -1;
  auto& alr = rocm::allocator();
  auto rd = [&](int i, int64_t& out) {
    auto sl = alr.decode_value_slot(i);
    if (sl.data) {
      (void)hipMemcpy(&out, sl.data, 8, hipMemcpyDeviceToHost);
    }
  };
  rd(0, s0);
  rd(1, s1);
  rd(2, s2);
  fprintf(stderr,
          "[g1s] post-launch: scratch written=%zu/%zu (orig out=%p) "
          "slot0=%lld slot1=%lld slot2=%lld => %s\n",
          written, g1s_state.n, g1s_state.orig_v2, (long long)s0, (long long)s1,
          (long long)s2,
          written > 0 ? "exec staged the NEW value (not stale)"
                      : "exec kept the OLD out pointer (stale packet)");
  fflush(stderr);
  (void)hipMemset(g1s_state.scratch, 0, g1s_state.n * 8);
}

// Pre-launch slot H2D: the slot H2D memcpy nodes recorded during stream
// capture do not reliably deliver the current host value before the graph
// kernels read the slot device buffer (r15-r19: APERTURE fault on the
// first graph launch with a stale/garbage slot value). Issue the H2D as a
// plain stream op immediately before hipGraphLaunch on the same stream so
// it is ordered before every graph node. Redundant (same value) if the
// in-graph memcpy nodes do work. MLX_G1_NO_PREH2D=1 disables.
static void g1_prelaunch_slot_h2d(hipStream_t st) {
  if (!graph_decode_mode())
    return;
  if (const char* e = std::getenv("MLX_G1_NO_PREH2D");
      e && e[0] == '1')
    return;
  bool use_kernel = std::getenv("MLX_G1_KSL") != nullptr;
  bool logh = std::getenv("MLX_G1_H2DLOG") != nullptr;
  auto& alc = rocm::allocator();
  for (int i = 0; i < 4; ++i) {
    auto sl = alc.decode_value_slot(i);
    if (sl.host && sl.data) {
      if (use_kernel) {
        int64_t* dst = static_cast<int64_t*>(sl.data);
        uint64_t val = (uint64_t)*sl.host;
        void* kargs[2] = {&dst, &val};
        hipError_t e = hipLaunchKernel(
            (void*)g1_slot_write_kernel, dim3(1), dim3(1), kargs, 0, st);
        g1_check_hip("hipLaunchKernel (pre-launch slot write, kernel)", e);
      } else {
        // Async on the launch stream: a blocking hipMemcpy lands on the
        // legacy default stream, which does NOT order against the
        // non-blocking decode stream (r43: slot1 still held the
        // capture-step value at readback). The host buffers are pinned,
        // so the async H2D is safe and stream-ordered before the graph
        // launch below.
        hipError_t e = hipMemcpyAsync(
            sl.data, sl.host, sizeof(int64_t), hipMemcpyHostToDevice, st);
        g1_check_hip("hipMemcpyAsync (pre-launch slot H2D)", e);
      }
      if (logh) {
        fprintf(stderr, "[g1h] slot%d %p <- %lld\n", i, sl.data,
                (long long)*sl.host);
        fflush(stderr);
      }
    }
  }
  // The idx (zero) slot must read 0 at every launch: the capture step's
  // kernels can leave arena garbage in it (and nothing re-zeroes it),
  // which a gather then reads as an index -> off-arena fault address.
  auto zs = alc.decode_zero_slot();
  if (zs.data) {
    hipError_t e = hipMemsetAsync(zs.data, 0, sizeof(int64_t), st);
    g1_check_hip("hipMemsetAsync (pre-launch zero-slot clear)", e);
    if (logh) {
      fprintf(stderr, "[g1h] zero %p <- 0\n", zs.data);
      fflush(stderr);
    }
  }
}
// G1H2D: record each in-graph H2D memcpy node's (host_src, dev_dst,
// bytes) into the slot's staging list and neutralize the node (0-byte
// copy) so it cannot overwrite the pre-launch-delivered value mid-graph
// with a stale one. Runs once per capture, on the source graph BEFORE
// the exec is (re)instantiated, so the instantiated exec carries the
// neutralized nodes. The [g1m] memcpy lines (original params) are dumped
// here so the diagnostic shows the values as captured.
void CommandEncoder::decode_capture_record_neutralize_h2d(int slot,
                                                       hipGraph_t g) {
  slot &= 1;
  if (slot < 0 || slot > 1)
    slot = 0;
  g1h2d_recs_[slot].clear();
  size_t qn = 0;
  if (hipGraphGetNodes(g, nullptr, &qn) != hipSuccess) {
    (void)hipGetLastError();
    return;
  }
  if (qn == 0)
    return;
  std::vector<hipGraphNode_t> mn(qn);
  if (hipGraphGetNodes(g, mn.data(), &qn) != hipSuccess) {
    (void)hipGetLastError();
    return;
  }
  const bool memdump = std::getenv("MLX_G1_MEMDUMP") != nullptr;
  for (size_t i = 0; i < qn; ++i) {
    hipGraphNodeType ty = (hipGraphNodeType)0;
    if (hipGraphNodeGetType(mn[i], &ty) != hipSuccess ||
        ty != hipGraphNodeTypeMemcpy)
      continue;
    hipMemcpy3DParms mp;
    if (hipGraphMemcpyNodeGetParams(mn[i], &mp) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    size_t bytes = (size_t)mp.extent.width * mp.extent.height *
                   mp.extent.depth;
    const void* sp = mp.srcPtr.ptr;
    void* dp = mp.dstPtr.ptr;
    if (mp.kind != hipMemcpyHostToDevice || !sp || !dp || bytes == 0)
      continue;
    // Confirm src is host and dst is device (skip anything else).
    hipPointerAttribute_t sa;
    hipPointerAttribute_t da;
    if (hipPointerGetAttributes(&sa, const_cast<void*>(sp)) != hipSuccess ||
        hipPointerGetAttributes(&da, dp) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    if (sa.type != hipMemoryTypeHost || da.type != hipMemoryTypeDevice)
      continue;
    g1h2d_recs_[slot].push_back({const_cast<void*>(sp), dp, bytes});
    if (memdump) {
      fprintf(stderr,
              "[g1m] n%zu kind=%d src=%p(h) dst=%p(d) bytes=%zu\n", i,
              (int)mp.kind, sp, dp, bytes);
      fflush(stderr);
    }
    // Neutralize: 0-byte copy = no-op, so the in-graph node can no
    // longer overwrite the pre-launch-delivered value with a stale one.
    mp.extent.width = 0;
    mp.extent.height = 0;
    mp.extent.depth = 0;
    hipError_t se = hipGraphMemcpyNodeSetParams(mn[i], &mp);
    g1_check_hip("hipGraphMemcpyNodeSetParams (neutralize H2D)", se, "h2d");
    if (se != hipSuccess) {
      (void)hipGetLastError();
    }
  }
  if (memdump && !g1h2d_recs_[slot].empty()) {
    fprintf(stderr,
            "[g1m] slot%d: %zu in-graph H2D node(s) recorded + neutralized; "
            "re-delivered pre-launch\n",
            slot, g1h2d_recs_[slot].size());
    fflush(stderr);
  }
}
// G1H2D: re-deliver the recorded in-graph H2D staging values on the
// launch stream immediately before hipGraphLaunch (same pattern as
// g1_prelaunch_slot_h2d): a plain stream op ordered before every graph
// node, so the device staging buffer holds the current host value when
// the graph's kernels read it.
void CommandEncoder::g1_prelaunch_recorded_h2d(int slot, hipStream_t st) {
  slot &= 1;
  if (!graph_decode_mode())
    return;
  if (const char* e = std::getenv("MLX_G1_NO_PREH2D"); e && e[0] == '1')
    return;
  const bool logh = std::getenv("MLX_G1_H2DLOG") != nullptr;
  for (auto& r : g1h2d_recs_[slot]) {
    hipError_t e = hipMemcpyAsync(r.dev_dst, r.host_src, r.bytes,
                                 hipMemcpyHostToDevice, st);
    g1_check_hip("hipMemcpyAsync (pre-launch recorded H2D)", e);
    if (logh) {
      fprintf(stderr, "[g1h] recorded %p <- %p (%zu)\n", r.dev_dst,
              r.host_src, r.bytes);
      fflush(stderr);
    }
  }
}
void set_current_prim(const char*) {}
void record_inline_launch() {
  g_inline_launches_.fetch_add(1, std::memory_order_relaxed);
}

// Per-arch op/MB caps for the build graph. Tunable via env.
// The earlier "corrupts at >3 nodes" was actually one bad op (Concatenate,
// whose multi-copy kernels corrupt when co-grouped); it is now graph-split in
// gpu::eval (is_graph_split_op), so large graphs are correct again.
static std::pair<int, int> get_graph_limits() {
  int ops = env::max_ops_per_buffer(50);
  int mb = env::max_mb_per_buffer(200);
  return {ops, mb};
}

Device::Device(int device) : device_(device) {
  make_current();
  {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, device_) == hipSuccess) {
      fprintf(
          stderr,
          "[mlx-rocm] bound HIP device %d: %s (%s) cus=%d warp=%d lds=%zuKB\n",
          device_,
          p.gcnArchName,
          p.name,
          p.multiProcessorCount,
          p.warpSize,
          p.sharedMemPerBlock / 1024);
      if (p.sharedMemPerBlock > 0) {
        max_shared_memory_per_block_ = static_cast<int>(p.sharedMemPerBlock);
      }
      if (p.warpSize > 0) {
        warp_size_ = p.warpSize;
      }
      if (p.multiProcessorCount > 0) {
        num_cus_ = p.multiProcessorCount;
      }
      if (p.maxThreadsPerBlock > 0) {
        max_threads_per_block_ = p.maxThreadsPerBlock;
      }
      // Optional debug/experiment override (RDNA is hardware wave32 including
      // RDNA4; only set this if you know you need a different launch width).
      // Wrong value vs device kernels → garbage QMV / decode.
      if (const char* fw = std::getenv("MLX_ROCM_FORCE_WARP_SIZE")) {
        int forced = std::atoi(fw);
        if (forced == 32 || forced == 64) {
          fprintf(
              stderr,
              "[mlx-rocm] MLX_ROCM_FORCE_WARP_SIZE=%d overrides device warp=%d\n",
              forced,
              warp_size_);
          warp_size_ = forced;
        } else {
          fprintf(
              stderr,
              "[mlx-rocm] ignoring MLX_ROCM_FORCE_WARP_SIZE=%s (want 32 or 64)\n",
              fw);
        }
      }
    }
  }
  // rocBLAS initialization is now lazy - done in get_rocblas_handle()
}

Device::~Device() {
  if (rocblas_) {
    rocblas_destroy_handle(rocblas_);
  }
}

rocblas_handle Device::get_rocblas_handle() {
  if (!rocblas_initialized_) {
    rocblas_initialized_ = true;
    make_current();

    // Check if the GPU architecture is supported by rocBLAS
    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, device_);    std::string arch_name = props.gcnArchName;

    // List of architectures supported by rocBLAS (based on TensileLibrary
    // files). These are the architectures that have TensileLibrary_lazy_*.dat.
    static const std::vector<std::string> supported_archs = {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1200",
        "gfx1201"};

    // Extract base architecture name (remove any suffix like :sramecc+:xnack-)
    std::string base_arch = arch_name;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }

    bool arch_supported = false;
    for (const auto& supported : supported_archs) {
      if (base_arch == supported) {
        arch_supported = true;
        break;
      }
    }

    if (!arch_supported) {
      rocblas_available_ = false;
      rocblas_ = nullptr;
      std::cerr << "Warning: rocBLAS does not support GPU architecture '"
                << arch_name << "'. "
                << "Matrix multiplication operations will not be available. "
                << "Supported architectures: gfx908, gfx90a, gfx942, gfx950, "
                << "gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, "
                << "gfx1152, gfx1200, gfx1201." << std::endl;
    } else {
      // G1: a mid-capture handle creation is a capture-region allocator call
      // (rocBLAS lazy-inits a workspace here); name it if it fails.
      rocblas_status status = rocblas_create_handle(&rocblas_);
      g1_check_rocblas("rocblas_create_handle (lazy init)", status);
      if (g1_fe_on() &&
          g_decode_capturing.load(std::memory_order_relaxed)) {
        fprintf(stderr,
                "[g1fe] NOTE: rocblas handle CREATED mid-capture (op %ld)\n",
                g1_trace_n);
      }
      if (status != rocblas_status_success) {
        rocblas_available_ = false;
        rocblas_ = nullptr;
        std::cerr
            << "Warning: rocBLAS initialization failed (status "
            << static_cast<int>(status)
            << "). Matrix multiplication operations will not be available."
            << std::endl;
      }
    }
  }
  if (!rocblas_available_) {
    throw std::runtime_error(
        "rocBLAS is not available on this GPU architecture. "
        "Matrix multiplication operations are not supported.");
  }
  return rocblas_;
}

bool Device::is_rocblas_available() {
  if (!rocblas_initialized_) {
    try {
      get_rocblas_handle();
    } catch (...) {
    }
  }
  return rocblas_available_;
}

bool Device::is_rocblas_bf16_available() {
  if (!rocblas_bf16_probed_) {
    rocblas_bf16_probed_ = true;
    rocblas_bf16_available_ = false;

    if (!is_rocblas_available()) {
      return false;
    }

    // Probe: run a tiny bf16 GEMM and check if the GPU survives.
    // rocBLAS may claim support but crash if the Tensile .co files
    // are corrupt or missing specific kernel variants.
    make_current();
    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    hipError_t err;

    err = hipMalloc(&a_ptr, 4 * 4 * 2); // 4x4 bf16
    if (err != hipSuccess)
      return false;
    err = hipMalloc(&b_ptr, 4 * 4 * 2);
    if (err != hipSuccess) {
      (void)hipFree(a_ptr);      return false;
    }
    err = hipMalloc(&c_ptr, 4 * 4 * 2);
    if (err != hipSuccess) {
      (void)hipFree(a_ptr);      (void)hipFree(b_ptr);      return false;
    }

    (void)hipMemset(a_ptr, 0, 4 * 4 * 2);
    (void)hipMemset(b_ptr, 0, 4 * 4 * 2);
    (void)hipMemset(c_ptr, 0, 4 * 4 * 2);

    float alpha = 1.0f, beta = 0.0f;
    rocblas_status status = rocblas_gemm_ex(
        rocblas_,
        rocblas_operation_none,
        rocblas_operation_none,
        4,
        4,
        4,
        &alpha,
        a_ptr,
        rocblas_datatype_bf16_r,
        4,
        b_ptr,
        rocblas_datatype_bf16_r,
        4,
        &beta,
        c_ptr,
        rocblas_datatype_bf16_r,
        4,
        c_ptr,
        rocblas_datatype_bf16_r,
        4,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        0);

    // Sync and check if the GPU is still alive
    hipError_t sync_err = G1_HL_SYNC("hipDeviceSynchronize (device_check)", hipDeviceSynchronize(), "device_check");
    // Clear any lingering error
    (void)hipGetLastError();

    (void)hipFree(a_ptr);    (void)hipFree(b_ptr);    (void)hipFree(c_ptr);
    if (status == rocblas_status_success && sync_err == hipSuccess) {
      rocblas_bf16_available_ = true;
    } else {
      // GPU may be in a bad state — need to reset
      (void)hipDeviceReset();
      // Re-initialize device
      make_current();
      // Re-create rocBLAS handle
      if (rocblas_) {
        rocblas_destroy_handle(rocblas_);
        rocblas_ = nullptr;
      }
      rocblas_status rs = rocblas_create_handle(&rocblas_);
      if (rs != rocblas_status_success) {
        rocblas_available_ = false;
      }
      std::cerr << "Warning: rocBLAS bfloat16 GEMM probe failed on this GPU. "
                << "Using fallback kernels for bf16 matmul." << std::endl;
    }
  }
  return rocblas_bf16_available_;
}

bool Device::has_native_wmma() {
  if (!wmma_probed_) {
    wmma_probed_ = true;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_) != hipSuccess) {
      has_native_wmma_ = false;
      return has_native_wmma_;
    }

    // Strip any ":sramecc+:xnack-" style suffix from gcnArchName.
    std::string base_arch = props.gcnArchName;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }

    // rocWMMA arch allowlist. Keep in sync with detect_rocm_hw_info() in
    // mlx/backend/rocm/quantized/qmm.hip. RDNA3.5: gfx1150/1151/1152 all have
    // WMMA; Device used to omit 1150/1152 which forced flash/qmm off on those
    // parts (incl. reduced-CU gfx1152 instances).
    static const std::vector<std::string> rocwmma_archs = {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx1200",
        "gfx1201",
    };
    for (const auto& a : rocwmma_archs) {
      if (base_arch == a) {
        has_native_wmma_ = true;
        break;
      }
    }
  }
  return has_native_wmma_;
}

bool Device::supports_cdna_mfma_gemm() {
  if (!cdna_mfma_probed_) {
    cdna_mfma_probed_ = true;
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_) != hipSuccess) {
      cdna_mfma_ok_ = false;
      return cdna_mfma_ok_;
    }
    std::string base_arch = props.gcnArchName;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }
    // CDNA2 (gfx90a) + CDNA3 (gfx942) only: clean v_mfma_f32_16x16x16bf16 and a
    // working hipBLASLt pointer-offset GEMM. gfx1151 (RDNA3.5) pegs on the
    // offset path; gfx908 (CDNA1) bf16 MFMA differs; RDNA/others use WMMA.
    cdna_mfma_ok_ = (base_arch == "gfx90a" || base_arch == "gfx942");
  }
  return cdna_mfma_ok_;
}

void Device::make_current() {
  // HIP's current device is per-thread, so the cache must be too — a process
  // global lets one thread's binding suppress another's, stranding allocations
  // on the wrong device in a multi-GPU / multi-stream-thread run.
  thread_local int current = -1;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

void Device::set_rocblas_stream(hipStream_t stream) {
  if (rocblas_stream_ != stream) {
    rocblas_status rs = rocblas_set_stream(get_rocblas_handle(), stream);
    g1_check_rocblas("rocblas_set_stream", rs);
    rocblas_stream_ = stream;
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  // Bind this device current before constructing/returning the encoder. Callers
  // reach this member directly (e.g. QuantizedMatmul::eval_gpu), and the
  // encoder's stream + the kernels launched on it must land on this device, not
  // whatever was current on the calling thread.
  make_current();
  std::lock_guard<std::mutex> lk(encoders_mtx_);
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    if (getenv("MLX_GDN_DEBUG")) {
      fprintf(stderr, "[g1] CommandEncoder for stream index=%d dev=%d\n", s.index, hip_device());
    }
    auto [inserted_it, success] =
        encoders_.emplace(s.index, std::make_unique<CommandEncoder>(*this));
    it = inserted_it;
  }
  return *it->second;
}

void Device::clear_encoders() {
  encoders_.clear();
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d),
      stream_(d),
      worker_(std::make_unique<Worker>(d.hip_device())) {
  std::tie(max_ops_per_graph_, max_mb_per_graph_) = get_graph_limits();
  if (use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    set_graph_active(true);
  }
}

CommandEncoder::~CommandEncoder() {
  for (auto& [key, pool] : exec_pool_) {
    for (auto& slot : pool) {
      (void)hipGraphExecDestroy(slot.exec);      if (slot.source_graph) {
        (void)hipGraphDestroy(slot.source_graph);
      }
    }
  }
  if (build_graph_) {
    (void)hipGraphDestroy(build_graph_);
    build_graph_ = nullptr;
  }
}

void CommandEncoder::add_temporary(const array& arr) {
  auto data = arr.data_shared_ptr();
  const array::Data* ptr = data.get();
  if (temporary_ptrs_.insert(ptr).second) {
    temporaries_.push_back(std::move(data));
  }
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  bytes_in_graph_ += arr.data_size();
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
}

void CommandEncoder::set_output_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
  active_outputs_.push_back(id);
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  node.id = std::to_string(node_count_++);
  std::vector<GraphNode> nodes;
  nodes.push_back(std::move(node));
  insert_graph_dependencies(std::move(nodes));
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  // Two edge sets combined:
  //  (1) REAL data-dependency edges (node_map_: buffer ptr -> producing node).
  //      These give each node a unique data-flow identity, which
  //      hipGraphExecUpdate relies on to re-map kernelParams correctly across
  //      reuse — a bare submission chain leaves same-signature nodes ambiguous
  //      and corrupts under ExecUpdate.
  //  (2) A submission-order CHAIN edge (last_node_ -> node) as a backstop. Not
  //      every migrated kernel registers all of its I/O (the dense GEMM path
  //      and a few others register nothing), so the real-dep graph alone has
  //      missing edges -> races -> coherent-but-wrong output. The chain is a
  //      superset of the true partial order (eager runs serial on one stream),
  //      so it fills every gap and makes the graph bit-correct vs eager, while
  //      the real edges still uniquely identify nodes for ExecUpdate.
  for (auto& node : nodes) {
    graph_nodes_key_ += node.node_type;
    graph_nodes_key_ += "-";
  }
  std::vector<GraphNode> deps;
  std::unordered_set<hipGraphNode_t> set_deps;
  for (auto d : active_deps_) {
    if (auto it = node_map_.find(d); it != node_map_.end()) {
      if (set_deps.insert(it->second.node).second) {
        deps.push_back(it->second);
      }
    }
  }
  active_deps_.clear();

  for (auto o : active_outputs_) {
    for (auto& node : nodes) {
      node_map_.emplace(o, node).first->second = node;
    }
  }
  active_outputs_.clear();

  for (auto& from : deps) {
    for (auto& to : nodes) {
      from_nodes_.push_back(from.node);
      to_nodes_.push_back(to.node);
      graph_deps_key_ += from.id;
      graph_deps_key_ += "-";
      graph_deps_key_ += to.id;
      graph_deps_key_ += "-";
    }
  }

  // (2) Submission-order CHAIN edge. Serialize every node behind the previously
  // inserted node so the graph's execution order is a superset of eager's
  // serial-on-one-stream order — even for kernels that register no I/O (dense
  // GEMM path and a few others), whose nodes would otherwise have no edges and
  // race their true producers/consumers at any cap > 1. The chain is
  // deterministic for a given node sequence, so identical kernel sequences
  // still produce identical topology and hipGraphExecUpdate stays valid. Skip
  // an edge already present as a real data dep (avoid an exact duplicate edge).
  hipGraphNode_t prev = last_node_;
  for (auto& to : nodes) {
    // A real-dep edge prev->to was already added iff prev is in set_deps; skip
    // then so we never add an exact duplicate edge (HIP rejects duplicates).
    // Internal batch nodes are never external producers, so they always chain.
    if (prev && !set_deps.count(prev)) {
      from_nodes_.push_back(prev);
      to_nodes_.push_back(to.node);
      graph_deps_key_ += "c-";
    }
    prev = to.node;
  }
  last_node_ = prev;
}

void CommandEncoder::add_kernel_node_raw(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params,
    bool persistent) {
  if (!use_hip_graphs()) {
    device_.make_current();
    // G1f: probe the capture status BEFORE the launch. If the stream is
    // already Invalidated (2) here, the invalidation came from an earlier
    // call (asynchronous); if Active (1), this launch is the first to
    // observe it. Names the exact transition point.
    if (g_decode_capturing.load(std::memory_order_relaxed)) {
      hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
      hipError_t cse = hipStreamIsCapturing(stream_, &cs);
      g1_check_hip("hipStreamIsCapturing (pre-launch probe)", cse);
      if (cs != hipStreamCaptureStatusActive) {
        g1_check_status(
            "pre-launch probe: stream NOT active", 1,
            cs == hipStreamCaptureStatusInvalidated
                ? "capture ALREADY INVALIDATED before this launch"
                : "capture status unexpected",
            "pre-launch");
      }
    }
    // G1X: under decode stream capture the graph node keeps REFERENCING
    // these arg slots, which live on the caller's stack (dead by replay).
    // Copy the values (24-entry cap; kernels read only their own first
    // narg entries) into a slot-owned persistent block. Entries beyond the
    // real arg count may be unmapped stack garbage — guard the read.
    if (g_decode_capturing.load(std::memory_order_relaxed) && params &&
        !persistent && !g1x_off()) {
      constexpr int G1X_CAP = 24;
      auto blk = decode_capture_persist_alloc(2 * G1X_CAP * 8);
      auto* vals = reinterpret_cast<uint64_t*>(blk);
      auto* ptrs =
          reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(blk) +
                                  G1X_CAP * 8);
      for (int a = 0; a < G1X_CAP; ++a) {
        void* v = params[a];
        uint64_t rv = 0;
        if (v) {
          uintptr_t u = reinterpret_cast<uintptr_t>(v);
          if (u >= 0x10000000 && u < 0x10000000000 && (u % 8) == 0) {
            rv = g1x_safe_read8(v);
          }
        }
        vals[a] = rv;
        ptrs[a] = &vals[a];
      }
      params = ptrs;
    }
    hipError_t le = hipLaunchKernel(
        func, grid_dim, block_dim, params, smem_bytes, stream_);
    g1_check_hip("hipLaunchKernel (add_kernel_node_raw)", le);
    if (le != hipSuccess) {
      if (g_decode_capturing.load(std::memory_order_relaxed)) {
        // A failed launch (hipErrorStreamCaptureInvalidated) ends the
        // capture. Reset the flag so later commits are not deferred and
        // synchronize()'s completion handler can fire (otherwise the
        // worker gate + deferred commit deadlock the caller); discard the
        // invalidated capture and clear the sticky last error.
        g_decode_capturing.store(false, std::memory_order_relaxed);
        set_stream_capturing(false);
        hipGraph_t stale = nullptr;
        (void)hipStreamEndCapture(stream_, &stale);
        if (stale)
          (void)hipGraphDestroy(stale);
        (void)hipGetLastError();
      }
      check_hip_error(
          "hipLaunchKernel( func, grid_dim, block_dim, params, smem_bytes, "
          "stream_ )",
          le);
    }
    node_count_++;
    return;
  }

  hipKernelNodeParams kernel_params = {};
  kernel_params.func = func;
  kernel_params.gridDim = grid_dim;
  kernel_params.blockDim = block_dim;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  add_kernel_node_kp(kernel_params);
}

// Key the node by its kernel FUNCTION + FULL launch dims so a reused exec is
// re-pointed only onto a structurally identical graph. The grid/block PRODUCT
// collided distinct shapes (e.g. 2097152x1x1 vs 1024x2048x1); encode each dim.
std::string CommandEncoder::kernel_node_key(const hipKernelNodeParams& kp) {
  std::string key = "K";
  key += std::to_string(reinterpret_cast<std::uintptr_t>(kp.func));
  key += "_";
  key += std::to_string(kp.gridDim.x);
  key += ",";
  key += std::to_string(kp.gridDim.y);
  key += ",";
  key += std::to_string(kp.gridDim.z);
  key += "x";
  key += std::to_string(kp.blockDim.x);
  key += ",";
  key += std::to_string(kp.blockDim.y);
  key += ",";
  key += std::to_string(kp.blockDim.z);
  key += "s";
  key += std::to_string(kp.sharedMemBytes);
  return key;
}

void CommandEncoder::add_kernel_node_kp(const hipKernelNodeParams& kp) {
  // A launch with any zero grid/block dim is a no-op: hipLaunchKernel tolerates
  // it (eager path), but hipGraphAddKernelNode rejects it with "invalid
  // argument". Skip it entirely — it does no work, and skipping is consistent
  // across tokens so the decode topology stays stable for replay.
  if (kp.func == nullptr || kp.gridDim.x == 0 || kp.gridDim.y == 0 ||
      kp.gridDim.z == 0 || kp.blockDim.x == 0 || kp.blockDim.y == 0 ||
      kp.blockDim.z == 0) {
    return;
  }
  std::string key = kernel_node_key(kp);

  hipGraphNode_t node;
  hipError_t kn_err =
      hipGraphAddKernelNode(&node, build_graph_, nullptr, 0, &kp);
  if (kn_err != hipSuccess) {
    // ROCm's hipGraphAddKernelNode rejects some kernels with invalid-argument
    // (certain WMMA / single-block / JIT kernels). In replay mode, don't split
    // — CAPTURE the launch into a child graph node (CUDA's model: capture
    // records any launched kernel that the manual node API won't accept), so
    // the chunk stays ~1 fragment. Else fall back to the eager graph-split.
    (void)hipGetLastError();
    static const bool cap_reject = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    if (cap_reject && !graph_decode_mode()) {
      device_.make_current();
      hipError_t be =
          hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
      if (be == hipSuccess) {
        (void)hipLaunchKernel(
            kp.func,
            kp.gridDim,
            kp.blockDim,
            kp.kernelParams,
            kp.sharedMemBytes,
            stream_);
        hipGraph_t child = nullptr;
        hipError_t ee = hipStreamEndCapture(stream_, &child);
        if (ee == hipSuccess && child) {
          size_t nn = 0;
          (void)hipGraphGetNodes(child, nullptr, &nn);
          if (nn > 0) {
            add_child_graph_node(child, key);
            (void)hipGraphDestroy(child);
            return;
          }
          (void)hipGraphDestroy(child);
        } else if (child) {
          (void)hipGraphDestroy(child);
        }
        (void)hipGetLastError();
      }
    }
    commit();
    device_.make_current();
    (void)hipLaunchKernel(
        kp.func,
        kp.gridDim,
        kp.blockDim,
        kp.kernelParams,
        kp.sharedMemBytes,
        stream_);
    record_inline_launch();
    return;
  }
  build_nodes_.push_back(node);
  build_node_params_.push_back(kp);
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::add_module_kernel_node(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params,
    std::shared_ptr<void> args_keepalive) {
  if (!use_hip_graphs()) {
    device_.make_current();
    // Replay mode: the recorded graph node is the only execution.
    if (decode_replay_suppress() &&
        !g_decode_capturing.load(std::memory_order_relaxed)) {
      return;
    }
    // G1X: same dangling-arg problem as add_kernel_node_raw — persist the
    // arg values when the caller did not already keep them alive.
    void* g1x_params = params;
    if (g_decode_capturing.load(std::memory_order_relaxed) && params &&
        !args_keepalive && !g1x_off()) {
      constexpr int G1X_CAP = 24;
      auto blk = decode_capture_persist_alloc(2 * G1X_CAP * 8);
      auto* vals = reinterpret_cast<uint64_t*>(blk);
      auto* ptrs = reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(blk) + G1X_CAP * 8);
      for (int a = 0; a < G1X_CAP; ++a) {
        void* v = params[a];
        uint64_t rv = 0;
        if (v) {
          uintptr_t u = reinterpret_cast<uintptr_t>(v);
          if (u >= 0x10000000 && u < 0x10000000000 && (u % 8) == 0) {
            rv = g1x_safe_read8(v);
          }
        }
        vals[a] = rv;
        ptrs[a] = &vals[a];
      }
      g1x_params = ptrs;
    }
    CHECK_HIP_ERROR(hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(func),
        grid_dim.x,
        grid_dim.y,
        grid_dim.z,
        block_dim.x,
        block_dim.y,
        block_dim.z,
        smem_bytes,
        stream_,
        reinterpret_cast<void**>(g1x_params),
        nullptr));
    node_count_++;
    return;
  }
  // Graph path: the node references `params` (which point into the kept-alive
  // KernelArgs) until commit instantiates the graph. A module hipFunction_t is
  // a valid hipKernelNodeParams.func on ROCm 7.13 (see device.h note).
  if (args_keepalive) {
    graph_node_args_.push_back(std::move(args_keepalive));
  }
  add_kernel_node_raw(
      func,
      grid_dim,
      block_dim,
      smem_bytes,
      params,
      /*persistent=*/args_keepalive != nullptr ||
          g_decode_capturing.load(std::memory_order_relaxed));
}

void CommandEncoder::add_child_graph_node(
    hipGraph_t child,
    const std::string& key) {
  // hipGraphExecUpdate does NOT refresh kernel params nested inside child-graph
  // nodes (confirmed: it returns success but the child keeps stale kernargs),
  // so embedding a child node silently breaks graph reuse across tokens.
  // Flatten the child's kernels into build_graph_ as TOP-LEVEL kernel nodes via
  // the normal add path so ExecUpdate refreshes them. The child's kernelParams
  // point at the caller's arg storage (kept alive like any other op), and
  // chain-edge ordering serializes the flattened nodes correctly. Fall back to
  // embedding the child as-is only if it contains non-kernel nodes (none
  // observed in practice).
  size_t n = 0;
  (void)hipGraphGetNodes(child, nullptr, &n);
  std::vector<hipGraphNode_t> cnodes(n);
  if (n) {
    (void)hipGraphGetNodes(child, cnodes.data(), &n);
  }
  // Topologically order the child's kernels (Kahn) so chain-edge serialization
  // never places a consumer before its producer.
  size_t ne = 0;
  (void)hipGraphGetEdges(child, nullptr, nullptr, &ne);  std::vector<hipGraphNode_t> cfrom(ne), cto(ne);
  if (ne) {
    (void)hipGraphGetEdges(child, cfrom.data(), cto.data(), &ne);  }
  bool all_kernels = n > 0;
  for (size_t i = 0; i < n; i++) {
    hipGraphNodeType t;
    if (hipGraphNodeGetType(cnodes[i], &t) != hipSuccess ||
        t != hipGraphNodeTypeKernel) {
      all_kernels = false;
      break;
    }
  }
  if (all_kernels) {
    std::unordered_map<hipGraphNode_t, int> indeg;
    std::unordered_map<hipGraphNode_t, std::vector<hipGraphNode_t>> succ;
    for (auto nd : cnodes)
      indeg[nd] = 0;
    for (size_t i = 0; i < ne; i++) {
      succ[cfrom[i]].push_back(cto[i]);
      indeg[cto[i]]++;
    }
    std::vector<hipGraphNode_t> order;
    order.reserve(n);
    for (auto nd : cnodes)
      if (indeg[nd] == 0)
        order.push_back(nd);
    for (size_t h = 0; h < order.size(); h++)
      for (auto s : succ[order[h]])
        if (--indeg[s] == 0)
          order.push_back(s);
    if (order.size() == n) {
      bool ok = true;
      std::vector<hipKernelNodeParams> kps(n);
      for (size_t i = 0; i < n && ok; i++) {
        kps[i] = {};
        if (hipGraphKernelNodeGetParams(order[i], &kps[i]) != hipSuccess)
          ok = false;
      }
      if (ok) {
        for (size_t i = 0; i < n; i++) {
          add_kernel_node_kp(
              kps[i]); // preserve full params (kernelParams+extra)
        }
        return;
      }
    }
  }
  // Fallback: embed the child as-is (non-kernel nodes or topo failure).
  hipGraphNode_t node;
  CHECK_HIP_ERROR(
      hipGraphAddChildGraphNode(&node, build_graph_, nullptr, 0, child));
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::maybe_commit() {
  if (use_hip_graphs()) {
    if (needs_commit()) {
      commit();
    }
    return;
  }
  if (node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer)) {
    commit();
  }
}

bool CommandEncoder::needs_commit() {
  if (!use_hip_graphs()) {
    return node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer);
  }
  // Decode-mode: never split mid-forward — the whole single-token forward
  // becomes one graph, committed once at finalize, refreshed via ExecUpdate.
  // Decode's live intermediates are small, so the per-graph caps (which bound
  // prefill) aren't needed here.
  if (graph_decode_mode()) {
    return false;
  }
  return (node_count_ > max_ops_per_graph_) ||
      ((bytes_in_graph_ >> 20) > static_cast<size_t>(max_mb_per_graph_));
}

// --- G1 root-cause instrumentation (gated by MLX_G1_TRACE) -----------------
long g1_trace_n{0};
inline bool g1_trace_on() {
  static const bool on = std::getenv("MLX_G1_TRACE") != nullptr;
  return on;
}
inline long g1_trace_ops_limit() {
  static const long lim = [] {
    const char* e = std::getenv("MLX_G1_TRACE_OPS");
    return e ? std::atol(e) : 0;
  }();
  return lim;
}

// --- G1 first-error tracking (gated by MLX_G1_TRACE / MLX_G1_FIRST_ERR) ---
// The capture crash (hipLaunchKernel "operation failed due to a previous
// error during capture" + rocblas_status_internal_error at handle free) is a
// SYMPTOM: the first failing hip/rocBLAS call in the capture region
// invalidates the capture sequence, and every later call on the stream then
// fails with the invalidation error. This tracker names the FIRST failing
// call: every instrumented hip*/rocblas* call in the capture region reports
// its non-success return; the first one wins and is logged once with the op
// index; capture continues so the downstream symptom chain stays visible.
// The tracker resets at capture begin; a summary prints at capture end.
struct G1FirstError {
  std::atomic<bool> reported{false};
  const char* call{""};
  int code{0};
  const char* msg{""};
  long op{-1};
  const char* phase{""};
};
inline G1FirstError& g1_first_error() {
  static G1FirstError fe;
  return fe;
}
inline bool g1_fe_on() {
  static const bool on = std::getenv("MLX_G1_TRACE") != nullptr ||
      std::getenv("MLX_G1_FIRST_ERR") != nullptr;
  return on;
}
inline void g1_record_first(const char* call, int code, const char* msg,
                           const char* phase) {
  if (code == 0 || !g1_fe_on())
    return;
  auto& fe = g1_first_error();
  if (fe.reported.exchange(true))
    return;
  fe.call = call;
  fe.code = code;
  fe.msg = msg;
  fe.op = g1_trace_n;
  fe.phase = phase ? phase : "in-capture";
  fprintf(stderr,
          "[g1fe] FIRST ERROR: %s -> %s (code %d) at op %ld [%s] "
          "capturing=%d\n",
          call, msg, code, g1_trace_n, fe.phase,
          g_decode_capturing.load(std::memory_order_relaxed));
}
void g1_check_hip(const char* call, hipError_t err, const char* phase) {
  if (err == hipSuccess)
    return;
  g1_record_first(call, static_cast<int>(err), hipGetErrorString(err), phase);
}
void g1_check_rocblas(const char* call, rocblas_status st) {
  if (st == rocblas_status_success)
    return;
  g1_record_first(call, static_cast<int>(st), "rocblas_status", "in-capture");
}
void g1_check_status(const char* call, int code, const char* msg,
                    const char* phase) {
  g1_record_first(call, code, msg, phase);
}
void g1_reset_first_error() {
  g1_first_error().reported.store(false);
}
void g1_first_error_summary(const char* phase) {
  auto& fe = g1_first_error();
  if (!fe.reported.load() || !g1_fe_on())
    return;
  fprintf(stderr,
          "[g1fe] summary (%s): first error was %s (code %d, %s) at op %ld\n",
          phase, fe.call, fe.code, fe.msg, fe.op);
}

// G1f: pre-warm rocBLAS OUTSIDE the capture region. During a capture step
// is_hipblaslt_available() is false (hipBLASLt's lazy init is non-capturable),
// so the step's first GEMM (QuantizedMatmul) falls to the rocBLAS path, whose
// handle / stream / workspace / Tensile kernels are ALL lazily initialised on
// the first gemm. Any of that inside the capture region invalidates the
// capture (next launch: 901 "previous error during capture"). Force it all
// here, on the capture stream, before BeginCapture:
//   1. rocblas_create_handle (via get_rocblas_handle),
//   2. rocblas_set_stream,
//   3. one tiny bf16 + f32 GEMM (async) -> workspace alloc + kernel load.
// The hipStreamSynchronize below completes the GEMMs (doubles as a final
// drain). Failures are g1fe-logged but non-fatal: the GEMM paths have
// fallbacks and the capture can still succeed.
static void g1_prewarm_rocblas(Device& dev, hipStream_t stream) {
  dev.make_current();
  rocblas_handle handle = nullptr;
  try {
    handle = dev.get_rocblas_handle();
  } catch (...) {
    // Arch unsupported / init failed: GEMM paths fall back to kernels.
    return;
  }
  dev.set_rocblas_stream(stream);
  const int n = 16;
  const size_t hbytes = size_t(n) * n * 2; // bf16
  const size_t fbytes = size_t(n) * n * 4; // f32
  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  void* fa = nullptr;
  void* fb = nullptr;
  void* fc = nullptr;
  hipError_t m1 = hipMalloc(&a, hbytes);
  g1_check_hip("hipMalloc (g1 prewarm bf16 a)", m1, "prewarm");
  hipError_t m2 = (m1 == hipSuccess) ? hipMalloc(&b, hbytes) : hipSuccess;
  g1_check_hip("hipMalloc (g1 prewarm bf16 b)", m2, "prewarm");
  hipError_t m3 =
      (m1 == hipSuccess && m2 == hipSuccess) ? hipMalloc(&c, hbytes) : hipSuccess;
  g1_check_hip("hipMalloc (g1 prewarm bf16 c)", m3, "prewarm");
  if (m1 == hipSuccess && m2 == hipSuccess && m3 == hipSuccess) {
    (void)hipMemset(a, 0, hbytes);
    (void)hipMemset(b, 0, hbytes);
    (void)hipMemset(c, 0, hbytes);
    float alpha = 1.0f;
    float beta = 0.0f;
    // Same call shape as the qmm bf16 capture path (dequant_rocblas_gemm).
    rocblas_status st = rocblas_gemm_ex(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        n, n, n,
        &alpha,
        a, rocblas_datatype_bf16_r, n,
        b, rocblas_datatype_bf16_r, n,
        &beta,
        c, rocblas_datatype_bf16_r, n,
        c, rocblas_datatype_bf16_r, n,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0, 0);
    g1_check_rocblas("rocblas_gemm_ex (g1 prewarm bf16)", st);
  }
  // f32 (non-quantized Matmul capture path: gemm_rocblas / qmm f32).
  hipError_t f1 = (m1 == hipSuccess) ? hipMalloc(&fa, fbytes) : hipSuccess;
  g1_check_hip("hipMalloc (g1 prewarm f32 a)", f1, "prewarm");
  hipError_t f2 =
      (m1 == hipSuccess && f1 == hipSuccess) ? hipMalloc(&fb, fbytes) : hipSuccess;
  g1_check_hip("hipMalloc (g1 prewarm f32 b)", f2, "prewarm");
  hipError_t f3 =
      (m1 == hipSuccess && f1 == hipSuccess && f2 == hipSuccess)
          ? hipMalloc(&fc, fbytes)
          : hipSuccess;
  g1_check_hip("hipMalloc (g1 prewarm f32 c)", f3, "prewarm");
  if (f1 == hipSuccess && f2 == hipSuccess && f3 == hipSuccess) {
    (void)hipMemset(fa, 0, fbytes);
    (void)hipMemset(fb, 0, fbytes);
    (void)hipMemset(fc, 0, fbytes);
    float alpha = 1.0f;
    float beta = 0.0f;
    rocblas_status st = rocblas_sgemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        n, n, n,
        &alpha,
        static_cast<const float*>(fa), n,
        static_cast<const float*>(fb), n,
        &beta,
        static_cast<float*>(fc), n);
    g1_check_rocblas("rocblas_sgemm (g1 prewarm f32)", st);
  }
  hipError_t sy = G1_HL_SYNC("hipStreamSynchronize (prewarm)", hipStreamSynchronize(stream), "prewarm");
  g1_check_hip("hipStreamSynchronize (g1 prewarm)", sy, "prewarm");
  (void)hipGetLastError();
  if (a)
    (void)hipFree(a);
  if (b)
    (void)hipFree(b);
  if (c)
    (void)hipFree(c);
  if (fa)
    (void)hipFree(fa);
  if (fb)
    (void)hipFree(fb);
  if (fc)
    (void)hipFree(fc);
  // G1f: prewarm the exact GEMM shapes logged during eager steps, so the
  // capture step's GEMMs (new shapes) do not trigger a per-shape lazy init
  // (Tensile kernel load) that does a stream-ordered call from a BLAS internal
  // thread on the capture stream (invalidates the capture, delayed).
  handle = dev.get_rocblas_handle();
  dev.set_rocblas_stream(stream);
  g1_prewarm_gemm_shapes(handle, stream);
  hipError_t sync2 = hipStreamSynchronize(stream);
  g1_check_hip("hipStreamSynchronize (g1 shape prewarm)", sync2, "prewarm");
  (void)hipGetLastError();
}

// --- Full decode-step stream capture (build-once / replay) ------------------
// The manual-node chain can't capture the ~1.4k library/JIT/memset ops that go
// through launch_kernel during decode. Instead, run the whole forward with
// graphs OFF (every kernel launches on stream_), wrap it in one stream capture,
// and relaunch that single exec each token. The deterministic decode arena
// makes every token's buffer addresses identical, so the captured exec's baked
// pointers stay valid; per-token input/position/GDN-state are injected into
// those fixed buffers before each relaunch.

void* CommandEncoder::decode_capture_persist_alloc(size_t n) {
  int slot = g_decode_capture_slot.load(std::memory_order_relaxed) & 1;
  if (slot < 0 || slot > 1)
    slot = 0;
  auto blk = std::make_shared<std::vector<uint8_t>>(n + 128);
  decode_cap_keepalive_[slot].push_back(blk);
  return blk->data();
}

void CommandEncoder::decode_capture_keep(std::shared_ptr<void> p) {
  int slot = g_decode_capture_slot.load(std::memory_order_relaxed) & 1;
  if (slot < 0 || slot > 1)
    slot = 0;
  decode_cap_keepalive_[slot].push_back(std::move(p));
}

bool CommandEncoder::decode_capture_begin(int slot) {
  g_decode_capture_slot.store(slot, std::memory_order_relaxed);
  device_.make_current();
  // Defensive: if a prior capture was left open (e.g., a failed step), end
  // and discard it first, or every later op records into it and never runs.
  // ROCm 10.1's hipStreamIsCapturing is 2-arg (status out-param) only.
  hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
  hipError_t ic = hipStreamIsCapturing(stream_, &cs);
  g1_check_hip("hipStreamIsCapturing (begin, defensive)", ic, "pre-capture");
  if (ic == hipSuccess && cs != hipStreamCaptureStatusNone) {
    hipGraph_t stale = nullptr;
    hipError_t se = hipStreamEndCapture(stream_, &stale);
    g1_check_hip("hipStreamEndCapture (begin, defensive)", se, "pre-capture");
    if (stale)
      (void)hipGraphDestroy(stale);
    (void)hipGetLastError();
  }
  // Before starting the capture, wait for the worker to finish running any
  // in-flight completion handlers (buffer frees via hipFreeAsync on the worker
  // thread), then DRAIN the stream. The previous step's hipLaunchHostFunc and
  // the worker's enqueued hipFreeAsync are still in flight on the GPU when
  // wait_idle returns (wait_idle only waits for the worker task queue, not
  // the stream). On ROCm a stream-ordered free enqueued from another thread
  // that completes on the GPU during a thread-local capture invalidates the
  // capture (the next launch fails with hipErrorStreamCaptureInvalidated,
  // 901). Two drain rounds: sync fires the host func (which re-signals the
  // worker, which enqueues new frees), so sync again to drain those.
  if (worker_) {
    worker_->wait_idle();
  }
  hipError_t s1 = hipStreamSynchronize(stream_);
  g1_check_hip("hipStreamSynchronize (pre-capture drain 1)", s1, "pre-capture");
  if (worker_) {
    worker_->wait_idle();
  }
  hipError_t s2 = hipStreamSynchronize(stream_);
  g1_check_hip("hipStreamSynchronize (pre-capture drain 2)", s2, "pre-capture");
  // G1f: pre-warm rocBLAS (handle/stream/workspace/kernels) BEFORE the
  // capture — see g1_prewarm_rocblas. Fresh first-error tracker so a
  // prewarm failure is attributed to this capture, not the previous one.
  if (g1_fe_on())
    g1_reset_first_error();
  g1_prewarm_rocblas(device_, stream_);
  g_decode_capturing.store(true, std::memory_order_relaxed);
  // G1f: mirror the capture flag into stream_capturing() — is_hipblaslt_available()
  // consults it to force the capture-safe rocBLAS path during capture (the flag
  // was previously never set, so the capture step's first GEMM could hit
  // hipBLASLt's non-capturable lazy init).
  set_stream_capturing(true);
  hipError_t e =
      hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
  g1_check_hip("hipStreamBeginCapture", e, "begin");
  if (e != hipSuccess) {
    g_decode_capturing.store(false, std::memory_order_relaxed);
    set_stream_capturing(false);
    (void)hipGetLastError();
    static const bool dbg = std::getenv("MLX_PURE_DEBUG") != nullptr;
    if (dbg)
      fprintf(stderr, "[cap] BeginCapture failed: %s\n", hipGetErrorString(e));
    return false;
  }
  // G1: fresh first-error tracker for this capture region.
  if (g1_fe_on())
    g1_reset_first_error();
  if (g1_trace_on()) {
    g1_trace_n = 0;
    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    (void)hipStreamIsCapturing(stream_, &st);
    fprintf(stderr,
            "[g1t] begin: stream=%p status_after_begin=%d\n",
            static_cast<void*>(stream_),
            static_cast<int>(st));
  }
  return true;
}

bool CommandEncoder::decode_capture_end_record(int slot) {
  device_.make_current();
  slot &= 1;
  static const bool dbg = std::getenv("MLX_PURE_DEBUG") != nullptr;
  if (g1_trace_on()) {
    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    hipError_t es = hipStreamIsCapturing(stream_, &st);
    g1_check_hip("hipStreamIsCapturing (end_record, pre)", es, "end");
    auto& al = rocm::allocator();
    fprintf(stderr,
            "[g1t] end_record: stream=%p status_before_end=%d ops_traced=%ld "
            "arena_off=%zu arena_hw=%zu arena_cap=%zu arena_ovf=%d\n",
            static_cast<void*>(stream_),
            static_cast<int>(st),
            g1_trace_n,
            al.decode_arena_offset(),
            al.decode_arena_high_water(),
            al.decode_arena_capacity(),
            al.decode_arena_overflowed() ? 1 : 0);
  }
  hipGraph_t g = nullptr;
  hipError_t ee = hipStreamEndCapture(stream_, &g);
  g1_check_hip("hipStreamEndCapture", ee, "end");
  g_decode_capturing.store(false, std::memory_order_relaxed);
  set_stream_capturing(false);
  // Run the completion handlers (buffer frees) that were deferred during the
  // capture. Doing this now (after the stream is no longer capturing) avoids
  // the cross-thread hipFreeAsync invalidating the capture.
  if (worker_) {
    worker_->signal_now();
  }
  if (ee != hipSuccess) {
    // Driver-dependent: a failed EndCapture may leave the stream capturing.
    // Release it so later ops execute instead of recording forever.
    hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
    hipError_t cs2 = hipStreamIsCapturing(stream_, &cs);
    g1_check_hip("hipStreamIsCapturing (end_record, defensive)", cs2, "end");
    if (cs2 == hipSuccess && cs != hipStreamCaptureStatusNone) {
      hipGraph_t stale = nullptr;
      hipError_t se = hipStreamEndCapture(stream_, &stale);
      g1_check_hip("hipStreamEndCapture (end_record, defensive)", se, "end");
      if (stale)
        (void)hipGraphDestroy(stale);
      (void)hipGetLastError();
    }
    (void)hipGetLastError();
    if (dbg)
      fprintf(stderr, "[cap] EndCapture failed: %s\n", hipGetErrorString(ee));
    g1_first_error_summary("end_record:EndCapture-fail");
    return false;
  }
  if (!g) {
    (void)hipGetLastError();
    g1_first_error_summary("end_record:empty-graph");
    return false;
  }
  size_t nn = 0;
  hipError_t gn = hipGraphGetNodes(g, nullptr, &nn);
  g1_check_hip("hipGraphGetNodes", gn, "end");

  // MLX_G1_NODEDUMP: dump every node's baked device pointers so a faulting
  // address can be attributed to a recorded op.
  if (std::getenv("MLX_G1_NODEDUMP")) {
    auto& al = rocm::allocator();
    void* ab = al.decode_arena_base_ptr();
    fprintf(stderr,
            "[nd] arena base=%p cap=%zu hw=%zu off=%zu\n",
            ab, al.decode_arena_capacity(), al.decode_arena_high_water(),
            al.decode_arena_offset());
    int ndump_n = 8;
    if (const char* en = std::getenv("MLX_G1_NODUMP_N"))
      ndump_n = std::atoi(en);
    std::vector<hipGraphNode_t> nodes(nn);
    if (hipGraphGetNodes(g, nodes.data(), &nn) == hipSuccess) {
      for (size_t i = 0; i < nn && i < (size_t)ndump_n; ++i) {
        hipGraphNodeType ty;
        if (hipGraphNodeGetType(nodes[i], &ty) != hipSuccess)
          continue;
        if (ty == hipGraphNodeTypeKernel) {
          hipKernelNodeParams kp;
          if (hipGraphKernelNodeGetParams(nodes[i], &kp) != hipSuccess)
            continue;
          fprintf(stderr,
                  "[nd] K%zu func=%p grid=[%u,%u,%u]\n",
                  i, static_cast<void*>(kp.func), kp.gridDim.x, kp.gridDim.y,
                  kp.gridDim.z);
          // Name the kernel so the faulting node can be found without
          // guessing arg counts.
          char buf[256];
          int nl = snprintf(
              buf, sizeof(buf), "%p", kp.func);
          (void)nl;
          Dl_info di;
          bool named = (dladdr(kp.func, &di) != 0);
          const char* nm = named ? di.dli_sname : "?";
          bool is_gather =
              named && strstr(nm, "gather_rows_kernel") != nullptr;
          if (is_gather) {
            fprintf(stderr, "[nd] G%zu %s\n", i, nm);
          } else {
            fprintf(stderr, "[nd] K%zu %s\n", i, nm);
          }
          if (kp.kernelParams) {
            auto** kparr = static_cast<void**>(kp.kernelParams);
            int maxa = 8;
            for (int a = 0; a < maxa; ++a) {
              void* v = kparr[a];
              if (!v)
                continue;
              bool safe =
                  reinterpret_cast<uintptr_t>(v) >= 0x10000000 &&
                  reinterpret_cast<uintptr_t>(v) < 0x10000000000 &&
                  reinterpret_cast<uintptr_t>(v) % 8 == 0;
              if (safe) {
                fprintf(stderr,
                        "[nd]  kp[%d]=%p u64=%016llx\n",
                        a, v, (unsigned long long)g1x_safe_read8(v));
              } else {
                fprintf(stderr, "[nd]  kp[%d]=%p (oob)\n", a, v);
              }
            }
            if (named && strstr(nm, "copy_g_byval") != nullptr) {
              // args 3 = shape (40B), 4 = strides (80B): dump fully.
              for (int a = 3; a <= 4; ++a) {
                void* v = kparr[a];
                if (!v)
                  continue;
                fprintf(stderr, "[nd]  struct arg %d: ", a);
                auto* bytes = reinterpret_cast<uint8_t*>(v);
                size_t nbytes = (a == 3) ? 40 : 80;
                for (size_t b = 0; b < nbytes; b += 8) {
                  fprintf(stderr, "%016llx ",
                          (unsigned long long)g1x_safe_read8(
                              reinterpret_cast<void*>(
                                  reinterpret_cast<uint8_t*>(v) + b)));
                }
                fprintf(stderr, "\n");
              }
            }
            if (is_gather) {
              // args 1..5 of gather_rows_kernel, guarded read (the arg
              // value buffers are host-side; a raw deref faulted the host
              // on one vGPU run).
              for (int a = 1; a < 6; ++a) {
                void* v = kparr[a];
                if (!v)
                  continue;
                fprintf(stderr,
                        "[nd]  kp[%d]=%p u64=%016llx\n",
                        a, v, (unsigned long long)g1x_safe_read8(v));
              }
            }
          }
        } else if (ty == hipGraphNodeTypeMemcpy) {
          hipMemcpy3DParms mp;
          if (hipGraphMemcpyNodeGetParams(nodes[i], &mp) == hipSuccess) {
            fprintf(stderr,
                    "[nd] M%zu %p(%zu) -> %p(%zu) kind=%d\n",
                    i, mp.srcPtr.ptr, mp.srcPtr.xsize, mp.dstPtr.ptr,
                    mp.dstPtr.xsize, (int)mp.kind);
          }
        }
      }
    }
  }
  // VA-range map: attribute every distinct device pointer in the graph to a
  // [base, base+size) allocation so a faulting address can be located.
  if (std::getenv("MLX_G1_NODEDUMP")) {
    auto& al = rocm::allocator();
    std::vector<void*> ptrs;
    if (al.decode_arena_base_ptr())
      ptrs.push_back(al.decode_arena_base_ptr());
    size_t nn2 = nn;
    std::vector<hipGraphNode_t> nodes2(nn2);
    if (hipGraphGetNodes(g, nodes2.data(), &nn2) == hipSuccess) {
      for (size_t i = 0; i < nn2; ++i) {
        hipGraphNodeType ty;
        if (hipGraphNodeGetType(nodes2[i], &ty) != hipSuccess)
          continue;
        if (ty == hipGraphNodeTypeKernel) {
          hipKernelNodeParams kp;
          if (hipGraphKernelNodeGetParams(nodes2[i], &kp) != hipSuccess)
            continue;
          if (!kp.kernelParams)
            continue;
          auto** kparr = static_cast<void**>(kp.kernelParams);
          for (int a = 0; a < 6; ++a) {
            void* v = kparr[a];
            if (v && (reinterpret_cast<uintptr_t>(v) < 0x10000000 ||
                     reinterpret_cast<uintptr_t>(v) >= 0x10000000000)) {
              v = nullptr;
            }
            if (!v)
              continue;
            uint64_t val = 0;
            if (reinterpret_cast<uintptr_t>(v) >= 0x100000000000ULL) {
              val = *(uint64_t*)v;
            }
            if (val > 0x100000000000ULL) {
              bool seen = false;
              for (auto& e : ptrs) {
                if (e == (void*)val) {
                  seen = true;
                  break;
                }
              }
              if (!seen)
                ptrs.push_back((void*)val);
            }
          }
        } else if (ty == hipGraphNodeTypeMemcpy) {
          hipMemcpy3DParms mp;
          if (hipGraphMemcpyNodeGetParams(nodes2[i], &mp) == hipSuccess) {
            if ((uint64_t)mp.srcPtr.ptr > 0x100000000000ULL)
              ptrs.push_back(mp.srcPtr.ptr);
            if ((uint64_t)mp.dstPtr.ptr > 0x100000000000ULL)
              ptrs.push_back(mp.dstPtr.ptr);
          }
        }
      }
    }
    for (auto* v : ptrs) {
      void* base = nullptr;
      size_t sz = 0;
      hipError_t e = hipMemGetAddressRange(&base, &sz, v);
      fprintf(stderr,
              "[ndr] %p range=[%p, +%zu) err=%d\n",
              v, base, sz, (int)e);
    }
  }
  if (dbg) {
    fprintf(stderr, "[cap] captured %zu nodes\n", nn);
    std::vector<hipGraphNode_t> nodes(nn);
    if (hipGraphGetNodes(g, nodes.data(), &nn) == hipSuccess) {
      int n_kernel = 0, n_memcpy = 0, n_memset = 0, n_event = 0, n_other = 0;
      for (auto& nd : nodes) {
        hipGraphNodeType ty;
        if (hipGraphNodeGetType(nd, &ty) != hipSuccess) {
          n_other++;
          continue;
        }
        switch (ty) {
          case hipGraphNodeTypeKernel:
            n_kernel++;
            break;
          case hipGraphNodeTypeMemcpy:
            n_memcpy++;
            break;
          case hipGraphNodeTypeMemset:
            n_memset++;
            break;
          case hipGraphNodeTypeEventRecord:
          case hipGraphNodeTypeWaitEvent:
            n_event++;
            break;
          default:
            n_other++;
            break;
        }
      }
      fprintf(
          stderr,
          "[cap] node types: kernel=%d memcpy=%d memset=%d event=%d other=%d\n",
          n_kernel,
          n_memcpy,
          n_memset,
          n_event,
          n_other);
    }
  }
  if (nn == 0) {
    (void)hipGraphDestroy(g);
    g1_first_error_summary("end_record:zero-nodes");
    return false;
  }
  // G1H2D: record + neutralize the in-graph H2D staging nodes BEFORE the
  // exec is (re)instantiated, so the instantiated exec carries them.
  decode_capture_record_neutralize_h2d(slot, g);
  // Refresh the slot's exec in place (hipGraphExecUpdate) when one exists —
  // the decode topology is stable across tokens, so the update succeeds and
  // the exec's baked pointers (arena buffers) stay valid. Fall back to a
  // fresh instantiate when there is no exec yet or the update fails (shape
  // change, node-count change).
  hipGraphExec_t exec = nullptr;
  bool refreshed = false;
  if (decode_cap_exec_[slot]) {
    hipGraphExecUpdateResult ur;
    hipGraphNode_t en;
    hipError_t ue = hipGraphExecUpdate(decode_cap_exec_[slot], g, &en, &ur);
    g1_check_hip("hipGraphExecUpdate", ue, "refresh");
    if (ue == hipSuccess && ur == hipGraphExecUpdateSuccess) {
      refreshed = true;
      // G1S: probe whether the updated exec staged the new values (the
      // recapture's own launch is the one that faults).
      g1s_arm(decode_cap_exec_[slot], g);
    } else {
      (void)hipGetLastError();
    }
  }
  if (!refreshed) {
    hipError_t ie = hipGraphInstantiate(&exec, g, nullptr, nullptr, 0);
    g1_check_hip("hipGraphInstantiate", ie, "instantiate");
    if (ie != hipSuccess) {
      if (dbg)
        fprintf(stderr, "[cap] Instantiate failed: %s\n", hipGetErrorString(ie));
      (void)hipGraphDestroy(g);
      (void)hipGetLastError();
      g1_first_error_summary("end_record:instantiate-fail");
      return false;
    }
    if (decode_cap_exec_[slot])
      (void)hipGraphExecDestroy(decode_cap_exec_[slot]);
    if (decode_cap_graph_[slot])
      (void)hipGraphDestroy(decode_cap_graph_[slot]);
    decode_cap_graph_[slot] = g;
    decode_cap_exec_[slot] = exec;
  } else {
    // The exec now references its own copy of the nodes; keep the source
    // graph alive for the slot's lifetime (same as the manual path).
    if (decode_cap_graph_[slot])
      (void)hipGraphDestroy(decode_cap_graph_[slot]);
    decode_cap_graph_[slot] = g;
  }
  // G1B: bisection — with MLX_G1_BISECT=N, re-point every kernel node
  // after the first N (hipGraphGetNodes order) at a no-op kernel so a
  // minimal faulting node range can be located. Diagnostic only.
  if (const char* be = std::getenv("MLX_G1_BISECT"); be) {
    int keep = std::atoi(be);
    if (keep > 0 && (size_t)keep < nn) {
      size_t qn = nn;
      std::vector<hipGraphNode_t> bn(nn);
      if (hipGraphGetNodes(g, bn.data(), &qn) == hipSuccess) {
        int neutralized = 0;
        // G1B2: node-type histogram (memset nodes are invisible to [nd]
        // dumps and G1B neutralized kernels only).
        int nt_k = 0, nt_m = 0, nt_ms = 0, nt_o = 0;
        for (size_t i = 0; i < qn; i++) {
          hipGraphNodeType ty = (hipGraphNodeType)0;
          if (hipGraphNodeGetType(bn[i], &ty) != hipSuccess)
            continue;
          if (ty == hipGraphNodeTypeKernel) nt_k++;
          else if (ty == hipGraphNodeTypeMemcpy) nt_m++;
          else if (ty == hipGraphNodeTypeMemset) nt_ms++;
          else nt_o++;
        }
        static int g1b2_prints = 0;
        if (g1b2_prints++ < 12)
          fprintf(stderr,
                  "[g1b] types: kernel=%d memcpy=%d memset=%d other=%d "
                  "ms_off=%d\n",
                  nt_k, nt_m, nt_ms, nt_o,
                  (int)(std::getenv("MLX_G1_BISECT_MS") != nullptr));
        fflush(stderr);
        if (const char* bms = std::getenv("MLX_G1_BISECT_MS");
            bms && bms[0] == '1') {
          for (size_t i = 0; i < qn; i++) {
            hipGraphNode_t nd = bn[i];
            hipGraphNodeType ty = (hipGraphNodeType)0;
            if (hipGraphNodeGetType(nd, &ty) != hipSuccess ||
                ty != hipGraphNodeTypeMemset)
              continue;
            hipMemsetParams mp = {};
            if (hipGraphMemsetNodeGetParams(nd, &mp) == hipSuccess) {
              mp.width = 1;
              mp.value = 0;
              (void)hipGraphExecMemsetNodeSetParams(decode_cap_exec_[slot],
                                                  nd, &mp);
            } else {
              (void)hipGetLastError();
            }
          }
        }
        for (size_t i = qn; i > (size_t)keep; i--) {
          hipGraphNode_t nd = bn[i - 1];
          hipGraphNodeType ty;
          if (hipGraphNodeGetType(nd, &ty) != hipSuccess ||
              ty != hipGraphNodeTypeKernel) {
            continue;
          }
          hipKernelNodeParams kp;
          if (hipGraphKernelNodeGetParams(nd, &kp) != hipSuccess) {
            (void)hipGetLastError();
            continue;
          }
          kp.func = (void*)g1b_noop;
          kp.gridDim = dim3(1, 1, 1);
          kp.blockDim = dim3(1, 1, 1);
          kp.sharedMemBytes = 0;
          (void)hipGraphExecKernelNodeSetParams(decode_cap_exec_[slot], nd,
                                              &kp);
          neutralized++;
        }
        fprintf(stderr,
                "[g1b] bisect: keeping first %d of %zu nodes, neutralized "
                "%d\n",
                keep,
                qn,
                neutralized);
        fflush(stderr);
      }
    }
  }
  // G1E: read back the exec's staged arg values right before the
  // first launch (end of record) — a mismatch vs the source graph's
  // values (above, [nd]) is a stale ExecUpdate/instantiate packet.
  if (std::getenv("MLX_G1_EXECDUMP") || std::getenv("MLX_G1_NODEDUMP"))
    g1e_source_dump(g, "endrec");
  // G1M: node-type histogram + memcpy-node params of the captured graph
  // (diagnostic: are the slot H2D memcpy nodes in the graph, and where
  // do they point?).
  if (std::getenv("MLX_G1_MEMDUMP")) {
    size_t qn = nn;
    std::vector<hipGraphNode_t> mn(nn);
    if (hipGraphGetNodes(g, mn.data(), &qn) == hipSuccess) {
      long mcn = 0, kcn = 0, mscn = 0, ocn = 0;
      for (size_t i = 0; i < qn; ++i) {
        hipGraphNodeType ty = (hipGraphNodeType)0;
        if (hipGraphNodeGetType(mn[i], &ty) != hipSuccess)
          continue;
        if (ty == hipGraphNodeTypeMemcpy) {
          ++mcn;
          // The memcpy param dump (with the original, pre-neutralization
          // src/dst/bytes) happens in decode_capture_record_neutralize_h2d
          // (before the exec is instantiated); here we only count the node
          // for the histogram (the params are 0 after neutralization).
          hipMemcpy3DParms mp;
          if (false && hipGraphMemcpyNodeGetParams(mn[i], &mp) == hipSuccess) {
            // Pointers can be host (H2D src) or device; tag each so a
            // (nil) src vs a real one is visible, and name the node's
            // predecessor kernel so the enqueuing op is identifiable.
            auto ptag = [](const void* p) -> const char* {
              if (!p)
                return "nil";
              hipPointerAttribute_t a;
              if (hipPointerGetAttributes(&a, const_cast<void*>(p)) !=
                  hipSuccess) {
                (void)hipGetLastError();
                return "?";
              }
              switch (a.type) {
                case hipMemoryTypeHost:
                  return "h";
                case hipMemoryTypeDevice:
                  return "d";
                case hipMemoryTypeManaged:
                  return "m";
                default:
                  return "?";
              }
            };
            size_t bytes = (size_t)mp.extent.width * mp.extent.height *
                          mp.extent.depth;
            const void* sp = mp.srcPtr.ptr;
            void* dp = mp.dstPtr.ptr;
            hipGraphNode_t dep = nullptr;
            size_t ndp = 0;
            const char* pname = "-";
            if (hipGraphNodeGetDependencies(mn[i], &dep, &ndp) == hipSuccess &&
                ndp > 0) {
              hipGraphNodeType dty = (hipGraphNodeType)0;
              if (hipGraphNodeGetType(dep, &dty) == hipSuccess &&
                  dty == hipGraphNodeTypeKernel) {
                hipKernelNodeParams kp;
                if (hipGraphKernelNodeGetParams(dep, &kp) == hipSuccess &&
                    kp.func) {
                  Dl_info di;
                  if (dladdr(kp.func, &di) && di.dli_sname)
                    pname = di.dli_sname;
                }
              }
            }
            fprintf(stderr,
                    "[g1m] n%zu kind=%d src=%p(%s) dst=%p(%s) bytes=%zu "
                    "pred=%s\n",
                    i, (int)mp.kind, sp, ptag(sp), dp, ptag(dp), bytes, pname);
            fflush(stderr);
          }
        } else if (ty == hipGraphNodeTypeKernel) {
          ++kcn;
          // Full sequence dump: name every kernel node so the memcpy
          // nodes' neighbours (the enqueueing op) are identifiable.
          hipKernelNodeParams kp;
          if (hipGraphKernelNodeGetParams(mn[i], &kp) == hipSuccess &&
              kp.func) {
            Dl_info di;
            const char* kn = "?";
            if (dladdr(kp.func, &di) && di.dli_sname)
              kn = di.dli_sname;
            int status = 0;
            char* dm = abi::__cxa_demangle(kn, nullptr, nullptr, &status);
            const char* knf = (status == 0 && dm) ? dm : kn;
            fprintf(stderr, "[g1m] n%zu K %s\n", i, knf);
            fflush(stderr);
            if (dm)
              free(dm);
          }
        } else if (ty == hipGraphNodeTypeMemset) {
          ++mscn;
        } else {
          ++ocn;
        }
      }
      fprintf(stderr,
              "[g1m] nodes: memcpy=%ld kernel=%ld memset=%ld other=%ld "
              "total=%zu\n",
              mcn, kcn, mscn, ocn, qn);
      fflush(stderr);
    }
  }
  // G1P: slot readback happens AFTER the pre-launch slot write (below),
  // just before hipGraphLaunch, so it sees the value the graph kernels
  // will actually read.
  // G1K: first kernel node's dyn-offset arg values (diagnostic: where does
  // the kernel's dyn offset pointer argument point?).
  if (std::getenv("MLX_G1_MEMDUMP")) {
    size_t qn = nn;
    std::vector<hipGraphNode_t> mn(nn);
    if (hipGraphGetNodes(g, mn.data(), &qn) == hipSuccess) {
      for (size_t i = 0; i < qn; ++i) {
        hipGraphNodeType ty = (hipGraphNodeType)0;
        if (hipGraphNodeGetType(mn[i], &ty) != hipSuccess ||
            ty != hipGraphNodeTypeKernel)
          continue;
        hipKernelNodeParams kp;
        if (hipGraphKernelNodeGetParams(mn[i], &kp) != hipSuccess)
          continue;
        auto** ka = static_cast<void**>(kp.kernelParams);
        if (!ka)
          break;
        auto read_at = [](void* p) -> uint64_t {
          uintptr_t u = reinterpret_cast<uintptr_t>(p);
          if (u < 0x1000 || (u & 7))
            return 0;
          uintptr_t base = u & ~4095ULL;
          bool second = (u - base) + 8 > 4095;
          unsigned char pres[2] = {1, 1};
          if (syscall(SYS_mincore, reinterpret_cast<void*>(base),
                     second ? 8192 : 4096,
                     pres) != 0)
            return 0;
          if (pres[0] == 0 || (second && pres[1] == 0))
            return 0;
          return *reinterpret_cast<uint64_t*>(p);
        };
        // The dynamic-copy node: args 12/13 are pointers to the
        // dyn-offset slot buffers (device VAs in [2^57, 2^61)).
        uint64_t v12 = read_at(ka[12]);
        uint64_t v13 = read_at(ka[13]);
        auto dev_va = [](uint64_t v) {
          return v >= 0x08000000000000ULL && v < 0x8000000000000000ULL;
        };
        if (!dev_va(v12) || !dev_va(v13))
          continue;
        uint64_t o12 = read_at((void*)v12);
        uint64_t o13 = read_at((void*)v13);
        fprintf(stderr,
                "[g1k] n%zu grid=%u,%u,%u a12ptr=%llx a12off=%llx "
                "a13ptr=%llx a13off=%llx\n",
                i, kp.gridDim.x, kp.gridDim.y, kp.gridDim.z,
                (unsigned long long)v12, (unsigned long long)o12,
                (unsigned long long)v13, (unsigned long long)o13);
        fflush(stderr);
        break;
      }
    }
  }
  // Stream capture records WITHOUT executing — run the exec once to actually
  // compute the record token's logits/state.
  if (std::getenv("MLX_G1_NO_LAUNCH")) {
    fprintf(stderr, "[g1n] NO_LAUNCH: skipping hipGraphLaunch (slot %d)\n", slot);
    fflush(stderr);
  } else {
    g1_prelaunch_slot_h2d(static_cast<hipStream_t>(stream_));
    g1_prelaunch_recorded_h2d(slot, static_cast<hipStream_t>(stream_));
    // G1P: slot readback AFTER the pre-launch slot write, before launch.
    if (std::getenv("MLX_G1_MEMDUMP")) {
      (void)hipStreamSynchronize(static_cast<hipStream_t>(stream_));
      auto& alr = rocm::allocator();
      for (int i = 0; i < 4; ++i) {
        auto sl = alr.decode_value_slot(i);
        if (sl.data) {
          int64_t dv = -1;
          (void)hipMemcpy(&dv, sl.data, 8, hipMemcpyDeviceToHost);
          fprintf(stderr,
                  "[g1p] slot%d host=%p dev=%p hostval=%lld devval=%lld\n",
                  i, (void*)sl.host, (void*)sl.data,
                  (long long)(sl.host ? *sl.host : -9), (long long)dv);
          fflush(stderr);
        }
      }
      auto zz = alr.decode_zero_slot();
      if (zz.data) {
        int64_t dv = -9;
        (void)hipMemcpy(&dv, zz.data, 8, hipMemcpyDeviceToHost);
        fprintf(stderr, "[g1p] zero dev=%lld\n", (long long)dv);
        fflush(stderr);
      }
    }
    hipError_t lt = hipGraphLaunch(decode_cap_exec_[slot], stream_);
    g1_check_hip("hipGraphLaunch (end_record)", lt, "launch");
    if (lt != hipSuccess)
      check_hip_error("hipGraphLaunch(decode_cap_exec_[slot], stream_)", lt);
    if (std::getenv("MLX_G1_NO_COMMIT")) {
      fprintf(stderr,
              "[g1n] NO_COMMIT: skipping worker commit (slot %d)\n",
              slot);
      fflush(stderr);
    } else {
      worker_->commit(stream_);
    }
    // G1S: after the (re)capture's own launch, check who the exec wrote.
    g1s_check(static_cast<hipStream_t>(stream_));
  }
  g1_first_error_summary("end_record:success");
  return true;
}

bool CommandEncoder::decode_capture_replay(int slot) {
  slot &= 1;
  if (!decode_cap_exec_[slot])
    return false;
  device_.make_current();
  // G1E: read back the exec's staged arg values before every replay
  // launch (stale-packet check across recaptures).
  if (std::getenv("MLX_G1_EXECDUMP") || std::getenv("MLX_G1_NODEDUMP"))
    g1e_source_dump(decode_cap_graph_[slot], "replay");
  g1_prelaunch_slot_h2d(static_cast<hipStream_t>(stream_));
  g1_prelaunch_recorded_h2d(slot, static_cast<hipStream_t>(stream_));
  CHECK_HIP_ERROR(hipGraphLaunch(decode_cap_exec_[slot], stream_));
  worker_->commit(stream_);
  return true;
}

void CommandEncoder::decode_capture_destroy() {
  for (int s = 0; s < 2; ++s) {
    if (decode_cap_exec_[s]) {
      (void)hipGraphExecDestroy(decode_cap_exec_[s]);      decode_cap_exec_[s] = nullptr;
      decode_cap_keepalive_[s].clear();
    }
    if (decode_cap_graph_[s]) {
      (void)hipGraphDestroy(decode_cap_graph_[s]);
      decode_cap_graph_[s] = nullptr;
    }
  }
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  temporary_ptrs_.clear();

  if (use_hip_graphs() && node_count_ > 0) {
    if (!from_nodes_.empty()) {
      hipError_t de = hipGraphAddDependencies(
          build_graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          from_nodes_.size());
      g1_check_hip("hipGraphAddDependencies (commit)", de);
      if (de != hipSuccess)
        check_hip_error("hipGraphAddDependencies(...)", de);
    }
    device_.make_current();

    hipGraphExec_t graph_exec = nullptr;
    bool build_graph_adopted =
        false; // build_graph_ became a slot's source graph
    std::shared_ptr<std::atomic<int>> inflight;
    ExecSlot* used_slot = nullptr;
    // Reuse a drained exec (inflight==0) for an identical kernel sequence. In
    // decode-mode the whole-forward graph recurs every token, so refresh the
    // cached exec's params in one hipGraphExecUpdate; otherwise (or if
    // ExecUpdate fails) reinstantiate into the slot. The slot owns the source
    // graph + arg Packs for the exec's life (CLR stores kernelParams by
    // pointer).
    static const bool prefill_replay_cu = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    const bool use_execupdate = graph_decode_mode() || prefill_replay_cu;
    auto& pool = exec_pool_[graph_nodes_key_ + ":" + graph_deps_key_];
    // For the stable decode topology, grow the pool to N execs (skip reuse
    // until then) so replay always finds a drained slot despite completion-flag
    // lag.
    static const size_t replay_slots = [] {
      const char* e = std::getenv("MLX_GRAPH_REPLAY_SLOTS");
      return e ? std::max<size_t>(2, std::atoi(e)) : 4;
    }();
    const std::string& grow_key =
        graph_decode_mode() ? decode_key_ : prefill_key_;
    const bool force_grow = use_execupdate && !grow_key.empty() &&
        (graph_nodes_key_ + ":" + graph_deps_key_) == grow_key &&
        pool.size() < replay_slots;
    for (auto& slot : pool) {
      if (force_grow)
        break;
      if (slot.inflight->load(std::memory_order_acquire) != 0) {
        continue;
      }
      bool refreshed = false;
      if (use_execupdate) {
        hipGraphExecUpdateResult ur;
        hipGraphNode_t en;
        if (hipGraphExecUpdate(slot.exec, build_graph_, &en, &ur) ==
                hipSuccess &&
            ur == hipGraphExecUpdateSuccess) {
          refreshed = true;
          build_graph_adopted = true; // exec now bound to build_graph_'s nodes
          slot.src_nodes = build_nodes_;
          if (slot.source_graph)
            (void)hipGraphDestroy(slot.source_graph);
          slot.source_graph = build_graph_;
        } else {
          (void)hipGetLastError();
        }
      }
      if (refreshed) {
        graph_exec = slot.exec;
      } else {
        // Reinstantiate into this slot from the new build graph.
        (void)hipGraphExecDestroy(slot.exec);        CHECK_HIP_ERROR(
            hipGraphInstantiate(&slot.exec, build_graph_, nullptr, nullptr, 0));
        graph_exec = slot.exec;
        slot.src_nodes = build_nodes_;
        if (slot.source_graph)
          (void)hipGraphDestroy(slot.source_graph);
        slot.source_graph = build_graph_;
        build_graph_adopted = true;
      }
      inflight = slot.inflight;
      used_slot = &slot;
      break;
    }
    if (graph_exec == nullptr) {
      CHECK_HIP_ERROR(
          hipGraphInstantiate(&graph_exec, build_graph_, nullptr, nullptr, 0));
      inflight = std::make_shared<std::atomic<int>>(0);
      pool.push_back({graph_exec, inflight, build_graph_, {}, build_nodes_});
      used_slot = &pool.back();
      build_graph_adopted = true;
    }
    inflight->store(1, std::memory_order_release);

    // Reclaim this chunk's deferred-free buffers once it has drained, bounding
    // graph-mode memory to a sliding window of chunks instead of a whole
    // forward (which OOMs a 32GB card). Free with a generation LAG so a buffer
    // is only released after the next few chunks have also launched — covers
    // cross-chunk / in-place references that a lag-0 free races
    // (use-after-free). Tunable via MLX_GRAPH_FREE_LAG. Opt-in
    // (MLX_GRAPH_FREE_LAG): per-chunk reclaim is currently racy (frees a buffer
    // the next chunk still references → UAF), so OFF by default — frees flush
    // safely at the per-token synchronize. The real fix is 100% buffer reuse
    // (deterministic per-forward addresses), not freeing.
    static const long free_lag = [] {
      const char* e = std::getenv("MLX_GRAPH_FREE_LAG");
      return e ? std::atol(e) : -1;
    }();
    uint64_t my_gen = graph_current_gen();
    graph_advance_gen();
    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));
    if (free_lag >= 0 && static_cast<long>(my_gen) > free_lag) {
      uint64_t fg = my_gen - free_lag;
      add_completed_handler([fg]() { free_graph_generation(fg); });
    }
    // Reclaim this chunk's stream-ordered POOL buffers now, via hipFreeAsync
    // queued right after the launch on stream_ (retires after the graph; no
    // blocking drain). This is the common case on the discrete pool and keeps
    // the peak bounded without stalling the pipeline.
    free_graph_generation_async(my_gen);
    // Backstop ONLY for the non-stream-ordered remainder (unified/slab buffers,
    // rare on the discrete path): if that residual backlog still exceeds a cap,
    // drain + flush. MLX_GRAPH_DEFER_MAX_MB (default 2048; 0 disables).
    static const size_t defer_cap = [] {
      const char* e = std::getenv("MLX_GRAPH_DEFER_MAX_MB");
      return static_cast<size_t>(e ? std::atoll(e) : 2048) << 20;
    }();
    if (defer_cap && graph_deferred_bytes() > defer_cap) {
      (void)hipStreamSynchronize(stream_);
      flush_graph_deferred_frees();
    }
    // The completion handler fires after the stream drains this commit's
    // launch; clear inflight so the slot can be reused next token.
    add_completed_handler(
        [inflight]() { inflight->store(0, std::memory_order_release); });

    // Lock in the stable decode topology key once it recurs on two consecutive
    // tokens (the first decode token after prefill differs). Thereafter,
    // matching tokens replay (build-once) — re-point a drained slot's params,
    // no rebuild.
    if (graph_decode_mode() && used_slot && decode_key_.empty()) {
      std::string fk = graph_nodes_key_ + ":" + graph_deps_key_;
      if (fk == pending_decode_key_) {
        decode_key_ = fk;
      } else {
        pending_decode_key_ = std::move(fk);
      }
    }
    // Same for prefill: lock the full-size-chunk topology once it recurs (chunk
    // 2 matches chunk 1), then later chunks build-once/replay it.
    if (!graph_decode_mode() && used_slot && prefill_key_.empty()) {
      std::string fk = graph_nodes_key_ + ":" + graph_deps_key_;
      if (fk == pending_prefill_key_) {
        prefill_key_ = fk;
      } else {
        pending_prefill_key_ = std::move(fk);
      }
    }

    // Reset build state for the next chunk.
    from_nodes_.clear();
    to_nodes_.clear();
    graph_nodes_key_.clear();
    graph_deps_key_.clear();
    node_map_.clear();
    active_deps_.clear();
    active_outputs_.clear();
    last_node_ = nullptr;
    bytes_in_graph_ = 0;
    build_nodes_.clear();
    build_node_params_.clear();

    // The exec references the current build's Packs by pointer (CLR stores
    // kernelParams by pointer) and is relaunched in later tokens, so the Packs
    // must outlive the slot's next use. Move them into the slot, releasing its
    // prior Packs — the slot's prior launch has drained (only inflight==0
    // reused). source_graph was adopted above when ExecUpdate/instantiate bound
    // the exec to build_graph_; if not adopted, destroy it. Build next fresh.
    used_slot->packs = std::move(graph_node_args_);
    graph_node_args_.clear();
    if (!build_graph_adopted) {
      (void)hipGraphDestroy(build_graph_);
    }
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
  }

  node_count_ = 0;

  // Put completion handlers in a batch.
  worker_->commit(stream_);
}

void CommandEncoder::synchronize() {
  (void)hipStreamSynchronize(stream_);
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  f.wait();
  (void)hipStreamSynchronize(stream_);
  // Stream is fully drained. Non-cached (no-reuse) execs reference these Packs
  // until now; cached-exec Packs live in their ExecSlot (clr#138) and are NOT
  // in these vectors, so clearing here is safe.
  graph_node_args_.clear();
  graph_node_args_prev_.clear();
  if (use_hip_graphs())
    flush_graph_deferred_frees();
}

// Global flag: true while any stream on this process is recording a HIP graph.
// Lazy library inits (e.g. hipblasLtCreate) abort the process if first called
// during capture, so they consult this to defer to a non-capturing path.
std::atomic<bool> g_stream_capturing{false};
bool stream_capturing() {
  return g_stream_capturing.load(std::memory_order_relaxed);
}
void set_stream_capturing(bool v) {
  g_stream_capturing.store(v, std::memory_order_relaxed);
}

std::atomic<bool> g_graph_active{false};
bool graph_active() {
  return g_graph_active.load(std::memory_order_relaxed);
}
void set_graph_active(bool v) {
  g_graph_active.store(v, std::memory_order_relaxed);
}

// Decode-mode: a single-token forward accrues into ONE graph (no mid-forward
// commit) that is refreshed via hipGraphExecUpdate and launched once per token.
// Set by the generation loop for Lstep==1 steps; prefill leaves it off so its
// large intermediates stay bounded by the per-graph caps. Disable entirely with
// MLX_GRAPH_DECODE=0.
bool graph_decode_mode() {
  static const bool enabled = [] {
    const char* e = std::getenv("MLX_GRAPH_DECODE");
    return !(e && std::string(e) == "0");
  }();
  return enabled && g_graph_decode_mode.load(std::memory_order_relaxed);
}
void set_graph_decode_mode(bool v) {
  g_graph_decode_mode.store(v, std::memory_order_relaxed);
}

std::unordered_map<int, Device>& get_devices() {
  static std::unordered_map<int, Device> devices;
  return devices;
}

Device& device(mlx::core::Device device) {
  auto& devices = get_devices();
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    // Set blocking sync flags on THIS device (per index, not a single global
    // bool: if device 0 were touched first the global gate would leave device 1
    // unflagged). Must happen while this device is current and before its
    // context is created — i.e. before the Device is constructed. Iterating
    // every device would create a context/queue on the other GPU too; on a
    // multi-GPU host that cross-device coexistence is what wedges the discrete
    // GPU's queue over a TB5 link, so touch only this device.
    (void)hipSetDevice(device.index);    (void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync);    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  // Bind the HIP current device to this stream's device. HIP's current device
  // is per-thread; everything that touches a stream goes through here (eval,
  // kernel launches, event record/wait, commit, completion callbacks). Without
  // binding, operations for a non-default GPU (--device 1) execute against
  // device 0 — the stream/event/kernel land on the wrong device and the queue
  // hangs. With HIP_VISIBLE_DEVICES the only device IS index 0 so the bug is
  // hidden.
  auto& d = device(s.device);
  d.make_current();
  // Flush any dirty host shadows before GPU kernels read device buffers.
  // CPU-side writes (array::init, array::data<T>) go to host shadow; GPU
  // kernels read from VRAM. Without this flush, kernels see stale data.
  rocm::allocator().flush_all_dirty_shadows();
  return d.get_command_encoder(s);
}

void clear_all_encoders() {
  auto& devices = get_devices();
  for (auto& [idx, dev] : devices) {
    dev.clear_encoders();
  }
}

} // namespace mlx::core::rocm

namespace mlx::core {
long decode_inline_launch_count() {
  return rocm::g_inline_launches_.load(std::memory_order_relaxed);
}
// Full decode-step stream-capture bridge.
bool decode_capture_begin() {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_begin(0);
}
bool decode_capture_end_record(int slot) {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_end_record(slot);
}
bool decode_capture_replay(int slot) {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_replay(slot);
}
void decode_capture_destroy() {
  rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_destroy();
}
} // namespace mlx::core
