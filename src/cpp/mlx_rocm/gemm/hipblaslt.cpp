// mlx-rocm MVP — minimal hipBLASLt GEMM (bf16 + fp8 e4m3).
//
// Design notes (mirrors the upstream fork's approach, see provenance in
// gemm.h): persistent hipblasLt handle, persistent 32 MB workspace, per-call
// descriptor/layout creation with a heuristic algo query, row-major layouts
// via HIPBLASLT_MATRIX_LAYOUT_ORDER. The upstream fork (hipblaslt_gemm.cpp)
// keeps cached layouts + heuristics per shape; this MVP keeps one shape per
// call, which is fine for the smoke path.
//
// gfx1201 note: fp8 e4m3 activation GEMM may be kernel-buggy on some RDNA4
// drivers (garbled long-context prefill on gfx1201 per upstream
// hipblaslt_gemm.cpp). mlx_rocm_gemm_fp8_probe() surfaces this; the serving
// stack can force bf16 with MLX_ROCM_GEMM_BF16=1 (see
// mlx_rocm_preferred_precision()).

#include "mlx_rocm/gemm/gemm.h"
#include "mlx_rocm/gemm/gemm_ref.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef MLX_ROCM_WITH_HIP
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

namespace {

constexpr size_t kWorkspaceBytes = 32u * 1024u * 1024u;
constexpr size_t kWorkspaceFallbackBytes = 4u * 1024u * 1024u;

bool hip_ok(hipError_t e, const char *ctx) {
  if (e == hipSuccess)
    return true;
  std::fprintf(stderr, "[mlx-rocm] %s: %s\n", ctx, hipGetErrorString(e));
  return false;
}

bool lt_ok(hipblasStatus_t e, const char *ctx) {
  if (e == HIPBLAS_STATUS_SUCCESS)
    return true;
  std::fprintf(stderr, "[mlx-rocm] %s: hipBLASLt status=%d\n", ctx, (int)e);
  return false;
}

struct LtState {
  hipblasLtHandle_t handle{nullptr};
  void *workspace{nullptr};
  size_t workspace_size{0};
  std::mutex mu;
};

LtState g_state;

bool ensure_state() {
  std::lock_guard<std::mutex> g(g_state.mu);
  if (g_state.handle)
    return true;
  if (!lt_ok(hipblasLtCreate(&g_state.handle), "hipblasLtCreate"))
    return false;
  for (size_t sz : {kWorkspaceBytes, kWorkspaceFallbackBytes}) {
    if (hipMalloc(&g_state.workspace, sz) == hipSuccess) {
      g_state.workspace_size = sz;
      break;
    }
  }
  if (!g_state.workspace) {
    std::fprintf(stderr, "[mlx-rocm] hipMalloc workspace failed\n");
    return false;
  }
  return true;
}

struct MatmulDescGuard {
  hipblasLtMatmulDesc_t d{nullptr};
  ~MatmulDescGuard() {
    if (d)
      hipblasLtMatmulDescDestroy(d);
  }
};

struct LayoutGuard {
  hipblasLtMatrixLayout_t l{nullptr};
  ~LayoutGuard() {
    if (l)
      hipblasLtMatrixLayoutDestroy(l);
  }
};

int make_cm_layout(LayoutGuard &out, hipDataType t, uint64_t rows,
                   uint64_t cols, int64_t ld) {
  return hipblasLtMatrixLayoutCreate(&out.l, t, rows, cols, ld) ==
             HIPBLAS_STATUS_SUCCESS
         ? 0
         : 1;
}

// Row-major D[m,n] = A[m,k] * B[k,n] computed in column-major BLAS terms as
// D_cm[n,m] (ld n) = B_cm[n,k] (ld n) * A_cm[k,m] (ld k), both OP_N — the
// same convention the upstream fork uses (column-major layouts, see its
// hipblaslt_gemm_impl op_a/op_b handling).
int gemm_device(const void *Ad, const void *Bd, void *Dd, int m, int n, int k,
                hipDataType ab_t, hipDataType d_t, hipblasComputeType_t comp) {
  if (!ensure_state())
    return 1;

  MatmulDescGuard md;
  if (!lt_ok(hipblasLtMatmulDescCreate(&md.d, comp, HIP_R_32F),
             "hipblasLtMatmulDescCreate"))
    return 1;
  int32_t op_n = static_cast<int32_t>(HIPBLAS_OP_N);
  hipblasLtMatmulDescSetAttribute(md.d, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n,
                                  sizeof(op_n));
  hipblasLtMatmulDescSetAttribute(md.d, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n,
                                  sizeof(op_n));

  LayoutGuard la, lb, ld_;
  if (make_cm_layout(la, ab_t, (uint64_t)n, (uint64_t)k, n) ||
      make_cm_layout(lb, ab_t, (uint64_t)k, (uint64_t)m, k) ||
      make_cm_layout(ld_, d_t, (uint64_t)n, (uint64_t)m, n)) {
    std::fprintf(stderr, "[mlx-rocm] layout create failed\n");
    return 1;
  }

  hipblasLtMatmulPreference_t pref = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
    return 1;
  hipblasLtMatmulPreferenceSetAttribute(pref,
                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &g_state.workspace_size,
                                        sizeof(g_state.workspace_size));

  hipblasLtMatmulHeuristicResult_t results[4] = {};
  int nres = 0;
  if (hipblasLtMatmulAlgoGetHeuristic(g_state.handle, md.d, la.l, lb.l, ld_.l,
                                      ld_.l, pref, 4, results, &nres) !=
          HIPBLAS_STATUS_SUCCESS ||
      nres <= 0) {
    std::fprintf(
        stderr,
        "[mlx-rocm] no hipBLASLt algo for this dtype combo/shape "
        "(m=%d n=%d k=%d) — the known gfx1201 fp8 GEMM bug may be in "
        "play; try MLX_ROCM_GEMM_BF16=1\n",
        m, n, k);
    hipblasLtMatmulPreferenceDestroy(pref);
    return 1;
  }

  float alpha = 1.0f, beta = 0.0f;
  hipblasStatus_t st =
      hipblasLtMatmul(g_state.handle, md.d, &alpha, Bd, la.l, Ad, lb.l, &beta,
                      Dd, ld_.l, Dd, ld_.l, &results[0].algo,
                      g_state.workspace, g_state.workspace_size,
                      /*stream*/ 0);
  hipblasLtMatmulPreferenceDestroy(pref);
  return lt_ok(st, "hipblasLtMatmul") ? 0 : 1;
}

// Shared H2D / gemm / D2H driver for the hostio variants.
int hostio(const void *Ah, const void *Bh, void *Dh, int m, int n, int k,
           hipDataType ab_t, hipDataType d_t, hipblasComputeType_t comp,
           int ab_elem_bytes) {
  const size_t ab = (size_t)ab_elem_bytes;
  const size_t a_bytes = (size_t)m * k * ab;
  const size_t b_bytes = (size_t)k * n * ab;
  const size_t d_bytes = (size_t)m * n * 2; // bf16 out

  void *Ad = nullptr, *Bd = nullptr, *Dd = nullptr;
  if (hipMalloc(&Ad, a_bytes) != hipSuccess ||
      hipMalloc(&Bd, b_bytes) != hipSuccess ||
      hipMalloc(&Dd, d_bytes) != hipSuccess) {
    std::fprintf(stderr, "[mlx-rocm] hipMalloc failed\n");
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Dd);
    return 1;
  }

  int rc = 0;
  if (hipMemcpy(Ad, Ah, a_bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(Bd, Bh, b_bytes, hipMemcpyHostToDevice) != hipSuccess) {
    rc = 1;
  } else {
    rc = gemm_device(Ad, Bd, Dd, m, n, k, ab_t, d_t, comp);
  }
  if (rc == 0 && hipMemcpy(Dh, Dd, d_bytes, hipMemcpyDeviceToHost) != hipSuccess)
    rc = 1;
  hipFree(Ad);
  hipFree(Bd);
  hipFree(Dd);
  return rc;
}

int probe_fp8() {
  const int m = 256, n = 256, k = 256;
  std::vector<uint8_t> Ah((size_t)m * k), Bh((size_t)k * n);
  std::vector<uint16_t> D_ref((size_t)m * n), D_dev((size_t)m * n, 0);

  // e4m3 codes in [0, 96): no NaN (0x7F/0xFF) and small magnitudes for a
  // sane reference.
  std::uint64_t seed = 0x9E3779B97F4A7C15ull;
  auto next = [&](uint8_t &x) {
    seed = seed * 636413629384795163ull + 1ull;
    x = (uint8_t)(seed % 96);
  };
  for (auto &x : Ah)
    next(x);
  for (auto &x : Bh)
    next(x);

  // Reference: decode e4m3 -> f32, f32 GEMM, round to bf16.
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float acc = 0.f;
      for (int l = 0; l < k; ++l)
        acc += mlx_rocm::gemm_ref::fp8e4m3_to_f32(Ah[(size_t)i * k + l]) *
               mlx_rocm::gemm_ref::fp8e4m3_to_f32(Bh[(size_t)l * n + j]);
      D_ref[(size_t)i * n + j] = mlx_rocm::gemm_ref::f32_to_bf16(acc);
    }

  int rc = mlx_rocm_gemm_hip_fp8_hostio(Ah.data(), Bh.data(), D_dev.data(), m,
                                        n, k);
  if (rc != 0) {
    std::fprintf(stderr,
                 "[mlx-rocm] fp8 probe: hip fp8 GEMM failed (rc=%d)\n", rc);
    return 1;
  }

  float max_rel = 0.f;
  size_t bad = 0;
  for (int i = 0; i < m * n; ++i) {
    float a = mlx_rocm::gemm_ref::bf16_to_f32(D_ref[i]);
    float b = mlx_rocm::gemm_ref::bf16_to_f32(D_dev[i]);
    float denom = a > 0.f ? a : -a;
    if (denom < 1e-3f)
      denom = 1e-3f;
    float rel = (b - a > 0.f ? b - a : a - b) / denom;
    if (rel > max_rel)
      max_rel = rel;
    if (rel > 0.05f)
      ++bad;
  }
  std::fprintf(stderr,
               "[mlx-rocm] fp8 probe: max_rel_err=%.4f bad=%zu/%zu -> %s\n",
               max_rel, bad, (size_t)m * n,
               (bad * 100 < m * n) ? "PASS" : "FAIL (gfx1201 fp8 GEMM bug?)");
  return bad * 100 < m * n ? 0 : 1;
}

} // namespace

#else // !MLX_ROCM_WITH_HIP

#endif

extern "C" {

int mlx_rocm_gemm_host_bf16(const uint16_t *A, const uint16_t *B, uint16_t *D,
                            int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || !A || !B || !D)
    return 1;
  mlx_rocm::gemm_ref::gemm_host_bf16(A, B, D, m, n, k);
  return 0;
}

#ifdef MLX_ROCM_WITH_HIP
int mlx_rocm_gemm_hip_bf16_hostio(const uint16_t *A, const uint16_t *B,
                                  uint16_t *D, int m, int n, int k) {
  return hostio(A, B, D, m, n, k, HIP_R_16BF, HIP_R_16BF,
                HIPBLAS_COMPUTE_32F, 2);
}

int mlx_rocm_gemm_hip_fp8_hostio(const uint8_t *A, const uint8_t *B,
                                 uint16_t *D, int m, int n, int k) {
  return hostio(A, B, D, m, n, k, HIP_R_8F_E4M3, HIP_R_16BF,
                HIPBLAS_COMPUTE_32F, 1);
}

int mlx_rocm_gemm_fp8_probe(void) {
  if (mlx_rocm_hip_device_count() == 0) {
    std::fprintf(stderr, "[mlx-rocm] fp8 probe: no HIP device\n");
    return 1;
  }
  return probe_fp8();
}

int mlx_rocm_hip_device_count(void) {
  int nd = 0;
  if (hipGetDeviceCount(&nd) != hipSuccess)
    return 0;
  return nd;
}

int mlx_rocm_backend_info(char *buf, int buflen) {
  if (!buf || buflen <= 1)
    return 1;
  int nd = mlx_rocm_hip_device_count();
  const char *arch = "unknown";
  if (nd > 0) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess)
      arch = prop.gcnArchName;
  }
  std::snprintf(buf, (size_t)buflen,
                "hip=1 n_dev=%d arch0=%s workspace=%zuMB", nd, arch,
                g_state.workspace_size / (1024u * 1024u));
  return 0;
}

#else
int mlx_rocm_gemm_hip_bf16_hostio(const uint16_t *, const uint16_t *,
                                  uint16_t *, int, int, int) {
  std::fprintf(stderr,
               "[mlx-rocm] built without ROCm support "
               "(rebuild with MLX_ROCM_WITH_HIP)\n");
  return 2;
}

int mlx_rocm_gemm_hip_fp8_hostio(const uint8_t *, const uint8_t *,
                                 uint16_t *, int, int, int) {
  std::fprintf(stderr,
               "[mlx-rocm] built without ROCm support "
               "(rebuild with MLX_ROCM_WITH_HIP)\n");
  return 2;
}

int mlx_rocm_gemm_fp8_probe(void) {
  std::fprintf(stderr, "[mlx-rocm] fp8 probe unavailable (no ROCm build)\n");
  return 1;
}

int mlx_rocm_hip_device_count(void) {
  return 0;
}

int mlx_rocm_backend_info(char *buf, int buflen) {
  if (!buf || buflen <= 1)
    return 1;
  std::snprintf(buf, (size_t)buflen, "hip=0 (host-only build)");
  return 0;
}
#endif

const char *mlx_rocm_preferred_precision(void) {
  const char *e = std::getenv("MLX_ROCM_GEMM_BF16");
  if (e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O'))
    return "bf16";
  return "fp8";
}

} // extern "C"
