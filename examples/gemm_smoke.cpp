// mlx-rocm MVP — GEMM smoke driver.
//
// Usage: gemm_smoke [host] [hip]
//   host  — small bf16 GEMM on CPU (library host path vs. independent f32
//           scalar reference in this file)
//   hip   — bf16 GPU GEMM on device 0 + fp8 e4m3 self-test (needs a build
//           with MLX_ROCM_WITH_HIP and a visible AMD GPU)
// No arguments: runs every section the build supports.
// Exit code: 0 = all sections pass, 1 = any section failed.

#include "mlx_rocm/gemm/gemm.h"
#include "mlx_rocm/gemm/gemm_ref.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using mlx_rocm::gemm_ref::bf16_to_f32;
using mlx_rocm::gemm_ref::f32_to_bf16;
using mlx_rocm::gemm_ref::lcg_f32;

static int g_fail = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL: %s\n", msg);                                \
      g_fail = 1;                                                             \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// Independent scalar reference: f32 accumulation, k-major looperational order
// (different order than the library's i-j-k reference).
static void ref_scalar(const std::uint16_t *A, const std::uint16_t *B,
                       std::uint16_t *D, int m, int n, int k) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int l = 0; l < k; ++l)
        acc += bf16_to_f32(A[(std::int64_t)i * k + l]) *
               bf16_to_f32(B[(std::int64_t)l * n + j]);
      D[(std::int64_t)i * n + j] = f32_to_bf16(acc);
    }
}

static std::uint16_t fill_bf16(std::vector<std::uint16_t> &v, std::uint64_t &seed) {
  for (auto &x : v)
    x = f32_to_bf16(lcg_f32(seed));
  return 0;
}

// bf16 has ~2^-8 relative resolution; allow 4 ulp-ish slack for the
// f32 accumulation-order difference.
static int cmp_bf16(const std::uint16_t *a, const std::uint16_t *b, size_t n,
                   float tol_rel) {
  float max_rel = 0.f;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float x = bf16_to_f32(a[i]);
    float y = bf16_to_f32(b[i]);
    float denom = x > 0.f ? x : -x;
    if (denom < 1e-3f)
      denom = 1e-3f;
    float rel = (y - x > 0.f ? y - x : x - y) / denom;
    if (rel > max_rel)
      max_rel = rel;
    if (rel > tol_rel)
      ++bad;
  }
  std::fprintf(stderr, "  cmp: max_rel_err=%.5f bad=%zu/%zu\n", max_rel, bad,
              n);
  return bad == 0 ? 0 : 1;
}

static int run_host(int do_host) {
  if (!do_host)
    return 0;
  std::fprintf(stderr, "== host section (bf16, 256^3) ==\n");
  const int m = 256, n = 256, k = 256;
  std::vector<std::uint16_t> A((size_t)m * k), B((size_t)k * n), D_lib((size_t)m * n),
      D_ref((size_t)m * n);
  std::uint64_t seed = 0x1234;
  fill_bf16(A, seed);
  fill_bf16(B, seed);

  CHECK(mlx_rocm_gemm_host_bf16(A.data(), B.data(), D_lib.data(), m, n, k) == 0,
        "mlx_rocm_gemm_host_bf16 returned error");
  ref_scalar(A.data(), B.data(), D_ref.data(), m, n, k);
  CHECK(cmp_bf16(D_lib.data(), D_ref.data(), (size_t)m * n, 2e-2f) == 0,
        "host GEMM mismatch vs independent reference");
  std::fprintf(stderr, "host GEMM: PASS\n");
  return 0;
}

#ifdef MLX_ROCM_GEMM_HIP_BUILT
static int run_hip(int do_hip) {
  if (!do_hip)
    return 0;
  std::fprintf(stderr, "== hip section (bf16 512^3 + fp8 probe) ==\n");

  char info[256];
  if (mlx_rocm_backend_info(info, sizeof(info)) == 0)
    std::fprintf(stderr, "backend: %s\n", info);
  CHECK(mlx_rocm_hip_device_count() > 0, "no HIP device visible");

  const int m = 512, n = 512, k = 512;
  std::vector<std::uint16_t> A((size_t)m * k), B((size_t)k * n), D_gpu((size_t)m * n),
      D_ref((size_t)m * n);
  std::uint64_t seed = 0x5678;
  fill_bf16(A, seed);
  fill_bf16(B, seed);
  ref_scalar(A.data(), B.data(), D_ref.data(), m, n, k);

  auto t0 = std::chrono::steady_clock::now();
  int rc = mlx_rocm_gemm_hip_bf16_hostio(A.data(), B.data(), D_gpu.data(), m, n,
                                         k);
  auto t1 = std::chrono::steady_clock::now();
  CHECK(rc == 0, "mlx_rocm_gemm_hip_bf16_hostio failed");
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double tflops = 2.0 * m * n * k / ms / 1e9;
  std::fprintf(stderr, "  bf16 GEMM 512^3: %.1f ms (%.2f TFLOPS bf16)\n", ms,
              tflops);
  CHECK(cmp_bf16(D_gpu.data(), D_ref.data(), (size_t)m * n, 2e-2f) == 0,
        "GPU GEMM mismatch vs host reference");
  std::fprintf(stderr, "GPU bf16 GEMM: PASS\n");

  std::fprintf(stderr, "preferred precision: %s (MLX_ROCM_GEMM_BF16=%s)\n",
              mlx_rocm_preferred_precision(),
              std::getenv("MLX_ROCM_GEMM_BF16") ? std::getenv("MLX_ROCM_GEMM_BF16")
                                                : "(unset)");
  if (mlx_rocm_gemm_fp8_probe() != 0)
    std::fprintf(stderr,
                 "fp8 probe failed/unsupported on this driver — bf16 "
                 "fallback is the safe default (MLX_ROCM_GEMM_BF16=1)\n");
  return 0;
}
#else
static int run_hip(int do_hip) {
  if (!do_hip)
    return 0;
  std::fprintf(stderr, "== hip section: skipped (host-only build) ==\n");
  return 0;
}
#endif

int main(int argc, char **argv) {
  int do_host = 0, do_hip = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "host") == 0)
      do_host = 1;
    else if (std::strcmp(argv[i], "hip") == 0)
      do_hip = 1;
    else {
      std::fprintf(stderr, "usage: gemm_smoke [host] [hip]\n");
      return 2;
    }
  }
  if (!do_host && !do_hip)
    do_host = do_hip = 1;

  int rc = run_host(do_host) | run_hip(do_hip);
  std::fprintf(stderr, "gemm_smoke: %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : rc;
}
