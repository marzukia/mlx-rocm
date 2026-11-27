// iu8 16x16x16 WMMA fragment probe v2 (gfx1201, P1-d leg 2, defect 2 oracle).
// Runs __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12 with deterministic
// per-lane patterns and dumps the V8i C (32 lanes x 8) + V2i A/B to global.
//
// Patterns (blockIdx.x):
//   0          : control, A = 1, B = 1 -> C = 16 everywhere (16 k terms)
//   1..32      : A = e_(p-1), B = 1  -> C != 0 iff row m of cell contains lane p-1
//   33..64     : A = 1, B = e_(p-33) -> C != 0 iff col n of cell contains lane p-33
//   65         : A = v(l), B = 1 (cross-check)
//   66         : A = 1, B = v(l)
//   67         : A = v(l), B = v(l)
// where v(l) = 2*l + l/8 (max 66 < 256, per-lane constant over all 8 iu8 bytes).
//
// Decoding: for each cell (lane, reg) build a 32-bit mask of which A-lanes
// appear (from patterns 1..32) and a 32-bit mask of which B-lanes appear
// (from patterns 33..64). 16 distinct A-masks -> m; 16 distinct B-masks -> n.
// The 256 cells must form the complete 16x16 (m,n) grid.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

using v2i = __attribute__((ext_vector_type(2))) int;
using v8i = __attribute__((ext_vector_type(8))) int;

#define NPAT 68

__global__ void probe(int* C, int* Ab, int* Bb) {
  int lane = threadIdx.x;             // 0..31
  int p    = blockIdx.x;
  int v    = 2*lane + (lane/8);      // per-lane value (0..66, fits iu8)
  int a = 0, b = 0;
  if (p == 0)        { a = 1; b = 1; }
  else if (p <= 32)  { a = (lane == p-1) ? 1 : 0; b = 1; }
  else if (p <= 64)  { a = 1; b = (lane == p-33) ? 1 : 0; }
  else if (p == 65)  { a = v; b = 1; }
  else if (p == 66)  { a = 1; b = v; }
  else               { a = v; b = v; }
  v2i a0, b0;
  for (int i = 0; i < 2; ++i) {
    uint8_t * pa = (uint8_t *) (&a0);
    uint8_t * pb = (uint8_t *) (&b0);
    for (int by = 0; by < 4; ++by) {
      pa[i*4 + by] = (uint8_t) a;
      pb[i*4 + by] = (uint8_t) b;
    }
  }
  v8i c;
  for (int i = 0; i < 8; ++i) c[i] = 0;
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, c, true);
  for (int i = 0; i < 8; ++i) C[(p*32 + lane)*8 + i] = c[i];
  Ab[(p*32 + lane)*2 + 0] = a0[0];
  Ab[(p*32 + lane)*2 + 1] = a0[1];
  Bb[(p*32 + lane)*2 + 0] = b0[0];
  Bb[(p*32 + lane)*2 + 1] = b0[1];
}

int main() {
  int * hC = new int[NPAT*32*8];
  int * hA = new int[NPAT*32*2];
  int * hB = new int[NPAT*32*2];
  int * dC, * dA, * dB;
  hipError_t e0 = hipMalloc(&dC, sizeof(int)*NPAT*32*8);
  hipError_t e1 = hipMalloc(&dA, sizeof(int)*NPAT*32*2);
  hipError_t e2 = hipMalloc(&dB, sizeof(int)*NPAT*32*2);
  if (e0 || e1 || e2) { printf("PROBE-FAIL hipMalloc\n"); return 1; }
  hipLaunchKernelGGL(probe, dim3(NPAT), dim3(32), 0, 0, dC, dA, dB);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    printf("PROBE-FAIL hipDeviceSynchronize: %s\n", hipGetErrorString(e));
    return 1;
  }
  hipMemcpy(hC, dC, sizeof(int)*NPAT*32*8, hipMemcpyDeviceToHost);
  hipMemcpy(hA, dA, sizeof(int)*NPAT*32*2, hipMemcpyDeviceToHost);
  hipMemcpy(hB, dB, sizeof(int)*NPAT*32*2, hipMemcpyDeviceToHost);
  for (int p = 0; p < NPAT; ++p) {
    printf("== pattern %d\n", p);
    for (int lane = 0; lane < 32; ++lane) {
      printf("lane %2d: A=%d %d B=%d %d C=",
             lane, hA[(p*32+lane)*2], hA[(p*32+lane)*2+1],
             hB[(p*32+lane)*2], hB[(p*32+lane)*2+1]);
      for (int l = 0; l < 8; ++l) printf(" %5d", hC[(p*32+lane)*8+l]);
      printf("\n");
    }
  }
  printf("PROBE-OK\n");
  return 0;
}
