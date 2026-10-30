// Step 0: direct HIP CU query — the value mlx uses (device.cpp logs
// hipGetDeviceProperties().multiProcessorCount as cus=).
//
// Build + run (guest, under flock):
//   /opt/rocm/core-10.1/bin/hipcc -o hipcu step0_hip_cu.cu
//   flock -w 60 /models/gpu-bench.lock -c ./hipcu
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  hipDeviceProp_t p;
  hipGetDeviceProperties(&p, 0);
  printf("name=%s arch=%s mc=%d warp=%d smem=%zu total=%zu\n", p.name,
         p.gcnArchName, p.multiProcessorCount, p.warpSize,
         (size_t)p.sharedMemPerBlock, (size_t)p.totalGlobalMem);
  return 0;
}
