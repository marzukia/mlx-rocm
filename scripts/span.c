// t2246 seam: per-step GPU span measurement via HIP events on the LEGACY
// default stream. The legacy default stream (stream 0) is the magic
// synchronization stream in HIP (like CUDA): an event recorded on it is
// ordered with respect to work on all other streams, so bracketing a step
// with two events on stream 0 measures the GPU-timeline span of the step
// regardless of which stream mlx uses.
//
// span_begin(): record evA (host-side, negligible; the event timestamp is
// stamped when the GPU reaches it).
// span_end():   record evB, wait for evB completion, return ms between
// evA and evB (GPU-timeline elapsed time: kernels + in-step idle gaps).
//
// Build: gcc -O2 -shared -fPIC span.c -o span.so -ldl
// Self-contained: the box has no usable ROCm hip headers, so the four
// needed libamdhip64 entry points are resolved via dlsym (stable ABI).
#include <stdio.h>
#include <dlfcn.h>

enum {
  _hipSuccess = 0,
  _hipErrorNotReady = 600,
  _hipErrorInvalidHandle = 400
};
typedef int hipError_t;
typedef void* hipEvent_t;   // struct ihipEvent_t* in hip 7.2
typedef void* hipStream_t; // struct ihipStream_t* in hip 7.2
typedef hipError_t (*hipEventCreateWithFlags_fn)(hipEvent_t*, unsigned);
typedef hipError_t (*hipEventRecord_fn)(hipEvent_t, hipStream_t);
typedef hipError_t (*hipEventQuery_fn)(hipEvent_t);
typedef hipError_t (*hipEventElapsedTime_fn)(float*, hipEvent_t, hipEvent_t);

static hipEventCreateWithFlags_fn p_hipEventCreateWithFlags;
static hipEventRecord_fn p_hipEventRecord;
static hipEventQuery_fn p_hipEventQuery;
static hipEventElapsedTime_fn p_hipEventElapsedTime;
static hipEvent_t evA = 0, evB = 0;
static int inited = 0;

int span_init(void) {
  if (inited) return 0;
  // libamdhip64 is already loaded by the process (via mlx, RTLD_LOCAL,
  // so RTLD_DEFAULT cannot see it). dlopen by soname returns the SAME
  // loaded object (dedupe by soname).
  void* h = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_NOLOAD);
  if (!h) h = dlopen("libamdhip64.so.7", RTLD_NOW);
  if (!h) h = dlopen("libamdhip64.so", RTLD_NOW);
  if (!h) {
    fprintf(stderr, "span: cannot open libamdhip64\n");
    return 1;
  }
  p_hipEventCreateWithFlags =
      (hipEventCreateWithFlags_fn)dlsym(h, "hipEventCreateWithFlags");
  p_hipEventRecord = (hipEventRecord_fn)dlsym(h, "hipEventRecord");
  p_hipEventQuery = (hipEventQuery_fn)dlsym(h, "hipEventQuery");
  p_hipEventElapsedTime =
      (hipEventElapsedTime_fn)dlsym(h, "hipEventElapsedTime");
  if (!p_hipEventCreateWithFlags || !p_hipEventRecord || !p_hipEventQuery ||
      !p_hipEventElapsedTime) {
    fprintf(stderr, "span: dlsym failures\n");
    return 1;
  }
  hipError_t e = p_hipEventCreateWithFlags(&evA, 0);
  fprintf(stderr, "span: evA create rc=%d ptr=%p\n", (int)e, (void*)evA);
  if (e != _hipSuccess || !evA) {
    return 1;
  }
  e = p_hipEventCreateWithFlags(&evB, 0);
  fprintf(stderr, "span: evB create rc=%d ptr=%p\n", (int)e, (void*)evB);
  if (e != _hipSuccess || !evB) {
    return 2;
  }
  inited = 1;
  return 0;
}

void span_begin(void) {
  if (!inited && span_init() != 0) return;
  hipError_t e = p_hipEventRecord(evA, (hipStream_t)0);
  if (e != _hipSuccess) fprintf(stderr, "span: begin rc=%d\n", (int)e);
}

double span_end(void) {
  if (!inited) return -1.0;
  hipError_t e = p_hipEventRecord(evB, (hipStream_t)0);
  if (e != _hipSuccess) {
    fprintf(stderr, "span: end record rc=%d\n", (int)e);
    return -1.0;
  }
  while (p_hipEventQuery(evB) == _hipErrorNotReady) {
  }
  float ms = -1.0f;
  e = p_hipEventElapsedTime(&ms, evA, evB);
  if (e != _hipSuccess) {
    fprintf(stderr, "span: elapsed rc=%d evA=%p evB=%p\n", (int)e, (void*)evA,
            (void*)evB);
    return -1.0;
  }
  return (double)ms;
}
