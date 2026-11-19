#pragma once
// W1 in_ba non-repro probe (cloudy order 2026-11-13, step 1b): allocator
// block tracking + hip-API log on pool blocks + pointer-classified buffer
// read. Env MLX_W1_INBA_ALLOC_PROBE=1, DEFAULT OFF (one static bool; zero
// behavior change when off).
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <hip/hip_runtime.h>

namespace w1probe {

bool enabled();

// allocator event tracking (block ptrs issued by the rocM allocator)
void note_alloc(void* ptr, size_t size);
void note_reuse(size_t size, void* ptr);
void note_recycle(size_t size, void* ptr);
void note_free(void* ptr, size_t size);

// hip API call on a pool block (membership-filtered; all calls logged with rc)
void api_log(const char* name, const void* p1, const void* p2, size_t n, int rc);

// pointer-classified buffer read: host/managed/unregistered = plain host
// memcpy (the host-pinned pool pointers reject hipMemcpy with InvalidValue);
// device = stream-ordered async copy + sync. Returns 0 on success.
int read_buf(const void* ptr, size_t n, std::vector<unsigned char>& out,
             hipStream_t stream);

// atexit: dump ring + counts to /tmp/w1inba_ap_<W1_PROBE_TAG>_pool.txt
void finish();

} // namespace w1probe
