// Copyright © 2025 Apple Inc.
//
// ROCm allocator — mirrors mlx/backend/cuda/allocator.h exactly.
// Device free/alloc: BufferCache recycle, hipMallocAsync / hipFreeAsync
// (or hipMalloc / hipFree when no stream/pool). Only ROCm-specific extras
// are host_shadow (discrete GPU CPU access) and optional decode arena /
// graph deferred free.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <deque>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlx::core::rocm {

using allocator::Buffer;

// Matches CudaBuffer + fields required for discrete-GPU host access and
// stream-ordered free (CUDA uses move_to_unified_memory instead of shadow).
// Field order is ABI-stable for brace-init sites outside this file
// (e.g. gemms/naive_gemm.hip): data, size, is_managed, device, ...
struct RocmBuffer {
  void* data;
  size_t size;
  bool is_managed;
  int device; // -1 for unified / non-pool
  void* host_shadow;
  bool host_dirty;
  void* alloc_stream; // stream used for hipMallocAsync / hipFreeAsync
};

// ---------------------------------------------------------------------------
// SmallSizePool — identical to CUDA (8-byte scalar freelist only).
// ---------------------------------------------------------------------------

class SmallSizePool {
 private:
  union Block {
    Block* next;
    RocmBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  bool data_managed_{true};
  Block* next_free_{nullptr};

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  RocmBuffer* malloc();
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf);
};

// ---------------------------------------------------------------------------
// DecodeArena — optional bump allocator for HIP-graph decode (not in CUDA).
// Training does not use this; only the graph-decode engine does.
// ---------------------------------------------------------------------------
struct DecodeArena {
  void* base{nullptr};
  size_t capacity{0};
  size_t offset{0};
  bool active{false};
  bool overflowed{false};
  int device{-1};
  std::deque<RocmBuffer> wrappers;
  size_t next_wrapper{0};
  size_t high_water{0};
  size_t floor_offset{0};
  size_t floor_wrapper{0};

  bool contains(const void* p) const {
    return base && p >= base && p < static_cast<const char*>(base) + capacity;
  }
};

// ---------------------------------------------------------------------------
// RocmAllocator — same control flow as CudaAllocator.
// ---------------------------------------------------------------------------

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_async(size_t size, int device, void* stream);
  void free(Buffer buffer) override {
    free(buffer, /*force=*/false);
  }
  void free(Buffer buffer, bool force);
  size_t size(Buffer buffer) const override;

  Buffer make_buffer(void* ptr, size_t size) override;
  void release(Buffer buffer) override;

  // Free device storage only (CUDA free_async).
  void free_async(RocmBuffer& buf, void* stream = nullptr);
  // Free device + delete shell (CUDA free_cuda_buffer).
  void free_rocm_buffer(RocmBuffer* buf);

  void ensure_host_shadow(RocmBuffer& buf);
  void flush_host_shadow(RocmBuffer& buf);
  // Flush all host shadows that have been written from the CPU back to VRAM.
  // Called before GPU kernel execution to ensure device buffers are current.
  void flush_all_dirty_shadows();
  void mark_dirty_shadow(RocmBuffer* buf);

  void* decode_arena_base_ptr() const {
    return decode_arena_.base;
  }
  // G1V: ownership-map accessors for fault-VA attribution (diagnostic).
  const std::deque<RocmBuffer>& decode_arena_wrappers() const {
    return decode_arena_.wrappers;
  }
  size_t decode_arena_floor_offset() const {
    return decode_arena_.floor_offset;
  }
  size_t decode_arena_floor_wrapper() const {
    return decode_arena_.floor_wrapper;
  }

    bool in_decode_arena(void* p) const {
    return decode_arena_.contains(p);
  }
  bool decode_arena_begin(size_t capacity, int device, void* stream);
  void decode_arena_reset();
  void decode_arena_freeze_floor();
  void decode_arena_reset_to_floor();
  void decode_arena_end();
  bool decode_arena_active() const {
    return decode_arena_.active;
  }
  size_t decode_arena_high_water() const {
    return decode_arena_.high_water;
  }
  size_t decode_arena_offset() const {
    return decode_arena_.offset;
  }
  size_t decode_arena_capacity() const {
    return decode_arena_.capacity;
  }
  bool decode_arena_overflowed() const {
    return decode_arena_.overflowed;
  }

  // A persistent per-step value slot for decode exec-cache replay: a small stable arena buffer + a pinned host buffer. The initial capture records an H2D memcpy (host->device) and the kernel reads the device buffer; the CPU re-writes the host value each replay step before the graph launch, so per-step values (the KV write offset, the SDPA K length) reach the replayed graph. host==nullptr when the decode arena is inactive (legacy by-value path).
  struct DecodeValueSlot {
    Buffer buf{nullptr};  // persistent (never freed), valid iff host != nullptr
    int64_t* host{nullptr};  // pinned
    void* data{nullptr};  // device data pointer (buf wraps the RocmBuffer; ptr() is the wrapper, data is the kernel-facing VA)
  };
  DecodeValueSlot decode_value_slot(int index);
  void decode_arena_reserve_wrappers(size_t n);
  // A persistent 8-byte device buffer holding 0, for the dynamic-copy "missing offset" case in decode mode: a per-step malloc + memset would shift the arena layout (and add a node), breaking the replay step's stable addresses. Zeroed once per arena incarnation.
  DecodeValueSlot decode_zero_slot();
  // K3 (t1082): per-step cursor into the 4 KB value slot 3 (gather out). Each small gather's row is placed at a distinct 256 B aligned offset so rows do not clobber each other in-graph (a decode step has several fast_rows gathers). Returns the row's offset, or the 4 KB capacity if it does not fit (the caller falls back to a stable-arena buffer). Reset with the arena.
  size_t decode_value_slot3_cursor(size_t size);
  // GD-idx (t1099 R3): the per-step token-idx staging slot. A 4 KB device buffer on its OWN hipMalloc (not arena: the token array is evaluated when its python object is created — before the step's first op, when a fresh capture's arena is not yet active) + a 16 B pinned host. Its device VA is stable across steps AND arena incarnations, so the capture can bake it into the embedding gather; the token H2D re-executes before every graph launch (eval-time copy + pre-launch re-delivery, both sourced from the persistent pinned buffer, which the neutralized in-graph node pattern keeps current). The per-step re-allocation of the old per-step pinned path moved the out VA under the baked pointer, so the replayed graph read the capture step's value (the R2 collapse).
  DecodeValueSlot decode_idx_slot();
  // Arm is set by the token staging (Load small path); consumed by the first single-row fast_rows gather of the step (the embedding). Other single-row gathers keep their own idx buffer. Cleared with the arena reset.
  void decode_idx_slot_arm();
  bool decode_idx_slot_take();
  // GD-idx (R4): the capture-time token-idx buffer VA (the recorded stage-slot
  // kernel node's source). On a replay step the pre-launch Load refreshes this
  // VA so the re-run node propagates the NEW token into the slot; the VA is a
  // stable arena input slot (preserved above the floor).
  void decode_idx_set_src_va(void* va, size_t nbytes);
  void* decode_idx_src_va();
  size_t decode_idx_src_nbytes();
  // GD-idx (R4): the persistent staging source buffer (lazy hipMalloc; the
  // pilot materializes it BEFORE the capture window — a hip* during a
  // capture invalidates it on ROCm 10.1). Never freed: the VA is baked
  // into every exec that references it (the slot's device buffer follows
  // the same rule).
  void* decode_idx_src_buffer();
  // GD-idx (R4): drop the recorded source (a fresh capture records a fresh
  // staging node; the buffer itself persists — only the record is
  // per-incarnation). Also disarms the slot (the old incarnation's arm is
  // stale).
  void decode_idx_reset_src();
  // True iff p is one of the persistent staging host buffers (value slots 0-3 or the idx slot). Pre-launch re-delivery of a recorded in-graph H2D must skip any other (per-step) source: it is freed at step end, and copying from it is a UAF read of the stale capture-step value.
  bool is_persistent_staging_host(const void* p) const;

  bool train_arena_begin(size_t capacity, int device, void* stream) {
    return decode_arena_begin(capacity, device, stream);
  }
  void train_arena_reset() {
    decode_arena_reset();
  }
  void train_arena_freeze_floor() {
    decode_arena_freeze_floor();
  }
  void train_arena_reset_to_floor() {
    decode_arena_reset_to_floor();
  }
  void train_arena_end() {
    decode_arena_end();
  }
  bool train_arena_active() const {
    return decode_arena_active();
  }
  size_t train_arena_high_water() const {
    return decode_arena_high_water();
  }
  bool train_arena_overflowed() const {
    return decode_arena_overflowed();
  }

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  RocmAllocator();
  friend RocmAllocator& allocator();

  RocmBuffer* arena_alloc(size_t size);

  std::mutex mutex_;
  size_t memory_limit_;
  size_t free_limit_;
  size_t total_memory_{0};
  size_t max_pool_size_;
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  std::vector<void*> free_streams_;
  std::vector<void*> mem_pools_;
  SmallSizePool scalar_pool_;
  // Track buffers with dirty host shadows (CPU wrote, needs flush to VRAM).
  std::unordered_set<RocmBuffer*> dirty_shadows_;
  DecodeArena decode_arena_;
  struct DecodeValueSlotState {
    Buffer buf{nullptr};
    int64_t* host{nullptr};
    void* data{nullptr};  // device data pointer (see DecodeValueSlot::data)
    void* base{nullptr};  // arena base at allocation time
  };
  DecodeValueSlotState decode_value_slot_state_[4];
  DecodeValueSlotState decode_zero_state_;
  struct DecodeIdxSlotState {
    Buffer buf{nullptr};
    int64_t* host{nullptr};  // pinned, 16 B, persistent
    void* data{nullptr};  // device, 4 KB, persistent
    // GD-idx (R4): the staging node's SOURCE — a 16 B device buffer with a
    // process-lifetime fixed VA (like the slot's device buffer). The
    // captured node (source -> slot) bakes this VA; every step's token
    // writer (the embedding gather's pinned-shadow H2D, or the token Load's
    // refresh) rewrites it in place, so the re-run node at the step-end
    // launch always carries the NEW token. The per-step idx buffer's VA
    // cannot be the node source: at the capture step the idx is a
    // plain-malloc buffer (the arena only activates in the first op's
    // prepare) that the host frees after the step, while at replay steps
    // the idx lands in the arena at the step-end offset — two different
    // VAs, so the re-run node read freed capture-time memory (the 264
    // attractor that survived R3).
    void* src_dev{nullptr};
  };
  DecodeIdxSlotState decode_idx_state_;
  bool idx_slot_armed_{false};
  void* decode_idx_src_va_{nullptr};
  size_t decode_idx_src_nbytes_{0};
  size_t decode_slot3_cursor_{0};
};

RocmAllocator& allocator();

class CommandEncoder;
Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::rocm
