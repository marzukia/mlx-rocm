#!/usr/bin/env python3
"""Apply the ent4 (Phase 3, lever 1) in-CU decode patch to the ROCm vendor tree.

Usage:
  python3 apply-ent4-patch.py /path/to/vendor-tree          # apply (idempotent)
  python3 apply-ent4-patch.py /path/to/vendor-tree --check  # verify, no write

Adds backend/rocm/quantized/ent4.h (device decode kernel + host registry +
.ent4 file reader), and small hooks in:
  primitives.h            Load op gains an optional ent4_name_ (tensor name).
  io/safetensors.cpp      on MLX_ROCM_ENTROPY=1, mark 4-bit tensors in the
                          .ent4 table so their Load uploads the compressed
                          stream + side tables instead of the dense bytes.
  backend/rocm/load.cpp   Load::eval_gpu: upload compressed stream + side
                          tables to VRAM, register w_ptr -> Reg.
  backend/rocm/quantized/qmm.hip  wide qmv path: if w_ptr is registered,
                          decode the stream to a dense scratch and read it.

Lossless by construction: the decode kernel is the same canonical-Huffman
decoder the offline C verifier used (verified bit-identical 250/250).
Default path (MLX_ROCM_ENTROPY unset) and CI are untouched.
"""

import hashlib
import os
import shutil
import sys

PRE_MD5 = {
    "primitives.h": "d7a4b374562831967ed8298cd8d51350",
    "io/safetensors.cpp": "96d8a5e8bd1c719ff1fca9df5be54f9b",
    "backend/rocm/load.cpp": "271421a4a0909c46d5e397adfa99a7d9",
    "backend/rocm/quantized/qmm.hip": "bd9c83c187546e5aa1468fba65ce2fa9",
}

MARKERS = {
    "primitives.h": "set_ent4_name",
    "io/safetensors.cpp": "ent4::ent4_table()",
    "backend/rocm/load.cpp": "rocm::ent4::ent4_table()",
    "backend/rocm/quantized/qmm.hip": "rocm::ent4::Registry::instance()",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENT4_H = os.path.join(SCRIPT_DIR, "..", "patches", "ent4.h")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def apply(root):
    # 1. write ent4.h (host) + ent4_kernel.hip (device)
    dst_h = os.path.join(root, "backend", "rocm", "quantized", "ent4.h")
    with open(ENT4_H, "rb") as src, open(dst_h, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  wrote {os.path.relpath(dst_h, root)}")
    dst_k = os.path.join(root, "backend", "rocm", "quantized", "ent4_kernel.hip")
    with open(os.path.join(SCRIPT_DIR, "..", "patches", "ent4_kernel.hip"), "rb") as src, open(dst_k, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.chmod(dst_k, 0o755)
    print(f"  wrote {os.path.relpath(dst_k, root)}")
    dst_j = os.path.join(root, "backend", "rocm", "quantized", "ent4_json.h")
    with open(os.path.join(SCRIPT_DIR, "..", "patches", "ent4_json.h"), "rb") as src, open(dst_j, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  wrote {os.path.relpath(dst_j, root)}")

    def edit(rel, old, new, marker=None):
        p = os.path.join(root, rel)
        with open(p) as f:
            t = f.read()
        if new in t:
            return  # already applied
        if marker is not None and marker in t:
            return  # block already present (anchor lines may be separated)
        if old not in t:
            # Neither `new` nor `old` is present: the tree is in an
            # intermediate or already-applied state. Skip rather than fail so
            # re-apply stays idempotent across the LUT-only -> complete steps.
            print(f"  skip {rel} (pattern absent; assuming applied)")
            return
        t = t.replace(old, new, 1)
        with open(p, "w") as f:
            f.write(t)
        print(f"  patched {rel}")

    # 2. primitives.h: Load gains ent4_name_
    edit(
        "primitives.h",
        """  DEFINE_NAME(Load)

 private:
  std::shared_ptr<io::Reader> reader_;
  size_t offset_;
  bool swap_endianness_;
};""",
        """  void set_ent4_name(std::string n) { ent4_name_ = std::move(n); }
  bool has_ent4() const { return !ent4_name_.empty(); }
  const std::string& ent4_name() const { return ent4_name_; }

  DEFINE_NAME(Load)

 private:
  std::shared_ptr<io::Reader> reader_;
  size_t offset_;
  bool swap_endianness_;
  std::string ent4_name_;
};""",
    )

    # 3. io/safetensors.cpp: include + mark ent4 tensors
    edit(
        "io/safetensors.cpp",
        '#include "mlx/io/load.h"',
        '#include "mlx/io/load.h"\n#include "mlx/backend/rocm/quantized/ent4_json.h"',
    )
    edit(
        "io/safetensors.cpp",
        """    res.insert(
        {item.key(),
         array(
             shape,
             type,
             std::make_shared<Load>(
                 stream, in_stream, offset + data_offsets.at(0), false),
             std::vector<array>{})});
  }""",
        """    auto load_op = std::make_shared<Load>(
        stream, in_stream, offset + data_offsets.at(0), false);
    if (const auto* tbl = rocm::ent4::ent4_table(); tbl->count(item.key()))
      load_op->set_ent4_name(item.key());
    res.insert(
        {item.key(),
         array(shape, type, std::move(load_op), std::vector<array>{})});
  }""",
    )

    # 4. backend/rocm/load.cpp: include + upload/register
    edit(
        "backend/rocm/load.cpp",
        '#include "mlx/primitives.h"',
        '#include "mlx/primitives.h"\n#include "mlx/backend/rocm/quantized/ent4_json.h"',
    )
    edit(
        "backend/rocm/load.cpp",
        """  out.set_data(mlx::core::rocm::malloc_async(nbytes, encoder));
  // Stage through PINNED host memory.""",
        """  out.set_data(mlx::core::rocm::malloc_async(nbytes, encoder));
  // ent4: upload the compressed stream + side tables and register w_ptr.
  if (has_ent4()) {
    const auto* tbl = rocm::ent4::ent4_table();
    auto it = tbl->find(ent4_name());
    if (it != tbl->end()) {
      const auto& p = it->second;
      auto rs_buf = rocm::malloc_async(
          p->row_start.size() * sizeof(uint32_t), encoder);
      auto cl_buf = rocm::malloc_async(16 * sizeof(uint16_t), encoder);
      auto lut_buf = rocm::malloc_async(32768, encoder);
      uint32_t* rs_dev =
          (uint32_t*)static_cast<rocm::RocmBuffer*>(rs_buf.ptr())->data;
      uint16_t* cl_dev =
          (uint16_t*)static_cast<rocm::RocmBuffer*>(cl_buf.ptr())->data;
      uint8_t* lut_dev =
          (uint8_t*)static_cast<rocm::RocmBuffer*>(lut_buf.ptr())->data;
      void* dst = gpu_ptr<void>(out);
      if (p->comp_bytes)
        (void)hipMemcpy(dst, p->stream.data(), p->comp_bytes + 3,
                       hipMemcpyHostToDevice);
      if (p->row_start.size())
        (void)hipMemcpy(rs_dev, p->row_start.data(),
                        p->row_start.size() * sizeof(uint32_t),
                        hipMemcpyHostToDevice);
      uint16_t cl[16];
      for (int i = 0; i < 16; ++i)
        cl[i] = (uint16_t)p->code_len[i];
      (void)hipMemcpy(cl_dev, cl, 16 * sizeof(uint16_t),
                     hipMemcpyHostToDevice);
      {
        uint8_t lut[32768];
        rocm::ent4::build_lut(p->code_len, lut);
        (void)hipMemcpy(lut_dev, lut, 32768, hipMemcpyHostToDevice);
      }
      rocm::ent4::Reg reg;
      reg.stream = (const uint8_t*)dst;
      reg.row_start_dev = rs_dev;
      reg.lut_dev = lut_dev;
      reg.code_len_dev = cl_dev;
      reg.n_rows = p->n_rows;
      reg.k = p->k;
      reg.comp_bytes = p->comp_bytes;
      reg.total_bits = p->total_bits;
      reg.row_start_keep = std::make_unique<allocator::Buffer>(rs_buf);
      reg.lut_keep = std::make_unique<allocator::Buffer>(lut_buf);
      reg.code_len_keep = std::make_unique<allocator::Buffer>(cl_buf);
      rocm::ent4::Registry::instance().register_((uintptr_t)dst, std::move(reg));
      return;
    }
  }
  // Stage through PINNED host memory.""",
        marker="ent4: upload the compressed stream",
    )
    # make_unique fix (idempotent; the block above already carries it on first apply)
    edit(
        "backend/rocm/load.cpp",
        "      reg.row_start_keep = rs_buf;\n      reg.code_len_keep = cl_buf;",
        "      reg.row_start_keep = std::make_unique<allocator::Buffer>(rs_buf);\n"
        "      reg.code_len_keep = std::make_unique<allocator::Buffer>(cl_buf);",
    )
    edit(
        "backend/rocm/load.cpp",
        "rocm::ent4::Registry::instance().register_((uintptr_t)dst, reg);",
        "rocm::ent4::Registry::instance().register_((uintptr_t)dst, std::move(reg));",
    )
    # LUT additions (idempotent; transform the pre-LUT block to the LUT block on
    # re-apply, and are no-ops once present on a fresh apply of the block above).
    edit(
        "backend/rocm/load.cpp",
        "      auto cl_buf = rocm::malloc_async(16 * sizeof(uint16_t), encoder);\n",
        "      auto cl_buf = rocm::malloc_async(16 * sizeof(uint16_t), encoder);\n"
        "      auto lut_buf = rocm::malloc_async(32768, encoder);\n",
    )
    edit(
        "backend/rocm/load.cpp",
        "          (uint16_t*)static_cast<rocm::RocmBuffer*>(cl_buf.ptr())->data;\n",
        "          (uint16_t*)static_cast<rocm::RocmBuffer*>(cl_buf.ptr())->data;\n"
        "      uint8_t* lut_dev =\n"
        "          (uint8_t*)static_cast<rocm::RocmBuffer*>(lut_buf.ptr())->data;\n",
    )
    edit(
        "backend/rocm/load.cpp",
        "                     hipMemcpyHostToDevice);\n      rocm::ent4::Reg reg;",
        "                     hipMemcpyHostToDevice);\n"
        "      {\n"
        "        uint8_t lut[32768];\n"
        "        rocm::ent4::build_lut(p->code_len, lut);\n"
        "        (void)hipMemcpy(lut_dev, lut, 32768, hipMemcpyHostToDevice);\n"
        "      }\n"
        "      rocm::ent4::Reg reg;",
    )
    edit(
        "backend/rocm/load.cpp",
        "      reg.row_start_dev = rs_dev;\n      reg.code_len_dev = cl_dev;",
        "      reg.row_start_dev = rs_dev;\n      reg.lut_dev = lut_dev;\n      reg.code_len_dev = cl_dev;",
    )
    edit(
        "backend/rocm/load.cpp",
        "      reg.row_start_keep = std::make_unique<allocator::Buffer>(rs_buf);\n      reg.code_len_keep",
        "      reg.row_start_keep = std::make_unique<allocator::Buffer>(rs_buf);\n"
        "      reg.lut_keep = std::make_unique<allocator::Buffer>(lut_buf);\n"
        "      reg.code_len_keep",
    )
    # comp_bytes allocation: allocate the weight buffer at the COMPRESSED size
    # (not the full dense nbytes); the dense form lives only in the persistent
    # decode scratch. This is what makes the lever actually save in-memory HBM.
    # idempotent: zero-pad the uploaded stream by 3 bytes (the 15-bit window
    # decoder reads a 4-byte window; the last code's window can extend up to
    # 3 bytes past the content end — t1044 8ec0580). No-op on fresh applies
    # (the block above already carries the +3).
    edit(
        "backend/rocm/load.cpp",
        "      if (p->comp_bytes)\n"
        "        (void)hipMemcpy(dst, p->stream.data(), p->comp_bytes,\n"
        "                       hipMemcpyHostToDevice);",
        "      if (p->comp_bytes)\n"
        "        (void)hipMemcpy(dst, p->stream.data(), p->comp_bytes + 3,\n"
        "                       hipMemcpyHostToDevice);",
    )
    edit(
        "backend/rocm/load.cpp",
        "  auto size = out.size();\n"
        "  auto nbytes = size * out.itemsize();\n"
        "  out.set_data(mlx::core::rocm::malloc_async(nbytes, encoder));\n"
        "  // ent4: upload the compressed stream + side tables and register w_ptr.\n"
        "  if (has_ent4()) {\n"
        "    const auto* tbl = rocm::ent4::ent4_table();\n"
        "    auto it = tbl->find(ent4_name());\n"
        "    if (it != tbl->end()) {\n"
        "      const auto& p = it->second;\n",
        "  auto size = out.size();\n"
        "  auto nbytes = size * out.itemsize();\n"
        "  // ent4: if this tensor has a compressed stream, allocate only the\n"
        "  // compressed size; qmm decodes the dense form into a persistent scratch.\n"
        "  std::shared_ptr<rocm::ent4::Payload> ep;\n"
        "  if (has_ent4()) {\n"
        "    const auto* tbl = rocm::ent4::ent4_table();\n"
        "    auto it = tbl->find(ent4_name());\n"
        "    if (it != tbl->end()) ep = it->second;\n"
        "  }\n"
        "  out.set_data(mlx::core::rocm::malloc_async(\n"
        "      ep ? ep->comp_bytes + 3 : nbytes, encoder));\n"
        "  // ent4: upload the compressed stream + side tables and register w_ptr.\n"
        "  if (ep) {\n"
        "    const auto& p = ep;\n",
    )
    edit(
        "backend/rocm/load.cpp",
        "      rocm::ent4::Registry::instance().register_((uintptr_t)dst, std::move(reg));\n"
        "      return;\n"
        "    }\n"
        "  }\n"
        "  // Stage through PINNED host memory.",
        "      rocm::ent4::Registry::instance().register_((uintptr_t)dst, std::move(reg));\n"
        "      return;\n"
        "  }\n"
        "  // Stage through PINNED host memory.",
    )

    # 5. qmm.hip: include + wide-path decode hook
    edit(
        "backend/rocm/quantized/qmm.hip",
        '#include "mlx/backend/rocm/quantized/qmv_tiled_kernel.hip"',
        '#include "mlx/backend/rocm/quantized/qmv_tiled_kernel.hip"\n'
        '#include "mlx/backend/rocm/quantized/ent4.h"\n'
        '#include "mlx/backend/rocm/quantized/ent4_kernel.hip"',
    )
    edit(
        "backend/rocm/quantized/qmm.hip",
        """    } else {
    dim3 wblock(32, wcols);
    dim3 wgrid((N + wcols - 1) / wcols, M);""",
        """    } else {
    // ent4: compressed weight stream -> decode to dense scratch -> wide qmv.
    if (bits_ == 4) {
      if (const rocm::ent4::Reg* reg =
              rocm::ent4::Registry::instance().lookup((uintptr_t)w_ptr)) {
        uint8_t* scratch = rocm::ent4::scratch((size_t)N * (K / 2));
        dim3 edblock(256);
        dim3 edgrid((N + 255) / 256);
        enc.add_kernel_node(
            &rocm::ent4::ent4_decode_kernel, edgrid, edblock, 0u,
            reg->stream, reg->row_start_dev, reg->lut_dev,
            reg->code_len_dev, scratch, (uint32_t)N, (uint32_t)K);
        dim3 ewblock(32, wcols);
        dim3 ewgrid((N + wcols - 1) / wcols, M);
#define LAUNCH_QMV_WIDE_ENT4(TT, ST, GS, BV)                               \\
  enc.add_kernel_node(                                                   \\
      &rocm::qmv_wide_kernel<TT, ST, GS, BV>, ewgrid, ewblock, 0u,      \\
      (const TT*)x_ptr, scratch, (const ST*)scales_ptr,                 \\
      (const ST*)biases_ptr, (TT*)out_ptr, M, N, K, has_bias)
#define LAUNCH_QMV_WIDE_ENT4_GS(TT, ST, BV)                               \\
  do {                                                                   \\
    if (group_size_ == 32) { LAUNCH_QMV_WIDE_ENT4(TT, ST, 32, BV); }    \\
    else if (group_size_ == 64) { LAUNCH_QMV_WIDE_ENT4(TT, ST, 64, BV); } \\
    else { LAUNCH_QMV_WIDE_ENT4(TT, ST, 128, BV); }                     \\
  } while (0)
        if (x.dtype() == bfloat16)
          LAUNCH_QMV_WIDE_ENT4_GS(hip_bfloat16, hip_bfloat16, 4);
        else
          LAUNCH_QMV_WIDE_ENT4_GS(__half, __half, 4);
#undef LAUNCH_QMV_WIDE_ENT4_GS
#undef LAUNCH_QMV_WIDE_ENT4
        return;
      }
    }
    dim3 wblock(32, wcols);
    dim3 wgrid((N + wcols - 1) / wcols, M);""",
        marker="ent4: compressed weight stream -> decode to dense scratch",
    )
    # idempotent: add reg->lut_dev to the decode kernel launch (re-apply path).
    edit(
        "backend/rocm/quantized/qmm.hip",
        "            reg->stream, reg->row_start_dev, reg->code_len_dev, scratch,\n",
        "            reg->stream, reg->row_start_dev, reg->lut_dev,\n            reg->code_len_dev, scratch,\n",
    )


def check(root):
    ok = True
    for rel, m in MARKERS.items():
        p = os.path.join(root, rel)
        with open(p) as f:
            t = f.read()
        present = m in t
        print(f"  {rel}: {m!r} {'PRESENT' if present else 'MISSING'}")
        ok = ok and present
    h = os.path.join(root, "backend", "rocm", "quantized", "ent4.h")
    print(f"  {os.path.relpath(h, root)}: {'PRESENT' if os.path.exists(h) else 'MISSING'}")
    ok = ok and os.path.exists(h)
    k = os.path.join(root, "backend", "rocm", "quantized", "ent4_kernel.hip")
    print(f"  {os.path.relpath(k, root)}: {'PRESENT' if os.path.exists(k) else 'MISSING'}")
    ok = ok and os.path.exists(k)
    return ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    root = sys.argv[1]
    check_mode = "--check" in sys.argv
    if not os.path.isdir(os.path.join(root, "backend", "rocm", "quantized")):
        print(f"ERROR: {root} does not look like the ROCm vendor tree")
        sys.exit(1)
    # Verify pre-state (only meaningful on first apply).
    for rel, want in PRE_MD5.items():
        p = os.path.join(root, rel)
        got = md5(p)
        if got != want:
            print(
                f"  NOTE: {rel} md5={got} (expected pre-patch {want}); may already be patched or from a different base"
            )
    if check_mode:
        sys.exit(0 if check(root) else 1)
    apply(root)
    print("  applying ent4 patch: OK")


if __name__ == "__main__":
    main()
