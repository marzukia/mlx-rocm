#!/usr/bin/env python3
"""t889: apply the t852 LX-hoist patch to the MR2 multi-row QMV kernel.

Usage:
  python3 apply-qmv-mr-lxhoist.py /path/to/vendor-tree          # apply (idempotent)
  python3 apply-qmv-mr-lxhoist.py /path/to/vendor-tree --check # verify state, no write

Conventions (mirrors scripts/apply-m6-patch.py):
  - pattern-based, idempotent (re-running is a no-op);
  - the pre-patch file is backed up as <name>.bak-t889 (the
    qwen3_next.py.bak-t718c convention); the backup is taken once, from
    the current (pre-patch) file, and is never overwritten.

Applies patches/qmv-mr-lxhoist.diff (recorded there as the git diff from
the MR2 lineage state to the final state) to:
  mlx/backend/rocm/quantized/qmv_wide_mr_kernel.hip
      - template parameter bool LXHOIST (A/B: MLX_QMV_MR_LX env at dispatch)
      - __shared__ float s_lx[2][4][32] (256 floats = 1 KB, DOUBLE-BUFFERED
        on k-iteration parity — fixes the single-buffer race where thread
        (c, m) could overwrite s_lx[m][c] in k+1 while another wavefront
        is still reading it in k's m-loop)
      - if constexpr (LXHOIST): per-(row, k-tile) x-chunk-sum hoist
        (32-term 4-bit / 16-term 8-bit FADD chain in the EXACT order of the
        t825 per-row lx) + in-k-loop __syncthreads
      - if constexpr (LXHOIST): m-loop reads s_lx[p][m][lane]; else the
        t825 m-loop verbatim (codegen-equivalent A/B baseline)
  mlx/backend/rocm/quantized/qmm.hip
      - MLX_QMV_MR_LX env (default 0 = t825 path; 1 = hoist on — t903:
        the t889 A/B verdict is a 0.7-31% loss on all 12 MR combos,
        so the t825 path is the default)
      - hoist gate: K % (32 * WPER) == 0 (K % 1024 4-bit / K % 512 8-bit);
        the MR dispatch gate itself is unchanged (shapes that miss the
        hoist gate run the t825 path)
      - launch qmv_wide_mr_kernel<TT, ST, GS, BV, lxhoist>

The pre-patch (MR2 lineage) state is md5-verified:
  qmv_wide_mr_kernel.hip 2352b75c190392e334d915200d971cb0
  qmm.hip                ab6bc086bcc1e1f70745b550d143a018
"""

import hashlib
import sys
from pathlib import Path

P1_old = "template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>\n__global__ void __launch_bounds__(1024) qmv_wide_mr_kernel("  # noqa: E501
P1_new = "template <typename T, typename ScaleT, int GROUP_SIZE, int BITS, bool LXHOIST>\n__global__ void __launch_bounds__(1024) qmv_wide_mr_kernel("  # noqa: E501
P2_old = "// WEIGHT REUSE (t825): the dequantized weight tile is computed ONCE per\n// k-tile (shift+mask+cvt hoisted out of the row loop into a register array)\n// and MAC'd against ALL M rows, instead of re-dequantizing per row. This\n// makes the per-row cost sub-additive (the per-row increment is the MAC only,\n// not dequant+MAC). The per-row math keeps the exact FMA order of\n// qmv_wide_kernel, so every row stays bit-identical to the per-row kernel.\n//\n// Grid: (ceil(N/16), ceil(M/4)); block: (32, 16).\n// LDS: min(4, M) * K * sizeof(T) bytes (32 KB at K = 4096, M = 4).\n//\n// Dispatch gate (qmm.hip): 4/8-bit affine with bias, transposed, non-batched,\n// M <= 4, K % 32 == 0, GROUP_SIZE in {32,64,128}, bf16/fp16 x, wave32,\n// LDS fits (<= 32 KB). M = 5..8 keeps the per-row qmv_wide_kernel.\n"  # noqa: E501
P2_new = "// WEIGHT REUSE (t825): the dequantized weight tile is computed ONCE per\n// k-tile (shift+mask+cvt hoisted out of the row loop into a register array)\n// and MAC'd against ALL M rows, instead of re-dequantizing per row. The\n// per-row math keeps the exact FMA order of qmv_wide_kernel, so every row\n// stays bit-identical to the per-row kernel.\n//\n// LX HOIST (t852): the per-row x-chunk-sum lx = sum_j sx[m*K + k0+j] is\n// COLUMN-INVARIANT (independent of the output column n), but the t825 kernel\n// recomputed it for every one of the block's 16 columns (16x redundancy) and\n// for every row (Mx redundancy). It is now computed ONCE per (row, k-tile)\n// by the lane whose threadIdx.y equals the row (the 32-term 4-bit / 16-term\n// 8-bit FADD chain in the EXACT order of the t825 per-row lx: q-major,\n// j-minor over the chunk, same values from sx), written to a 256-float\n// DOUBLE-BUFFERED LDS buffer (buffer parity = k-iteration parity),\n// __syncthreads, and read by all lanes. The double buffer is required for\n// correctness: with a single buffer, thread (c, m) could overwrite\n// s_lx[m][c] in k-iteration k+1 while another wavefront (c, r') is still\n// reading s_lx[m][c] in k-iteration k's m-loop (different threadIdx.y =\n// different wavefront = different speed; no barrier between the m-loop\n// read and the next hoist write). The double buffer makes iteration k\n// (buffer p) and k+1 (buffer p^1) use disjoint buffers, and the in-k\n// barrier B_{k+1} orders iteration k's read before iteration k+2's write\n// to buffer p. The per-row ALU increment\n// drops from 66 to 34 ops (32 FMA + 2 FMA); the per-row math keeps the exact\n// FMA order of qmv_wide_kernel (same chunk order, same inner unroll, same\n// acc += s*la + b*lx), so every row stays bit-identical to the per-row\n// kernel (lossless gate: maxdiff 0.0). The in-k-loop __syncthreads requires\n// uniform k-iteration counts across lanes: the dispatch gate enables the\n// hoist only when K % (32 * WPER) == 0 (K % 1024 == 0 for 4-bit,\n// K % 512 == 0 for 8-bit); shapes that do not satisfy it run the t825 path\n// (LXHOIST = false, no in-k barrier).\n//\n// A/B (t889): LXHOIST is a template parameter selected at dispatch by\n// MLX_QMV_MR_LX (default 0 = t825 path; 1 = hoist on — the t889 A/B\n// verdict: a 0.7-31% loss on all 12 MR combos). Each instantiation\n// compiles only its m-loop path (if constexpr), so the A/B baseline\n// (LXHOIST = false) is codegen-equivalent to the t825 kernel (the x-staging\n// barrier and the dispatch structure are the only shared code). This makes\n// the gain gate (method doc section 2: gain >= 3% to count) measurable in\n// one session, free of the cross-run drift recorded in REPORT.md section 6.1\n// (17% on an unchanged kernel between back-to-back runs).\n//\n// Grid: (ceil(N/16), ceil(M/4)); block: (32, 16).\n// LDS: min(4, M) * K * sizeof(T) bytes (32 KB at K = 4096, M = 4) + 1 KB\n// (s_lx, double-buffered, always declared; used only by the hoist path).\n// The 32 KB gate bounds the x-staging only; the 1 KB s_lx sits on top\n// (33 KB total at the max shape, well within the 64 KB per-block limit).\n//\n// Dispatch gate (qmm.hip): 4/8-bit affine with bias, transposed, non-batched,\n// M <= 4, K % 32 == 0, GROUP_SIZE in {32,64,128}, bf16/fp16 x, wave32,\n// LDS fits (<= 32 KB). The hoist additionally requires K % (32 * WPER) == 0\n// (K % 1024 == 0 for 4-bit, K % 512 == 0 for 8-bit); the MR dispatch gate\n// itself is unchanged by the hoist. M = 5..8 keeps the per-row\n// qmv_wide_kernel.\n"  # noqa: E501
P3_old = "  T* sx = reinterpret_cast<T*>(sx_bytes);  // [nrows][K]\n  int lane = threadIdx.x;"
P3_new = "  T* sx = reinterpret_cast<T*>(sx_bytes);  // [nrows][K]\n  // LX HOIST (t852): per-row, per-lane x-chunk-sum buffer (256 floats =\n  // 1 KB, DOUBLE-BUFFERED on k-iteration parity). Written once per (row,\n  // k-tile) by the thread (threadIdx.x == lane, threadIdx.y == row); read\n  // by all threads after __syncthreads(). The double buffer avoids the\n  // single-buffer race (thread (c, m) overwriting s_lx[m][c] in k+1 while\n  // another wavefront still reads it in k's m-loop). Declared\n  // unconditionally (1 KB, always fits); used only by the hoist path.\n  __shared__ float s_lx[2][MAXM][32];\n  int lane = threadIdx.x;"  # noqa: E501
P4b_old = "  for (int k = lane * WPER; k < K; k += 32 * WPER) {\n    int g = k / GROUP_SIZE;"
P4b_new = "  for (int k = lane * WPER; k < K; k += 32 * WPER) {\n    // Double-buffer parity (uniform across lanes: k - lane*WPER is a\n    // multiple of 32*WPER, so the division is exact and lane-invariant).\n    int p = ((k - lane * WPER) / (32 * WPER)) & 1;\n    int g = k / GROUP_SIZE;"  # noqa: E501
P4_old = "    for (int m = 0; m < MAXM; ++m) {\n      if (m >= nrows)\n        break;\n      float la = 0.0f, lx = 0.0f;\n      if constexpr (BITS == 4) {\n#pragma unroll\n        for (int q = 0; q < 4; ++q) {\n          int kb = k + q * 8;\n#pragma unroll\n          for (int j = 0; j < 8; ++j) {\n            float xv = static_cast<float>(sx[m * K + kb + j]);\n            la += xv * wn[q * 8 + j];\n            if (has_bias)\n              lx += xv;\n          }\n        }\n      } else {\n#pragma unroll\n        for (int q = 0; q < 4; ++q) {\n          int kb = k + q * 4;\n#pragma unroll\n          for (int j = 0; j < 4; ++j) {\n            float xv = static_cast<float>(sx[m * K + kb + j]);\n            la += xv * wn[q * 4 + j];\n            if (has_bias)\n              lx += xv;\n          }\n        }\n      }\n      acc[m] += s * la + b * lx;\n    }\n"  # noqa: E501
P4_new = "    if constexpr (LXHOIST) {\n      // LX HOIST (t852): compute the column-invariant per-row x-chunk-sum\n      // ONCE per (row, k-tile). The FADD chain is the EXACT order of the\n      // t825 per-row lx (q-major, j-minor over the 32/16-value chunk, same\n      // values from sx), so the value is bit-identical to the t825 per-row\n      // lx. The thread (lane, row) writes s_lx[row][lane]; all threads read\n      // it after the barrier. When !has_bias the consumer multiplies by\n      // b == 0.0f, so 0.0f * (finite chain) == 0.0f == the t825\n      // (unaccumulated) lx — the result is bit-identical in both cases.\n      {\n        int ly = threadIdx.y;\n        if (ly < nrows) {\n          float lx = 0.0f;\n          if constexpr (BITS == 4) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q)\n#pragma unroll\n              for (int j = 0; j < 8; ++j)\n                lx += static_cast<float>(sx[ly * K + k + q * 8 + j]);\n          } else {\n#pragma unroll\n            for (int q = 0; q < 4; ++q)\n#pragma unroll\n              for (int j = 0; j < 4; ++j)\n                lx += static_cast<float>(sx[ly * K + k + q * 4 + j]);\n          }\n          s_lx[p][ly][lane] = lx;\n        }\n      }\n      __syncthreads();\n    }\n    if constexpr (LXHOIST) {\n      for (int m = 0; m < MAXM; ++m) {\n        if (m >= nrows)\n          break;\n        float la = 0.0f;\n        if constexpr (BITS == 4) {\n#pragma unroll\n          for (int q = 0; q < 4; ++q) {\n            int kb = k + q * 8;\n#pragma unroll\n            for (int j = 0; j < 8; ++j)\n              la += static_cast<float>(sx[m * K + kb + j]) * wn[q * 8 + j];\n          }\n        } else {\n#pragma unroll\n          for (int q = 0; q < 4; ++q) {\n            int kb = k + q * 4;\n#pragma unroll\n            for (int j = 0; j < 4; ++j)\n              la += static_cast<float>(sx[m * K + kb + j]) * wn[q * 4 + j];\n          }\n        }\n        acc[m] += s * la + b * s_lx[p][m][lane];\n      }\n    } else {\n      // t825 path (A/B baseline): per-row lx accumulated in the same inner\n      // loop (EXACT t825 order: la FMA then lx FADD per value, guarded by\n      // has_bias), so this instantiation is codegen-equivalent to the t825\n      // kernel.\n      for (int m = 0; m < MAXM; ++m) {\n        if (m >= nrows)\n          break;\n        float la = 0.0f, lx = 0.0f;\n        if constexpr (BITS == 4) {\n#pragma unroll\n          for (int q = 0; q < 4; ++q) {\n            int kb = k + q * 8;\n#pragma unroll\n            for (int j = 0; j < 8; ++j) {\n              float xv = static_cast<float>(sx[m * K + kb + j]);\n              la += xv * wn[q * 8 + j];\n              if (has_bias)\n                lx += xv;\n            }\n          }\n        } else {\n#pragma unroll\n          for (int q = 0; q < 4; ++q) {\n            int kb = k + q * 4;\n#pragma unroll\n            for (int j = 0; j < 4; ++j) {\n              float xv = static_cast<float>(sx[m * K + kb + j]);\n              la += xv * wn[q * 4 + j];\n              if (has_bias)\n                lx += xv;\n            }\n          }\n        }\n        acc[m] += s * la + b * lx;\n      }\n    }\n"  # noqa: E501

P5_old = '      static int mr_cols_env = parse_non_negative_int_env("MLX_QMV_MR_COLS", 0);\n      int mcols = (mr_cols_env >= 8) ? mr_cols_env : 16;\n      size_t mr_lds = (size_t)std::min(4, M) * (size_t)K * 2;\n      bool mr_ok = use_mr_qmv && (M <= 4) && (mr_lds <= 32768);\n      if (!mr_ok) {'  # noqa: E501
P5_new = '      static int mr_cols_env = parse_non_negative_int_env("MLX_QMV_MR_COLS", 0);\n      // t889: A/B toggle for the t852 LX hoist (default 0 = t825 path;\n      // 1 = hoist — the same kernel with the in-k barrier; the t889 A/B\n      // verdict: a 0.7-31% loss on all 12 MR combos, so the t825 path\n      // is the default).\n      static int mr_lx_env = parse_non_negative_int_env("MLX_QMV_MR_LX", 0);\n      int mcols = (mr_cols_env >= 8) ? mr_cols_env : 16;\n      size_t mr_lds = (size_t)std::min(4, M) * (size_t)K * 2;\n      bool mr_ok = use_mr_qmv && (M <= 4) && (mr_lds <= 32768);\n      // t852: the MR kernel\'s in-k-loop __syncthreads (LX hoist) requires\n      // uniform k-iteration counts across lanes: K % (32 * WPER) == 0\n      // (K % 1024 == 0 for 4-bit, K % 512 == 0 for 8-bit). Shapes that do\n      // not satisfy it (or MLX_QMV_MR_LX=0) run the t825 path\n      // (LXHOIST = false) — the MR dispatch gate itself is unchanged.\n      bool lxhoist = (mr_lx_env != 0) &&\n          (K % ((bits_ == 4) ? 1024 : 512) == 0);\n      if (!mr_ok) {'  # noqa: E501
P6_old = "#define LAUNCH_QMV_WIDE_MR(TT, ST, GS, BV)                                   \\\n  enc.add_kernel_node(                                                      \\\n      &rocm::qmv_wide_mr_kernel<TT, ST, GS, BV>, mrgrid, mrblock,           \\\n      (uint32_t)mr_lds, (const TT*)x_ptr, w_ptr, (const ST*)scales_ptr,     \\\n      (const ST*)biases_ptr, (TT*)out_ptr, M, N, K, has_bias)\n#define LAUNCH_QMV_WIDE_MR_GS(TT, ST, BV)                                    \\\n  do {                                                                      \\\n    if (group_size_ == 32) { LAUNCH_QMV_WIDE_MR(TT, ST, 32, BV); }         \\\n    else if (group_size_ == 64) { LAUNCH_QMV_WIDE_MR(TT, ST, 64, BV); }    \\\n    else { LAUNCH_QMV_WIDE_MR(TT, ST, 128, BV); }                          \\\n  } while (0)\n      if (x.dtype() == bfloat16) {\n        if (bits_ == 4) { LAUNCH_QMV_WIDE_MR_GS(hip_bfloat16, hip_bfloat16, 4); }\n        else { LAUNCH_QMV_WIDE_MR_GS(hip_bfloat16, hip_bfloat16, 8); }\n      } else {\n        if (bits_ == 4) { LAUNCH_QMV_WIDE_MR_GS(__half, __half, 4); }\n        else { LAUNCH_QMV_WIDE_MR_GS(__half, __half, 8); }\n      }"  # noqa: E501
P6_new = "#define LAUNCH_QMV_WIDE_MR(TT, ST, GS, BV, LX)                               \\\n  enc.add_kernel_node(                                                      \\\n      &rocm::qmv_wide_mr_kernel<TT, ST, GS, BV, LX>, mrgrid, mrblock,     \\\n      (uint32_t)mr_lds, (const TT*)x_ptr, w_ptr, (const ST*)scales_ptr,     \\\n      (const ST*)biases_ptr, (TT*)out_ptr, M, N, K, has_bias)\n#define LAUNCH_QMV_WIDE_MR_GS(TT, ST, BV, LX)                               \\\n  do {                                                                      \\\n    if (group_size_ == 32) { LAUNCH_QMV_WIDE_MR(TT, ST, 32, BV, LX); }     \\\n    else if (group_size_ == 64) { LAUNCH_QMV_WIDE_MR(TT, ST, 64, BV, LX); } \\\n    else { LAUNCH_QMV_WIDE_MR(TT, ST, 128, BV, LX); }                      \\\n  } while (0)\n      // LXHOIST is a template parameter (compile-time), so the runtime\n      // lxhoist value selects between the two instantiations (true = t852\n      // hoist, false = t825 path).\n      if (x.dtype() == bfloat16) {\n        if (bits_ == 4) {\n          if (lxhoist) { LAUNCH_QMV_WIDE_MR_GS(hip_bfloat16, hip_bfloat16, 4, true); }\n          else { LAUNCH_QMV_WIDE_MR_GS(hip_bfloat16, hip_bfloat16, 4, false); }\n        } else {\n          if (lxhoist) { LAUNCH_QMV_WIDE_MR_GS(hip_bfloat16, hip_bfloat16, 8, true); }\n          else { LAUNCH_QMV_WIDE_MR_GS(hip_bfloat16, hip_bfloat16, 8, false); }\n        }\n      } else {\n        if (bits_ == 4) {\n          if (lxhoist) { LAUNCH_QMV_WIDE_MR_GS(__half, __half, 4, true); }\n          else { LAUNCH_QMV_WIDE_MR_GS(__half, __half, 4, false); }\n        } else {\n          if (lxhoist) { LAUNCH_QMV_WIDE_MR_GS(__half, __half, 8, true); }\n          else { LAUNCH_QMV_WIDE_MR_GS(__half, __half, 8, false); }\n        }\n      }"  # noqa: E501


BASE_MD5 = {
    "qmv_wide_mr_kernel.hip": "2352b75c190392e334d915200d971cb0",
    "qmm.hip": "ab6bc086bcc1e1f70745b550d143a018",
}

KERNEL_PAIRS = [
    ("P1 template line", P1_old, P1_new),
    ("P2 header comment", P2_old, P2_new),
    ("P3 s_lx declaration", P3_old, P3_new),
    ("P4b double-buffer parity", P4b_old, P4b_new),
    ("P4 hoist block + m-loop", P4_old, P4_new),
]
QMM_PAIRS = [
    ("P5 dispatch env + gate", P5_old, P5_new),
    ("P6 launch template arg", P6_old, P6_new),
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def apply_file(p: Path, pairs: list) -> None:
    s = p.read_text()
    done = all(new in s for _, _, new in pairs)
    if done:
        print(f"already patched: {p}")
        return
    missing = [name for name, _, new in pairs if new not in s]
    for name, old, new in pairs:
        if new in s:
            continue
        if old not in s:
            raise SystemExit(f"PATCH FAIL ({name}): pattern not found in {p}")
        s = s.replace(old, new, 1)
    bak = p.with_name(p.name + ".bak-t889")
    if not bak.exists():
        bak.write_text(p.read_text())
        print(f"backup: {bak}")
    p.write_text(s)
    print(f"patched: {p} ({', '.join(missing)})")


def check_file(p: Path, pairs: list) -> bool:
    s = p.read_text()
    return all(new in s for _, _, new in pairs)


def main() -> None:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--check"):
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    k = root / "mlx" / "backend" / "rocm" / "quantized" / "qmv_wide_mr_kernel.hip"
    q = root / "mlx" / "backend" / "rocm" / "quantized" / "qmm.hip"
    for p in (k, q):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if len(sys.argv) == 3:
        ok = check_file(k, KERNEL_PAIRS) and check_file(q, QMM_PAIRS)
        print(f"{'PATCHED' if ok else 'NOT-PATCHED'}: {k}")
        print(f"{'PATCHED' if ok else 'NOT-PATCHED'}: {q}")
        sys.exit(0 if ok else 1)
    # md5 guard: only apply from the verified MR2 lineage state (or the
    # already-patched state); a .bak-t889 may exist from a previous run.
    for p, want, pairs in (
        (k, BASE_MD5["qmv_wide_mr_kernel.hip"], KERNEL_PAIRS),
        (q, BASE_MD5["qmm.hip"], QMM_PAIRS),
    ):
        if p.with_name(p.name + ".bak-t889").exists():
            continue  # already patched in a previous run; idempotent
        got = md5(p)
        if got != want and not all(nw in p.read_text() for _, _, nw in pairs):
            raise SystemExit(
                f"MD5-GUARD: {p.name} = {got}, expected {want} (the MR2 lineage state) or the patched state"
            )
    apply_file(k, KERNEL_PAIRS)
    apply_file(q, QMM_PAIRS)


if __name__ == "__main__":
    main()
