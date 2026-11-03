#!/usr/bin/env python3
"""AR-step-change: qmm.hip dispatch gates (all DEFAULT OFF).

Applies three opt-in gates to the canonical qmm.hip:
  B1  MLX_ROCM_M1_LX=1        M=1 -> int8-dot4 LX path (rows-agnostic)
  B2  MLX_ROCM_M1_ROWSMSG=1   M=1 -> qmv_rows_msg morphology (bit-identical)
  B3  MLX_QMV_WCOLS=n         wide block-height tuning (default 8)
Plus a faithful port of the t1434 REFWD_M1 env (rows-aware, no-op with
QMV_ROWS=1) so the standing config stays semantically identical.
"""
import sys

p = sys.argv[1]
s = open(p).read()
orig = s

# --- Block 1: LX gate + REFWD port + M1_LX + lx_cols guard ---
old1 = """  static const char* lx_env = std::getenv("MLX_QMV_LX");
  if (lx_env != nullptr && (bits_ == 4 || bits_ == 8) &&"""
new1 = """  static const char* lx_env = std::getenv("MLX_QMV_LX");
  // REFWD-M1 (DEFAULT OFF; ported from the t1434 wheel, rows-aware):
  // re-route ONLY the M=1 QMMs to the int8-dot4 LX path when QMV_ROWS is
  // OFF. With MLX_ROCM_QMV_ROWS set (standing config), M=1 stays on the
  // wide default. With the env unset, this term is inert.
  static const bool refwd_m1 =
      (std::getenv("MLX_ROCM_REFWD_M1") != nullptr);
  static const bool rows_active =
      (std::getenv("MLX_ROCM_QMV_ROWS") != nullptr);
  // AR-step-change B1 (DEFAULT OFF): re-route the M=1 (eager decode /
  // correction re-forward) QMMs to the int8-dot4 LX path. Unlike the
  // rows-aware REFWD_M1 gate, M1_LX engages even with MLX_ROCM_QMV_ROWS
  // on - the rows kernel only serves M = 2..4, so the two are disjoint.
  // With MLX_ROCM_M1_LX unset, lx_engage reduces to (lx_env != nullptr) -
  // zero behavior change.
  static const bool m1_lx =
      (std::getenv("MLX_ROCM_M1_LX") != nullptr);
  const bool lx_engage =
      (lx_env != nullptr) ||
      (refwd_m1 && M == 1 && !rows_active) ||
      (m1_lx && M == 1);
  if (lx_engage && (bits_ == 4 || bits_ == 8) &&"""
assert s.count(old1) == 1, "block1 anchor"
s = s.replace(old1, new1)

old1b = """    int lx_cols = (lx_env[0] == '1' && lx_env[1] == '6') ? 16 : 32;"""
new1b = """    // Guard the cols read: lx_env may be null when engaged via
    // REFWD/M1_LX; MLX_QMV_LX=16 still selects 16, otherwise default 32
    // cols.
    int lx_cols = 32;
    if (lx_env != nullptr && lx_env[0] == '1' && lx_env[1] == '6')
      lx_cols = 16;"""
assert s.count(old1b) == 1, "block1b anchor"
s = s.replace(old1b, new1b)

# --- Block 2: rows gate M=1 extension ---
old2 = """      bool rows_ok = use_rows_qmv && (M >= 2) && (M <= 4) &&
          (N >= rows_min_n) && (K % 32 == 0);"""
new2 = """      // AR-step-change B2 (DEFAULT OFF): M=1 on the rows-msg 192-thread /
      // 24-col morphology. The kernel is M-natural (MAXM = 4, `if
      // (m >= M) break`) and the per-row math transcribes qmv_wide
      // verbatim (row-exact contract: output row r ==
      // quantized_matmul(x[r:r+1]) bit for bit), so M=1 is bit-identical
      // to the per-row wide path. M=1 band floor:
      // MLX_ROCM_M1_ROWSMSG_MIN_N (default 0 = all N); the M=2..4 band is
      // unchanged (rows_min_n).
      static bool use_m1_rowsmsg =
          (std::getenv("MLX_ROCM_M1_ROWSMSG") != nullptr);
      static int m1_rowsmsg_min_n =
          parse_non_negative_int_env("MLX_ROCM_M1_ROWSMSG_MIN_N", 0);
      bool rows_ok =
          ((use_rows_qmv && (M >= 2) && (M <= 4) && (N >= rows_min_n)) ||
           (use_m1_rowsmsg && (M == 1) && (N >= m1_rowsmsg_min_n))) &&
          (K % 32 == 0);"""
assert s.count(old2) == 1, "block2 anchor"
s = s.replace(old2, new2)

# --- Block 3: wcols tuning ---
old3 = "    int wcols = hw_info.is_low_cu_igpu ? 4 : 8;"
new3 = """    // AR-step-change B3 (DEFAULT 8 / 4 on low-CU iGPU): wide
    // block-height tuning. The kernel is blockDim.y-agnostic (col =
    // blockIdx.x * blockDim.y + threadIdx.y), so any height >= 4 works.
    static int wcols_env = parse_non_negative_int_env("MLX_QMV_WCOLS", 0);
    int wcols =
        (wcols_env >= 4) ? wcols_env : (hw_info.is_low_cu_igpu ? 4 : 8);"""
assert s.count(old3) == 1, "block3 anchor"
s = s.replace(old3, new3)

open(p, "w").write(s)
assert s != orig
print("PATCHED OK: %s" % p)
