"""t1354 CPU unit gate for the rows-tuning kernel variants (ws32 / kb2 /
warps).

Mirrors the exact float32 arithmetic of the `qmv_rows_msg_*` kernels
(base, ws32, kb2) in the lane / row / column structure, and checks:

  (a) the ws32 and kb2 variants are BIT-IDENTICAL to the base kernel
      (maxdiff 0.0 — the knob contract; the GPU round re-verifies on the
      wheel with the same 1e-2 gate),
  (b) the base kernel matches the scalar reference (per-output fp32
      sequential dequant-MAC) within maxdiff <= 1e-2 (accumulation-order
      noise only: the kernel sums 32-lane partials in the 5-stage
      butterfly; the reference in k order),
  (c) the per-(lane, k-tile) weight-decode ALU count is 512 (base inline
      at M=4: 4 rows x 4 cols x 32) vs 128 (ws32 pre-decoded: 4 cols x
      32) at 4-bit — the FMA-analysis premise the expected-ms table in
      REPORT-rows-tuning.md rests on (pinned in the source-shape test).

No GPU needed, zero deps (pure Python; every kernel op is one
struct-packed float32 op, in kernel order, so the comparison is at
floating-point precision, not just statistical).

Weight packing (mirrors the kernel load): the 128-bit pack-interleaved
load is at byte offset (k * BITS) / 8; the 4 words decode
q-major / j-minor:
  4-bit: out[q * 8 + j] = (word_q >> 4j) & 0xF  = element k + q * 8 + j
  8-bit: out[q * 4 + j] = (word_q >> 8j) & 0xFF = element k + q * 4 + j
i.e. decoded value j is element k + j in both bit-widths (the 8-bit 16B
load spans exactly the 16-element tile).
"""

import random
import struct


def f32(x):
    """One IEEE-754 single-precision rounding (a kernel float op)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _lane_tiles_base(K, WPER):
    """Base / ws32 k-tile k-starts, per lane (k = lane*WPER + i*32*WPER)."""
    tiles = []
    for i in range((K - 1) // (32 * WPER) + 1):
        for lane in range(32):
            k = lane * WPER + i * 32 * WPER
            if k < K:
                tiles.append((lane, k))
    return tiles


def _lane_tiles_kb2(K, WPER):
    """kb2 k-tile k-starts, per lane: the lane's tiles 2i and 2i+1 of the
    base sequence per step (k = lane*WPER + (2i + t)*32*WPER, t = 0 then 1).

    Must produce the SAME k sequence as _lane_tiles_base (per lane, in
    order) — that is the kb2 bit-exactness argument (2x ILP: both tiles'
    128-bit weight loads are in flight before the first FMA).
    """
    tiles = []
    for i in range((K - 1) // (64 * WPER) + 1):
        for lane in range(32):
            for t in range(2):
                k = lane * WPER + (2 * i + t) * 32 * WPER
                if k < K:
                    tiles.append((lane, k))
    return tiles


def _decode(w4_row, k, BITS):
    """The shift+mask decode chain (q-major, j-minor), float32-cast.

    Mirrors the 128-bit pack-interleaved load at byte offset
    (k * BITS) / 8: word q = the 8 (4-bit) / 4 (8-bit) elements
    k + q * nvals .. k + q * nvals + nvals - 1 packed little-endian.
    """
    nvals = 8 if BITS == 4 else 4
    out = []
    for q in range(4):
        w = 0
        for i in range(nvals):
            w |= int(w4_row[k + q * nvals + i]) << (BITS * i)
        shift = 4 if BITS == 4 else 8
        mask = 0xF if BITS == 4 else 0xFF
        for j in range(nvals):
            out.append(f32((w >> (shift * j)) & mask))
    return out  # decoded value j = element k + j (both bit-widths)


def _kernel(x, w4, scales, biases, has_bias, M, N, K, BITS, GROUP_SIZE, variant, warps=6):
    """One kernel's fp32 output (pre bf16 cast), lane-structure mirror.

    `variant` in {"base", "ws32", "kb2"}. `warps` only changes the block
    packing (col0 = block * (warps * 4) + wid * 4); the per-(wave, lane,
    col) math is identical, so the output is packing-invariant — the
    assertion the WARPS knob's bit-exactness rests on.
    """
    WPER = 128 // BITS
    tiles = _lane_tiles_base(K, WPER) if variant in ("base", "ws32") else _lane_tiles_kb2(K, WPER)
    tiles_by_lane = {}
    for lane, k in tiles:
        tiles_by_lane.setdefault(lane, []).append(k)

    ng = (K + GROUP_SIZE - 1) // GROUP_SIZE
    out = [[0.0] * N for _ in range(M)]
    for col0 in range(0, N, warps * 4):
        for wid in range(warps):
            for c in range(4):
                col = col0 + wid * 4 + c
                if col >= N:
                    continue
                # acc[m][lane]
                acc = [[0.0] * 32 for _ in range(M)]
                for lane in range(32):
                    a = [0.0] * M
                    for k in tiles_by_lane.get(lane, []):
                        g = k // GROUP_SIZE
                        if g >= ng:
                            continue
                        s = f32(scales[col][g])
                        b = f32(biases[col][g]) if has_bias else f32(0.0)
                        # The per-tile weight decode (32 at 4-bit / 16 at
                        # 8-bit shift+mask+cvt values). ws32 decodes per
                        # column ONCE before the row loop; base decodes
                        # per (row, column) — same values, so same wdec.
                        wdec = _decode(w4[col], k, BITS)
                        for m in range(M):
                            xn = [f32(x[m][k + j]) for j in range(WPER)]
                            lx = f32(0.0)
                            for j in range(WPER):
                                lx = f32(lx + xn[j])
                            la = f32(0.0)
                            for j in range(WPER):
                                la = f32(la + f32(xn[j] * wdec[j]))
                            # acc += s * la + b * lx  (kernel op order)
                            t1 = f32(s * la)
                            t2 = f32(b * lx)
                            a[m] = f32(a[m] + f32(t1 + t2))
                    for m in range(M):
                        acc[m][lane] = a[m]
                # the 5-stage shfl_down butterfly, float32
                for m in range(M):
                    v = acc[m][:]
                    for o in (16, 8, 4, 2, 1):
                        v = [f32(v[i] + v[(i + o) % 32]) for i in range(32)]
                    out[m][col] = v[0]
    return out


def _scalar_ref(x, w4, scales, biases, has_bias, M, N, K, BITS, GROUP_SIZE):
    """Scalar reference: out[m, n] = sum over k in k order of
    x[m, k] * (s_g * w4[n, k] + b_g), float32 one op at a time."""
    out = [[0.0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            acc = f32(0.0)
            for k in range(K):
                g = k // GROUP_SIZE
                s = f32(scales[n][g])
                b = f32(biases[n][g]) if has_bias else f32(0.0)
                wq = f32(w4[n][k])
                acc = f32(acc + f32(x[m][k] * f32(f32(s * wq) + b)))
            out[m][n] = acc
    return out


def _maxdiff(a, b):
    md = 0.0
    for i in range(len(a)):
        for j in range(len(a[0])):
            d = abs(a[i][j] - b[i][j])
            if d > md:
                md = d
    return md


def _make(M, N, K, BITS, GROUP_SIZE, seed=0, has_bias=True):
    rs = random.Random(seed)
    xmax = 16 if BITS == 4 else 256
    w4 = [[rs.randint(0, xmax - 1) for _ in range(K)] for _ in range(N)]
    ng = (K + GROUP_SIZE - 1) // GROUP_SIZE
    scales = [[f32(rs.uniform(0.001, 0.01)) for _ in range(ng)] for _ in range(N)]
    biases = [[f32(rs.uniform(-0.5, 0.5)) for _ in range(ng)] for _ in range(N)]
    x = [[f32(rs.uniform(-1.0, 1.0)) for _ in range(K)] for _ in range(M)]
    return x, w4, scales, biases


def test_kb2_tile_sequence_is_the_base_sequence():
    """The kb2 k sequence equals the base k sequence, per lane, in order."""
    for K, BITS in ((4096, 4), (12288, 4), (4096, 8), (512, 4), (512, 8)):
        WPER = 128 // BITS
        b = _lane_tiles_base(K, WPER)
        k2 = _lane_tiles_kb2(K, WPER)
        b_by_lane = {}
        for lane, k in b:
            b_by_lane.setdefault(lane, []).append(k)
        k2_by_lane = {}
        for lane, k in k2:
            k2_by_lane.setdefault(lane, []).append(k)
        assert b_by_lane == k2_by_lane, (K, BITS)


def test_variants_bit_identical_to_base():
    """ws32 / kb2 (and warps=4 packing) == base, maxdiff 0.0 (gate <= 1e-2)."""
    cases = [
        # M, N, K, BITS, GROUP_SIZE, has_bias (model shape + band edges)
        (4, 24, 512, 4, 64, True),
        (4, 24, 512, 8, 64, True),
        (2, 24, 512, 4, 32, True),
        (3, 24, 512, 4, 128, False),
        (4, 128, 4096, 4, 64, True),
        (4, 25, 1024, 4, 64, True),  # N not a multiple of 24 (col guard)
    ]
    for M, N, K, BITS, GS, hb in cases:
        x, w4, s, b = _make(M, N, K, BITS, GS, has_bias=hb)
        base = _kernel(x, w4, s, b, hb, M, N, K, BITS, GS, "base")
        for variant in ("ws32", "kb2"):
            v = _kernel(x, w4, s, b, hb, M, N, K, BITS, GS, variant)
            assert _maxdiff(v, base) == 0.0, (variant, M, N, K, BITS, GS)
        w4p = _kernel(x, w4, s, b, hb, M, N, K, BITS, GS, "base", warps=4)
        assert _maxdiff(w4p, base) == 0.0, (M, N, K)


def test_base_within_1e2_of_scalar_reference():
    """base vs the scalar reference: maxdiff <= 1e-2 (order noise only)."""
    for M, N, K, BITS, GS, hb in (
        (4, 24, 512, 4, 64, True),
        (4, 24, 512, 8, 64, False),
        (4, 24, 1024, 4, 128, True),
        (2, 24, 512, 4, 32, True),
    ):
        x, w4, s, b = _make(M, N, K, BITS, GS, has_bias=hb)
        base = _kernel(x, w4, s, b, hb, M, N, K, BITS, GS, "base")
        ref = _scalar_ref(x, w4, s, b, hb, M, N, K, BITS, GS)
        md = _maxdiff(base, ref)
        assert md <= 1e-2, (md, M, N, K, BITS, GS)


def test_source_shape_decodes_and_warps_arg():
    """The FMA-analysis premise pinned in source shape:
    - base: the shift+mask decode chain sits INSIDE the per-(m, c) MAC
      loop (512 decodes per (lane, k-tile) at M=4);
    - ws32: one pre-decode per column into wdec[c][...] (128 per (lane,
      k-tile) at 4-bit) and the MAC loop reads wdec (no shift+mask);
    - kb2: two tiles per lane step, both guards, k = k0 + t*WPER;
    - all four launch forms in the guest patch pass rows_warps (6 base /
      4) and the knob defaults keep the base path (WS 4 / KBLOCK 1 /
      WARPS 6)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    d = root / "vendor" / "mlx-rocm-upstream" / "mlx" / "backend" / "rocm" / "quantized"
    ktext = (d / "qmv_rows_msg_kernel.hip").read_text()
    # the dispatch lives in the guest patch (repo qmm.hip is pre-dispatch
    # by repo convention — the vendored tree is not in the repo)
    qtext = (root / "patches" / "qmv-rows-tuning.diff").read_text()
    i_base = ktext.index("qmv_rows_msg_kernel(")
    i_ws32 = ktext.index("qmv_rows_msg_ws32_kernel(")
    i_kb2 = ktext.index("qmv_rows_msg_kb2_kernel(")
    base = ktext[i_base:i_ws32]
    ws32 = ktext[i_ws32:i_kb2]
    kb2 = ktext[i_kb2:]
    # base: the decode chain is inside the per-(m,c) MAC loop
    assert ">> (4 * j)" in base and ">> (8 * j)" in base
    # ws32: pre-decode into wdec[c][...]; the MAC loop reads it
    assert "wdec[c][q * 8 + j] =" in ws32
    # kb2: two tiles per step (tiles 2i, 2i+1 of the base sequence)
    # with the per-tile guard
    assert "k0 + t * 32 * WPER" in kb2 and "if (k >= K)" in kb2
    # all four launch forms pass the runtime warps arg (patch + lines)
    for k in (
        "qmv_rows_msg_kernel<",
        "qmv_rows_msg_hoist_kernel<",
        "qmv_rows_msg_ws32_kernel<",
        "qmv_rows_msg_kb2_kernel<",
    ):
        assert k in qtext, k
    assert qtext.count("has_bias, rows_warps)") == 4
    assert "int warps = 6" in ktext
    # knob defaults: all unset -> base path (warps 6, WS 4, KBLOCK 1)
    assert 'MLX_QMV_ROWS_WS", 4)' in qtext
    assert 'MLX_QMV_ROWS_KBLOCK", 1)' in qtext
    assert 'MLX_QMV_ROWS_WARPS", 6)' in qtext
    assert "rows_ws32)" in qtext and "rows_kb2)" in qtext
