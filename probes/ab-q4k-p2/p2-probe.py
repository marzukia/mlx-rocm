#!/usr/bin/env python3
"""P2 probe (2026-11-27 ruling): their Q4_K dp4a dot (lib-ab3 4738da13) in
isolation at the SIX trio classes (M=512, the SoT shapes). Direct
mmq_ab_gemm ctypes call + refgen2-math CPU ref. corr before any timing.
Per-class lines: P2-<class>-{XGEN,DEQUANT-START,REF-DONE,IDENTITY,HOOK-RC,
SYNC,CORR,OUTSTAT,DONE}. Class = P2_CLASS env, shape = P2_M/P2_N/P2_K env."""
import ctypes, os, sys, time
import numpy as np
_RTLD_NOLOAD = os.RTLD_NOLOAD
D = "/home/andryo/gemm1-leg2"
VP = "/home/andryo/venv-hbmfix"
CLS = os.environ["P2_CLASS"]
M = int(os.environ["P2_M"])
N = int(os.environ["P2_N"])
K = int(os.environ["P2_K"])
TAG = "P2-" + CLS

def f16_to_f32(h):
    h = h.astype(np.uint32)
    sign = (h & 0x8000) << 16
    exp = (h & 0x7C00) >> 10
    mant = h & 0x03FF
    r = np.zeros(h.shape, dtype=np.uint32)
    norm = exp > 0
    r[norm] = sign[norm] | (((exp[norm] - 0x0F) + 0x7F) << 23) | (mant[norm] << 13)
    inf = (exp == 0x1F) & (mant == 0)
    nan = (exp == 0x1F) & (mant != 0)
    r[inf] = sign[inf] | 0x7F800000
    r[nan] = sign[nan] | 0x7FC00000 | (mant[nan] << 13)
    sub = (exp == 0) & (mant != 0)
    m = mant[sub].astype(np.uint32)
    b = sum(((m >= t).astype(np.uint32)) for t in (512, 256, 128, 64, 32, 16, 8, 4, 2))
    r[sub] = sign[sub] | ((b + 103) << 23) | ((m << (23 - b)) - 0x00800000)
    return r.view(np.float32).copy()

def dequant_blocks_np(w):
    d = f16_to_f32(w[:, 0:2].copy().reshape(-1).view(np.uint16))
    dmin = f16_to_f32(w[:, 2:4].copy().reshape(-1).view(np.uint16))
    scales = w[:, 4:16]
    qs = w[:, 16:144]
    out = np.zeros((w.shape[0], 256), dtype=np.float32)
    for e in range(256):
        t = e // 64; l = e % 32; j = e // 32
        code = (qs[:, t * 32 + l] >> (4 * (j % 2))) & 15
        if j < 4:
            sc = (scales[:, j] & 63).astype(np.float32)
            m = (scales[:, j + 4] & 63).astype(np.float32)
        else:
            sc = ((scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)).astype(np.float32)
            m = ((scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)).astype(np.float32)
        out[:, e] = (d * sc) * code.astype(np.float32) - (dmin * m)
    return out

# x: K=4096 classes use the exact refgen2 ref2-x.npy (seed-42 bf16, the P1 input);
# K=12288 class uses the identical refgen2 generation at (512,12288).
if K == 4096:
    x16 = np.load(D + "/ref2-x.npy")
    assert x16.shape == (M, K)
else:
    x = np.random.default_rng(42).normal(size=(M, K)).astype(np.float32)
    xu32 = np.ascontiguousarray(x).view(np.uint32)
    rounded = xu32 + 0x7FFF + ((xu32 >> 16) & 1)
    x16 = ((rounded & 0xFFFF0000) >> 16).astype(np.uint16)
    print("%s-XGEN K=%d seed42-refgen2-math" % (TAG, K), flush=True)

raw = np.fromfile(D + "/bfiledump-N%dk%d.bin" % (N, K), dtype=np.uint8)
NB = K * N // 256
assert raw.size == NB * 144, raw.size
print("%s-DEQUANT-START NB=%d" % (TAG, NB), flush=True)
Wb = dequant_blocks_np(raw.reshape(NB, 144))
W = Wb.reshape(K, N)
xf = (x16.astype(np.uint32) << 16).view(np.float32)
y_ref = np.zeros((M, N), dtype=np.float32)
for c0 in range(0, M, 256):
    y_ref[c0:c0 + 256] = xf[c0:c0 + 256] @ W
print("%s-REF-DONE y_absmax=%.4g" % (TAG, float(np.abs(y_ref).max())), flush=True)

import hashlib
def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b: break
            h.update(b)
    return h.hexdigest()

site = VP + "/lib/python3.14/site-packages"
HIBPATH = "/home/andryo/rcp/new724/tree/opt/rocm-7.2.4/lib/libamdhip64.so.7"
lib = ctypes.CDLL(site + "/mlx/lib/libmlx.so", mode=ctypes.RTLD_GLOBAL)
print("%s-IDENTITY libmlx_md5=%s M=%d N=%d K=%d" % (TAG, md5f(site + "/mlx/lib/libmlx.so"), M, N, K), flush=True)
try:
    hip = ctypes.CDLL(HIBPATH, mode=_RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
except OSError:
    hip = ctypes.CDLL(HIBPATH, mode=ctypes.RTLD_GLOBAL)
for f, a, r in (("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
                ("hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
                ("hipFree", [ctypes.c_void_p], None),
                ("hipStreamSynchronize", [ctypes.c_void_p], ctypes.c_int)):
    fn = getattr(hip, f)
    fn.argtypes = a
    if r: fn.restype = r
lib.mmq_ab_gemm.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_bool,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_void_p]
lib.mmq_ab_gemm.restype = ctypes.c_bool
def alloc(n):
    p = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(p), n) == 0
    return p
x_dev = alloc(x16.nbytes)
w_dev = alloc(raw.nbytes)
o_dev = alloc(M * N * 4)
assert hip.hipMemcpy(x_dev, x16.ctypes.data, x16.nbytes, 1) == 0
assert hip.hipMemcpy(w_dev, raw.ctypes.data, raw.nbytes, 1) == 0
print("%s-LAUNCH t0=%.2f" % (TAG, time.time()), flush=True)
ok = lib.mmq_ab_gemm(x_dev, True, w_dev, o_dev, True, M, N, K, K, M, 2, None)
print("%s-HOOK-RC %s" % (TAG, int(ok)), flush=True)
t0 = time.time()
rc = hip.hipStreamSynchronize(None)
wall = time.time() - t0
print("%s-SYNC rc=%d wall=%.3f" % (TAG, rc, wall), flush=True)
if rc != 0:
    print("%s-FAIL sync rc=%d" % (TAG, rc), flush=True); sys.exit(3)
buf = (ctypes.c_float * (M * N))()
assert hip.hipMemcpy(buf, o_dev, M * N * 4, 2) == 0
out = np.frombuffer(buf, dtype=np.float32).reshape(M, N).copy()
maxrel = float(np.abs(out - y_ref).max() / (np.abs(y_ref).max() + 1e-12))
rms = float(np.sqrt(((out - y_ref) ** 2).mean()) / np.sqrt((y_ref ** 2).mean()))
print("%s-CORR maxrel=%.4g rmsrel=%.4g" % (TAG, maxrel, rms), flush=True)
print("%s-OUTSTAT out_absmax=%.4g frac_zero=%.4g out8=%s y8=%s"
      % (TAG, float(np.abs(out).max()), float((out == 0).mean()),
         np.round(out[0, :8], 4), np.round(y_ref[0, :8], 4)), flush=True)
print("%s-DONE wall=%.3f verdict=%s" % (TAG, wall, "PASS" if maxrel < 2e-2 else "CHECK"), flush=True)
