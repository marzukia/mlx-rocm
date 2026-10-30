"""ent4 verifier: full-stream decode of every tensor in an .ent4 file and
bit-compare against the original packed U32 4bit tensors in the safetensors.

Usage:
  python3 ent4_verify.py /models/qwen3-9b-mlx.ent4 \
      /models/model-00001-of-00002.safetensors /models/model-00002-of-00002.safetensors

Exits non-zero on any mismatch. Also prints per-tensor E[L] and compression
ratios (the measured GB/token numbers for the report).
"""

import ctypes
import json
import mmap
import struct
import sys

import numpy as np


def load_lib():
    import glob
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "ent4_verify.so"), "/tmp/ent4_verify.so", "ent4_verify.so"]
    lib = None
    for c in cands:
        lib = glob.glob(c)
        if lib:
            break
    assert lib, "build with: gcc -O3 -fPIC -shared -o ent4_verify.so ent4_verify.c"
    lib = ctypes.CDLL(lib[0])
    lib.ent4_decode.restype = ctypes.c_int
    lib.ent4_decode.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    return lib


def read_safetensors(path):
    with open(path, "rb") as fh:
        (hl,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hl))
    with open(path, "rb") as fh:
        fh.seek(8 + hl)
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    return hdr, mm


def unpack_u32(w, n, kdiv8):
    b = w.view(np.uint8).reshape(n, kdiv8, 4)
    lo = (b >> 4).astype(np.uint8)
    hi = (b & 0xF).astype(np.uint8)
    return np.stack([lo, hi], axis=-1).reshape(n, kdiv8 * 8)


def main():
    ent4_path, shards = sys.argv[1], sys.argv[2:]
    with open(ent4_path + ".index.json") as _f:
        idx = json.load(_f)
    names = idx["tensors"]
    lib = load_lib()

    sf = {}
    for shard in shards:
        hdr, mm = read_safetensors(shard)
        for name, meta in hdr.items():
            if not isinstance(meta, dict) or meta.get("dtype") != "U32":
                continue
            sf[name] = (hdr, mm, meta)

    with open(ent4_path, "rb") as f:
        magic = f.read(16)
        assert magic == b"MLXENT4V0" + b"\x00" * 7, magic
        ver, n_tensors = struct.unpack("<II", f.read(8))
        assert n_tensors == len(names), (n_tensors, len(names))

        bad = 0
        tot_raw = tot_comp = 0
        for name in names:
            n, k, comp_bytes, total_bits, plain = struct.unpack("<IIII B", f.read(17))
            lengths = np.frombuffer(f.read(16), dtype=np.uint8)
            rowstart = np.frombuffer(f.read(4 * n), dtype=np.uint32).copy()
            # +3 zero bytes: the 15-bit window decoder reads a 4-byte window and
            # the last code's window can extend up to 3 bytes past the stream
            # content end (ported from run-1044 8ec0580).
            stream = np.frombuffer(f.read(comp_bytes) + b"\x00" * 3, dtype=np.uint8)
            out = np.zeros(n * k, dtype=np.uint8)
            ep = ctypes.c_uint64(0)
            er = ctypes.c_uint32(0)
            rc = lib.ent4_decode(
                stream.ctypes.data_as(ctypes.c_char_p),
                n,
                k,
                rowstart.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                lengths.ctypes.data_as(ctypes.c_char_p),
                total_bits,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(ep),
                ctypes.byref(er),
            )
            hdr, mm, meta = sf[name]
            o0, _ = meta["data_offsets"]
            w = np.frombuffer(mm, dtype=np.uint32, count=n * (k // 8), offset=o0)
            orig = unpack_u32(w, n, k // 8).ravel()
            ok = rc == 0 and np.array_equal(out, orig)
            cnt = np.bincount(out, minlength=16)
            e_l = float(cnt @ lengths) / int(cnt.sum())
            raw = n * k // 2
            tot_raw += raw
            tot_comp += comp_bytes
            print(
                f"{'OK ' if ok else 'BAD'} {'PLAIN' if plain else 'COMP '} {name} N={n} K={k} rc={rc} "
                f"E[L]={e_l:.4f} raw={raw / 1e6:.1f}MB comp={comp_bytes / 1e6:.1f}MB",
                flush=True,
            )
            if not ok:
                bad += 1
                first = np.nonzero(out != orig)[0]
                if len(first):
                    i = first[0]
                    print(f"  first diff at {i}: got {out[i]} want {orig[i]} pos={ep.value} row={er.value}", flush=True)
        print(
            f"TOTAL raw={tot_raw / 1e9:.4f}GB comp={tot_comp / 1e9:.4f}GB ratio={tot_comp / tot_raw:.4f} bad={bad}",
            flush=True,
        )
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
