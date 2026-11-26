#!/usr/bin/env python3
"""Block-level d/dmin census for Q2_K (ggml type 10 in the perclass b12062 tree) tensors in a GGUF file.

Usage: python3 q2k_block_census.py <file.gguf> [tensor-name ...]

Counts d/dmin halves that parse as NaN or Inf across all 256-element blocks. The data-section start is padded to the 32-byte GGUF alignment (base = DS + (-DS % 32)); using the unpadded header end shifts every block by the pad and makes qs bytes parse as d/dmin halves (phantom NaN rate ~2.9%).

Exit 0 = no NaN/Inf halves found; 1 = issues found; 2 = usage/read error.
"""
import struct
import sys
import numpy as np


def parse_header(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    f.read(4)
    tc = struct.unpack("<Q", f.read(8))[0]
    kvc = struct.unpack("<Q", f.read(8))[0]
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def rs(t):
        if t == 8:
            m = struct.unpack("<Q", f.read(8))[0]
            f.read(m)
        else:
            f.read(sizes[t])

    for _ in range(kvc):
        nl = struct.unpack("<Q", f.read(8))[0]
        f.read(nl)
        tt = struct.unpack("<I", f.read(4))[0]
        if tt == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if et in sizes:
                f.seek(n * sizes[et], 1)
            else:
                for _ in range(n):
                    rs(et)
        else:
            rs(tt)
    tens = []
    for _ in range(tc):
        nl = struct.unpack("<Q", f.read(8))[0]
        name = f.read(nl).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
        t = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        nel = 1
        for d in dims:
            nel *= d
        tens.append((name, t, off, nel))
    DS = f.tell()
    return f, tens, DS


def census(f, tens, base0, wants):
    tot_blocks = 0
    tot_nan = 0
    tot_inf = 0
    bad = []
    n_tensors = 0
    for (name, t, off, nel) in tens:
        if t != 10:
            continue
        if wants is not None and name not in wants:
            continue
        n_tensors += 1
        nb = nel // 256
        f.seek(base0 + off)
        raw = np.frombuffer(f.read(nb * 84), dtype=np.uint8).copy().reshape(nb, 84)
        dv = np.frombuffer(np.ascontiguousarray(raw[:, 80:82]).tobytes(), dtype=np.float16)
        mn = np.frombuffer(np.ascontiguousarray(raw[:, 82:84]).tobytes(), dtype=np.float16)
        a = int(np.isnan(dv).sum() + np.isnan(mn).sum())
        b = int(np.isinf(dv).sum() + np.isinf(mn).sum())
        tot_blocks += nb
        tot_nan += a
        tot_inf += b
        if a or b:
            bad.append((name, a, b))
    return n_tensors, tot_blocks, tot_nan, tot_inf, bad


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    wants = set(argv[2:]) or None
    try:
        f, tens, DS = parse_header(path)
    except (ValueError, struct.error, OSError) as e:
        print("census: read error: %s" % e)
        return 2
    base0 = DS + (-DS) % 32
    n_tensors, tot_blocks, tot_nan, tot_inf, bad = census(f, tens, base0, wants)
    print("base=%d Q2_K tensors=%d blocks=%d nan-halves=%d inf-halves=%d" % (base0, n_tensors, tot_blocks, tot_nan, tot_inf))
    for x in bad[:10]:
        print("BAD: %s nan=%d inf=%d" % x)
    return 1 if (tot_nan or tot_inf) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
