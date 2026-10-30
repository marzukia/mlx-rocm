"""Offline ent4 compressor: 4bit safetensors tensors -> per-tensor shared
canonical-Huffman compressed stream (docs/entropy4.md format spec).

CPU-only, no GPU. Reads the packed U32 [N, K/8] 4bit code tensors out of the
safetensors shards, builds one canonical Huffman codebook per tensor, encodes
the code stream bit-exactly, and writes the .ent4 side file.

Usage:
  python3 ent4_compress.py /models/model-0000X-of-NNNNN.safetensors ... \
      --out /models/qwen3-9b-mlx.ent4

Lossless by construction: the decoder reconstructs the exact 4bit code values;
the downstream dequant (scales/biases, untouched) is then bit-identical.
"""

import argparse
import heapq
import json
import mmap
import os
import struct
import time

import numpy as np

MAGIC = b"MLXENT4V0" + b"\x00" * 7
VER = 0


def read_safetensors(path):
    with open(path, "rb") as fh:
        (hl,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hl))
    with open(path, "rb") as fh:
        fh.seek(8 + hl)
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    return hdr, mm


def unpack_u32(w, n, kdiv8):
    """[N, K/8] u32 -> [N, K] int8 codes. Code j of the u32 = (w >> 4j) & 0xF,
    i.e. byte j//2, low nibble first — the qmv read order."""
    b = w.view(np.uint8).reshape(n, kdiv8, 4)
    lo = (b >> 4).astype(np.uint8)
    hi = (b & 0xF).astype(np.uint8)
    return np.stack([lo, hi], axis=-1).reshape(n, kdiv8 * 8)


def huffman_lengths(counts):
    """Classic Huffman; returns 16 code lengths (0 for absent symbols)."""
    cnt = [int(c) for c in counts]
    present = [i for i in range(16) if cnt[i] > 0]
    if not present:
        return [0] * 16
    if len(present) == 1:
        return [0 if cnt[i] == 0 else 1 for i in range(16)]
    heap = [(cnt[i], i) for i in present]
    heapq.heapify(heap)
    nxt = 16
    parent = {}
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent[a[1]] = nxt
        parent[b[1]] = nxt
        heapq.heappush(heap, (a[0] + b[0], nxt))
        nxt += 1
    out = [0] * 16
    for i in present:
        d = 0
        cur = i
        while cur in parent:
            cur = parent[cur]
            d += 1
        out[i] = d
    return out


def canonical_codes(lengths):
    """Canonical Huffman codes: symbols sorted by (length, symbol); the running
    code shifts left by (ln - prev_len) between length changes."""
    codes = [0] * 16
    code = 0
    prev_len = 0
    for ln, sym in sorted((lg, i) for i, lg in enumerate(lengths) if lg > 0):
        code <<= ln - prev_len
        codes[sym] = code
        code += 1
        prev_len = ln
    return codes


def _bits_to_bytes(bits):
    """Flat MSB-first bit array -> bytes. Bit index 0 = MSB of byte 0 (the
    decoder reads bytes MSB-first). Pads the tail with zeros."""
    n = len(bits)
    padded = (n + 7) // 8 * 8
    if padded != n:
        bits = np.concatenate([bits, np.zeros(padded - n, dtype=np.uint8)])
    rows = bits.reshape(-1, 8)
    out = np.zeros(rows.shape[0], dtype=np.uint8)
    for i in range(8):
        out |= rows[:, i] << (7 - i)
    return out.tobytes()


def _rss_mb():
    """Process RSS in MB (diagnostics; ported from t1044 764fc08)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as psf:
            for line in psf:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _class_of(name):
    if "lm_head" in name or "embed" in name:
        return "el"
    if "attn" in name:
        return "attn"
    return "mlp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--gate",
        type=float,
        default=3.9,
        help="compress only tensors with measured E < gate; others stay plain 4-bit "
        "(flag byte in the per-tensor header). 0 = compress all (legacy).",
    )
    args = ap.parse_args()
    try:
        _main(args)
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def _main(args):
    # pass 1: enumerate U32 tensors from headers (no data loads)
    entries = []
    for shard in args.shards:
        hdr, _ = read_safetensors(shard)
        for name, meta in hdr.items():
            if name == "__metadata__" or meta["dtype"] != "U32":
                continue
            n, kdiv8 = meta["shape"]
            entries.append((shard, name, n, kdiv8))

    print(f"{len(entries)} 4bit tensors", flush=True)
    total_raw = total_comp = 0
    names = []
    tensor_info = []
    class_stat = {}  # class -> [tensors, plain, measured_bits, written_bits]
    t_tensor0 = time.time()
    with open(args.out, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", VER, len(entries)))
        for i, (shard, name, n, kdiv8) in enumerate(entries):
            t0 = time.time()
            hdr, mm = read_safetensors(shard)
            o0, _ = hdr[name]["data_offsets"]
            w = np.frombuffer(mm, dtype=np.uint32, count=n * kdiv8, offset=o0)
            codes = unpack_u32(w, n, kdiv8)  # [n, k] uint8
            k = kdiv8 * 8
            counts = np.bincount(codes.ravel(), minlength=16)
            lengths = huffman_lengths(counts)
            cvec = np.array(lengths, dtype=np.int64)
            e_l = float(counts @ cvec) / int(counts.sum())
            # Per-tensor gating (decision C): if the measured entropy is above
            # the gate, a canonical 4-bit encoding (identity symbols, lengths
            # all 4, stream = raw packed bytes) decodes through the same LUT
            # kernel and beats/equals any Huffman coding on this tensor.
            plain = args.gate > 0 and e_l >= args.gate
            if plain:
                lengths = np.full(16, 4, dtype=np.int64)
                rowstart = (np.arange(n, dtype=np.int64) * k * 4).astype(np.uint32)
                total_bits = n * k * 4
                enc = w.tobytes()  # raw packed 4-bit bytes (high nibble first)
            else:
                cv = np.array(canonical_codes(lengths), dtype=np.uint32)
                rowbits = []
                pads = []
                enc = bytearray()
                for c0 in range(0, n, 4096):
                    cc = codes[c0 : c0 + 4096]
                    ncc = cc.shape[0]
                    cnt = np.stack([(cc == s).sum(axis=1) for s in range(16)], axis=1).astype(np.int64)
                    rb = cnt @ cvec
                    rowbits.append(rb)
                    total = int(rb.sum())
                    # int32: max chunk cumsum = 4096 rows * 12288 codes * 15 bits
                    # = 7.5e8 < 2^31 (the global cum below stays int64).
                    # Ported from t1044 4f4a9e1 (peak per-chunk 8 GB -> 2 GB).
                    lc = cvec[cc.ravel()].astype(np.int32)
                    off = np.cumsum(lc) - lc
                    bits = np.zeros(total, dtype=np.uint8)
                    for s in range(16):
                        ns = int(cnt[:, s].sum())
                        if ns == 0:
                            continue
                        m = cc == s
                        p = off.reshape(ncc, -1)[m]  # chunk-local index (bits is local)
                        v = int(cv[s])
                        ls = int(lengths[s])
                        # v is already the code value; its ls-bit form is v itself
                        # (left zero-padded by the (v >> (ls-1-bb)) extraction).
                        for bb in range(ls):
                            if (v >> (ls - 1 - bb)) & 1:
                                bits[p + bb] = 1
                    aligned = (total + 7) // 8 * 8
                    enc.extend(_bits_to_bytes(bits))
                    pads.append(aligned - total)  # chunk-end padding bits
                allrb = np.concatenate(rowbits) if rowbits else np.zeros(0)
                # global row-starts = per-row prefix sums + chunk paddings:
                # rows in chunk c (c*4096 .. (c+1)*4096-1) shift by sum(pads[:c]).
                cum = np.concatenate([[0], allrb.cumsum()]).astype(np.int64)
                rowstart = cum[:n].copy()
                for j, p in enumerate(pads):
                    rowstart[(j + 1) * 4096 :] += p
                rowstart = rowstart.astype(np.uint32)
                total_bits = int(allrb.sum())  # real row bits (decoder end-check base)

            raw_bytes = n * k // 2  # n*k codes * 4 bits / 8
            comp_bytes = len(enc)
            total_raw += raw_bytes
            total_comp += comp_bytes
            names.append(name)
            tensor_info.append(
                {
                    "name": name,
                    "class": _class_of(name),
                    "plain": bool(plain),
                    "E": round(e_l, 4),
                    "n": n,
                    "k": k,
                    "comp_bytes": comp_bytes,
                },
            )
            st = class_stat.setdefault(_class_of(name), [0, 0, 0.0, 0, 0])
            st[0] += 1
            st[1] += int(plain)
            st[2] += e_l * n * k
            st[3] += total_bits
            st[4] += n * k
            out.write(struct.pack("<IIII B", n, k, comp_bytes, int(total_bits), 1 if plain else 0))
            out.write(struct.pack("<16B", *lengths))
            out.write(rowstart.tobytes())
            out.write(bytes(enc))
            print(
                f"[{i}] {'PLAIN' if plain else 'COMP '} {name} N={n} K={k} E[L]={e_l:.4f} "
                f"raw={raw_bytes / 1e6:.1f}MB "
                f"comp={comp_bytes / 1e6:.1f}MB "
                f"x{raw_bytes / comp_bytes:.4f} "
                f"t={time.time() - t0:.1f}s rss={_rss_mb()}MB "
                f"total_t={time.time() - t_tensor0:.0f}s",
                flush=True,
            )
            del codes, w, enc
            if not plain:
                del allrb, cum

    n_plain = sum(1 for t in tensor_info if t["plain"])
    print(
        f"GATE {args.gate}: plain={n_plain} compressed={len(tensor_info) - n_plain}",
        flush=True,
    )
    for c in ("el", "attn", "mlp"):
        if c not in class_stat:
            continue
        nt, npl, mbits, wbits, ncodes = class_stat[c]
        print(
            f"class {c}: {nt} tensors, {npl} plain, E_huff={mbits / ncodes:.4f}, written={wbits / ncodes:.4f} bits/sym",
            flush=True,
        )
    with open(args.out + ".index.json", "w") as f:
        json.dump(
            {
                "format": "ent4",
                "version": VER,
                "magic": MAGIC.decode(),
                "tensors": names,
                "gate": args.gate,
                "tensor_info": tensor_info,
            },
            f,
            indent=1,
        )
    print(
        f"TOTAL raw={total_raw / 1e9:.4f}GB comp={total_comp / 1e9:.4f}GB x{total_raw / total_comp:.4f} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
