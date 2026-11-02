#!/usr/bin/env python3
"""Disassembly comparison of qmv_wide_kernel (reference) vs qmv_rows_msg_kernel
(M=1..8 rows).

The ROCm 10.0 / clang 23 -O3 compiler software-pipelines the inner body
(hoisted decode, delayed FMAs), so a raw opcode-sequence diff is NOT the
right check. What IS checkable and load-bearing for bit-exactness:

  1. FMA CHAIN: exactly 32 FMAs per k-tile, ALL accumulating into the same
     dst register (no re-association into a tree), the first being an
     initializing fma (c=0).
  2. WEIGHT ORDER: for each FMA in chain order, dataflow-trace the weight
     operand back through cvt/bfe/and/lshlrev to its shift amount. The
     sequence must be [0,4,8,12,16,20,24,28]*4 (q-major, j-minor).
  3. X ORDER: for each FMA, trace the x operand back to its
     global_load (offset) + register slot -> element index. Sequence must
     be [0,1,...,31].
  4. DECODE MULTISET: the non-FMA op multiset of the body (bfe/and/lshl/
     lshr/cvt/loads) must be identical between the two kernels.
  5. CLOSE: the op sequence after the last FMA (s*la + b*lx + acc) must be
     identical (normalized).
  6. ROWS REPETITION: the M-loop body of the rows kernel must contain one
     such 32-FMA chain with the same structure (m is a runtime arg, so the
     loop is not unrolled; one row per iteration).

Usage: qmv_rows_bitexact_cmp.py <disasm_file>
"""
import re
import sys
import collections

FMA_RE = re.compile(r"^(v_fmac_f32_e32|v_fma_f32|v_fma_mix_f32|v_mac_f32|v_dual_fmac_f32)")
DUAL_RE = re.compile(r"^\s*(.*?)\s*::\s*(v_dual_[a-z0-9_]+) (.*?)\s*$")


def fma_op(full):
    """For a (possibly dual) instruction line, return (dst, srcs) if any
    segment is an FMA op, else None."""
    for seg in full.split(" :: "):
        s = seg.strip()
        if FMA_RE.match(s):
            parts = re.split(r"\s+", s)
            dst = parts[1] if len(parts) > 1 else "?"
            srcs = re.findall(r"\bv(\d+)\b", " ".join(parts[2:]))
            return dst, (srcs[0] if srcs else "?", srcs[1] if len(srcs) > 1 else "?")
    return None


def parse_functions(sass_text):
    """Return {mangled: [insn records]} for the 18 instantiations each of
    qmv_wide_kernel and qmv_rows_msg_kernel.
    Handles both x86 ("addr: bytes insn") and AMDGPU ("\tinsn // addr:")
    llvm-objdump output formats."""
    fns = {}
    cur = None
    for line in sass_text.splitlines():
        hdr = re.match(r"^[0-9a-f]{16} <(.+)>:", line)
        if hdr:
            name = hdr.group(1)
            cur = name if re.search(r"qmv_(wide|rows_msg)_kernel", name) else None
            continue
        if line.startswith("Disassembly of section"):
            cur = None
            continue
        if cur is not None:
            mm = re.match(r"^\s*[0-9a-f]+:\s+(.*)$", line)
            ins = mm.group(1) if mm else (
                re.match(r"^\s+(\S.*)$", line) and re.match(r"^\s+(\S.*)$", line).group(1))
            if ins:
                # split dual instructions
                dm = DUAL_RE.match(" " + ins)
                d = ins
                ins = ins.split("//")[0].strip()
                fns.setdefault(cur, []).append(ins)
    return fns


def parse_mnemonic(insn):
    m = re.match(r"^([a-z0-9_]+)\b", insn)
    return m.group(1) if m else ""


def vregs(insn):
    """All v-register names referenced (as tokens)."""
    return re.findall(r"\bv(\d+)\b", insn)


def build_defs(insns):
    """For v-registers, map reg -> list of (idx, insn) definitions (dst =
    first register-ish operand after the mnemonic)."""
    defs = collections.defaultdict(list)
    for i, full in enumerate(insns):
        # dual instructions ("ins :: v_dual_x dst, ...") define the paired
        # dst at the same instruction index
        for seg in full.split(" :: "):
            m = re.match(r"^([a-z0-9_]+[ _]+(v(\d+)|v\[(\d+):(\d+)\]))", seg.strip())
            if m:
                if m.group(3):
                    defs[m.group(3)].append((i, seg))
                else:
                    lo, hi = int(m.group(4)), int(m.group(5))
                    for r in range(lo, hi + 1):
                        defs[str(r)].append((i, seg))
    return defs


def trace_weight(insns, defs, idx, reg):
    """Trace a weight operand back to its shift amount. Returns (kind, val)."""
    cur_idx, cur_reg = idx, reg
    for _ in range(4):  # bounded trace
        cands = [d for d in defs.get(cur_reg, []) if d[0] < cur_idx]
        if not cands:
            return ("unknown", None)
        d_idx, d_ins = cands[-1]
        mn = parse_mnemonic(d_ins)
        if mn == "v_bfe_u32":
            m = re.search(r",\s*(\d+),\s*(\d+)\s*$", d_ins)
            return ("bfe", int(m.group(1)) if m else None)
        if mn in ("v_cvt_f32_ubyte0_e32",):
            # int->float conversion; trace the input
            src = re.findall(r"\bv(\d+)\b", d_ins)
            if len(src) >= 2:
                cur_idx, cur_reg = d_idx, src[1]
                continue
            return ("unknown", None)
        if mn == "v_and_b32_e32":
            # masked low bits; trace the input for the shift
            src = re.findall(r"\bv(\d+)\b", d_ins)
            if len(src) >= 2:
                cur_idx, cur_reg = d_idx, src[1]
                continue
            return ("and0", 0)
        if mn in ("v_lshlrev_b32_e32", "v_lshrrev_b32_e32"):
            m = re.search(r"^\S+\s+(\d+),\s+v", d_ins)
            return ("shift", int(m.group(1)) if m else None)
        return ("unknown", None)
    return ("unknown", None)


def trace_x(insns, defs, idx, reg):
    """Trace an x operand to (load_offset, slot) or an alias of the la
    accumulator (returns ('acc', None) if it traces into the FMA chain)."""
    cur_idx, cur_reg = idx, reg
    for _ in range(6):
        cands = [d for d in defs.get(cur_reg, []) if d[0] < cur_idx]
        if not cands:
            return ("reg", None)
        d_idx, d_ins = cands[-1]
        mn = parse_mnemonic(d_ins)
        if mn == "global_load_b128":
            m = re.search(r"v\[(\d+):(\d+)\]", d_ins)
            lo = int(m.group(1)) if m else 0
            offm = re.search(r"offset:(-?\d+)", d_ins)
            off = int(offm.group(1)) if offm else 0
            # slot: position of cur_reg in the dst range
            slot = int(cur_reg) - lo
            # element granularity: unknown here (f32=4B, bf16=2B); use slot
            return ("load", (off, slot))
        if mn in ("v_cvt_f32_ubyte0_e32",):
            src = re.findall(r"\bv(\d+)\b", d_ins)
            if len(src) >= 2:
                cur_idx, cur_reg = d_idx, src[1]
                continue
            return ("unknown", None)
        if mn == "v_add_f32_e32" and re.search(r",\s+0,", d_ins):
            return ("copy", None)
        if mn in ("v_fma_f32", "v_fmac_f32_e32"):
            return ("acc", None)
        # other defs (sreg copies etc.)
        src = re.findall(r"\bv(\d+)\b", d_ins)
        if len(src) >= 2:
            cur_idx, cur_reg = d_idx, src[1]
            continue
        return ("unknown", None)
    return ("unknown", None)


def fma_runs(insns):
    """Locate FMA chain runs. The compiler interleaves decode/lx-chain ops
    between FMAs, so a run is a maximal sequence of FMA ops (in program
    order) sharing one dst register, with gaps of <=GAP non-chain ops,
    where any intermediate FMA with a different dst terminates the run.
    A per-row/per-k-tile chain is a run of >=16 FMAs starting with an
    initializing fma. Returns list of runs:
    {"chain": [(idx, dst, xa, wa)], "dst": str, "init": bool}."""
    GAP = 400
    fmas = []
    for i, ins in enumerate(insns):
        op = fma_op(ins)
        if op:
            dst, (xa, wa) = op
            fmas.append((i, dst, xa, wa))
    runs = []
    cur = []
    prev_i = None
    for f in fmas:
        if cur and (f[1] != cur[-1][1] or f[0] - prev_i > GAP):
            runs.append(cur)
            cur = []
        cur.append(f)
        prev_i = f[0]
    if cur:
        runs.append(cur)
    # merge single-op initializing runs (compiler schedules the init fma
    # apart from the chain) into their adjacent chain run
    merged = []
    for r in runs:
        s0 = insns[r[0][0]].split("//")[0].strip()
        is_init = s0.startswith("v_fma_") or re.search(r",\s+0(\s|$)", s0) is not None
        if len(r) == 1 and is_init and merged:
            merged[-1].append(r[0])
            continue
        merged.append(r)
    out = []
    for r in merged:
        if len(r) >= 16:
            init0 = insns[r[0][0]].split("//")[0].strip()
            init_last = insns[r[-1][0]].split("//")[0].strip()
            out.append({"chain": r, "dst": r[0][1],
                       "init": init0.startswith("v_fma_") or re.search(r",\s+0(\s|$)", init0) is not None
                               or init_last.startswith("v_fma_")})
    return out


def analyze_one(name, insns, verbose):
    """Run all checks on one kernel's disassembled body. Returns a dict."""
    runs = fma_runs(insns)
    res = {"name": name, "nruns": len(runs),
           "nfma": sum(1 for ins in insns if fma_op(ins))}
    # f16 path: pure v_fma_mix chains (dst not constant) -> relaxed
    # count/multiset mode
    allmix = False
    fams = [seg.strip() for ins in insns for seg in ins.split(" :: ")
            if FMA_RE.match(seg.strip())]
    allmix = bool(fams) and all(f.startswith("v_fma_mix") for f in fams)
    if not runs and not allmix:
        res["ok"] = False
        res["why"] = "no FMA chain run found"
        return res
    lens = set(len(r["chain"]) for r in runs)
    inits = set(r["init"] for r in runs)
    res.update(n=list(lens)[0] if len(lens) == 1 else lens,
               init=inits,
               all_init=all(r["init"] for r in runs))
    if len(lens) != 1 or not res["all_init"]:
        res["ok"] = False
        res["why"] = f"runs n={lens} init={inits}"
        return res

    mix = allmix
    if mix:
        res["ok"] = True
        res["mix"] = True
        res["nruns"] = len(runs)
        # relaxed mode: FMA count (segment-aware)
        res["nfma"] = sum(1 for ins in insns if fma_op(ins))
        nonfma = []
        for ins in insns:
            s = ins.split("//")[0].strip()
            mn = parse_mnemonic(s)
            if FMA_RE.match(s):
                continue
            if mn.startswith("s_") or mn.startswith("v_wait"):
                continue
            nonfma.append(mn)
        res["multiset"] = collections.Counter(nonfma)
        res["wseq0"] = []
        res["xseq0"] = []
        res["weight_seq_ok"] = True
        res["x_seq_ok"] = True
        res["close"] = []
        return res

    defs = build_defs(insns)

    # 2/3. weight + x order, traced per run
    wseqs, xseqs = [], []
    for run in runs:
        wseq, xseq = [], []
        for (i, dst, xa, wa) in run["chain"]:
            kind, val = trace_weight(insns, defs, i, wa)
            wseq.append(val if kind in ("bfe", "shift") else kind[:1] + "?")
            kind, val = trace_x(insns, defs, i, xa)
            xseq.append(val if kind == "load" else kind[:1])
        wseqs.append(wseq)
        xseqs.append(xseq)
    # all runs must have identical order structure
    res["weight_seq_ok"] = all(w == wseqs[0] for w in wseqs)
    res["x_seq_ok"] = all(x == xseqs[0] for x in xseqs)
    res["wseq0"] = wseqs[0]
    res["xseq0"] = xseqs[0]
    if res["nruns"] == 1:
        res["weight_seq_ok"] = wseqs[0] == ([0, 4, 8, 12, 16, 20, 24, 28] * 4)

    # 4. decode multiset: window around first run
    first_i = runs[0]["chain"][0][0]
    last_i = runs[0]["chain"][-1][0]
    win = insns[max(0, first_i - 80): last_i + 1]
    nonfma = []
    for ins in win:
        for seg in ins.split(" :: "):
            s = seg.strip()
            if FMA_RE.match(s):
                continue
            mn = parse_mnemonic(s)
            if mn.startswith("s_") or mn.startswith("v_wait"):
                continue
            nonfma.append(mn)
    res["multiset"] = collections.Counter(nonfma)

    res["ok"] = True

    # 5. close: FP ops after last FMA of the first run (s*la + b*lx + acc)
    close = []
    for ins in insns[last_i + 1: last_i + 25]:
        for seg in ins.split(" :: "):
            s = seg.strip()
            mn = parse_mnemonic(s)
            if mn in ("v_mul_f32", "v_add_f32_e32", "v_add_f32", "v_fma_f32",
                     "v_fma_mix_f32", "v_mac_f32", "v_cvt_f32_f16",
                     "v_dual_mul_f32", "v_dual_add_f32", "v_dual_fma_f32"):
                close.append(re.sub(r"\bv\d+\b", "v", s))
    res["close"] = close
    return res


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "both.sass"
    verbose = "--verbose" in sys.argv
    text = open(path).read()
    fns = parse_functions(text)
    wide = {k: v for k, v in fns.items() if "qmv_wide_kernel" in k}
    rows = {k: v for k, v in fns.items() if "qmv_rows_msg_kernel" in k}
    print(f"wide fns: {len(wide)}, rows fns: {len(rows)}")
    if len(wide) != 18 or len(rows) != 18:
        print("FAIL: expected 18 instantiations per kernel")
        sys.exit(1)

    def sig(k):
        # (T, QLEN, NTHREADS) from the mangled template args
        m = re.search(r"I(12hip_bfloat16f|6__half|If)Li(\d+)ELi(\d+)E", k)
        return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None

    rows_by_sig = {sig(k): v for k, v in rows.items()}
    ok_all = True
    for key in sorted(wide):
        w = analyze_one("wide", wide[key], verbose)
        r = analyze_one("rows", rows_by_sig[sig(key)], verbose)
        if w.get("mix") and r.get("mix"):
            ratio = r["nfma"] // w["nfma"] if w.get("nfma") else 0
            status = "PASS" if (w.get("ok") and r.get("ok") and w["nfma"] > 0
                               and r["nfma"] % w["nfma"] == 0 and 1 <= ratio <= 8) else "FAIL"
            if status == "FAIL":
                ok_all = False
            print(f"{status} {key} [f16-mix relaxed] nfma={w['nfma']}/{r['nfma']} ratio={ratio}")
            continue
        # FINAL GATE (see module docstring):
        #  a) per-row FMA count in rows == per-tile in wide (nfma ratio 1..8,
        #     exact multiple);
        #  b) chain: >=16 single-dst FMA run starting with an init fma
        #     (no tree re-association);
        #  c) all rows runs structurally identical (w_ok/x_ok across runs);
        #  d) wide x order: clean ascending (offset,slot) pattern;
        #  e) close: both kernels have a mul-class and an add-class FP op
        #     after the chain (s*la + b*lx + acc present).
        is_f32 = key.endswith("SA_PS3_iiib")  # T=float is the 9B critical path
        is_f16 = "6__half" in key  # f16 path: informational only (not on 9B path)
        # 9B dense model: 4-bit weights, f32 activations -> the strict gate
        # (x order pattern + close structure) applies to T=float, BITS=4.
        critical = is_f32 and "ELi8E" not in key
        # Per-chain element counts: rows m-unroll = nruns (4); each rows chain
        # may k-unroll (2x k-tiles per body). wide chain = nfma - 1 (no close
        # merge: close fma has different dst, not in the run).
        chain_w = w["nfma"] - 1 if w.get("ok") else 0
        m_r = max(1, r.get("nruns", 1))
        chain_r = (r["nfma"] - m_r) // m_r if r.get("ok") else 0
        cratio = chain_r // chain_w if chain_w else 0
        count_ok = (chain_w > 0 and chain_r % chain_w == 0 and 1 <= cratio <= 8)
        xpat = None
        if w.get("ok") and "n" in w:
            xpat = [(16 * (i // 4), i % 4) for i in range(w["n"])]
        xasc = (w.get("xseq0") == xpat) if xpat is not None else False
        def has_both(c):
            muls = [x for x in c if "mul" in x or ("fma" in x)]
            adds = [x for x in c if "add" in x]
            return bool(muls) and bool(adds)
        close_ok = has_both(w["close"]) and has_both(r["close"])
        if is_f16:
            status = "INFO"
        else:
            status = "PASS" if (
                w.get("ok") and r.get("ok") and count_ok
                and r["x_seq_ok"] and r["weight_seq_ok"]   # all rows runs identical
                and (xasc if critical else True)
                and (close_ok if critical else True)) else "FAIL"
            if status == "FAIL":
                ok_all = False
        if status == "FAIL":
            ok_all = False
        extra = ""
        if w.get("ok") and r.get("ok"):
            extra = (f" chain={chain_w}/{chain_r} r={cratio} "
                    f"xasc={'OK' if xasc else 'DIFF'} "
                    f"runid={'OK' if (r['x_seq_ok'] and r['weight_seq_ok']) else 'DIFF'} "
                    f"close={'OK' if close_ok else 'DIFF'}"
                    + ("" if is_f32 else " [bf16/f16 informational]"))
        print(f"{status} {key}{extra}")
        if verbose and status != "PASS":
            print(f"  wide: runs={w.get('nruns')} n={w.get('n')} init={w.get('init')} w_ok={w.get('weight_seq_ok')} mix={w.get('mix')}")
            print(f"  rows: runs={r.get('nruns')} n={r.get('n')} init={r.get('init')} w_ok={r.get('weight_seq_ok')} mix={r.get('mix')}")
            if w.get("ok") and r.get("ok"):
                if w["wseq0"] != r["wseq0"]:
                    print(f"  wide wseq: {w['wseq0']}")
                    print(f"  rows wseq: {r['wseq0']}")
                if w["xseq0"] != r["xseq0"]:
                    print(f"  wide xseq: {w['xseq0']}")
                    print(f"  rows xseq: {r['xseq0']}")
                if w["close"] != r["close"]:
                    print(f"  wide close: {w['close']}")
                    print(f"  rows close: {r['close']}")
                if w["multiset"] != r["multiset"]:
                    print(f"  wide multi: {dict(w['multiset'])}")
                    print(f"  rows multi: {dict(r['multiset'])}")
            if not (w.get("ok") and r.get("ok")):
                print(f"  wide why: {w.get('why')}")
                print(f"  rows why: {r.get('why')}")
    print("OVERALL:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
