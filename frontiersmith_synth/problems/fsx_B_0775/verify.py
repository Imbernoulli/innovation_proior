#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_0775 (hash vs published key-family sweep).

Reads the instance (<in>), the participant's mixer-pipeline artifact (<out>), validates
feasibility strictly, computes the objective (max bucket load over all published
families), normalizes against the checker's own fixed reference pipeline, and prints
"Ratio: <float in [0,1]>" on the last line. <ans> is an unused empty placeholder.
"""
import sys
import math
from collections import Counter

MASK64 = (1 << 64) - 1
M = 1024
SHIFT = 54  # 64 - log2(M)
MAX_STAGES = 4
MAX_SALT_TOTAL = 8192

# Checker's own fixed reference pipeline: one multiply (a DIFFERENT constant from the
# greedy reference's textbook choice) + top-bit extraction. Positive, never catastrophic.
BASELINE_A = 0xFF51AFD7ED558CCD


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        m = int(next(it))
        f_count = int(next(it))
        fams = []
        for _ in range(f_count):
            kind = next(it)
            if kind == "AP":
                start = int(next(it)); stride = int(next(it)); count = int(next(it))
                fams.append(("AP", start, stride, count))
            elif kind == "COSET":
                base = int(next(it)); lo = int(next(it)); width = int(next(it))
                fams.append(("COSET", base, lo, width))
            elif kind == "FLOAT":
                exp = int(next(it)); mb = int(next(it)); ms = int(next(it)); count = int(next(it))
                fams.append(("FLOAT", exp, mb, ms, count))
            else:
                raise ValueError("bad instance family kind %r" % kind)
    except (StopIteration, ValueError) as e:
        raise RuntimeError("malformed instance: %s" % e)
    return m, fams


def materialize(fam):
    kind = fam[0]
    if kind == "AP":
        _, start, stride, count = fam
        return [(start + i * stride) & MASK64 for i in range(count)]
    if kind == "COSET":
        _, base, lo, width = fam
        winmask = ((1 << width) - 1) << lo
        base = base & (~winmask & MASK64)
        return [base | (pattern << lo) for pattern in range(1 << width)]
    if kind == "FLOAT":
        _, exp, mant_base, mant_stride, count = fam
        mant_mask = (1 << 40) - 1
        return [((exp << 40) | ((mant_base + i * mant_stride) & mant_mask)) & MASK64
                for i in range(count)]
    raise RuntimeError("bad family kind")


def parse_pipeline(toks):
    """Parse the participant's stage pipeline strictly. Raises ValueError/StopIteration
    on ANY malformed/out-of-range/non-finite/extra/missing token; caller treats that as
    infeasible."""
    it = iter(toks)
    s = int(next(it))
    if s < 0 or s > MAX_STAGES:
        raise ValueError("S out of range")
    reduce_mode = next(it)
    if reduce_mode not in ("MODM", "TOPBITS"):
        raise ValueError("bad REDUCE")
    stages = []
    salt_total = 0
    for _ in range(s):
        tag = next(it)
        if tag == "MUL":
            a = int(next(it)); b = int(next(it))
            if not (0 <= a < (1 << 64)) or not (0 <= b < (1 << 64)):
                raise ValueError("MUL out of range")
            stages.append(("MUL", a, b))
        elif tag == "ROT":
            r = int(next(it))
            if not (0 <= r <= 63):
                raise ValueError("ROT out of range")
            stages.append(("ROT", r))
        elif tag == "XORFOLD":
            r = int(next(it))
            if not (1 <= r <= 63):
                raise ValueError("XORFOLD out of range")
            stages.append(("XORFOLD", r))
        elif tag == "SALT":
            t = int(next(it)); idx = int(next(it))
            if not (1 <= t <= 12) or not (0 <= idx <= 63):
                raise ValueError("SALT header out of range")
            tsize = 1 << t
            salt_total += tsize
            if salt_total > MAX_SALT_TOTAL:
                raise ValueError("SALT table budget exceeded")
            table = []
            for _ in range(tsize):
                v = int(next(it))
                if not (0 <= v < (1 << 64)):
                    raise ValueError("SALT value out of range")
                table.append(v)
            stages.append(("SALT", t, idx, table))
        else:
            raise ValueError("bad stage tag %r" % tag)
    # no leftover tokens allowed
    try:
        next(it)
        raise ValueError("trailing tokens after a well-formed pipeline")
    except StopIteration:
        pass
    return stages, reduce_mode


def apply_pipeline(x, stages, reduce_mode):
    v = x & MASK64
    for st in stages:
        if st[0] == "MUL":
            _, a, b = st
            v = (a * v + b) & MASK64
        elif st[0] == "ROT":
            _, r = st
            if r:
                v = ((v << r) | (v >> (64 - r))) & MASK64
        elif st[0] == "XORFOLD":
            _, r = st
            v ^= (v >> r)
        elif st[0] == "SALT":
            _, t, idx, table = st
            idx_field = (v >> idx) & ((1 << t) - 1)
            v ^= table[idx_field]
    if reduce_mode == "MODM":
        return v % M
    return v >> SHIFT  # TOPBITS


def peak_of_peaks(fams_keys, stages, reduce_mode):
    worst = 0
    for keys in fams_keys:
        c = Counter(apply_pipeline(x, stages, reduce_mode) for x in keys)
        peak = max(c.values())
        if peak > worst:
            worst = peak
    return worst


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>", file=sys.stderr)
        print("Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]

    m, fams = read_instance(in_path)
    if m != M:
        # generator invariant; if ever violated, treat as our own bug -> fail loudly
        # but still print a Ratio so the harness doesn't hang.
        print("Ratio: 0.0")
        sys.exit(0)
    fams_keys = [materialize(f) for f in fams]

    try:
        with open(out_path) as f:
            toks = f.read().split()
        if not toks:
            fail("empty output")
        stages, reduce_mode = parse_pipeline(toks)
    except (StopIteration, ValueError) as e:
        fail(str(e))

    fval = peak_of_peaks(fams_keys, stages, reduce_mode)
    if fval <= 0 or not math.isfinite(fval):
        fail("non-positive/non-finite objective")

    baseline_stages = [("MUL", BASELINE_A, 0)]
    bval = peak_of_peaks(fams_keys, baseline_stages, "TOPBITS")

    sc = min(1000.0, 100.0 * bval / max(1e-9, fval))
    ratio = sc / 1000.0
    print("B=%d F=%d" % (bval, fval))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
