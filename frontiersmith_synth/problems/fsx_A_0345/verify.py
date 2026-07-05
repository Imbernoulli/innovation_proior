#!/usr/bin/env python3
"""Deterministic checker for the autocorrelation / thermal-cross-interference
constant.

CLI:  python3 verify.py <in> <out> <ans>   (ans is an ignored placeholder)

Prints exactly one final line `Ratio: <float in [0,1]>`.
Any feasibility violation -> `Ratio: 0.0`.
Exact integer arithmetic (Python big ints + Fraction); bit-for-bit deterministic.
"""
import sys
from fractions import Fraction


def fail(reason):
    sys.stdout.write("%s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    n = int(toks[0])
    M = int(toks[1])
    return n, M


def max_self_conv(f):
    """Exact max_k sum_i f[i]*f[k-i] over ints. O(n^2)."""
    n = len(f)
    best = 0
    for k in range(2 * n - 1):
        i0 = 0 if k - n + 1 < 0 else k - n + 1
        i1 = k if k < n - 1 else n - 1
        s = 0
        for i in range(i0, i1 + 1):
            s += f[i] * f[k - i]
        if s > best:
            best = s
    return best


def c1(f, n):
    """c1 = 2n * max_conv / sum^2  as an exact Fraction (or None if sum<=0)."""
    s = sum(f)
    if s <= 0:
        return None
    mx = max_self_conv(f)
    return Fraction(2 * n * mx, s * s)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, M = read_instance(in_path)

    # ---- parse participant output strictly ----
    try:
        with open(out_path) as fh:
            raw = fh.read()
    except Exception:
        fail("cannot read output")

    toks = raw.split()
    if len(toks) != n:
        fail("wrong token count: got %d expected %d" % (len(toks), n))

    f = []
    for t in toks:
        tl = t.lower()
        if tl in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            fail("non-finite token")
        try:
            v = int(t)
        except ValueError:
            fail("non-integer token: %r" % t)
        if v < 0 or v > M:
            fail("token out of range [0,%d]: %d" % (M, v))
        f.append(v)

    if sum(f) <= 0:
        fail("all-zero profile")

    F = c1(f, n)
    if F is None or F <= 0:
        fail("degenerate profile")

    # ---- internal baseline B: a concentrated coolant block ----
    w = max(1, n // 5)
    block = [1] * w + [0] * (n - w)
    B = c1(block, n)

    # minimization normalization
    sc = 100.0 * float(B) / max(1e-9, float(F))
    if sc > 1000.0:
        sc = 1000.0
    ratio = sc / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0

    sys.stdout.write("c1=%s B=%s\nRatio: %.6f\n" % (str(F), str(B), ratio))


if __name__ == "__main__":
    main()
