#!/usr/bin/env python3
"""Deterministic checker for the Mountain Rescue Relay Density problem.

Usage: python3 verify.py <in> <out> <ans>   (ans is an empty placeholder, ignored)

Reads the instance (n, U) and the participant's relay vector f. Validates strictly,
then scores the scale-invariant autocorrelation constant

    c(f) = 2 * n * max_k conv(f,f)_k / (sum f)^2      (minimize)

against an internal reference baseline B (the half-trail unit block). Prints exactly
one line ending in `Ratio: <x>` with x in [0,1]. Any violation -> Ratio: 0.0.
"""
import sys


def fail(reason):
    print("INVALID (%s) Ratio: 0.0" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    n = int(toks[0])
    U = int(toks[1])
    return n, U


def peak_autocorr(f):
    """max_k sum_{i+j=k} f_i f_j, exact integer arithmetic. O(n^2)."""
    n = len(f)
    g = [0] * (2 * n - 1)
    for a in range(n):
        fa = f[a]
        if fa == 0:
            continue
        base = a
        for b in range(n):
            fb = f[b]
            if fb:
                g[base + b] += fa * fb
    return max(g)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    n, U = read_instance(in_path)

    # ---- parse participant output strictly ----
    try:
        with open(out_path) as fh:
            raw = fh.read().split()
    except Exception:
        fail("unreadable output")

    if len(raw) != n:
        fail("expected %d integers, got %d tokens" % (n, len(raw)))

    f = []
    for tok in raw:
        # reject non-integer, nan, inf, floats, signs+garbage
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer token '%s'" % tok[:16])
        if v < 0 or v > U:
            fail("value %d out of [0,%d]" % (v, U))
        f.append(v)

    S = sum(f)
    if S <= 0:
        fail("total relay strength is zero")

    # ---- objective F = c(f) ----
    P = peak_autocorr(f)
    # F = 2*n*P / S^2  (rational, keep numerator/denominator as ints)
    F_num = 2 * n * P
    F_den = S * S
    # F = F_num / F_den

    # ---- internal reference baseline B = half-block of ceil(n/2) unit stations ----
    b = (n + 1) // 2                 # block length
    # block of b ones: peak autocorr = b, sum = b  ->  B = 2*n*b / b^2 = 2*n / b
    B_num = 2 * n
    B_den = b
    # B = B_num / B_den

    # ratio = 0.1 * B / F = 0.1 * (B_num/B_den) / (F_num/F_den)
    #       = 0.1 * B_num * F_den / (B_den * F_num)
    if F_num <= 0:
        # peak autocorr can't be 0 when S>0, but guard anyway
        fail("degenerate objective")
    ratio = 0.1 * (B_num * F_den) / (B_den * F_num)
    if ratio > 1.0:
        ratio = 1.0
    if ratio < 0.0:
        ratio = 0.0

    print("c=%.6f B=%.6f Ratio: %.6f" % (F_num / F_den, B_num / B_den, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
