#!/usr/bin/env python3
# Deterministic checker for the Sonar Costas-Permutation problem (format C).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# The artifact is a permutation p of {0,...,n-1}: p[c] = the row of the single
# sonar mark in column c. Quality = number of displacement-vector COINCIDENCES
# (collisions); fewer is better (a Costas array has zero). Score normalizes
# against the checker's own trivial baseline (the identity permutation).
# Prints "... Ratio: <r>" with r in [0, 1]; any infeasibility prints Ratio: 0.0.
import sys
from collections import Counter


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def coincidences(p):
    """Number of repeated displacement vectors: sum over vectors of (mult-1).
    Vector for the ordered pair of columns (i<j) is (j-i, p[j]-p[i])."""
    n = len(p)
    c = Counter()
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, n):
            c[(j - i, p[j] - pi)] += 1
    return sum(v - 1 for v in c.values())


def main():
    # ---- read instance ----
    try:
        itoks = open(sys.argv[1]).read().split()
        n = int(itoks[0])
    except Exception:
        fail("bad instance")
    if n < 1 or n > 10000:
        fail("n out of range")

    # ---- read participant artifact (bounded, strict) ----
    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not otoks:
        fail("empty output")
    # reject an over-long dump outright (guards against huge/garbage floods)
    if len(otoks) > n + 5:
        fail("too many tokens")
    if len(otoks) != n:
        fail("expected exactly n integers, got %d" % len(otoks))

    p = []
    for t in otoks:
        # int() rejects nan/inf/floats/garbage outright -> infeasible
        try:
            v = int(t)
        except Exception:
            fail("non-integer token %r" % t)
        if v < 0 or v >= n:
            fail("value %d out of range [0,%d]" % (v, n - 1))
        p.append(v)

    # must be a genuine permutation (one mark per row and per column)
    if len(set(p)) != n:
        fail("not a permutation (repeated row)")

    # ---- objective: minimize coincidences ----
    F = coincidences(p)

    # ---- internal trivial baseline: the identity permutation p[i]=i.
    # Its displacement vectors are all of the form (h, h), so vector (h,h) occurs
    # (n-h) times; coincidences = sum_{h=1}^{n-1} (n-h-1) = (n-1)(n-2)/2. ----
    B = (n - 1) * (n - 2) // 2
    if B <= 0:
        B = 1  # degenerate tiny n; keep score well-defined

    # minimization normalization: a solution 10x better than baseline caps at 1.0
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("n=%d coincidences=%d baseline=%d Ratio: %.6f" % (n, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
