#!/usr/bin/env python3
"""
Deterministic checker for fsx_A_0085 -- "Corridor Congestion Constant".

Reads:
  <in>  : first token n = number of corridor segments.
  <out> : n non-negative real habitat-density values f_0..f_{n-1}.

Objective (MINIMIZE) -- normalized peak pairwise-encounter index:
    c1(f) = 2 * n * max_k ( sum_{i+j=k} f_i f_j )  /  ( sum_i f_i )^2

Internal baseline B = c1 of the "half-block" construction the checker builds
itself: put unit density in the first floor(n/2) segments, zero elsewhere
(all the habitat crammed into one contiguous half of the corridor). This is a
trivially feasible allocation and is deliberately congested.

Score (minimization):
    sc  = min(1000.0, 100.0 * B / F)
    Ratio = sc / 1000.0
so reproducing the baseline -> 0.1 ; a 10x lower index caps at 1.0.
"""
import sys


def objective(f, n):
    s = sum(f)
    # autoconvolution c_k = sum_{i+j=k} f_i f_j , k = 0 .. 2n-2
    m = len(f)
    cmax = 0.0
    for k in range(2 * m - 1):
        lo = max(0, k - (m - 1))
        hi = min(k, m - 1)
        acc = 0.0
        for i in range(lo, hi + 1):
            acc += f[i] * f[k - i]
        if acc > cmax:
            cmax = acc
    return 2.0 * n * cmax / (s * s)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as fh:
        n = int(fh.read().split()[0])

    with open(out_path) as fh:
        toks = fh.read().split()

    if len(toks) != n:
        fail("expected %d values, got %d" % (n, len(toks)))

    f = []
    for t in toks:
        try:
            v = float(t)
        except ValueError:
            fail("non-numeric token %r" % t)
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite value")
        if v < 0.0:
            fail("negative density")
        if v > 1e6:
            fail("density exceeds 1e6")
        f.append(v)

    s = sum(f)
    if s <= 1e-9:
        fail("total density must be positive")

    F = objective(f, n)

    # internal baseline: half-block
    base = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    B = objective(base, n)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
