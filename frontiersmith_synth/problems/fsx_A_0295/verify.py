#!/usr/bin/env python3
"""
Deterministic checker for fsx_A_0295 -- "Reservoir Spectral-Load Constant".

Reads:
  <in>  : first token n = number of reservoirs.
  <out> : n non-negative real capacity values f_0..f_{n-1}.

Objective (MINIMIZE) -- normalized spectral-load index (squared L2 norm of the
autoconvolution / stress spectrum), scale-invariant:

    L(f) = n * ( sum_k c_k^2 )  /  ( sum_i f_i )^4 ,
    where c_k = sum_{i+j=k} f_i f_j  (k = 0 .. 2n-2).

Internal baseline B = L of the "half-block" construction the checker builds
itself: unit capacity in the first floor(n/2) reservoirs, zero elsewhere (all
the buffering crammed into one contiguous half of the river). Trivially
feasible and deliberately brittle.

Score (minimization):
    sc    = min(1000.0, 100.0 * B / F)
    Ratio = sc / 1000.0
so reproducing the baseline -> 0.1 ; a 10x lower index caps at 1.0.
"""
import sys
import math


def objective(f, n):
    # autoconvolution c_k = sum_{i+j=k} f_i f_j , k = 0 .. 2n-2
    m = len(f)
    s = 0.0
    for v in f:
        s += v
    energy = 0.0
    for k in range(2 * m - 1):
        lo = k - (m - 1)
        if lo < 0:
            lo = 0
        hi = k
        if hi > m - 1:
            hi = m - 1
        acc = 0.0
        for i in range(lo, hi + 1):
            acc += f[i] * f[k - i]
        energy += acc * acc
    return n * energy / (s ** 4)


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
        if not math.isfinite(v):
            fail("non-finite value")
        if v < 0.0:
            fail("negative capacity")
        if v > 1e6:
            fail("capacity exceeds 1e6")
        f.append(v)

    s = sum(f)
    if s <= 1e-9:
        fail("total capacity must be positive")

    F = objective(f, n)
    if not math.isfinite(F) or F <= 0.0:
        fail("degenerate objective")

    # internal baseline: half-block
    base = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    B = objective(base, n)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
