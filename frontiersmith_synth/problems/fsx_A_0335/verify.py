#!/usr/bin/env python3
"""Deterministic checker for the quantum-lab-wiring minimum-peak-crosstalk problem.

Usage: python3 verify.py <in> <out> <ans>   (ans is an ignored placeholder)

Reads n from <in>, a non-negative coupling profile f (n reals) from <out>.
Feasibility is validated STRICTLY (count, finiteness, non-negativity, positive sum);
any violation -> `Ratio: 0.0`.  Otherwise computes the scale-free normalized peak
crosstalk c1(f) and normalizes against the checker's own naive boundary-loaded
baseline.  Minimization: smaller c1 -> higher score.  Prints exactly one `Ratio:`.
"""
import sys
import math


def read_int(path):
    with open(path) as fh:
        toks = fh.read().split()
    return int(toks[0])


def read_profile(path):
    """Return list of floats, or None if the file is unparseable / has non-finite."""
    try:
        with open(path) as fh:
            toks = fh.read().split()
    except Exception:
        return None
    vals = []
    for tk in toks:
        try:
            v = float(tk)
        except Exception:
            return None
        if not math.isfinite(v):   # reject nan / inf
            return None
        vals.append(v)
    return vals


def c1(f):
    """Normalized peak self-convolution: 2*n*max(conv(f,f)) / (sum f)^2."""
    n = len(f)
    s = sum(f)
    # full self-convolution, indices 0 .. 2n-2
    peak = 0.0
    for k in range(2 * n - 1):
        lo = max(0, k - (n - 1))
        hi = min(k, n - 1)
        acc = 0.0
        for i in range(lo, hi + 1):
            acc += f[i] * f[k - i]
        if acc > peak:
            peak = acc
    return 2.0 * n * peak / (s * s)


def baseline_profile(n):
    """Naive 'boundary-loaded' construction: push drive toward the two edge ports."""
    b = []
    for i in range(n):
        x = (i + 0.5) / n
        b.append(1.0 + 8.0 * ((2.0 * x - 1.0) ** 4))
    return b


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n = read_int(inf)

    f = read_profile(outf)
    if f is None or len(f) != n:
        print("Infeasible: expected %d finite numbers. Ratio: 0.0" % n)
        return
    for v in f:
        if v < -1e-9:
            print("Infeasible: negative coupling. Ratio: 0.0")
            return
    f = [max(0.0, v) for v in f]
    if sum(f) <= 1e-9:
        print("Infeasible: zero total drive. Ratio: 0.0")
        return

    F = c1(f)
    B = c1(baseline_profile(n))

    if not math.isfinite(F) or F <= 1e-12:
        print("Infeasible: degenerate objective. Ratio: 0.0")
        return

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("c1=%.6f baseline=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
