#!/usr/bin/env python3
"""Deterministic scorer for the coral-reef survey low-discrepancy problem.

    python3 verify.py <in> <out> <ans>

<in>  : the instance (see gen.py).
<out> : the participant's added stations -- exactly A = M-K lines, each "x y" in [0,1]^2.
<ans> : ignored placeholder.

Objective (MINIMIZE): the exact L-infinity star discrepancy of the FULL station set
S = (K fixed stations) U (A added stations), computed over all anchored boxes [0,x)x[0,y).

The star discrepancy of a set S of n points in [0,1]^2 is
    D*(S) = sup_{(a,b) in [0,1]^2} | #{p in S : p <= (a,b)} / n  -  a*b |.
It is attained on the finite grid of point coordinates (union {1}); we evaluate both the
closed-box (count-heavy) and open-box (volume-heavy) discrepancy at every grid corner, which
is exact.  Smaller star discrepancy = the survey samples every sub-region of the reef evenly.

Scoring: internal baseline B = discrepancy of the trivial construction (all A added stations
dumped at the reef centre (0.5,0.5)).  Minimization normalization:
    sc = min(1000, 100 * B / F);  Ratio = sc/1000.
"""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    d = int(next(it)); M = int(next(it)); K = int(next(it))
    fixed = []
    for _ in range(K):
        x = float(next(it)); y = float(next(it))
        fixed.append((x, y))
    return d, M, K, fixed


def read_points(path, count):
    """Read exactly `count` (x,y) pairs; return None on ANY malformation."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) != 2 * count:
        return None
    pts = []
    for i in range(count):
        try:
            x = float(toks[2 * i]); y = float(toks[2 * i + 1])
        except ValueError:
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if x < -1e-9 or x > 1.0 + 1e-9 or y < -1e-9 or y > 1.0 + 1e-9:
            return None
        pts.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return pts


def star_discrepancy(points):
    """Exact L-infinity star discrepancy of a 2D point set."""
    n = len(points)
    if n == 0:
        return 1.0
    xs = sorted(set(p[0] for p in points) | {1.0})
    ys = sorted(set(p[1] for p in points) | {1.0})
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    best = 0.0
    inv = 1.0 / n
    for a in xs:
        for b in ys:
            vol = a * b
            c_closed = 0
            c_open = 0
            for j in range(n):
                if px[j] <= a and py[j] <= b:
                    c_closed += 1
                if px[j] < a and py[j] < b:
                    c_open += 1
            d1 = c_closed * inv - vol      # closed box: count can exceed volume
            d2 = vol - c_open * inv        # open box: volume can exceed count
            if d1 > best:
                best = d1
            if d2 > best:
                best = d2
    return best


def fail(reason):
    print("reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inp, outp, _ans = sys.argv[1], sys.argv[2], sys.argv[3]
    d, M, K, fixed = read_instance(inp)
    A = M - K
    added = read_points(outp, A)
    if added is None:
        fail("output is not exactly A=%d finite in-range (x,y) pairs" % A)

    full = list(fixed) + added
    F = star_discrepancy(full)

    # internal trivial baseline: all A added stations at the reef centre
    base_set = list(fixed) + [(0.5, 0.5)] * A
    B = star_discrepancy(base_set)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("discrepancy F=%.6f baseline B=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
