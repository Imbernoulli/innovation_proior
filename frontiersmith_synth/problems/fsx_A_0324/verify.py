#!/usr/bin/env python3
"""Deterministic checker for the apiary landing-pad spread (Heilbronn-type) problem.

CLI:  python3 verify.py <in> <out> <ans>     (ans ignored)

Reads n from <in> and a participant layout (n coordinate pairs) from <out>.
Validates feasibility STRICTLY (exactly n finite pads, all inside the closed unit
triangle within tol). Objective F = min triangle area over all C(n,3) triples.
Baseline B = the faintest-triangle area of a FIXED pseudo-random reference layout
the checker builds itself. Prints  'Ratio: <s/1000>'  and exits 0.
"""
import sys
import math

TOL = 1e-6


def read_n(path):
    with open(path) as f:
        tok = f.read().split()
    return int(tok[0])


# --- fixed 64-bit LCG (reproducible, no external deps) --------------------------
_MASK = (1 << 64) - 1


def _lcg_stream(seed):
    state = seed & _MASK
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK
        yield (state >> 11) / float(1 << 53)   # uniform in [0,1)


REF_SEED = 12345
REF_K = 150


def reference_layout(n, seed=REF_SEED, K=REF_K):
    """The checker's internal baseline B: the BEST of K fixed pseudo-random folded
    layouts (uniform-in-triangle) under the min-triangle-area objective. This is a
    calibrated, decent-but-suboptimal reference (best-of-K plateaus quickly, so the
    faintest triangle of a truly optimized layout still beats it several-fold)."""
    g = _lcg_stream(seed)
    best_pts = None
    best_val = -1.0
    for _ in range(K):
        pts = []
        for _ in range(n):
            a = next(g)
            b = next(g)
            if a + b > 1.0:
                a, b = 1.0 - a, 1.0 - b
            pts.append((a, b))
        v = min_triangle_area(pts)
        if v > best_val:
            best_val = v
            best_pts = pts
    return best_pts


def min_triangle_area(pts):
    n = len(pts)
    best = float("inf")
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dxj = xj - xi
            dyj = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                area = 0.5 * abs(dxj * (yk - yi) - (xk - xi) * dyj)
                if area < best:
                    best = area
    return best


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    n = read_n(sys.argv[1])

    # --- parse participant output strictly ---
    try:
        with open(sys.argv[2]) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(toks) != 2 * n:
        fail("expected %d numbers (n=%d pads), got %d" % (2 * n, n, len(toks)))

    vals = []
    for t in toks:
        try:
            v = float(t)
        except ValueError:
            fail("non-numeric token %r" % t)
        if not math.isfinite(v):
            fail("non-finite token %r" % t)
        vals.append(v)

    pts = [(vals[2 * i], vals[2 * i + 1]) for i in range(n)]

    # --- feasibility: inside the closed unit triangle ---
    for (x, y) in pts:
        if x < -TOL or y < -TOL or (x + y) > 1.0 + TOL:
            fail("pad (%g,%g) outside unit triangle" % (x, y))

    F = min_triangle_area(pts)
    if not math.isfinite(F) or F <= 0.0:
        # collinear / degenerate -> faintest triangle has zero area
        print("degenerate faintest triangle F=%g" % F)
        print("Ratio: 0.0")
        sys.exit(0)

    B = min_triangle_area(reference_layout(n))
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.10g B=%.10g" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
