#!/usr/bin/env python3
"""Deterministic checker for the aquarium-floor probe placement problem.

CLI:  python3 verify.py <in> <out> <ans>
  <in>  : the instance ("d M")
  <out> : participant artifact = M lines, each "x y", coords in [0,1]^2
  <ans> : ignored placeholder

Objective (MINIMIZE): the EXACT 2-D star discrepancy D*(P) of the emitted
point set P.  D*(P) = sup over anchored boxes B(b)=[0,b1)x[0,b2) of
    | (#points in B) / M  -  vol(B) | .
The supremum is attained on the finite grid of point coordinates, so it is
computed exactly (no sampling, no randomness, no time).

Scoring (minimization):  the checker builds its OWN trivial feasible baseline
B (all fittings on a single vertical manifold x=1/2) and normalizes:
    sc    = min(1000, 100 * D_baseline / max(1e-9, D_participant))
    Ratio = sc / 1000
so a solution equal to the baseline scores ~0.1 and a 10x-lower discrepancy
caps at 1.0.  ANY feasibility violation prints "Ratio: 0.0".
"""
import sys
import math


# ----------------------------------------------------------------------------
# exact 2-D star discrepancy (pure python, O(M^3), M small)
# ----------------------------------------------------------------------------
def star_discrepancy(pts):
    n = len(pts)
    if n == 0:
        return 1.0
    xs = sorted(set(p[0] for p in pts))
    ys = sorted(set(p[1] for p in pts))
    xg = xs + [1.0]
    yg = ys + [1.0]
    best = 0.0
    # "+" one-sided term: sup( vol - open_count/n )  over grid U {1}
    for bx in xg:
        for by in yg:
            vol = bx * by
            op = 0
            for (px, py) in pts:
                if px < bx and py < by:
                    op += 1
            v = vol - op / n
            if v > best:
                best = v
    # "-" one-sided term: sup( closed_count/n - vol )  over point grid
    for bx in xs:
        for by in ys:
            vol = bx * by
            cl = 0
            for (px, py) in pts:
                if px <= bx and py <= by:
                    cl += 1
            v = cl / n - vol
            if v > best:
                best = v
    return best


def baseline_points(M):
    # trivial feasible construction the checker owns: a single vertical line.
    return [(0.5, (i + 0.5) / M) for i in range(M)]


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # --- read instance ---
    try:
        with open(in_path) as f:
            toks = f.read().split()
        d = int(toks[0])
        M = int(toks[1])
    except Exception:
        fail("bad instance")
        return
    if d != 2:
        fail("dimension not supported")

    # --- read participant artifact ---
    try:
        with open(out_path) as f:
            data = f.read()
    except Exception:
        fail("no output")
        return
    raw = data.split()
    if len(raw) != 2 * M:
        fail("expected exactly %d numbers, got %d" % (2 * M, len(raw)))

    vals = []
    for tk in raw:
        try:
            v = float(tk)
        except Exception:
            fail("non-numeric token")
            return
        if not math.isfinite(v):
            fail("non-finite coordinate")
        vals.append(v)

    tol = 1e-9
    pts = []
    for i in range(M):
        x = vals[2 * i]
        y = vals[2 * i + 1]
        if x < -tol or x > 1.0 + tol or y < -tol or y > 1.0 + tol:
            fail("coordinate out of [0,1]")
        # clamp into the closed unit square
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        pts.append((x, y))

    # --- score ---
    F = star_discrepancy(pts)
    B = star_discrepancy(baseline_points(M))
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("M=%d participant_discrepancy=%.9f baseline_discrepancy=%.9f" % (M, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
