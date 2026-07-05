#!/usr/bin/env python3
"""
Deterministic checker for the "quantum lab wiring" low-discrepancy probe problem.

Instance (stdin the solver read):  "d M"   with d == 2.
Artifact (participant stdout):  exactly M points, each two floats in [0,1].

Objective (MINIMISE):  F = exact 2-D star discrepancy of the emitted point set,
    D*(P) = sup over corner boxes [0,a)x[0,b) of | #{p in box}/M - a*b |.
The supremum is attained on the finite grid of point coordinates, so it is
computed EXACTLY (no sampling, no randomness).

Feasibility (all enforced strictly; ANY violation -> Ratio: 0.0):
  * output is exactly 2*M whitespace-separated float tokens,
  * every token finite (nan/inf rejected) and in [0.0, 1.0].

Baseline B (built by the checker itself): the star discrepancy of the trivial
"main diagonal" layout p_i = ((i+0.5)/M, (i+0.5)/M), which piles all probes on
the diagonal and leaves whole off-diagonal boxes empty.  Score (minimisation):
    sc = min(1000, 100 * B / max(1e-9, F)); print  Ratio: sc/1000
=> reproducing the diagonal scores ~0.1; a 10x lower discrepancy caps at 1.0.
"""
import sys
import math


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def star_discrepancy_2d(pts, m):
    """Exact 2-D star discrepancy over half-open corner boxes [0,a)x[0,b)."""
    xs = sorted(set(p[0] for p in pts))
    ys = sorted(set(p[1] for p in pts))

    # D+  (over-representation): closed count, grid = coordinate values.
    #   at a = x-coord the box [0,a')x... , a'->a+ , includes points with x<=a.
    dplus = 0.0
    for a in xs:
        ptsx = [p for p in pts if p[0] <= a]  # prune on x once
        for b in ys:
            cnt = 0
            for p in ptsx:
                if p[1] <= b:
                    cnt += 1
            val = cnt / m - a * b
            if val > dplus:
                dplus = val

    # D-  (under-representation): open count, grid = coordinate values U {1}.
    xs2 = xs + [1.0]
    ys2 = ys + [1.0]
    dminus = 0.0
    for a in xs2:
        ptsx = [p for p in pts if p[0] < a]
        for b in ys2:
            cnt = 0
            for p in ptsx:
                if p[1] < b:
                    cnt += 1
            val = a * b - cnt / m
            if val > dminus:
                dminus = val

    return max(dplus, dminus)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- instance ----
    with open(inf) as f:
        itoks = f.read().split()
    d = int(itoks[0])
    m = int(itoks[1])
    if d != 2:
        fail("only d==2 supported")

    # ---- baseline B = star discrepancy of the main-diagonal layout ----
    diag = [((i + 0.5) / m, (i + 0.5) / m) for i in range(m)]
    B = star_discrepancy_2d(diag, m)
    if B <= 0.0:
        B = 1e-9

    # ---- participant artifact ----
    with open(outf) as f:
        toks = f.read().split()
    if len(toks) != 2 * m:
        fail("token count mismatch: expected %d got %d" % (2 * m, len(toks)))

    pts = []
    for i in range(m):
        try:
            x = float(toks[2 * i]); y = float(toks[2 * i + 1])
        except (ValueError, TypeError):
            fail("coordinate not a float (garbage?)")
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate (nan/inf)")
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            fail("coordinate outside [0,1]")
        pts.append((x, y))

    F = star_discrepancy_2d(pts, m)
    if F <= 0.0:
        F = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.8f B=%.8f M=%d" % (F, B, m))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
