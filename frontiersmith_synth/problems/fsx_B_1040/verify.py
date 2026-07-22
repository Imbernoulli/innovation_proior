#!/usr/bin/env python3
"""
Deterministic checker for the locator-beacon anchor-placement problem
(format C, family anchor-set-geometry-regimes, minimize).
CLI: python3 verify.py <in> <out> <ans>  (ans is an empty placeholder, ignored).
Prints "... Ratio: <r>" with r in [0,1] on its own final line, exit 0 always.
Any feasibility violation -> "Ratio: 0.0".

Mechanism (see statement.md for the full write-up):
  For a target point t and the set of anchors with clear line-of-sight to t,
  form the 2x2 "bearing information" matrix M = sum_i u_i u_i^T over unit
  bearing vectors u_i = (cos th_i, sin th_i). trace(M) = n_vis exactly (each
  u_i u_i^T has trace 1), so the uncertainty is
      GDOP(t) = sqrt(n_vis / det(M))          if n_vis >= 2 and det(M) >= DET_EPS
      GDOP(t) = PEN                            otherwise (unlocalizable / collinear)
  A scenario's cost is the MEAN GDOP over its target list; the instance cost F
  is the MAX over the K scenarios (the worst regime the single anchor layout
  must still serve). Smaller F is better (minimize).
"""
import math
import sys
from fractions import Fraction

PEN = 30.0
DET_EPS = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_instance(path):
    try:
        lines = open(path).read().splitlines()
    except Exception:
        fail("cannot read instance")
    p = 0
    try:
        W, H, A, K, T = map(int, lines[p].split()); p += 1
        grid = []
        for _ in range(H):
            row = lines[p]; p += 1
            if len(row) != W:
                raise ValueError("bad grid row width")
            grid.append(row)
        scenarios = []
        for _ in range(K):
            name = lines[p].strip(); p += 1
            targets = []
            for _ in range(T):
                r, c = map(int, lines[p].split()); p += 1
                targets.append((r, c))
            scenarios.append((name, targets))
    except Exception as e:
        fail("malformed instance (%s)" % e)
    return W, H, A, K, T, grid, scenarios


def open_cells(grid):
    H = len(grid); W = len(grid[0])
    return [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']


def los_clear(grid, p, q):
    """EXACT grid line-of-sight (a supercover / grid-traversal test, with a
    conservative no-diagonal-corner-cut rule): the open segment from p to q
    is subdivided, in exact rational arithmetic, at every point where it
    crosses a row or column cell boundary (a half-integer line). Between two
    consecutive crossings the segment stays inside exactly one cell --
    evaluating its midpoint identifies that cell exactly (no possibility of
    a diagonal "corner cut" through an unsampled wall, the failure mode of
    naive integer-step rounding). Separately: whenever a row-crossing and a
    column-crossing land at the EXACT SAME parameter t, the segment grazes a
    single grid corner shared by four cells and threads between only the two
    cells on ONE diagonal -- but a wall on the OTHER diagonal has zero real
    thickness at that mathematical point, and a real beacon must not count
    as "visible" by grazing it. All four cells touching such a corner are
    required to be open (the standard no-corner-cutting rule), not just the
    two the continuous path actually passes through."""
    (r0, c0), (r1, c1) = p, q
    dr, dc = r1 - r0, c1 - c0
    if dr == 0 and dc == 0:
        return True
    row_ts = set()
    col_ts = set()
    if dr != 0:
        lo, hi = (r0, r1) if dr > 0 else (r1, r0)
        for rb in range(lo, hi):
            t = Fraction(2 * (rb - r0) + 1, 2 * dr)
            if 0 < t < 1:
                row_ts.add(t)
    if dc != 0:
        lo, hi = (c0, c1) if dc > 0 else (c1, c0)
        for cb in range(lo, hi):
            t = Fraction(2 * (cb - c0) + 1, 2 * dc)
            if 0 < t < 1:
                col_ts.add(t)

    for t in row_ts & col_ts:
        rb = int(r0 + t * dr - Fraction(1, 2))  # exact: r0 + t*dr == rb + 0.5
        cb = int(c0 + t * dc - Fraction(1, 2))
        for rr in (rb, rb + 1):
            for cc in (cb, cb + 1):
                if not (0 <= rr < len(grid) and 0 <= cc < len(grid[0])) or grid[rr][cc] == '#':
                    return False

    ts = sorted(row_ts | col_ts | {Fraction(0), Fraction(1)})
    for i in range(len(ts) - 1):
        tm = (ts[i] + ts[i + 1]) / 2
        rr = round(r0 + tm * dr)
        cc = round(c0 + tm * dc)
        if grid[rr][cc] == '#':
            return False
    return True


def gdop(grid, target, anchors):
    sxx = syy = sxy = 0.0
    n = 0
    for (ar, ac) in anchors:
        if (ar, ac) == target:
            continue
        dr = ar - target[0]; dc = ac - target[1]
        d = math.hypot(dr, dc)
        if d < 1e-9:
            continue
        if not los_clear(grid, target, (ar, ac)):
            continue
        u, v = dc / d, dr / d
        sxx += u * u; syy += v * v; sxy += u * v
        n += 1
    if n < 2:
        return PEN
    det = sxx * syy - sxy * sxy
    if det < DET_EPS:
        return PEN
    return math.sqrt(n / det)


def worst_regime_mean(grid, scenarios, anchors):
    means = []
    for name, targets in scenarios:
        vals = [gdop(grid, t, anchors) for t in targets]
        means.append(sum(vals) / len(vals))
    return max(means)


def farthest_point_baseline(grid, A):
    """The checker's own internal reference construction (== what greedy.py
    implements): spread A anchors over the open cells by iterated
    farthest-point sampling, purely by Euclidean distance -- no notion of
    line-of-sight, bearing, or per-scenario target geometry at all.
    Tie-break is FULLY SPECIFIED (not left to hash/set iteration order):
    candidates are scanned in sorted (r, c) order and ties for the maximum
    min-distance keep the FIRST (lexicographically smallest) candidate."""
    cells = sorted(open_cells(grid))
    start = cells[len(cells) // 2]
    chosen = [start]
    remaining = [p for p in cells if p != start]
    while len(chosen) < A and remaining:
        best = None; bestd = -1
        for p in remaining:
            dmin = min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen)
            if dmin > bestd:
                bestd = dmin; best = p
        chosen.append(best)
        remaining.remove(best)
    return chosen


def main():
    W, H, A, K, T, grid, scenarios = parse_instance(sys.argv[1])
    open_set = set(open_cells(grid))

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != 2 * A:
        fail("expected exactly %d numbers (%d anchors x 2 coords), got %d" % (2 * A, A, len(otoks)))

    anchors = []
    for i in range(A):
        try:
            rv = float(otoks[2 * i]); cv = float(otoks[2 * i + 1])
        except Exception:
            fail("bad anchor token %d" % i)
        if not (math.isfinite(rv) and math.isfinite(cv)):
            fail("non-finite anchor %d" % i)
        if rv != round(rv) or cv != round(cv):
            fail("anchor %d not integer grid coordinates" % i)
        r, c = int(round(rv)), int(round(cv))
        if not (0 <= r < H and 0 <= c < W):
            fail("anchor %d = (%d,%d) out of bounds" % (i, r, c))
        if (r, c) not in open_set:
            fail("anchor %d = (%d,%d) is not an open cell" % (i, r, c))
        anchors.append((r, c))

    if len(set(anchors)) != A:
        fail("anchor positions are not all distinct")

    F = worst_regime_mean(grid, scenarios, anchors)

    baseline_anchors = farthest_point_baseline(grid, A)
    B = worst_regime_mean(grid, scenarios, baseline_anchors)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
