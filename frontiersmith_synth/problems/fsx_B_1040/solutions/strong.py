# TIER: strong
# The insight: the design problem is really a COVER of K viewpoint-geometry
# requirements, one per scenario -- and the checker's own scoring function
# (worst-regime mean GDOP) is fully computable from the input, so instead of
# a geometry heuristic blind to line-of-sight (the greedy tier), directly
# search for anchors that reduce it.
#
# 1. Seed with the same Euclidean farthest-point spread the greedy tier uses
#    (guarantees no scenario region is left with literally zero nearby
#    anchors -- a purely bearing-driven forward search can get stuck
#    starving one scenario while it fixates on another).
# 2. Refine by repeated single-anchor EXCHANGE: for each anchor, try
#    swapping it for every other open cell and keep the swap if it improves
#    the LEXICOGRAPHIC objective (max scenario mean, then sum of scenario
#    means). The lexicographic tie-break is what lets the search keep
#    chipping away at an under-served scenario even while the currently-
#    worst scenario (typically the corridor, where anchors on the same
#    straight stretch are exactly collinear and give det(M)=0) is unmoved --
#    a plain "does this reduce the max" acceptance rule stalls out because a
#    lone anchor can't fix a scenario that needs a SECOND, non-collinear
#    anchor to escape the singular case.
# 3. Bearings are visibility-gated (line-of-sight through walls) and
#    precomputed once per (candidate, target) pair, since LOS/bearing between
#    a fixed candidate and a fixed target does not depend on which OTHER
#    anchors are chosen -- this is what keeps the exchange search fast.
import math
import sys
from fractions import Fraction

PEN = 30.0
DET_EPS = 1e-6


def read_instance():
    data = sys.stdin.read().splitlines()
    p = 0
    W, H, A, K, T = map(int, data[p].split()); p += 1
    grid = []
    for _ in range(H):
        grid.append(data[p]); p += 1
    scenarios = []
    for _ in range(K):
        name = data[p].strip(); p += 1
        targets = []
        for _ in range(T):
            r, c = map(int, data[p].split()); p += 1
            targets.append((r, c))
        scenarios.append((name, targets))
    return W, H, A, grid, scenarios


def open_cells(grid):
    H = len(grid); W = len(grid[0])
    return [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']


def los_clear(grid, p, q):
    """EXACT grid line-of-sight (matches verify.py, including the
    no-diagonal-corner-cutting rule): subdivide the segment, in exact
    rational arithmetic, at every row/column cell-boundary crossing so no
    diagonal "corner cut" through an unsampled wall cell is possible, and
    whenever a row- and column-crossing coincide exactly (a corner graze)
    require all four cells touching that corner to be open."""
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
        rb = int(r0 + t * dr - Fraction(1, 2))
        cb = int(c0 + t * dc - Fraction(1, 2))
        for rr in (rb, rb + 1):
            for cc in (cb, cb + 1):
                if not (0 <= rr < len(grid) and 0 <= cc < len(grid[0])) or grid[rr][cc] == '#':
                    return False

    ts = sorted(row_ts | col_ts | {Fraction(0), Fraction(1)})
    for i in range(len(ts) - 1):
        tm = (ts[i] + ts[i + 1]) / 2
        rr = round(r0 + tm * dr); cc = round(c0 + tm * dc)
        if grid[rr][cc] == '#':
            return False
    return True


def farthest_point(cells, A):
    # Same fully-specified tie-break as greedy.py / verify.py's baseline:
    # sorted-order scan, first (lexicographically smallest) cell wins ties.
    cells_sorted = sorted(cells)
    start = cells_sorted[len(cells_sorted) // 2]
    chosen = [start]
    remaining = [p for p in cells_sorted if p != start]
    while len(chosen) < A and remaining:
        best = None; bestd = -1
        for p in remaining:
            dmin = min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen)
            if dmin > bestd:
                bestd = dmin; best = p
        chosen.append(best)
        remaining.remove(best)
    return chosen


def precompute_bearings(grid, cells, targets):
    bearing = {}
    for ci, (ar, ac) in enumerate(cells):
        for ti, t in enumerate(targets):
            if (ar, ac) == t or not los_clear(grid, t, (ar, ac)):
                continue
            dr = ar - t[0]; dc = ac - t[1]
            d = math.hypot(dr, dc)
            if d < 1e-9:
                continue
            bearing[(ci, ti)] = (dc / d, dr / d)
    return bearing


def tgt_gdop(acc_entry):
    sxx, syy, sxy, n = acc_entry
    if n < 2:
        return PEN
    det = sxx * syy - sxy * sxy
    if det < DET_EPS:
        return PEN
    return math.sqrt(n / det)


def scen_means(acc, scen_target_idx):
    return [sum(tgt_gdop(acc[ti]) for ti in idxs) / len(idxs) for idxs in scen_target_idx.values()]


def lex_key(acc, scen_target_idx):
    means = scen_means(acc, scen_target_idx)
    return (max(means), sum(means))


def solve(grid, scenarios, A, rounds=20):
    cells = open_cells(grid)
    cell_idx = {p: i for i, p in enumerate(cells)}
    targets_all = []
    scen_target_idx = {}
    off = 0
    for name, targets in scenarios:
        targets_all.extend(targets)
        scen_target_idx[name] = list(range(off, off + len(targets)))
        off += len(targets)
    ntargets = len(targets_all)
    bearing = precompute_bearings(grid, cells, targets_all)

    def add_bearing(acc, ci, sign=1):
        for ti in range(ntargets):
            b = bearing.get((ci, ti))
            if b is None:
                continue
            u, v = b
            acc[ti][0] += sign * u * u
            acc[ti][1] += sign * v * v
            acc[ti][2] += sign * u * v
            acc[ti][3] += sign

    seed_pts = farthest_point(cells, A)
    chosen = [cell_idx[p] for p in seed_pts]
    acc = [[0.0, 0.0, 0.0, 0] for _ in range(ntargets)]
    for ci in chosen:
        add_bearing(acc, ci, 1)
    remaining = set(range(len(cells))) - set(chosen)

    for _ in range(rounds):
        improved = False
        for pos in range(len(chosen)):
            cur = chosen[pos]
            add_bearing(acc, cur, -1)
            base_key = lex_key(acc, scen_target_idx)
            best_ci = cur; best_key = base_key
            for ci in remaining:
                add_bearing(acc, ci, 1)
                key = lex_key(acc, scen_target_idx)
                add_bearing(acc, ci, -1)
                if key < best_key:
                    best_key = key; best_ci = ci
            add_bearing(acc, best_ci, 1)
            if best_ci != cur:
                chosen[pos] = best_ci
                remaining.discard(best_ci); remaining.add(cur)
                improved = True
        if not improved:
            break

    return [cells[ci] for ci in chosen]


def main():
    W, H, A, grid, scenarios = read_instance()
    anchors = solve(grid, scenarios, A)
    out = []
    for (r, c) in anchors:
        out.append(f"{r} {c}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
