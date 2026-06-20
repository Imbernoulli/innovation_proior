**Problem.** AHC039: maximize `a - b + 1` (mackerel minus sardine caught, +1) with one
axis-aligned rectilinear simple net under `4 <= m <= 1000`, axis-parallel edges, simplicity, and
perimeter `<= 4x10^5`. This rung carves a rectilinear region out of a coarse grid by greedy
region-growing, with the rung-1 rectangle kept as a fallback so it dominates the baseline.

**Key idea.** Bucket the sea into a `G x G` grid; cell weight = `(#mackerel - #sardine)`. A net is
a connected, hole-free subset of cells whose outer boundary traces a rectilinear simple polygon,
and `a - b` is the sum of chosen-cell weights. Grow from the densest seed, at each step adding the
highest-weight admissible frontier cell (allowing a negative step to bridge gaps), rejecting any
add that would exceed the perimeter budget or punch a hole (checked by flood-filling the outside).
Track the best total weight seen and restore that snapshot. Trace the boundary, merge collinear
runs, emit the polygon.

**Why these choices.** Grid cells give an arbitrary rectilinear boundary for free, so the net can
bend around sardine that a rectangle cannot. The hole check (border flood-fill) keeps the region a
single simple polygon; the running boundary-edge count keeps perimeter legal. A purely-positive
greedy freezes small, so bridging through negative cells with a best-snapshot restore is essential.
Because one grid resolution is never right for every layout, we run `G in {30,40,50}` and keep the
best by internal weight, and we add a prefix-sum **best-rectangle** candidate so the rung never
scores below rung 1. The internal weight only ranks candidates; the frozen evaluator's exact count
is the reported score.

**Hyperparameters / contract.** Grid resolutions `G in {30,40,50}`; perimeter safe-budget `396000`
(margin for grid-line rounding under the `4x10^5` cap); patience `G*G` greedy steps below the best
before stopping; rectangle candidate on a `50x50` grid. Deterministic. Reads instance on stdin,
writes `m` then `m` vertices on stdout.

```python
#!/usr/bin/env python3
"""Rung 2: grid-cell greedy region growing (multi-resolution + rectangle fallback).

Bucket [0,10^5]^2 into a G x G grid; cell weight = (#mackerel - #sardine).
Greedily grow a connected, hole-free cell region from the densest seed, adding
at each step the admissible frontier cell of highest weight, while keeping the
traced rectilinear boundary's perimeter <= 4e5. Keep the best running snapshot.
We trace the outer boundary into a simple rectilinear polygon. We try several
grid resolutions and also a single best rectangle, and emit the candidate with
the highest internal (a-b) estimate.
"""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); idx = 1
    mack = []
    for _ in range(n):
        mack.append((int(data[idx]), int(data[idx+1]))); idx += 2
    sard = []
    for _ in range(n):
        sard.append((int(data[idx]), int(data[idx+1]))); idx += 2

    PERIM_BUDGET = 400000
    SAFE_BUDGET = 396000  # margin for grid-line rounding in the traced polygon

    def build_grid(G):
        cw = 100000 / G
        W = [[0]*G for _ in range(G)]
        def gi(v):
            c = int(v / cw); return min(max(c, 0), G-1)
        for (x, y) in mack:
            W[gi(x)][gi(y)] += 1
        for (x, y) in sard:
            W[gi(x)][gi(y)] -= 1
        return W, cw

    def neighbors(i, j, G):
        if i+1 < G: yield i+1, j
        if i-1 >= 0: yield i-1, j
        if j+1 < G: yield i, j+1
        if j-1 >= 0: yield i, j-1

    def grow(G):
        W, cw = build_grid(G)
        inset = [[False]*G for _ in range(G)]
        bi = bj = 0; bw = -10**9
        for i in range(G):
            for j in range(G):
                if W[i][j] > bw:
                    bw = W[i][j]; bi, bj = i, j
        inset[bi][bj] = True
        weight = W[bi][bj]
        bedges = 4  # boundary unit-edges of current region

        def delta_edges(i, j):
            d = 4
            for ni, nj in neighbors(i, j, G):
                if inset[ni][nj]:
                    d -= 2  # shared edge removed from both
            return d

        def creates_hole():
            seen = [[False]*G for _ in range(G)]
            dq = deque()
            for i in range(G):
                for j in range(G):
                    if (i in (0, G-1) or j in (0, G-1)) and not inset[i][j] and not seen[i][j]:
                        seen[i][j] = True; dq.append((i, j))
            while dq:
                i, j = dq.popleft()
                for ni, nj in neighbors(i, j, G):
                    if not inset[ni][nj] and not seen[ni][nj]:
                        seen[ni][nj] = True; dq.append((ni, nj))
            for i in range(G):
                for j in range(G):
                    if not inset[i][j] and not seen[i][j]:
                        return True
            return False

        best_weight = weight
        best_snapshot = [row[:] for row in inset]
        steps_since_best = 0
        PATIENCE = G * G  # generous: climb through dips
        forbidden = set()
        frontier = set()
        for ni, nj in neighbors(bi, bj, G):
            frontier.add((ni, nj))

        while frontier:
            best = None; bestgain = -10**9; bestde = 0
            for (i, j) in frontier:
                if (i, j) in forbidden:
                    continue
                de = delta_edges(i, j)
                if (bedges + de) * cw > SAFE_BUDGET:
                    continue
                g = W[i][j]
                if g > bestgain:
                    bestgain = g; best = (i, j); bestde = de
            if best is None:
                break
            i, j = best
            inset[i][j] = True
            if creates_hole():
                inset[i][j] = False
                forbidden.add(best)
                frontier.discard(best)
                continue
            bedges += bestde
            weight += bestgain
            frontier.discard(best)
            for ni, nj in neighbors(i, j, G):
                if not inset[ni][nj] and (ni, nj) not in forbidden:
                    frontier.add((ni, nj))
            if weight > best_weight:
                best_weight = weight
                best_snapshot = [row[:] for row in inset]
                steps_since_best = 0
            else:
                steps_since_best += 1
                if steps_since_best > PATIENCE:
                    break

        inset = best_snapshot
        poly = trace(inset, G, cw)
        return poly, best_weight

    def trace(inset, G, cw):
        def cx(i): return int(round(i*cw))
        edges = {}
        for i in range(G):
            for j in range(G):
                if not inset[i][j]:
                    continue
                below = inset[i][j-1] if j-1 >= 0 else False
                if not below: edges[(i, j)] = (i+1, j)
                above = inset[i][j+1] if j+1 < G else False
                if not above: edges[(i+1, j+1)] = (i, j+1)
                left = inset[i-1][j] if i-1 >= 0 else False
                if not left: edges[(i, j+1)] = (i, j)
                right = inset[i+1][j] if i+1 < G else False
                if not right: edges[(i+1, j)] = (i+1, j+1)
        if not edges:
            return None
        start = next(iter(edges))
        cells = []; cur = start; cap = len(edges) + 5
        while True:
            cells.append(cur)
            cur = edges.get(cur)
            if cur is None or cur == start or len(cells) > cap:
                break
        pts = [(cx(ci), cx(cj)) for (ci, cj) in cells]
        m = len(pts); poly = []
        for k in range(m):
            a = pts[(k-1) % m]; b = pts[k]; c = pts[(k+1) % m]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                continue
            poly.append(b)
        if len(poly) < 4 or len(poly) > 1000:
            return None
        return poly

    def best_rectangle(G=50):
        # best axis-aligned rectangle on a GxG grid via 2D prefix sums of
        # (mackerel - sardine), respecting the perimeter budget.
        W, cw = build_grid(G)
        PS = [[0]*(G+1) for _ in range(G+1)]
        for i in range(G):
            for j in range(G):
                PS[i+1][j+1] = W[i][j] + PS[i][j+1] + PS[i+1][j] - PS[i][j]
        def rs(i1, i2, j1, j2):
            return PS[i2][j2]-PS[i1][j2]-PS[i2][j1]+PS[i1][j1]
        def cx(i): return int(round(i*cw))
        best = -10**9; br = (0, 0, 1, 1)
        for i1 in range(G):
            for i2 in range(i1+1, G+1):
                w = cx(i2)-cx(i1)
                for j1 in range(G):
                    for j2 in range(j1+1, G+1):
                        h = cx(j2)-cx(j1)
                        if 2*(w+h) > PERIM_BUDGET:
                            continue
                        s = rs(i1, i2, j1, j2)
                        if s > best:
                            best = s; br = (i1, i2, j1, j2)
        i1, i2, j1, j2 = br
        x1, x2, y1, y2 = cx(i1), cx(i2), cx(j1), cx(j2)
        return best, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    candidates = []
    for G in (30, 40, 50):
        poly, w = grow(G)
        if poly is not None:
            candidates.append((w, poly))
    rw, rpoly = best_rectangle(50)
    candidates.append((rw, rpoly))
    candidates.sort(key=lambda t: t[0], reverse=True)
    w, poly = candidates[0]
    print(len(poly))
    for (x, y) in poly:
        print(x, y)


if __name__ == "__main__":
    main()
```
