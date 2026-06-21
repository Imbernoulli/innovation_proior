The best single rectangle averaged $2704$ over the five seeds, and the feedback told me exactly where it leaked: on the seeds where mackerel and sardine overlap it hemorrhages sardine — $b=1795$ on seed 2, $1207$ on seed 5 — because a four-cornered box has no way to bend its boundary around a sardine pocket or to reach two mackerel pockets without swallowing the gap between them. The problem hands me a far richer net, any rectilinear simple polygon up to a thousand vertices, and the rectangle uses four of them. I want a net whose boundary can follow the *shape* of where the mackerel are, and the most direct way to get an arbitrary rectilinear region is to build it out of grid cells: choose a connected set of cells and its outer boundary is automatically a rectilinear polygon.

I propose **grid-cell greedy region growing** with a rectangle fallback, run at multiple resolutions. I bucket the sea into a $G\times G$ grid and give each cell a weight $(\#\text{mackerel}-\#\text{sardine})$. A net is now just a subset of cells, $a-b$ is the sum of the weights of the chosen cells, and the boundary — the unit edges where an inside cell meets an outside cell or the grid border — is a rectilinear closed curve. As long as the subset is connected and hole-free, that curve is a single simple polygon I can trace and emit. This turns "design an arbitrary rectilinear net" into "select a good connected cell region," which I attack greedily: start from the single densest cell and grow, at each step looking at the frontier (the outside cells adjacent to my current region) and adding the highest-weight one.

Two invariants must hold under every addition, and each gets a cheap incremental check. Perimeter must stay legal: adding a cell changes the boundary-edge count by a known local amount — it removes the shared edges with neighbors already inside and adds its own exposed edges, so $\Delta_{\text{edges}} = 4 - 2\cdot(\#\text{inside neighbors})$ — and I keep a running boundary count `bedges` so I can reject any cell that would push the traced perimeter over $4\times10^5$. I enforce this against a safe budget of $396000$ rather than the raw $4\times10^5$, leaving a margin for grid-line rounding in the traced polygon. The second invariant is hole-freeness: if adding a cell would enclose an empty pocket, the region stops being a simple polygon, so I flood-fill the outside from the grid border and forbid any addition that leaves an unreachable empty cell.

The load-bearing design choice is what the greedy is *allowed to add*. A pure "add only positive-weight cells" greedy stops far too early — the frontier of a good region is often a ring of slightly sardine-heavy cells, and beyond that ring sit rich mackerel cells the greedy will never reach because it refuses to step through the negative collar, so the region freezes small, below even the rectangle. So I let the greedy take the highest-weight frontier cell at each step *even when that weight is negative*: it can spend a little to bridge a gap. The danger is obvious — an unrestrained downhill walk would wander off — so I track the best total weight the region has ever had, snapshot the cell set at that maximum, and at the end *restore that best snapshot* rather than wherever the walk happened to stop. Bridging is allowed but never charged to the final answer unless it paid off. A patience counter of $G^2$ steps below the best stops a walk that is going nowhere.

Two cheap defenses make the rung robust rather than fragile, and both follow from honest limits. A single grid resolution is never right for every layout — too coarse and the boundary cannot hug the fish, too fine and each cell is so small that the fixed perimeter budget wraps far less area and the region stays small — so I run the greedy at $G\in\{30,40,50\}$ and keep whichever region scores best by the internal weight; trying three is nearly free. And I include the rung-1 rectangle itself as one more candidate, computed by the same prefix-sum sweep, so that whenever the grown region fails to beat the box I fall back to the box. This guarantees the rung *dominates* rung 1: it is the rectangle plus the option of a carved rectilinear region, picking the better of the two by internal estimate, with the frozen exact evaluator having the final word.

This is the right second step and no more. Its engine is a one-shot forward greedy with no reversibility: it cannot take back a cell that looked good when the region was small but became a liability once the shape settled, the perimeter ceiling lets a ragged, many-notched boundary saturate the budget on a mediocre region, and it is at the mercy of a single grid resolution per run. The way past all three is the same move — stop treating region construction as a forward-only greedy and start treating it as *search*, where cells can be added and removed, downhill steps are accepted to escape the traps, and the only price is making each candidate move cheap enough to afford millions. That is the next rung.

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
