We must place one axis-aligned rectilinear simple net in the sea to maximize $\max(0, a - b + 1)$,
the mackerel caught minus the sardine caught, under $4 \le m \le 1000$ vertices, axis-parallel
edges, simplicity, and perimeter at most $4 \times 10^5$. The $N = 5000$ mackerel and $N = 5000$
sardine come from overlapping clustered shoals, so the two species interleave and the ideal net is
an irregular rectilinear region, not a box. The natural baseline — a single best axis-aligned
rectangle — does well when one mackerel cluster sits apart from the sardine, but it hemorrhages
sardine wherever the shoals overlap, because a rectangle is convex and single-bodied: it cannot bend
its boundary around a sardine pocket inside a mackerel shoal, and it cannot reach two separated
mackerel pockets without swallowing the sardine-rich gap between them. The problem hands us up to a
thousand vertices; the rectangle uses four. We need a net whose boundary can follow the *shape* of
where the mackerel are.

I propose a **grid-cell greedy region-growing** method. The most direct way to get a rectilinear
region of arbitrary shape is to build it out of grid cells: bucket the sea into a $G \times G$ grid,
give each cell weight $(\#\text{mackerel} - \#\text{sardine})$ inside it, and let the net be a
connected subset of cells. The outer boundary of such a subset — the unit edges where an inside cell
meets an outside cell or the grid border — is automatically a rectilinear closed curve, and as long
as the subset is connected and hole-free, that curve is a single simple polygon we can trace and
emit. With this representation $a - b$ is just the sum of the chosen cells' weights, so "design an
arbitrary rectilinear net" becomes "select a good connected cell region," a combinatorial problem I
attack greedily: start from the single densest cell and, at each step, add the frontier cell (an
outside cell adjacent to the region) that most increases the total weight.

Two invariants must hold under every addition, and each gets a cheap incremental check. The
perimeter must stay legal: adding a cell changes the boundary edge count by a known local amount —
it removes the edges shared with neighbors already inside and adds its exposed edges, which for a
cell with $s$ inside-neighbors is $\Delta = 4 - 2s$ — so I keep a running boundary-edge count and
reject any cell that would push the traced perimeter past $4 \times 10^5$. And I must not punch a
hole: if adding a cell would enclose an empty pocket, the region stops being a simple polygon, so I
flood-fill the outside from the grid border and forbid any addition that leaves an unreachable empty
cell.

The first wall is that a pure "add only positive-weight cells" greedy stops far too early. The
frontier of a good region is often a ring of slightly sardine-heavy cells, and beyond that ring sit
rich mackerel cells the greedy never reaches because it refuses to step through the negative collar;
the region freezes small, below even the rectangle. The fix is to let the greedy take the
*highest-weight* frontier cell at each step even when that weight is negative — so it can spend a
little to bridge a gap — but to track the best total weight the region has ever attained and, at the
end, restore that best snapshot rather than wherever the walk happened to stop. Bridging is thus
allowed but never charged to the final answer unless it pays off. I also guard against wandering too
long below the best with a patience counter of $G \times G$ steps.

Two cheap defenses make the method robust rather than brilliant, and they are deliberate. A static
grid resolution is never right for every layout — too coarse and the boundary cannot hug the fish,
too fine and each cell is so small that the fixed perimeter budget covers little area — so I run the
greedy at several resolutions, $G \in \{30, 40, 50\}$, and keep whichever region scores best by the
internal weight; trying three is nearly free. And I include the best prefix-sum rectangle itself as
one more candidate, so that whenever the grown region fails to beat the box, I simply fall back to
the box. This guarantees the method *dominates* the rectangle baseline: it is the rectangle plus the
option of a carved rectilinear region, picking the better of the two by internal estimate, with the
exact evaluator having the final word. The internal weight only ranks candidates; the reported score
is always the frozen evaluator's exact count.

What this buys is exactly the overlapping-shoal case: the grown region's rectilinear boundary carves
sardine out of a mackerel shoal where the rectangle could not. What it cannot do is undo itself — it
is a single forward pass, so a cell that looked good early can become a liability once the shape is
settled, and the greedy cannot remove it; the perimeter ceiling makes this worse by letting a ragged
boundary saturate the budget on a mediocre region. Those are failures of irreversibility, and they
are the opening for replacing greedy growth with reversible local search.

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
