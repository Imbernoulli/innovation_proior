We are given the sea $[0, 10^5] \times [0, 10^5]$ with $N = 5000$ mackerel (targets) and $N = 5000$
sardine (penalties) at integer coordinates, and we must place one net — a closed, axis-aligned
rectilinear simple polygon — to maximize $\max(0, a - b + 1)$, where $a$ is the mackerel caught and
$b$ the sardine caught (a fish inside the net or on its boundary counts as caught). The net must
have $4 \le m \le 1000$ vertices, axis-parallel edges, no self-intersection, and total edge length
at most $4 \times 10^5$. Before reaching for any region-shaping machinery, I want the simplest net
that is guaranteed legal and a concrete objective to beat. The full freedom of a thousand-vertex
rectilinear polygon is exactly what makes the problem hard, but it also makes it easy to produce an
*illegal* net by accident — a self-intersection, an overlong perimeter. So the right first move is
the one shape that cannot go wrong.

I propose the **best axis-aligned rectangle**: choose a single box to maximize $a - b$, emit it as a
four-vertex net. A rectangle is always legal — four vertices sit inside the $m$ bounds, its four
edges are axis-parallel, it is trivially simple, and its perimeter $2(w+h)$ is controllable — so the
only question is which box. The key observation is that the objective inside a box is a sum of
per-point contributions: stamp weight $+1$ at every mackerel and $-1$ at every sardine, and then
$a - b$ for a box equals the total signed weight the box covers. Maximizing $a - b$ over all
axis-aligned boxes is therefore the classic maximum-weight axis-aligned rectangle problem on a
signed point set. I do not need to try every real-valued box, because the optimum's sides can always
be snapped to coordinate lines, making the search finite; and I make it cheap with prefix sums.

Concretely, I lay down a coarse grid of candidate cut lines across each axis — a uniform set of $K$
lines, refinable at the cost of more lines — and bucket every fish into the grid cell it falls in,
accumulating per-cell $(\#\text{mackerel} - \#\text{sardine})$. I then build a 2D prefix-sum table
$\mathrm{PS}$ over the cells, so the signed weight of any grid-aligned rectangle is a constant-time
difference of four entries,
$$\text{weight}(i_1,i_2,j_1,j_2) = \mathrm{PS}[i_2][j_2] - \mathrm{PS}[i_1][j_2] - \mathrm{PS}[i_2][j_1] + \mathrm{PS}[i_1][j_1].$$
Now I sweep over all choices of grid lines $(x_1 < x_2,\, y_1 < y_2)$, read each box's $a - b$ in
$O(1)$, and keep the best — while skipping any box whose perimeter $2(w+h)$ would exceed
$4 \times 10^5$. That perimeter check matters even for a rectangle: a box too wide or too tall is
illegal no matter how rich its catch, so the budget is a real constraint I bake into the loop rather
than discover at scoring time.

There is one discipline I impose on the scoring, because it will matter for every later, more
flexible net. The grid used for the prefix-sum search is a discretization: a box snapped to coarse
grid lines only approximates the true best box, and a fish near a boundary line could be counted on
the wrong side. But the evaluator is exact — a real integer point-in-polygon test with
boundary-counts-as-inside — so the number I optimize internally (a grid estimate) and the number I am
graded on (the exact count) can differ slightly. For a rectangle this gap is small and harmless: the
grid merely picks a good box, and the exact evaluator reports its true $a - b + 1$. I make the grid
resolution fine enough that the estimate tracks the truth, and I report only the evaluator's exact
score, never the internal estimate.

This does surprisingly well when the mackerel cluster densely in one compact region with sardine
mostly elsewhere, and it fails exactly where the problem is built to make it fail — overlapping
shoals, where any box large enough to capture a mackerel pocket also swallows the sardine threaded
through it, and any box small enough to dodge the sardine clips off mackerel at the edges. That
limitation is in the shape, not the tuning: with one convex body and four corners, a rectangle
cannot reach around a sardine cluster or bridge two mackerel pockets. It is the right floor — the
simplest guaranteed-legal net, found exactly up to grid resolution by a fast budget-respecting
sweep — and it sets the honest baseline every later rung must beat.

```python
#!/usr/bin/env python3
"""AHC039 rung 1: best axis-aligned single rectangle maximizing a - b.

Stamp +1 at each mackerel, -1 at each sardine; the value of an axis-aligned box
is the signed weight it covers. Coarse-grid the sea, bucket per-cell weight,
build 2D prefix sums, and sweep all grid-line boxes in O(1) each under the
perimeter budget. Emit the best box as a 4-vertex rectangle.
"""
import sys, bisect


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); idx = 1
    mack = []
    for _ in range(n):
        mack.append((int(data[idx]), int(data[idx + 1]))); idx += 2
    sard = []
    for _ in range(n):
        sard.append((int(data[idx]), int(data[idx + 1]))); idx += 2

    K = 60  # grid resolution per axis (candidate cut lines)
    xs = sorted(set([0, 100000] + [int(i * 100000 / K) for i in range(K + 1)]))
    ys = sorted(set([0, 100000] + [int(i * 100000 / K) for i in range(K + 1)]))
    nx, ny = len(xs), len(ys)

    W = [[0] * ny for _ in range(nx)]

    def cell(px, py):
        ci = min(max(bisect.bisect_right(xs, px) - 1, 0), nx - 2)
        cj = min(max(bisect.bisect_right(ys, py) - 1, 0), ny - 2)
        return ci, cj

    for (x, y) in mack:
        ci, cj = cell(x, y); W[ci][cj] += 1
    for (x, y) in sard:
        ci, cj = cell(x, y); W[ci][cj] -= 1

    # 2D prefix sums over cells
    PS = [[0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(nx - 1):
        for j in range(ny - 1):
            PS[i + 1][j + 1] = W[i][j] + PS[i][j + 1] + PS[i + 1][j] - PS[i][j]

    def rectsum(i1, i2, j1, j2):  # cells [i1,i2) x [j1,j2)
        return PS[i2][j2] - PS[i1][j2] - PS[i2][j1] + PS[i1][j1]

    best = -10**9; bestrect = (0, 0, 1, 1)
    for i1 in range(nx - 1):
        for i2 in range(i1 + 1, nx):
            w = xs[i2] - xs[i1]
            for j1 in range(ny - 1):
                for j2 in range(j1 + 1, ny):
                    h = ys[j2] - ys[j1]
                    if 2 * (w + h) > 400000:    # perimeter budget
                        continue
                    s = rectsum(i1, i2, j1, j2)
                    if s > best:
                        best = s; bestrect = (i1, i2, j1, j2)

    i1, i2, j1, j2 = bestrect
    x1, x2, y1, y2 = xs[i1], xs[i2], ys[j1], ys[j2]
    print(4)
    print(x1, y1)
    print(x2, y1)
    print(x2, y2)
    print(x1, y2)


if __name__ == "__main__":
    main()
```
