**Problem.** AHC039: place one axis-aligned rectilinear simple polygon (the net) in `[0,10^5]^2`
to maximize `a − b + 1`, where `a` = mackerel inside (boundary counts inside) and `b` = sardine
inside, under `4 ≤ m ≤ 1000` vertices, axis-parallel edges, simplicity, and perimeter `≤ 4×10^5`.
This rung emits the strongest single axis-aligned **rectangle** (a 4-vertex baseline).

**Key idea.** Stamp weight `+1` at every mackerel and `−1` at every sardine; the value `a − b` of
any axis-aligned box is the total weight it covers, so the best rectangle is the maximum-weight
axis-aligned rectangle on a signed point set. Discretize the sea into a coarse grid of candidate
cut lines, bucket per-cell `(#mackerel − #sardine)`, build a 2D prefix-sum table, and sweep all
`(x1<x2, y1<y2)` grid-line boxes in O(1) each, skipping any whose perimeter `2(w+h)` exceeds the
budget. Emit the best box as a 4-vertex polygon.

**Why these choices.** A rectangle is the simplest *always-legal* net (4 vertices, axis-parallel
edges, simple, perimeter controllable), so it is a zero-risk floor. The prefix-sum sweep finds the
exact best box up to grid resolution in O(G^4) on a G×G grid, and the perimeter check is folded
into the loop so no illegal box is ever considered. The internal score is a grid estimate; the
*reported* score is the frozen evaluator's exact integer point-in-polygon count, so the
discretization only guides the choice and never inflates the result. The rectangle is deliberately
the floor: convex and single-bodied, it cannot separate interleaved mackerel and sardine — that is
left to later rungs.

**Hyperparameters / contract.** Grid resolution `K = 60` cut lines per axis (uniform). Perimeter
budget `4×10^5` enforced per candidate. Output: `m = 4` and the four corners of the best box.
Deterministic. Reads instance on stdin (`N`; `N` mackerel `x y`; `N` sardine `x y`), writes the net
on stdout.

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
