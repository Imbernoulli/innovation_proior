Before reaching for anything clever I want the simplest object that is guaranteed valid and gives me a number every later net must beat. The constraints are forgiving about one shape in particular: a single axis-aligned rectangle is always a legal net. It has four vertices, which sits inside $4 \le m \le 1000$; its four edges are axis-parallel by construction; it is trivially simple; and as long as I keep it from spanning the whole grid its perimeter stays under $4\times10^5$. So a rectangle is a zero-risk floor I can stand on, and the only real question is *which* rectangle.

I propose the **best axis-aligned rectangle via a 2D prefix-sum maximum-weight box**. The objective inside a rectangle is clean: if I fix the four sides at $x\in[x_1,x_2]$, $y\in[y_1,y_2]$, then $a-b$ is just the number of mackerel in that box minus the number of sardine in it. That is a sum of per-point contributions where each mackerel counts $+1$ and each sardine counts $-1$, so if I stamp a weight $+1$ at every mackerel and $-1$ at every sardine, the value of a rectangle is exactly the total weight it covers. Maximizing $a-b$ over all axis-aligned boxes is therefore the classic *maximum-weight axis-aligned rectangle* problem on a signed point set. I cannot try every real-valued box, but I do not need to: the optimum's sides can always be snapped to coordinate lines through the data, so the search is finite, and prefix sums make it cheap.

The mechanism is a coarse-grid sweep. I lay down a uniform set of $K=60$ candidate cut lines per axis across $[0,10^5]$ — a coarse grid is enough to find a strong box, and refining it only costs more grid lines. I bucket every fish into its grid cell, accumulating per-cell $(\#\text{mackerel}-\#\text{sardine})$ into a table $W$, then build a 2D prefix-sum table $PS$ over the cells so that the signed weight of *any* grid-aligned rectangle is a constant-time difference of four entries,
$$\text{rectsum}(i_1,i_2,j_1,j_2)=PS[i_2][j_2]-PS[i_1][j_2]-PS[i_2][j_1]+PS[i_1][j_1].$$
Now I sweep over all $(x_1<x_2,\;y_1<y_2)$ choices of grid lines, read each box's $a-b$ in $O(1)$, and keep the best. On a $G\times G$ grid this is $O(G^4)$, which $G=60$ keeps well within budget.

One design choice is load-bearing: I fold the perimeter ceiling directly into the loop, skipping any box whose perimeter $2(w+h)$ would exceed $4\times10^5$ before I even score it. A box that is too wide or too tall is illegal no matter how good its catch, so the perimeter budget is a real constraint even on a rectangle, and checking it at scoring time rather than discovering it afterward keeps every candidate legal by construction. A second subtlety I get right because it haunts every later rung: the grid I optimize on is a discretization, so a box snapped to coarse grid lines is only an approximation and a fish near a boundary line could be counted on the wrong side. But the *evaluator* is exact — a real integer point-in-polygon test with on-boundary-counts-as-inside — so the grid estimate only *guides* the choice; I report the evaluator's exact $a-b+1$, never the internal estimate, so the discretization can never inflate the result.

I am proposing the bounding box as the *right* first rung and nothing more. Its limitation is baked into its shape, not its tuning: convex and single-bodied, with only four corners, it cannot separate interleaved species. Any rectangle large enough to capture a mackerel pocket also swallows the sardine threaded through it; any rectangle small enough to dodge that sardine clips off mackerel at the edges. It cannot reach around a sardine cluster, bridge two separated mackerel pockets without paying for the sardine-rich gap between them, or cut a single notch. The moment I want to exclude a sardine pocket sitting inside a mackerel shoal I need a boundary that can bend — a rectilinear region with many vertices — and that is what the next rung builds, by giving up the single box for a *grid of cells* I can selectively include.

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
