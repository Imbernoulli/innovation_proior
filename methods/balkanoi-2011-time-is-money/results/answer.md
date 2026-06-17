# Time is Money — minimizing (Σ time)·(Σ cost) over spanning trees

## Problem

Connected graph, $V \le 200$ vertices, $E \le 10000$ edges; each edge $e$ has a time $t_e$ and a cost $c_e$ ($1 \le t_e, c_e < 256$). Among all spanning trees $T$, minimize

$$\Big(\sum_{e\in T} t_e\Big)\cdot\Big(\sum_{e\in T} c_e\Big),$$

and output the tree. The objective is a product of two sums, so it is **not** additive over edges — Kruskal/Prim cannot be applied to the raw weights.

## Key idea

Map each spanning tree $T$ to the planar point $P(T) = (X, Y) = (\sum t_e,\ \sum c_e)$. The objective is $X\cdot Y$, minimized over the finite (but exponential) set $S$ of achievable points, all in the positive quadrant.

1. **The optimum is a lower-left convex-hull vertex of $S$.** If $p=(X_*,Y_*)$ minimizes the product and $k=X_*Y_*$, the branch $Y=k/X$ is convex in the positive quadrant. Its tangent at $p$ has equation $Y_*X+X_*Y=2k$, and every achievable point $(X,Y)$ has $XY\ge k$, so AM-GM gives $Y_*X+X_*Y\ge 2k$. Thus the optimum is supported by a line with positive normal on the lower-left convex hull. If it lies inside a flat hull edge, $XY$ along that negative-slope segment is concave, so one endpoint has product no larger; an optimal hull vertex always exists.

2. **A hull vertex = the minimizer of a linear functional = an MST.** For a direction $(a,b)$ with $a,b\ge 0$,
$$a X + b Y = \sum_{e\in T}\big(a\,t_e + b\,c_e\big),$$
which is a single sum of nonnegative per-edge weights $w_e = a t_e + b c_e$. Its minimizer over spanning trees is the **minimum spanning tree** under weight $w$ — one Kruskal call ($O(E\log E)$) yields the hull vertex extreme in direction $(a,b)$.

3. **Hull-probing recursion.** Start from the two axis-extreme vertices: $s = $ MST under $(1,0)$ (min total time, leftmost) and $e = $ MST under $(0,1)$ (min total cost, bottommost). For a segment $s\to e$, probe the down-left normal direction $(a,b)=(s_Y-e_Y,\ e_X-s_X)$ (both $\ge 0$). The MST under it returns the point $m$ that bulges furthest below segment $se$. With `cross((s - e), (m - e))`, a genuine lower-left bulge has positive sign; if that value is `> 0`, recurse on $(s,m)$ and $(m,e)$, otherwise the hull there is the segment $se$ and stop. Evaluate $X\cdot Y$ at every vertex found; keep the best, then rerun the winning MST to print its edges.

This enumerates exactly the lower-left hull vertices, so the global minimum of $X\cdot Y$ is found.

## Code

```python
def mst(n, edges, weight_fn):
    """Minimize sum over a spanning tree of a LINEAR per-edge weight.
    n: number of vertices (0-indexed); edges: list of (u, v, t, c).
    weight_fn(t, c) -> number: the per-edge weight to minimize over spanning trees.
    Returns (sum_t, sum_c, chosen_edge_indices) for the resulting tree.
    Standard Kruskal + union-find (disjoint-set with path compression)."""
    m = len(edges)
    order = sorted(range(m), key=lambda i: weight_fn(edges[i][2], edges[i][3]))
    pa = list(range(n))
    def find(x):
        while pa[x] != x:
            pa[x] = pa[pa[x]]                   # path compression
            x = pa[x]
        return x
    sum_t = sum_c = 0
    chosen = []
    for i in order:                             # Kruskal: take the lightest safe edge
        u, v, t, c = edges[i]
        ru, rv = find(u), find(v)
        if ru != rv:
            pa[ru] = rv
            sum_t += t
            sum_c += c
            chosen.append(i)
    return (sum_t, sum_c, chosen)


def solve(n, edges):
    """edges: list of (u, v, t, c), 0-indexed vertices, graph connected.
    Returns (best_product, sum_t, sum_c, tree_edge_indices)."""
    best = {"prod": None, "dir": None}          # running best product + winning direction

    def probe_dir(a, b):
        # one MST extreme in direction (a, b): blend the weight w_e = a*t_e + b*c_e,
        # evaluate the real objective X*Y at the returned hull vertex, update best.
        X, Y, chosen = mst(n, edges, lambda t, c: a * t + b * c)
        prod = X * Y                            # the true objective at this hull vertex
        if best["prod"] is None or prod < best["prod"]:
            best["prod"] = prod
            best["dir"] = (a, b)
        return (X, Y, chosen)

    def below(s, e, m):
        # cross((s - e), (m - e)) > 0  <=>  m is on the lower-left side of segment s->e
        return (s[0]-e[0]) * (m[1]-e[1]) - (s[1]-e[1]) * (m[0]-e[0]) > 0

    def hull(s, e):
        # probe the normal of segment s->e pointing down-left (both components >= 0)
        a, b = s[1] - e[1], e[0] - s[0]
        if a == 0 and b == 0:
            return
        mx, my, _ = probe_dir(a, b)
        m = (mx, my)
        if below(s, e, m):                      # found a new lower-left hull vertex
            hull(s, m)
            hull(m, e)

    sx, sy, _ = probe_dir(1, 0)                 # leftmost vertex: minimize total time
    ex, ey, _ = probe_dir(0, 1)                 # bottommost vertex: minimize total cost
    hull((sx, sy), (ex, ey))

    a, b = best["dir"]                           # rebuild the tree under the winning direction
    X, Y, tree = mst(n, edges, lambda t, c: a * t + b * c)
    return best["prod"], X, Y, tree


if __name__ == "__main__":
    edges = [(0,1,1,5), (1,2,2,3), (2,3,4,1), (0,3,3,2), (0,2,5,5), (1,3,2,2)]
    prod, st, sc, tree = solve(4, edges)
    print(prod, st, sc)                              # 45 9 5
    for i in tree:
        u, v, t, c = edges[i]
        print(u, v)
```

## Complexity

- Each `mst` call: $O(E\log E)$ (edge sort + union-find with path compression).
- Number of `mst` calls $= O(H)$, where $H$ is the number of lower-left convex-hull vertices. With coordinates bounded by $X,Y \in [V-1,\ (V-1)M]$ ($M = \max$ single weight $= 255$), $H = O(VM)$ in the worst case, and far smaller in practice.
- Total: $O(E\log E \cdot VM)$ worst case; typically only a handful of MSTs.
- Memory: $O(V + E)$.
