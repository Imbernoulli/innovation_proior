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

Single-file C++17, reading from stdin and writing to stdout. Input: first line `V E`; then `E` lines `u v t c` (0-indexed endpoints, edge time, edge cost). Output: first line the minimum-product tree's total time and total cost `X Y` (so `X*Y` is the minimal achievable product), then the `V-1` tree edges, one `u v` per line.

```cpp
// Time is Money (Balkan OI 2011, balkan11_timeismoney).
// Reads from stdin: first line "V E"; then E lines "u v t c" (0-indexed
// endpoints, edge time t, edge cost c). Writes to stdout: first line the
// minimum-product tree's total time and total cost "X Y" (product X*Y is
// minimized over spanning trees), then the V-1 tree edges, one "u v" per line.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Edge { int u, v, t, c; };

int n, m;
vector<Edge> edges;
vector<int> parent;

int find_set(int x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}

ll best_prod;            // smallest X*Y seen so far
int best_a, best_b;      // direction (a,b) that achieved it

// Kruskal MST minimizing the linear blend w_e = a*t_e + b*c_e over spanning
// trees. Returns (X, Y) = (sum of times, sum of costs) of the resulting tree.
// If print is true, also emits the chosen edges' endpoints to stdout.
pair<int,int> getmst(ll a, ll b, bool print) {
    vector<int> order(m);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int i, int j) {
        ll wi = a * edges[i].t + b * edges[i].c;
        ll wj = a * edges[j].t + b * edges[j].c;
        return wi < wj;
    });
    parent.resize(n);
    iota(parent.begin(), parent.end(), 0);
    ll X = 0, Y = 0;
    for (int idx : order) {
        int ru = find_set(edges[idx].u), rv = find_set(edges[idx].v);
        if (ru != rv) {
            parent[ru] = rv;
            X += edges[idx].t;
            Y += edges[idx].c;
            if (print) printf("%d %d\n", edges[idx].u, edges[idx].v);
        }
    }
    if (!print) {
        ll prod = (ll)X * (ll)Y;
        if (prod < best_prod) { best_prod = prod; best_a = (int)a; best_b = (int)b; }
    }
    return {(int)X, (int)Y};
}

// cross((s - e), (m - e)) > 0  <=>  m is strictly on the lower-left side of
// segment s->e, i.e. a genuine new lower-left hull vertex.
bool below(pair<int,int> s, pair<int,int> e, pair<int,int> mid) {
    ll cross = (ll)(s.first - e.first) * (mid.second - e.second)
             - (ll)(s.second - e.second) * (mid.first - e.first);
    return cross > 0;
}

// Probe the down-left normal of segment s->e; recurse on each half if the
// returned point bulges strictly below the segment.
void hull(pair<int,int> s, pair<int,int> e) {
    int a = s.second - e.second;   // both >= 0 along the lower hull
    int b = e.first - s.first;
    if (a == 0 && b == 0) return;
    pair<int,int> mid = getmst(a, b, false);
    if (below(s, e, mid)) {
        hull(s, mid);
        hull(mid, e);
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    edges.resize(m);
    for (int i = 0; i < m; i++)
        scanf("%d %d %d %d", &edges[i].u, &edges[i].v, &edges[i].t, &edges[i].c);

    best_prod = LLONG_MAX;
    best_a = 1; best_b = 0;

    pair<int,int> s = getmst(1, 0, false);   // leftmost: minimize total time
    pair<int,int> e = getmst(0, 1, false);   // bottommost: minimize total cost
    hull(s, e);

    pair<int,int> ans = getmst(best_a, best_b, false);   // realize the winning point
    printf("%d %d\n", ans.first, ans.second);
    getmst(best_a, best_b, true);            // reprint the winning tree's edges
    return 0;
}
```

## Complexity

- Each `getmst` call: $O(E\log E)$ (edge sort + union-find with path compression).
- Number of `getmst` calls $= O(H)$, where $H$ is the number of lower-left convex-hull vertices. With coordinates bounded by $X,Y \in [V-1,\ (V-1)M]$ ($M = \max$ single weight $= 255$), $H = O(VM)$ in the worst case, and far smaller in practice.
- Total: $O(E\log E \cdot VM)$ worst case; typically only a handful of MSTs. Sums fit comfortably in 32-bit ($X, Y \le 199 \cdot 255 < 51000$), but the product and the blended weights use `long long` to be safe.
- Memory: $O(V + E)$.
