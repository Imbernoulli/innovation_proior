The problem asks for a spanning tree of an undirected graph where every edge has a time and a cost, and the goal is to minimize the product of the total time and total cost. A natural first attempt is to run a standard minimum spanning tree algorithm such as Kruskal or Prim, because spanning-tree problems so often reduce to additive edge weights. That does not work here. MST algorithms minimize the sum of fixed per-edge weights, which relies on the cut and cycle properties. When the objective is the product of two sums, swapping one edge for another changes the value in a way that depends on the current totals of the whole tree, not just on the two edge weights. There is no fixed per-edge weight that makes Kruskal correct, so a plain MST on either time or cost only optimizes one factor while ignoring the other.

The right way to look at the problem is to forget the tree as a combinatorial object and focus on the pair of numbers it produces: total time X and total cost Y. Every spanning tree maps to a point (X, Y) in the positive quadrant, and we want the point with the smallest product X times Y. The level sets of this objective are hyperbolas, and in the positive quadrant they are convex toward the origin. That means the minimizing point must lie on the lower-left frontier of the set of achievable points. More specifically, it must be a vertex of the lower-left convex hull of that set. If a point were dominated by another achievable point with smaller or equal X and smaller or equal Y, its product would be larger. Even if the optimum lies on a flat hull edge, the product is concave along that edge, so one of the endpoints is at least as good. Therefore we only need to enumerate the vertices of the lower-left convex hull.

The method I propose is the Balkan OI 2011 Time Is Money algorithm, also known as the minimum-product spanning tree via convex-hull probing. It is built on a simple but powerful observation: minimizing a linear functional a X + b Y over spanning trees is exactly an MST problem. Expanding the expression gives a sum over edges of a times time plus b times cost, so the per-edge weight is just a linear blend of the two original weights. For non-negative a and b, running Kruskal on the blended weights returns the achievable point that is extreme in direction (a, b). Each such MST call gives one lower-left hull vertex. So instead of enumerating the exponentially many trees, we fish out hull vertices one by one using ordinary MST computations.

To find all relevant hull vertices, start with the two axis-extreme points. One MST with direction (1, 0) minimizes total time and gives the leftmost point. Another MST with direction (0, 1) minimizes total cost and gives the bottommost point. Every lower-left hull vertex lies between these two extremes. For any segment joining two known hull vertices, take the normal direction pointing down and to the left, which is (sY - eY, eX - sX) for segment endpoints s and e. Run an MST with that blended weight. The returned point is the one that bulges furthest below the segment. If it is strictly below the segment, it is a new hull vertex; recurse separately on the two sub-segments. If it lies on the segment, the hull there is flat and the recursion stops. Because each genuine split discovers a new hull vertex, the process terminates after O(H) MST calls, where H is the number of hull vertices. With the given weight bounds, H is bounded by O(V times max weight), but in practice it is very small.

During every MST call, including the initial extremes and each recursive probe, I evaluate the true objective X times Y and keep the best value seen along with the direction that produced it. Once the hull search finishes, I run one final MST using the stored best direction to reconstruct an actual tree that attains the minimum product. That tree is returned as the answer.

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
            pa[x] = pa[pa[x]]
            x = pa[x]
        return x
    sum_t = sum_c = 0
    chosen = []
    for i in order:
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
    best = {"prod": None, "dir": None}

    def probe_dir(a, b):
        X, Y, chosen = mst(n, edges, lambda t, c: a * t + b * c)
        prod = X * Y
        if best["prod"] is None or prod < best["prod"]:
            best["prod"] = prod
            best["dir"] = (a, b)
        return (X, Y, chosen)

    def below(s, e, m):
        return (s[0]-e[0]) * (m[1]-e[1]) - (s[1]-e[1]) * (m[0]-e[0]) > 0

    def hull(s, e):
        a, b = s[1] - e[1], e[0] - s[0]
        if a == 0 and b == 0:
            return
        mx, my, _ = probe_dir(a, b)
        m = (mx, my)
        if below(s, e, m):
            hull(s, m)
            hull(m, e)

    sx, sy, _ = probe_dir(1, 0)
    ex, ey, _ = probe_dir(0, 1)
    hull((sx, sy), (ex, ey))

    a, b = best["dir"]
    X, Y, tree = mst(n, edges, lambda t, c: a * t + b * c)
    return best["prod"], X, Y, tree
```
