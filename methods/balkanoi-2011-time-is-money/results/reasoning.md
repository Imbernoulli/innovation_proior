OK, let me think about this from scratch. I have a connected graph, every edge carries two numbers — a time $t_e$ and a cost $c_e$, both positive. I have to pick a spanning tree, and the thing I'm scored on is $\big(\sum t_e\big)\cdot\big(\sum c_e\big)$. Minimize that.

My first instinct is the obvious one: it's a spanning tree, minimize *some* edge weight, that's a minimum spanning tree, I have Kruskal and Prim in my pocket, done in $O(E\log E)$. So let me just try to bend the objective into a sum and run an MST.

But it won't bend. An MST minimizes $\sum_{e\in T} w_e$ for a *fixed per-edge weight* $w_e$. The reason Kruskal and Prim work at all is additivity: the value of a tree is the sum of independent per-edge contributions, so the cut property and the cycle property hold — for any cut, the lightest crossing edge is safe; for any cycle, the heaviest edge is useless. Both arguments are exchange arguments that only make sense when swapping one edge changes the total by exactly the difference of two edge weights.

Here the value is $\big(\sum t_e\big)\big(\sum c_e\big)$, a *product of two sums*. If I swap edge $f$ out for edge $g$, the change in value is

$$(\Sigma t - t_f + t_g)(\Sigma c - c_f + c_g) - \Sigma t\,\Sigma c,$$

and that depends on the current totals $\Sigma t$ and $\Sigma c$ of the *whole tree*, not just on $f$ and $g$. The marginal "weight" of an edge depends on what else is in the tree. So there's no fixed $w_e$, the cut property is dead, and a plain MST is simply the wrong tool. I can't greedily build this.

So what do I actually have? A spanning tree $T$ doesn't really matter to the objective as a *tree* — all the objective sees is the pair of numbers $\big(\sum t_e, \sum c_e\big)$. Two completely different trees with the same total time and total cost are interchangeable to me. That's a strong hint to stop thinking about trees as combinatorial objects and start thinking about the *summary statistics* they produce.

Let me make that literal. To each spanning tree $T$ associate the point

$$P(T) = \Big(\sum_{e\in T} t_e,\ \sum_{e\in T} c_e\Big) = (X, Y)$$

in the plane. There are exponentially many trees, so there are (up to coincidences) lots of points, all sitting in the positive quadrant since all weights are positive: $X \ge V-1$, $Y \ge V-1$. Call this finite set of achievable points $S$. My job is just: over all points $(X,Y) \in S$, minimize $X\cdot Y$.

Now this is a cleaner question. Forget trees for a second — I have a cloud of points and I want the one minimizing the product of its coordinates. What do the level sets of $X\cdot Y$ look like? They're hyperbolas $XY = k$. In the positive quadrant each one is a curve bowing away from the origin, convex toward the origin, and as $k$ shrinks the hyperbola slides down toward the corner. Minimizing $XY$ means: find the smallest $k$ such that the hyperbola $XY=k$ still touches my point cloud. I want to push that hyperbola as far down-and-left as I can until it just kisses a point of $S$.

Picture that. The hyperbola is convex (bulging toward the origin). I'm sliding it down until it first touches $S$. Where can the first touch happen? It has to happen at a point of $S$ that's on the *lower-left frontier* of the cloud — the part facing the origin. A point that's strictly above-and-to-the-right of some other achievable point can never be the minimizer: if there's a tree with $X' \le X$ and $Y' \le Y$ and at least one strict, then $X'Y' < XY$ (positive coordinates), so that dominated point is beaten. The minimizer must be Pareto-minimal — nothing dominates it.

But I can say something sharper than "Pareto-minimal." Suppose the best point is $p=(X_*,Y_*)$ with value $k=X_*Y_*$. The branch $Y=k/X$ is convex in the positive quadrant. Its tangent at $p$ has equation $Y_*X+X_*Y=2k$, and every achievable point $q=(X,Y)$ has $XY\ge k$, so by AM-GM, $Y_*X+X_*Y \ge 2\sqrt{X_*Y_*XY}\ge 2k$. All achievable points lie on or above that tangent half-plane, while $p$ lies on the boundary. So $p$ is a support point of the lower-left convex hull. If the minimizing point sits in the middle of a flat hull edge, the edge has $X$ increasing and $Y$ decreasing, and $XY$ along that segment is a concave quadratic, so one endpoint has product no larger. I can therefore choose an optimum that is a *vertex of the lower-left convex hull* of $S$. That's the whole reduction: I don't need to look at all of $S$, just at the hull vertices, and there should be far fewer of those.

Good, but two problems. (1) $S$ is exponentially large — I can't materialize it to take a hull. (2) Even if I could, I need a way to *generate* hull vertices, because I can only access $S$ through "build me a tree," not by listing it.

Let me attack (2) first, because it might rescue (1). A vertex of a convex hull has a defining property: it's the unique point that's extreme in some direction. Pick a direction $(a,b)$ and ask for the point of $S$ minimizing the linear functional

$$a\cdot X + b\cdot Y.$$

For $a,b \ge 0$ this support-function minimization picks out a lower-left hull vertex — the one you'd touch by sweeping a line with normal $(a,b)$ in from the lower-left. So *if* I can minimize $aX + bY$ over $S$ for any chosen $(a,b)$, I can fish out hull vertices one at a time by choosing directions.

And now — can I minimize $aX + bY$ over achievable points? Write it out:

$$a\,X + b\,Y = a\sum_{e\in T} t_e + b\sum_{e\in T} c_e = \sum_{e\in T}\big(a\,t_e + b\,c_e\big).$$

Stare at that. It collapsed back into a *single sum over the edges of $T$*, with per-edge weight $w_e = a\,t_e + b\,c_e$. And $a,b \ge 0$ with $t,c$ positive makes $w_e \ge 0$. Minimizing a sum of fixed per-edge weights over spanning trees — that's an MST. The exact tool I threw away at the start works perfectly here, just not on the raw weights: I run Kruskal on the *blended* weight $a t_e + b c_e$, and the resulting MST is precisely the achievable point that's extreme in direction $(a,b)$. One MST call = one hull vertex. That dissolves problem (1) too: I never build $S$; each $O(E\log E)$ MST hands me a hull vertex on demand.

So the shape of the algorithm is: enumerate the lower-left hull vertices by probing directions, evaluate $XY$ at each, keep the best. The only thing left is *which directions to probe*. The hull vertices correspond to a range of normal directions sweeping from "straight left" (minimize $X$ alone) to "straight down" (minimize $Y$ alone). I obviously can't try every real direction — that's a continuum.

Let me think about the two ends first. Direction $(1,0)$: minimize $X = \sum t_e$ — that's just an ordinary MST on the times, and it gives me the leftmost hull vertex, call it $s$ (smallest total time). Direction $(0,1)$: minimize $Y = \sum c_e$ — MST on the costs, the bottommost vertex $e$ (smallest total cost). Every hull vertex I care about lives on the chain between $s$ and $e$.

Now, between two known adjacent-ish hull vertices, how do I find whether there's another one between them, and which direction reveals it? Suppose I have a segment from $s$ (up and to the left) to $e$ (down and to the right). The genuine lower-left hull either *is* the straight segment $se$, or it dips below it with one or more vertices. The vertex that dips the *most* below segment $se$ is the extreme point in the direction perpendicular to $se$ pointing away from the origin — down and to the left. If I probe that normal direction with an MST and the returned point lands strictly below the line through $s$ and $e$, I've found a real new hull vertex $m$ between them; then the hull on $[s,m]$ and on $[m,e]$ might dip further still, so I recurse into both halves. If instead the MST in that direction returns a point *on* the segment $se$ (nothing strictly below), then there is no hull vertex between $s$ and $e$ — the hull there is exactly the edge $se$, and I stop. This is a divide-and-conquer over the angular range, splitting at the extreme point each time, the way you'd refine a curve by repeatedly sampling the point of maximum deviation.

I should pin down the normal direction and the test concretely, signs included, because getting them backwards silently breaks everything. Going from $s$ to $e$ along the lower hull, $X$ increases and $Y$ decreases, so the segment vector is $e - s = (e_X - s_X,\ e_Y - s_Y)$ with $e_X \ge s_X$ and $e_Y \le s_Y$. I want the normal pointing down-left (toward the origin), so that minimizing the functional finds the point that bulges furthest below the segment. Take the direction

$$(a,b) = \big(s_Y - e_Y,\ e_X - s_X\big).$$

Both components are $\ge 0$: $s_Y - e_Y \ge 0$ and $e_X - s_X \ge 0$ — good, so $w_e = a t_e + b c_e \ge 0$ and Kruskal is legitimate. Run the MST under this $(a,b)$ to get a candidate point $m$. Now I need "is $m$ strictly below segment $se$?" That's an orientation test. Using the signed cross product, $m$ is on the lower-left side of segment $se$ exactly when

$$\big(s_X - e_X\big)\big(m_Y - e_Y\big) - \big(s_Y - e_Y\big)\big(m_X - e_X\big) > 0,$$

i.e. the cross product of $(s - e)$ and $(m - e)$ is positive. If that holds, $m$ is a new hull vertex, recurse on $(s,m)$ and $(m,e)$; otherwise $m$ lies on segment $se$ and I'm done with this interval. (When I run the MST I also just evaluate $X\cdot Y$ at every point it hands me and keep the running minimum — including at $s$ and $e$ themselves — so I never miss a candidate.)

Why does this find *every* relevant hull vertex and not get stuck? Each lower-left hull vertex is the unique maximizer of *some* support direction in the first quadrant; the recursion, by always probing the exact perpendicular of the current segment, tests precisely the direction that would reveal a vertex between the two current endpoints. If the interval truly has a vertex between $s$ and $e$, the perpendicular probe finds the deepest one (it's extreme in that normal direction), splits there, and recurses; if it has none, the probe returns a point on the segment and the interval is certified flat. By induction on the number of vertices in an interval, every hull vertex between the two extreme points $s$ and $e$ is eventually produced. And it terminates: every recursive split consumes one genuinely new hull vertex, the lower-left hull has finitely many vertices, so the recursion tree is finite. (How many? The coordinates are bounded — $X,Y$ range in $[V-1,\ (V-1)M]$ with $M$ the largest single weight — so the hull can have at most on the order of $V\cdot M$ vertices, and in practice far fewer, which is why this runs fast.)

Let me also double check that the *answer* I want is genuinely among these vertices and not lost on a hull edge. I argued the minimizer of $XY$ is a hull vertex; the recursion produces exactly the hull vertices; and I evaluate $XY$ at each one as I go. So the global minimum of the product is the smallest $XY$ over the vertices I enumerate. The one subtlety is that the *minimizing tree* might not be unique and I only need *a* minimizer — fine, I keep the direction $(a,b)$ that achieved the best product, and at the very end I rerun Kruskal once under that direction to reconstruct and print the actual tree edges. The MST under the optimal direction realizes the optimal point, so its edge set is a valid answer.

Let me write it. The MST step is the same primitive every time — sort the edges by some per-edge weight, run Kruskal with a union-find, hand back the resulting point $(\sum t, \sum c)$ and its edge set — so I'll pull it out as a free-standing `mst(n, edges, weight_fn)` that takes the per-edge weight as a function and returns $(\text{sum\_t}, \text{sum\_c}, \text{chosen})$. Then probing direction $(a,b)$ is just calling it with `weight_fn = lambda t, c: a*t + b*c`. The hull search lives in `solve(n, edges)`, which keeps the running best (product, and the direction that achieved it), gets the two axis-extreme vertices with directions $(1,0)$ and $(0,1)$, recurses between them — `below(s,e,m)` is the sign of the cross product, `probe(s,e)` shoots the perpendicular and recurses if the midpoint is strictly below — and finally reconstructs the winning tree.

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
```

So the whole thing comes down to: the product objective only sees the pair $(\sum t, \sum c)$, so each tree is a point in the plane; the minimizer of $XY$ sits at a vertex of the lower-left convex hull of those points; a hull vertex is the extreme point in some direction $(a,b)$; and minimizing $aX+bY$ collapses to $\sum (a t_e + b c_e)$, a plain MST — so I can fish out hull vertices with Kruskal; and I find all the ones that matter by starting from the two axis-extreme trees and recursively probing the perpendicular of each segment, splitting at whatever point bulges below it until nothing does.
