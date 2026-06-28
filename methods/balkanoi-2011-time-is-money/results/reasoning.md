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

But I can say something sharper than "Pareto-minimal." Suppose the best point is $p=(X_*,Y_*)$ with value $k=X_*Y_*$. The branch $Y=k/X$ is convex in the positive quadrant. Its tangent at $p$ has equation $Y_*X+X_*Y=2k$, and every achievable point $q=(X,Y)$ has $XY\ge k$, so by AM-GM, $Y_*X+X_*Y \ge 2\sqrt{X_*Y_*XY}\ge 2k$. All achievable points lie on or above that tangent half-plane, while $p$ lies on the boundary. So $p$ is a support point of the lower-left convex hull. If the minimizing point sits in the middle of a flat hull edge, the edge has $X$ increasing and $Y$ decreasing, and $XY$ along that segment is a concave quadratic, so one endpoint has product no larger. So I can choose an optimum that is a *vertex of the lower-left convex hull* of $S$, which means I do not need to look at all of $S$, just at the hull vertices — and there should be far fewer of those. I will want to sanity-check this concavity claim on a real cloud of points later, because "the optimum is a hull vertex" is precisely the kind of statement that is easy to assert and easy to be wrong about.

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

I do not trust myself on the sign of that cross product — it is exactly the kind of thing I get backwards — so let me actually push numbers through it on a small graph before I believe it. Take 4 vertices and the edges (as $(u,v,t,c)$): $(0,1,1,5),\ (1,2,2,3),\ (2,3,4,1),\ (0,3,3,2),\ (0,2,5,5),\ (1,3,2,2)$. First the two axis probes. Minimizing total time $X$: the cheapest-time spanning tree picks the small $t$'s — edge $0$ ($t=1$), edge $1$ ($t=2$), edge $5$ ($t=2$), connecting $\{0,1\},\{1,2\},\{1,3\}$, which spans all four vertices; that gives $X = 1+2+2 = 5$ and $Y = 5+3+2 = 10$. So $s = (5,10)$, product $50$. Minimizing total cost $Y$: take the small $c$'s — edge $2$ ($c=1$), edge $3$ ($c=2$), edge $5$ ($c=2$); vertices $\{2,3\},\{0,3\},\{1,3\}$ span everything, $X = 4+3+2 = 9$, $Y = 1+2+2 = 5$. So $e = (9,10)$? — no: $Y = 5$, so $e = (9,5)$, product $45$. Already $e$ beats $s$.

Now the perpendicular probe between $s=(5,10)$ and $e=(9,5)$. The direction is $(a,b) = (s_Y - e_Y,\ e_X - s_X) = (10-5,\ 9-5) = (5,4)$, both nonnegative — the sign came out right. Blended weight $w_e = 5t_e + 4c_e$ per edge: edge $0 \to 5{\cdot}1+4{\cdot}5 = 25$, edge $1 \to 22$, edge $2 \to 24$, edge $3 \to 23$, edge $4 \to 45$, edge $5 \to 18$. Kruskal in increasing $w$: take edge $5$ ($w=18$, joins $1,3$), edge $1$ ($w=22$, joins $2$), edge $3$ ($w=23$, joins $0$) — three edges, spanning; that is $X = 2+2+3 = 7$, $Y = 2+3+2 = 7$, so $m = (7,7)$, product $49$. Is $m$ strictly below segment $se$? The straight line through $(5,10)$ and $(9,5)$ has slope $-5/4$, so at $X=7$ it sits at $Y = 10 - \tfrac54(7-5) = 7.5$; my point has $Y=7 < 7.5$, so it genuinely dips below. And the cross product: $(s_X-e_X)(m_Y-e_Y) - (s_Y-e_Y)(m_X-e_X) = (5-9)(7-5) - (10-5)(7-9) = (-4)(2) - (5)(-2) = -8 + 10 = +2 > 0$. Positive, matching the "below the line" verdict from the slope check — so the sign convention is right, and $(7,7)$ is a real third hull vertex. (As a control, a point *above* the line, say $(7,8)$, gives cross $= (-4)(3)-(5)(-2) = -12+10 = -2 < 0$ — correctly rejected.) The recursion now splits at $(7,7)$ and probes $[s,m]$ and $[m,e]$; both of those return points already seen (no new dip), so the hull there is flat and it stops. Over the three vertices $\{(5,10),(9,5),(7,7)\}$ the products are $\{50,45,49\}$, smallest $45$ at $(9,5)$.

That is the algorithm's answer, but is $45$ actually the global optimum, or did the hull argument quietly drop a better tree? A 4-vertex graph with 6 edges has few enough spanning trees to just enumerate all $\binom{6}{3}$ triples, keep the ones that span, and read off $\min XY$ directly. Doing that, there are $16$ spanning trees; their distinct $(X,Y)$ points are $(5,10),(6,10),(7,7),(7,8),(7,9),(8,8),(8,12),(9,5),(9,6),(9,10),(9,12),(10,9),(10,10),(10,11),(11,8),(11,9)$, and the smallest product over *all* of them is indeed $45$, at $(9,5)$. So the convex-hull shortcut found the true optimum while only ever touching three of the sixteen points — the brute force agrees with it exactly. Reassuring, and it also shows the cheaper failure mode that worried me — the optimum hiding strictly inside the cloud — does not happen: $(9,5)$ is a genuine hull vertex.

That last example is a little too easy on the method, though, because the optimum turned out to be one of the two *axis* extremes ($e$, the min-cost tree). If the answer were always an axis extreme I would not need the hull recursion at all — two MSTs would do. So I want to see a case where the optimum is a *non-axis* hull vertex, to confirm the recursive probing is actually pulling its weight. Searching small random 5-vertex graphs for one where neither the min-time nor the min-cost tree is optimal turns one up: edges $(0,1,5,2),(0,2,2,6),(2,3,5,8),(1,4,9,1),(1,4,6,3),(4,0,9,5),(0,2,9,6),(1,2,4,9)$. Here the min-time tree is $(17,26)$ (product $442$) and the min-cost tree is $(21,17)$ (product $357$); but the perpendicular probe between them, direction $(26-17,\,21-17)=(9,4)$, returns $(18,19)$ with product $342$ — strictly below both endpoints. Brute force over all spanning trees of this graph confirms $342$ at $(18,19)$ is the global minimum. So when the optimum lies between the axis extremes, the recursion is exactly what surfaces it; the two MSTs alone would have reported $357$ and been wrong by a real margin.

With the sign convention pinned down and the method validated on both an axis-optimum and an interior-optimum example, I can ask why it should find *every* relevant hull vertex in general. Each lower-left hull vertex is the unique maximizer of *some* support direction in the first quadrant; the recursion, by always probing the exact perpendicular of the current segment, tests precisely the direction that would reveal a vertex between the two current endpoints. If the interval truly has a vertex between $s$ and $e$, the perpendicular probe finds the deepest one (it is extreme in that normal direction), splits there, and recurses; if it has none, the probe returns a point on the segment and the interval is certified flat. By induction on the number of hull vertices strictly inside an interval — the deepest one is split off, and each half has strictly fewer — every hull vertex between the two extreme points $s$ and $e$ is produced. Termination follows from the same induction: every recursive split consumes one genuinely new hull vertex, and the lower-left hull has finitely many, so the recursion tree is finite. (How many? The coordinates are bounded — $X,Y$ range in $[V-1,\ (V-1)M]$ with $M$ the largest single weight — so the hull can have at most on the order of $V\cdot M$ vertices, and in practice far fewer, which is why this runs fast; in the two examples above it was 3 and 3.)

That settles where the answer lives: the minimizer of $XY$ is a hull vertex (the AM-GM/concavity argument above), the recursion produces exactly the hull vertices, and I evaluate $XY$ at each as I go, so the global minimum of the product is the smallest $XY$ over the vertices I enumerate — which is what the brute-force checks just confirmed on two graphs. The one remaining subtlety is that the *minimizing tree* need not be unique and I only need *a* minimizer: I keep the direction $(a,b)$ that achieved the best product, and at the very end rerun Kruskal once under that direction to reconstruct and print the actual tree edges. The MST under the winning direction realizes the winning point, so its edge set is a valid answer.

Let me write it as a single self-contained C++17 program that reads the graph from stdin and prints the answer to stdout, since that is the form the judge scores. The MST step is the same primitive every time — sort the edges by some per-edge weight, run Kruskal with a union-find, hand back the resulting point $(\sum t, \sum c)$ — so I'll pull it out as a free-standing `getmst(a, b, print)` that blends the weight as $w_e = a\,t_e + b\,c_e$, returns the point $(X, Y)$, and (when not printing) folds $X\cdot Y$ into a global running best together with the direction that achieved it. The hull search is the recursive `hull(s, e)`: it forms the down-left normal $(a,b) = (s_Y - e_Y,\ e_X - s_X)$, probes it with one `getmst`, and recurses on both halves when the returned point is strictly below the segment — `below(s,e,m)` being the sign of the cross product. `main` reads $V, E$ and the edges, gets the two axis-extreme vertices with directions $(1,0)$ and $(0,1)$, drives the recursion between them, then reruns `getmst` under the winning direction to print first the point $(X, Y)$ and then the tree's edges. The input is `V E` followed by $E$ lines `u v t c`; the output is `X Y` (the minimum-product tree's total time and total cost) followed by the $V-1$ tree edges as `u v`. Sums fit in 32-bit, but the blend and the product use `long long` to stay safe.

The recursive lower-hull probe -- especially the perpendicular blend `(s_Y - e_Y, e_X - s_X)` and the `below` cross-product sign -- is the part I'd most easily get wrong under time pressure; if I weren't confident in that hull recursion, I'd fall back to a slower exhaustive spanning-tree check that I could prove end to end, because a plain correct submission beats an ambitious broken one.

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

Looking back at the chain: the product objective only sees the pair $(\sum t, \sum c)$, so each tree is a point in the plane; the minimizer of $XY$ sits at a vertex of the lower-left convex hull of those points; a hull vertex is the extreme point in some direction $(a,b)$; and minimizing $aX+bY$ collapses to $\sum (a t_e + b c_e)$, a plain MST — so I can fish out hull vertices with Kruskal; and I find the ones that matter by starting from the two axis-extreme trees and recursively probing the perpendicular of each segment, splitting at whatever point bulges below it until nothing does. The two graphs I worked by hand and by brute force both came out matching the hull search ($45$ at an axis extreme, $342$ at an interior vertex the two axis MSTs would have missed), which is the evidence that makes me trust the whole reduction rather than just hope the picture is right.
