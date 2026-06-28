The problem asks for a spanning tree of an undirected graph where every edge has a time and a cost, and the goal is to minimize the product of the total time and total cost. A natural first attempt is to run a standard minimum spanning tree algorithm such as Kruskal or Prim, because spanning-tree problems so often reduce to additive edge weights. That does not work here. MST algorithms minimize the sum of fixed per-edge weights, which relies on the cut and cycle properties. When the objective is the product of two sums, swapping one edge for another changes the value in a way that depends on the current totals of the whole tree, not just on the two edge weights. There is no fixed per-edge weight that makes Kruskal correct, so a plain MST on either time or cost only optimizes one factor while ignoring the other.

The right way to look at the problem is to forget the tree as a combinatorial object and focus on the pair of numbers it produces: total time X and total cost Y. Every spanning tree maps to a point (X, Y) in the positive quadrant, and we want the point with the smallest product X times Y. The level sets of this objective are hyperbolas, and in the positive quadrant they are convex toward the origin. That means the minimizing point must lie on the lower-left frontier of the set of achievable points. More specifically, it must be a vertex of the lower-left convex hull of that set. If a point were dominated by another achievable point with smaller or equal X and smaller or equal Y, its product would be larger. Even if the optimum lies on a flat hull edge, the product is concave along that edge, so one of the endpoints is at least as good. Therefore we only need to enumerate the vertices of the lower-left convex hull.

The method I propose is the Balkan OI 2011 Time Is Money algorithm, also known as the minimum-product spanning tree via convex-hull probing. It is built on a simple but powerful observation: minimizing a linear functional a X + b Y over spanning trees is exactly an MST problem. Expanding the expression gives a sum over edges of a times time plus b times cost, so the per-edge weight is just a linear blend of the two original weights. For non-negative a and b, running Kruskal on the blended weights returns the achievable point that is extreme in direction (a, b). Each such MST call gives one lower-left hull vertex. So instead of enumerating the exponentially many trees, we fish out hull vertices one by one using ordinary MST computations.

To find all relevant hull vertices, start with the two axis-extreme points. One MST with direction (1, 0) minimizes total time and gives the leftmost point. Another MST with direction (0, 1) minimizes total cost and gives the bottommost point. Every lower-left hull vertex lies between these two extremes. For any segment joining two known hull vertices, take the normal direction pointing down and to the left, which is (sY - eY, eX - sX) for segment endpoints s and e. Run an MST with that blended weight. The returned point is the one that bulges furthest below the segment. If it is strictly below the segment, it is a new hull vertex; recurse separately on the two sub-segments. If it lies on the segment, the hull there is flat and the recursion stops. Because each genuine split discovers a new hull vertex, the process terminates after O(H) MST calls, where H is the number of hull vertices. With the given weight bounds, H is bounded by O(V times max weight), but in practice it is very small.

During every MST call, including the initial extremes and each recursive probe, I evaluate the true objective X times Y and keep the best value seen along with the direction that produced it. Once the hull search finishes, I run one final MST using the stored best direction to reconstruct an actual tree that attains the minimum product. That tree is the answer.

Concretely, this is a single-file C++17 program that reads the graph from stdin and writes the answer to stdout. The input is a first line `V E`, then `E` lines `u v t c` giving each edge's 0-indexed endpoints, time, and cost. The output is a first line `X Y` with the minimum-product tree's total time and total cost (so `X*Y` is the minimal achievable product), followed by the `V-1` tree edges, one `u v` per line. Edge sums fit in 32-bit integers, but the blended weights and the product use `long long` to stay safe.

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
