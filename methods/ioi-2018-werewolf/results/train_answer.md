We are handed an undirected graph on $N$ vertices, numbered $0$ through $N-1$, with $M$ edges, and a pile of $Q$ queries. Each query $(S,E,L,R)$ with $L\le R$ asks a yes/no question about a walk from $S$ to $E$ that runs in two phases: at the start we may only step on vertices whose index is at least $L$, then at exactly one moment we flip a switch at some vertex, and from that vertex onward we may only step on vertices whose index is at most $R$. The switch vertex itself must be legal for both halves, so it lives in $[L,R]$. The literal way to decide one query is a two-color BFS over a $2N$-state graph — states $(v,0)$ "at $v$, still in phase one" and $(v,1)$ "at $v$, already switched", with intra-phase edges gated by $\ge L$ and $\le R$ and a free switch edge $(v,0)\to(v,1)$ whenever $L\le v\le R$ — and then asking whether $(E,1)$ is reachable from $(S,0)$. That is correct and runs in $O(N+M)$ per query, but with both $N$ and $Q$ up to a few hundred thousand, $O(Q(N+M))$ is hopeless. The queries must share preprocessing instead of each touching the whole graph.

The first move is to strip the two-color trick down to its essence. Phase one is a walk from $S$ that never leaves the set of vertices with index $\ge L$, so the switch vertex must lie in the connected component of $S$ in the subgraph induced by $\{v:v\ge L\}$ — call it $V_L$. Phase two is a walk from the switch vertex to $E$ over vertices $\le R$; since the graph is undirected, that is the same as the component of $E$ in the subgraph induced by $\{v:v\le R\}$ — call it $V_R$. A vertex in $V_L$ already has index $\ge L$ and a vertex in $V_R$ already has index $\le R$, so any vertex in $V_L\cap V_R$ automatically lies in $[L,R]$ and is a legal switch; the explicit switch constraint comes for free. The entire question therefore collapses to whether $V_L$ and $V_R$ share a vertex, with two cheap edge cases: if $S<L$ then $V_L=\varnothing$ and the answer is no, and if $E>R$ then $V_R=\varnothing$ and the answer is no. Recomputing both sets per query is still two BFS floods, $O(N+M)$ each, and the waste is that nearby thresholds produce almost identical sets. We want to compute these reachable sets once, as a structure, and then read each query off it.

I propose to solve this with two Kruskal reconstruction trees feeding a single offline 2D range-intersection sweep. The lever is monotonicity: as $L$ decreases, the subgraph "$\ge L$" only gains vertices and edges, so components only ever merge, never split. That is an incremental-union process, exactly what disjoint-set union is built for. Sweep $L$ from high to low, turning vertices on in decreasing index order; when vertex $w$ turns on, union it with every already-on neighbor (those are precisely the neighbors with larger index, which turned on earlier). The merge history of such a sweep is a laminar family — any two components ever seen are either disjoint or nested, since lowering the threshold only fuses — and a laminar family on $N$ leaves is a tree. So rather than throw the merge history away, build it explicitly. The leaves are the $N$ original vertices, each tagged with its own value. Each time we turn on a vertex $w$, we create one fresh internal node tagged with $w$ and make it the parent of the current component-roots of all already-on neighbors together with the leaf $w$; crucially we mint a new internal node for *every* vertex turned on, even an isolated one with no on-neighbors yet, so that every vertex owns exactly one internal node carrying its threshold value — a uniformity that makes the query climb clean.

Climbing from a leaf to the root, the internal nodes were created later and later in the high-to-low sweep, and "later" means *smaller* tag, so tags strictly decrease upward. For a query $(S,L)$, the set $V_L$ is whatever component $S$ landed in once the threshold had only dropped to $L$, i.e. only vertices with value $\ge L$ had turned on; in the tree that is the subtree under the highest ancestor of $S$ whose tag is still $\ge L$. Because tags decrease upward, we find it by climbing while the parent's tag is $\ge L$ and stopping the instant it would dip below $L$. The decisive payoff is that a subtree's leaves form a *contiguous interval* if we number leaves by a DFS traversal of the tree. So after building the tree we run a DFS, assign each leaf a position $0,1,2,\dots$ in visitation order, and give each node the interval $[\mathrm{lo},\mathrm{hi}]$ of leaf-positions beneath it. The arbitrary scattered set $V_L$ becomes an interval $[a_1,b_1]$ — the easy line-graph picture, recovered on a general graph by relabeling leaves along the tree.

The other side is the mirror image. The constraint there is $\le R$, so the subgraph grows as $R$ increases; we turn vertices on in *increasing* index order, union each with its already-on smaller-numbered neighbors, and build the analogous tree, whose tags increase along the sweep. We climb from $E$ while the parent's tag is $\le R$, and $V_R$ is the subtree we stop at — again a contiguous interval $[a_2,b_2]$, but in *this* tree's DFS leaf order. The snag is that the two trees order the leaves differently, so each vertex $v$ has two positions: $x_v$ in the high tree and $y_v$ in the low tree. $V_L$ is the interval $[a_1,b_1]$ in the $x$-coordinate and $V_R$ is the interval $[a_2,b_2]$ in the $y$-coordinate, living in different coordinate systems, so they cannot be intersected on one line. The resolution is to view each original vertex as a point $(x_v,y_v)$ in the plane. Then $V_L\cap V_R\ne\varnothing$ is exactly the question of whether any vertex-point falls in the axis-aligned rectangle
$$[a_1,b_1]\times[a_2,b_2],$$
a rectangle-nonempty test over $N$ fixed points.

Counting points in axis-aligned rectangles over a static point set is standard offline work. Lay the $N$ points along the $x$-axis — an array whose position $x_v$ holds the value $y_v$ — and sweep $x$ upward, inserting each point's $y$ into a Fenwick tree (BIT) keyed by $y$. A rectangle count is a difference of two "$x$-prefix, $y$-range" counts: the number of points with $x\le b_1$ and $y\in[a_2,b_2]$ minus the number with $x\le a_1-1$ and $y\in[a_2,b_2]$. So for each query we register two events, a $+1$ at $x=b_1$ and (when $a_1>0$) a $-1$ at $x=a_1-1$, each asking the BIT for the count of inserted $y$ in $[a_2,b_2]$. After the sweep, a query's signed total is the number of its vertices in the rectangle; positive means $V_L$ and $V_R$ share a vertex, hence yes. Queries already settled empty by $S<L$ or $E>R$ are simply no. For scale, the upward climb in each tree must not walk parent pointers one edge at a time, since a path-shaped merge history would make a query linear; because the tags along any leaf-to-root path are monotone, we precompute $2^j$-ancestor jump tables (binary lifting) and a jump is valid exactly when the candidate ancestor still has tag $\ge L$ on the lower-bound side or $\le R$ on the upper-bound side, making each component-range lookup logarithmic. The whole pipeline runs in $O\big((N+M)\,\alpha(N)+(N+Q)\log N\big)$ time.

The deliverable is a single self-contained C++17 program. It reads `N M Q`, then
`M` lines `u v` (0-based undirected edges), then `Q` lines `S E L R`, and prints
`YES`/`NO` for each query. Vertex indices fit in `int`; the Fenwick prefix
counts are kept in `long long`.

```cpp
// Reads: N M Q, then M lines "u v" (0-based undirected edges), then Q lines
// "S E L R"; prints for each query "YES" iff a two-phase walk S->E exists
// (vertices >= L until one switch vertex in [L,R], then vertices <= R), else "NO".
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> par;
    DSU(int n) : par(n) { iota(par.begin(), par.end(), 0); }
    int find(int x) {
        int root = x;
        while (par[root] != root) root = par[root];
        while (par[x] != root) { int nx = par[x]; par[x] = root; x = nx; }
        return root;
    }
};

struct Tree {
    vector<int> parent;                 // forest parent (root points to self)
    vector<int> tag;                    // tag[node]: the vertex that created it
    vector<int> left, right;            // contiguous leaf-interval per node
    vector<int> place;                  // leaf position of each original vertex
    vector<vector<int>> jump;           // binary-lifting table over parent
};

// Turn vertices on in `order`; turning on w hangs the components of all already-on
// neighbors (and the leaf w) under a fresh internal node tagged with value w.
// Leaves 0..n-1 keep their own value as tag.
static Tree build_structure(int n, const vector<vector<int>>& adj,
                            const vector<int>& order) {
    int cap = 2 * n;
    vector<int> parent(cap), tag(cap);
    iota(parent.begin(), parent.end(), 0);
    iota(tag.begin(), tag.end(), 0);
    vector<vector<int>> children(cap);
    DSU dsu(cap);
    vector<char> added(n, 0);
    int nxt = n;
    for (int w : order) {
        added[w] = 1;
        vector<int> roots;
        int r0 = dsu.find(w);
        roots.push_back(r0);
        unordered_set<int> seen;
        seen.insert(r0);
        for (int u : adj[w]) {
            if (added[u]) {
                int r = dsu.find(u);
                if (!seen.count(r)) { seen.insert(r); roots.push_back(r); }
            }
        }
        int node = nxt++;
        tag[node] = w;
        for (int r : roots) { parent[r] = node; dsu.par[r] = node; }
        children[node] = roots;
        dsu.par[node] = node;           // node is the new component root
    }

    // DFS leaf order: each subtree's leaves become a contiguous interval.
    vector<int> left(nxt, 0), right(nxt, -1), place(n, 0);
    int timer = 0;
    vector<int> rootsList;
    for (int x = 0; x < nxt; x++) if (parent[x] == x) rootsList.push_back(x);
    sort(rootsList.begin(), rootsList.end());
    for (int root : rootsList) {
        // iterative post-order: stack of (node, child-index)
        vector<pair<int,int>> st;
        st.push_back({root, 0});
        while (!st.empty()) {
            int node = st.back().first;
            int ci = st.back().second;
            if (node < n) {
                place[node] = timer;
                left[node] = right[node] = timer;
                timer++;
                st.pop_back();
            } else if (ci < (int)children[node].size()) {
                st.back().second = ci + 1;
                st.push_back({children[node][ci], 0});
            } else {
                int lo = INT_MAX, hi = INT_MIN;
                for (int c : children[node]) { lo = min(lo, left[c]); hi = max(hi, right[c]); }
                left[node] = lo; right[node] = hi;
                st.pop_back();
            }
        }
    }

    Tree t;
    t.parent.assign(parent.begin(), parent.begin() + nxt);
    t.tag.assign(tag.begin(), tag.begin() + nxt);
    t.left = move(left);
    t.right = move(right);
    t.place = move(place);

    // Binary-lifting table over parent pointers.
    t.jump.push_back(t.parent);
    int bit = 1;
    while ((1 << bit) <= nxt) {
        const vector<int>& prev = t.jump.back();
        vector<int> cur(nxt);
        for (int x = 0; x < nxt; x++) cur[x] = prev[prev[x]];
        t.jump.push_back(move(cur));
        bit++;
    }
    return t;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, q;
    if (!(cin >> n >> m >> q)) return 0;
    vector<vector<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> S(q), E(q), L(q), R(q);
    for (int i = 0; i < q; i++) cin >> S[i] >> E[i] >> L[i] >> R[i];

    // human side: vertices on high-to-low; threshold L is a lower bound.
    vector<int> orderHigh(n), orderLow(n);
    for (int i = 0; i < n; i++) { orderHigh[i] = n - 1 - i; orderLow[i] = i; }
    Tree first = build_structure(n, adj, orderHigh);
    Tree second = build_structure(n, adj, orderLow);

    // vertex v becomes the point (first.place[v], second.place[v]).
    vector<int> axisValue(n, 0);
    for (int v = 0; v < n; v++) axisValue[first.place[v]] = second.place[v];

    // For each query, locate the high interval [xlo,xhi] and low interval [ylo,yhi].
    vector<char> usable(q, 0);
    vector<int> xlo(q), xhi(q), ylo(q), yhi(q);
    for (int i = 0; i < q; i++) {
        if (S[i] < L[i] || E[i] > R[i]) continue;     // a side is empty
        // climb in high tree while ancestor tag >= L
        int x = S[i];
        for (int j = (int)first.jump.size() - 1; j >= 0; j--) {
            int y = first.jump[j][x];
            if (y != x && first.tag[y] >= L[i]) x = y;
        }
        int a1 = first.left[x], b1 = first.right[x];
        // climb in low tree while ancestor tag <= R
        int z = E[i];
        for (int j = (int)second.jump.size() - 1; j >= 0; j--) {
            int y = second.jump[j][z];
            if (y != z && second.tag[y] <= R[i]) z = y;
        }
        int a2 = second.left[z], b2 = second.right[z];
        usable[i] = 1;
        xlo[i] = a1; xhi[i] = b1; ylo[i] = a2; yhi[i] = b2;
    }

    // Offline x-sweep with a Fenwick tree over y: rectangle count by x-prefix diff.
    // events[x] holds tuples (qid, sign, ylo, yhi).
    vector<vector<array<int,4>>> events(n);
    for (int i = 0; i < q; i++) {
        if (!usable[i]) continue;
        if (xlo[i] > 0) events[xlo[i] - 1].push_back({i, -1, ylo[i], yhi[i]});
        events[xhi[i]].push_back({i, +1, ylo[i], yhi[i]});
    }

    vector<long long> fenwick(n + 1, 0);
    auto addValue = [&](int idx) {
        idx += 1;
        while (idx <= n) { fenwick[idx] += 1; idx += idx & (-idx); }
    };
    auto prefixValue = [&](int idx) -> long long {  // #inserted with y in [0..idx]
        idx += 1;
        long long total = 0;
        while (idx > 0) { total += fenwick[idx]; idx -= idx & (-idx); }
        return total;
    };

    vector<long long> hits(q, 0);
    for (int x = 0; x < n; x++) {
        addValue(axisValue[x]);
        for (const auto& ev : events[x]) {
            int qid = ev[0], sign = ev[1], yl = ev[2], yh = ev[3];
            long long count = prefixValue(yh) - (yl > 0 ? prefixValue(yl - 1) : 0);
            hits[qid] += (long long)sign * count;
        }
    }

    string out;
    out.reserve(q * 4);
    for (int i = 0; i < q; i++)
        out += (usable[i] && hits[i] > 0) ? "YES\n" : "NO\n";
    cout << out;
    return 0;
}
```
