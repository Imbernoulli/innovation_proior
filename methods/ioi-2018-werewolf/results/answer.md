# Kruskal reconstruction trees + 2D range-intersection for the two-phase reachability query

## Problem

An undirected graph on $N$ vertices ($0..N-1$), $M$ edges, and $Q$ queries
$(S,E,L,R)$ with $L \le R$. A query asks whether some walk goes $S \to E$ in two
phases: it stays on vertices with index $\ge L$ until a single switch vertex $v$
(which must satisfy $L \le v \le R$), then stays on vertices with index $\le R$
to the end. Answer each query yes/no.

## Key idea

**Reduce a query to a set intersection.** Let
$V_L$ = vertices reachable from $S$ using only vertices $\ge L$ (the component of
$S$ in the subgraph induced by $\{v \ge L\}$), and
$V_R$ = vertices reachable from $E$ using only vertices $\le R$ (the component of
$E$ in the subgraph induced by $\{v \le R\}$; the graph is undirected, so
reaching $E$ from a switch vertex equals reaching the switch vertex from $E$). A
valid switch vertex is exactly a vertex in $V_L \cap V_R$ — and any such vertex
has index $\ge L$ and $\le R$ automatically, so the answer is **yes iff
$V_L \cap V_R \neq \varnothing$**. (If $S < L$ then $V_L = \varnothing$; if
$E > R$ then $V_R = \varnothing$ — those queries are immediately no.)

**Make the reachable sets contiguous via two Kruskal reconstruction trees.** As
$L$ decreases the subgraph "$\ge L$" only gains vertices, so components only
merge — a laminar (nested-or-disjoint) family, i.e. a tree. Build it: turn
vertices on in **decreasing** index order; turning on $w$ creates a fresh
internal node tagged $w$, made the parent of the components of all already-on
neighbors (and of the leaf $w$). Tags strictly decrease from any leaf to the
root, so $V_L$ is the subtree under the **highest ancestor of $S$ whose tag is
$\ge L$**, found in logarithmic time with binary lifting over parent pointers. In DFS leaf order,
that subtree is a **contiguous interval** $[a_1, b_1]$. The wolf side is the
mirror: turn vertices on in **increasing** index order; $V_R$ is the subtree
under the highest ancestor of $E$ with tag $\le R$, a contiguous interval
$[a_2, b_2]$ in *that* tree's leaf order.

**Intersect via a 2D rectangle test.** The two trees order leaves differently,
so vertex $v$ gets two coordinates: $x_v = \text{posH}[v]$ (high-tree leaf
index) and $y_v = \text{posL}[v]$ (low-tree leaf index). The query
"$V_L \cap V_R \neq \varnothing$" becomes: **is there a vertex-point with
$x \in [a_1,b_1]$ and $y \in [a_2,b_2]$?** — a rectangle-nonempty test over $N$
static points. Answer all queries offline by sweeping $x$ and maintaining a
Fenwick tree (BIT) over $y$: a rectangle count is the $x$-prefix difference
$\text{cnt}(b_1) - \text{cnt}(a_1-1)$, each a $y$-range sum in the BIT.

## Algorithm

1. Build the **high tree** (vertices on in decreasing order) and **low tree**
   (increasing order); each via a DSU that records the merge tree.
2. DFS each tree to give every leaf a position and every node a contiguous
   leaf-interval $[\text{lo},\text{hi}]$.
3. Build binary-lifting tables for both trees. For each query, jump upward from
   $S$ in the high tree while the candidate ancestor's tag is $\ge L$ to get
   $[a_1,b_1]$, and from $E$ in the low tree while the candidate ancestor's tag
   is $\le R$ to get $[a_2,b_2]$; if $S<L$ or $E>R$, answer no.
4. Lay vertices out as points $(\text{posH},\text{posL})$ and answer every
   rectangle query with one $x$-sweep over a BIT on the $y$-axis.

## Code

A single-file C++17 program. It reads `N M Q`, then `M` lines `u v` (0-based
undirected edges), then `Q` lines `S E L R`, and prints `YES`/`NO` per query.
Index counts fit in `int`, but Fenwick prefix counts use `long long`.

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

## Complexity

- Building the two trees and their DFS leaf orders touches each vertex and edge a
  constant number of times through a near-linear DSU; binary-lifting tables add
  $O(N\log N)$ preprocessing.
- Each query does two logarithmic ancestor searches and contributes two Fenwick
  events. The sweep performs $O(N+Q)$ Fenwick operations, each $O(\log N)$.
- Total $O\big((N+M)\,\alpha(N) + (N+Q)\log N\big)$ time and
  $O(N\log N + M + Q)$ memory.
