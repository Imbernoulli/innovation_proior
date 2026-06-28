# Centroid decomposition for the exact-length tree path with fewest edges

## Problem

Tree on $n$ vertices, each edge a non-negative integer length, plus a target
$K$. Among all simple paths between two distinct vertices whose lengths sum to
exactly $K$, return the minimum number of edges, or $-1$ if no path totals $K$.

## Key idea

**Divide and conquer over paths-through-a-vertex.** Every path either passes
through a chosen vertex $u$ or lies entirely in one of the pieces left when $u$
is deleted. Handle all paths through $u$ now, delete $u$, and recurse on each
piece. A fixed path is processed at the first deleted vertex on that path; after
that deletion its endpoints are no longer together in any recursive piece, so it
is not processed again.

**Choose $u$ to be the centroid** — the vertex whose deletion leaves every piece
with at most $n/2$ vertices. Such a vertex always exists (at most one piece can
exceed $n/2$; step toward an oversized piece and the walk never backtracks, so
it halts at a balanced vertex), so the recursion has depth $O(\log n)$. The
centroid is found in $O(n)$: root the piece, compute subtree sizes $\mathrm{sz}$,
and pick the vertex minimizing
$\max\!\big(\,\text{(size }-\mathrm{sz}(u)),\ \max_{w \text{ child}} \mathrm{sz}(w)\big)$.

**Combine paths through the centroid in $O(\text{size})$.** A path through $u$ is
two half-paths into *different* neighbor-subtrees of $u$ (or $u$ itself as an
endpoint). DFS each neighbor-subtree to list every $(\text{cost},\text{depth})$ =
(summed length, edge count) from $u$, pruning once cost $> K$. Keep an array
$A[c]$ = minimum depth of a half-path of length $c$ seen in *earlier* subtrees.
Per subtree, **query then fill**: for each $(c,d)$, if $A[K-c]$ is set the glued
path has length exactly $K$ with $d + A[K-c]$ edges (and if $c=K$, depth $d$
alone, with $u$ an endpoint); then record $A[c]\leftarrow\min(A[c],d)$. Querying
before filling guarantees the two halves lie in different subtrees, so no path
doubles back through one edge.

**$O(1)$ reset via a freshness stamp.** $A$ is shared across all centroids over
the whole recursion; clearing $K+1$ slots per centroid would be $O(nK)$. Instead
a per-centroid stamp marks which slots are live: $A[c]$ counts only when
`seen[c]` equals the current stamp, and bumping the stamp invalidates everything
in $O(1)$.

## Complexity

$T(n) = O(n) + \sum_i T(n_i)$ with each $n_i \le n/2$, so $O(\log n)$ levels; each
level does $O(\text{level size})$ work and the level sizes sum to $\le n$, giving
**$O(n\log n)$ time**. Memory is $O(n)$ for the tree plus $O(K)$ for the two
helper arrays.

## Code

A single self-contained C++17 program: it reads `n K` then `n-1` lines `u v w`
(0-based vertices, non-negative weights) from stdin, and prints the minimum
number of edges on a path of total length exactly $K$, or $-1$ if none.

```cpp
// Reads "n K" then n-1 lines "u v w" (0-based vertices, non-negative weights);
// prints the minimum number of edges on a simple path of total length exactly K,
// or -1 if no such path exists. O(n log n) via centroid decomposition.
#include <bits/stdc++.h>
using namespace std;

int n;
long long K;
vector<vector<pair<int, long long>>> g;        // g[u] = list of (neighbor, weight)

vector<char> removed_;                          // centroid already deleted from tree
vector<int> sz;                                 // subtree sizes within current piece
vector<int> par;                                // reusable parent array

// best_depth[c] = min #edges of a centroid->node half-path of total length c,
// among neighbor-subtrees processed SO FAR. Live only when seen[c] == stamp
// (a per-centroid stamp gives O(1) reset, no O(K) wipe).
vector<int> best_depth;
vector<long long> seen;
long long stamp = 0;
long long answer = -1;

// Iterative DFS over the current component: fill order[] and par[], then sizes.
void calc_size(int root, vector<int>& order) {
    order.clear();
    vector<int> st = {root};
    par[root] = root;
    while (!st.empty()) {
        int cur = st.back(); st.pop_back();
        order.push_back(cur);
        for (auto& e : g[cur]) {
            int nxt = e.first;
            if (!removed_[nxt] && nxt != par[cur]) {
                par[nxt] = cur;
                st.push_back(nxt);
            }
        }
    }
    for (int cur : order) sz[cur] = 1;
    for (int i = (int)order.size() - 1; i >= 0; --i) {   // children before parents
        int cur = order[i];
        if (par[cur] != cur) sz[par[cur]] += sz[cur];
    }
}

// The centroid minimizes the largest piece left after its deletion.
int find_centroid(const vector<int>& order, int total) {
    int best = order[0], best_max = total + 1;
    for (int cur : order) {
        int mx = total - sz[cur];                // the "upward" piece
        for (auto& e : g[cur]) {
            int nxt = e.first;
            if (!removed_[nxt] && nxt != par[cur] && sz[nxt] > mx)
                mx = sz[nxt];                    // a child subtree
        }
        if (mx < best_max) { best_max = mx; best = cur; }
    }
    return best;
}

// Collect (cost, depth) of every half-path into the subtree entered via 'start';
// 'centroid' is start's parent, so the DFS never crosses back through it into a
// sibling subtree. Prune once cost > K.
void dfs_collect(int start, long long c0, int centroid,
                 vector<pair<long long, int>>& out) {
    out.clear();
    // stack of (node, cost, depth, parent)
    vector<tuple<int, long long, int, int>> st;
    st.emplace_back(start, c0, 1, centroid);
    while (!st.empty()) {
        auto [cur, cost, depth, p] = st.back(); st.pop_back();
        if (cost > K) continue;
        out.emplace_back(cost, depth);
        for (auto& e : g[cur]) {
            int nxt = e.first;
            if (!removed_[nxt] && nxt != p)
                st.emplace_back(nxt, cost + e.second, depth + 1, cur);
        }
    }
}

void process(int start) {
    // Iterative worklist over pieces to avoid recursion depth O(n) on a line graph.
    vector<int> work = {start};
    vector<int> order;
    vector<pair<long long, int>> half;
    while (!work.empty()) {
        int s = work.back(); work.pop_back();
        if (removed_[s]) continue;
        calc_size(s, order);
        if ((int)order.size() == 1) continue;
        int total = sz[s];
        int c = find_centroid(order, total);

        ++stamp;
        for (auto& e : g[c]) {                   // one neighbor-subtree at a time
            int nb = e.first;
            if (removed_[nb]) continue;
            dfs_collect(nb, e.second, c, half);
            for (auto& pr : half) {              // query against EARLIER subtrees
                long long cost = pr.first; int depth = pr.second;
                if (cost == K && (answer == -1 || depth < answer))
                    answer = depth;              // centroid is an endpoint
                long long need = K - cost;
                if (need >= 0 && need <= K && seen[need] == stamp) {
                    long long cand = (long long)depth + best_depth[need];
                    if (answer == -1 || cand < answer) answer = cand;
                }
            }
            for (auto& pr : half) {              // then fill, visible to LATER subtrees
                long long cost = pr.first; int depth = pr.second;
                if (seen[cost] != stamp || depth < best_depth[cost]) {
                    seen[cost] = stamp;
                    best_depth[cost] = depth;
                }
            }
        }

        removed_[c] = 1;                         // delete centroid, recurse on pieces
        for (auto& e : g[c])
            if (!removed_[e.first]) work.push_back(e.first);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> K)) return 0;             // empty input -> nothing to do

    g.assign(n, {});
    for (int i = 0; i < n - 1; ++i) {
        int u, v; long long w;
        cin >> u >> v >> w;
        g[u].emplace_back(v, w);
        g[v].emplace_back(u, w);
    }

    removed_.assign(n, 0);
    sz.assign(n, 0);
    par.assign(n, -1);
    best_depth.assign(K + 1, 0);
    seen.assign(K + 1, -1);

    if (n >= 1) process(0);
    cout << answer << "\n";
    return 0;
}
```
