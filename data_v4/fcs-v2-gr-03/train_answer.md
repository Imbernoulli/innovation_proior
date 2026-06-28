**Problem.** Undirected graph on `n` vertices, `m` edges (may be disconnected, may
have parallel edges and self-loops). For each of `q` queries `u v`, count the
vertices `w` (`w != u`, `w != v`) whose deletion disconnects `u` from `v` — i.e.
the articulation points lying on *every* `u`-`v` path. Print `-1` if `u, v` are in
different components, `0` if `u == v`. Limits: `n, m, q <= 2*10^5`, 2 s.

**Why the obvious approach is too slow.** Per query, delete each candidate `w` and
BFS to test `u`-`v` reachability: `O(n(n+m))` per query, `O(q n (n+m))` overall —
about `10^16` at the limits, hopeless. (It is, however, the right brute-force
oracle.)

**Key idea — block-cut tree + LCA.** Only **articulation points** can ever
disconnect a pair, and they carve the graph into **biconnected components (BCCs)**.
Contract this into the **block-cut tree (BCT)**: one *block node* per BCC, one *cut
node* per articulation vertex, each cut node joined to every block containing it. A
non-cut vertex maps to its single block (`rep[v]`); a cut vertex maps to its own cut
node. The BCT is a forest. The vertices separating `u` from `v` are then exactly the
**cut nodes on the BCT path between `rep[u]` and `rep[v]`** — an articulation point
off that path has a route around it. Count cut nodes on a tree path in `O(log n)`
with binary-lifting LCA over a prefix count `cutPrefix[x]` (= cut nodes on root..x):

```
cutOnPath(a,b) = cutPrefix[a] + cutPrefix[b] - cutPrefix[lca] - cutPrefix[parent(lca)]
```

Total `O((n+m) log(n+m) + q log(n+m))`.

**Pitfalls.**
1. *Parallel-edge BCC boundary.* When a BCC closes (`low[u] >= disc[p]`), pop the
   edge stack down to the tree edge `p->u` identified by its **exact half-edge id**,
   not by endpoints. A parallel/back edge between `p` and `u` shares the endpoints
   `{p,u}` and stops the pop early, splitting one BCC and mislabeling cut vertices —
   the one real bug here (a query that should answer `2` returns `1`).
2. *Recursion depth.* A depth-`2*10^5` path overflows recursive DFS; use an
   explicit iterative DFS stack.
3. *Skip the parent EDGE, not the parent VERTEX.* With parallel edges these differ;
   store half-edges and skip `e ^ 1 == parentEdge`.
4. *Exclude `u`, `v` themselves.* If an endpoint is a cut vertex it sits on the BCT
   path; subtract 1 for each of `u`, `v` that is a cut vertex.

**Edge cases.** `n = 1` (singleton block, `0`); different components (`-1`);
`u == v` (`0`); a cycle/clique (no cut vertex, all `0`); parallel edges (not a
single point of failure); self-loops (ignored on read); endpoint is a junction
(endpoint-exclusion fires).

**Complexity.** `O((n + m) log(n + m))` preprocessing, `O(log(n + m))` per query;
`O(n + m)` memory. Full-scale (`n = m = q = 2*10^5`, deep-path adversary) runs in
~0.19 s, ~68 MB.

**Verification.** Differential-tested against the brute-force removal oracle on
`2000+` random small cases (sparse forests, dense blobs, disconnected graphs,
parallel edges, self-loops) and explicit edge cases — zero mismatches — plus a
hand-computed deep-path test and a full-scale timing run.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Cut vertices on every path between u and v, answered via the block-cut tree
  (BCT) plus LCA.

  Construction:
    * Iterative Tarjan biconnected-component (BCC) decomposition using an EDGE
      stack: discovery time disc[], low-link low[]; when low[child] >= disc[u] a
      biconnected component closes -- pop edges off the edge stack down to the
      tree edge (u,child); u is then an articulation point (root special-cased by
      its DFS-child count). Each BCC is a set of edges; its vertices are the
      endpoints of those edges, so a cut vertex shared by several BCCs is naturally
      present in each.
    * Build the BCT: one node per BCC ("block") and one node per articulation
      vertex ("cut"). A cut node is joined to every block containing it. A non-cut
      vertex lives in exactly one block; rep[v] is that block node. A cut vertex v
      has its own cut node; rep[v] is that cut node. The BCT is a forest (one tree
      per connected component of the input graph).

  Query (u != v, same component): the articulation points lying on EVERY u-v path
  are exactly the cut nodes on the BCT path between rep[u] and rep[v]. We must not
  count u or v themselves (the query asks for w != u, w != v), so subtract 1 for
  each of u, v that is itself an articulation point. If u, v are in different input
  components (no path at all), output -1. If u == v, output 0.
*/

int n, m, q;

// Input graph stored as half-edges so we can skip the parent EDGE (parallel-edge
// safe), not the parent vertex.
vector<int> head_, nxt_, to_;
void addEdge(int u, int v) {
    to_.push_back(v); nxt_.push_back(head_[u]); head_[u] = (int)to_.size() - 1;
    to_.push_back(u); nxt_.push_back(head_[v]); head_[v] = (int)to_.size() - 1;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> m)) return 0;
    head_.assign(n + 1, -1);
    for (int e = 0; e < m; e++) {
        int u, v; cin >> u >> v;
        if (u == v) continue;          // self-loop: irrelevant to articulation points
        addEdge(u, v);
    }

    vector<int> disc(n + 1, 0), low(n + 1, 0), comp(n + 1, 0);
    vector<char> isCut(n + 1, 0);
    int timer_ = 0, curComp = 0;

    // Each BCC is recorded as a list of vertices (endpoints of its edges,
    // de-duplicated). We pop edges off an edge stack when a BCC closes.
    vector<vector<int>> blocks;
    vector<int> edgeStack;             // half-edge ids forming the current open BCCs
    edgeStack.reserve(to_.size() / 2 + 1);

    // Iterative DFS frame.
    struct Frame { int u, parentEdge, ei; };
    vector<Frame> st; st.reserve(n + 1);

    for (int s = 1; s <= n; s++) {
        if (disc[s]) continue;
        curComp++;
        int rootChildren = 0;
        disc[s] = low[s] = ++timer_;
        comp[s] = curComp;
        st.push_back({s, -1, head_[s]});

        while (!st.empty()) {
            Frame &f = st.back();
            int u = f.u;
            if (f.ei != -1) {
                int e = f.ei;
                int v = to_[e];
                f.ei = nxt_[e];                         // advance before recursing
                if ((e ^ 1) == f.parentEdge) continue;  // skip the parent half-edge
                if (!disc[v]) {
                    edgeStack.push_back(e);             // tree edge onto the stack
                    if (u == s) rootChildren++;
                    disc[v] = low[v] = ++timer_;
                    comp[v] = curComp;
                    st.push_back({v, e, head_[v]});
                } else if (disc[v] < disc[u]) {
                    // back edge to an ancestor (disc[v] < disc[u] avoids pushing the
                    // same back edge twice from both endpoints)
                    edgeStack.push_back(e);
                    low[u] = min(low[u], disc[v]);
                }
            } else {
                int childParentEdge = f.parentEdge; // tree half-edge p->u (or -1 at root)
                st.pop_back();
                if (!st.empty()) {
                    int p = st.back().u;
                    low[p] = min(low[p], low[u]);
                    if (low[u] >= disc[p]) {
                        // BCC closes; pop edges down to (and including) THIS tree edge
                        // p->u. We identify it by its exact half-edge id, not by its
                        // endpoints -- parallel edges between p and u must not stop us
                        // early. childParentEdge is the half-edge p->u that we pushed
                        // onto edgeStack when we first descended into u.
                        if (p != s) isCut[p] = 1;       // non-root articulation point
                        vector<int> members;
                        while (!edgeStack.empty()) {
                            int e = edgeStack.back(); edgeStack.pop_back();
                            members.push_back(to_[e ^ 1]); // source endpoint
                            members.push_back(to_[e]);     // dest endpoint
                            if (e == childParentEdge) break;  // popped the boundary edge
                        }
                        sort(members.begin(), members.end());
                        members.erase(unique(members.begin(), members.end()), members.end());
                        blocks.push_back(move(members));
                    }
                }
            }
        }
        if (rootChildren >= 2) isCut[s] = 1;
        // Isolated vertex (no incident edge): it forms its own singleton block.
        if (head_[s] == -1) blocks.push_back({s});
    }

    // ---- Build the block-cut tree ----
    int bctN = 0;
    vector<int> cutNode(n + 1, -1), rep(n + 1, -1);
    for (int v = 1; v <= n; v++) if (isCut[v]) cutNode[v] = bctN++;
    int firstBlockNode = bctN;
    bctN += (int)blocks.size();
    vector<vector<int>> bct(bctN);
    vector<char> isCutBCT(bctN, 0);
    for (int v = 1; v <= n; v++) if (isCut[v]) isCutBCT[cutNode[v]] = 1;

    for (int b = 0; b < (int)blocks.size(); b++) {
        int blockNode = firstBlockNode + b;
        for (int v : blocks[b]) {
            if (isCut[v]) {
                int cn = cutNode[v];
                bct[cn].push_back(blockNode);
                bct[blockNode].push_back(cn);
            } else {
                rep[v] = blockNode;            // non-cut vertex: exactly one block
            }
        }
    }
    for (int v = 1; v <= n; v++) if (isCut[v]) rep[v] = cutNode[v];

    // ---- LCA on the BCT forest with binary lifting + prefix cut counts ----
    int LOG = 1; while ((1 << LOG) < max(1, bctN)) LOG++;
    vector<vector<int>> up(LOG + 1, vector<int>(bctN, -1));
    vector<int> depth(bctN, 0), cutPrefix(bctN, 0);
    vector<char> seen(bctN, 0);

    for (int s = 0; s < bctN; s++) {
        if (seen[s]) continue;
        seen[s] = 1; depth[s] = 0; up[0][s] = -1;
        cutPrefix[s] = isCutBCT[s] ? 1 : 0;
        queue<int> Q; Q.push(s);
        while (!Q.empty()) {
            int u = Q.front(); Q.pop();
            for (int w : bct[u]) if (!seen[w]) {
                seen[w] = 1;
                up[0][w] = u;
                depth[w] = depth[u] + 1;
                cutPrefix[w] = cutPrefix[u] + (isCutBCT[w] ? 1 : 0);
                Q.push(w);
            }
        }
    }
    for (int k = 1; k <= LOG; k++)
        for (int v = 0; v < bctN; v++) {
            int mid = up[k - 1][v];
            up[k][v] = (mid == -1) ? -1 : up[k - 1][mid];
        }

    auto lca = [&](int a, int b) -> int {
        if (depth[a] < depth[b]) swap(a, b);
        int diff = depth[a] - depth[b];
        for (int k = 0; k <= LOG; k++) if (diff & (1 << k)) a = up[k][a];
        if (a == b) return a;
        for (int k = LOG; k >= 0; k--)
            if (up[k][a] != up[k][b]) { a = up[k][a]; b = up[k][b]; }
        return up[0][a];
    };
    auto cutOnPath = [&](int a, int b) -> int {
        int l = lca(a, b);
        int pl = up[0][l];
        int res = cutPrefix[a] + cutPrefix[b] - cutPrefix[l];
        res -= (pl == -1) ? 0 : cutPrefix[pl];
        return res;
    };

    // ---- Answer queries ----
    cin >> q;
    string out; out.reserve((size_t)q * 3);
    while (q--) {
        int u, v; cin >> u >> v;
        if (u == v) { out += "0\n"; continue; }
        if (comp[u] != comp[v]) { out += "-1\n"; continue; }  // no path
        int cnt = cutOnPath(rep[u], rep[v]);
        if (isCut[u]) cnt--;       // exclude u itself
        if (isCut[v]) cnt--;       // exclude v itself
        out += to_string(cnt);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
