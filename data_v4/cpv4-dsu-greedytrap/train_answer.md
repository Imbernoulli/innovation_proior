**Problem.** There are `n` islands joined by `m` bidirectional bridges; bridge `i` connects `u_i, v_i` with capacity `c_i`, and the whole graph is connected (parallel bridges allowed). A path's *throughput* is the minimum capacity on it. For each of `q` requests `(s, t)` with `s != t`, output the **maximum throughput over all `s`–`t` paths** — the widest-path / maximin bottleneck. Read everything from stdin, print one integer per request.

**Why the obvious greedy is wrong.** Answering locally fails because the maximin objective is global.
- *Direct-edge greedy* (use the best bridge straight between `s` and `t`): on `0–1:6, 1–2:5, 2–3:4, 0–3:2`, request `0 3` returns the direct `2`, but the detour `0→1→2→3` has throughput `min(6,5,4)=4`. The widest path is not the shortest.
- *Highest-neighbour walk* (always step to the widest neighbour): on `0–1:10, 1–3:1, 0–2:3, 2–3:8`, request `0 3` walks `0→1` (cap 10) and dies at the cap-`1` edge `1–3`, returning `1`; but `0→2→3 = min(3,8)=3`. A wide first hop can dead-end into a narrow exit.
Both greedies are discarded.

**Key idea — DSU over capacity-sorted edges (max spanning tree).** Add bridges from **highest** capacity to **lowest**, maintaining components with a disjoint-set union. The widest-path bottleneck of `(s, t)` equals the capacity `c*` at which they *first* join the same component:
- *Achievable:* when they merge, all edges added so far have capacity `>= c*` and connect `s` to `t`, so a path of throughput `>= c*` exists.
- *Optimal:* before `c*` they were separate, so every edge of capacity `> c*` sits inside non-bridging components; any `s`–`t` path must cross an edge of capacity `<= c*`.

That `c*` is exactly the minimum-capacity edge on the unique `s`–`t` path of the **maximum spanning tree** the DSU builds. Build the tree by keeping each edge that merges two components, then answer each query with a path-minimum BFS over the tree.

**Two pitfalls to get right.**
1. *Min vs max spanning tree.* The edges must be sorted **descending** by capacity. A reflexive ascending `sort` builds the *minimum* spanning tree and returns a path dominated by *low* capacities — it reproduces the wrong direct-edge greedy value (`2` instead of `4` on the sample). One `>` in the comparator is the whole fix.
2. *Bottleneck initialization.* In the per-query BFS, seed `best[s] = +infinity` (`LLONG_MAX`), not `0`. The empty prefix (no bridge crossed yet) imposes no constraint, so the first edge's capacity should win the `min`; seeding `0` collapses every running `min(best[x], c)` to `0`. Use a distinct `UNVISITED = LLONG_MIN` guard — capacities are `>= 1`, so it never collides with a real bottleneck.

**Edge cases.** `n = 2`, single bridge → its capacity. Parallel edges → descending sort keeps the stronger one as a tree edge and discards the weaker as a cycle, automatically. Capacities near `10^9` → the answer is one capacity (no sums), carried in `long long`, no truncation; the `LLONG_MAX` start sentinel is only ever consumed by `min`, never added to. Connectivity is guaranteed, so every `t` is reachable.

**Complexity.** Sorting `O(m log m)`; tree construction `m` near-`O(1)` DSU ops; `q` queries each an `O(n)` tree BFS, `O(n·q) <= 4·10^6`. Memory `O(n + m)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, r;
    DSU(int n) : p(n), r(n, 0) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, q;
    if (!(cin >> n >> m >> q)) return 0;

    // edges stored as (capacity, u, v) so a plain sort orders by capacity first.
    vector<array<long long, 3>> edges(m);
    for (int i = 0; i < m; i++) {
        long long u, v, c;
        cin >> u >> v >> c;
        edges[i] = {c, u, v};
    }

    vector<int> S(q), T(q);
    for (int k = 0; k < q; k++) cin >> S[k] >> T[k];

    // Maximum spanning tree via Kruskal on DSU: add edges in DECREASING capacity.
    sort(edges.begin(), edges.end(),
         [](const array<long long, 3>& a, const array<long long, 3>& b) {
             return a[0] > b[0];
         });

    DSU dsu(n);
    vector<vector<pair<int, long long>>> tree(n);  // adjacency of the chosen tree
    for (auto& e : edges) {
        int u = (int)e[1], v = (int)e[2];
        long long c = e[0];
        if (dsu.unite(u, v)) {
            tree[u].push_back({v, c});
            tree[v].push_back({u, c});
        }
    }

    // Bottleneck(s, t) = min capacity on the unique s-t path of the max spanning tree.
    // The graph is connected, so the path always exists. n, q <= 2000 => O(q*n) BFS.
    const long long UNVISITED = LLONG_MIN;
    for (int k = 0; k < q; k++) {
        int s = S[k], t = T[k];
        vector<long long> best(n, UNVISITED);
        best[s] = LLONG_MAX;  // empty prefix: bottleneck is +infinity
        deque<int> bfs;
        bfs.push_back(s);
        while (!bfs.empty()) {
            int x = bfs.front();
            bfs.pop_front();
            for (auto& pr : tree[x]) {
                int y = pr.first;
                long long c = pr.second;
                if (best[y] == UNVISITED) {
                    best[y] = min(best[x], c);
                    bfs.push_back(y);
                }
            }
        }
        cout << best[t] << "\n";
    }
    return 0;
}
```
