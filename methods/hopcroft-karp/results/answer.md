# Hopcroft-Karp Maximum Bipartite Matching

## Problem

Given a bipartite graph `G = (L, R, E)`, find a maximum-cardinality matching: a largest set of edges with no shared endpoint.

## Method

The algorithm keeps a valid matching in `match_l` and `match_r`. It repeatedly searches for augmenting paths, because by Berge's criterion a matching is maximum exactly when no augmenting path exists.

Each phase does two things:

1. `build_layers()` runs a BFS from all free left vertices through alternating edges. It records left-side layers in `level` and the first reachable free-right distance in `free_level`; expansion beyond that distance is skipped, so the phase is restricted to shortest augmenting paths.
2. `find_path(u)` runs a DFS inside that layered graph. It only advances to a matched left vertex in the next layer, and it only accepts a free right vertex exactly at `free_level`. Successful and failed left vertices are removed from the current phase search, so the phase augments along a maximal vertex-disjoint set of shortest augmenting paths.

The phase-increase proof is the key invariant. Let a phase choose `t` vertex-disjoint shortest augmenting paths, each of length `ell`, and let `M'` be the matching after flipping them. If a later augmenting path `P` for `M'` is disjoint from those paths, then it was already an augmenting path for `M`, so maximality rules out `|P| <= ell`. If `P` touches a chosen path, the shared vertex is internal to `P`, and the unique `M'`-matched edge at that vertex lies on both paths. Thus `(chosen paths) xor P` has at most `t ell + |P| - 1` edges, but it is also the symmetric difference between `M` and a matching of size `|M| + t + 1`, so it contains at least `t + 1` old augmenting paths, each of length at least `ell`. Therefore `|P| >= ell + 1`, and bipartiteness makes the next possible length at least `ell + 2`. Splitting the run after `floor(sqrt(|M*|))` phases, every remaining augmenting path contains at least `floor(sqrt(|M*|))` current-matching edges, so the remaining deficit is `O(sqrt(|M*|))`. Thus there are `O(sqrt V)` phases, each scanning `O(E)` edges, for total time `O(E sqrt V)` and space `O(V + E)`.

## Code

```cpp
// Maximum bipartite matching via Hopcroft-Karp.
// Reads from stdin: n (left size) m (right size) e (edge count), then e edges
// "u v" with u in [0,n), v in [0,m). Writes the matching size on the first
// line, then one matched "u v" pair per line.
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    long long e;
    if (!(cin >> n >> m >> e)) return 0;

    vector<vector<int>> adj(n);
    for (long long i = 0; i < e; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    const int INF = n + m + 1;
    vector<int> match_l(n, -1), match_r(m, -1), level(n, INF);
    int free_level = INF;

    // BFS: assign left-side layers, record shortest free-right distance.
    auto build_layers = [&]() -> bool {
        free_level = INF;
        vector<int> queue;
        queue.reserve(n);
        for (int u = 0; u < n; ++u) {
            if (match_l[u] == -1) {
                level[u] = 0;
                queue.push_back(u);
            } else {
                level[u] = INF;
            }
        }
        size_t head = 0;
        while (head < queue.size()) {
            int u = queue[head++];
            int next_level = level[u] + 1;
            if (next_level >= free_level) continue;
            for (int v : adj[u]) {
                int w = match_r[v];
                if (w == -1) {
                    free_level = next_level;
                } else if (level[w] == INF) {
                    level[w] = next_level;
                    queue.push_back(w);
                }
            }
        }
        return free_level != INF;
    };

    // DFS inside the layered graph; flip edges on success, mark vertices dead.
    function<bool(int)> find_path = [&](int u) -> bool {
        int next_level = level[u] + 1;
        for (int v : adj[u]) {
            int w = match_r[v];
            if (w == -1) {
                if (next_level == free_level) {
                    match_l[u] = v;
                    match_r[v] = u;
                    level[u] = INF;
                    return true;
                }
            } else if (next_level < free_level && level[w] == next_level && find_path(w)) {
                match_l[u] = v;
                match_r[v] = u;
                level[u] = INF;
                return true;
            }
        }
        level[u] = INF;
        return false;
    };

    while (build_layers()) {
        for (int u = 0; u < n; ++u) {
            if (match_l[u] == -1 && level[u] == 0) {
                find_path(u);
            }
        }
    }

    int size = 0;
    string out;
    for (int u = 0; u < n; ++u) {
        if (match_l[u] != -1) ++size;
    }
    out += to_string(size);
    out += '\n';
    for (int u = 0; u < n; ++u) {
        if (match_l[u] != -1) {
            out += to_string(u);
            out += ' ';
            out += to_string(match_l[u]);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```
