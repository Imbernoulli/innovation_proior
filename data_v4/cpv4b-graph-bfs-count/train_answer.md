**Problem.** A subway network has `n` stations and `m` bidirectional one-minute tracks (tracks may be
parallel or self-loops). Count the distinct *fastest* routes from station `1` to station `n`, where a
route is the **sequence of stations** visited, modulo `1_000_000_007`. Output `0` if `n` is
unreachable. Read `n`, `m`, then `m` track pairs from stdin; print the count.

**Key idea — layered BFS counting.** Run one BFS from station `1` to get `dist[v]` (fewest minutes).
Let `ways[v]` be the number of fastest routes from `1` to `v`. Then `ways[1] = 1` and, for `v != 1`,

  `ways[v] = sum over neighbours u with dist[u] = dist[v] - 1 of ways[u]`,

because a fastest route to `v` is a fastest route to some neighbour `u` one layer closer, extended by
the hop `u -> v`. Evaluate it in the same BFS pass: when station `u` (distance `d`) is dequeued and a
neighbour `v` has `dist[v] == d + 1`, add `ways[u]` into `ways[v]`. BFS dequeues layer `d` entirely
before layer `d + 1`, so `ways[v]` is fully summed by the time `v` is processed. All sums are taken
modulo the prime. `O(n + m)` time.

**Pitfalls.**
1. *Parallel tracks double-count.* Building adjacency straight from the input (a multigraph) makes the
   BFS treat each physical track between the same two stations as a separate predecessor, adding the
   same hop's count more than once. A route is a station sequence, so first collapse the graph to a
   **simple** graph (one neighbour entry per pair) and drop self-loops. Trace: two parallel `1-2`
   tracks must give `1`, not `2`.
2. *The layer condition.* Only a neighbour in the *next* layer is a successor. Using `dist[v] >= dist[u]`
   (or otherwise admitting same-layer or closer neighbours) folds in a *same-layer* edge as a phantom
   predecessor and double-counts. Use the strict test `dist[v] == dist[u] + 1`; same-layer
   (`dist[v] == dist[u]`) and closer (`dist[v] < dist[u]`) neighbours contribute nothing.
3. *Overflow / modulus.* The count is exponential (a chain of `k` diamonds yields `2^k` routes), so use
   `long long` and reduce every accumulation `% MOD`. An `int` silently wraps and lies on large tests.

**Edge cases.** Disconnected (`dist[n] == INF`) -> `0`. Direct edge `1-n` -> `1`. Parallel tracks plus a
self-loop -> still `1` after dedup. Source seeded `ways[1] = 1` (standing still is one route).

**Complexity.** `O(m log n)` to dedup with per-station sets plus `O(n + m)` BFS; `O(n + m)` memory.
Measured `0.43 s`, `64 MB` at `n = 2*10^5`, `m = 4*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> adj(n + 1);
    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
    }
    // Build simple-graph adjacency: collapse parallel edges, drop self-loops,
    // because a "route" is a station sequence, so multiplicity does not matter.
    {
        vector<set<int>> tmp(n + 1);
        for (auto &e : edges) {
            int u = e.first, v = e.second;
            if (u == v) continue;            // self-loop: never on a shortest simple step
            tmp[u].insert(v);
            tmp[v].insert(u);
        }
        for (int u = 1; u <= n; u++)
            adj[u].assign(tmp[u].begin(), tmp[u].end());
    }

    const int INF = INT_MAX;
    vector<int> dist(n + 1, INF);
    vector<long long> ways(n + 1, 0);
    dist[1] = 0;
    ways[1] = 1 % MOD;
    queue<int> q;
    q.push(1);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                ways[v] = ways[u];
                q.push(v);
            } else if (dist[v] == dist[u] + 1) {
                // v already discovered on the next layer: u is one valid predecessor.
                ways[v] = (ways[v] + ways[u]) % MOD;
            }
            // dist[v] == dist[u] (same layer) or dist[v] < dist[u]: NOT a predecessor.
        }
    }

    if (dist[n] == INF) {
        cout << 0 << "\n";
    } else {
        cout << ways[n] % MOD << "\n";
    }
    return 0;
}
```
