**Problem.** A directed graph of `n` depots and `m` one-way toll roads is given; road `u -> v` adds toll `w` (`0 <= w <= 10^9`), and tolls sum along a route. Read `n m` and the `m` triples `u v w` from stdin; print the minimum total toll of a directed route from depot `1` to depot `n`, or `-1` if `n` is unreachable. With `n = 1` the answer is `0`.

**Key idea — Dijkstra with a binary heap.** Weights are non-negative, so Dijkstra is correct: keep `dist[v]`, push `(dist, v)` onto a min-heap, pop the smallest, skip stale pops, and relax outgoing edges. The settled-vertex invariant — the first pop of a vertex carries its true shortest distance — holds because the heap pops in nondecreasing key order and `w >= 0` means extending a path never lowers its cost. Complexity `O((n + m) log n)`.

**Correctness.** Initialize `dist[1] = 0`, all others `INF`. Each `dist[v]` is only ever lowered by a relaxation `dist[v] = dist[u] + w` from a *popped* (hence final-distance) `u`, so it is an upper bound on the true distance throughout and equals it once `v` is popped. The lazy-deletion heap may hold outdated copies of a vertex; the guard `if (d > dist[u]) continue;` discards them, so each vertex is processed (its edges relaxed) at most once. At the end `dist[n]` is the minimum total toll, or remains `INF` iff `n` is unreachable.

**Pitfalls.**
1. *32-bit overflow — the headline trap.* A route can chain up to `n - 1 ≈ 2*10^5` roads of toll up to `10^9`, so a distance reaches `~2*10^14`, far past the signed-`int` cap `2147483647`. If `dist`, the heap keys, or the relaxation `d + w` are 32-bit, the sum wraps to a negative value and silently corrupts the answer. On the worked sample an `int` build computes `1600000000 + 800000000` and prints `-1894967296` instead of `2400000000`. Use `long long` for every distance quantity and force the add into 64-bit: `long long nd = d + (long long)w;`.
2. *Sentinel underflow.* With `INF = LLONG_MAX`, computing `INF + w` would overflow. It never happens here because Dijkstra only relaxes edges out of a popped, finite-distance vertex; an unreachable vertex is never popped, so its `INF` is read only in comparisons, never in an addition.
3. *Stale heap entries.* Without the `d > dist[u]` skip, vertices get re-processed; the strict `>` keeps exactly the fresh copy.

**Edge cases.** `n = 1` -> `dist[1] = 0` is printed (already at the port). `m = 0` or `n` in a separate component -> `dist[n] = INF` -> `-1`. Self-loops and zero-weight roads never improve a distance (`w >= 0`), so no infinite loop and no spurious re-push. Parallel roads are handled naturally by relaxation keeping the cheaper. The heaviest finite answer `~2*10^14` fits comfortably in `long long`.

**Complexity.** `O((n + m) log n)` time, `O(n + m)` space — well within `n <= 2*10^5`, `m <= 5*10^5` at a 2-second limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<pair<int,int>>> adj(n + 1); // (neighbor, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX;
    vector<long long> dist(n + 1, INF);
    dist[1] = 0;

    priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                   greater<pair<long long,int>>> pq;
    pq.push({0LL, 1});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue; // stale entry
        for (auto [v, w] : adj[u]) {
            long long nd = d + (long long)w; // long long: path cost can exceed 32-bit int
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    if (dist[n] == INF) cout << -1 << "\n";
    else cout << dist[n] << "\n";
    return 0;
}
```
