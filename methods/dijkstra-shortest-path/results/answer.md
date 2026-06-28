# Dijkstra's Shortest Path Algorithm

## Problem

Given a graph whose edges (roads) have **nonnegative** lengths — possibly directed — find the
minimum total length of a path from a source node `P` to a target `Q`, and recover the path
itself. The search state should avoid a sorted or repeatedly scanned full edge list.

## Key idea

Grow the set `A` of nodes whose shortest distance from `P` is known, one node at a time, in
order of increasing distance. For a frontier node `v`, its tentative distance `t[v]` is the
best known route that reaches some settled node and then crosses one edge into `v`.

If `u` has the smallest tentative distance, then `u` is final. Any path from `P` to `u` first
leaves `A` at some frontier node `y` via an edge `(x, y)` with `x` in `A`. The prefix to `x`
costs at least the known shortest distance `d[x]`, so the route's prefix through this
particular crossing costs at least `d[x] + length(x, y)`. The tentative distance `t[y]` is
the best crossing into `y` from the settled set, so that prefix costs at least `t[y]`. Since
`t[y] >= t[u]`, and the remaining suffix from `y` to `u` has nonnegative length, every path
to `u` costs at least `t[u]`. Since `t[u]` is itself a real path, it equals the true shortest
distance. This is exactly where nonnegative lengths are required.

## Algorithm

Maintain settled nodes, best tentative distances, predecessors, and a min-heap keyed by
tentative distance.

1. Put `P` in the frontier with distance 0.
2. Extract the smallest heap entry. If it is stale or already settled, skip it; otherwise
   settle that node.
3. If the node is `Q`, trace predecessors backward to recover the route.
4. For each outgoing edge `(u, v)` with `v` unsettled, compute `d(u) + length(u, v)`.
   If that improves `v`'s best known distance, record the new distance and predecessor and
   push the new heap entry.
5. Repeat until `Q` is settled or the frontier empties.

Works for directed edges. Correctness requires nonnegative lengths.

With a linear scan to choose the minimum frontier node, the time is `O(V^2 + E)` and the live
branch records stay `O(V)` besides the external road source. The heap implementation below
inspects each outgoing edge of a settled node once; each successful improvement pushes one
heap entry, and lazy deletion leaves stale entries in the heap. Its time is
`O((E + V) log(E + 1))` and its extra heap space can be `O(E)`. For a simple graph,
`log(E + 1) = O(log V)`, so this is commonly written `O((E + V) log V)` and often
`O(E log V)` when the edge term dominates.

## Code

Single-file C++17 reading from stdin: `n m s t`, then `m` directed roads `u v w` (0-based
city codes, nonnegative lengths). Prints the minimum total length from `s` to `t` and one
shortest route, or `UNREACHABLE`.

```cpp
// Dijkstra single-source shortest path (lazy-heap variant).
// Reads from stdin: "n m s t", then m lines "u v w" (0-based nodes, w >= 0,
// directed edges); prints the minimum total length from s to t followed by the
// node sequence of one shortest path, or "UNREACHABLE" if t cannot be reached.
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    if (!(cin >> n >> m >> s >> t)) return 0;

    vector<vector<pair<int, long long>>> adj(n);  // adj[u] = {(v, length)}
    for (int i = 0; i < m; ++i) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});  // lengths are nonnegative; roads may be directed
    }

    const long long INF = numeric_limits<long long>::max();
    vector<long long> best(n, INF);   // best tentative distance per node
    vector<int> pred(n, -1);          // predecessor on the recorded route
    vector<char> settled(n, 0);       // nodes whose shortest distance is final

    // Min-heap of (tentative distance, node); the frontier with set-II distances.
    priority_queue<pair<long long, int>, vector<pair<long long, int>>,
                   greater<pair<long long, int>>> heap;
    best[s] = 0;
    heap.push({0, s});

    while (!heap.empty()) {
        auto [dist, u] = heap.top();
        heap.pop();
        if (dist > best[u]) continue;   // stale key left by a later improvement
        if (settled[u]) continue;
        settled[u] = 1;                 // u joins A; dist is its true shortest distance
        if (u == t) break;

        // Only routes through the newly settled u can improve the frontier.
        for (auto [v, w] : adj[u]) {
            if (settled[v]) continue;
            long long cand = dist + w;
            if (cand < best[v]) {
                best[v] = cand;
                pred[v] = u;
                heap.push({cand, v});
            }
        }
    }

    if (best[t] == INF) {
        cout << "UNREACHABLE\n";
        return 0;
    }

    // Reconstruct the route by tracing predecessors back to the source.
    vector<int> path;
    for (int node = t; node != -1; node = pred[node]) path.push_back(node);
    reverse(path.begin(), path.end());

    cout << best[t] << "\n";
    for (size_t i = 0; i < path.size(); ++i)
        cout << path[i] << " \n"[i + 1 == path.size()];
    return 0;
}
```

The heap may contain stale entries after a better tentative distance is found; comparing the
popped distance with `best[u]` and skipping settled nodes gives lazy deletion without an
explicit decrease-key.
