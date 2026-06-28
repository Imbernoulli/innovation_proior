The demonstration calls for a shortest road route between two Dutch cities on a map small enough for a 1956 audience to check, yet the ARMAC has almost no memory. A road map is just points connected by segments with nonnegative lengths, possibly one-way, and the goal is the minimum-length route from a source to a target. The brute-force idea of enumerating every route is out of reach, and the known alternatives are too heavy for this machine. Ford's label-correcting method keeps a tentative distance for every node and repeatedly scans the entire edge set to relax violations, which means storing and re-scanning all branches many times before anything stabilizes. Kruskal's and Loberman-Weinberger's spanning-tree ideas sort all edges up front, so they too require the whole branch list to be resident at once. What is needed is a way to commit to answers one city at a time while touching only a tiny active slice of the map.

The method is Dijkstra's shortest-path algorithm. Instead of aiming straight at the destination, it grows a settled set of cities whose true shortest distance from the source is already known, starting with the source at distance zero. For every city not yet settled but reachable by a single road from the settled set, it keeps a tentative distance equal to the best known route that reaches some settled city and then crosses one road into that frontier city. The city on the frontier with the smallest tentative distance is moved into the settled set, and its distance becomes final. The reason this is safe is that all road lengths are nonnegative: any route to that city must leave the settled set somewhere, and the first crossing already costs at least as much as the smallest frontier tentative distance, while the remaining suffix cannot make the total any smaller. Once a city is settled it is never revisited, so the algorithm only ever needs one candidate branch per frontier city and one predecessor branch per settled city, rather than the full edge list.

When a city is newly settled, the only fresh information comes from its outgoing roads. For each neighbor not yet settled, the algorithm computes the candidate distance through the newly settled city; if this improves the neighbor's best known distance, the neighbor's tentative distance and predecessor are updated. Choosing the next city to settle is naturally implemented by a min-priority queue keyed by tentative distance. Stale queue entries left behind by later improvements are harmless: when one surfaces, it is discarded if the city is already settled or if its key is worse than the current best distance. This keeps the bookkeeping frugal and the code simple. A linear scan over the frontier would also work and uses only O(V) live records, while the heap version runs in O((E + V) log V) time for sparse graphs.

Concretely, the program reads the map from stdin as `n m s t`, then `m` directed roads
`u v w` (0-based city codes, nonnegative lengths `w`), and prints the minimum total length
from `s` to `t` followed by one shortest route, or `UNREACHABLE`.

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
