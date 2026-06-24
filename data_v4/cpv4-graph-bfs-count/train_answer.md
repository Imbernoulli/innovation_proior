**Problem.** An unweighted undirected graph of `n` stations and `m` bidirectional hops is given, with a source `s` and a sink `t`. Count, modulo `1_000_000_007`, the number of distinct minimum-hop itineraries from `s` to `t`, where an itinerary is the sequence of hops taken (so parallel edges are distinct connections and each yields a distinct route). Print `0` if `t` is unreachable and `1` if `s == t`.

**Key idea — BFS distances plus a one-pass count.** Let `dist[v]` be the BFS hop distance from `s` and `ways[v]` the number of minimum-hop itineraries from `s` to `v`. A hop `u -> v` lies on a shortest route to `v` exactly when `u` is one layer closer, `dist[u] + 1 == dist[v]`; summing over such hops,

  `ways[v] = sum of ways[u]` over hops `(u,v)` with `dist[u] + 1 == dist[v]`,  with `ways[s] = 1`.

Run a single BFS from `s`. When an edge first discovers `w`, set `dist[w] = dist[u]+1` and seed `ways[w] = ways[u]`. When a later edge reaches an already-discovered `w` with `dist[w] == dist[u] + 1`, add `ways[u]` into `ways[w]` (mod). Reduce modulo `1e9+7` at every addition.

**Why it is correct.** BFS dequeues vertices in nondecreasing distance, so when `u` (layer `dist[u]`) is dequeued, every contributor to `ways[u]` came from layer `dist[u]-1` and was already processed; `ways[u]` is therefore final before it is read. The predicate `dist[w] == dist[u] + 1` admits exactly the previous-layer contributors and excludes same-layer and behind neighbours, which lie on no shortest route. Hence the BFS computes the recurrence above exactly.

**Pitfalls.**
1. *Same-layer / add-back double-count (the central trap).* A naive `else { ways[w] += ways[u]; }` that fires for *every* already-seen neighbour wrongly accumulates across same-distance edges (`dist[u] == dist[w]`) and back toward the source along the undirected edge (`dist[w] < dist[u]`). Trace `[1-2,1-3,2-4,3-4,3-5,4-6,5-6]` plus decoys `2-3,4-5`: the unguarded version corrupts `ways` immediately; gating on `dist[w] == dist[u] + 1` fixes it and the decoys contribute nothing, giving the correct `3`.
2. *Reading a not-yet-final count.* A two-pass version that iterates vertices in **index** order (`for v = 1..n`) instead of distance order reads a half-built `ways[v]` whenever a contributor of `v` has a larger index. On `[1-7,1-8,7-2,8-2,2-3]` (`s=1,t=3`) it outputs `0` instead of `2`, because it pushes `ways[2]` to `3` before `7,8` have filled `ways[2]`. The one-pass BFS avoids this for free, since the queue already serves vertices in nondecreasing-distance order.
3. *Overflow.* Counts reach `W^(k-1)` (e.g. `10^10`), so keep them `long long` and reduce `% MOD` at every addition; a pre-reduction sum stays below `2*MOD`.

**Edge cases.** `s == t` -> `1` (`ways[s]` is pre-seeded and never overwritten, since a neighbour tests `0 == dist[u]+1`). `t` unreachable -> `0` via the explicit `dist[t] == INF` guard. Single isolated station -> `1`. Self-loops are dropped at read time (and would be harmless anyway: `dist[u] == dist[u]+1` is always false). Parallel edges are kept, each contributing once, matching the hop-sequence definition (a doubled `1-2` gives `2`). Same-distance "decoy" edges contribute nothing via the predicate.

**Complexity.** `O(n + m)` time and `O(n + m)` space — one BFS over the adjacency list.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    if (!(cin >> n >> m >> s >> t)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        if (u == v) continue;              // ignore self-loops: they never help a shortest path
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    const int INF = INT_MAX;
    vector<int> dist(n + 1, INF);
    vector<long long> ways(n + 1, 0);

    // BFS from s; layer-by-layer so every vertex is finalized in nondecreasing distance order.
    queue<int> q;
    dist[s] = 0;
    ways[s] = 1;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int w : adj[u]) {
            if (dist[w] == INF) {              // first time we reach w
                dist[w] = dist[u] + 1;
                ways[w] = ways[u];             // start its count from this predecessor
                q.push(w);
            } else if (dist[w] == dist[u] + 1) {
                // another shortest predecessor on the previous layer
                ways[w] = (ways[w] + ways[u]) % MOD;
            }
            // dist[w] == dist[u] (same layer) or dist[w] < dist[u]: contributes nothing.
        }
    }

    long long ans = (dist[t] == INF) ? 0 : ways[t] % MOD;
    cout << ans << "\n";
    return 0;
}
```
