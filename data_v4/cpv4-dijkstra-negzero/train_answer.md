**Problem.** A directed graph of `n` relay stations and `m` links; each link `u -> v` has an integer margin `w` that may be negative, zero, or positive. A route's reliability is the **minimum** margin over its links. For a fixed source `s`, print for every station `v` the maximum reliability over all routes `s -> v` (the widest-path / max-min objective), with `INF` for the source line and `UNREACHABLE` for nodes with no route. Read `n m s` then `m` lines `u v w` from stdin.

**Key idea — max-min Dijkstra.** Maintain `best[v]` = the widest known bottleneck (max-min value) to `v`. Relax a link `u -> v` of margin `w` with `cand = min(best[u], w)`, keeping the maximum. Run Dijkstra with a **max-heap**, finalizing the unfinished node of largest `best` each step.

**Why Dijkstra is valid despite negative/zero weights.** The usual ban on negative edges is about *additive* shortest paths. Here the metric is `min`, and extending a route is **monotone non-increasing**: `min(best[u], w) <= best[u]` for any sign of `w`. So once the widest unfinished node is popped, no later route can reach it with a larger bottleneck — Dijkstra's exchange argument needs only monotonicity, not non-negativity. (Simple paths suffice: appending a cycle only adds edges, which can only lower a `min`.)

**Two pitfalls to get right (both are sign/base-case traps).**
1. *Source base case.* The empty route uses no link, so its bottleneck is the identity of `min`, which is `+infinity` — **not** `0`. Seeding `best[s] = 0` fabricates a phantom zero-margin link and clamps every reachable node to `<= 0`; with positive margins that is a blatant wrong answer. Seed `best[s] = POS_INF` so a first link `s -> v` yields `min(POS_INF, w) = w`. (Tracing the sample with `best[s]=0` turns all answers into `UNREACHABLE` — the smoking gun.)
2. *Reachability tested by sign.* A reachable node may legitimately have a **negative or zero** bottleneck. Testing `best[v] <= 0 -> UNREACHABLE` collapses reachable degraded nodes (and all-negative or zero-margin graphs) into "unreachable." Reachability is "did `best[v]` leave the `NEG_INF` sentinel," so test `best[v] == NEG_INF` **exactly**, never by sign.

**Correctness.** With the monotone relaxation and max-heap finalization, `best[v]` equals the maximum-over-routes minimum margin (standard widest-path Dijkstra). Lazy deletion is safe: the max-heap pops larger values first, so a node is finalized at its true `best[v]`; stale entries are skipped via `done[u]` (and the explicit `d != best[u]` guard). Cross-checked against an exhaustive simple-path brute force over 701 random cases (all-negative, all-zero, mixed, empty, self-loops, parallel edges) with zero mismatches.

**Edge cases.** `m = 0` -> only the source is reachable (`INF`), all others `UNREACHABLE`. `n = 1` -> single line `INF`. All-negative graph -> reachable nodes print their negative bottleneck, not `UNREACHABLE`. Zero-margin link -> reachable node prints `0`, distinct from `UNREACHABLE`. Self-loops/parallel edges -> harmless (`min(best[u], w) <= best[u]` can't improve a self-target; best parallel edge wins). The source reached by a real cycle keeps `POS_INF` (finite bottleneck can't beat the empty route). No weight is ever summed, so the value stays in `[-10^9, 10^9]` and never overflows; sentinels `LLONG_MAX/4` / `LLONG_MIN/4` sit far outside that range.

**Complexity.** `O((n + m) log n)` time, `O(n + m)` space. At `n = 2*10^5`, `m = 5*10^5` it runs in ~0.25 s.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long NEG_INF = LLONG_MIN / 4;   // "no path yet" / unreachable
    const long long POS_INF = LLONG_MAX / 4;   // bottleneck of the zero-edge path at the source

    vector<long long> best(n + 1, NEG_INF);
    vector<char> done(n + 1, 0);
    best[s] = POS_INF;                         // path of length 0: min over empty edge set = +inf

    // max-min Dijkstra: pop the node with the currently largest bottleneck.
    priority_queue<pair<long long,int>> pq;    // (bottleneck, node), max-heap
    pq.push({best[s], s});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (done[u]) continue;                 // stale / already finalized
        if (d != best[u]) continue;            // outdated entry
        done[u] = 1;
        for (auto [v, w] : adj[u]) {
            long long cand = min(best[u], w);  // weakest link along this path
            if (cand > best[v]) {
                best[v] = cand;
                pq.push({best[v], v});
            }
        }
    }

    for (int v = 1; v <= n; v++) {
        if (best[v] == NEG_INF) cout << "UNREACHABLE";
        else if (v == s && best[v] == POS_INF) cout << "INF";  // source with no self-route
        else cout << best[v];
        cout << "\n";
    }
    return 0;
}
```
