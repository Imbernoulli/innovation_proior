**Problem.** A directed graph on `n` relay stations with strictly positive integer latencies (parallel
segments and self-loops allowed) and one source `s`. For every station report the number of distinct
minimum-latency routes from `s`, modulo `1e9+7`, where two routes differ if their sequence of segments
differs (so parallel segments count separately); print `-1` for unreachable stations. The source has one
route (the empty route), so its answer is `1`.

**Key idea — count during Dijkstra.** Run Dijkstra from `s` while carrying `cnt[i]` alongside `dist[i]`.
When relaxing a segment `u -> v` with `nd = dist[u] + w`:

- `nd <  dist[v]`: a strictly shorter route appeared, the old routes to `v` are obsolete, so **reset**
  `dist[v] = nd` and `cnt[v] = cnt[u]`, and push `v`.
- `nd == dist[v]`: another shortest route appeared, so **accumulate** `cnt[v] = (cnt[v] + cnt[u]) % MOD`
  and do **not** push (the distance did not change, so nothing new settles).

Initialize `dist[s]=0, cnt[s]=1`. The boundary is `< ` for reset and `==` for accumulate — never `<=`.

**Correctness.** The invariant is: *when station `u` is popped at its final distance, `cnt[u]` is already
the true number of minimum-latency routes to `u`.* Every minimum-latency route to `u` ends with a tight
segment `x -> u` (`dist[x] + w = dist[u]`), and because weights are strictly positive, `dist[x] < dist[u]`.
Dijkstra pops in nondecreasing distance, so every such `x` was popped earlier with its final `cnt[x]`, and
at that expansion the segment `x -> u` folded `cnt[x]` into `cnt[u]` exactly once (per parallel copy).
Hence `cnt[u]` is complete when `u` is expanded. Base case: `cnt[s]=1`. Parallel tight segments are handled
automatically because the relaxation loop visits each segment of `adj[u]` independently.

**Pitfalls.**
1. *Double-count from expanding a station twice.* If you push `v` on the equal-latency branch and omit the
   stale-entry guard, a settled station gets expanded again and re-pours its count into its neighbors. On a
   small graph this turned `cnt = 3` into `5`. Fix: push only on strict improvement, and add
   `if (d > dist[u]) continue;` so any leftover (stale) queue entry is discarded. Then every station is
   expanded exactly once, at its final distance.
2. *Off-by-one on the comparison.* Writing `<=` for the reset branch resets counts that should have been
   accumulated, undercounting (e.g. `cnt[4]` collapses from `4` to `1` on the sample). Reset is strictly
   `<`; equal is `==`.
3. *Overflow / modulus.* Distances reach `~2*10^14`, so `dist[]` and the relaxation accumulator must be
   `long long` (32-bit silently overflows). Reduce the count modulo `1e9+7` on **every** addition; a chain
   of doubling diamonds overflows 64 bits otherwise.

**Edge cases.** Source prints `1` (empty route); a self-loop on any node relaxes to a strictly larger
distance and is ignored, so it never inflates a count. Unreachable stations keep `dist = INF` and print
`-1`. `n = 1, m = 0` prints `1`. Two parallel source edges give `cnt = 2`. `INF` is only compared, never
has a weight added to it (relaxation only fires from popped, finite-distance nodes), so no `INF + w`
overflow.

**Complexity.** Standard binary-heap Dijkstra: `O((n + m) log n)` time, `O(n + m)` memory. The counting
adds only `O(1)` work per relaxation.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;
    const long long MOD = 1000000007LL;

    vector<vector<pair<int,int>>> adj(n); // adj[u] = list of (to, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF);
    vector<long long> cnt(n, 0); // number of shortest routes mod MOD

    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0;
    cnt[s] = 1;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;             // stale entry: a shorter dist[u] is already final
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {                // strictly shorter route to v: reset its count
                dist[v] = nd;
                cnt[v] = cnt[u];
                pq.push({nd, v});
            } else if (nd == dist[v]) {         // another shortest route to v: accumulate once
                cnt[v] = (cnt[v] + cnt[u]) % MOD;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) cout << -1 << "\n";
        else cout << (cnt[i] % MOD) << "\n";
    }
    return 0;
}
```
