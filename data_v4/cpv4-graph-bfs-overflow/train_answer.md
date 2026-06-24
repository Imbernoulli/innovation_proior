**Problem.** A broadcast starts at city `1` and floods over `m` bidirectional, unweighted fiber links
among `n` cities. Each reached city `v` has a *depth* equal to its minimum hop count from city `1`,
and relaying there costs `depth(v) * w[v]` for a non-negative importance `w[v]` (the source has depth
`0`). Cities unreachable from city `1` contribute nothing. Read `n m`, the importances, and the
edges from stdin; print the total relay cost over all reachable cities.

**Key idea — BFS depths, then a 64-bit weighted sum.** Because every link costs exactly one hop, the
minimum hop count from city `1` is exactly what breadth-first search computes: BFS visits cities in
non-decreasing depth order and fixes each city's depth the first time it is discovered, in
`O(n + m)`. Run BFS from city `1` to get `dist[v]` (using a sentinel `-1` for "unreached"), then
output `sum over reachable v of dist[v] * w[v]`.

**Why BFS (not DFS or Dijkstra).** DFS reports the length of *some* path, not the minimum — on the
triangle `1-2`, `2-3`, `1-3` it can stamp `depth(3) = 2` when the true depth is `1` — so it is wrong
for "minimum hops." Dijkstra is correct but pays an unnecessary `log` factor when all edges have equal
weight. BFS is the exact, fastest fit; correctness is the standard layer argument (a city dequeued at
depth `d` only discovers neighbours at depth `d + 1`, and the first discovery is the minimum).

**Pitfalls.**
1. *Overflow — the headline.* Depths reach `n - 1 ~ 2*10^5` and importances reach `10^6`, so a single
   product `depth(v) * w[v]` reaches `~2*10^11` and the total reaches `~2*10^16`. Both blow past the
   32-bit `int` ceiling (`~2.1*10^9`). The accumulator must be `long long`, **and** the multiply must
   be widened *before* it happens: `total += (long long)dist[v] * w[v];`. Writing `total += dist[v] *
   w[v];` with two `int`s overflows the product first and only widens the already-broken result. The
   trap is silent: the 26-cost sample passes either way, and an `int` version prints negative garbage
   (e.g. `-1760880640`) only on large hidden tests. The worst-case total `w_max * (n-1)*n/2 ~ 2*10^16`
   sits safely under the `long long` ceiling `~9.2*10^18`, so 64-bit always suffices.
2. *Reachability test.* Use `dist[v] != -1`, not `dist[v] > 0`. The source has depth `0` and IS
   reachable; `> 0` would silently treat it as unreached. (It costs `0` here, so the bug hides on
   small cases — exactly why it is dangerous.)
3. *Self-loops / duplicate edges.* Harmless: a repeated or self-neighbour is already visited when
   seen, so the sentinel check skips it. No special handling needed.

**Edge cases.** `n = 1, m = 0` → `0`. `m = 0, n > 1` → only city `1` reachable → `0`. All importances
zero → `0`. Disconnected graph → unreachable cities stay `-1` and are skipped (the sample's city `6`).
Maximum instance `n = m = 2*10^5`, all `w = 10^6` → total `~2*10^16`, positive and exact in
`long long`.

**Complexity.** `O(n + m)` time for BFS and the sum, `O(n + m)` memory for the adjacency list. At
`n = m = 2*10^5` it runs in about `0.04 s`, far inside the `1 s` limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> w(n + 1);
    for (int v = 1; v <= n; v++) cin >> w[v];

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    // BFS from city 1 over the unweighted graph to get hop-distances.
    const int INF = -1;
    vector<int> dist(n + 1, INF);
    queue<int> q;
    dist[1] = 0;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    // Sum dist[v] * w[v] over all reachable cities. With dist up to n-1
    // (~2e5) and w up to 1e6, a single product reaches ~2e11 and the whole
    // sum reaches ~2e16, so 64-bit accumulation is mandatory; int overflows.
    long long total = 0;
    for (int v = 1; v <= n; v++) {
        if (dist[v] != INF) {
            total += (long long)dist[v] * w[v];
        }
    }

    cout << total << "\n";
    return 0;
}
```
