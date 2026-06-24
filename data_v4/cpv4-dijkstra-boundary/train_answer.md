**Problem.** A courier starts at station `1` at time `0` in a directed graph; a connection `(u, v, w)` costs `w` minutes. Each station `v` has a curfew `c[v]`: being present at `v` is allowed only while `t < c[v]` (strict). Find the earliest legal arrival time at station `n`, or `-1` if none exists — including when the courier may not even legally stand at station `1` at time `0` (`c[1] = 0`). Read `n`, `m`, the curfews, and the `m` edges from stdin; print one integer.

**Key idea — Dijkstra with a strict-curfew relaxation gate.** Edge weights are positive and the constraint depends only on *arrival time*, so the right per-node label is the earliest legal arrival, and that label is monotone: a smaller arrival at `u` can only produce smaller, equally-or-more-legal arrivals downstream. That is exactly Dijkstra's settle-once condition, so run a min-heap Dijkstra. When relaxing edge `u -> v` with candidate arrival `nd = dist[u] + w`, accept it only if it is **legal and improving**:

- legal: `nd < c[v]` (strict — arriving exactly at the curfew is forbidden),
- improving: `nd < dist[v]`.

Seed the start the same way: the origin is an arrival at time `0`, so set `dist[1] = 0` only when `0 < c[1]`. Answer is `dist[n]`, or `-1` if it stays infinite.

**Why it's correct.** Waiting never helps (arriving earlier satisfies every curfew a later arrival would and yields smaller successor times), so an optimal route is a simple path and the earliest-legal-arrival label is well defined. Positive weights plus monotone labels give the standard Dijkstra guarantee: the first pop of a node is its true earliest legal arrival. The curfew gate only refuses to create illegal labels — equivalent to deleting an edge for that arrival time — so it does not disturb the ordering invariant. This was cross-checked against an independent exhaustive simple-path enumeration on hundreds of boundary-dense random graphs with zero mismatches.

**Pitfalls.**
1. *Off-by-one at the curfew (the whole point).* The legal region is `t < c[v]`, strict. Coding `nd <= c[v]` admits the single forbidden instant `t = c[v]` and changes the answer: e.g. one edge `1->2 (5)` with `c[2] = 5` has true answer `-1`, but `<=` reports `5`. A traced two-node case exposes it; tighten to `<`.
2. *The start is also an arrival.* Seeding `dist[1] = 0` unconditionally lets an illegal origin (`c[1] = 0`) leak through — an `n = 1` graph then prints `0` instead of `-1`. Gate the seed on `0 < c[1]`.
3. *Overflow.* With `n, m` up to `2*10^5` and weights up to `10^9`, arrival times reach `~2*10^14`; use `long long` for distances, the running sum, and curfews. An `int` is a silent wrong-answer.

**Edge cases.** `n = 1` -> `0` if `c[1] > 0`, else `-1`; every route into `n` blocked or landing on/after a curfew -> `-1`; `m = 0`, `n > 1` -> `-1`; parallel edges -> each candidate considered independently, the min legal one survives; `INF = LLONG_MAX/4` is never added to, so the unreachable test `dist[n] >= INF` is safe.

**Complexity.** `O((n + m) log n)` time, `O(n + m)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> c(n + 1);
    for (int v = 1; v <= n; v++) cin >> c[v];

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX / 4;
    vector<long long> dist(n + 1, INF);

    // You start at station 1 at time 0.
    // Being present at station v at any time t with t >= c[v] is forbidden,
    // so an arrival time t at v is LEGAL only if t < c[v] (strict).
    // Departure happens at the same instant as arrival (no waiting helps),
    // so the only constraint per node is the strict-inequality arrival check.

    // Start station must itself be legal at time 0.
    if (0 < c[1]) {
        dist[1] = 0;
    }

    priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                   greater<pair<long long,int>>> pq;
    if (dist[1] == 0) pq.push({0, 1});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            // Arriving at v at time nd is allowed only if nd < c[v] (strict boundary).
            if (nd < c[v] && nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    if (dist[n] >= INF) cout << -1 << "\n";
    else cout << dist[n] << "\n";
    return 0;
}
```
