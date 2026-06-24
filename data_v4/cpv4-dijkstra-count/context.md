# Counting minimum-latency relay routes in a fiber backbone

## Research question

A continental fiber backbone is modeled as a directed graph on `n` relay stations `0..n-1`. Each fiber
segment is a directed edge `u -> v` carrying a positive integer latency `w`. Two stations may be joined
by **several distinct fiber segments** (parallel edges), and a segment may loop from a station to itself.
Signals originate at one designated source station `s`.

For every station `i`, define its *minimum latency* as the smallest total latency of any directed route
from `s` to `i`, and define the *number of minimum-latency routes* as how many distinct directed routes
from `s` to `i` achieve exactly that minimum latency. Two routes are distinct if they differ in their
**sequence of fiber segments** — so two parallel segments between the same pair of stations count as two
different routes. Output, for every station, the number of minimum-latency routes **modulo `1e9+7`**, or
`-1` if the station is unreachable from `s`.

Because all latencies are strictly positive, no route can revisit a station while keeping the latency
minimal, so the count is always finite. The point of the problem is the counting layer on top of a
shortest-path computation: parallel segments, equal-latency ties, and the modulus all conspire to make
a double-count or off-by-one easy to introduce.

## Input / output contract

- Input (stdin): the first line has three integers `n m s`
  (`1 <= n <= 2*10^5`, `0 <= m <= 5*10^5`, `0 <= s < n`).
  Then `m` lines follow, each `u v w` describing a directed segment `u -> v` with latency `w`
  (`0 <= u,v < n`, `1 <= w <= 10^9`). Parallel edges and self-loops may appear.
- Output (stdout): `n` lines. Line `i` (0-indexed) is the number of minimum-latency routes from `s` to
  station `i`, taken modulo `1e9+7`, or `-1` if station `i` is unreachable. Station `s` itself has
  exactly one route (the empty route), so its answer is `1`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the network

```
5 7 0
0 1 2
0 2 2
1 4 3
2 4 3
0 3 4
3 4 1
2 4 3
```

the answer is

```
1
1
1
1
4
```

Station 4 has minimum latency `5`, reached by `0->1->4` (one route), by `0->2->4` (two parallel
segments `2->4`, so two routes), and by `0->3->4` (one route): `1 + 2 + 1 = 4`.

## Background

The minimum latencies themselves are a textbook single-source shortest-path computation; with positive
weights, Dijkstra finalizes each station in nondecreasing distance order. The open part is layering a
*count* on top. Two families of approach are on the table before committing:

- **Count by a DAG-DP over the shortest-path subgraph.** First compute all distances, then keep only the
  *tight* segments `u -> v` with `dist[u] + w == dist[v]`; these form a DAG. The number of routes to `v`
  is the sum of the counts of its tight predecessors (each parallel tight segment contributing
  separately). This needs a valid topological order and a second pass.
- **Count during the Dijkstra relaxation itself.** Maintain `cnt[i]` alongside `dist[i]`: on a strictly
  shorter route to `v`, copy the predecessor's count; on an equal-latency route to `v`, add it. This is a
  single pass but is delicate — *when* the addition happens relative to a station being finalized
  determines whether a route is counted once, twice, or zero times.

## Evaluation settings

Judged on hidden tests covering: graphs with many parallel segments between the same pair, self-loops,
chains of equal-latency diamonds that double the count repeatedly (so the answer overflows 64 bits before
the modulus and must be reduced every step), unreachable stations, the source equal to various nodes,
single-node graphs (`n = 1`, `m = 0`), and large graphs (`n = 2*10^5`, `m = 5*10^5`) where an `O(m log n)`
method is required and 32-bit distance accumulators overflow.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

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

    // TODO: run Dijkstra from s while maintaining cnt[] so that for every node
    //       cnt[i] is the number of minimum-latency routes from s to i (mod MOD).

    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) cout << -1 << "\n";
        else cout << (cnt[i] % MOD) << "\n";
    }
    return 0;
}
```
