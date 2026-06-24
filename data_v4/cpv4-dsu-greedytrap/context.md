# Widest-path bottlenecks on an island bridge network

## Research question

There are `n` islands (numbered `0 .. n-1`) joined by `m` bidirectional bridges. Bridge `i` connects
islands `u_i` and `v_i` and has a **capacity** `c_i` — the most trucks per hour it can carry. The
network using all bridges is **connected** (you can get between any two islands). There may be several
bridges between the same pair of islands, and capacities may repeat.

For a path between two islands, its *throughput* is the **minimum** capacity of any bridge on it (the
narrowest bridge is the bottleneck). For `q` shipping requests, request `k` gives a source `s_k` and a
destination `t_k` (`s_k != t_k`) and asks for the **maximum throughput over all `s_k`–`t_k` paths** —
the *widest path* / maximin bottleneck. Output that value for each request.

This is the bottleneck-connectivity question that underlies reliability routing, network-capacity
planning, and "at what threshold do two nodes become connected" subproblems. The single direct bridge
between the endpoints is almost never the answer, and a hop-by-hop greedy walk can be arbitrarily bad,
so the exact structure — and the right data structure — matters.

## Input / output contract

- Input (stdin): the first line has three integers `n m q`
  (`2 <= n <= 2000`, `1 <= m <= 200000`, `1 <= q <= 2000`).
  Then `m` lines, each `u v c` (`0 <= u, v < n`, `u != v`, `1 <= c <= 10^9`) describing a bridge.
  Then `q` lines, each `s t` (`0 <= s, t < n`, `s != t`) describing a request.
  The graph formed by all bridges is guaranteed connected.
- Output (stdout): `q` lines; line `k` is the maximum throughput (widest-path bottleneck) for request
  `k`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: islands `0–1` (cap 6), `1–2` (cap 5), `2–3` (cap 4), `0–3` (cap 2). The request `0 3` has a
direct bridge of capacity `2`, but the path `0→1→2→3` has throughput `min(6,5,4) = 4`, so the answer is
`4`. The request `1 3` answers `min(5,4) = 4`.

## Background

Two families of approach are on the table before committing to one:

- **Greedy "best obvious bridge."** Answer each request from something local: the single
  highest-capacity bridge directly between `s` and `t`, or a hop-by-hop walk that always steps to the
  highest-capacity neighbour and reports the minimum along that walk. Both are `O(1)` or `O(path)` per
  request and trivial to write; the open question is whether a locally-best choice can reconstruct a
  globally-widest path under the maximin objective.
- **Union–Find over capacity-sorted edges (Kruskal-style).** Add bridges from highest capacity to
  lowest, maintaining connected components with a disjoint-set union. Two islands' widest-path
  bottleneck equals the capacity of the edge at which they *first* land in the same component — which
  is exactly the minimum edge on their path in the **maximum spanning tree**. The open question is the
  precise claim (why the max-spanning-tree path-minimum is the maximin bottleneck) and the exact
  bookkeeping.

## Evaluation settings

Judged on hidden tests covering: requests whose answer is *not* the direct edge between the endpoints;
"trap" instances where the highest-capacity first hop leads into a low-capacity dead end; parallel
edges between the same pair with different capacities; `n = 2` (one bridge); large capacities near
`10^9` (the answer is a single capacity, so it must be carried in a 64-bit-safe value and never
truncated); and large connected graphs with `m` up to `2*10^5` and up to `2000` requests.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, r;
    DSU(int n) : p(n), r(n, 0) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, q;
    if (!(cin >> n >> m >> q)) return 0;

    vector<array<long long, 3>> edges(m);        // (capacity, u, v)
    for (int i = 0; i < m; i++) {
        long long u, v, c;
        cin >> u >> v >> c;
        edges[i] = {c, u, v};
    }
    vector<int> S(q), T(q);
    for (int k = 0; k < q; k++) cin >> S[k] >> T[k];

    // TODO: compute, for each request (S[k], T[k]), the maximum over all paths
    // of the minimum bridge capacity along the path (the widest-path bottleneck).

    return 0;
}
```
