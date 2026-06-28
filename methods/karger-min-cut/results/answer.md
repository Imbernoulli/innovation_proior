# Karger's Randomized Minimum Cut (and Karger–Stein)

## Problem

Given a connected undirected graph G = (V, E) on n vertices and m edges (possibly a weighted multigraph), find a **global minimum cut**: a partition of V into two nonempty sides minimizing the number (or total weight) of crossing edges. Unlike the s–t min cut, no source/sink is fixed — the minimum is over all bipartitions. The classical route reduces this to n−1 max-flow computations (O(mn)-ish, flow machinery throughout). The goal is an elementary, flow-free method that is competitive or better, at least with high probability.

## Key idea

A global min cut is *sparse*: the handshake identity Σ deg(u) = 2m plus "isolating one vertex is a valid cut" force the min-cut value k ≤ 2m/n, i.e. m ≥ nk/2. So a uniformly random edge is a cut edge with probability only k/m ≤ 2/n. **Edge contraction** — merging the two endpoints of an edge into one supernode, discarding the resulting self-loops, keeping parallel edges as multiplicities — commits two vertices to the same side and never lowers the min cut. Therefore contracting random edges preserves a fixed min cut with high per-step probability, especially early. Contract down to two supernodes and the edges between them are a cut of G; if no cut edge was ever contracted, it is *the* min cut.

## The Contraction Algorithm (Karger)

```
Contract(G):
  repeat until 2 vertices remain:
    choose an edge uniformly at random (weighted: prob ∝ edge weight)
    contract its endpoints (merge, drop self-loops, keep parallel edges)
  return the cut between the two surviving supernodes
```

**Success probability.** Fix a min cut C of value k. At i supernodes the contracted graph still has min cut ≥ k, hence ≥ ik/2 edges, so the step picks a C-edge with probability ≤ 2/i and survives with probability ≥ (i−2)/i. Telescoping from i = n down to i = 3:

  P[C survives] ≥ Π_{i=3}^{n} (i−2)/i = 2 / (n(n−1)) = 1 / C(n,2).

More generally, C survives **contraction to t vertices** with probability ≥ C(t,2)/C(n,2) ≈ (t/n)².

**Amplification.** Repeat T = ⌈C(n,2) · ln n⌉ ≈ ½ n² ln n independent runs and keep the smallest cut; failure ≤ (1 − 1/C(n,2))^T ≤ e^{−ln n} = 1/n. Each run is n−2 contractions at O(n) each ⇒ O(n²) per run ⇒ **O(n⁴ log n)** total, flow-free.

**Free corollary.** Distinct min cuts have disjoint "this run outputs exactly C" events, each of probability ≥ 1/C(n,2), so a graph has at most **C(n,2) = O(n²)** distinct minimum cuts.

## The Recursive Contraction Algorithm (Karger–Stein)

The risk lives in the *late* contractions: the per-step kill probability 2/i is negligible while i is large and only bites when i is small. Sharing the safe early prefix and branching only where risk concentrates removes the waste. Contract to the survival-½ threshold (t/n)² = ½ ⇒ **t ≈ n/√2**, then branch into **two** independent continuations and recurse.

```
Recursive-Contract(G, n):
  if n ≤ 6:  return Contract(G) down to 2 vertices      # brute base case
  t ← 1 + ⌈n/√2⌉                                         # survival ≥ 1/2
  repeat twice:  G' ← Contract(G, t);  recurse on (G', t)
  return the smaller of the two cut values
```

**Time.** T(n) = 2·T(⌈n/√2 + 1⌉) + O(n²). Branching 2, shrink √2 ⇒ Θ(n²) work at each of Θ(log n) levels ⇒ **T(n) = O(n² log n)** (critical case: n^{log_{√2}2} = n²).

**Success probability.** With p = P(n/√2), one branch survives the contraction (≥ ½) and then succeeds (p), so P(n) ≥ 1 − (1 − ½p)² = p − p²/4. The base case has p₀ ≥ 1/C(6,2) = 1/15. Setting z_k = 4/p_k − 1 gives z₀ ≤ 59 and z_{k+1} = z_k + 1 + 1/z_k, so z_k = Θ(k), p_k = Θ(1/k); with 2 log₂ n + O(1) levels, **P(n) = Ω(1/log n)**.

**Total.** Repeat Recursive-Contract O(log n / P(n)) = O(log² n) times and keep the smallest cut, giving failure probability 1/poly(n) for finding a minimum cut. The same repetition count can make the miss probability for any fixed minimum cut O(1/n⁴); union with the C(n,2) min-cut bound gives high probability for finding all minimum cuts. Total time is **O(n² log³ n)**.

## Equivalent view (Kruskal / random-weight MST)

Assign each edge an iid uniform random weight and run Kruskal's MST with union-find; stopping just before the last merge — equivalently, deleting the single heaviest MST edge — splits the graph into the two supernodes of one contraction run. A contraction run *is* a random-weight MST with the final union withheld, so union-find / MST machinery applies directly.

## Runnable code

Single self-contained C++17 program. It reads `n m` then `m` lines `u v` (1-based endpoints of an undirected edge) from stdin and prints the global min-cut value to stdout. A contraction run is held as a flat edge list plus a union-find (the Kruskal-in-disguise view): process edges in a uniformly random order, union endpoints until the target supernode count remains, and the cut value is the count of edges whose endpoints fall in different components. Vertex labels fit in `int`; the crossing-edge count is accumulated in `long long`.

```cpp
// Global minimum cut of a connected undirected (multi)graph via Karger / Karger-Stein.
// Reads from stdin: "n m" then m lines "u v" (1-based vertices); prints the global min-cut value.
#include <bits/stdc++.h>
using namespace std;

static std::mt19937 rng(0xC0FFEE);

// One contraction run on an edge multigraph held as a Union-Find over vertices
// plus a flat edge list. We process edges in a uniformly random order and union
// their endpoints (skipping self-loops) until exactly `t` supernodes remain;
// this is exactly "contract a uniformly random edge" repeated. Equivalent to a
// random-weight Kruskal MST with the last unions withheld.
struct DSU {
    vector<int> p, r;
    int comps;
    DSU(int n) : p(n), r(n, 0), comps(n) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        comps--;
        return true;
    }
};

// Contract `edges` (each {u,v}) down to `t` supernodes using a random edge order.
// Returns the DSU labeling. n = number of vertices currently in play.
static DSU contract(int n, const vector<pair<int,int>> &edges, int t) {
    DSU dsu(n);
    vector<int> order(edges.size());
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rng);
    for (int idx : order) {
        if (dsu.comps <= t) break;
        dsu.unite(edges[idx].first, edges[idx].second);
    }
    return dsu;
}

// Given a DSU that has contracted to exactly 2 supernodes, count crossing edges.
static long long cutValue(const DSU &dsu_const, const vector<pair<int,int>> &edges) {
    DSU dsu = dsu_const; // local copy so find() path-compression is harmless
    long long crossing = 0;
    for (auto &e : edges)
        if (dsu.find(e.first) != dsu.find(e.second)) crossing++;
    return crossing;
}

// Compress the current contracted graph into a smaller vertex set + edge list,
// dropping self-loops, so recursion shrinks the working size.
static void rebuild(DSU &dsu, const vector<pair<int,int>> &edges,
                    int &outN, vector<pair<int,int>> &outEdges) {
    unordered_map<int,int> relabel;
    relabel.reserve(dsu.comps * 2);
    auto id = [&](int root) -> int {
        auto it = relabel.find(root);
        if (it != relabel.end()) return it->second;
        int v = (int)relabel.size();
        relabel[root] = v;
        return v;
    };
    outEdges.clear();
    for (auto &e : edges) {
        int a = dsu.find(e.first), b = dsu.find(e.second);
        if (a == b) continue;                 // self-loop after contraction
        outEdges.push_back({id(a), id(b)});
    }
    outN = (int)relabel.size();
}

// Karger-Stein recursion: contract to ~n/sqrt(2), branch twice, recurse, take min.
static long long fastMinCut(int n, const vector<pair<int,int>> &edges) {
    if (n <= 6) {
        DSU dsu = contract(n, edges, 2);
        return cutValue(dsu, edges);
    }
    int t = 1 + (int)ceil(n / sqrt(2.0));
    long long best = LLONG_MAX;
    for (int b = 0; b < 2; b++) {
        DSU dsu = contract(n, edges, t);
        int subN; vector<pair<int,int>> subE;
        rebuild(dsu, edges, subN, subE);
        best = min(best, fastMinCut(subN, subE));
    }
    return best;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, m;
    if (!(cin >> n >> m)) return 0;

    vector<pair<int,int>> edges;
    edges.reserve((size_t)m);
    for (long long i = 0; i < m; i++) {
        long long u, v;
        cin >> u >> v;
        if (u == v) continue;                 // ignore self-loops in input
        edges.push_back({(int)(u - 1), (int)(v - 1)}); // to 0-based
    }
    if (n <= 1) { cout << 0 << "\n"; return 0; }

    // Repeat the recursive procedure O(log^2 n) times and keep the smallest cut,
    // driving failure to 1/poly(n).
    int trials = max(2, (int)ceil(pow(log(max(2.0, (double)n)), 2.0)));
    long long best = LLONG_MAX;
    for (int it = 0; it < trials; it++)
        best = min(best, fastMinCut((int)n, edges));

    cout << best << "\n";
    return 0;
}
```
