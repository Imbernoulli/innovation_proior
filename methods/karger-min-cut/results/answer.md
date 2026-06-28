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
// Exact global minimum cut of an undirected unweighted multigraph.
// Reads from stdin: "n m" then m lines "u v" (1-based vertices); prints the min-cut value.
#include <bits/stdc++.h>
using namespace std;

static long long stoerWagner(vector<vector<long long>> weight) {
    int n = (int)weight.size();
    if (n <= 1) return 0;

    vector<int> vertices(n);
    iota(vertices.begin(), vertices.end(), 0);
    long long best = numeric_limits<long long>::max();

    for (int active = n; active > 1; --active) {
        vector<long long> addedWeight(active, 0);
        vector<char> used(active, false);
        int previous = -1;

        for (int step = 0; step < active; ++step) {
            int selected = -1;
            for (int i = 0; i < active; ++i) {
                if (!used[i] && (selected == -1 || addedWeight[i] > addedWeight[selected])) {
                    selected = i;
                }
            }

            if (step == active - 1) {
                best = min(best, addedWeight[selected]);

                for (int i = 0; i < active; ++i) {
                    if (i == selected) continue;
                    weight[vertices[previous]][vertices[i]] += weight[vertices[selected]][vertices[i]];
                    weight[vertices[i]][vertices[previous]] = weight[vertices[previous]][vertices[i]];
                }
                vertices.erase(vertices.begin() + selected);
                break;
            }

            used[selected] = true;
            previous = selected;
            for (int i = 0; i < active; ++i) {
                if (!used[i]) {
                    addedWeight[i] += weight[vertices[selected]][vertices[i]];
                }
            }
        }
    }

    return best;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long nInput, m;
    if (!(cin >> nInput >> m)) return 0;
    if (nInput <= 1) {
        cout << 0 << "\n";
        return 0;
    }

    int n = (int)nInput;
    vector<vector<long long>> weight(n, vector<long long>(n, 0));
    for (long long i = 0; i < m; i++) {
        long long u, v;
        cin >> u >> v;
        --u; --v;
        if (u == v) continue;                 // self-loops never cross a cut
        weight[(int)u][(int)v]++;
        weight[(int)v][(int)u]++;
    }

    cout << stoerWagner(weight) << "\n";
    return 0;
}
```
