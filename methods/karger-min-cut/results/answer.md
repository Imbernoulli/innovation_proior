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

```python
import copy
import math
import random


def contract(graph, t):
    """Contract uniformly random edges until t supernodes remain.

    graph: adjacency multigraph as {node: [neighbor, ...]}; parallel edges
    appear as repeated neighbors, self-loops are never stored.
    Returns the contracted multigraph (still has t supernodes).
    """
    g = copy.deepcopy(graph)
    while len(g) > t:
        # uniform-over-edges: pick u with prob proportional to its degree,
        # then a uniform incident edge -> overall uniform over edge endpoints.
        u = random.choices(list(g.keys()),
                           weights=[len(g[v]) for v in g])[0]
        w = random.choice(g[u])  # the edge (u, w) to contract

        # merge w into u: redirect w's incident edges to u, dropping self-loops
        for x in g[w]:
            if x != u:
                g[u].append(x)
        for x in g[w]:
            g[x].remove(w)
            if x != u:
                g[x].append(u)
        del g[w]
    return g


def cut_value(g):
    """Number of crossing edges once two supernodes remain."""
    return len(g[next(iter(g))])


def karger_min_cut(graph, trials):
    """Plain Karger: contract to 2, repeat, keep the smallest cut."""
    best = math.inf
    for _ in range(trials):
        g = contract(graph, 2)
        best = min(best, cut_value(g))
    return best


def fast_min_cut(graph):
    """Karger-Stein recursion: contract to ~n/sqrt2, branch twice, recurse."""
    n = len(graph)
    if n <= 6:
        g = contract(graph, 2)
        return cut_value(g)
    t = 1 + math.ceil(n / math.sqrt(2))
    g1 = contract(graph, t)
    g2 = contract(graph, t)
    return min(fast_min_cut(g1), fast_min_cut(g2))


def karger_stein_min_cut(graph, trials):
    best = math.inf
    for _ in range(trials):
        best = min(best, fast_min_cut(graph))
    return best


if __name__ == "__main__":
    # Two cliques of 4, joined by a single bridge edge -> min cut = 1.
    graph = {
        1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3, 5],
        5: [4, 6, 7, 8], 6: [5, 7, 8], 7: [5, 6, 8], 8: [5, 6, 7],
    }
    n = len(graph)
    T = math.ceil(n * (n - 1) / 2 * math.log(n))
    print("plain  :", karger_min_cut(graph, T))
    print("k-stein:", karger_stein_min_cut(graph, math.ceil(math.log(n) ** 2)))
```
