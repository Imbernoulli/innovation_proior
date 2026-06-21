# Context: Single-Source Shortest Paths with Negative Integer Weights

## Research question

Given a directed graph $G=(V,E,w)$ with $m$ edges, $n$ vertices, **integer** edge weights that may be negative, and a source $s$, compute the shortest-path distances $\operatorname{dist}_G(s,v)$ from $s$ to every vertex $v$ (equivalently, a shortest-path tree), or correctly report that some cycle reachable from $s$ has negative total weight. Let $W \ge 2$ be the smallest integer such that every edge weight is $\ge -W$.

When all weights are non-negative this is solved in near-linear time. The only fully general tool that handles negative weights — Bellman–Ford — costs $O(mn)$. The question is how to compute shortest-path distances on general directed graphs with integer edge weights.

This problem is foundational: it sits underneath transshipment, min-cost flow, the assignment problem, and many graph primitives, and it has driven "waves" of algorithmic progress since the 1950s.

## Background

**The two textbook algorithms.**
- *Dijkstra's algorithm* runs in $O(m + n\log n)$ time but is correct only for **non-negative** edge weights. Its correctness rests on a greedy invariant: once the vertex with the smallest tentative distance is extracted, that distance is final.
- *Bellman–Ford* (Shimbel 1955; Ford 1956; Bellman 1958; Moore 1959) handles negative weights and detects negative cycles, but runs in $O(mn)$: it performs up to $n-1$ rounds of relaxing all $m$ edges. Its real cost is proportional to how many *negative* edges a shortest path must traverse — a path that uses few negative edges is found quickly; a path that threads many negative edges forces many rounds.

**Price functions / reweighting (Johnson 1977).** For a function $\phi:V\to\mathbb{Z}$ define the *reduced* weight
$$w_\phi(u,v) = w(u,v) + \phi(u) - \phi(v).$$
For any $u,v$, $\operatorname{dist}_{G_\phi}(u,v) = \operatorname{dist}_G(u,v) + \phi(u) - \phi(v)$, and for any **cycle** $C$, $w_\phi(C)=w(C)$ (the $\phi$ terms telescope). Hence $\phi$ preserves the set of shortest paths and the set of negative cycles: $G$ and $G_\phi$ are *equivalent*. If a $\phi$ makes every reduced weight non-negative, one can run Dijkstra on $G_\phi$ and recover the true distances. Johnson's own choice $\phi(v)=\operatorname{dist}_G(s,v)$ (from a super-source reaching all vertices) achieves this. Price functions are also exactly the "heuristic" of $A^*$ and have roots in Dreyfus (1969) and Braess (1971).

**The scaling era (1980s–90s).** Gabow (1985), Gabow–Tarjan (1989), and Goldberg (1995) introduced *scaling* for shortest paths. Goldberg's bit-scaling framework maintains an integral price function $p$ that is *$\varepsilon$-feasible* ($w_p(e)\ge -\varepsilon$ for all $e$) and halves $\varepsilon$ each round via a subroutine REFINE; after $O(\log W)$ rounds $\varepsilon$ reaches $1$, and a $1$-feasible integral $p$ has $w_p(e)\ge 0$ for all $e$ (since reduced weights are integral). Goldberg explicitly notes: *"the basic problem solved at each iteration of the bit-scaling method is a special version of the shortest-paths problem where the arc lengths are integers greater or equal to $-1$."* The whole family reached $O(m\sqrt n \log W)$.

**The continuous-optimization wave (2017–).** A separate line — Cohen–Madry–Sankowski–Vladu (2017), Axiotis–Madry–Vladu (2020), van den Brand–Lee–Nanongkai–Peng–Saranurak–Sidford–Song–Wang (2020) and relatives — attacks the *more general* min-cost-flow / transshipment problem with interior-point methods and sophisticated dynamic graph data structures, and gets negative-weight SSSP as a special case. This produced $\tilde O((m+n^{1.5})\log W)$ on moderately dense graphs and $m^{4/3+o(1)}\log W$ on sparse graphs.

**Planar special case.** For planar digraphs, near-linear time has been known since Fakcharoenphol–Rao (2001), with the current best $O(n\log^2 n/\log\log n)$ (Mozes–Wulff-Nilsen 2010).

**Low-diameter decomposition (since the 1980s).** A recurring tool in *parallel/distributed/dynamic* algorithms is to delete a small fraction of edges so the remaining graph breaks into pieces of small diameter. For **undirected** graphs this is classical (Awerbuch 1985; Awerbuch–Goldberg–Luby–Plotkin 1989; Linial–Saks 1993; Bartal 1996; Miller–Peng–Xu 2013 via random exponential/geometric shifts; Blelloch et al. 2014). The Miller–Peng–Xu style grows balls of geometrically random radius around vertices and cuts the boundary; the probability an edge is cut scales like (edge length)·(log n)/(diameter parameter). For **directed** graphs the picture is subtler — one must use *strongly connected components* and a *weak* diameter notion ($\operatorname{dist}_G(u,v)\le D$ and $\operatorname{dist}_G(v,u)\le D$ for $u,v$ in the same SCC, measured in the original graph). A directed version was built by Bernstein–Probst-Gutenberg–Wulff-Nilsen (2020) for the *dynamic* setting (their subroutine "Partition"); it works on non-negative weights and is geared to maintenance under updates. Every application of these decompositions has been to problems without negative weights, and to the parallel/distributed/dynamic models rather than to classical sequential shortest paths.

## Baselines

**Bellman–Ford ($O(mn)$).** Relax all edges $n-1$ times. Handles negatives, detects negative cycles.

**Dijkstra ($O(m+n\log n)$).** Near-linear; correct when all edge weights are non-negative.

**Johnson reweighting (Johnson 1977).** Run Bellman–Ford once from a super-source to get $\phi(v)=\operatorname{dist}(s,v)$, then Dijkstra on $G_\phi$. Provides a clean reduction: with a non-negativizing price function $\phi$ in hand, a single Dijkstra suffices.

**Scaling algorithms — Gabow 1985, Gabow–Tarjan 1989, Goldberg 1995 ($O(m\sqrt n\log W)$).** Maintain an integral $\varepsilon$-feasible price function and halve $\varepsilon$ over $O(\log W)$ rounds; each round restores feasibility with a routine dominated by a graph-search/assignment subroutine costing $O(m\sqrt n)$.

**Continuous-optimization / min-cost-flow ($\tilde O((m+n^{1.5})\log W)$ dense; $m^{4/3+o(1)}\log W$ sparse).** Interior-point methods on the flow LP, accelerated by dynamic graph data structures.

**Planar/structured algorithms (Fakcharoenphol–Rao 2001; Mozes–Wulff-Nilsen 2010).** Near-linear algorithms for planar digraphs, using planar separators.

## Evaluation settings

The natural yardsticks (all pre-existing):
- **Inputs:** directed graphs with integer edge weights, possibly negative; both with and without negative cycles. Sparse ($m=\Theta(n)$) and dense regimes. Adversarial graphs that force many negative edges onto shortest paths. Standard road-network instances (e.g. DIMACS Shortest Path Challenge graphs such as the USA road networks), and synthetic graphs with negative weights injected (e.g. via random price perturbation, or by negating a random fraction of weights).
- **Correctness check:** the output tree must be a valid arborescence and a true shortest-path tree — verified by confirming no edge $(u,v)$ has $\operatorname{dist}(v) > \operatorname{dist}(u)+w(u,v)$ — or, in the cycle case, the returned cycle must have negative total weight in the input graph.
- **Metric:** asymptotic running time as a function of $m,n,W$ (and the realized constants/log-factors), measured against the $O(mn)$ Bellman–Ford baseline and the $O(m\sqrt n\log W)$ scaling baseline.
- **Model:** standard word-RAM / comparison-addition model; randomization allowed (Las Vegas preferred: always correct, fast with high probability).

## Code framework

The pre-existing primitives are: graphs with integer weights, Dijkstra on non-negative graphs, Bellman–Ford for the general case, and the price-function reweighting identity. Everything specific to the new method goes into one empty slot.

```python
import math, random
from dataclasses import dataclass

# ---- existing primitives -------------------------------------------------

@dataclass
class Graph:
    n: int
    adj: dict          # u -> list of (v, weight)   integer weights, may be negative
    def out_edges(self, u): return self.adj[u]
    def vertices(self):     return range(self.n)

def dijkstra(G, s):
    """Shortest-path tree from s. Correct ONLY when all weights >= 0. O(m + n log n)."""
    # standard binary-heap Dijkstra
    ...

def bellman_ford(G, s):
    """Distances from s, or report a negative cycle. O(m*n). The general but slow tool."""
    ...

def reweight(G, phi):
    """Return G_phi with w_phi(u,v) = w(u,v) + phi[u] - phi[v].
       Preserves shortest paths and cycle weights (Johnson 1977)."""
    H = Graph(G.n, {u: [] for u in G.vertices()})
    for u in G.vertices():
        for (v, w) in G.out_edges(u):
            H.adj[u].append((v, w + phi[u] - phi[v]))
    return H

# ---- the slot the method will occupy -------------------------------------

def make_nonnegative_price(G):
    """Given an integer-weighted digraph with no negative cycle, return an integral
       price function phi such that every reduced weight w_phi(e) >= 0.
       TODO: the procedure we will design goes here. With such a phi we can finish
             by a single Dijkstra on reweight(G, phi)."""
    pass  # TODO

def sssp(G, s):
    """Negative-weight SSSP: return a shortest-path tree from s, or a negative cycle."""
    phi = make_nonnegative_price(G)          # TODO: the contribution
    if phi is None:
        return ("negative_cycle", ...)       # TODO: witness extraction
    return dijkstra(reweight(G, phi), s)
```

The contribution is whatever fills `make_nonnegative_price` (and the witness path for the cycle case): a way to produce a non-negativizing integral price function, using only Dijkstra, Bellman–Ford, reweighting, and elementary graph operations.
