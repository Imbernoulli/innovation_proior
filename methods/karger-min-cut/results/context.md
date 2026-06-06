# Context — Finding a global minimum cut of an undirected graph

## Research question

Given a connected undirected graph G = (V, E) on n = |V| vertices and m = |E| edges (possibly a weighted multigraph), find a **global minimum cut**: a partition of V into two nonempty sides (S, V∖S) that minimizes the number of edges crossing between them (or, in the weighted case, their total weight). "Global" means we are *not* given a source s and sink t — we want the cut of least value over *all* ways of splitting the vertices into two groups.

The cut value of a graph measures its weakest point: the fewest edges whose removal disconnects it. It is the basic measure of network reliability (how many links must fail before the network splits), the cost of the cheapest way to two-color a system into communicating clusters, and a building block in graph partitioning, image segmentation, and clustering. The pain point is concrete: the classical route to a global min cut is to *reduce it to flow*. A single s–t min cut equals a single max flow (max-flow/min-cut), but a global cut has no fixed s and t, so one must consider all the ways the two sides could fall. Naively that is C(n,2) source–sink pairs; the standard improvement fixes one vertex as a perpetual source and runs n−1 max-flow computations against it (any global min cut separates that fixed vertex from *something*). Either way the cost is dominated by repeated max-flow, on the order of O(mn) per flow with the better algorithms and O(mn²) or worse overall — heavy, and built on the full machinery of augmenting paths, residual graphs, and flow data structures. A solution worth having would: avoid flow entirely; be elementary enough to state in two lines; and run in time competitive with, or better than, the flow-based bounds, at least with high probability.

## Background

**The handshake identity and the single-vertex cut.** Two elementary facts about any undirected graph bound the min cut from above. First, summing degrees double-counts edges: Σ_{u} deg(u) = 2m, so the average degree is 2m/n. Second, isolating a single vertex u on one side gives a valid cut of value exactly deg(u). Therefore the global min-cut value k satisfies k ≤ deg(u) for *every* u, hence k ≤ (average degree) = 2m/n. Rearranged, this says any graph whose min cut is k must be reasonably dense: m ≥ nk/2. These are the only structural facts the randomized approach will lean on, and they are knowable from the degree sequence alone.

**Cuts are sparse relative to the whole graph.** The same inequality, read the other way, says the cut edges are a *small fraction* of all edges: at most k of the m ≥ nk/2 edges cross a given min cut, so a uniformly random edge crosses it with probability at most k/m ≤ 2/n. As n grows, an arbitrary edge is overwhelmingly *internal* to one side of any fixed min cut. This sparsity — "the min cut is a needle, the bulk of the graph is hay" — is the latent structure that a method could try to exploit, but flow-based algorithms make no use of it.

**Merging vertices as a decision primitive.** Searching for a cut is, operationally, deciding for each pair of vertices whether they land on the same side or opposite sides. There is an operation that *commits* to "same side": **edge contraction**. Contracting an edge {u, v} replaces u and v with a single supernode whose incident edges are the union of u's and v's; any edge that ran between u and v becomes a self-loop and is discarded, while edges to a common neighbor w become **parallel edges** and are kept. Contraction shrinks the vertex set by one and turns a graph into a *multigraph* (parallel edges carry the multiplicity that records how many original edges run between two supernodes). Crucially, a cut of the contracted multigraph corresponds exactly to a cut of G that keeps every contracted pair together — so contraction never *invents* small cuts: the min cut of the contracted graph is always ≥ the min cut of G, and equals it precisely when no contracted edge crossed the original min cut.

**Spanning-tree machinery already exists.** Minimum spanning tree algorithms (Kruskal's, with union-find) process edges in weight order and merge components, which is structurally the same "commit two vertices to the same side" move as contraction. Decades of optimization have made union-find and MST near-linear. Any cut method phrased in terms of edge-by-edge merging could in principle borrow this machinery.

## Baselines

**Flow-based global min cut (Gomory–Hu; Hao–Orlin).** Compute a global min cut by reduction to s–t min cuts. Fix an arbitrary vertex s; for each other vertex t, the global min cut either separates s from t (caught by the s–t min cut) or does not — so computing the s–t min cut for all n−1 choices of t and taking the smallest yields the global min cut. Each s–t min cut is a max-flow computation. The Gomory–Hu tree computes all-pairs min cuts with n−1 flow computations; Hao–Orlin (1992) amortizes the n−1 flows into roughly the cost of a single parametric max-flow, O(mn log(n²/m)). Core idea is exact and deterministic. Its gap: it is entirely built on flow — residual graphs, augmenting paths or push–relabel, and their data structures — and it makes no use of the fact that the cut is a tiny fraction of the graph. The cost is tied to the cost of flow, which on dense graphs is large, and the algorithm is intricate.

**Deterministic contraction — the Stoer–Wagner / Nagamochi–Ibaraki line.** A min cut can be found *without* flow by repeated "maximum adjacency" (legal) orderings: build an ordering of vertices where each next vertex is the most strongly connected to those already chosen; the last two vertices in the ordering form a min "s–t" cut for that s,t, which is then contracted, and the process repeats n−1 times. Each phase is O(m + n log n) with a heap, giving O(mn + n² log n) overall, deterministic, flow-free. Core idea: a clever *deterministic* ordering guarantees the last edge examined is a genuine s–t min cut, so contracting it is safe. Its gap: the ordering must be recomputed from scratch every phase, the analysis is delicate, and the n−1 phases give an inherently O(mn)-ish cost; it does not exploit cut sparsity to go below that, and it is not obvious how to recurse to do better.

**The uniform-random-edge heuristic.** Repeatedly pick a *random* edge and contract it until two supernodes remain, then read off the cut between them. Cheap and elementary, and intuitively appealing because a random edge tends to come from dense, tightly-knit regions and so tends to merge vertices that *belong* together. Its gap as a baseline: stated as a one-shot heuristic it has no guarantee — one unlucky contraction of a cut edge ruins the run — and no analysis quantifying how often it succeeds or how to amplify it into a reliable algorithm. Turning this heuristic into a method with a provable success probability, and then making it fast, is exactly the open problem.

## Evaluation settings

The natural yardstick is correctness probability against the true global min cut, and running time, as functions of n and m. Test instances are connected undirected (multi)graphs: random graphs at various densities; graphs with a planted sparse cut (e.g. two dense clusters joined by a few bridge edges, so the min-cut value is known by construction); and standard benchmark graphs used for partitioning. The metrics are: the empirical fraction of independent runs that recover a cut of the true minimum value (compared against the analytic per-run success bound), the number of repetitions needed to reach a target overall failure probability (e.g. 1/n or 1/poly(n)), and wall-clock / asymptotic running time versus the flow-based and deterministic-contraction baselines. Memory is O(m) for the adjacency multigraph. No outcomes are reported here — these are the conditions under which a method would be measured.

## Code framework

The primitives that already exist: an adjacency-multigraph representation (a map from supernode to a list of incident supernodes, with parallel edges as repeats and no self-loops), a uniform-random choice over edges, the contraction operation that merges two supernodes, a routine to read off the crossing-edge count when two supernodes remain, and an outer loop that repeats a randomized run and keeps the best. The open slots are the policy that drives a single run down to a target number of supernodes, and the control structure that turns one cheap-but-unreliable run into a reliable algorithm.

```python
import copy
import math
import random


def contract(graph, t):
    """Drive the multigraph down to t supernodes by merging edges.

    graph: {node: [neighbor, ...]} multigraph (parallel edges = repeats,
    self-loops never stored). Returns the contracted multigraph.
    """
    g = copy.deepcopy(graph)
    while len(g) > t:
        u, w = _choose_edge(g)        # TODO: which edge to merge
        _merge(g, u, w)               # commit u,w to the same side
    return g


def _choose_edge(g):
    # TODO: the rule for selecting the edge to contract
    pass


def _merge(g, u, w):
    # absorb w into u; redirect w's edges, drop the u-w self-loops
    for x in g[w]:
        if x != u:
            g[u].append(x)
    for x in g[w]:
        g[x].remove(w)
        if x != u:
            g[x].append(u)
    del g[w]


def cut_value(g):
    """Crossing-edge count once two supernodes remain."""
    return len(g[next(iter(g))])


def min_cut(graph):
    # TODO: the control structure that turns single runs into a
    # reliable global-min-cut algorithm (how far each run contracts,
    # how runs are combined / repeated to amplify the success probability)
    pass
```
