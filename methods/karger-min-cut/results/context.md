# Context — Finding a global minimum cut of an undirected graph

## Research question

Given a connected undirected graph G = (V, E) on n = |V| vertices and m = |E| edges (possibly a weighted multigraph), find a **global minimum cut**: a partition of V into two nonempty sides (S, V∖S) that minimizes the number of edges crossing between them (or, in the weighted case, their total weight). "Global" means we are *not* given a source s and sink t — we want the cut of least value over *all* ways of splitting the vertices into two groups. The deliverable is a single self-contained C++17 program that reads the graph from stdin and writes the minimum-cut value to stdout.

The cut value of a graph measures its weakest point: the fewest edges whose removal disconnects it. It is the basic measure of network reliability (how many links must fail before the network splits), the cost of the cheapest way to two-color a system into communicating clusters, and a building block in graph partitioning, image segmentation, and clustering. The question is how to find a global minimum cut efficiently.

## Background

**The handshake identity and the single-vertex cut.** Two elementary facts about any undirected graph bound the min cut from above. First, summing degrees double-counts edges: Σ_{u} deg(u) = 2m, so the average degree is 2m/n. Second, isolating a single vertex u on one side gives a valid cut of value exactly deg(u). Therefore the global min-cut value k satisfies k ≤ deg(u) for *every* u, hence k ≤ (average degree) = 2m/n. Rearranged, this says any graph whose min cut is k must be reasonably dense: m ≥ nk/2. These structural facts are knowable from the degree sequence alone.

**Cuts are sparse relative to the whole graph.** The same inequality, read the other way, says the cut edges are a *small fraction* of all edges: at most k of the m ≥ nk/2 edges cross a given min cut, so a uniformly random edge crosses it with probability at most k/m ≤ 2/n. As n grows, an arbitrary edge is overwhelmingly *internal* to one side of any fixed min cut.

**Edge contraction.** A standard graph operation, used for instance inside the deterministic-contraction baseline below. Contracting an edge {u, v} replaces u and v with a single supernode whose incident edges are the union of u's and v's; any edge that ran between u and v becomes a self-loop and is discarded, while edges to a common neighbor w become **parallel edges** and are kept. Contraction shrinks the vertex set by one and turns a graph into a *multigraph* (parallel edges carry the multiplicity that records how many original edges run between two supernodes).

**Spanning-tree machinery already exists.** Minimum spanning tree algorithms (Kruskal's, with union-find) process edges in weight order and merge components as they go. Decades of optimization have made union-find and MST near-linear.

## Baselines

**Flow-based global min cut (Gomory–Hu; Hao–Orlin).** Compute a global min cut by reduction to s–t min cuts. Fix an arbitrary vertex s; for each other vertex t, the global min cut either separates s from t (caught by the s–t min cut) or does not — so computing the s–t min cut for all n−1 choices of t and taking the smallest yields the global min cut. Each s–t min cut is a max-flow computation. The Gomory–Hu tree computes all-pairs min cuts with n−1 flow computations; Hao–Orlin (1992) amortizes the n−1 flows into roughly the cost of a single parametric max-flow, O(mn log(n²/m)). The method is exact and deterministic, built on residual graphs, augmenting paths or push–relabel, and their data structures.

**Deterministic contraction — the Stoer–Wagner / Nagamochi–Ibaraki line.** A min cut can be found *without* flow by repeated "maximum adjacency" orderings: build an ordering of vertices where each next vertex is the most strongly connected to those already chosen; the last two vertices in the ordering form a min "s–t" cut for that s,t, which is then contracted, and the process repeats n−1 times. Each phase is O(m + n log n) with a heap, giving O(mn + n² log n) overall, deterministic, flow-free. A clever deterministic ordering guarantees the last edge examined is a genuine s–t min cut, so contracting it is safe.

**The uniform-random-edge heuristic.** Repeatedly pick a *random* edge and contract it until two supernodes remain, then read off the cut between them. Cheap and elementary: a random edge tends to come from dense, tightly-knit regions and so tends to merge vertices that belong together.

## Evaluation settings

The natural yardstick is correctness probability against the true global min cut, and running time, as functions of n and m. Test instances are connected undirected (multi)graphs: random graphs at various densities; graphs with a planted sparse cut (e.g. two dense clusters joined by a few bridge edges, so the min-cut value is known by construction); and standard benchmark graphs used for partitioning. The metrics are: the empirical fraction of independent runs that recover a cut of the true minimum value, the number of repetitions needed to reach a target overall failure probability (e.g. 1/n or 1/poly(n)), and wall-clock / asymptotic running time versus the flow-based and deterministic-contraction baselines. Memory is O(m) for the adjacency multigraph. These are the conditions under which a method would be measured.

## Input-output contract

Write a single self-contained C++17 program. It reads `n m` from stdin, followed by `m` lines `u v` giving 1-based endpoints of undirected edges, and writes one integer to stdout: the value of a global minimum cut. Vertex labels fit in `int`; the crossing-edge count should be stored in `long long`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, m;
    if (!(cin >> n >> m)) return 0;

    vector<pair<int, int>> edges;
    edges.reserve(static_cast<size_t>(m));
    for (long long i = 0; i < m; i++) {
        long long u, v;
        cin >> u >> v;
        edges.push_back({static_cast<int>(u), static_cast<int>(v)});
    }

    long long result = 0;
    // TODO: compute the required value from n and edges.

    cout << result << "\n";
    return 0;
}
```
