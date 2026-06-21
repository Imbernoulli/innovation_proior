Metric TSP asks for the shortest closed tour that visits every city exactly once and returns to the start, given that all pairwise distances satisfy the triangle inequality. The exact problem is NP-hard, so the goal is a polynomial-time approximation whose worst-case tour length is provably within a constant factor of the optimum `OPT`. The standard starting point is the minimum spanning tree: deleting any edge from an optimal tour leaves a spanning tree, so `d(MST) <= OPT`. The simplest way to turn the MST into a tour is to double every tree edge, walk around the doubled tree, and shortcut repeated vertices. This gives a factor-2 approximation, but it is wasteful: the only reason for doubling is to make every vertex have even degree so that an Eulerian circuit exists, yet doubling pays for a complete second copy of the tree and changes the parity of vertices that were already fine.

The shortcoming is therefore not the lower bound but the upper-bound construction. We need a cheaper way to make the MST Eulerian. The key observation is that only the odd-degree vertices of the MST are parity defects; every even-degree vertex already satisfies Euler's condition. If we could add edges that flip the parity of exactly those odd-degree vertices and nothing else, the resulting multigraph would have all even degrees and would still be connected, but we would pay only for the parity repair rather than for a second full spanning tree.

The Christofides-Serdyukov algorithm does exactly this. Start by computing a minimum spanning tree `T`. Let `O` be the set of odd-degree vertices of `T`. By the handshake lemma, the sum of all degrees is even, so the number of odd-degree vertices is even and a perfect matching on `O` exists. Add to `T` a minimum-weight perfect matching `M` on the complete graph induced by `O`. Each matching edge raises the degree of its two endpoints by one, turning every odd vertex even while leaving even vertices untouched. The multigraph `H = T ∪ M` is connected and all its vertices have even degree, so `H` has an Eulerian circuit. Shortcutting repeated vertices along that circuit yields a valid Hamiltonian tour, and shortcutting cannot increase length because the distances are metric.

Why is this a `3/2`-approximation? The MST part already satisfies `d(T) <= OPT`. For the matching, list the vertices of `O` in the order they appear along an optimal tour and shortcut the paths between consecutive `O`-vertices. This produces a cycle `C_O` on `O` whose length is at most `OPT`. Because `|O|` is even, `C_O` has an even number of edges and can be partitioned into two perfect matchings on `O`. The cheaper of these two matchings costs at most half of `d(C_O)`, hence at most `OPT/2`. The minimum-weight perfect matching `M` is no more expensive than either of them, so `d(M) <= OPT/2`. Adding the two bounds gives `d(T) + d(M) <= (3/2) OPT`, and the Eulerian circuit uses each of those edges exactly once. Shortcutting only helps, so the final tour is also at most `3/2` times optimal.

The running time is polynomial: the MST is computed greedily in `O(n^2 log n)` or `O(n^2)` time depending on the implementation, and the bottleneck is the minimum-weight perfect matching on `O`, which Edmonds' blossom algorithm solves in roughly `O(n^3)` time for a complete graph. Shortcutting and extracting the Eulerian circuit are linear in the number of edges. The whole procedure is therefore `O(n^3)`.

```python
import networkx as nx
from networkx.utils import not_implemented_for


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def christofides(G, weight="weight", tree=None):
    """3/2-approximation for metric TSP on a complete graph with the triangle inequality."""

    # Strip self-loops and require a complete graph.
    loop_nodes = nx.nodes_with_selfloops(G)
    try:
        node = next(loop_nodes)
    except StopIteration:
        pass
    else:
        G = G.copy()
        G.remove_edge(node, node)
        G.remove_edges_from((n, n) for n in loop_nodes)
    N = len(G) - 1
    if any(len(nbrdict) != N for n, nbrdict in G.adj.items()):
        raise nx.NetworkXError("G must be a complete graph.")

    # 1. Minimum spanning tree: d(tree) <= OPT.
    if tree is None:
        tree = nx.minimum_spanning_tree(G, weight=weight)

    # 2. Restrict to the odd-degree vertices of the tree (|O| is even by the handshake lemma).
    L = G.copy()
    L.remove_nodes_from([v for v, degree in tree.degree if not (degree % 2)])

    # 3. Minimum-weight maximum-cardinality matching on O. On a complete even-sized set this is perfect.
    MG = nx.MultiGraph()
    MG.add_edges_from(tree.edges)
    edges = nx.min_weight_matching(L, weight=weight)
    MG.add_edges_from(edges)          # tree + matching => every vertex has even degree

    # 4-6. Eulerian circuit, then shortcut to a Hamiltonian tour.
    return _shortcutting(nx.eulerian_circuit(MG))


def _shortcutting(circuit):
    """Walk the Eulerian circuit and skip already-visited cities."""
    nodes = []
    for u, v in circuit:
        if v in nodes:
            continue
        if not nodes:
            nodes.append(u)
        nodes.append(v)
    nodes.append(nodes[0])            # close the tour
    return nodes
```
