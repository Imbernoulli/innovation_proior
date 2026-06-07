# The Christofides-Serdyukov algorithm for metric TSP

## Problem

Given a complete graph on `n` cities with a metric distance function `d` (nonnegative, symmetric,
triangle inequality `d(x,y) <= d(x,z) + d(z,y)`), find a minimum-cost Hamiltonian tour. The exact
problem is NP-hard, so exact polynomial time is not expected unless P = NP. The algorithm returns,
in `O(n^3)` time, a tour of length at most `3/2` times the optimum `OPT`.

## Key idea

Two computable handles on `OPT` drive the analysis:

1. **MST lower bound.** Deleting one edge from any Hamiltonian tour leaves a spanning tree, so the
   minimum spanning tree satisfies `d(MST) <= OPT`.
2. **Parity correction by matching, not by doubling.** To walk a tree as a closed tour you need an
   Eulerian (all-even-degree) structure. Doubling every MST edge achieves that but costs `2 d(MST)`
   (ratio 2). Instead, fix parity *only where it is broken*: add a minimum-weight perfect matching on
   the odd-degree vertices `O` of the MST. Each added edge flips the parity of its two endpoints, so
   matching `O` turns every odd vertex even and leaves the even ones untouched. `|O|` is even by the
   handshake lemma, so the matching exists.

The matching is cheap: `d(matching) <= OPT/2`. Hence the Eulerian tour of `MST + matching` costs
`d(MST) + d(matching) <= OPT + OPT/2 = (3/2) OPT`, and shortcutting it into a Hamiltonian tour does
not increase the cost (triangle inequality).

## Algorithm

1. Compute a minimum spanning tree `T` of `G`.
2. Let `O` be the set of odd-degree vertices of `T` (`|O|` is even).
3. Compute a minimum-weight perfect matching `M` on the complete graph over `O`.
4. Form the multigraph `H = T ∪ M`; every vertex of `H` has even degree and `H` is connected.
5. Find an Eulerian circuit of `H`.
6. Shortcut repeated vertices to obtain a Hamiltonian tour.

## Analysis (ratio 3/2)

- **`d(T) <= OPT`:** the optimal tour minus an edge is a spanning tree, no cheaper than the MST.
- **`d(M) <= OPT/2`:** list the odd-degree vertices `O` in the order the optimal tour visits them.
  Shortcut each segment between consecutive `O`-vertices to get a cycle `C_O` on `O` with
  `d(C_O) <= OPT`. Since `|O|` is even, alternating the edges of `C_O` splits it into two perfect
  matchings `N1, N2` with `d(N1) + d(N2) = d(C_O)`. So
  `d(M) <= min(d(N1), d(N2)) <= d(C_O)/2 <= OPT/2`.
- **Combine:** the Eulerian circuit of `H` uses each edge once, costing `d(T) + d(M) <= (3/2) OPT`;
  shortcutting preserves the bound.

Running time is `O(n^3)`, dominated by the minimum-weight perfect matching (Edmonds' blossom
algorithm).

## Code

```python
import networkx as nx
from networkx.utils import not_implemented_for


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def christofides(G, weight="weight", tree=None):
    """3/2-approximation for metric TSP on a complete graph with the triangle inequality."""

    # Strip self-loops; require a complete graph.
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

    # 2. Restrict to the odd-degree vertices of the tree (|O| even by the handshake lemma).
    L = G.copy()
    L.remove_nodes_from([v for v, degree in tree.degree if not (degree % 2)])

    # 3. Minimum-weight maximum-cardinality matching on O. min_weight_matching sets
    #    new_weight = (max_edge_weight + 1) - original_weight and calls max_weight_matching
    #    with maxcardinality=True; on complete even O this is perfect.
    MG = nx.MultiGraph()
    MG.add_edges_from(tree.edges)
    edges = nx.min_weight_matching(L, weight=weight)
    MG.add_edges_from(edges)             # tree + matching => every vertex even degree

    # 4-6. Eulerian circuit of the even-degree multigraph, then shortcut to a Hamiltonian tour.
    return _shortcutting(nx.eulerian_circuit(MG))


def _shortcutting(circuit):
    """Walk the Eulerian circuit; skip already-seen cities, going to the next new one."""
    nodes = []
    for u, v in circuit:
        if v in nodes:
            continue
        if not nodes:
            nodes.append(u)
        nodes.append(v)
    nodes.append(nodes[0])
    return nodes
```
