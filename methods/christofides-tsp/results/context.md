## Research question

We are given a finite set of cities `X` with a distance function `d` on every pair, and we want the
shortest closed tour that visits each city exactly once and returns to the start — a minimum-cost
Hamiltonian cycle. The distances are *metric*: for all `x, y, z`,

- `d(x, y) >= 0`, with `d(x, y) = 0` iff `x = y`;
- `d(x, y) = d(y, x)` (symmetry);
- `d(x, y) <= d(x, z) + d(z, y)` (triangle inequality).

Finding the exact optimum is NP-hard even in this metric special case, so an exact polynomial-time
method is out of reach unless P = NP. The goal is therefore an *approximation algorithm*:
one that runs in polynomial time and returns a valid tour whose length is provably within a fixed
constant factor `c` of the optimum, for every instance. The quality of such an algorithm is its
worst-case ratio `(cost of returned tour) / OPT`, and the whole problem is to make `c` as close to 1
as we can prove, with an algorithm that actually runs fast.

The defining difficulty is the proof, not the heuristic. A heuristic is easy to write; a *guarantee*
requires a quantity we can compute in polynomial time that provably *lower-bounds* OPT, so that we
can sandwich the returned tour between our construction (an upper bound) and that lower bound.

## Background

**The metric assumption is what makes guarantees possible.** Without the triangle inequality, no
constant-factor approximation can exist unless P = NP: a reduction from Hamiltonian cycle blows up
non-edges to an enormous weight, so any constant-factor tour would have to avoid them and thus
decide Hamiltonicity. The triangle inequality is exactly the property that lets us take *shortcuts*
without penalty: if a walk visits `... a, b, c ...` and we skip `b`, replacing `a -> b -> c` by the
direct edge `a -> c` cannot increase length, since `d(a, c) <= d(a, b) + d(b, c)`. Shortcutting is
the workhorse that converts any closed walk that covers all the cities into a Hamiltonian tour no
longer than the walk.

**Spanning trees and the MST lower bound.** A spanning tree connects all `n` cities with `n - 1`
edges and no cycle. The minimum spanning tree (MST) — the cheapest such tree — is computable in
polynomial time by Prim's or Kruskal's algorithm. The structural fact that makes the MST a *lower
bound on OPT*: delete any single edge `e` from an optimal Hamiltonian cycle `C*`. What remains is a
path through every city, hence a spanning tree, of cost `d(C*) - d(e)`. Since the MST is the cheapest
spanning tree,

```
d(MST) <= d(C*) - d(e) <= d(C*) = OPT.
```

The MST is the standard handle on OPT for this problem.

**Euler's theorem and parity.** A connected multigraph has a closed walk traversing every edge
exactly once — an Eulerian circuit — if and only if every vertex has even degree (Euler, 1736). The
*handshake lemma* says the sum of all vertex degrees equals twice the number of edges, hence is
even, so the number of odd-degree vertices in any graph is even. These two facts are the basic
parity tools.

**Matchings.** A matching is a set of edges no two of which share a vertex; a perfect matching on a
vertex set `S` pairs up all of `S` (requires `|S|` even). On a complete edge-weighted graph,
Edmonds (1965) gave the first polynomial-time algorithm for minimum/maximum-weight matching in
*general* (non-bipartite) graphs — the "blossom" algorithm — and implementations such as
Karzanov's `O(n^3 log n)` version made it concretely fast. Before 1965 there was no polynomial method
for general-graph weighted matching, so this is a relatively recent tool.

Matchings interact with cycles through parity: any cycle alternates structure along its edges, and
on an even cycle that alternation closes up consistently, whereas on an odd cycle it clashes when it
wraps around.

**The Chinese postman connection.** A closely related problem was being studied intensively in the
early 1970s: the Chinese postman problem — find a shortest closed walk traversing *every edge* of a
given graph. Edmonds and Johnson (1973), Christofides (1973), and Serdyukov (1974) all attack it.
The obstruction there is the same parity obstruction: a graph fails to have an Eulerian circuit
exactly because of its odd-degree vertices, and the work in these papers is about cheaply repairing
that parity defect so every edge can be traversed in one closed walk.

## Baselines

**Nearest-neighbour and greedy heuristics.** Building a tour by repeatedly walking to the closest
unvisited city, or by greedily adding cheapest edges, runs fast but does not supply the constant
worst-case ratio needed here. Useful in practice, useless for a constant-ratio proof.

**The twice-around (double-MST) 2-approximation.** This tree-based baseline already exploits the MST
lower bound. Build the MST `M`. Walk around it with a hand on the wall: this traverses each tree edge
once in each direction, a closed walk of cost exactly `2 d(M)` in which every vertex is visited and
(since each edge is doubled) has even degree, so the walk is Eulerian. Shortcut repeated vertices to
get a Hamiltonian tour `T`. Then

```
d(T) <= 2 d(M)            (shortcutting the doubled-tree walk, triangle inequality)
     <= 2 d(OPT)          (MST lower bound)
```

So `T` is within a factor of 2 of optimal. The factor of 2 comes *entirely* from doubling the whole
tree, and doubling serves a single purpose: to make every vertex's degree even. It is a blunt
instrument — it pays for a second full copy of the MST and changes the parity of every vertex at
once, whether or not a given vertex's degree was a problem to begin with. Whether the factor 2 of
this tree-based approach can be pushed below 2 is the open question.

## Evaluation settings

The natural yardstick is the worst-case approximation ratio over all metric instances: the supremum
of `(returned tour length) / OPT`. One also reports asymptotic running time as a function of the
number of cities `n`, since a guarantee is only useful if it is achieved in polynomial time. Test
inputs are complete graphs whose weights satisfy the triangle inequality, including Euclidean point
sets in the plane, shortest-path metrics of weighted graphs, and adversarial families designed to
stress worst-case behavior.

## Code framework

The available primitives are a metric represented as a complete weighted graph, a polynomial MST
routine, a polynomial weighted-matching routine for general graphs, an Eulerian-circuit extractor
for an even-degree connected multigraph, and the shortcutting operation. A tree-based tour builder
can use the same validation and shortcutting harness regardless of how the tree is made Eulerian.

```python
import networkx as nx
from networkx.utils import not_implemented_for


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def metric_tsp_tour(G, weight="weight", tree=None):
    # G: complete undirected graph, weights satisfy the triangle inequality.
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

    if tree is None:
        tree = nx.minimum_spanning_tree(G, weight=weight)

    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(tree.edges)

    # TODO: add edges from G so every vertex degree in multigraph becomes even at low cost.

    return _shortcutting(nx.eulerian_circuit(multigraph))


def _shortcutting(circuit):
    # Walk the Eulerian circuit; on revisiting a city, skip it and go to the next new one.
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
