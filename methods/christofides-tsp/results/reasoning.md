OK, let me think this through from scratch. I have `n` cities, all pairwise distances, and the
distances are metric — nonnegative, symmetric, and they obey `d(x,y) <= d(x,z) + d(z,y)`. I want the
shortest tour that visits every city once and comes home. Finding it exactly is hopeless in
polynomial time unless P = NP. So I'm not chasing the optimum; I'm chasing a *guarantee*: a tour I can
build fast and prove is never more than some fixed factor `c` times the optimal length `OPT`, on every
instance.

And right away I notice that the hard part isn't producing *a* tour — any ordering of the cities is a
tour. The hard part is the inequality. To say "my tour `<= c * OPT`" I need a number I can actually
compute that is provably `<= OPT`, because I can never compute `OPT` itself. So the real question is:
what lower bound on `OPT` do I have that's also polynomial-time?

Let me stare at the structure of a tour. A tour is a Hamiltonian cycle — `n` edges, every vertex
degree 2, connected. Now do something violent to an optimal tour `C*`: delete one edge `e`. What's
left is a path through all `n` cities — `n - 1` edges, connected, no cycle. That's a spanning tree.
Its cost is `d(C*) - d(e)`, which is certainly at most `d(C*) = OPT`. The cheapest spanning tree, the
MST, is no more expensive than that one, so

    d(MST) <= d(C*) - d(e) <= d(C*) = OPT.

And the MST I *can* compute in polynomial time, Prim or Kruskal. So there it is — my lower-bound
handle on `OPT` is the MST. Everything I prove from here is going to be measured against `d(MST)`,
and since `d(MST) <= OPT`, bounds in terms of `d(MST)` become bounds in terms of `OPT` for free.

Good. Now the other direction — I need to manufacture an actual tour out of the MST, and bound its
cost from above. The MST is a tree; it's not a tour, it has leaves, it has branch points. How do I
turn a tree into something I can walk as a closed loop? Here's where the triangle inequality earns
its keep. Suppose I have *any* closed walk that visits every city — it's allowed to repeat cities,
go back and forth, whatever — as long as it's a single closed route covering all of `X`. I can
collapse it into a genuine tour: walk along it, and whenever the next city is one I've already
visited, skip it and head straight to the next *new* city. Replacing `... a -> b -> c ...` (where I
skip `b`) by the direct hop `a -> c` can't cost more, because `d(a,c) <= d(a,b) + d(b,c)`. So
*shortcutting never increases length*. That means: if I can build a cheap closed walk covering all
cities, I can convert it to a Hamiltonian tour no more expensive than the walk. The whole game
reduces to building a cheap covering walk.

So how do I get a closed covering walk out of the tree? A tree by itself has no closed walk that
uses each edge once — its vertices have all sorts of degrees, and to *traverse edges* in a single
closed loop I need an Eulerian structure. Let me recall the Euler condition: a connected multigraph
has a closed walk using every edge exactly once precisely when every vertex has even degree. My tree
flunks that badly — every leaf has degree 1, which is odd.

The standard fix, the one everyone reaches for: double every edge of the tree. Walk around the tree
keeping your hand on the wall; you go down each edge and back up, traversing each edge twice. Now
every vertex's degree is exactly doubled, hence even, the doubled tree is connected, so an Eulerian
circuit exists. Its cost is exactly `2 d(MST)`, because I used every tree edge twice. Shortcut it to
a tour `T`, and

    d(T) <= 2 d(MST) <= 2 d(OPT).

A factor-2 approximation, clean. I could stop here. But let me look hard at where that 2 comes from,
because it feels wasteful.

The entire factor of 2 is the doubling. And *why* did I double? Only to make every degree even. That
was the one and only job: satisfy Euler's parity condition. But look — most vertices of the tree were
already fine, or rather, doubling was overkill for them. I paid for a *complete second copy of the
whole MST* just to fix parity. That can't be necessary. Doubling fixes the parity of *every* vertex,
including ones I didn't need to touch.

Let me get precise about which vertices are actually broken. In the original tree `M`, a vertex needs
fixing iff its degree is odd (Euler wants all-even). Call `O` the set of odd-degree vertices of `M`.
The even-degree vertices of `M` are already legal; I should leave them completely alone. The leaves
(degree 1) are in `O`, and maybe some internal branch points too. Doubling flipped the parity of
everybody; what I actually want is to flip the parity of *only the vertices in `O`*, and pay as
little as possible to do it.

So reframe the problem: I have the tree `M`, with its even-degree vertices happy and its odd set `O`
unhappy. I want to add some edges so that exactly the vertices in `O` get their degree parity
flipped (odd to even) and the even ones stay even. What set of added edges flips the parity of
exactly a prescribed set of vertices? An edge `{u, v}` flips the parity of `u` and of `v`. So if I
add a collection of edges in which each vertex of `O` is an endpoint an *odd* number of times and
each non-`O` vertex an *even* number of times... the cleanest such collection is one where every
vertex of `O` is hit *exactly once* and nobody else is hit at all. That's a set of edges pairing up
the vertices of `O` with no shared endpoints — a *perfect matching on `O`*. Add a perfect matching on
`O` to `M`: every vertex in `O` gains exactly 1 to its degree (odd becomes even), every vertex not in
`O` gains 0 (stays even). The result is all-even. Exactly the surgical fix I wanted.

Wait — does a perfect matching on `O` even exist? It needs `|O|` to be even. Is it? Sum over all
vertices of `deg(v)` counts each edge twice, so it's `2 |E|`, an even number. The even-degree
vertices contribute an even total; for the grand total to be even, the odd-degree vertices must be
even in *count*. So `|O|` is even. A perfect matching on `O` exists. (This is the handshake lemma, and
it's the little fact that makes the whole idea legal.)

Now, *which* perfect matching? I want the cheapest one, to keep the added cost small — the
minimum-weight perfect matching `P` on the complete graph over the vertices of `O`. Can I compute
that in polynomial time? Yes — Edmonds' blossom algorithm does minimum-weight perfect matching on a
general (non-bipartite) graph in polynomial time. So the step is constructive.

And actually I've seen this exact move before, in a different problem. The Chinese postman problem —
traverse every edge of a graph by the shortest closed walk — has precisely this shape: the graph
fails to be Eulerian only because of its odd-degree vertices, and the early solutions patch that
failure by adding a minimum-weight matching on the odd-degree vertices to restore parity. I'm
transplanting that device: there it patched a graph so every edge could be traversed; here it patches
the MST so a covering walk exists. Same parity surgery, new host.

So the construction is: take `M`, take a minimum-weight perfect matching `P` on its odd-degree
vertices `O`, throw the edges of `P` together with the edges of `M` into one multigraph `H` (if an
edge happens to be in both, keep both copies). Every vertex of `H` is now even degree, and `H`
contains `M` so it's connected. Euler gives me a closed walk using every edge of `H` exactly once,
of cost `d(M) + d(P)`. Shortcut it to a Hamiltonian tour. The tour costs at most `d(M) + d(P)`.

Now the moment of truth — what is this worth? I have `d(M) <= d(OPT)` already. If I can show
`d(P)` is small relative to `OPT`, I win. Suppose, optimistically, `d(P) <= d(OPT)/2`. Then the tour
is `<= d(OPT) + d(OPT)/2 = (3/2) d(OPT)`. That would be `3/2`, a real improvement on 2. But that's a
*hope*; I haven't earned the `d(P) <= d(OPT)/2`. Let me see if it's actually true, because the bound
on the matching is the entire ballgame.

`P` is the *minimum* perfect matching on `O`. To bound a minimum, I just need to exhibit *some*
perfect matching on `O` that's cheap; `P` is at most that. Where do I find a cheap perfect matching
on `O`? I should mine it out of `OPT` itself, since `OPT` is the thing I'm comparing against.

List the vertices of `O` in the order the optimal tour visits them: `o_1, o_2, ..., o_k`, wrapping
around from `o_k` back to `o_1`. Between `o_i` and `o_{i+1}` on the optimal tour there may be a whole
path through cities not in `O`. Replace that whole path by the direct edge `{o_i, o_{i+1}}`. The
triangle inequality says the direct edge costs no more than the path it replaces. Do that for every
gap between consecutive odd vertices. I get a cycle `C_O` on exactly the vertices in `O`, and its
cost is at most the cost of the optimal tour:

    d(C_O) <= d(OPT).

Now `k = |O|` is even, so `C_O` has an even number of edges. Color its edges alternately: odd-position
edges one color, even-position edges the other. Because the cycle length is even, the colors match
when I wrap around. Each color class touches every vertex of `O` exactly once: every vertex of the
cycle has two incident cycle edges, one of each color. So each color class is a perfect matching on
`O`. Call them `N1` and `N2`. Together they partition the cycle edges, so

    d(N1) + d(N2) = d(C_O).

One of the two is at most the average:

    min(d(N1), d(N2)) <= (d(N1) + d(N2))/2 = d(C_O)/2.

And `P`, the minimum perfect matching on `O`, is no more expensive than either `N1` or `N2` — they're
both perfect matchings on `O`, and `P` is the cheapest:

    d(P) <= min(d(N1), d(N2)) <= d(C_O)/2 <= d(OPT)/2.

There it is. The `d(P) <= d(OPT)/2` I was hoping for is real, and the reason is genuinely pretty: the
optimal tour, after shortcutting onto the even set `O`, is two perfect matchings stacked together.
The cheaper one costs at most half that shortcut cycle, and my minimum matching beats even that.
The factor of 1/2 wasn't an assumption; it fell out of splitting an even cycle into two matchings.

Let me make sure I didn't cheat on the parity of the split. The cycle `C_O` has an even number of
edges precisely because `|O|` is even; if `|O|` were odd the alternating 2-coloring would clash when
it wrapped around and the color classes wouldn't be matchings. But `|O|` *is* even by the handshake
lemma — the same fact that let `P` exist in the first place. The two places I needed `|O|` even are
the same fact. Good, consistent.

So assemble the bound end to end. Tour cost `<= d(H) = d(M) + d(P) <= d(OPT) + d(OPT)/2 =
(3/2) d(OPT)`. The doubling's factor of 2 became `1 + 1/2`: the `1` is the MST (which I had to pay
either way, it's the lower bound itself), and the `1/2` is the matching that surgically fixes parity
only where it's broken — instead of doubling, I paid for half a tour's worth of matching. That's the
whole improvement, and it's exactly the slack I spotted in the doubling argument: I stopped paying to
fix vertices that were never broken.

Let me also pin down the running time, because a guarantee with an exponential algorithm is worthless.
MST is cheap. Shortcutting and the Eulerian circuit are linear in the edges. The expensive step is the
minimum-weight perfect matching on `O` — Edmonds' blossom algorithm, polynomial, roughly `O(n^3)` with
a good implementation. So the whole thing is `O(n^3)`, dominated by the matching. Polynomial, done.

Now let me write it as real code. I keep the NetworkX harness: strip self-loops, check that the graph
is complete, compute the MST if the caller didn't pass one, keep only the odd-degree tree vertices in
a copy of `G`, match them, put tree edges and matching edges into a `MultiGraph`, take an Eulerian
circuit, and shortcut it.

```python
import networkx as nx
from networkx.utils import not_implemented_for


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def christofides(G, weight="weight", tree=None):
    """3/2-approximation for metric TSP on a complete graph G with the triangle inequality."""

    # Strip self-loops; require a complete graph (every vertex adjacent to every other).
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

    # Step 1: the MST. d(tree) <= OPT, our lower-bound handle.
    if tree is None:
        tree = nx.minimum_spanning_tree(G, weight=weight)

    # Step 2: O = the odd-degree vertices of the tree -- the only ones whose parity is broken.
    # Restrict G to O so we can match within it. (|O| is even by the handshake lemma,
    # so a perfect matching exists.)
    L = G.copy()
    L.remove_nodes_from([v for v, degree in tree.degree if not (degree % 2)])

    # Step 3: minimum-weight maximum-cardinality matching on O. min_weight_matching
    # sets new_weight = (max_edge_weight + 1) - original_weight and calls
    # max_weight_matching with maxcardinality=True; on complete even O this is perfect.
    # The even-cycle split above gives d(matching) <= OPT/2.
    MG = nx.MultiGraph()
    MG.add_edges_from(tree.edges)          # the tree
    edges = nx.min_weight_matching(L, weight=weight)
    MG.add_edges_from(edges)               # + the parity-fixing matching => all even degree

    # Step 4 & 5: Eulerian circuit of the even-degree connected multigraph,
    # then shortcut repeated vertices into a Hamiltonian tour (triangle inequality keeps it cheap).
    return _shortcutting(nx.eulerian_circuit(MG))


def _shortcutting(circuit):
    """Walk the Eulerian circuit; skip already-seen cities, going straight to the next new one."""
    nodes = []
    for u, v in circuit:
        if v in nodes:
            continue
        if not nodes:
            nodes.append(u)
        nodes.append(v)
    nodes.append(nodes[0])     # close the tour
    return nodes
```
