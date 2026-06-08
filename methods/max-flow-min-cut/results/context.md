# Context: maximum flow and minimum cut in a capacitated network

## Research question

A network of cities is connected by rail links. Each link can carry only so much traffic
per day — its *capacity*. Pick two cities, a *source* and a *sink*. How much traffic can be
pushed, in steady state, from source to sink, when the flow may split across many parallel
routes and recombine? This is the maximum-flow problem, posed in the mid-1950s by T. E.
Harris as a model of rail traffic.

The same network carries a dual question, and it is the one the original military study
actually cared about. If you wanted to *stop* all traffic from source to sink by destroying
links, which links would you destroy, and what is the least total capacity you must remove?
A set of links whose removal disconnects source from sink is a *cut*; its cost is the sum of
the capacities removed. The military framing (rail interdiction) asks for the cheapest cut —
"the bottleneck."

These two problems pull in opposite directions: one party wants to move as much as possible,
the other wants to block all of it as cheaply as possible. The pain point is twofold. First,
there is no clean way to *compute* either optimum on a real network of a hundred-odd links
short of setting up a large linear program. Second — and this is the deeper gap — even if a
procedure hands you a flow it calls maximal, nothing in the procedure proves it: there is no
*certificate* that no larger flow exists. A solution would have to deliver both a flow and a
matching guarantee that the flow cannot be beaten.

## Background

**Networks, flows, and conservation.** A directed graph `D = (V, A)` with two distinguished
vertices, source `s` and sink `t`. A flow is a function `f : A → ℝ₊`
obeying the *conservation law* at every vertex other than source and sink: the total flow
entering a vertex equals the total leaving it,
`Σ_{a into v} f(a) = Σ_{a out of v} f(a)`.
The *value* of a flow is the net flow leaving the source,
`value(f) = Σ_{a out of s} f(a) − Σ_{a into s} f(a)`,
and conservation makes this equal to the net flow entering the sink. A *capacity* function
`c : A → ℝ₊` constrains the flow: `f` is feasible if `f(a) ≤ c(a)` on every arc. An arc with
`f(a) = c(a)` is *saturated*; one with `f(a) = 0` is *avoided*.

**Cuts.** For a vertex subset `W` with `s ∈ W`, `t ∉ W`, the set `δ⁺(W)` of arcs leaving `W`
(tail in `W`, head outside) is an `s`–`t` cut. Its capacity is `cap(δ⁺(W)) = Σ_{a ∈ δ⁺(W)} c(a)`.
The definition is *asymmetric*: arcs running back into `W` are not counted. Intuitively a cut
is a wall separating source from sink, and its capacity is the total carrying power of the
links crossing the wall in the forward direction.

**The obvious bound.** Anything that flows from source to sink must cross every such wall, so
no flow can exceed the capacity of any cut. This is intuitively clear and easy to make exact
(the flow-across-a-cut computation): for any flow `f` and any cut `δ⁺(W)`, expand
`value(f)` as a sum of net-flows over the vertices of `W`; conservation kills every interior
term, internal `W`-to-`W` arcs cancel, and what survives is the flow crossing the cut
forward minus the flow crossing it backward — at most the forward flow, at most the forward
capacity. So `value(f) ≤ cap(δ⁺(W))` for *every* flow and *every* cut. The maximum flow is
therefore `≤` the minimum cut. The whole question is whether this gap is ever forced open, or
whether it always closes.

**The genesis (rail interdiction).** The problem reached Ford and Fulkerson at RAND from a
1955 classified report by T. E. Harris and General F. S. Ross, *Fundamentals of a Method for
Evaluating Rail Net Capacities*, a study of the Soviet and Eastern-European railway network.
They modeled the rail system as a graph — not with junctions as vertices (too fine) but with
railway *operating divisions* (administrative districts) as vertices, and the total transport
capacity between adjacent divisions as edge weights — yielding 44 vertices and 105 edges.
Their object was the *minimum cut*: the cheapest way to interdict (bomb) the network so that
no traffic could move. They observed that a railway net is not just a bundle of independent
through-lines — cutting every line in a "set of through lines" need not disconnect the net,
because alternative routings survive — so the bottleneck had to be computed on the whole
graph. The dual flow value and the bottleneck cut both came out, in their hand computation,
to 163,000 tons.

**Linear programming and duality.** The maximum-flow problem is a linear program: maximize a
linear functional (the value) over the polytope of feasible flows cut out by the conservation
equations and the capacity inequalities. Dantzig's simplex method (Dantzig 1951, *Activity
Analysis of Production and Allocation*, Cowles Commission) solves such programs by walking
vertices of the polytope. Every linear program has a dual, and the strong-duality phenomenon
— optimum of the primal equals optimum of the dual — is the general fact under which equality
of the two network optima, should it hold, would be a combinatorial instance. But simplex on
the flow LP is, in the words of the rail study, cumbersome, and demands input accuracy a
hundred-edge rail net cannot supply.

## Baselines

**The simplex method (Dantzig 1951).** Set the flow up as a linear program — one variable per
arc (or per source-to-sink route), conservation equalities, capacity inequalities — and run
simplex, pivoting from vertex to vertex of the feasible polytope until the value can no
longer increase. *Core idea:* general-purpose LP optimization; it will return a maximal flow,
and via LP duality a dual optimum. *Gap:* it is heavy machinery for a combinatorial problem.
It does not exploit the network structure, the hand computation is laborious on a realistic
graph, and it gives no transparent, network-level certificate that a planner can read off the
map — exactly what the rail study said it could not justify ("the calculation would be
cumbersome; and, even if it could be performed, sufficiently accurate data could not be
obtained").

**The flooding heuristic (Boldyreff 1955).** Push as much flow as possible greedily through
the network from source toward sink; when a vertex becomes a bottleneck (more arrives than can
be sent onward), return the excess to the origin and continue. *Core idea:* a fast, hand-
computable, almost game-like greedy procedure ("can be taught to a ten-year-old boy in a few
minutes"). *Gap:* it does **not** guarantee optimality. Greedy commitment of flow along an
early path can block a globally better routing, and once flow is committed the procedure has
no way to *retract* it. Boldyreff could only *speculate* that for usual railway networks a
single flooding plus bottleneck removal "should lead to a maximal flow." There is no proof
and no certificate; the method can stop short of the true maximum with no way to detect it.

**Per-route chain-flow decomposition.** Represent a flow as a collection of source-to-sink
chains, each carrying a nonnegative amount, with the constraint that the chains through any
arc do not overload it. *Core idea:* makes "value of the flow" a sum over routes and turns
the feasible flows into a convex polytope, so a maximum exists and the set of maxima is
convex. *Gap:* the existence argument is non-constructive — it proves a maximum flow exists
and (with care) that the minimum cut equals it, but it does not by itself *produce* the flow
or the cut, and it does not tell a computer or a planner how to find them.

## Evaluation settings

The natural object is a capacitated directed graph with a designated source and sink — for
instance the Harris–Ross rail model (railway operating divisions as the ~44 vertices, ~105
inter-division links as edges, daily tonnage as capacities) or any transportation / pipeline
/ communication network of that kind. The yardsticks are: the value of the flow produced
(does it match a cut?), the capacity of the cut produced (does it match the flow?), the
number of elementary steps the procedure takes (hand-computable on a hundred-edge graph?),
and whether the output includes a checkable certificate of optimality. Capacities may be
integers (tonnages), rationals, or in principle reals, and a method should be analyzed for
whether it terminates and how its work scales with the number of vertices `n = |V|` and arcs
`m = |E|`.

## Code framework

The primitives that exist beforehand are just a directed graph with capacities, a notion of
flow on its arcs, and graph search. The scaffold therefore keeps the capacity table, an
empty flow table, a generic search for a source-to-sink route, a generic push operation
along such a route, and an empty slot for the certificate-producing stop condition.

```python
from collections import deque

def max_flow(cap, s, t):
    # cap[u][v] = capacity of arc u->v. Missing entries mean capacity 0.
    vertices = {s, t}
    for u, nbrs in cap.items():
        vertices.add(u)
        vertices.update(nbrs)

    capacity = {u: dict(cap.get(u, {})) for u in vertices}
    neighbors = {u: set() for u in vertices}
    for u in vertices:
        for v in capacity[u]:
            neighbors[u].add(v)
            neighbors[v].add(u)

    flow = {u: {v: 0 for v in capacity[u]} for u in vertices}

    def available_room(u, v):
        # TODO: define how much a route step from u to v can still change the flow.
        pass

    def find_source_sink_route():
        # TODO: graph search from s to t using only route steps with available room.
        pass

    def push(parent):
        # TODO: choose the route bottleneck and update the flow along that route.
        pass

    def certifying_cut():
        # TODO: when no route remains, return the source side of the proving cut.
        pass

    # TODO: repeatedly search, push, and stop with the certificate when search fails.
    pass
```
