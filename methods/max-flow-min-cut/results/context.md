## Research Question

A transportation network has two distinguished terminals. Traffic starts at one terminal, ends at the other, and may split and recombine across intermediate routes. Each directed link has a capacity, meaning the largest steady amount it can carry. The immediate question is how much total traffic can be moved from the origin to the destination without violating any link capacity or accumulating traffic at intermediate vertices.

The same network has a second, adversarial question. Which links must be removed, and with what least total capacity, to separate the origin from the destination? In the rail setting that motivated the problem, this was the operational question: identify a bottleneck in a large railway system, not merely a set of locally busy tracks.

These two quantities look related. A route-packing answer gives a feasible traffic pattern. A separating set gives an upper bound on every possible traffic pattern. The broad question is whether and how the two can be connected.

## Objects Already On The Table

A directed network is a finite directed graph `G = (V,E)` with a source `s`, a sink `t`, and a nonnegative capacity `c(u,v)` on each arc. A feasible flow assigns a nonnegative number `f(u,v)` to every arc such that `0 <= f(u,v) <= c(u,v)`.

Every internal vertex obeys conservation: the total inflow equals the total outflow. The value of the flow is the net amount leaving `s`,

`|f| = sum_v f(s,v) - sum_v f(v,s)`.

By conservation, this is also the net amount entering `t`.

A separating cut is a partition `(S,T)` of the vertices with `s in S` and `t in T`. Its capacity counts only arcs crossing from the source side to the sink side:

`cap(S,T) = sum_{u in S, v in T} c(u,v)`.

The direction matters. Arcs pointing from `T` back into `S` do not help carry traffic out of `S`, so they are not part of the forward bottleneck.

## The Obvious Upper Bound

For any feasible flow and any cut, summing the net outflow over all vertices on the source side cancels every arc internal to that side. What remains is the flow crossing the cut forward minus the flow crossing it backward:

`|f| = sum_{S->T} f - sum_{T->S} f`.

Since backward flow is nonnegative and each forward flow is at most capacity,

`|f| <= sum_{S->T} f <= sum_{S->T} c = cap(S,T)`.

Thus every cut is an upper bound on every flow.

## Existing Approaches

One approach is to formulate the flow problem as a linear program and apply a general method such as simplex. That is mathematically legitimate and treats the graph as a large set of equations and inequalities.

Another approach is greedy flooding: push traffic along available routes until bottlenecks appear, return excess, and continue. This is easy to perform on a map and fits the operational setting.

A third approach is route decomposition: view a flow as a sum of path flows. This makes existence and convexity visible.

## Evaluation And Code Scaffold

The setting is a finite directed graph with nonnegative capacities, source `s`, and sink `t`. The desired output is a feasible flow, a separating cut, and a local certificate showing that the two values match. Integer capacities should lead to an integer flow; route choices should be controlled enough to avoid a runtime depending only on the magnitude of capacities.

The available implementation primitives are a capacity table, a mutable flow table, graph search, a way to measure unused capacity, and a stopping rule that can also explain why stopping is correct.

```python
from collections import deque

def solve(cap, s, t):
    vertices = {s, t}
    for u, nbrs in cap.items():
        vertices.add(u)
        vertices.update(nbrs)

    capacity = {u: dict(cap.get(u, {})) for u in vertices}
    flow = {u: {v: 0 for v in capacity[u]} for u in vertices}

    def available_step(u, v):
        # TODO: quantify how a step from u to v can still change the current flow.
        pass

    def find_route():
        # TODO: search from s to t using only steps with positive available change.
        pass

    def improve(parent):
        # TODO: change the current flow along the found route without violating feasibility.
        pass

    def certificate():
        # TODO: when no route remains, report the cut that certifies optimality.
        pass

    pass
```
