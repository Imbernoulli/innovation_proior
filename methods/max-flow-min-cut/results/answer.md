# The Max-Flow Min-Cut Theorem

## Problem

In a directed graph `G = (V, E)` with a source `s`, a sink `t`, and a nonnegative capacity
`c(u→v)` on each arc, a *flow* is a function `f : E → ℝ₊` that is bounded by capacity
(`f(e) ≤ c(e)`) and conserved at every vertex except `s` and `t` (inflow = outflow). Its
*value* `|f|` is the net flow out of `s`. An `(s,t)`-*cut* `(S, T)` is a partition of `V` with
`s ∈ S`, `t ∈ T`; its *capacity* `‖S,T‖ = Σ_{u∈S, v∈T} c(u→v)` counts only the arcs crossing
forward. We want the maximum flow value and the minimum cut capacity, with a certificate that
each is optimal.

## Key idea

Maximum flow and minimum cut are dual, and one constructive procedure solves both: augment the
flow along source-to-sink paths in the *residual graph* (which contains, for each arc, a
forward "push more" edge and a reverse "cancel existing flow" edge) until no such path remains.
At that moment the set of vertices still reachable from `s` in the residual graph defines a cut
every one of whose forward arcs is saturated and whose backward arcs carry no flow — so its
capacity equals the flow value, certifying both optima simultaneously.

## The theorem

**Max-Flow Min-Cut Theorem.**
In every flow network, the maximum value of an `(s,t)`-flow equals the minimum capacity of an
`(s,t)`-cut:
```
max_{f}  |f|   =   min_{(S,T)}  ‖S,T‖.
```
If all capacities are integers, there is an integer-valued maximum flow (Integrality Theorem).

## Proof

**Weak duality (flow across a cut).** For any flow `f` and any cut `(S,T)`,
```
|f| = ∂f(s) = Σ_{v∈S} ∂f(v)                         (∂f(v)=0 for interior v, by conservation)
    = Σ_{u∈S, v∈T} f(u→v) − Σ_{u∈T, v∈S} f(u→v)     (internal S–S arcs cancel)
    ≤ Σ_{u∈S, v∈T} f(u→v)                           (drop the backward term, which is ≥ 0)
    ≤ Σ_{u∈S, v∈T} c(u→v) = ‖S,T‖.                  (f ≤ c on forward arcs)
```
Hence `max |f| ≤ min ‖S,T‖`. Equality holds **iff** every `S→T` arc is *saturated*
(`f = c`, second inequality tight) and every `T→S` arc is *avoided* (`f = 0`, first
inequality tight).

**Residual graph.** Given a feasible flow `f`, define the residual capacity
```
c_f(u→v) = (c(u→v) − f(u→v))  +  f(v→u),
```
the forward slack on arc `u→v` plus the cancellable flow on the reverse arc `v→u`. The
residual graph `G_f` is the set of pairs with `c_f > 0`. If both real arcs exist, an
augmentation step from `u` to `v` can cancel flow on `v→u` and then push any remaining
amount on `u→v`; the two effects add because both increase the net flow from `u` to `v`.

**Augmenting-path theorem.** Exactly one of two cases holds.

- *There is an `s→t` path `P` in `G_f`.* Let `F = min_{e∈P} c_f(e) > 0`. For each residual
  step `u→v`, cancel `min(F, f(v→u))` units on the reverse real arc, then push the remaining
  amount on the forward real arc `u→v`. The definition of `F` keeps every changed arc in
  `[0, c]`; each interior path vertex receives `F` units of net change and sends `F` units on;
  and the first step raises source net outflow by `F`. Thus `|f'| = |f| + F`, so `f` is
  **not** maximum.
- *There is no `s→t` path in `G_f`.* Let `S = {v : v reachable from s in G_f}` and `T = V∖S`;
  then `s∈S`, `t∈T`. For `u∈S, v∈T`: no residual edge leaves `S`, so any real arc `u→v` has
  `c(u→v) − f(u→v) = 0` (**saturated**) and any real arc `v→u` has `f(v→u) = 0` (**avoided**,
  else its reverse residual edge would reach `v`). By the equality condition,
  `|f| = ‖S,T‖`, so `f` is a maximum flow and `(S,T)` a minimum cut.

The feasible-flow polytope is nonempty and compact, so a maximum flow exists; for that flow
the first case is impossible, hence the second case supplies an equal cut. Therefore
`max |f| = min ‖S,T‖`. ∎

**Integrality / termination.** With integer capacities, `f = 0` and every `c_f` start
integral and stay integral, so each augmentation has `F ≥ 1` and raises `|f|` by ≥ 1; the
value is bounded by `‖{s}, V∖{s}‖`, so the process halts, with an integral optimum. (Rational
capacities scale to this case; with irrational capacities and bad path choices the process can
fail to terminate.)

## Algorithm and the Edmonds–Karp bound

Ford–Fulkerson: from the zero flow, repeatedly find an augmenting path in `G_f` and augment,
until none exists; then read off the cut `S`. With *arbitrary* augmenting paths the number of
augmentations can be `Θ(|f*|)` (exponential in the input size). **Edmonds–Karp:** always
choose a *shortest* (fewest-arc) augmenting path, found by breadth-first search. Then the BFS
distance `δ(s, v)` is monotone non-decreasing across augmentations: any newly created residual
edge is the reverse of an edge on the previous shortest path, and using it in a shorter new path
would force the endpoint distance to have increased by two, a contradiction. Once an edge is a
bottleneck, it must be recreated by a later reverse use before it can be a bottleneck again, so
one endpoint's BFS distance rises by at least two between such events. Each edge is therefore a
bottleneck `O(V)` times, giving `O(VE)` augmentations at `O(E)` each, hence **`O(VE²)`** total —
independent of the capacity values.

```python
from collections import deque

def max_flow(cap, s, t):
    """cap[u][v] = capacity of arc u->v. Returns (value, flow, min_cut_S)."""
    if s == t:
        raise ValueError("source and sink must be distinct")

    vertices = {s, t}
    for u, nbrs in cap.items():
        vertices.add(u)
        vertices.update(nbrs)

    capacity = {u: dict(cap.get(u, {})) for u in vertices}
    neighbors = {u: set() for u in vertices}
    for u in vertices:
        for v, c_uv in capacity[u].items():
            if c_uv < 0:
                raise ValueError("capacities must be nonnegative")
            neighbors[u].add(v)
            neighbors[v].add(u)          # the opposite direction may cancel flow

    flow = {u: {v: 0 for v in capacity[u]} for u in vertices}

    def available_room(u, v):
        # residual capacity: unused u->v capacity plus cancellable v->u flow
        return capacity[u].get(v, 0) - flow[u].get(v, 0) + flow[v].get(u, 0)

    def find_source_sink_route():
        # Edmonds-Karp: BFS gives a shortest augmenting path in the residual graph.
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in parent and available_room(u, v) > 0:
                    parent[v] = u
                    if v == t:
                        return parent
                    q.append(v)
        return None

    def push(parent):
        # Bottleneck residual capacity along the path.
        F, v = float("inf"), t
        while parent[v] is not None:
            u = parent[v]
            F = min(F, available_room(u, v))
            v = u

        # Implement each residual step by canceling reverse flow first,
        # then pushing any remaining amount on the forward arc.
        v = t
        while parent[v] is not None:
            u = parent[v]
            cancel = min(F, flow[v].get(u, 0))
            if cancel:
                flow[v][u] -= cancel
            forward = F - cancel
            if forward:
                flow[u][v] += forward
            v = u
        return F

    def certifying_cut():
        # Vertices still reachable from s in the residual graph form the cut.
        S, q = {s}, deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in S and available_room(u, v) > 0:
                    S.add(v)
                    q.append(v)
        return S

    value = 0
    while True:
        parent = find_source_sink_route()
        if parent is None:
            break
        value += push(parent)

    return value, {u: dict(flow[u]) for u in vertices if flow[u]}, certifying_cut()
```

## Why it works, in one line

The value of any flow is its net crossing of any cut, hence at most the cut's forward
capacity; a residual graph whose reverse edges undo committed flow lets augmenting paths reach
the optimum; and when no augmenting path remains, the source-reachable set is a cut that is
forced saturated-forward and idle-backward, so flow value = cut capacity — proving max-flow =
min-cut and certifying both at once.
