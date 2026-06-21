# The Max-Flow Min-Cut Theorem

## Statement

In a directed network `G=(V,E)` with source `s`, sink `t`, and nonnegative capacities `c(e)`, the maximum value of a feasible `s-t` flow equals the minimum capacity of an `s-t` cut:

```text
max_f |f| = min_{S: s in S, t not in S} sum_{u in S, v not in S} c(u,v).
```

If all capacities are integers, there is an integer-valued maximum flow.

## Certificate Lemma

For any feasible flow `f` and cut `(S,T)`,

```text
|f| = sum_{S->T} f - sum_{T->S} f
     <= sum_{S->T} f
     <= sum_{S->T} c
     = cap(S,T).
```

Equality holds exactly when every arc from `S` to `T` is saturated and every arc from `T` to `S` carries zero flow. Thus one matching flow/cut pair certifies both optima.

## Residual Augmentation

For a current flow `f`, define residual capacity

```text
c_f(u,v) = c(u,v) - f(u,v) + f(v,u),
```

where missing arcs have capacity and flow zero. The first term is unused forward capacity; the second is flow on the reverse arc that can be canceled.

If the residual graph has an `s-t` path `P`, push

```text
F = min_{(u,v) in P} c_f(u,v)
```

along it. Each residual step cancels reverse flow first and then adds any remaining amount forward. This keeps all arc flows in `[0,c]`, preserves conservation at internal vertices, and increases the flow value by `F`.

If no residual `s-t` path exists, let `S` be the vertices reachable from `s` in the residual graph. Then `t` is outside `S`. No residual edge leaves `S`, so every original arc from `S` to `T` is saturated and every original arc from `T` to `S` carries zero flow. By the certificate lemma, `|f|=cap(S,T)`, so `f` is maximum and `(S,T)` is minimum.

## Algorithm

```python
from collections import deque


def max_flow_min_cut(cap, s, t):
    """Return (value, flow, S), where S is the source side of a minimum cut.

    cap[u][v] is the capacity of arc u->v. Missing arcs have capacity 0.
    Uses shortest residual augmenting paths, i.e. Edmonds-Karp.
    """
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
            neighbors[v].add(u)

    flow = {u: {v: 0 for v in capacity[u]} for u in vertices}

    def residual(u, v):
        return capacity[u].get(v, 0) - flow[u].get(v, 0) + flow[v].get(u, 0)

    def bfs():
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in parent and residual(u, v) > 0:
                    parent[v] = u
                    if v == t:
                        return parent
                    q.append(v)
        return None

    def augment(parent):
        amount = float("inf")
        v = t
        while parent[v] is not None:
            u = parent[v]
            amount = min(amount, residual(u, v))
            v = u

        v = t
        while parent[v] is not None:
            u = parent[v]
            cancel = min(amount, flow[v].get(u, 0))
            if cancel:
                flow[v][u] -= cancel
            forward = amount - cancel
            if forward:
                flow[u][v] += forward
            v = u
        return amount

    value = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        value += augment(parent)

    S = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if v not in S and residual(u, v) > 0:
                S.add(v)
                q.append(v)

    return value, {u: dict(flow[u]) for u in vertices if flow[u]}, S
```

With integer capacities, each augmentation raises the value by at least `1`, so the basic method terminates and returns an integral optimum. With breadth-first shortest augmenting paths, residual distances never decrease and each edge can be a bottleneck only `O(V)` times; hence there are `O(VE)` augmentations and total running time `O(VE^2)`.
