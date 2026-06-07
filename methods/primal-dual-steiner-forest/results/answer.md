# Primal–Dual 2-approximation for Steiner Forest

## Problem

Given an undirected graph `G = (V, E)` with edge costs `c_e ≥ 0` and `p` terminal pairs
`(s_j, t_j)`, find a minimum-cost edge set `F` so that every `s_j` is connected to `t_j` in
`(V, F)`. The optimum may be a disconnected forest. The problem is NP-hard; the algorithm returns a
solution of cost at most `2·OPT` in polynomial time.

## Key idea

Feasibility is a cut-covering condition: `s_j`–`t_j` are connected iff `F` hits `δ(S)` for every set
`S` that separates the pair. The LP relaxation has a dual that assigns a width `y_S ≥ 0` ("moat") to
each separating set, with `∑_{S: e∈δ(S)} y_S ≤ c_e` per edge; any feasible dual lower-bounds OPT.
The algorithm never solves the LP — it *grows* the dual. Maintaining the connected components of the
edges bought so far, it raises the moats of all **active** components (those still separating an
unconnected pair) **uniformly and simultaneously**. It stores vertex labels
`d(v)=∑_{S:v∈S}y_S`; an edge `(u,v)` crossing two current components goes tight when
`d(u)+d(v)=c_e`, is bought, and merges its two components. A final **reverse-delete** removes edges
(latest-bought first) that are not needed for feasibility.

Growing one moat at a time gives only a `k`-approximation (one common source, `k` sinks: `k` star
edges charged to a single moat). Growing all active moats at once spreads the charge so that, at
every iteration, the bought edges crossing the active moats number at most twice the active moats —
the source of the factor 2.

## Algorithm

1. Components ← singletons; vertex labels `d(v) ← 0`; `F ← ∅`.
2. While some component is active (separates an unconnected pair):
   - For each edge `e=(u,v)` across two components, slack `= c_e − d(u) − d(v)` is consumed at rate
     `=` number of active endpoint components (1 or 2); the next tight edge minimizes `slack/rate`,
     giving the uniform growth amount `ε`.
   - Raise every active component's vertices by `ε`; add the tight edge to `F`; merge its
     components.
3. Reverse-delete: for each edge in reverse order of addition, remove it if all pairs stay connected.
4. Return `F`.

## Why it is a 2-approximation

Every kept edge is tight, so `cost(F') = ∑_{e∈F'} ∑_{S:e∈δ(S)} y_S = ∑_S y_S·|F'∩δ(S)|`. Writing
`y_S = ∑_{t: S active} ε_t` and regrouping by iteration, it suffices that at each iteration `t`,
`∑_{S∈A_t}|F'∩δ(S)| ≤ 2|A_t|`. Contract the current components to nodes and add an edge for each
kept `F'`-edge across components; the growth phase makes `H` a forest, and reverse-delete leaves no
inactive (blue) leaf, so every inactive node has degree ≥ 2. Then for active (red) nodes
`∑_{red} deg = ∑_{all} deg − ∑_{blue} deg ≤ 2(|R|+|B|−1) − 2|B| = 2|R|−2 ≤ 2|R| = 2|A_t|`.
Summing the increments, `cost(F') ≤ 2∑_S y_S ≤ 2·OPT`. The dual `∑_S y_S` is a per-instance
lower-bound certificate. Keeping the `−2`, and writing `K` for the initial number of active
components, gives `cost(F') ≤ (2 − 2/K)∑_S y_S ≤ (2 − 2/K)OPT`; this is the sharper
`2 − 2/K` form. Special cases: one terminal pair reduces to bidirectional Dijkstra (optimal
`s`–`t` path); all-pairs connectivity reduces to Kruskal (optimal MST).

## Working code

```python
def steiner_forest(n, edges, pairs):
    """Min-cost forest connecting each (s, t) in `pairs`.
    n: vertices 0..n-1; edges: (u, v, cost) with cost >= 0; pairs: (s, t)."""
    parent = list(range(n))                       # union-find over vertices

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d = [0.0] * n                                  # d[v] = sum of moats containing v

    def active(root):                              # separates some unconnected pair?
        for s, t in pairs:
            if (find(s) == root) != (find(t) == root):
                return True
        return False

    chosen = []                                    # edges bought, in order
    while True:
        roots = {find(v) for v in range(n)}
        act = {r for r in roots if active(r)}
        if not act:                                # all pairs connected -> feasible
            break
        best_eps, best_edge = None, None           # earliest edge to go tight
        for (u, v, c) in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            rate = (ru in act) + (rv in act)       # active endpoints: 1 or 2
            if rate == 0:
                continue
            eps = (c - d[u] - d[v]) / rate
            if best_eps is None or eps < best_eps:
                best_eps, best_edge = eps, (u, v, c)
        if best_edge is None:
            raise ValueError("no feasible edge can connect the remaining terminal pairs")
        for x in range(n):                         # grow every active moat by eps
            if find(x) in act:
                d[x] += best_eps
        u, v, c = best_edge
        chosen.append((u, v, c))
        parent[find(u)] = find(v)                  # buy tight edge, merge

    forest = list(chosen)                          # reverse-delete cleanup
    for e in reversed(chosen):
        removed = False
        trial = []
        for x in forest:
            if not removed and x is e:
                removed = True
                continue
            trial.append(x)
        if not removed:
            continue
        if _all_connected(n, trial, pairs):
            forest = trial
    return forest


def _all_connected(n, forest, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v, _c) in forest:
        parent[find(u)] = find(v)
    return all(find(s) == find(t) for s, t in pairs)
```
