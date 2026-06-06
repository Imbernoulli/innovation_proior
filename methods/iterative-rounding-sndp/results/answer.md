# Iterative rounding for survivable network design (factor-2)

## Problem

Undirected graph `G = (V, E)`, nonnegative edge costs `c_e`, and for each vertex pair an integer
requirement `r(uv)`. Find a minimum-cost subgraph `H` containing at least `r(uv)` edge-disjoint paths
between `u` and `v` for every pair. The problem is NP-hard and APX-hard; iterative rounding returns,
in polynomial time, a feasible subgraph of cost at most `2 * OPT`, with the factor independent of the
maximum requirement.

## Key idea

By Menger, edge-disjoint paths equal min cuts, so feasibility is the cut condition
`|delta_H(S)| >= f(S)` for all `S`, where `f(S) = max_{u in S, v not in S} r(uv)` is **weakly
supermodular** (`f(empty)=f(V)=0`, and for all `A,B` either `f(A)+f(B) <= f(A∪B)+f(A∩B)` or
`f(A)+f(B) <= f(A\B)+f(B\A)`). Relax to the covering LP

    min  sum_e c_e x_e
    s.t. x(delta(S)) >= f(S)   for all S subset V
         0 <= x_e <= 1.

Two facts make a clean factor-2 rounding possible.

1. **Structural theorem.** In every basic feasible (vertex) solution `x` of this LP there is an edge
   with `x_e >= 1/2`.
2. **Closure of the class.** For any fixed edge set `F`, the residual requirement
   `g(S) = f(S) - |delta_F(S)|` is again weakly supermodular (the cut function `|delta_F|` is
   symmetric submodular, hence posimodular; subtracting it preserves one of the two weak-super
   inequalities).

Round every `>= 1/2` edge up to `1`, fix it, replace `f` by the residual `g`, and recurse. Because
each committed edge is at least half-paid by the LP, the total cost is at most twice the LP optimum,
which lower-bounds `OPT`.

## Why the structural theorem holds

**Uncrossing → laminar basis.** Restrict to the support, `0 < x_e < 1`. Tight sets satisfy
`x(delta(S)) = f(S)`. The cut function is submodular and posimodular:
`x(delta(S)) + x(delta(T)) >= x(delta(S∪T)) + x(delta(S∩T))` and
`>= x(delta(S\T)) + x(delta(T\S))`. If two tight sets cross, chaining the matching weak-super
inequality of `f` through the cut inequality and LP feasibility forces equality throughout, so the
uncrossed sets are also tight and (since all `x_e > 0`, the cross-edge set is empty) their constraint
vectors span the same space. Uncrossing repeatedly yields a laminar family `L` with `x` the unique
solution of `{x(delta(S)) = f(S): S in L}`, the vectors `chi(delta(S))` independent, and `|L| = |E|`.

**Token counting.** Suppose `0 < x_e < 1/2` for all `e`. Give each edge one token; edge `e = (u,v)`
sends `x_e` to the smallest set of `L` containing `u`, `x_e` to the smallest containing `v`, and
`1 - 2 x_e` to the smallest set containing both (all three positive, summing to 1). For `S in L` with
children `R_1, ..., R_k`, subtracting tight equalities gives that the tokens `S` collects equal
`|B| + |C| + (f(S) - sum_i f(R_i))` — a positive integer (positive since the surviving edge set
`A ∪ B ∪ C` is nonempty by independence), hence `>= 1`. Every maximal set leaves its boundary edges'
`1 - 2 x_e` tokens unclaimed, so strictly fewer than `|E|` tokens are collected while all `|L|` sets
get one: `|L| < |E|`, contradicting `|L| = |E|`. Therefore some `x_e >= 1/2`. (If every `f(S)` is even,
rescaling the rules pushes the threshold to `x_e = 1`.)

## Algorithm

1. `F <- empty`.
2. Repeat: form the residual cut requirement `g(S) = f(S) - |delta_F(S)|`. If `g <= 0` everywhere,
   return `F`.
3. Solve the covering LP for `g` to a basic optimal (vertex) solution `x*` via a cutting-plane loop
   whose separation oracle is a min-cut / max-flow per demand pair (all pairs by a Gomory-Hu tree).
4. Add every edge with `x*_e >= 1/2` to `F` (the theorem guarantees at least one). Go to 2.

## Analysis (ratio 2)

Per round with rounded set `R = {e : x*_e >= 1/2}`:
`cost(R) = sum_{e in R} c_e <= 2 sum_{e in R} c_e x*_e`. Restricting `x*` to the
remaining edges is feasible for the next residual LP (the class is closed), so
`LP(next) <= LP(current) - sum_{e in R} c_e x*_e`. Telescoping over rounds,
`cost(F) <= 2 sum_rounds (LP(current) - LP(next)) <= 2 * LP(initial) <= 2 * OPT`. The factor is `2`
regardless of `r_max`.

## Code

```python
import networkx as nx
import pulp

def requirement_on_cut(S, r):
    # f(S) = max over pairs split by S of r(uv): cut form of the demands
    best = 0
    for (u, v), req in r.items():
        if (u in S) ^ (v in S):
            best = max(best, req)
    return best

def delta(S, edges):
    return [e for e in edges if (e[0] in S) ^ (e[1] in S)]

def add_capacity(H, e, cap):
    u, v = e
    if H.has_edge(u, v):
        H[u][v]["cap"] += cap
    else:
        H.add_edge(u, v, cap=cap)

def edge_cost(costs, e):
    if e in costs:
        return costs[e]
    return costs[(e[1], e[0])]

def separation_oracle(x, free_edges, fixed_edges, V, r, tol=1e-7):
    H = nx.Graph()
    H.add_nodes_from(V)
    for e in free_edges:
        add_capacity(H, e, max(0.0, float(x.get(e, 0.0))))
    for e in fixed_edges:
        add_capacity(H, e, 1.0)

    tree = nx.gomory_hu_tree(H, capacity="cap")
    for (u, v), req in r.items():
        if req <= 0 or u == v:
            continue
        path = nx.shortest_path(tree, u, v)
        cut_edge = min(zip(path, path[1:]), key=lambda ab: tree[ab[0]][ab[1]]["weight"])
        cut_value = tree[cut_edge[0]][cut_edge[1]]["weight"]
        if cut_value < req - tol:
            witness = tree.copy()
            witness.remove_edge(*cut_edge)
            return frozenset(nx.node_connected_component(witness, u))
    return None

def solve_covering_lp_to_vertex(edges, V, r, fixed_edges, costs):
    free = [e for e in edges if e not in fixed_edges]
    prob = pulp.LpProblem("cut_cover", pulp.LpMinimize)
    xv = {e: pulp.LpVariable(f"x_{e}", lowBound=0, upBound=1) for e in free}
    prob += pulp.lpSum(edge_cost(costs, e) * xv[e] for e in free)
    solver = pulp.PULP_CBC_CMD(msg=False)
    while True:
        status = prob.solve(solver)                # CBC/CLP returns a basic optimum for the active LP
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"residual LP is {pulp.LpStatus[status]}")
        x = {e: (xv[e].value() or 0.0) for e in free}
        S = separation_oracle(x, free, fixed_edges, V, r)
        if S is None:
            return x
        cross = [e for e in free if (e[0] in S) ^ (e[1] in S)]
        fixed = len(delta(S, fixed_edges))
        rhs = requirement_on_cut(S, r) - fixed
        prob += pulp.lpSum(xv[e] for e in cross) >= rhs

def all_satisfied(V, fixed_edges, r):
    return separation_oracle({}, [], fixed_edges, V, r) is None

def cover_cut_requirements(V, edges, costs, r):
    edges = [tuple(e) for e in edges]
    F = set()
    while True:
        if all_satisfied(V, F, r):
            return F
        x = solve_covering_lp_to_vertex(edges, V, r, F, costs)
        R = [e for e, val in x.items() if val >= 0.5 - 1e-9]
        if not R:
            raise RuntimeError("expected an edge with value at least 1/2 in the basic residual solution")
        F.update(R)
```
