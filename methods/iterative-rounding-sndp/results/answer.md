# Iterative rounding for survivable network design (factor-2)

## Problem

Undirected graph `G = (V, E)`, nonnegative edge costs `c_e`, and for each vertex pair an integer
requirement `r(uv)`. Find a minimum-cost subgraph `H` containing at least `r(uv)` edge-disjoint paths
between `u` and `v` for every pair. The problem is NP-hard and APX-hard; on feasible instances,
iterative rounding returns, in polynomial time, a feasible subgraph of cost at most `2 * OPT`, with the
factor independent of the maximum requirement.

## Key idea

By Menger, edge-disjoint paths equal min cuts, so feasibility is the cut condition
`|delta_H(S)| >= f(S)` for all `S`, where `f(S) = max_{u in S, v not in S} r(uv)` is **weakly
supermodular** (`f(empty)=f(V)=0`, and for all `A,B` either `f(A)+f(B) <= f(A∪B)+f(A∩B)` or
`f(A)+f(B) <= f(A\B)+f(B\A)`). Relax to the covering LP

    min  sum_e c_e x_e
    s.t. x(delta(S)) >= f(S)   for all S subset V
         0 <= x_e <= 1.

Two facts make a clean factor-2 rounding possible.

1. **Structural theorem.** In every basic feasible (vertex) solution `x` of a nonzero residual LP
   there is an edge with `x_e >= 1/2`.
2. **Closure of the class.** For any fixed edge set `F`, the residual requirement
   `g(S) = f(S) - |delta_F(S)|` is again weakly supermodular (the cut function `|delta_F|` is
   symmetric submodular, hence posimodular; subtracting it preserves one of the two weak-super
   inequalities).

Round every `>= 1/2` edge up to `1`, fix it, replace `f` by the residual `g`, and recurse. Because
each committed edge is at least half-paid by the LP, the total cost is at most twice the LP optimum,
which lower-bounds `OPT`.

## Why the structural theorem holds

**Uncrossing → laminar basis.** Work on the positive support. Edges with `x_e = 0` are deleted, and an
edge with `x_e >= 1/2` already proves the theorem, so the hard case has no active variable bounds.
Tight sets satisfy `x(delta(S)) = f(S)`. The cut function is submodular and posimodular:
`x(delta(S)) + x(delta(T)) >= x(delta(S∪T)) + x(delta(S∩T))` and
`>= x(delta(S\T)) + x(delta(T\S))`. If two tight sets cross, chaining the matching weak-super
inequality of `f` through the cut inequality and LP feasibility forces equality throughout, so the
uncrossed sets are also tight and the extra cross-edge term in the matching cut identity has zero
`x`-mass; since all support values are positive, that edge set is empty and the crossing row is spanned
by the uncrossed rows plus the row it crossed. Take a maximal laminar family of tight sets. If it did
not span all tight rows, choose an outside tight set crossing the fewest family members; uncrossing it
with one crossed member gives an outside tight row crossing fewer members, a contradiction. Thus, in
the hard support case, some laminar family `L` has `x` as the unique solution of
`{x(delta(S)) = f(S): S in L}`, the vectors `chi(delta(S))` independent, and `|L| = |E|`.

**Token counting.** Suppose `0 < x_e < 1/2` for all `e`. Give each edge one token. Edge `e=(u,v)`
sends `x_e` to the smallest set containing `u`, `x_e` to the smallest set containing `v`, and
`1 - 2x_e` to the smallest set containing both endpoints, whenever those sets exist; unassigned pieces
remain leftover. For `S in L` with children `R_1, ..., R_k`, subtracting tight equations gives
`x(delta(S)) - sum_i x(delta(R_i)) = f(S) - sum_i f(R_i)`. With
`O = S \ (R_1 ∪ ... ∪ R_k)`, the surviving edge classes are: `A`, edges from `O` to `V \ S`; `B`, edges
from `O` to one child; and `C`, endpoints in two different children. Thus
`x(A) - x(B) - 2x(C) = f(S) - sum_i f(R_i)`, and the tokens collected by `S` equal
`sum_A x_e + sum_B(1-x_e) + sum_C(1-2x_e) + |D| = |B| + |C| + |D| + f(S) - sum_i f(R_i)`,
where `D` is the class of edges with both endpoints in `O`, which cancel from the subtraction but send
their whole token to `S`. This is a positive integer: if no positive piece reached `S`, then
`chi(delta(S))` would be dependent on the child vectors, and integrality comes from `f`. Hence every
set gets at least one token. A maximal set has a support edge crossing it, and that edge's
`1 - 2x_e` both-endpoints token is unclaimed, so strictly fewer than `|E|` tokens are collected while
all `|L|` sets get one: `|L| < |E|`, contradicting `|L| = |E|`. Therefore some `x_e >= 1/2`.

## Algorithm

1. `F <- empty`.
2. Repeat: form the residual cut requirement `g(S) = f(S) - |delta_F(S)|`. If `g <= 0` everywhere,
   return `F`.
3. Solve the covering LP for `g` to a basic optimal (vertex) solution `x*`. The runnable code below
   adds violated cut rows with Gomory-Hu / max-flow separation until the solution satisfies every cut,
   and raises if the returned active-LP optimum does not contain the half-paid edge the theorem
   guarantees for a basic solution.
4. Add every edge with `x*_e >= 1/2` to `F` (the theorem guarantees at least one). Go to 2.

## Analysis (ratio 2)

Per round with rounded set `R = {e : x*_e >= 1/2}`:
`cost(R) = sum_{e in R} c_e <= 2 sum_{e in R} c_e x*_e`. For every cut `S`, deleting the rounded
coordinates leaves
`x*(delta(S)) - x*_R(delta(S)) >= g_current(S) - x*_R(delta(S)) >= g_current(S) - |delta_R(S)|`,
which is exactly `g_next(S)`. Thus
`LP(next) <= LP(current) - sum_{e in R} c_e x*_e`. Telescoping over rounds,
`cost(F) <= 2 sum_rounds (LP(current) - LP(next)) <= 2 * LP(initial) <= 2 * OPT`. The factor is `2`
regardless of `r_max`.

## Code

```python
from itertools import combinations

import networkx as nx
import pulp

def requirement_on_cut(S, r):
    # Menger turns pair requirements into one cut requirement.
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
    rev = (e[1], e[0])
    if rev in costs:
        return costs[rev]
    raise KeyError(f"missing cost for edge {e}")

def separation_oracle(x, free_edges, fixed_edges, V, r, tol=1e-7):
    # The fixed edges count integrally; the remaining edges carry their LP values.
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
        if u not in H or v not in H:
            return frozenset({u})
        path = nx.shortest_path(tree, u, v)
        cut_edge = min(zip(path, path[1:]), key=lambda ab: tree[ab[0]][ab[1]]["weight"])
        cut_value = tree[cut_edge[0]][cut_edge[1]]["weight"]
        if cut_value < req - tol:
            witness = tree.copy()
            witness.remove_edge(*cut_edge)
            return frozenset(nx.node_connected_component(witness, u))
    return None

def solve_covering_lp_to_vertex(edges, V, r, fixed_edges, costs):
    fixed_edges = set(fixed_edges)
    free = [e for e in edges if e not in fixed_edges]
    prob = pulp.LpProblem("sndp_cut_cover", pulp.LpMinimize)
    xv = {e: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1) for i, e in enumerate(free)}
    prob += pulp.lpSum(edge_cost(costs, e) * xv[e] for e in free)
    solver = pulp.PULP_CBC_CMD(msg=False)

    while True:
        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"residual LP is {pulp.LpStatus[status]}")

        x = {e: (xv[e].value() or 0.0) for e in free}
        S = separation_oracle(x, free, fixed_edges, V, r)
        if S is None:
            return x

        cross = delta(S, free)
        rhs = requirement_on_cut(S, r) - len(delta(S, fixed_edges))
        if rhs > 1e-7 and not cross:
            raise RuntimeError("residual instance is infeasible with the remaining edges")
        prob += pulp.lpSum(xv[e] for e in cross) >= rhs

def all_satisfied(V, fixed_edges, r):
    return separation_oracle({}, [], fixed_edges, V, r) is None

def cover_cut_requirements(V, edges, costs, r):
    edges = [tuple(e) for e in edges]
    F = set()

    while not all_satisfied(V, F, r):
        x = solve_covering_lp_to_vertex(edges, V, r, F, costs)
        R = [e for e, val in x.items() if val >= 0.5 - 1e-9]
        if not R:
            raise RuntimeError("expected an edge with value at least 1/2 in the basic solution")
        F.update(R)

    return F

def solution_cost(F, costs):
    return sum(edge_cost(costs, e) for e in F)

def is_feasible(V, F, r):
    H = nx.Graph()
    H.add_nodes_from(V)
    H.add_edges_from(F)
    for (u, v), req in r.items():
        if req <= 0:
            continue
        if nx.edge_connectivity(H, u, v) < req:
            return False
    return True

def brute_force_optimum(V, E, costs, r):
    best_cost = float("inf")
    best_set = None
    for size in range(len(E) + 1):
        for chosen in combinations(E, size):
            F = set(chosen)
            cost = solution_cost(F, costs)
            if cost >= best_cost:
                continue
            if is_feasible(V, F, r):
                best_cost = cost
                best_set = F
    return best_cost, best_set

if __name__ == "__main__":
    V = [0, 1, 2, 3]
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    costs = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1, (0, 2): 5}
    r = {(0, 2): 2}

    F = cover_cut_requirements(V, E, costs, r)
    alg_cost = solution_cost(F, costs)
    opt_cost, opt_set = brute_force_optimum(V, E, costs, r)

    print("chosen edges:", sorted(F))
    print("algorithm cost:", alg_cost)
    print("exact optimum:", opt_cost, sorted(opt_set))
    print("feasible:", is_feasible(V, F, r))
    print("within 2x:", alg_cost <= 2 * opt_cost)
```
