OK, let me think this through from scratch. I have an undirected graph with edge costs, and for each
pair of vertices a requirement `r(uv)` — the number of edge-disjoint `u`-`v` paths the final subgraph
must contain so the network survives `r(uv) - 1` failures. I want the cheapest subgraph meeting all
of them. I assume the input graph itself can meet these requirements; otherwise there is no finite
optimum to approximate. Finding that optimum is hopeless in polynomial time, so I'm chasing a
*guarantee* — a feasible subgraph I can build fast and prove is never more than some fixed factor
times `OPT`, and crucially a factor that doesn't blow up as the requirements get large.

First let me get the constraint into a usable form, because "edge-disjoint paths" is a routing
statement and I'd rather have a counting statement. Menger: the max number of edge-disjoint `u`-`v`
paths equals the min number of edges separating `u` from `v`. So "at least `r(uv)` edge-disjoint
paths between `u` and `v`" is the same as "every cut that splits `u` from `v` keeps at least `r(uv)`
of my chosen edges." If I aggregate over all pairs and define, for a vertex set `S`,
`f(S) = max over pairs (u,v) split by S of r(uv)`, then my subgraph `H` is feasible exactly when
`|delta_H(S)| >= f(S)` for every `S`. Good — now the whole problem is: pick a min-cost edge set whose
cut-degree dominates the function `f`. Both endpoints of the problem are about cuts now, which is what
I can compute with max-flow.

I also need to know what kind of set function this `f` is, because the rounding proof will have to
move tight cuts around without breaking feasibility. Take two sets `A` and `B`, and split the ground
set into four atoms: `P=A∩B`, `Q=A\B`, `R=B\A`, and `W=V\(A∪B)`. A pair that witnesses `f(A)` must be
of type `P-R`, `P-W`, `Q-R`, or `Q-W`. Those four types are cut, respectively, by
`A∩B` and `B\A`, by `A∩B` and `A∪B`, by `A\B` and `B\A`, and by `A\B` and `A∪B`. A pair that
witnesses `f(B)` must be of type `P-Q`, `P-W`, `R-Q`, or `R-W`, and those are cut, respectively, by
`A∩B` and `A\B`, by `A∩B` and `A∪B`, by `B\A` and `A\B`, and by `B\A` and `A∪B`. Now I check the
four types for the `A` witness against the four types for the `B` witness. In every combination I can
put the two witnessed requirements on distinct members of `A∩B, A∪B`, which gives the
union/intersection inequality, or on distinct members of `A\B, B\A`, which gives the difference
inequality. The only two combinations not covered immediately are `P-W` against `R-Q` and `Q-R`
against `P-W`. In those, the pair witnessing `f(A)` also crosses `B`, so its requirement is at most
`f(B)`, and the pair witnessing `f(B)` also crosses `A`, so its requirement is at most `f(A)`. The two
maxima are equal, and either the union/intersection pair or the two difference sets carry that common
value twice. That is exactly
`f(A)+f(B) <= f(A∪B)+f(A∩B)` or
`f(A)+f(B) <= f(A\B)+f(B\A)`. This weak supermodularity is the piece of structure that makes
uncrossing possible.

So the relaxation writes itself. Put a variable `x_e in [0,1]` on each edge, minimize `sum c_e x_e`,
and for every `S` impose `x(delta(S)) := sum_{e in delta(S)} x_e >= f(S)`. The LP optimum is a lower
bound on the integral optimum, so if I can round a fractional solution to an integral one losing only
a constant factor, I'm done. The catch is obvious: there are exponentially many cut constraints, one
per subset. I can separate over them — given `x`, the most violated cut for a pair is just the min
`u`-`v` cut under weights `x_e`, a max-flow; all pairs can be handled via a Gomory-Hu tree. For now
assume I can get an optimal fractional `x`.

What does the prior art do with this? The primal-dual augmentation line — Goemans, Goldberg, Plotkin,
Shmoys, Tardos, Williamson 1994, building on Goemans-Williamson's constrained-forest primal-dual —
raises connectivity one unit at a time. At layer `k` you already have a subgraph that's `(k-1)`-
connected where required, and you run a `0/1` cut-covering primal-dual step to lift the still-deficient
pairs to `k`. Each layer is a clean `2`-approximation against its residual, but the layers *stack*:
you pay for `r_max` of them and the bound comes out around `2 H(r_max) = 2(1 + 1/2 + ... + 1/r_max)`.
That logarithmic dependence on the requirement is exactly the thing I want to kill. The reason it's
there is structural: the analysis never looks at the whole problem at once, it charges connectivity
layer by layer. So the lesson is — don't decompose by connectivity level. Reason about the entire LP
at one shot.

If I'm going to reason about the whole LP, the most natural move is threshold rounding: solve the LP,
round up every edge with `x_e >= 1/2` to `1`, throw away the rest, done. Why `1/2`? Because if every
edge I commit to had `x_e >= 1/2`, then `c_e <= 2 c_e x_e`, so the committed cost is at most
`2 * sum c_e x_e = 2 * LP <= 2 * OPT`. A factor `2`, independent of `r_max`. That's the target ratio,
sitting right there — *if* the rounding is valid.

But it isn't valid as stated, and I can see two separate problems. One: after I round up the
`>= 1/2` edges, the remaining requirements aren't met — rounding up a half-ish set doesn't on its own
satisfy the cuts. Two, and worse: what if the LP solution has *no* edge with `x_e >= 1/2`? Picture a
Petersen graph with unit costs and requirement `1` between every pair: it is three-edge-connected, so
the uniform assignment `x_e = 1/3` satisfies every cut constraint, and the optimum face can contain
that completely non-half-paid point. If every coordinate is below `1/2`, threshold rounding at `1/2`
commits to nothing and makes no progress, and if I drop the threshold to `1/3` my factor becomes `3`.
So the whole plan hinges on a structural question I don't yet know the answer to: **is there always an
edge with `x_e >= 1/2`?**

Let me not give up on `1/2` yet, because the answer might depend on *which* fractional solution I
take. An arbitrary optimal point of the polytope can be ugly — the optimum face can contain a point
that's `1/3` everywhere. But the *vertices* of the polytope are special: a basic feasible solution is
the unique solution of `|E|` linearly independent tight constraints. Vertices have rigid combinatorial
structure that interior points don't, and an LP solver can be made to hand me a vertex. So the precise
question is: **at a vertex `x` of a nonzero residual LP, with `0 < x_e < 1` on its support, must some
`x_e >= 1/2`?** If yes, threshold-`1/2` rounding has fuel at every step, and I can iterate: round those
edges, fix them, re-solve on what's left, repeat. Let me try to prove it, and if it's false the
counterexample will tell me the right threshold.

To get a grip on a vertex I need to understand which cut constraints are tight together. I am looking
at row vectors on the positive support: if `x_e = 0`, that edge can be deleted from the current
coordinate space, and if `x_e >= 1/2`, the lemma I want is already finished. The tight cut constraints
are the sets `S` with `x(delta(S)) = f(S)`. There can be a horrendous tangle of them. But the cut
function is gentle: for any weights, `x(delta(.))` is submodular and, being symmetric, also
posimodular:
`x(delta(S)) + x(delta(T)) >= x(delta(S∪T)) + x(delta(S∩T))`,
`x(delta(S)) + x(delta(T)) >= x(delta(S\T)) + x(delta(T\S))`.
And the requirement `f` is weakly supermodular: for any `A, B`, at least one of
`f(A)+f(B) <= f(A∪B)+f(A∩B)` or `f(A)+f(B) <= f(A\B)+f(B\A)` holds. Take two tight sets `S, T` that
cross (all four of `S∩T`, `S\T`, `T\S`, and `V \ (S∪T)` nonempty), and take whichever weakly-super
inequality `f` gives me — say the union/intersection one. Then the chain is
`f(S) + f(T) = x(delta(S)) + x(delta(T)) >= x(delta(S∪T)) + x(delta(S∩T)) >= f(S∪T) + f(S∩T) >= f(S) + f(T)`.
The first equality is tightness of `S, T`; the next is submodularity of the cut; the next is LP
feasibility at `S∪T` and `S∩T`; the last is weak supermodularity of `f`. The chain starts and ends at
the same number, so every inequality is an equality. That hands me two things at once. `S∪T` and `S∩T`
are *also tight*. And submodularity held with equality — but the characteristic-vector identity says
`chi(delta(S)) + chi(delta(T)) = chi(delta(S∪T)) + chi(delta(S∩T)) + 2 chi(E(S\T, T\S))`,
and `x(delta(S)) + x(delta(T)) = x(delta(S∪T)) + x(delta(S∩T))` forces `x(E(S\T, T\S)) = 0`; since
every edge on the support has `x_e > 0`, there are *no support edges* between `S\T` and `T\S`. So the
identity on the support collapses to
`chi(delta(S)) + chi(delta(T)) = chi(delta(S∪T)) + chi(delta(S∩T))` — the four
constraint-vectors are linearly dependent in the obvious way, meaning replacing `{S, T}` by
`{S∪T, S∩T}` keeps the linear span of my tight constraints unchanged. (If `f` had instead given the
`S\T, T\S` inequality, the same chain with the posimodular cut inequality replaces `{S,T}` by
`{S\T, T\S}` and kills edges between `S∩T` and `V \ (S∪T)` instead.) Either way the crossing row can
be written from uncrossed tight rows plus the row it crossed.

I should be careful about the word "less," because a blind replacement can create new crossings
elsewhere. Let `Tight` be the family of all tight sets, and take a maximal laminar subfamily `M` of
`Tight`. I want `M` to span every tight row. Suppose it doesn't. Then there is a tight set `S` whose
row is not in `span(M)`; choose such an `S` crossing as few sets of `M` as possible. Since `M` is
maximal laminar, `S` crosses some `L in M`. The row identity above says either
`chi(delta(S)) = chi(delta(S\L)) + chi(delta(L\S)) - chi(delta(L))` or
`chi(delta(S)) = chi(delta(S∩L)) + chi(delta(S∪L)) - chi(delta(L))`, with the two new sets tight.
Because `chi(delta(S))` is not in `span(M)` and `chi(delta(L))` is, at least one of the two new tight
rows is also outside `span(M)`. In the first case, say it is `S\L`. Any member of the laminar family
that crosses `S\L` also crosses `S`, while `L` crosses `S` but not `S\L`; so `S\L` crosses fewer sets
of `M` than `S` does, contradicting the choice of `S`. If the outside row is `L\S`, `S∩L`, or `S∪L`,
laminarity with `L` gives the same comparison: every set of `M` crossing that candidate must already
cross `S`, and `L` itself crossed `S` but no longer crosses the candidate. So `span(M)=span(Tight)`. I
pull a maximal linearly independent subfamily `L` from this laminar `M`. In the only case left to
contradict, every remaining coordinate has `0 < x_e < 1/2`, so no lower or upper variable bound is
active in the support space. Since `x` is a vertex, the cut rows have to pin all `|E|` support
variables by themselves: `|L| = |E|`, the vectors `chi(delta(S))` for `S in L` are independent, and
`L` is laminar. That's my handle: a vertex is described by a laminar family of tight cuts of size
exactly `|E|`.

Now back to the question — must some `x_e >= 1/2`? Suppose not. If an edge has `x_e = 0`, I can delete
it from the support and keep the same vertex argument in the smaller space; if an edge has `x_e = 1`,
then I already have an edge above the threshold. So the only dangerous case is
`0 < x_e < 1/2` for every support edge. I want a contradiction with `|L| = |E|`.

The disjoint case tells me what the contradiction should look like. If `L` is just a collection of
maximal disjoint sets, then each `S in L` has `x(delta(S)) = f(S) >= 1`. With every edge below `1/2`,
two boundary edges cannot reach one unit, so each set needs at least three boundary endpoints. The
sets are disjoint, so `m = |L| = |E|` sets ask for at least `3m` endpoints while `m` edges provide only
`2m`. That is the right pressure: the vertex wants too many independent tight cuts for the number of
edge endpoints available.

Nesting is where that endpoint count leaks. Leaves still demand many boundary endpoints, but internal
sets can share edges with their children; a parent with one child can look as if it contributes a new
equation almost for free. Independence already rules out the most naive leak: if a set `S` has a single
child `C` and owns no edge endpoint outside `C`, then `delta(S)=delta(C)`; if it owns exactly one such
endpoint, then `x(delta(S))` and `x(delta(C))` differ by exactly one fractional `x_e`; two integer
right-hand sides cannot differ by a number strictly between `0` and `1/2`. That pushes a rough endpoint
argument to `1/3`, but not to `1/2`. I need the count to charge internal sets without double-counting
the same endpoint again and again.

Tokens fit the shape better than raw endpoints. Give each edge one token. For edge `e=(u,v)`, if there
is a smallest set of `L` containing `u`, send `x_e` to it; if there is a smallest set containing `v`,
send another `x_e` to it. If there is a smallest set containing both endpoints, send `1 - 2x_e` to it.
All assigned pieces are positive because `0 < x_e < 1/2`, and the pieces from one edge sum to at most
`x_e + x_e + (1 - 2x_e) = 1`; if some required smallest set does not exist, that part of the token
simply remains unassigned.

Now fix a set `S in L` with children `R_1, ..., R_k`. I subtract the child tight equations from the
parent tight equation:
`x(delta(S)) - sum_i x(delta(R_i)) = f(S) - sum_i f(R_i)`.
Let `O = S \ (R_1 ∪ ... ∪ R_k)`, the part of `S` owned directly by `S`. The only edges with nonzero
coefficient in the subtraction fall into three classes: `A`, edges from `O` to `V \ S`; `B`, edges from
`O` to one child; and `C`, edges between two different children. Then the subtraction is
`x(A) - x(B) - 2x(C) = f(S) - sum_i f(R_i)`.
An `A` edge gives `S` exactly `x_e` by the endpoint rule. A `B` edge gives `S` `x_e` from the endpoint
outside the children and `1 - 2x_e` from the both-endpoints rule, for a total of `1 - x_e`. A `C` edge
gives `S` `1 - 2x_e` by the both-endpoints rule. There is one more token-only class, `D`: edges with
both endpoints in `O`. They cancel out of the tight-equation subtraction, but all three pieces of their
one token go to `S`, so each contributes exactly `1`. The amount collected by `S` is therefore
`sum_{e in A} x_e + sum_{e in B} (1 - x_e) + sum_{e in C} (1 - 2x_e) + |D|`
`= x(A) + |B| - x(B) + |C| - 2x(C) + |D|`
`= |B| + |C| + |D| + f(S) - sum_i f(R_i)`.
The last expression is an integer. The collected amount is strictly positive: if it were zero, every
positive token piece into `S` would be absent, so `A`, `B`, `C`, and `D` would all be empty; then in
particular no edge has a nonzero coefficient in
`chi(delta(S)) - sum_i chi(delta(R_i))`, contradicting the independence of the laminar basis. A
positive integer is at least one. Every set in `L` collects at least one token.

But at least one positive piece is never collected. Take a maximal set `R in L`. It cannot be `V`,
because `f(V)=0` and the vector `chi(delta(V))` is zero, so it could not be in an independent basis.
It also cannot have `delta(R)` empty on the support, because then `x(delta(R))=f(R)=0` and its vector
would again be zero. So some support edge crosses `R`. No laminar set contains both endpoints of that
edge: any such set would overlap `R` and, by laminarity, would have to strictly contain `R`, contrary to
maximality. The `1 - 2x_e` piece of that crossing edge is positive and unassigned. Total collected
token mass is therefore strictly less than the `|E|` unit tokens available, while the `|L|` sets each
collect at least one. That forces `|L| < |E|`, contradicting `|L| = |E|`. The assumption was false:
some edge has `x_e >= 1/2`. The threshold survives exactly because the token `1 - 2x_e` stays positive
below one half; beyond one half this proof has no slack to spend.

Now the rounding has fuel. But I still have the *first* problem from earlier: rounding the big edges to
`1` doesn't satisfy the cuts by itself. The fix is to not finish in one shot — iterate. Round every
edge with `x_e >= 1/2` up to `1`, *fix* them permanently, and solve the *residual* problem on the rest.
For this to make sense the residual must be the same kind of problem, or my structural theorem won't
apply at the next step. After committing an edge set `F`, what's left to require? Each cut `S` already
has `|delta_F(S)|` edges from `F`, so the remaining requirement is `g(S) = f(S) - |delta_F(S)|`. Is `g`
still weakly supermodular? The cut function `h(S)=|delta_F(S)|` is symmetric and submodular, hence also
posimodular. If `f(A)+f(B) <= f(A∪B)+f(A∩B)`, then submodularity gives
`h(A)+h(B) >= h(A∪B)+h(A∩B)`, and subtracting the second inequality from the first gives
`g(A)+g(B) <= g(A∪B)+g(A∩B)`. If `f` instead gives the set-difference inequality, posimodularity of
`h` gives the matching `h(A)+h(B) >= h(A\B)+h(B\A)`, and the same subtraction gives
`g(A)+g(B) <= g(A\B)+g(B\A)`. So `g` is weakly supermodular, the residual is another instance of
"cover a weakly-super function by a graph," and my theorem applies again. The iteration is legitimate.

So the algorithm: maintain a fixed set `F`, initially empty. While the residual requirement
`g = f - |delta_F|` is not everywhere `<= 0`, solve the covering LP for `g` to a *vertex*, round up
all edges with `x_e >= 1/2` (the theorem guarantees at least one), add them to `F`, and repeat. When
`g <= 0` everywhere, `F` is feasible. Termination is clear — each round fixes at least one new edge.

Now the factor. Let me charge round by round. In a round with vertex solution `x*` and rounded set
`R = { e : x*_e >= 1/2 }`, the cost I commit is
`sum_{e in R} c_e <= 2 sum_{e in R} c_e x*_e`. I need the right-hand side to telescope, so I should
charge only the fractional mass on the edges I just fixed, not the whole current LP value. Take `x*`
and delete its `R`-coordinates.
For any cut `S`, the remaining fractional mass is
`x*(delta(S)) - x*_R(delta(S)) >= g_current(S) - x*_R(delta(S)) >= g_current(S) - |delta_R(S)| = g_next(S)`,
because every rounded coordinate is at most `1`. So the restriction is feasible for the next residual
LP, and `LP(next) <= LP(current) - sum_{e in R} c_e x*_e`. Telescoping over
all rounds,
`cost(F) = sum_rounds sum_{e in R} c_e <= sum_rounds 2 sum_{e in R} c_e x*_e`,
and that is at most
`2 sum_rounds (LP(current) - LP(next)) <= 2 * LP(initial)`.
And `LP(initial) <= OPT`. So `cost(F) <= 2 * OPT`. A factor `2`, with no dependence on `r_max` — the
thing the layered primal-dual couldn't give, because I reasoned about the whole LP vertex at once
instead of one connectivity layer at a time. The `2` traces straight back to the `1/2` in the
structural theorem: I only ever commit to edges that are at least half-paid-for by the LP, so I
overpay by at most a factor of two, every round, forever.

Last piece: actually solving the covering LP, since it has a constraint per cut. In the polynomial
version I separate against the cut residual directly. Given a candidate `x`, build the graph with
capacity `x_e` on each free edge and capacity `1` on each fixed edge. A cut is violated exactly when
this total capacity is below `f(S)`, which is the same as saying some demanded pair split by the cut
has a min `u`-`v` cut value below `r(uv)`. A Gomory-Hu tree gives all those pair min-cut values with
`n - 1` max-flows; when a demanded pair is short, removing the minimum-weight tree edge on its path
gives a violated shore `S`, and I add
`sum_{free e in delta(S)} x_e >= f(S) - |delta_F(S)|`. Once the LP solution has no violated cut, it is
optimal for the full residual cut relaxation. For the proof I ask the LP routine for a basic optimum;
in the runnable code below, the active relaxation is solved by CBC/CLP and the progress guard catches
any failure to return a half-paid edge. Let me write it.

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

So the causal chain, start to finish: Menger turns "edge-disjoint paths" into "every cut has enough
edges," which makes the requirement a weakly-supermodular cut function and gives a covering LP whose
optimum lower-bounds `OPT`. I want each rounding step to look at the whole residual LP rather than the
layered primal-dual that pays `H(r_max)`, so I aim threshold rounding at `1/2`, where committing only
half-paid edges costs a factor `2`. The threshold only works if every vertex of a residual LP that is
not already covered has a `>= 1/2` coordinate; uncrossing the tight cuts (cut submodularity +
posimodularity against weak supermodularity of `f`) shows such a vertex is pinned by a laminar family of
tight cuts of size `|E|`, and a token / endpoint count over that laminar family proves that "all
coordinates `< 1/2`" would force `|L| < |E|`, a contradiction — so a `>= 1/2` edge always exists until
the residual is already covered, and `1/2` is exactly the limit the count permits.
The residual `f - |delta_F|` stays weakly supermodular, so I round the `>= 1/2` edges, fix them, and
recurse; telescoping the per-round `2 * (LP drop)` bound gives `cost(F) <= 2 * LP <= 2 * OPT`,
independent of the requirements. The LP is solved by Gomory-Hu / max-flow separation over cuts.
