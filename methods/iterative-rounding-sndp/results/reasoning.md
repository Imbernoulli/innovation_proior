OK, let me think this through from scratch. I have an undirected graph with edge costs, and for each
pair of vertices a requirement `r(uv)` — the number of edge-disjoint `u`-`v` paths the final subgraph
must contain so the network survives `r(uv) - 1` failures. I want the cheapest subgraph meeting all
of them. Finding the exact optimum is hopeless in polynomial time, so I'm not chasing `OPT`; I'm
chasing a *guarantee* — a feasible subgraph I can build fast and prove is never more than some fixed
factor times `OPT`, and crucially a factor that doesn't blow up as the requirements get large.

First let me get the constraint into a usable form, because "edge-disjoint paths" is a routing
statement and I'd rather have a counting statement. Menger: the max number of edge-disjoint `u`-`v`
paths equals the min number of edges separating `u` from `v`. So "at least `r(uv)` edge-disjoint
paths between `u` and `v`" is the same as "every cut that splits `u` from `v` keeps at least `r(uv)`
of my chosen edges." If I aggregate over all pairs and define, for a vertex set `S`,
`f(S) = max over pairs (u,v) split by S of r(uv)`, then my subgraph `H` is feasible exactly when
`|delta_H(S)| >= f(S)` for every `S`. Good — now the whole problem is: pick a min-cost edge set whose
cut-degree dominates the function `f`. Both endpoints of the problem are about cuts now, which is what
I can compute with max-flow.

So the relaxation writes itself. Put a variable `x_e in [0,1]` on each edge, minimize `sum c_e x_e`,
and for every `S` impose `x(delta(S)) := sum_{e in delta(S)} x_e >= f(S)`. The LP optimum is a lower
bound on the integral optimum, so if I can round a fractional solution to an integral one losing only
a constant factor, I'm done. The catch is obvious: there are exponentially many cut constraints, one
per subset. I'll worry about *solving* the LP later (I can separate over it — given `x`, the most
violated cut for a pair is just the min `u`-`v` cut under weights `x_e`, a max-flow; all pairs via a
Gomory-Hu tree). For now assume I can get an optimal fractional `x`.

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
simple cycle on three required terminals each needing connectivity `2`; an LP can sit at `x_e = 2/3`
on a few edges, or you can cook up fractional solutions where everything hovers around `1/3`. If every
coordinate is below `1/2`, threshold rounding at `1/2` commits to nothing and makes no progress, and
if I drop the threshold to `1/3` my factor becomes `3`. So the whole plan hinges on a structural
question I don't yet know the answer to: **is there always an edge with `x_e >= 1/2`?**

Let me not give up on `1/2` yet, because the answer might depend on *which* fractional solution I
take. An arbitrary optimal point of the polytope can be ugly — the optimum face can contain a point
that's `1/3` everywhere. But the *vertices* of the polytope are special: a basic feasible solution is
the unique solution of `|E|` linearly independent tight constraints. Vertices have rigid combinatorial
structure that interior points don't, and an LP solver can be made to hand me a vertex. So the precise
question is: **at a vertex `x` (with `0 < x_e < 1` on its support), must some `x_e >= 1/2`?** If yes,
threshold-`1/2` rounding has fuel at every step, and I can iterate: round those edges, fix them, re-
solve on what's left, repeat. Let me try to prove it, and if it's false the counterexample will tell
me the right threshold.

To get a grip on a vertex I need to understand which cut constraints are tight together. The tight
constraints are the sets `S` with `x(delta(S)) = f(S)`. There can be a horrendous tangle of them. But
the cut function is gentle: for any weights, `x(delta(.))` is submodular and, being symmetric, also
posimodular:
`x(delta(S)) + x(delta(T)) >= x(delta(S∪T)) + x(delta(S∩T))`,
`x(delta(S)) + x(delta(T)) >= x(delta(S\T)) + x(delta(T\S))`.
And the requirement `f` is weakly supermodular: for any `A, B`, at least one of
`f(A)+f(B) <= f(A∪B)+f(A∩B)` or `f(A)+f(B) <= f(A\B)+f(B\A)` holds. Watch what happens when two tight
sets `S, T` *cross* (all four of `S∩T, S\T, T\S, complement` nonempty). Take whichever weakly-super
inequality `f` gives me — say the union/intersection one. Then chain it:
`f(S) + f(T) = x(delta(S)) + x(delta(T)) >= x(delta(S∪T)) + x(delta(S∩T)) >= f(S∪T) + f(S∩T) >= f(S) + f(T)`.
The first equality is tightness of `S, T`; the next is submodularity of the cut; the next is LP
feasibility at `S∪T` and `S∩T`; the last is weak supermodularity of `f`. The chain starts and ends at
the same number, so every inequality is an equality. That hands me two things at once. `S∪T` and `S∩T`
are *also tight*. And submodularity held with equality — but the characteristic-vector identity says
`chi(delta(S)) + chi(delta(T)) = chi(delta(S∪T)) + chi(delta(S∩T)) + 2 chi(E(S\T, T\S))`,
and `x(delta(S)) + x(delta(T)) = x(delta(S∪T)) + x(delta(S∩T))` forces `x(E(S\T, T\S)) = 0`; since
every edge on the support has `x_e > 0`, there are *no* edges between `S\T` and `T\S`. So the identity
collapses to `chi(delta(S)) + chi(delta(T)) = chi(delta(S∪T)) + chi(delta(S∩T))` — the four
constraint-vectors are linearly dependent in the obvious way, meaning replacing `{S, T}` by
`{S∪T, S∩T}` keeps the linear span of my tight constraints unchanged. (If `f` had instead given the
`S\T, T\S` inequality, the same chain with the posimodular cut inequality replaces `{S,T}` by
`{S\T, T\S}` and kills edges between `S∩T` and the complement instead.) Either way an uncrossing step
removes a crossing without shrinking the span, and the quantity `sum_{S} |S|^2` strictly increases, so
it terminates. I end with a *laminar* family of tight sets that still spans everything. From that
laminar family I can pull a maximal linearly independent subfamily `L`, and since `x` is a vertex it's
the unique solution of `|E|` independent tight constraints, so `|L| = |E|`, the vectors `chi(delta(S))`
for `S in L` are independent, and `L` is laminar. That's my handle: a vertex is described by a laminar
family of tight cuts of size exactly `|E|`.

Now back to the question — must some `x_e >= 1/2`? Suppose not: `0 < x_e < 1/2` for every edge. I want
a contradiction with `|L| = |E|`. Let me start with the easiest possible laminar family to build
intuition, then generalize.

Suppose `L` is a bunch of *disjoint* sets (all maximal, no nesting). Each `S in L` has `f(S) >= 1` and
`x(delta(S)) = f(S) >= 1`. If every `x_e < 1/2`, then to make `x(delta(S)) >= 1` I need at least three
edges in `delta(S)` (two could give at most just under `1`). So each `S` swallows at least three edge
*endpoints*. The sets are disjoint, so the `m = |L| = |E|` sets demand at least `3m` endpoints. But
`m` edges have only `2m` endpoints total. `3m > 2m` — contradiction. So in the all-disjoint case some
edge is `>= 1/2`. The shape of the argument is clear: count endpoints, the "below `1/2`" assumption
forces each set to be expensive in endpoints, and there just aren't enough endpoints to go around.

Now let the family nest. With internal nodes the naive count weakens, and let me first see how far it
weakens. If every internal node has at least two children, then in a forest the number of internal
nodes `h` is less than the number of leaves `k`. If `x_e < 1/2` forced each leaf to have `>= 3` edges
leaving it I'd get `3k` endpoints — but I have to be careful, edges can run between sets. The cleanest
version: if I only insist `x_e < 1/3` to start, each leaf needs `>= 4` edges leaving it, giving `>= 4k`
endpoints from the disjoint leaves; with `h < k` and `h + k = m` we get `k > m/2`, so `4k > 2m`,
again more endpoints than exist. So even with nesting I easily get *some* `x_e >= 1/3`. The `1/3`
falls out cheaply; the trouble is squeezing it up to `1/2`, and the nesting plus "internal node with
exactly one child" cases are where the naive endpoint count loses too much.

The single-child case is the subtle one — a set `S` with one child `C` and no endpoints of its own
seems to add a constraint for free. But it can't, because the constraints are independent. If `S` had
*no* endpoint owned by it (no edge whose smaller containing set is `S`), then `delta(S) = delta(C)`
exactly and `chi(delta(S)) = chi(delta(C))` — a dependence, contradiction. If `S` owned exactly one
endpoint, of edge `e`, then `x(delta(S)) = x(delta(C)) +/- x_e`, but both `x(delta(S)) = f(S)` and
`x(delta(C)) = f(C)` are positive integers while `x_e in (0,1)` — impossible. So every set owns at
least two endpoints. That patches the single-child leak and re-confirms `1/3` in general.

But `1/3` is a factor-`3` algorithm, and I want `2`. The endpoint count is too lossy because it treats
every set the same; I need an invariant that *propagates up the laminar tree* and ends up reducing the
general case to the easy disjoint case. Let me hunt for the right potential. For a set `S in L`, let
`alpha(S)` be the number of sets of `L` inside `S` (including `S` itself), and `beta(S)` the number of
edges with *both* endpoints inside `S`. I'll conjecture the invariant

`f(S) >= alpha(S) - beta(S)`  for all `S in L`,

and try to prove it bottom-up. Intuition: `alpha(S)` is how many tight constraints live inside `S` and
`beta(S)` is how many edges are "used up" internally; the requirement `f(S)` has to be at least the
net number of internal constraints not absorbed by internal edges. If this holds, summing over the
maximal sets (roots) `R_1, ..., R_h` gives `sum f(R_i) >= sum alpha(R_i) - sum beta(R_i) = |L| - sum beta(R_i) = m - (#internal edges)`. Every support edge must cross at least one tight set, or its column in
all the tight equations would be zero and the vertex would not be pinned down; so every edge is either
internal to one root or crosses a root. Thus `m - (#edges internal to the roots)` is exactly the number of
edges crossing the roots. So I'm back to disjoint sets `R_i` with their crossing edges, and the easy
endpoint contradiction finishes it under `x_e < 1/2`. Let me prove the invariant.

Leaf `S`: `alpha(S) = 1`, `beta(S) = 0`, and `f(S) >= 1`. So `f(S) >= 1 = alpha - beta`. Base case
holds.

Internal `S` with children `C_1, ..., C_k`. Let `gamma(S)` be the number of edges with both endpoints
inside `S` but not inside any single child — edges that either run between two different children or
cross exactly one child without leaving `S`. Then `beta(S) = gamma(S) + sum_i beta(C_i)`, because an
edge internal to `S` is either internal to some child or counted by `gamma`. Apply the inductive
hypothesis to each child, `beta(C_i) >= alpha(C_i) - f(C_i)`:
`beta(S) >= gamma(S) + sum_i alpha(C_i) - sum_i f(C_i)`.
And `sum_i alpha(C_i) = alpha(S) - 1` (everything inside `S` except `S` itself is inside exactly one
child). So `beta(S) >= gamma(S) + alpha(S) - 1 - sum_i f(C_i)`. To conclude `beta(S) >= alpha(S) - f(S)`
I just need

`gamma(S) >= sum_i f(C_i) - f(S) + 1`.

Now compute the right side from the tight constraints. Subtract the children's tight equalities from
the parent's: classify edges by how they meet `S` and the children. Let `E_cc` be edges between two
different children, `E_cp` edges crossing exactly one child but staying inside `S`, `E_po` edges
crossing `S` but no child, `E_co` edges crossing both a child and `S`. Then
`f(S) = x(delta(S)) = sum_{E_co} x_e + sum_{E_po} x_e` and
`sum_i f(Ci) = sum_i x(delta(Ci)) = sum_{E_co} x_e + sum_{E_cp} x_e + 2 sum_{E_cc} x_e`
(an `E_cc` edge crosses two children so it's counted twice). Subtract:
`sum_i f(Ci) - f(S) = 2 sum_{E_cc} x_e + sum_{E_cp} x_e - sum_{E_po} x_e`,
so the inequality I need becomes
`gamma(S) >= 2 sum_{E_cc} x_e + sum_{E_cp} x_e - sum_{E_po} x_e + 1`,
and recall `gamma(S) = |E_cc| + |E_cp|`. Split into cases.

If `gamma(S) = 0`: then `E_cc` and `E_cp` are empty. The right side is `- sum_{E_po} x_e + 1`. By
independence, `chi(delta(S))` can't be a combination of the `chi(delta(C_i))`, so `E_po` is nonempty
and `sum_{E_po} x_e > 0`, making the right side strictly less than `1`. But the left side
`sum_i f(Ci) - f(S) + 1` is an integer, so it's `<= 0 = gamma(S)`. Good.

If `gamma(S) >= 1`: drop the `- sum_{E_po} x_e` term (it only helps) and bound
`2 sum_{E_cc} x_e + sum_{E_cp} x_e + 1`. Here's where the assumption `x_e < 1/2` finally bites. For
`E_cc`, `2 x_e < 1` per edge, so `2 sum_{E_cc} x_e < |E_cc|` whenever `E_cc` is nonempty. For `E_cp`,
`x_e < 1/2` gives `sum_{E_cp} x_e < |E_cp|/2 <= |E_cp|`. Since `gamma(S) = |E_cc| + |E_cp| >= 1`, at
least one of these strict bounds is active, and together
`2 sum_{E_cc} x_e + sum_{E_cp} x_e < |E_cc| + |E_cp| = gamma(S)`. Therefore
`sum_i f(Ci) - f(S) + 1 <= 2 sum_{E_cc} x_e + sum_{E_cp} x_e + 1 < gamma(S) + 1`, and being an integer
strictly below `gamma(S) + 1` it's `<= gamma(S)`. Exactly the inequality I needed. The invariant
`f(S) >= alpha(S) - beta(S)` propagates to `S`.

So under `x_e < 1/2` everywhere the invariant holds at every set, hence at the roots, hence
`sum f(R_i) >= (number of edges crossing the roots)`. Call those crossing edges `E'` and treat the
roots as disjoint sets: each `R_i` needs `x(delta(R_i)) = f(R_i) >= 1` with all weights below `1/2`,
so it owns more than `2 f(R_i)` crossing endpoints — at least `2 f(R_i) + 1`. Summing over the disjoint
roots, the total endpoints among `E'` is at least `2 sum f(R_i) + h > 2 |E'|`. But `|E'|` edges have
only `2 |E'|` endpoints. Contradiction. So the assumption fails: **some edge has `x_e >= 1/2`.** The
threshold survives. It's `1/2` precisely because that's where the endpoint count flips — `2 x_e < 1`
on cross-child edges and `x_e < 1/2` on single-cross edges is exactly what makes `gamma(S)` dominate;
any larger threshold and the counting no longer closes, which is why `1/2`, not more, is the
guarantee.

Let me re-derive this a second way, because the endpoint bookkeeping with `E_cc/E_cp/E_po` is fiddly
and I want to be sure. Same setup — vertex `x`, all `0 < x_e < 1/2`, laminar `L` with `|L| = |E|`.
Give each edge one token. Edge `e = (u, v)` hands its token out by two rules. Rule 1: send `x_e` to
the smallest set of `L` containing `u`, and another `x_e` to the smallest set containing `v`. Rule 2:
send `1 - 2 x_e` to the smallest set containing *both* `u` and `v` (if one exists). Since `0 < x_e < 1/2`,
all of `x_e`, `1 - x_e`, `1 - 2 x_e` are strictly positive, and the three pieces sum to
`x_e + x_e + (1 - 2 x_e) = 1` — at most one whole token leaves each edge (Rule 2 may find no set, in
which case some token is left over). Now count what a set `S in L` with children `R_1, ..., R_k`
collects. Subtract the children's tight equalities from `S`'s:
`x(delta(S)) - sum_i x(delta(R_i)) = f(S) - sum_i f(R_i)`, an integer. The edges that survive this
subtraction are exactly three types relative to `S` and `∪ R_i`: `A` = one endpoint in `S` and none in
the children (smallest containing set is `S` for that endpoint, contributes `+x_e` to the left via
Rule 1); `B` = one endpoint in some child, the other endpoint in `S` but outside all children (Rule 1
sends `x_e` and Rule 2 sends `1 - 2 x_e`, together `1 - x_e`); `C` = both endpoints in two different
children (Rule 2 sends `1 - 2 x_e`). And the subtraction reads `x(A) - x(B) - 2 x(C) = f(S) - sum f(R_i)`.
So the tokens `S` gathers total
`sum_A x_e + sum_B (1 - x_e) + sum_C (1 - 2 x_e) = x(A) + |B| - x(B) + |C| - 2 x(C) = |B| + |C| + (f(S) - sum_i f(R_i))`,
an integer. It's also strictly positive, because `A ∪ B ∪ C` can't be empty — if it were,
`chi(delta(S)) = sum_i chi(delta(R_i))`, contradicting independence — and each surviving edge donates
a strictly positive amount. A positive integer is `>= 1`, so **every set in `L` collects at least one
token.** Finally, take a maximal set `R in L`. Its vector is nonzero, so `f(R) = x(delta(R)) > 0` and
some edge crosses `R`; by laminarity, such an edge has no set containing both endpoints, because any such
set would have to strictly contain `R`. Its Rule-2 token `1 - 2 x_e > 0` is never claimed. There's leftover token mass. Total
tokens handed out and collected is therefore strictly less than the `|E|` tokens that existed, while
every one of the `|L|` sets grabbed at least one, forcing `|L| < |E|`. That contradicts `|L| = |E|`.
Same conclusion, cleaner: some `x_e >= 1/2`. (And I notice: if every `f(S)` were *even*, I could rescale
the rules — Rule 1 sends `x_e / 2`, Rule 2 sends `1 - x_e` — and the integrality of `(f(S) - sum f(R_i))/2`
pushes the threshold all the way to `1`, recovering the fact that the spanning-subgraph / tour LP with
even requirements always has an integral edge. Nice sanity check, but for general `f` it's `1/2`.)

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
`g(A)+g(B) <= g(A\B)+g(B\A)`. So `g` is weakly supermodular, the residual is another instance of "cover a weakly-super
function by a graph," and my theorem applies again. The iteration is legitimate.

So the algorithm: maintain a fixed set `F`, initially empty. While the residual requirement
`g = f - |delta_F|` is not everywhere `<= 0`, solve the covering LP for `g` to a *vertex*, round up
all edges with `x_e >= 1/2` (the theorem guarantees at least one), add them to `F`, and repeat. When
`g <= 0` everywhere, `F` is feasible. Termination is clear — each round fixes at least one new edge.

Now the factor. Let me charge round by round. In a round with vertex solution `x*` and rounded set
`R = { e : x*_e >= 1/2 }`, the cost I commit is
`sum_{e in R} c_e <= 2 sum_{e in R} c_e x*_e`. I need the right-hand side to telescope, so I should charge
only the fractional mass on the edges I just fixed, not the whole current LP value. Take `x*` and delete its `R`-coordinates;
the restriction is feasible for the next residual LP (because `g` dropped by exactly the cuts `R`
covers and stays in the class), so `LP(next) <= LP(current) - sum_{e in R} c_e x*_e`. Telescoping over
all rounds,
`cost(F) = sum_rounds sum_{e in R} c_e <= sum_rounds 2 sum_{e in R} c_e x*_e <= 2 sum_rounds (LP(current) - LP(next)) <= 2 * LP(initial)`.
And `LP(initial) <= OPT`. So `cost(F) <= 2 * OPT`. A factor `2`, with no dependence on `r_max` — the
thing the layered primal-dual couldn't give, because I reasoned about the whole LP vertex at once
instead of one connectivity layer at a time. The `2` traces straight back to the `1/2` in the
structural theorem: I only ever commit to edges that are at least half-paid-for by the LP, so I
overpay by at most a factor of two, every round, forever.

Last piece: actually solving the covering LP, since it has a constraint per cut. I separate against the
cut residual directly. Given a candidate `x`, build the graph with capacity `x_e` on each free edge and
capacity `1` on each fixed edge. A cut is violated exactly when this total capacity is below
`f(S)`, which is the same as saying some demanded pair split by the cut has a min `u`-`v` cut value below
`r(uv)`. A Gomory-Hu tree gives all those pair min-cut values with `n - 1` max-flows; when a demanded
pair is short, removing the minimum-weight tree edge on its path gives a violated shore `S`, and I add
`sum_{free e in delta(S)} x_e >= f(S) - |delta_F(S)|`. Feed violated cuts to the LP solver until none is
violated; a solver returning a basic optimal solution hands me the vertex my theorem needs. Let me write it.

```python
import networkx as nx
import pulp

def requirement_on_cut(S, r):
    # f(S) = max over pairs split by S of r(uv): the cut form of the demands
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
        status = prob.solve(solver)               # CBC/CLP returns a basic optimum for the active LP
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

So the causal chain, start to finish: Menger turns "edge-disjoint paths" into "every cut has enough
edges," which makes the requirement a weakly-supermodular cut function and gives a covering LP whose
optimum lower-bounds `OPT`. I want each rounding step to look at the whole residual LP rather than the layered
primal-dual that pays `H(r_max)`, so I aim threshold rounding at `1/2`, where committing only
half-paid edges costs a factor `2`. The threshold only works if a vertex always has a `>= 1/2`
coordinate; uncrossing the tight cuts (cut submodularity + posimodularity against weak supermodularity
of `f`) shows a vertex is pinned by a laminar family of tight cuts of size `|E|`, and a token / endpoint
count over that laminar family proves that "all coordinates `< 1/2`" would force `|L| < |E|`, a
contradiction — so a `>= 1/2` edge always exists, and `1/2` is exactly the limit the count permits.
The residual `f - |delta_F|` stays weakly supermodular, so I round the `>= 1/2` edges, fix them, and
recurse; telescoping the per-round `2 * (LP drop)` bound gives `cost(F) <= 2 * LP <= 2 * OPT`,
independent of the requirements. The LP is solved by Gomory-Hu / max-flow separation over cuts.
