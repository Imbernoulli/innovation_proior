## Research question

We are given an undirected graph `G = (V, E)` with a nonnegative cost `c_e` on every edge, and for
each unordered pair of vertices `u, v` an integer connectivity requirement `r(uv) >= 0`. We must
choose a minimum-cost subgraph `H` of `G` such that, for every pair, `H` contains at least `r(uv)`
edge-disjoint paths between `u` and `v`. This is the survivable network design problem (also called
the generalized Steiner network problem): build the cheapest network that keeps every required pair
connected even after up to `r(uv) - 1` edge failures.

The problem is NP-hard and APX-hard even in special cases, so an exact polynomial-time algorithm is
out of reach. The target is an approximation guarantee that does **not** degrade as the requirements
grow: a polynomial-time algorithm returning a feasible subgraph whose cost is at most a fixed
constant factor times the optimum `OPT`, for every instance and every requirement vector. The whole
difficulty is the proof of the factor — producing *a* feasible subgraph is easy; producing one with a
provable constant ratio is not.

The defining technical obstacle: the natural lower bound on `OPT` is a linear program with one
constraint per cut of the graph — exponentially many constraints — and the gap between that LP and
the integer optimum is what any rounding must overcome. We need a structural handle on the *vertices*
of that LP polytope strong enough to convert a fractional solution into an integral one while losing
only a constant factor, uniformly in the requirements.

## Background

**Edge-disjoint paths are cuts (Menger).** By Menger's theorem, the maximum number of edge-disjoint
`u`-`v` paths in a subgraph `H` equals the minimum number of edges whose removal separates `u` from
`v` in `H`. So requiring `r(uv)` edge-disjoint paths is exactly requiring that **every** cut
separating `u` from `v` contains at least `r(uv)` edges of `H`. Collecting the requirements pairwise,
for a vertex set `S` define
`f(S) = max_{u in S, v not in S} r(uv)`,
with `f(empty) = f(V) = 0`. Then `H` is feasible iff `|delta_H(S)| >= f(S)` for every `S subset V`,
where `delta_H(S)` is the set of edges of `H` with exactly one endpoint in `S`. The entire problem
collapses to: choose a minimum-cost edge set whose cut-degree dominates the function `f`.

**Weakly supermodular (skew-supermodular) requirement functions.** The function `f` above has a
structural property that turns out to be the linchpin. Call `f : 2^V -> Z` *weakly supermodular* if
`f(empty) = f(V) = 0` and for all `A, B subset V` at least one of
`f(A) + f(B) <= f(A union B) + f(A intersect B)` or
`f(A) + f(B) <= f(A \ B) + f(B \ A)`
holds. The cut-maximum `f(S) = max_{u in S, v not in S} r(uv)` is weakly supermodular: a short case
analysis on where the requirement-maximizing pair lies relative to `A` and `B` always validates one
of the two inequalities. This class is broader than the SNDP cut function and is the right level of
abstraction; the same machinery covers any weakly-supermodular `f`.

**The cut function is symmetric and submodular.** For any fixed edge set `F`, the map
`S -> |delta_F(S)|` is symmetric (`|delta_F(S)| = |delta_F(V \ S)|`) and submodular:
`|delta_F(S)| + |delta_F(T)| >= |delta_F(S union T)| + |delta_F(S intersect T)|`.
A symmetric submodular function is also *posimodular*:
`|delta_F(S)| + |delta_F(T)| >= |delta_F(S \ T)| + |delta_F(T \ S)|`.
The fractional version `x(delta(S)) = sum_{e in delta(S)} x_e` inherits both inequalities. These two
inequalities, together with the characteristic-vector identities
`chi(delta(S)) + chi(delta(T)) = chi(delta(S union T)) + chi(delta(S intersect T)) + 2 chi(E(S\T, T\S))`
and
`chi(delta(S)) + chi(delta(T)) = chi(delta(S\T)) + chi(delta(T\S)) + 2 chi(E(S intersect T, complement))`,
are the raw material for reasoning about which cut constraints can be simultaneously tight at a vertex
of the LP polytope.

**Linear programming relaxations and extreme points.** Relaxing the integral choice `x_e in {0,1}` to
`x_e in [0,1]` gives a covering LP. Its optimum lower-bounds `OPT`. A *basic feasible solution*
(extreme point / vertex) of a polytope in `R^E` is the unique solution of some `|E|` linearly
independent tight constraints drawn from the constraint system. Extreme points are the objects whose
support structure can be controlled — a generic optimal point need not have any nice coordinate, but
a vertex does, and LP solvers can be made to return a vertex. The governing prior fact is that for
covering an exponentially-large family of cut constraints, the tight constraints at a vertex can be
*uncrossed* into a laminar family (no two sets properly overlap), and a laminar family on `n`
vertices has at most `2n - 1` sets — a strong combinatorial restriction.

**Laminar families and their forests.** A family `L` of subsets of `V` is laminar if any two members
are disjoint or nested. Such a family is naturally a rooted forest: `C` is a child of `S` if `C` is
the largest member strictly inside `S`. Leaves, internal nodes, and "endpoints owned by a set" (an
edge endpoint whose smallest containing member is `S`) are the bookkeeping primitives for any
counting argument over `L`.

## Baselines

**Primal-dual / augmentation, with factor growing in the requirement.** The state-of-the-art constant
before a requirement-independent bound was the primal-dual augmentation approach of Goemans, Goldberg,
Plotkin, Shmoys, Tardos, and Williamson (1994), building on the Goemans-Williamson primal-dual method
for constrained forest problems. It raises the connectivity one unit at a time: at phase `k`, it has a
subgraph that is `(k-1)`-connected where required and runs a primal-dual `0/1` cut-covering step to
buy a layer raising deficient pairs to `k`. Each phase is a `2`-approximation against the *residual*
LP, but the phases stack, giving a total ratio of about `2 H(r_max) = 2 (1 + 1/2 + ... + 1/r_max)`.
The gap it leaves open: the factor grows logarithmically in the maximum requirement `r_max`, because
the analysis charges each connectivity layer separately rather than reasoning about the whole LP at
once.

**Doubling-based bounds for uniform connectivity.** For the special case of `k`-edge-connected
spanning subgraph (all `r(uv) = k`), one can take a fractional solution and exploit even-ness or
splitting-off to get small constants, and for `2`-edge-connected spanning subgraph there are
combinatorial `2`-approximations. These do not extend to arbitrary pairwise requirements: with
Steiner vertices (vertices that need not be connected to anything) and heterogeneous `r(uv)`,
doubling a tree or a single fractional structure neither yields feasibility cheaply nor gives a
constant independent of the requirements.

**Threshold LP rounding.** A direct idea is to solve the covering LP and round up every edge with
`x_e >= tau` for a fixed threshold `tau`. If such an edge is *guaranteed to exist* in the solution
and `tau = 1/2`, rounding it loses only a factor `2` on that edge. The open question this leaves —
and the entire crux of a requirement-independent guarantee — is whether some coordinate of a vertex
solution is always at least `1/2`, for the full weakly-supermodular cut LP. Naive instances show that
an *arbitrary* optimal point can be `1/3`-ish everywhere; the question is specifically about extreme
points.

## Evaluation settings

The natural yardstick is the integrality gap and the worst-case approximation ratio of the covering
LP `min sum c_e x_e` s.t. `x(delta(S)) >= f(S)`, `0 <= x_e <= 1`, measured against the integer
optimum `OPT` over all instances. Instances are undirected weighted graphs with arbitrary nonnegative
edge costs, arbitrary integer pairwise requirements `r(uv)` (including `0` for non-required pairs and
Steiner vertices with all-zero requirements), and the maximum requirement `r_max` ranging from `1`
(Steiner forest) to large. Feasibility is verified by checking, for every pair, that the chosen
subgraph admits `r(uv)` edge-disjoint paths — equivalently that every `u`-`v` min cut has at least
`r(uv)` edges — computable by max-flow. The LP itself, having exponentially many cut constraints, is
evaluated through a separation oracle: for a candidate `x`, the most violated constraint is found by
computing, for each demand pair, the minimum `u`-`v` cut under edge weights `x_e` (a max-flow), with
all pairs handled together via a Gomory-Hu tree (`n - 1` max-flow computations).

## Code framework

The pieces that already exist: a graph data structure with cuts and max-flow / min-cut, a Gomory-Hu
tree routine for all-pairs min cuts, and a generic LP solver that can return a basic (vertex) optimal
solution. The scaffold:

```python
import networkx as nx
import pulp

def requirement_on_cut(S, r):
    # f(S) = max_{u in S, v not in S} r(uv); the cut form of the demands
    best = 0
    for (u, v), req in r.items():
        if (u in S) ^ (v in S):
            best = max(best, req)
    return best

def delta(S, edges):
    pass  # TODO

def add_capacity(H, e, cap):
    pass  # TODO

def edge_cost(costs, e):
    pass  # TODO

def separation_oracle(x, free_edges, fixed_edges, V, r):
    # find a cut S with x(delta(S)) + |delta_fixed(S)| < f(S), or None if feasible
    # (Gomory-Hu / min-cut separation over capacities x_e plus fixed-edge capacity 1)
    pass  # TODO

def solve_covering_lp_to_vertex(edges, V, r, fixed_edges, costs):
    # cutting-plane loop using separation_oracle; return a basic optimal solution x
    pass  # TODO

def all_satisfied(V, fixed_edges, r):
    pass  # TODO

def cover_cut_requirements(V, edges, costs, r):
    F = set()
    while True:
        if all_satisfied(V, F, r):
            return F
        x = solve_covering_lp_to_vertex(edges, V, r, F, costs)
        # TODO: choose coordinates to fix and continue on the cut residual
        pass
    return F
```
