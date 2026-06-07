## Research question

We are given an undirected graph `G = (V, E)` with a nonnegative cost `c_e` on every edge, and for
each unordered pair of vertices `u, v` an integer connectivity requirement `r(uv) >= 0`. We must
choose a minimum-cost subgraph `H` of `G` such that, for every pair, `H` contains at least `r(uv)`
edge-disjoint paths between `u` and `v`. This is the survivable network design problem (also called
the generalized Steiner network problem): build the cheapest network that keeps every required pair
connected even after up to `r(uv) - 1` edge failures.

The problem is NP-hard and APX-hard even in special cases, so an exact polynomial-time algorithm is
out of reach. The target is an approximation guarantee that does **not** degrade as the requirements
grow: for every feasible instance and every requirement vector, a polynomial-time algorithm returning a
feasible subgraph whose cost is at most a fixed constant factor times the optimum `OPT`. The whole
difficulty is the proof of the factor; when the input graph can meet the requirements, taking all
edges proves feasibility, but gives no useful cost guarantee.

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
structural property that supports uncrossing. Call `f : 2^V -> Z` *weakly supermodular* if
`f(empty) = f(V) = 0` and for all `A, B subset V` at least one of
`f(A) + f(B) <= f(A union B) + f(A intersect B)` or
`f(A) + f(B) <= f(A \ B) + f(B \ A)`
holds. The cut-maximum `f(S) = max_{u in S, v not in S} r(uv)` is weakly supermodular. Partition
`V` into `P=A intersect B`, `Q=A \ B`, `R=B \ A`, and `W=V \ (A union B)`. A pair witnessing `f(A)`
lies in one of `P-R`, `P-W`, `Q-R`, or `Q-W`; these pairs are cut respectively by
`(A intersect B, B \ A)`, `(A intersect B, A union B)`, `(A \ B, B \ A)`, or
`(A \ B, A union B)`. A pair witnessing `f(B)` lies in one of `P-Q`, `P-W`, `R-Q`, or `R-W`; these
are cut respectively by `(A intersect B, A \ B)`, `(A intersect B, A union B)`,
`(B \ A, A \ B)`, or `(B \ A, A union B)`. Checking the four possible witness types for `f(A)`
against the four for `f(B)`, every pair of witnesses can be assigned to distinct members of
`{A intersect B, A union B}` or to distinct members of `{A \ B, B \ A}`. The only two exceptional
type-pairs are `P-W` against `R-Q` and `Q-R` against `P-W`; in both, the witness for `f(A)` also
crosses `B`, and the witness for `f(B)` also crosses `A`, so maximality gives
`f(A) <= f(B) <= f(A)`. The two values are equal, and either the union/intersection pair or the two
difference sets carries that common value twice. This proves one of the two
weak-supermodular inequalities. This class is broader than the SNDP cut function, and the same
uncrossing machinery applies to any weakly-supermodular `f`.

**The cut function is symmetric and submodular.** For any fixed edge set `F`, the map
`S -> |delta_F(S)|` is symmetric (`|delta_F(S)| = |delta_F(V \ S)|`) and submodular:
`|delta_F(S)| + |delta_F(T)| >= |delta_F(S union T)| + |delta_F(S intersect T)|`.
A symmetric submodular function is also *posimodular*:
`|delta_F(S)| + |delta_F(T)| >= |delta_F(S \ T)| + |delta_F(T \ S)|`.
The fractional version `x(delta(S)) = sum_{e in delta(S)} x_e` inherits both inequalities. These two
inequalities, together with the characteristic-vector identities
`chi(delta(S)) + chi(delta(T)) = chi(delta(S union T)) + chi(delta(S intersect T)) + 2 chi(E(S\T, T\S))`
and
`chi(delta(S)) + chi(delta(T)) = chi(delta(S\T)) + chi(delta(T\S)) + 2 chi(E(S intersect T, V \ (S union T)))`,
are the raw material for reasoning about which cut constraints can be simultaneously tight at a vertex
of the LP polytope.

**Linear programming relaxations and extreme points.** Relaxing the integral choice `x_e in {0,1}` to
`x_e in [0,1]` gives a covering LP. Its optimum lower-bounds `OPT`. A *basic feasible solution*
(extreme point / vertex) of a polytope in `R^E` is the unique solution of some `|E|` linearly
independent tight constraints, including active variable bounds if any. For the structural rounding
argument, zero-valued variables are deleted from the support and a variable already at `1` is already
a rounding edge, so the hard case is pinned by tight cut rows alone. Extreme points are the objects
whose support structure can be controlled — a generic optimal point need not have any nice coordinate,
but a vertex does, and LP solvers can be made to return a vertex. The governing prior fact is that for
covering an exponentially-large family of cut constraints, the tight constraints at a vertex can be
*uncrossed* into a laminar family (no two sets properly overlap), and a laminar family on `n`
vertices has at most `2n - 1` sets — a strong combinatorial restriction.

**Laminar families and their forests.** A family `L` of subsets of `V` is laminar if any two members
are disjoint or nested. Such a family is naturally a rooted forest: `C` is a child of `S` if `C` is
the largest member strictly inside `S`. Leaves, internal nodes, and "endpoints owned by a set" (an
edge endpoint whose smallest containing member is `S`) are the bookkeeping primitives for any
counting argument over `L`.

## Baselines

**Primal-dual / augmentation, with factor growing in the requirement.** The best available general
approach toward this target was the primal-dual augmentation method of Goemans, Goldberg,
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
solution is always at least `1/2`, for every nonzero weakly-supermodular residual cut LP. Naive
instances show that an *arbitrary* optimal point can be `1/3`-ish everywhere; the question is
specifically about extreme points.

## Evaluation settings

The natural yardstick is the integrality gap and the worst-case approximation ratio of the covering
LP `min sum c_e x_e` s.t. `x(delta(S)) >= f(S)`, `0 <= x_e <= 1`, measured against the integer
optimum `OPT` over feasible instances. Instances are undirected weighted graphs with arbitrary nonnegative
edge costs, arbitrary integer pairwise requirements `r(uv)` (including `0` for non-required pairs and
Steiner vertices with all-zero requirements), and the maximum requirement `r_max` ranging from `1`
(Steiner forest) to large. Feasibility is verified by checking, for every pair, that the chosen
subgraph admits `r(uv)` edge-disjoint paths — equivalently that every `u`-`v` min cut has at least
`r(uv)` edges — computable by max-flow. The LP itself, having exponentially many cut constraints, is
evaluated through a separation oracle: for a candidate `x`, the most violated constraint is found by
computing, for each demand pair, the minimum `u`-`v` cut under edge weights `x_e` (a max-flow), with
all pairs handled together via a Gomory-Hu tree (`n - 1` max-flow computations).

## Code framework

The available pieces are a graph data structure with cuts and max-flow / min-cut, a Gomory-Hu tree
routine for all-pairs min cuts, and a generic LP solver with a simplex/basic-solution interface for
the currently generated cut relaxation. The residual LP has variables only on unfixed edges and
constraints `x(delta_free(S)) >= f(S) - |delta_fixed(S)|`. The scaffold:

```python
from itertools import combinations

import networkx as nx
import pulp

def requirement_on_cut(S, r):
    # f(S) = max r(uv) over demand pairs split by S
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

def separation_oracle(x, free_edges, fixed_edges, V, r, tol=1e-7):
    # find a cut S with x(delta_free(S)) + |delta_fixed(S)| < f(S), or None
    pass  # TODO

def solve_covering_lp_to_vertex(edges, V, r, fixed_edges, costs):
    # add violated residual cut constraints until the LP solution satisfies every cut
    pass  # TODO

def all_satisfied(V, fixed_edges, r):
    pass  # TODO

def cover_cut_requirements(V, edges, costs, r):
    F = set()
    while not all_satisfied(V, F, r):
        x = solve_covering_lp_to_vertex(edges, V, r, F, costs)
        # TODO: choose which fractional coordinates become fixed edges
        pass
    return F

def solution_cost(F, costs):
    pass  # TODO

def is_feasible(V, F, r):
    pass  # TODO

def brute_force_optimum(V, edges, costs, r):
    pass  # TODO

if __name__ == "__main__":
    # TODO: instantiate a tiny graph, run the solver, and compare with brute force
    pass
```
