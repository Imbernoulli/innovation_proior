## Research question

We are given an undirected graph `G = (V, E)` with a nonnegative cost `c_e` on every edge, and for
each unordered pair of vertices `u, v` an integer connectivity requirement `r(uv) >= 0`. We must
choose a minimum-cost subgraph `H` of `G` such that, for every pair, `H` contains at least `r(uv)`
edge-disjoint paths between `u` and `v`. This is the survivable network design problem (also called
the generalized Steiner network problem): build the cheapest network that keeps every required pair
connected even after up to `r(uv) - 1` edge failures.

The problem is NP-hard and APX-hard even in special cases, so an exact polynomial-time algorithm is
out of reach. The question is what approximation ratio a polynomial-time algorithm can achieve, and
how it depends (if at all) on the requirement values. The natural lower bound on the integer optimum
is a linear program with one constraint per cut of the graph — exponentially many constraints — and
rounding a fractional solution of that LP into an integral feasible subgraph is the core
algorithmic step.

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
independent tight constraints, including active variable bounds if any. Extreme points are the objects
whose support structure can be controlled — a generic optimal point need not have any nice coordinate,
but a vertex does, and LP solvers can be made to return a vertex. With exponentially many cut
constraints, the only handle on a vertex is the family of cut rows that are tight there, and what
structure that tight family can be forced to have is exactly what an integrality argument must pin
down.

**Laminar families and their forests.** A family `L` of subsets of `V` is laminar if any two members
are disjoint or nested. Such a family is naturally a rooted forest: `C` is a child of `S` if `C` is
the largest member strictly inside `S`, and a laminar family on `n` vertices has at most `2n - 1`
members.

## Baselines

**Primal-dual / augmentation, with factor growing in the requirement.** The best available general
approach toward this target was the primal-dual augmentation method of Goemans, Goldberg,
Plotkin, Shmoys, Tardos, and Williamson (1994), building on the Goemans-Williamson primal-dual method
for constrained forest problems. It raises the connectivity one unit at a time: at phase `k`, it has a
subgraph that is `(k-1)`-connected where required and runs a primal-dual `0/1` cut-covering step to
buy a layer raising deficient pairs to `k`. Each phase is a `2`-approximation against the *residual*
LP, giving a total ratio of about `2 H(r_max) = 2 (1 + 1/2 + ... + 1/r_max)`.

**Doubling-based bounds for uniform connectivity.** For the special case of `k`-edge-connected
spanning subgraph (all `r(uv) = k`), one can take a fractional solution and exploit even-ness or
splitting-off to get small constants, and for `2`-edge-connected spanning subgraph there are
combinatorial `2`-approximations.

**Threshold LP rounding.** A direct idea is to solve the covering LP and round up every edge whose
fractional value clears a fixed threshold `tau`, since an edge with `x_e >= tau` is rounded at a cost
blow-up of only `1/tau` on that edge. Whether this buys anything depends entirely on how large a
coordinate the relaxation can be forced to expose, and on whether rounding a few edges and re-solving
actually drives the requirements down.

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
the currently generated cut relaxation. The scaffold:

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

def solve_covering_lp(edges, V, r, fixed_edges, costs):
    # add violated cut constraints until the LP solution satisfies every cut
    pass  # TODO

def all_satisfied(V, fixed_edges, r):
    pass  # TODO

def cover_cut_requirements(V, edges, costs, r):
    F = set()
    # TODO
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
