# Context

## Research question

Given a complete undirected graph on `n` cities with a symmetric cost matrix `(c_ij)`, the
symmetric traveling-salesman problem asks for a minimum-weight tour — a cycle visiting every
vertex exactly once. Exact solution proceeds by branch-and-bound: recursively split the set of
tours (by forcing some edges in and others out) and discard any branch whose cheapest possible
tour already exceeds the best tour found so far. The whole method lives or dies on the **lower
bound** computed at each node of the search tree. A weak bound prunes almost nothing and the
tree explodes; a bound that is both **tight** (close to the true optimum cost of the subproblem)
and **cheap** (computable in roughly the time of a spanning-tree calculation, since it is
recomputed at thousands of nodes) is what would make exact solution of 40-, 50-, 60-city
instances feasible. The precise problem is therefore: find a function `LB ≤ C*` (the optimum
tour cost) that is as large as possible while remaining far cheaper to evaluate than the tour
problem itself, and that adapts — can be driven upward — as the search wants a sharper estimate.

## Background

The cost of an optimum tour `C*` is what branch-and-bound must under-estimate. Several relaxations
of "tour" were known to give lower bounds, each leaving a gap:

A tour is a connected spanning subgraph in which every vertex has degree exactly 2. Relaxing the
degree-2 requirement to *connectivity* alone gives a spanning structure; relaxing connectivity
to a *perfect 2-matching* gives the assignment relaxation. Each removes a constraint and so each
optimal relaxed object costs no more than the optimal tour.

Three pieces of mathematics sit underneath everything that follows.

**Minimum spanning trees are easy.** Kruskal (1956) and Prim/Dijkstra (1957–1959) compute a
minimum spanning tree of a weighted graph in low-order polynomial time by a greedy edge or
vertex scan. A spanning tree on `k` vertices has `k−1` edges; degrees are unconstrained. Gale
(1968) and Kruskal (1956) connect the greedy/matroid view of MST to exactly such selection
problems, and the greedy MST routine produces, as a by-product, the sensitivity of the tree
cost to forcing or forbidding individual edges.

**Lagrangian relaxation of integer programs.** When a hard combinatorial problem
`min cᵀx s.t. Ax = b, x ∈ X` has a set of "complicating" equality constraints `Ax = b` whose
removal leaves an *easy* problem over `X`, one can dualize them: for a multiplier vector `π`,
`L(π) = min_{x∈X} [cᵀx + πᵀ(b − Ax)]` is a lower bound on the optimum for every `π`, because at
the true optimum the bracketed term added is zero and the minimization over the larger set `X`
can only lower the value. The best such bound is `max_π L(π)`. As a minimum of finitely many
affine functions of `π`, `L(π)` is concave and piecewise linear — hence non-differentiable at
the breakpoints where the inner minimizer switches. The systematic exploitation of this
structure for integer programming was being worked out at the time (Geoffrion); the degree
constraints of the TSP are a textbook instance of "complicating" equalities over an otherwise
tractable set.

**The relaxation method for linear inequalities.** Agmon (1954) and Motzkin & Schoenberg (1954),
following an idea of Motzkin, studied solving a consistent system of linear inequalities
`a_iᵀx ≥ b_i` by iterated projection: pick a violated inequality, and step from the current point
toward (and across) its bounding hyperplane along the normal. Agmon's basic lemma (2.1): if `x`
is on the wrong side of an oriented hyperplane `π` and the solution `y` on the right side, and
`x_r` is the orthogonal projection of `x` onto `π`, then for `0 < λ < 2`,
`|x + λ(x_r − x) − y| < |x − y|` — the relaxed projection strictly decreases the Euclidean
distance to every solution point, and Agmon proves a linear convergence rate. The relaxation
parameter `λ ∈ (0, 2)` and the "decrease distance to the solution set without necessarily
decreasing the residual" behavior are the load-bearing facts.

**Empirical state of TSP solving.** The diagnostic fact motivating the work is that the bounds
in use produced search trees too large to exhaust for instances past roughly 20–30 cities; the
assignment relaxation and the bare minimum-spanning-structure bound left visible gaps to `C*`,
and that gap, multiplied across the branch-and-bound tree, was the bottleneck. The structural
symptom is visible in the relaxed spanning objects themselves: they are cheap because they ignore
the tour's degree-2 condition, and the resulting degree surplus and deficit are exactly where the
bound loses contact with a Hamiltonian cycle.

## Baselines

**Bare spanning relaxation.** Drop both degree-2 and exact-connectivity-into-a-cycle; keep a
spanning connected structure. Its minimum weight is computed by a greedy MST routine and lower-
bounds `C*` because a tour contains a spanning tree. Gap: the structure is free to pile many
edges onto a few hub vertices, so the bound is loose, and there is no knob to tighten it.

**Assignment relaxation.** Relax a tour to a minimum-cost perfect 2-matching / assignment.
Solvable by the Hungarian / assignment algorithm; its dual gives values `u_i, v_j` with
`u_i + v_j ≤ c_ij`. It bounds `C*` from below but typically returns subtours (2-cycles and short
cycles) and so leaves a gap. Its dual variables also show that pricing constraints can be useful
when the primal relaxed object is easy but structurally wrong.

**Column-generation linear program / steepest ascent over node prices.** Treat the best
achievable per-vertex-price bound as an optimization: maximize, over price vectors, the
spanning-relaxation cost computed under price-perturbed edge weights. One can attack this as a
large linear program with one constraint per candidate structure, generating columns as needed,
or by a steepest-ascent procedure that increases the objective at each step. Gap: with the
number of candidate structures astronomically large, the LP/simplex route and the
function-increasing ascent route were both found to be slow, and the ascent's iteration count
grew rapidly with `n`. This leaves the need for a lighter ascent rule.

## Evaluation settings

The natural yardsticks are symmetric TSP instances from the literature and constructed test
families: published instances (e.g. the Dantzig 42-city problem, Croes 20-city, Karg–Thompson
57-city), `random(M)` instances with `c_ij` drawn i.i.d. from a discrete uniform distribution on
`{0,…,M}`, `random Euclidean(M)` instances with `n` points dropped uniformly in an `M × M`
square and `c_ij` the Euclidean distance, and `p × q` knight's-tour problems (cities are board
squares, cost 0 between knight-move-adjacent squares and ∞ otherwise) used to test for the
existence of a Hamiltonian circuit. Instance sizes of interest range from 20 up to 64 cities.
The metrics are: the value of the lower bound relative to the optimum tour cost `C*` (how tight),
the number of branch-and-bound nodes generated (how much pruning), and wall-clock time on the
machines of the day (an IBM 360/91). Each run is parameterized by a step-size control, a control
on when to stop ascending and branch, and an upper bound on `C*` used to discard subproblems.

## Code framework

Pre-existing primitives: a dense symmetric cost matrix, a greedy minimum-spanning-tree routine,
and a branch-and-bound shell that maintains a list of subproblems, each pinning some edges in and
some out, with associated lower bounds. The open slot is the subproblem bound routine.

```python
import numpy as np

def min_spanning_tree(weight):
    """Greedy MST on a dense symmetric weight matrix. Returns (edges, degrees, cost).
    Prim/Kruskal primitive."""
    ...

def lower_bound(cost, forced_edges, forbidden_edges, upper_bound=None):
    """Lower bound on tours satisfying the edge decisions in a branch-and-bound node."""
    # TODO: replace the plain spanning-tree bound with a stronger MST-cheap relaxation.
    pass

def branch_and_bound(cost, upper_bound):
    """Maintain a list of subproblems (X_in, X_out, bound); repeatedly take the least
    bound, tighten it via `lower_bound`, discard if it exceeds `upper_bound`, else
    split on an edge."""
    ...
```
