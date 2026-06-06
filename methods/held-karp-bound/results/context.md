# Context: lower bounds for branch-and-bound on the symmetric TSP (circa 1970)

## Research question

We want to solve the symmetric travelling-salesman problem to proven optimality: given `n`
cities and a symmetric cost matrix `c(i,j)`, find a minimum-cost Hamiltonian cycle (a "tour")
that visits every city exactly once and returns to the start. The only exact tool that scales
past tiny instances is branch-and-bound: recursively split the set of tours (by forcing an
edge in or out), and at each node of the search tree compute a **lower bound** on the cost of
the best tour consistent with the decisions made so far. If that lower bound already exceeds
the cost of some tour we have in hand, the whole subtree is pruned without ever being
explored.

The entire efficiency of such a search rests on one quantity: how tight the lower bound is. A
bound that sits far below the true optimum prunes almost nothing, the tree explodes, and the
method is useless beyond a handful of cities. A bound that hugs the optimum prunes
aggressively and keeps the tree small. So the precise problem is: **find a lower bound on the
optimal tour cost that is (a) cheap enough to recompute at every node of a large search tree,
and (b) tight enough — close to the true optimum — that branch-and-bound actually terminates
on instances of interesting size.** The cheap, obvious bounds available at the time are far too
loose; closing that gap is the goal.

## Background

A tour is a very constrained object. Three facts about it are load-bearing here.

First, a tour is a connected spanning subgraph on all `n` nodes with exactly `n` edges in
which **every node has degree exactly 2**. Connectivity plus "n nodes, n edges" forces exactly
one cycle, and degree-2-everywhere forces that cycle to be a single Hamiltonian cycle rather
than a union of smaller cycles.

Second, relaxations of "tour" that drop some of these constraints are computable in polynomial
time, whereas the tour itself is not. The minimum spanning tree (MST) — the cheapest connected
spanning subgraph — has `n-1` edges and is computed in near-linear time by classical greedy
algorithms (Prim's method grows a tree one cheapest-crossing-edge at a time; Kruskal's adds
globally cheapest edges that don't form a cycle). An MST drops the cycle and the degree
constraint entirely. Its cost is a lower bound on the tour cost (a tour minus one edge is a
spanning tree whose cost is no larger than the tour in the usual nonnegative distance setting,
so OPT-tour >= MST), but a weak one: it has no cycle and its degrees are wildly uneven, so it
looks nothing like a tour and its cost sits well below the optimum.

Third, the degree-2 requirement is an **equality constraint per node**, and equality
constraints are exactly the kind of thing Lagrangian relaxation was built to handle: move the
constraint into the objective with a multiplier (a price) per node, solve the now-unconstrained
(easier) problem, and adjust the prices to push the relaxed solution back toward feasibility.
The general theory says that for any fixed prices the relaxed optimum is a lower bound on the
constrained optimum, and that the best (largest) such bound is obtained by maximizing over the
prices — the **Lagrangian dual**. Maximizing a Lagrangian dual is maximizing a function that is
a pointwise minimum of linear pieces, hence concave but not smooth; the tool for maximizing a
nonsmooth concave function using only the easy-to-compute relaxed solutions is the
**subgradient method** of Shor and others: step in the direction of a subgradient with a
diminishing step size.

The pain point that motivates everything below: MST and the trivially-cheap per-node bounds
(half the sum of each city's two cheapest edges) are all far too loose to drive
branch-and-bound, yet the tour itself is intractable. We need a relaxation that lives *between*
"spanning tree" and "tour" — close enough to a tour that its cost is a sharp bound, but still
polynomial to compute.

## Baselines

**Minimum spanning tree bound.** Take the cheapest connected spanning subgraph; in the usual
nonnegative distance setting its cost lower-bounds the tour cost because deleting any one edge
of a tour leaves a spanning tree no more expensive than the tour. Computable in near-linear
time (Prim/Kruskal). Gap/limitation: it has `n-1` edges, no cycle, and degrees that range from
1 (leaves) to large hubs, so it is structurally far from a tour and the bound is loose.

**Sum-of-two-cheapest-edges bound.** Every node in a tour has exactly two incident edges, so
the tour cost is at least half the sum over nodes of that node's two cheapest incident edges.
Trivial to compute. Limitation: it ignores connectivity and global structure entirely and is
typically even looser than the MST bound; it cannot reflect any interaction between edges.

**Assignment-problem / 2-matching relaxation.** Drop connectivity, keep "degree 2": ask for the
cheapest set of edges giving every node degree 2 (a 2-factor), solvable as a matching/assignment
problem. Limitation: the solution decomposes into several disjoint subtours, so the bound is
loose and the relaxed solution can be far from a single Hamiltonian cycle; there is no built-in
mechanism to discourage the subtours.

**Linear-programming relaxation with subtour-elimination.** Write tour membership as a 0/1
program: one variable per edge, degree-2 equality at every node, and one inequality per vertex
subset forbidding subtours; relax integrality and solve the LP. This gives a strong bound, but
there are exponentially many subtour constraints, so it must be solved with cutting planes or a
separation oracle — heavy machinery to invoke at every node of a branch-and-bound tree, and not
obviously cheap.

## Evaluation settings

The natural yardstick is a set of symmetric TSP instances with a full symmetric cost matrix:
random Euclidean instances (cities drawn uniformly in the unit square, costs = pairwise
distances) at sizes from a handful up to a few dozen cities where the true optimum can still be
found, plus structured geographic instances (inter-city road or great-circle distances). The
quantities of interest are the bound's gap to the known optimum (how much of OPT it recovers),
the size of the branch-and-bound tree the bound induces, and the time to compute the bound
itself — including how that cost scales as `n` grows and how it behaves when recomputed
thousands of times inside a search. Small instances where the optimum is computable by
exhaustive permutation serve to confirm that a lower-bound routine never exceeds the true
optimum.

## Code framework

The primitives that already exist: a symmetric cost matrix, a minimum-spanning-tree routine,
a way to keep only cheap candidate edges when dense graphs are too slow, an upper-bound tour
heuristic, and basic vector arithmetic. The open slot is the lower-bound routine itself: it
must choose an easy relaxation, evaluate it under whatever auxiliary costs it introduces, and
return a certified value for the original complete graph.

```python
import numpy as np

def minimum_spanning_tree_edges(weight_matrix, allowed_edges=None):
    """Existing primitive: minimum spanning tree on a dense symmetric matrix,
    optionally restricted to an already-connected candidate-edge graph."""
    ...  # standard library routine

def cheap_candidate_edges(cost, width):
    """Existing primitive: keep a small symmetric set of low-cost candidate
    edges, then add enough spanning-tree edges to keep the graph connected."""
    ...  # nearest-neighbour filtering plus a connectivity repair

def tour_upper_bound(cost):
    """Existing primitive: return the cost of any complete tour, used only as
    a target value by iterative lower-bound tightening."""
    ...  # heuristic tour construction

def tsp_lower_bound(cost, parameters=None):
    """Return a certified lower bound on the optimal symmetric-TSP tour cost."""
    pass  # TODO: fill in the relaxation, auxiliary costs, updates, and final certification
```
