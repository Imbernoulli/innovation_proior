# Context: approximating the maximum cut of a weighted graph

## Research question

Given an undirected graph G=(V,E) with nonnegative edge weights w_ij, partition the vertices
into two sets (S, V\S) so as to maximize the total weight of edges crossing the partition. This
is MAX-CUT. Computing the exact optimum is NP-hard — it is one of Karp's original twenty-one
NP-complete problems, and the weighted optimization version is NP-hard as well. So the realistic
goal is not exactness but a *guarantee*: a polynomial-time algorithm that on every instance
returns a cut of weight at least c times the optimum, for the largest constant c we can prove.
The pain point is sharp. A trivial randomized assignment — flip an independent fair coin for each
vertex — already cuts each edge with probability 1/2, so it achieves expected weight exactly
(1/2) of the total edge weight, hence at least (1/2)·OPT. For decades the question was whether
anything could provably beat this factor of 1/2. Local-search and greedy heuristics improve in
practice but resist a worst-case guarantee better than 1/2. What is needed is a relaxation whose
optimum upper-bounds OPT tightly *and* from which an integral cut can be recovered without losing
much, with both facts provable.

## Background

The standard encoding uses a sign variable per vertex. Let y_i ∈ {-1,+1}, with y_i=+1 meaning
i ∈ S. Edge (i,j) is cut exactly when y_i ≠ y_j, i.e. when y_i y_j = -1, and the quantity
(1 - y_i y_j)/2 equals 1 on a cut edge and 0 otherwise. The weight of the cut is therefore

    (1/2) · Σ_{(i,j)∈E} w_ij (1 - y_i y_j),

and MAX-CUT is the maximization of this expression over y ∈ {-1,+1}^n. Written in matrix form the
objective is a quadratic form y^T M y plus a constant, with M indefinite; this is a nonconvex
integer quadratic program, which is where the hardness lives.

The prevailing tool for designing approximation algorithms in the early 1990s was linear
programming relaxation followed by rounding: relax the integrality, solve an LP, round the
fractional solution back, and bound the loss. For MAX-CUT this hits a structural wall. A relaxation
that describes a cut only by independent scalar/edge variables loses the parity constraints on odd
cycles. Take the triangle K_3 with unit weights: every genuine cut separates exactly two of the
three edges, so OPT = 2, while the naive edge-variable LP can set all three "is this edge cut"
variables to 1 and reach value 3. Adding the triangle inequality fixes this particular triangle,
but it exposes the general disease rather than curing it: linear descriptions need many odd-cycle
and cut-polytope inequalities to enforce global cut consistency. On complete graphs, the same
independent edge relaxation sets every edge variable to 1, while the best integral cut has only
floor(n^2/4) crossing edges out of n(n-1)/2, so the integral/LP ratio tends to 1/2. Strong
lift-and-project LP hierarchies (Lovász–Schrijver) also remain stuck near 1/2 + ε after a linear
number of rounds. So the scalar LP route does not give a provable way past the coin-flip factor.

A separate body of background makes the eventual escape possible: semidefinite programming.
A matrix Y is positive semidefinite iff it is the Gram matrix of a set of vectors, Y_ij = ⟨v_i,v_j⟩.
Optimizing a linear function of the entries of Y subject to Y ⪰ 0 and linear constraints on Y is a
convex program — a semidefinite program — solvable to any additive accuracy ε in time polynomial
in the size and log(1/ε) by the ellipsoid method or interior-point methods. Equivalently one
optimizes directly over vectors {v_i} with linear constraints on their inner products. This
"vector program" viewpoint, and the fact that any PSD Y factors as Y = Q Q^T or Y = Q^T Q so the
vectors can be recovered as rows or columns of a Cholesky factorization or matrix square root, are
the load-bearing facts.

The geometry of the unit sphere supplies the last ingredient. For two unit vectors at angle θ, a
random direction drawn from a spherically symmetric distribution induces a uniformly oriented
hyperplane through the origin, and the chance that the hyperplane separates the two vectors is
proportional to their angular separation. This linear-in-angle behavior — contrasted with the
inner-product, which moves like cos θ — is the seed of the whole analysis.

## Baselines

**Random ±1 assignment (the 1/2 baseline).** Set each y_i to +1 or -1 by an independent fair
coin. Edge (i,j) is cut iff the two coins disagree, probability 1/2, so E[cut weight] = (1/2)·Σ w_ij
≥ (1/2)·OPT. Deterministic via the method of conditional expectations. Core idea: every edge is
cut half the time. Gap it leaves open: it ignores the graph structure entirely; it cannot tell a
graph whose OPT is the whole edge weight from one whose OPT is barely above half, so it cannot
beat the factor 1/2.

**LP relaxation + rounding.** Introduce edge variables z_ij ∈ [0,1] for "edge (i,j) is cut," solve
a linear relaxation, and round. Core idea: replace the integer program by a polytope that contains
all cuts and optimize the linear objective Σ w_ij z_ij. Specific limitation: simple scalar
relaxations miss odd-cycle consistency. The K_3 LP value 3 versus OPT = 2 is the smallest warning
sign; complete graphs drive the independent edge-variable ratio down to 1/2 asymptotically, and
linear lift-and-project hierarchies remain near 1/2 for a linear number of rounds. LP-based
rounding by itself therefore does not give a provable route beyond the coin-flip guarantee.

**Local search / greedy heuristics.** Start from any cut and move a vertex across whenever it
increases the cut; or add vertices greedily. Core idea: monotone improvement to a local optimum.
At a local optimum each vertex has at least half its incident weight cut, giving ≥ (1/2)·(total
weight) again. Specific limitation: strong in practice but its provable worst-case guarantee is
still 1/2; no better constant is known from local moves alone.

## Evaluation settings

The yardstick is the worst-case approximation ratio: the infimum over all weighted graphs of
ALG/OPT, where ALG is the (expected, for randomized algorithms) weight of the returned cut and OPT
is the maximum cut weight. Instances range over arbitrary undirected graphs with nonnegative edge
weights; structured families used to probe tightness include complete graphs, odd cycles, random
graphs, and graphs derived from MAX-2SAT and other
constraint-satisfaction reductions. Hardness landmarks that frame what a guarantee can hope to be:
exact MAX-CUT is NP-hard, and approximating it within a factor better than 16/17 ≈ 0.941 is
NP-hard.

## Code framework

The pre-existing pieces are a graph representation, a convex-optimization solver exposed through a
modeling layer (a CVXPY-style interface over an interior-point SDP backend), and the standard
numerical-linear-algebra tools (Gaussian sampling, matrix square root / Cholesky, sign). What is
missing is the relaxation itself and the procedure that turns its solution into an integral cut.
The scaffold uses the common unweighted edge-list form; weights would appear as coefficients on the
same edge terms.

```python
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm   # PSD factorization tool that already exists

def relax_and_solve(n, edges):
    """Build and solve a convex relaxation of MAX-CUT over the n vertices.
    The decision object and the constraints that make it a valid, tractable
    relaxation are exactly what we must discover."""
    # TODO: choose the relaxed decision variable and its feasible set
    # TODO: express the cut objective as a linear function of that variable
    # TODO: solve the convex program and return its solution object
    pass

def round_to_cut(solution, n):
    """Turn the relaxed solution into y in {-1,+1}^n with provably small loss.
    The rounding rule is the second thing we must discover."""
    # TODO: map the relaxed solution to an integral assignment
    pass

def cut(x, edges):
    """Edges crossing the partition described by x in {-1,+1}^n."""
    return [(i, j) for (i, j) in edges if np.sign(x[i] * x[j]) < 0]
```
