# LP-rounding 2-approximation for weighted vertex cover

## Problem
Given G = (V, E) with vertex weights w : V → Q₊, find a minimum-weight vertex cover — a set C ⊆ V meeting
every edge. The problem is NP-complete, so the target is a polynomial-time algorithm with a proven worst-case
approximation ratio.

## Key idea
Model vertex cover as a 0/1 integer program, **relax** the integrality constraint to obtain a linear program
(solvable in polynomial time), solve it, and **round**. The LP optimum is a lower bound that certifies the
ratio; the edge constraints force half-integral rounding to work, giving a factor of 2.

## The programs
Integer program (exact; optimum = OPT):

  min Σ_{v∈V} w_v x_v   s.t.  x_u + x_v ≥ 1  ∀(u,v)∈E,  x_v ∈ {0,1}.

LP relaxation:

  min Σ_{v∈V} w_v x_v   s.t.  x_u + x_v ≥ 1  ∀(u,v)∈E,  0 ≤ x_v ≤ 1.

Write OPT_f for the LP optimum. The upper bounds do not change the optimum: replacing any feasible x_v > 1
by 1 preserves all edge constraints and never increases a nonnegative objective.

## The algorithm
1. Solve the LP relaxation; let x* be an optimal solution. An extreme optimal solution can be chosen.
2. Output C = { v ∈ V : x*_v ≥ 1/2 } (round every coordinate ≥ 1/2 up to 1, the rest down to 0).

## Correctness and the 2-approximation

**Lemma (relaxation lower bound).** OPT_f ≤ OPT.
*Proof.* Every integral cover x ∈ {0,1}^V satisfies 0 ≤ x_v ≤ 1 and the same edge constraints, so it is feasible
for the LP. Thus the LP minimizes the same objective over a superset of the integral feasible points, and its
optimum cannot exceed the integral optimum: OPT_f ≤ OPT. ∎

**Lemma (feasibility of C).** For every edge (u,v), x*_u + x*_v ≥ 1, so max(x*_u, x*_v) ≥ 1/2; hence at least
one endpoint lies in C. Every edge is covered, so C is a vertex cover. ∎

**Theorem (factor 2).** w(C) ≤ 2·OPT.
*Proof.* For each v: if x*_v ≥ 1/2 then v ∈ C and its contribution w_v satisfies w_v ≤ 2·w_v x*_v (since
1 ≤ 2x*_v); if x*_v < 1/2 then v ∉ C and contributes 0 ≤ 2·w_v x*_v. Summing,
  w(C) = Σ_{v∈C} w_v ≤ Σ_{v∈V} 2 w_v x*_v = 2·OPT_f ≤ 2·OPT. ∎

The threshold 1/2 is 1/p for p = 2, the number of vertices on an edge. For the general set-cover instance,
where each element lies in at most f sets, the same argument with threshold 1/f gives an f-approximation
(every covering inequality with f terms summing to ≥ 1 has a term ≥ 1/f); vertex cover is the case f = 2.

## Half-integrality of the vertex-cover polytope

**Theorem (Nemhauser–Trotter).** Every extreme-point (basic feasible) solution of the boxed LP relaxation has
x_v ∈ {0, 1/2, 1}.

*Proof 1 (perturbation).* Let x be feasible but not half-integral. Put
V₊ = {v : 1/2 < x_v < 1}, V₋ = {v : 0 < x_v < 1/2} (not both empty). For small ε > 0 define
  y = x + ε on V₊, x − ε on V₋, x elsewhere;  z = x − ε on V₊, x + ε on V₋, x elsewhere.
Both stay in [0,1] for small ε. For a slack edge (x_u + x_v > 1), small ε preserves the constraint. For a
tight edge (x_u + x_v = 1), both moved endpoints must be one in V₊ and one in V₋, so the ±ε cancel. Exactly
one moved endpoint and one half-integral endpoint cannot be tight: a strict value below 1/2 cannot be completed
to exactly 1 by 0, 1/2 or 1, and neither can a strict value above 1/2. If neither endpoint is moved, the tight
pairs are {1/2,1/2}, {0,1} and {1,0}, and the constraint is unchanged. So y, z are feasible, distinct from x,
and x = (y+z)/2, so x is not extreme. Contrapositive: extreme ⇒ half-integral. ∎

*Proof 2 (determinant).* In a basis matrix, edge rows have two nonzeros and bound rows have one, with entries
in {0, ±1}. Decompose a square basis matrix into nonseparable blocks. For one block, induct on size: a row or
column with 0 nonzeros gives determinant 0; a row or column with 1 nonzero reduces by expansion; if every row
and column has exactly 2 nonzeros, the block is a single cycle. After ordering rows cyclically, only two
permutation products survive in the determinant expansion, each ±1, so the determinant is 0 or ±2. Thus every
nonsingular block has determinant ±1 or ±2. Cramer's rule applied blockwise gives denominators at most 2; with
0 ≤ x_v ≤ 1, every basic coordinate is 0, 1/2 or 1. ∎

**Consequence.** Simplex returns an extreme point, hence a {0, 1/2, 1} solution, and the algorithm is
literally "round the halves up." The vertex-cover LP can also be solved by a single min-cut: split each v
into a_v, b_v of weight w_v/2, replace each edge (i,j) by (a_i,b_j) and (a_j,b_i); an optimal bipartite cover
maps back to x_v = 1 / 1/2 / 0 as both / one / neither copy is chosen.

**Fixing-variables (persistency, Nemhauser–Trotter).** With P = {x*_v = 1}, Q = {x*_v = 1/2}, R = {x*_v = 0}:
R has no edges to R or Q, so every neighbor of an R-vertex is in P. For any optimal integer cover S, let
A = P \ S and B = R ∩ S. If w(A) > w(B), lowering A from 1 to 1/2 and raising B from 0 to 1/2 keeps all LP
constraints feasible (any edge leaving A has its other endpoint in S) and lowers the LP objective by
(w(A)-w(B))/2, contradicting optimality of x*. Hence w(A) ≤ w(B). Replacing S by (S ∪ P) \ R stays feasible
and does not increase cost, so some optimal integer cover contains P and is disjoint from R. Only Q is
undetermined, and the factor-2 doubling occurs solely on Q.

## Implementation
```python
import pulp

def vertex_cover_lp_rounding(G, w):
    prob = pulp.LpProblem("vc_relaxation", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", lowBound=0, upBound=1) for v in G.nodes}
    prob += pulp.lpSum(w[v] * x[v] for v in G.nodes)                  # min sum w[v] x[v]
    for u, v in G.edges:
        prob += x[u] + x[v] >= 1                                      # edge constraints
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP solve failed with status {pulp.LpStatus[status]}")
    # every edge has an endpoint with x* >= 1/2; keep those => feasible cover, w(C) <= 2*OPT
    tol = 1e-9
    return {v for v in G.nodes if x[v].value() >= 0.5 - tol}
```
