# Context: approximating the maximum cut of a weighted graph

## Research question

Given an undirected graph G = (V, E) on n vertices with nonnegative edge weights w_ij = w_ji, the maximum cut problem asks for a vertex set S ⊆ V that maximizes the weight of the edges crossing between S and its complement, w(S, S̄) = Σ_{i∈S, j∉S} w_ij. The problem is NP-complete — it is one of Karp's original list, and stays NP-complete even when every edge has weight 1 — so an exact polynomial algorithm is not expected. The realistic goal is a **ρ-approximation algorithm**: a polynomial-time algorithm that on every input returns a cut of weight at least ρ times the optimum (for a randomized algorithm, expected weight at least ρ·OPT). The constant ρ is the performance guarantee, and the whole game is to push ρ as high as one can *prove* in the worst case.

## Background

**The 1/2 barrier and existing algorithms.** The simplest algorithm assigns each vertex independently to S or S̄ by a fair coin flip. An edge (i, j) is cut with probability exactly 1/2, so the expected cut weight is (1/2)·Σ_{i<j} w_ij. Since the optimum cut can be at most the total weight, this gives expected cut ≥ (1/2)·OPT — a 1/2-approximation (Sahni and Gonzales 1976 give a deterministic greedy version with the same guarantee). Later analyses (Vitányi 1981; Poljak and Turzík 1982; Haglin and Venkatesan 1991; Hofmeister and Lefmann 1995) add a vanishing lower-order term of the form 1/2 + 1/poly(n,m), leaving the worst-case constant at 1/2. All of these analyses bound the returned cut as a *fraction of the total edge weight* Σ_{i<j} w_ij.

**Encoding a cut as a quadratic form.** Assign each vertex a variable y_i ∈ {−1, +1}, with S = {i : y_i = +1}. Then ½(1 − y_i y_j) is 1 when y_i ≠ y_j (the edge is cut) and 0 otherwise, so the cut weight is exactly Σ_{i<j} ½ w_ij(1 − y_i y_j). MAX CUT is the integer quadratic program of maximizing this over y ∈ {−1, +1}^n. The same data has a spectral face: define the Laplacian L with L_ij = −w_ij for i ≠ j and L_ii = Σ_j w_ij. For x ∈ {−1, +1}^n, the identity x^T L x = Σ_{i<j} w_ij (x_i − x_j)^2 holds, and (x_i − x_j)^2 is 4 on cut edges and 0 otherwise, so x_S^T L x_S = 4·w(S, S̄).

**The eigenvalue upper bound (Delorme and Poljak 1993).** This spectral form yields a computable upper bound that is empirically far tighter than the total weight. By the Rayleigh principle, for any symmetric M and any nonzero x, λ_max(M) ≥ x^T M x / x^T x. Introduce a *correcting vector* u with Σ_i u_i = 0 and U = diag(u). For x_S ∈ {−1,+1}^n we have x_S^T U x_S = Σ_i u_i = 0, so 4·mc(G) = x_S^T L x_S = x_S^T (L + U) x_S ≤ λ_max(L + U)·(x_S^T x_S) = n·λ_max(L + U). Thus mc(G) ≤ (n/4)·λ_max(L + U) for *every* correcting vector u, and minimizing the right side over all correcting vectors gives a bound φ(G) = min_u (n/4)·λ_max(L + diag(u)) computable in polynomial time to arbitrary precision. Delorme and Poljak measured how tight φ(G) is and found it strikingly close to mc(G): the worst ratio mc(G)/φ(G) they could exhibit is the 5-cycle, 32/(25 + 5√5) ≈ 0.88445.

**Semidefinite programming as the optimization engine.** A semidefinite program optimizes a linear function of a symmetric matrix subject to linear equality constraints and the constraint that the matrix be positive semidefinite. The PSD cone is convex, so this is convex programming; it inherits a clean Lagrangian duality from cone programming, and for any ε > 0 a solution within additive ε of optimal can be found in time polynomial in the input size and log(1/ε) (via the ellipsoid method or interior-point methods). A symmetric matrix A is PSD iff all its eigenvalues are nonnegative iff it factors as A = B^T B; the columns of B are then vectors whose pairwise inner products are the entries of A (A is their Gram matrix), and B can be recovered from A by Cholesky or eigendecomposition. Lovász's bound on the Shannon capacity of a graph, and the Lovász–Schrijver lift-and-project machinery, had already shown SDP gives tighter relaxations of combinatorial problems than linear programming.

## Baselines

- **Fair-coin / Sahni–Gonzales (1976), ρ = 1/2.** Assign each vertex to S independently with probability 1/2 (or greedily place vertices to maximize the partial cut). Each edge is cut w.p. 1/2, so expected cut = (1/2)Σ w_ij ≥ (1/2)OPT.

- **Linear-programming / combinatorial relaxations.** The natural LP and combinatorial relaxations of the cut polytope have integrality gap; the triangle C3 is a standard test instance. These approaches bound the cut as a fraction of the total weight.

- **Delorme–Poljak eigenvalue bound (1993), an upper bound.** φ(G) = min over correcting vectors of (n/4)λ_max(L + diag(u)) is a polynomial-time, empirically tight *upper bound* on mc(G). It supplies a number certifying that mc(G) ≤ φ(G).

## Evaluation settings

The natural yardstick is the worst-case approximation ratio: the infimum over all weighted graphs of (algorithm's expected cut)/(OPT), proved analytically rather than measured. Instances of interest for stress-testing the ratio are small dense graphs where relaxations are loose — the triangle C3, the 5-cycle C5, complete graphs, and families of random weighted graphs. The relevant primitives for an implementation are a convex/semidefinite optimization solver and a graph Laplacian; metrics are the relaxation's optimal value (an upper bound on OPT) and the weight of the rounded integral cut.

## Code framework

Pre-existing primitives: a graph library that builds the Laplacian L (e.g. `networkx.laplacian_matrix`), a numerical linear-algebra library for eigendecomposition and matrix products (`numpy`), and a convex-optimization modeling layer for matrix-valued conic models (`cvxpy` with an appropriate solver). The harness computes the quadratic cut form and leaves one empty slot: produce a {−1,+1} assignment together with a computable upper bound on OPT.

```python
import numpy as np
import networkx as nx
import cvxpy as cvx

def max_cut(graph: nx.Graph):
    # Cut as a quadratic form via the (scaled) Laplacian.
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # --- the one open slot ---
    # TODO: produce a {-1,+1} assignment and an upper bound on OPT.
    colors = ...   # TODO: vertex assignment in {-1, +1}
    bound = ...    # TODO: upper bound on OPT

    score = colors @ laplacian @ colors.T     # weight of the produced cut
    return colors, score, bound
```
