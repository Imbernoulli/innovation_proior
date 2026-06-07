# Goemans–Williamson MAX CUT (SDP + random hyperplane rounding)

## Problem
Given a weighted undirected graph G = (V, E) with weights w_ij ≥ 0, find S ⊆ V maximizing the cut weight w(S, S̄) = Σ_{i∈S, j∉S} w_ij. MAX CUT is NP-complete; for two decades the best provable approximation ratio was 1/2 (independent fair coin per vertex / Sahni–Gonzales greedy). The 1/2 barrier is structural: those analyses certify only a fraction of the total weight Σ w_ij, and no graph-independent fraction above 1/2 is possible because OPT itself can be only about half of Σ w_ij.

## Key idea
Break 1/2 by comparing against a tighter, polynomial-time-computable upper bound on OPT. Write the cut as an integer quadratic program in y_i ∈ {−1, +1}; relax each scalar y_i (a unit vector in R¹) to a unit vector u_i ∈ Rⁿ and each product y_i y_j to the inner product u_i · u_j. In the Gram matrix Y = (u_i · u_j) this is a semidefinite program (linear objective, Y_ii = 1, Y ⪰ 0), solvable to additive ε in polynomial time. Recover the vectors by factoring Y. Round by a **random hyperplane** through the origin: draw a uniform unit normal r and set S = {i : u_i · r ≥ 0}. An edge at angle θ_ij = arccos(u_i · u_j) is cut with probability exactly θ_ij/π — proportional to the very angle the relaxation maximizes.

## Algorithm
1. Form the (scaled) Laplacian; the cut equals x^T(L/4)x for x ∈ {−1,+1}ⁿ since x^T L x = 4·w(S, S̄).
2. Solve the SDP: maximize Σ_{i<j} ½ w_ij(1 − Y_ij) subject to Y_ii = 1, Y ⪰ 0. Optimal value Z*_P ≥ OPT (it equals the Delorme–Poljak eigenvalue bound min_{Σu_i=0} (n/4)λ_max(L + diag(u))).
3. Factor Y = UU^T (Cholesky / eigendecomposition); row i of U is the unit vector u_i.
4. Draw r with i.i.d. N(0,1) coordinates (a spherical Gaussian → uniform direction). Set S = {i : u_i · r ≥ 0}.

## Guarantee
Pr[edge (i,j) cut] = θ_ij/π, so E[W] = Σ w_ij θ_ij/π. Edge-wise, θ/π ≥ α·½(1 − cos θ) with
α = min_{0<θ≤π} (2/π)·θ/(1 − cos θ) = 0.87856…, attained at θ* = 2.331122 rad (≈ 133.6°), the nonzero root of cos θ + θ sin θ = 1. Hence E[W] ≥ α·Z*_P ≥ α·OPT, giving a (0.87856 − ε)-approximation for any ε > 0 (the ε absorbs solving the SDP to additive accuracy in time polynomial in the input and log(1/ε)). This is the first improvement over 1/2 in roughly twenty years and the first use of semidefinite programming in approximation-algorithm design.

## Code

```python
import numpy as np
import networkx as nx
import cvxpy as cvx

def max_cut(graph):
    """Returns (colors in {-1,+1}^n, cut weight of this coloring, SDP upper bound)."""
    # cut weight = x^T (L/4) x for x in {-1,+1}^n, since x^T L x = 4 * cut
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # SDP relaxation: Y = Gram matrix of unit vectors u_i (Y >= 0, diag = 1)
    X = cvx.Variable(laplacian.shape, PSD=True)
    objective = cvx.Maximize(cvx.trace(laplacian @ X))        # linear relaxation objective
    constraints = [cvx.diag(X) == 1]                          # ||u_i|| = 1
    cvx.Problem(objective, constraints).solve()

    bound = float(np.trace(laplacian @ X.value))              # Z*_P >= OPT

    # recover vectors u_i by eigendecomposing the Gram matrix
    gram = np.array(X.value, dtype=float)
    gram = 0.5 * (gram + gram.T)
    evals, evects = np.linalg.eigh(gram)
    keep = evals > 1e-6
    sdp_vectors = evects[:, keep] @ np.diag(np.sqrt(evals[keep]))

    # random-hyperplane rounding: r uniform on the sphere via a Gaussian
    r = np.random.randn(sdp_vectors.shape[1])
    r /= np.linalg.norm(r)                                    # normalization optional; only sign matters
    colors = np.sign(sdp_vectors @ r)                        # S = {i : u_i . r >= 0}
    colors[colors == 0] = 1

    score = float(colors @ laplacian @ colors.T)              # weight of the produced cut
    return colors, score, bound
```

Because the algorithm is randomized, repeat the rounding step several times with fresh r and keep the best cut; each draw has expected weight at least 0.878·OPT, and keeping the best observed cut can only improve the returned score.
