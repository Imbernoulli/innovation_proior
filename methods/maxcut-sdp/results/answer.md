# Goemans-Williamson MAX-CUT

## Problem

Given an undirected graph G=(V,E) with nonnegative edge weights w_ij, partition V into two sides
to maximize the total weight of edges crossing the partition (MAX-CUT). Exact MAX-CUT is NP-hard;
the goal is a polynomial-time algorithm with the best provable worst-case ratio ALG/OPT. The
trivial fair-coin assignment already guarantees 1/2; the question is how to provably beat it.

## Key idea

Encode side membership as a sign y_i ∈ {-1,+1}; edge (i,j) is cut iff y_i y_j = -1, and the cut
weight is (1/2) Σ_{(i,j)} w_ij (1 - y_i y_j). The objective sees only the products y_i y_j.
Independent edge-variable LPs miss odd-cycle consistency: on K_3 the naive LP reaches 3 while
OPT=2, and on complete graphs the integral/LP ratio tends to 1/2. Escape by relaxing each sign to a
**unit vector** v_i on the sphere in R^n and replacing y_i y_j by ⟨v_i,v_j⟩. The pairwise inner
products form a positive-semidefinite Gram matrix with unit diagonal, so the relaxation is a
**semidefinite program**, solvable to additive ε in polynomial time, with optimum SDP ≥ OPT. Recover
an integral cut by **random-hyperplane rounding**: draw a Gaussian direction r and set
y_i = sign⟨v_i,r⟩.

## Algorithm and analysis

SDP relaxation (over the Gram matrix X_ij = ⟨v_i,v_j⟩):

    maximize  (1/2) Σ_{(i,j)∈E} w_ij (1 - X_ij)   s.t.  X ⪰ 0,  X_ii = 1.

Rounding: factor X = Q Q^T (matrix square root / Cholesky), use row i of Q as v_i, draw
r ~ N(0,I_n), and set y_i = sign(⟨v_i,r⟩) = sign((Q r)_i).

Guarantee. For an edge with angle θ_ij = arccos⟨v_i,v_j⟩ ∈ [0,π], the projection of the
spherically-symmetric Gaussian onto the plane of v_i,v_j is uniform in direction, so the hyperplane
separates them with probability exactly

    Pr[(i,j) cut] = θ_ij / π.

Hence E[cut weight] = Σ w_ij θ_ij/π. With

    α = (2/π) · min_{0<θ≤π}  θ/(1 - cos θ) ≈ 0.87856,

attained at θ* ≈ 2.3311 rad ≈ 133.56° (where (1 - cos θ*) = θ* sin θ*, cos θ* ≈ -0.689), one has
θ/π ≥ α·(1 - cos θ)/2 for all θ ∈ (0,π]. Therefore

    E[cut weight] = Σ w_ij θ_ij/π ≥ α · Σ w_ij (1 - cos θ_ij)/2 = α·SDP ≥ α·OPT,

a 0.87856-approximation in expectation. Repeating the rounding and keeping the best cut is useful
in practice; derandomization can recover a cut with value at least the expectation.

## Code

This is the standard unweighted edge-list implementation; weighted edges multiply the corresponding
objective terms by w_ij.

```python
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def gw(n, edges):
    """Goemans-Williamson MAX-CUT: returns x in {-1,+1}^n with E[cut] >= 0.87856 * OPT."""
    # SDP relaxation over the Gram matrix X_ij = <v_i, v_j>
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]                              # X PSD  <=>  X is a Gram matrix
    constraints += [X[i, i] == 1 for i in range(n)]     # unit vectors: ||v_i|| = 1
    objective = sum(0.5 * (1 - X[i, j]) for (i, j) in edges)   # cut weight in vector form
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    # random-hyperplane rounding
    Q = sqrtm(X.value).real        # X = Q Q^T; row i of Q is the vector v_i
    r = np.random.randn(n)          # Gaussian => spherically symmetric hyperplane through 0
    x = np.sign(Q @ r)              # y_i = sign(<v_i, r>)
    return x

def cut(x, edges):
    """Edges crossing the partition described by x in {-1,+1}^n."""
    return [(i, j) for (i, j) in edges if np.sign(x[i] * x[j]) < 0]
```

## Optimality (background)

MAX-CUT is NP-hard, and NP-hard to approximate beyond 16/17 ≈ 0.941. Under the Unique Games
Conjecture, the constant α ≈ 0.87856 is optimal: no polynomial-time algorithm achieves a better
worst-case ratio.
