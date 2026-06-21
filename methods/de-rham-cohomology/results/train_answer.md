The problem is to turn the local language of calculus into a detector of global shape. On a small patch of a manifold everything looks like Euclidean space, so the usual tests report that a curl-free vector field should be a gradient, a divergence-free field should be a curl, and more generally a closed differential form should have a primitive. These local tests are blind to holes: a loop that winds around a missing point is invisible inside any single coordinate chart, yet the loop cannot be shrunk to a point and the hoped-for global potential does not exist. Potential theory, Stokes' theorem on its own, and singular homology each capture part of the story, but none of them directly records the failure of a smooth differential equation to have a smooth solution. What is needed is a calculus-level invariant that keeps the differential-form data and quotients out the local clutter.

The right construction is de Rham cohomology. Let M be a smooth manifold and let Omega^k(M) be the space of smooth k-forms. The exterior derivative d raises degree by one and satisfies d^2 = 0, so every exact form d eta is automatically closed. The k-th de Rham cohomology group is the quotient H_dR^k(M) = ker(d: Omega^k -> Omega^(k+1)) / im(d: Omega^(k-1) -> Omega^k). It is precisely the space of closed k-forms modulo those that are globally differentials of lower-degree forms. The Poincare lemma removes all positive-degree local information: on any chart diffeomorphic to R^n every closed form of positive degree is exact. Therefore H_dR^k measures only global obstructions. Integration over smooth k-cycles gives a well-defined pairing, and de Rham's theorem says this pairing identifies H_dR^k(M) with real singular cohomology H^k(M; R). In short, de Rham cohomology is the space of periods of closed forms, with exact forms discarded because their periods are zero.

The following Python script implements the finite-dimensional algebraic skeleton of this construction for a cochain complex. It stores the coboundary maps d^k as real matrices and computes the dimensions of the cohomology spaces ker(d^k) / im(d^(k-1)) by rank-nullity. It also includes a one-dimensional circle example, whose first cohomology is one-dimensional.

```python
import numpy as np
from scipy.linalg import null_space, orth

def cohomology_dimensions(coboundaries, eps=1e-9):
    """
    Compute dimensions of H^k = ker(d^k) / im(d^(k-1))
    for a cochain complex  C^0 --d0--> C^1 --d1--> C^2 ...
    coboundaries[k] is the matrix of d^k : C^k -> C^(k+1).
    """
    dims = []
    for k, d in enumerate(coboundaries):
        rank_d = np.linalg.matrix_rank(d, tol=eps)
        ker_d = d.shape[1] - rank_d
        if k == 0:
            rank_prev = 0
        else:
            rank_prev = np.linalg.matrix_rank(coboundaries[k - 1], tol=eps)
        dims.append(ker_d - rank_prev)
    return dims

# Example: cochain complex for a triangulated circle S^1.
# Vertices: 0, 1. Edges: a (0->1), b (1->0).
# d^0 maps a vertex function f to edge differences.
d0 = np.array([[ -1,  1],   # edge a: f(1)-f(0)
               [  1, -1]],  # edge b: f(0)-f(1)
              dtype=float)
# No 2-cells, so d^1 is the zero map from edges to nothing.
d1 = np.zeros((0, 2), dtype=float)

dims = cohomology_dimensions([d0, d1])
print("H^0 dimension:", dims[0])  # 1
print("H^1 dimension:", dims[1])  # 1
```
