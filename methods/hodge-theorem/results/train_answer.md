De Rham cohomology records the topology of a smooth manifold as closed differential forms modulo exact forms, but that description is deliberately a quotient. A single cohomology class is an entire affine family of smooth representatives, and nothing in the definition itself picks one. Existing ideas do not solve this representative problem cleanly. Taking an arbitrary closed form ignores the ambiguity entirely; computing periods over homology cycles identifies the class as a functional on chains, but it does not return a canonical form living on the manifold; the scalar Laplacian on functions only captures degree-zero cohomology, since harmonic functions on a compact connected manifold are constant; and Helmholtz decompositions from vector calculus are tied to Euclidean or boundary-decay assumptions. Boundary versions of Hodge theory also complicate the picture, because harmonic forms must be paired with absolute or relative boundary conditions before they line up with cohomology.

The right fix is the Hodge theorem, also called the Hodge decomposition theorem. On a smooth compact oriented Riemannian manifold without boundary, the metric gives the space of differential forms an L2 inner product. That inner product lets the geometry itself choose a representative inside each de Rham class by picking the form of smallest L2 norm. The formal adjoint d* of the exterior derivative d is defined by integration by parts, and the Hodge Laplacian on k-forms is Delta = d d* + d* d. Because the manifold is closed, the identity <Delta alpha, alpha> = ||d alpha||^2 + ||d* alpha||^2 shows that a form is harmonic exactly when it is both closed and co-closed. The operator Delta is elliptic and self-adjoint, and compactness gives a finite-dimensional harmonic space, a Green operator G, and orthogonal projection H onto the harmonic forms satisfying I = H + Delta G = H + G Delta.

Expanding Delta in this identity yields the orthogonal Hodge decomposition Omega^k(M) = Harm^k(M) ⊕ im d ⊕ im d*. The three summands are mutually orthogonal: a harmonic form is killed by d and d*, exact and co-exact forms are orthogonal because d^2 = 0, and the projection H picks out the harmonic part of any form. For a closed form omega, the co-exact summand must vanish, leaving omega = h + d a with h harmonic. Thus every de Rham class has a harmonic representative. Uniqueness follows because an exact harmonic form h = d a has norm squared ||h||^2 = <h, d a> = <d* h, a> = 0, so h = 0. The map Harm^k(M) -> H^k_dR(M) sending h to its class is therefore an isomorphism. Moreover, the orthogonal decomposition gives ||h + d a||^2 = ||h||^2 + ||d a||^2, so the harmonic representative is the unique minimizer of L2 energy in its class.

The following code illustrates the same algebraic structure on a finite triangulated circle. The coboundary operator d0 plays the role of the exterior derivative on 0-forms, its transpose plays the role of d*, and the nullspace of the discrete Laplacian gives the harmonic forms. A random closed 1-cochain is projected onto its harmonic representative; the residual is exact, confirming the decomposition.

```python
import numpy as np

def circle_complex(n=6):
    """Return oriented edges (i, i+1 mod n) of a triangulated circle."""
    verts = np.arange(n)
    return np.column_stack((verts, np.roll(verts, -1)))

def coboundary_0(edges, n):
    """d0 maps vertex values to edge differences."""
    E = edges.shape[0]
    d0 = np.zeros((E, n))
    for k, (i, j) in enumerate(edges):
        d0[k, i] = -1.0
        d0[k, j] = 1.0
    return d0

def harmonic_basis(L, tol=1e-10):
    """Orthonormal basis for ker(L) via SVD."""
    _, s, vh = np.linalg.svd(L)
    rank = np.sum(s > tol)
    return vh[rank:, :].T  # columns form an orthonormal basis

# Build the circle and the 0 -> 1 coboundary operator
n = 6
edges = circle_complex(n)
d0 = coboundary_0(edges, n)
d0star = d0.T

# Laplacians: L0 on 0-cochains, L1 on 1-cochains (d1 is zero in 1D)
L0 = d0star @ d0
L1 = d0 @ d0star

H0 = harmonic_basis(L0)  # should be constants, dimension 1
H1 = harmonic_basis(L1)  # should be constant edge values, dimension 1

# Project a random closed 1-cochain onto its harmonic representative
rng = np.random.default_rng(0)
c = rng.normal(size=edges.shape[0])
h = H1 @ (H1.T @ c)
residual = c - h

# Verify that the residual is exact: it is the gradient of a vertex function
x, *_ = np.linalg.lstsq(d0, residual, rcond=None)
print("b0:", H0.shape[1], "b1:", H1.shape[1])
print("Harmonic representative:", h)
print("Residual is exact?", np.allclose(d0 @ x, residual))
print("Norm minimization:", np.linalg.norm(c)**2,
      np.linalg.norm(h)**2 + np.linalg.norm(residual)**2)
```

This discrete example captures the theorem's core mechanism: once a metric provides an inner product, the Laplacian detects closed and co-closed forms, elliptic compactness makes the harmonic projection well defined, and the orthogonal decomposition guarantees exactly one energy-minimizing representative per cohomology class.