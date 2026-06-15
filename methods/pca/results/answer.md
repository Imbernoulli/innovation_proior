# Principal Component Analysis (PCA), distilled

PCA finds an ordered set of orthogonal directions in feature space along which a cloud of
correlated points varies most, and re-expresses the data in the first few of them. The same
directions arise three equivalent ways: the flat that minimizes the sum of squared
*perpendicular* distances to the points (geometric / total least squares), the directions of
maximal projected variance (variance accounting), and the eigenvectors of the data's
second-moment (covariance or correlation) matrix. It is the canonical *linear* dimensionality
reduction.

## Problem it solves

Given `n` points in `q`-dimensional space, all coordinates measured with error (no privileged
dependent variable), produce a unique, symmetric, low-dimensional summary: a small number of
new mutually-uncorrelated variables that, in order, capture as much of the total variation as
possible. The regression line fails because it is asymmetric (different line per choice of
dependent variable); factor-analytic models fail because they fix the number of latent factors
by hypothesis and are rotationally indeterminate.

## Key idea

1. **Center.** The best-fitting flat passes through the centroid (the second moment of a point
   set is least about the mean), so subtract the mean: `Xc = X - mean(X)`.
2. **Perpendicular fit == max variance.** For a line through the origin with unit direction `l`,
   total scatter splits Pythagoreanly: `sum |x|^2 = (variance along l) + (perpendicular residual)`,
   and `sum |x|^2` is constant. So minimizing perpendicular residual is identical to maximizing
   the variance of the projection onto `l`.
3. **The optimum is an eigenproblem.** In the perpendicular-plane derivation, stationarity gives
   `C l + (Q/n)l = 0`; multiplying by `l` and using `||l|| = 1` gives `Q/n = -Sigma^2`, the
   residual variance with a minus sign. For a line, maximizing projected variance `l^T C l`
   subject to `||l|| = 1` gives `C l = lambda l`: the principal directions are the eigenvectors
   of `C`, and each eigenvalue `lambda` is the variance captured along that direction. Take
   eigenvectors in order of decreasing eigenvalue. The determinantal "characteristic equation"
   `det(C - lambda I) = 0` is Cauchy's; its roots are real, and here non-negative because they
   are variances.
4. **Order and deflate.** The first component is the top eigenvector; the next maximizes the
   *residual* variance. In the loading normalization `a_1^T a_1 = k_1`, the residual correlation
   matrix is `R' = R - a_1 a_1^T`; with a unit covariance eigenvector this is the equivalent
   `C' = C - lambda_1 v_1 v_1^T`. Eigenvector orthogonality makes this automatic; a full
   eigendecomposition yields all components at once, ordered.
5. **Project.** Re-express each centered point in the principal-axis frame: `Xc @ V[:, :k]`.

The fraction of total variance retained by `k` components is
`(lambda_1 + ... + lambda_k) / sum_i lambda_i` — the stopping rule.

**Covariance vs correlation.** If the variables share a meaningful scale, diagonalize the
covariance (center only). If units are incommensurable, standardize first
(`z_i = (x_i - mean_i)/sigma_i`) and diagonalize the correlation matrix. This choice is the
choice of metric on the space.

## Computation: SVD of the centered data

For the exact full-SVD implementation, do not form `C = Xc^T Xc` and call `eigh` — that squares
the condition number and loses accuracy on the small-variance directions. Instead take the
singular value decomposition of the centered data:
```
Xc = U S V^T   (S descending)
=>  Xc^T Xc = V S^2 V^T
```
so the right singular vectors `V` are the eigenvectors of the covariance (the principal
directions), the variance of component `i` is `explained_variance_i = S_i^2 / (n - 1)`, and the
embedding is `Xc V[:, :k] = (U S)[:, :k]`. Singular-vector signs are arbitrary; fix them
deterministically (e.g. force the largest-magnitude entry of each vector positive) so the output
is reproducible. The hand-computable version of the same object is **power iteration**
(`a <- C a`, normalize, repeat — the iterate aligns with the top eigenvector because its
eigenvalue dominates geometrically) followed by deflation for each successive component.

## Working code

```python
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def svd_flip(U: NDArray[np.float64], Vt: NDArray[np.float64]):
    """Deterministic signs from rows of Vt, as in full SVD with u_based_decision=False."""
    max_abs = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs])
    return U * signs[np.newaxis, :], Vt * signs[:, np.newaxis]


class PCA:
    """Project data onto its top-k directions of greatest variance,
    via the SVD of the centered data matrix."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def _fit_full_svd(self, X: NDArray[np.float64]):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.mean_ = X.mean(axis=0)                      # centroid
        Xc = X - self.mean_                              # center
        U, S, Vt = linalg.svd(Xc, full_matrices=False)  # SVD, not eig(Xc^T Xc)
        U, Vt = svd_flip(U, Vt)                          # reproducible signs
        k = self.n_components
        self.components_ = Vt[:k]                        # top-k principal axes (rows)
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)      # variance per component
        self.singular_values_ = S[:k]
        total = (S ** 2).sum() / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.n_samples_ = n_samples
        return U, S, Vt, Xc

    def fit(self, X: NDArray[np.float64]) -> "PCA":
        self._fit_full_svd(X)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        return Xc @ self.components_.T                   # coordinates in principal frame

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        U, S, Vt, Xc = self._fit_full_svd(X)
        # Xc @ components_.T == (U S)[:, :k]
        return U[:, : self.n_components] * S[: self.n_components]
```

This is the standard linear-algebra core of `sklearn.decomposition.PCA`'s full SVD solver:
center, full SVD of the centered matrix, `svd_flip(..., u_based_decision=False)`-style sign
determinism from the rows of `Vt`, `components_ = Vt[:k]`, projection onto the top-k right
singular vectors, and `explained_variance_ = S**2 / (n_samples - 1)`.

## Why each choice

- **Perpendicular (orthogonal) residual, not vertical** — symmetric in all variables; no
  variable is assumed error-free, so the cloud has one unique fitting direction.
- **Center at the centroid** — the optimal flat provably passes through the mean; removes the
  location nuisance and leaves a pure direction-finding problem.
- **Eigenvectors of the second-moment matrix** — the constrained variance maximum *is* an
  eigenproblem; the eigenvalue is the captured variance, giving a built-in importance ordering.
- **Deflation / orthogonality** — successive components maximize residual variance, which are
  the remaining eigenvectors; mutually orthogonal by construction.
- **SVD of centered data, not `eig` of the Gram matrix** — avoids squaring the condition number;
  same principal directions and variances, more accurately in the full solver.
- **`svd_flip`** — singular-vector signs are arbitrary; fixing them from the rows of `Vt` makes
  the embedding reproducible.
