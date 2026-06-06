# The Johnson–Lindenstrauss Lemma

## Problem

Given n points in R^d (d arbitrarily large) and a distortion parameter 0 < ε < 1, find a map
f : R^d → R^k with k as small as possible such that every pairwise squared distance is preserved
within a (1 ± ε) factor:

    (1 − ε) ‖u − v‖² ≤ ‖f(u) − f(v)‖² ≤ (1 + ε) ‖u − v‖²    for all points u, v.

## Key idea

A single **random linear projection** does the job, and the target dimension depends only on
log n and ε — **not on d**.

1. **Linearity reduces distances to norms.** For a linear f, f(u) − f(v) = f(u − v), so
   preserving the distance u↔v is preserving the norm of the difference vector w = u − v. The
   problem becomes: preserve the norm of each of the C(n,2) difference vectors.

2. **Distributional JL (the engine).** Let R be a k × d Gaussian random matrix whose entries
   have mean 0 and standard deviation 1/√k; equivalently, draw standard Gaussian rows and
   scale the map by 1/√k. For any fixed unit vector x, ‖f(x)‖² has mean 1 and concentrates:
   using the χ² moment generating function E[e^{sX²}] = (1 − 2s)^{−1/2} (X ~ N(0,1)) through
   Markov's inequality, with β the deviation factor,

       Pr[ ‖f(x)‖² ≤ β ] ≤ exp( (k/2)(1 − β + ln β) )   (β < 1),
       Pr[ ‖f(x)‖² ≥ β ] ≤ exp( (k/2)(1 − β + ln β) )   (β > 1).

   The ambient dimension d cancels out of these bounds. Setting β = 1 − ε and using
   ln(1 − x) ≤ −x − x²/2 gives the lower tail exp(−kε²/4). Setting β = 1 + ε and using
   ln(1 + x) ≤ x − x²/2 + x³/3 gives the upper tail
   exp(−k(ε²/2 − ε³/3)/2), which is binding for 0 < ε < 1. The two-sided per-vector failure
   probability is therefore at most 2·exp(−k(ε²/2 − ε³/3)/2).

3. **Union bound.** Choosing

       k ≥ 4 (ε²/2 − ε³/3)^{−1} ln n    (≈ 8 ln n / ε² for small ε)

   makes each pair fail with probability ≤ 2/n². Over all C(n,2) pairs the union-bound failure
   probability is C(n,2)·2/n² = 1 − 1/n < 1, so with positive probability a *single* random matrix
   preserves every distance at once. Repeating O(n) draws boosts the success probability and gives
   a randomized polynomial-time construction.

## Choices of random matrix (unscaled entries mean 0 and variance 1; final entries scaled by 1/√k)

- **Gaussian:** unscaled entries N(0,1), implemented with standard deviation 1/√k — gives the exact
  χ² MGF.
- **Rademacher / ±1 coins:** entries ±1 with probability 1/2 each — sub-Gaussian, even moments
   dominated by the Gaussian's, so the same tail holds; integer arithmetic, one bit per entry.
- **Sparse (database-friendly):** entries √3·{+1 w.p. 1/6, 0 w.p. 2/3, −1 w.p. 1/6} — two-thirds
   zeros give a 3× speedup; the √3 is forced by variance 1, and 2/3 is the edge for the same
   moment-domination proof. In the scikit-learn-style generator, density = 1/3 reproduces this
   Achlioptas distribution; more generally s = 1/density gives nonzeros ±√s/√k with probability
   1/(2s) each.

For the Achlioptas sign/sparse family, requesting failure probability at most n^−β gives
k₀ = (4 + 2β)/(ε²/2 − ε³/3) · log n.

## Code

```python
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement

def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    eps = np.asarray(eps)
    n_samples = np.asarray(n_samples)
    if np.any(eps <= 0.0) or np.any(eps >= 1):
        raise ValueError("eps must be in (0, 1)")
    if np.any(n_samples <= 0):
        raise ValueError("n_samples must be positive")
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int64)

def _check_density(density, n_features):
    if density == "auto":
        density = 1 / np.sqrt(n_features)
    elif density <= 0 or density > 1:
        raise ValueError("density must be in (0, 1]")
    return density

def _check_input_size(n_components, n_features):
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")

def _gaussian_random_matrix(n_components, n_features, random_state=None):
    _check_input_size(n_components, n_features)
    rng = check_random_state(random_state)
    return rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(n_components),
        size=(n_components, n_features),
    )

def _sparse_random_matrix(n_components, n_features, density="auto", random_state=None):
    _check_input_size(n_components, n_features)
    density = _check_density(density, n_features)
    rng = check_random_state(random_state)
    if density == 1:
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return components / np.sqrt(n_components)

    indices = []
    offset = 0
    indptr = [offset]
    for _ in range(n_components):
        nnz = rng.binomial(n_features, density)
        indices_i = sample_without_replacement(n_features, nnz, random_state=rng)
        indices.append(indices_i)
        offset += nnz
        indptr.append(offset)
    indices = np.concatenate(indices)
    data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1
    components = sp.csr_array((data, indices, indptr), shape=(n_components, n_features))
    return np.sqrt(1 / density) / np.sqrt(n_components) * components

class RandomProjection:
    def __init__(self, n_components="auto", eps=0.1, kind="gaussian",
                 density="auto", dense_output=False, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.kind = kind
        self.density = density
        self.dense_output = dense_output
        self.random_state = random_state

    def fit(self, X):
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        if self.n_components == "auto":
            k = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)
        else:
            k = self.n_components
        self.n_components_ = int(k)
        if self.kind == "gaussian":
            self.components_ = _gaussian_random_matrix(
                self.n_components_, n_features, random_state=random_state
            )
        else:
            self.density_ = _check_density(self.density, n_features)
            self.components_ = _sparse_random_matrix(
                self.n_components_, n_features, density=self.density_,
                random_state=random_state
            )
        return self

    def transform(self, X):
        return safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
```
