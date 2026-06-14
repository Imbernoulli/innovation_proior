# Kernel ridge regression, distilled

Kernel ridge regression (KRR) combines ridge regression (squared-error fit with an L2 penalty)
with the kernel trick. It learns a function `f(x) = sum_t c_t K(x_t, x)` that is linear in a
kernel-induced feature space and nonlinear in the original input, with the dual coefficients
solving a single regularized linear system `(K + a I) c = y`. It is the same predictive object
as the Gaussian-process posterior mean (Kriging), and the same model form as support-vector
regression but with squared loss instead of an epsilon-insensitive loss — which is exactly what
gives KRR a closed form.

## Problem it solves

Fit a flexible nonlinear regressor from `(x_1, y_1), ..., (x_T, y_T)` under squared error,
when (a) a linear-in-`x` model is too rigid, (b) the natural route to nonlinearity — fitting a
linear model in a rich feature map `phi(x)` — is computationally impossible because `phi` is
huge or infinite-dimensional, and (c) the fit must be regularized so it is stable and
generalizes instead of interpolating noise.

## Key idea

Express the regularized fit so its cost depends on the number of examples `T`, not on the
feature dimension. Penalized least squares in feature space has the solution
`w = (Phi'Phi + a I)^{-1} Phi'y`; the push-through identity
`(Phi'Phi + a I)^{-1} Phi' = Phi'(Phi Phi' + a I)^{-1}` rewrites it as
`w = Phi'(Phi Phi' + a I)^{-1} y = sum_t alpha_t phi(x_t)` — so the optimal weight is a
combination of the training feature vectors (the representer form), and every place `phi`
appears it appears only as an inner product `phi(u) . phi(v)`. Replace those inner products by
a kernel `K(u, v) = phi(u) . phi(v)`; then `phi` never has to be computed:

```
K_{s,t} = K(x_s, x_t)            (T x T Gram matrix)
(K + a I) c = y                  =>   c = (K + a I)^{-1} y       (one regularized solve)
f(x) = sum_t c_t K(x_t, x) = k(x)' c,   k(x)_t = K(x_t, x)
```

The only objects are the `T x T` kernel matrix and one linear solve, independent of the feature
dimension — this is what defeats the curse of dimensionality of explicit feature-space
regression.

## Three equivalent derivations (same predictor)

- **Push-through on primal ridge.** `(Phi'Phi + a I)^{-1} Phi' = Phi'(Phi Phi' + a I)^{-1}`
  moves the inverse from `(feature x feature)` to `(T x T)`; predictions then use `phi` only via
  inner products. Gives `c = (K + a I)^{-1} y`, `f(x) = k(x)' c`.

- **Constrained dual (Lagrange/KKT).** Minimize `a ||w||^2 + sum_t xi_t^2` s.t.
  `y_t - w . phi(x_t) = xi_t`. Stationarity gives `w = (1/2a) sum_t alpha_t phi(x_t)` and
  `xi_t = alpha_t / 2` (each multiplier is twice its residual). Substituting back yields
  `(K + a I) alpha = 2a y`, i.e. `alpha = 2a (K + a I)^{-1} y`; the `2a` cancels in the
  prediction `w . phi(x) = (1/2a) k(x)' alpha = y'(K + a I)^{-1} k(x)`. The stored coefficient
  is the clean `c = (K + a I)^{-1} y`.

- **RKHS / representer theorem.** Minimize `sum_t (y_t - f(x_t))^2 + a ||f||^2` over the kernel's
  Hilbert space. The representer theorem (`f = f_par + f_perp`; the data term sees only `f_par`,
  the penalty kills `f_perp`) forces a finite representation `f = sum_t c_t K(., x_t)`.
  Substitution gives objective `c'K^2 c - 2 y'K c + y'y + a c'K c`; stationarity gives
  `K((K + a I)c - y) = 0`. Since `K` may be singular, diagonalize `K = U diag(d_i) U'`: for
  every `d_i > 0`, the coordinate equation is `(d_i + a) z_i = r_i`, while `d_i = 0`
  coordinates do not change the function or its RKHS norm. The always-defined representative
  used by the linear solve is `(K + a I)c = y`.

## Why these choices

- **Squared loss (not epsilon-insensitive / hinge):** makes the optimality condition *linear*
  in the coefficients, so the dual is a single linear system `(K + a I) c = y` with a one-shot
  closed form (Cholesky solve). The price is a *dense* `c` (no sparsity); SVR's
  epsilon-insensitive loss gives a sparse solution but only via a quadratic program. For medium
  `T`, the exact linear solve is preferable.
- **L2 penalty `a I`:** `K` is PSD but can be singular / ill-conditioned (near-duplicate
  inputs). Adding `a I` lifts every eigenvalue by `a`, making `K + a I` strictly positive
  definite (always invertible, Cholesky-stable) and shrinking the coefficients. `a -> 0` is pure
  interpolation (overfits, unstable); large `a` shrinks the fit toward the zero/prior-mean
  function. `a` is the interpolation-to-smoothing dial.
- **Dual / kernel form:** the `T x T` solve works even for infinite-dimensional `phi`; the
  push-through identity guarantees it is the same solution as the (infeasible) primal.
- **Gaussian / RBF kernel `K(x, y) = exp(-gamma ||x - y||^2)`:** symmetric positive definite for
  any `gamma > 0` (always a legal kernel, so `K + a I` is PD), infinite-dimensional feature map
  (universal), and a single bandwidth knob `gamma`. Large `gamma` = narrow bump = local,
  high-capacity, can overfit; small `gamma` = broad = smooth, can underfit.
- **`gamma = 1 / n_features` default:** keeps the exponent argument `gamma ||x - y||^2` of order
  1 across dimensions (since `||x - y||^2` grows ~linearly in `n` for standardized inputs), so
  the kernel neither saturates to 0 nor sticks at 1.
- **Put inputs on comparable numeric scales:** the RBF rides on Euclidean distance, which is
  scale-sensitive. Centering/scaling numeric columns keeps one raw unit choice from dominating
  the distance; categorical information must be encoded numerically before it can enter the
  kernel.

## Bayesian / Kriging equivalence

With prior `w ~ N(0, (1/2a) I)` and Gaussian observation noise of variance `1/2`,
`cov(y_s, y_t) = (1/(2a))(K + a I)_{s,t}` and `cov(y_t, w . phi(x)) = (1/(2a)) k_t(x)`.
Gaussian conditioning gives posterior mean `k(x)'(K + a I)^{-1} y` — KRR's formula. More
generally, for a zero-mean GP with prior covariance amplitude `tau^2` and independent noise
variance `sigma^2`, the posterior mean is `k(x)'(K + (sigma^2/tau^2) I)^{-1} y`; the ridge
constant is the noise-to-prior-variance ratio.

## Final algorithm

```
fit(X, y):
    K = kernel(X, X)                 # T x T Gram matrix
    K += a * I                       # the only place the regularizer enters
    c = solve(K, y, assume_a="pos")  # Cholesky; (K + a I) c = y; lstsq fallback if singular
    store c and X

predict(X_new):
    K_test = kernel(X_new, X_fit)    # m x T cross-kernel
    return K_test @ c                # f(x) = sum_t c_t K(x_t, x)
```

## Working code

A faithful kernel ridge regressor (the structure used by the scikit-learn estimator: a
kernel matrix, the diagonal lift `K + a I`, a Cholesky solve for `dual_coef_`, and a
cross-kernel matrix-vector product at predict time):

```python
import numpy as np
from scipy import linalg


def rbf_kernel(A, B=None, gamma=None):
    """K(x, y) = exp(-gamma * ||x - y||^2), vectorized over all pairs of rows."""
    if B is None:
        B = A
    if gamma is None:
        gamma = 1.0 / A.shape[1]
    sq_dist = (np.sum(A * A, axis=1)[:, None]
               + np.sum(B * B, axis=1)[None, :]
               - 2.0 * A @ B.T)
    return np.exp(-gamma * np.clip(sq_dist, 0.0, None))


class KernelRidge:
    """f(x) = sum_t dual_coef_[t] * K(x_t, x), with (K + alpha I) dual_coef_ = y."""

    def __init__(self, alpha=1.0, kernel="rbf", gamma=None):
        self.alpha = alpha            # L2 / Tikhonov strength (= a)
        self.kernel = kernel
        self.gamma = gamma            # for RBF, None means 1/n_features

    def _kernel(self, A, B=None):
        if self.kernel != "rbf":
            raise ValueError("This compact version implements the RBF path.")
        return rbf_kernel(A, B, gamma=self.gamma)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ravel = False
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            ravel = True

        K = self._kernel(X, None)                        # T x T Gram matrix
        K[np.diag_indices_from(K)] += self.alpha         # (K + alpha I): PD for alpha > 0
        try:
            dual_coef = linalg.solve(K, y, assume_a="pos", overwrite_a=False)
        except np.linalg.LinAlgError:                    # numerically singular -> least squares
            dual_coef = linalg.lstsq(K, y)[0]
        self.dual_coef_ = dual_coef.ravel() if ravel else dual_coef
        self.X_fit_ = X                                  # predict needs K(x_new, x_t)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        K_test = self._kernel(X, self.X_fit_)            # m x T cross-kernel
        return K_test @ self.dual_coef_                  # f(x) = sum_t c_t K(x_t, x)
```

For an RBF run, feed the estimator numeric features on comparable scales before calling
`KernelRidge(alpha=..., kernel="rbf", gamma=None)`, letting the RBF kernel default use
`gamma = 1 / n_features`.
