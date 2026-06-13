# Ridge regression, distilled

Ridge regression estimates the coefficients of a linear model by minimizing the residual sum
of squares plus an L2 penalty on the coefficients. Equivalently, it adds `lambda I` to the
diagonal of `X'X` before inverting. The penalty makes the estimate well-defined and stable
even when `X'X` is ill-conditioned or singular (collinear predictors, or `p > n`), at the cost
of a small bias — and for an appropriate penalty strength it has strictly lower total mean
squared error than ordinary least squares. It is the same object as Tikhonov regularization of
the linear inverse problem, the posterior mean under an isotropic Gaussian prior on the
coefficients, and (in modern training) a linear layer fit with squared-error loss and
decoupled weight decay on its weights.

## Problem it solves

Fit `y = X beta + eps` (`eps` mean-zero, `Cov = sigma^2 I`) when the columns of `X` are
nonorthogonal. Ordinary least squares `beta_hat = (X'X)^{-1}X'y` is unbiased and BLUE
(Gauss-Markov), but its covariance `sigma^2 (X'X)^{-1} = sigma^2 sum_j d_j^{-2} v_j v_j'`
(where `X = U D V'`, `d_j` the singular values) blows up like `1/d_j^2` along
small-singular-value directions, so coefficients are huge, sign-unstable, and high-variance;
and when some `d_j = 0` (exact dependence or `p > n`), `X'X` is singular and OLS is undefined.

## Key idea

Add a positive multiple of the identity to `X'X`, equivalently penalize the coefficient norm:

```
beta_hat(lambda) = (X'X + lambda I)^{-1} X' y
                 = argmin_beta  ||y - X beta||^2 + lambda ||beta||^2,    lambda >= 0.
```

`X'X + lambda I` is positive definite for `lambda > 0` (PSD `+` PD `=` PD), so it is always
invertible. The estimator is the unique global minimizer of a strictly convex objective
(Hessian `2(X'X + lambda I) > 0`). At `lambda = 0` it is OLS; as `lambda -> infinity` it
shrinks to `0`.

## Three equivalent views

- **Penalized least squares / constrained fit.** `||y - X beta||^2 + lambda||beta||^2` is, by
  Lagrangian duality, `min ||y - X beta||^2` s.t. `||beta||^2 <= c`; `lambda` is the KKT
  multiplier (larger `lambda` ⇔ tighter budget). Geometrically: the ellipsoidal RSS contours
  meet the L2 ball, shrinking the OLS solution toward the origin.
- **SVD / shrinkage.** `beta_hat(lambda) = V (D'D + lambda I)^{-1} D' U' y`, so each singular
  direction's coordinate is scaled by `d_j^2/(d_j^2 + lambda)` in `(0,1]` — `~1` for large `d_j`,
  `~0` for the unstable small `d_j`. Effective degrees of freedom `= sum_j d_j^2/(d_j^2+lambda)`
  (`= p` at `lambda=0`, `-> 0` as `lambda -> infinity`). A soft version of dropping principal
  components.
- **Bayesian.** With prior `beta ~ N(0, (sigma^2/lambda) I)` and likelihood
  `y|beta ~ N(X beta, sigma^2 I)`, the posterior is Gaussian with mean (and mode)
  `(X'X + lambda I)^{-1}X'y = beta_hat(lambda)`. So `lambda` is the prior precision — the
  strength of the belief that the coefficients are small — and the isotropic `lambda I` is the
  noncommittal spherical prior.

## Why a biased estimator can win (bias-variance and the existence theorem)

Let `W = (X'X + lambda I)^{-1} X'X`, so `beta_hat(lambda) = W beta_hat` and
`E[beta_hat(lambda)] = W beta != beta` — biased, so Gauss-Markov (which only covers unbiased
linear estimators) does not apply. Its mean squared error decomposes as

```
MSE[beta_hat(lambda)] = sigma^2 tr[W (X'X)^{-1} W']  +  beta'(W - I)'(W - I) beta.
                          \____ variance (down) ____/     \____ squared bias (up) ____/
```

The variance term decreases from `sigma^2 tr[(X'X)^{-1}]` (OLS) toward `0`; the bias term
increases from `0` toward `beta'beta`. Working with the second-moment matrices
`M(lambda) = E[(beta_hat(lambda)-beta)(beta_hat(lambda)-beta)']`,

```
M(0) - M(lambda) = lambda (X'X+lambda I)^{-1} [ 2 sigma^2 I + lambda sigma^2 (X'X)^{-1}
                                                - lambda beta beta' ] (X'X+lambda I)^{-1},
```

which is positive definite whenever the bracketed matrix is positive definite. A sufficient
condition is `2 sigma^2 I - lambda beta beta' > 0`, i.e. for

```
0 < lambda < 2 sigma^2 / (beta'beta).
```

For `beta != 0`, this interval is nonempty whenever `sigma^2 > 0`; if `beta = 0`, the rank-one
subtraction disappears. Thus there exists `lambda > 0` with
`MSE[beta_hat(lambda)] < MSE[beta_hat(0)]` whenever OLS is a meaningful full-rank comparator. In
rank-deficient settings the estimator remains well-defined, but this particular OLS-comparison proof
uses the full-rank case. The engine is that the variance saving is first-order in `lambda` while the
bias cost is second-order. In the orthonormal case `X'X = I`,
`MSE[beta_hat(lambda)] = (1+lambda)^{-2}[p sigma^2 + lambda^2 beta'beta]`, minimized at
`lambda* = p sigma^2 / (beta'beta)`. This scalar optimum and the matrix-dominance interval are
different facts; the interval is a guaranteed small-`lambda` range, not a claim that it contains the
scalar optimum for every `p`.

## Practical details

- **Standardize** the columns of `X` to unit variance and **center** `y` so that
  `sum_j beta_j^2` is a fair, comparable budget; leave the **intercept unpenalized** (otherwise
  the fit is not equivariant to shifting `y`).
- **L2, not L1:** the squared norm gives the closed form, strict convexity, eigenbasis-
  proportional shrinkage, and the Gaussian-prior reading. Ridge keeps all variables (no exact
  zeros) — it shrinks, it does not select.
- **Choosing `lambda`:** the optimum depends on the unknown `beta`, `sigma^2`. In practice
  trace the coefficient paths `beta_hat_j(lambda)` and take the smallest `lambda` past which
  they stabilize, or choose `lambda` by cross-validation. The existence theorem motivates searching
  positive penalties; held-out error chooses the predictive value empirically.

## Working code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- (a) classical closed form ----
class RidgeRegression:
    """Minimize ||y - X w||^2 + lam * ||w||^2; w = (X'X + lam I)^{-1} X'y."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.w = None

    def fit(self, X, y):
        # Assumes X standardized, y centered (intercept handled outside, unpenalized).
        p = X.shape[1]
        A = X.T @ X + self.lam * np.eye(p)   # PD, invertible even when X'X is singular
        b = X.T @ y
        self.w = np.linalg.solve(A, b)       # solve (X'X + lam I) w = X'y
        return self

    def fit_svd(self, X, y):
        # Numerically stabler: shrink each singular direction by d^2/(d^2+lam).
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        self.w = (Vt.T * (d / (d**2 + self.lam))) @ (U.T @ y)
        return self

    def predict(self, X):
        return X @ self.w


# ---- (b) linear head trained with AdamW weight decay ----
class RidgeHead(nn.Module):
    """A single linear layer trained with MSE loss and weight decay on slopes."""

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)   # [B]


def train_ridge_head(model, loader, lam):
    # Use AdamW weight_decay as the ridge penalty on slopes only; leave bias unpenalized.
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.linear.weight], "weight_decay": lam},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    model.train()
    for features, target in loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(features), target)   # data term
        loss.backward()                              # AdamW applies decoupled shrinkage to slope weights
        optimizer.step()
    return model
```

Both implement the ridge penalty: `(a)` solves the regularized normal equations directly, `(b)` uses
an `nn.Linear` prediction head trained with MSE loss while AdamW applies decoupled
`weight_decay=lambda` to the slope weights.
