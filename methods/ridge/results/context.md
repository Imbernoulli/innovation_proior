## Research question

I have a linear model `y = X beta + eps`, with an `n x p` design matrix `X`, an unknown
coefficient vector `beta in R^p`, and noise `eps` with `E[eps] = 0`, `Cov(eps) = sigma^2 I`.
I want to estimate `beta` (or, equivalently, build the linear predictor `x -> x' beta`) from
the data. The default tool is ordinary least squares. On well-behaved (orthogonal) designs it
is excellent. The case of interest is *nonorthogonal* `X` — columns that are strongly
correlated, nearly linearly dependent, or more numerous than the observations (`p > n`). The
question is how to estimate `beta` across the whole range from orthogonal to super-collinear
designs, and what figure of merit to judge such an estimate by.

## Background

The field state rests on a small number of load-bearing results about the linear model.

**The least-squares / maximum-likelihood estimator and its optimality.** Minimizing the
residual sum of squares `||y - X beta||^2` (Gauss and Legendre, early 1800s) gives, when
`X'X` is invertible, the closed form

```
beta_hat = (X'X)^{-1} X' y,
```

with `E[beta_hat] = beta` (unbiased) and covariance `Cov(beta_hat) = sigma^2 (X'X)^{-1}`. Under
Gaussian noise this is also the maximum-likelihood estimator. The **Gauss-Markov theorem**
states that among all *linear unbiased* estimators of `beta`, least squares has the smallest
variance (it is BLUE — the best linear unbiased estimator).

**The role of the eigenvalues of `X'X`.** The optimality is about unbiasedness; it concerns
the linear unbiased class, not absolute error magnitude. Write the eigendecomposition
`X'X = sum_j d_j^2 v_j v_j'` (the `d_j` are the singular values of `X`, the `v_j` the right
singular vectors). Then

```
Cov(beta_hat) = sigma^2 (X'X)^{-1} = sigma^2 sum_j d_j^{-2} v_j v_j'.
```

A direction `v_j` with a small singular value `d_j` contributes variance proportional to
`1/d_j^2`. Collinearity *is* the presence of such small singular values: when two covariates
are nearly proportional, the design separates their contributions only weakly along that
direction. The associated symptoms — coefficients with large magnitudes, unstable signs, large
standard errors — are observed facts about least squares on correlated data. In the limit of
exact linear dependence, or `p > n`, some `d_j = 0`, `X'X` is singular, its inverse does not
exist, and the normal equations `X'X beta = X'y` have a whole affine space of solutions.

**Ill-posed inverse problems and stabilization.** Independently of statistics, the theory of
*ill-posed* inverse problems studies the same situation in the abstract: solving `A f = g`
for `f` when `A` has tiny singular values amplifies noise. Tikhonov (1963, "Solution of
incorrectly formulated problems and the regularization method") showed that an unstable inverse
problem can be made well behaved by trading raw data fidelity against a controlled size /
smoothness requirement on the solution — accepting a controlled distortion of the solution in
exchange for a more stable estimate.

**Shrinkage and the inadmissibility of the unbiased estimator.** A decision-theoretic thread
examines the price of unbiasedness. Stein (1956) and James & Stein (1961) showed that for
estimating the mean of a `p`-dimensional Gaussian with `p >= 3`, the obvious unbiased estimator
(the sample value itself) is **inadmissible** under summed squared-error loss: the explicitly
biased shrinkage estimator `(1 - (p-2) sigma^2 / ||x||^2) x`, which pulls the estimate toward
the origin, has strictly lower total mean squared error everywhere. Gauss-Markov does not cover
it because it is biased; Gauss-Markov protects only the unbiased class.

**Mean squared error as a figure of merit.** A scalar quality measure for an estimator
`theta_hat` of `theta` is `MSE(theta_hat) = E[(theta_hat - theta)^2] =
Var(theta_hat) + Bias(theta_hat)^2` — the variance-plus-squared-bias decomposition. Least
squares is the zero-bias, all-variance point of this decomposition.

## Baselines

**Ordinary least squares (OLS) / maximum likelihood.** Pick `beta` minimizing the residual sum
of squares; closed form `beta_hat = (X'X)^{-1} X' y`. Unbiased, BLUE by Gauss-Markov, MLE under
Gaussian noise. Requires `X'X` invertible.

**Variable subset selection.** Drop some covariates entirely (forward / backward / best-subset
selection) so the remaining columns are better conditioned and the model is more interpretable.
A covariate is either kept or dropped — a discrete, all-or-nothing operation on the column set.

**Principal-component regression.** Project the covariates onto the top principal components of
`X` (the high-singular-value directions) and regress on those, discarding the low-singular-value
directions. A component is kept whole or dropped whole, selected by its variance in `X`.

**Generalized / pseudo-inverse solution.** In the rank-deficient case, select the minimum-norm
least-squares solution `(X'X)^+ X' y` via the Moore-Penrose inverse, which returns a unique
answer when `X'X` is singular. It is a least-squares fit along the estimable directions.

## Evaluation settings

The natural yardsticks are standard multiple-regression datasets where collinearity is real
and where prediction on held-out data is the target:

- **Multiple linear regression on correlated-predictor data** — e.g. the diabetes dataset
  (`n = 442` patients, `p = 10` standardized baseline clinical measurements: age, sex, BMI,
  blood pressure, six serum measurements), predicting a quantitative disease-progression
  score. Several predictors are correlated. Predictors are standardized to zero mean and unit
  sample variance and the response is centered, so the coefficient scale is comparable and the
  intercept is handled separately.
- **High-dimensional / `p > n` regression** — e.g. gene-expression designs where the number
  of probes exceeds the number of samples, so `X'X` is singular; the estimator must still
  produce a fit.
- **Supervised prediction from fixed feature vectors** — a frozen, pre-computed feature
  representation per example (a dense embedding) with a continuous target; the model is a
  prediction head mapping the feature vector to a scalar, trained on a training split and
  scored on held-out examples.
- Protocol and metrics: hold-out or `k`-fold cross-validation; out-of-sample prediction error
  (mean squared error) or rank correlation between predicted and true targets; for studying
  the estimator's behavior, coefficient stability and standard errors under candidate fits.

## Code framework

The estimator plugs into the standard supervised linear-model harness that already exists:
data arrive as a feature matrix and a target vector (centered / standardized as usual), there
is a linear prediction function `x -> x' w`, a squared-error training objective, and a `fit`
routine that solves for the weights. Two interchangeable substrates are in common use and the
method must drop into either: a **closed-form** solver that forms the normal-equations system
and solves it, and a **gradient-descent** solver that trains a single linear layer with an
optimizer on the squared-error loss. The scaffold below is the bare least-squares machinery
with one empty hook for the estimator design.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- closed-form least-squares substrate ----
class LinearModel:
    """Linear predictor x -> x' w fit by a squared-error objective on (X, y).
    X is assumed centered/standardized; the intercept (if any) is handled outside w."""

    def __init__(self, n_features):
        self.w = np.zeros(n_features)

    def fit(self, X, y):
        # Least-squares normal equations: (X'X) w = X'y.
        A = X.T @ X
        b = X.T @ y
        # TODO: fill in the estimation rule that keeps w well-defined and
        #       well-behaved even when A is ill-conditioned or singular.
        #       For now, the bare least-squares solve:
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        return X @ self.w


# ---- gradient-descent substrate (single linear head) ----
class PredictionHead(nn.Module):
    """A single linear layer mapping a fixed feature vector to a scalar prediction."""

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)   # [B]


def train(model, loader, optimizer):
    model.train()
    for features, target in loader:
        optimizer.zero_grad()
        pred = model(features)
        loss = F.mse_loss(pred, target)
        # TODO: fill in the training-side version of the estimator design.
        loss.backward()
        optimizer.step()
```

The single empty slot in both substrates is the estimator design that determines how the fit
is computed; the final code fills exactly that hook.
