## Research question

I have a linear model `y = X beta + eps`, with an `n x p` design matrix `X`, an unknown
coefficient vector `beta in R^p`, and noise `eps` with `E[eps] = 0`, `Cov(eps) = sigma^2 I`.
I want to estimate `beta` (or, equivalently, build the linear predictor `x -> x' beta`) from
the data. The default tool is ordinary least squares, and on well-behaved designs it is
excellent. The problem appears the moment the columns of `X` are *nonorthogonal* — strongly
correlated, nearly linearly dependent, or outright more numerous than the observations. In
that regime the least-squares estimate becomes wildly unstable: individual coefficients swing
to huge magnitudes, flip sign on a tiny perturbation of the data, and have enormous
estimation variance, even when the *fit* `X beta` is fine. In the extreme `p > n` case the
estimate is not even defined. The goal is an estimator that stays well-defined and stable
across the whole range from orthogonal to super-collinear designs, that controls this
variance, and — ideally — that can be shown to estimate `beta` *better* (in total mean
squared error) than least squares does, not just more cheaply. What "better" can mean here,
and whether such an estimator can exist, is the open question; least squares already occupies
the throne of unbiased linear estimators.

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
crowns it: among all *linear unbiased* estimators of `beta`, least squares has the smallest
variance (it is BLUE — the best linear unbiased estimator). This theorem is the prevailing
wisdom and the thing any challenger must reckon with: it appears to say least squares cannot
be beaten.

**The diagnostic failure mode: small eigenvalues of `X'X`.** The optimality is about
unbiasedness; it says nothing about absolute magnitude of error when `X'X` is ill-conditioned.
Write the eigendecomposition `X'X = sum_j d_j^2 v_j v_j'` (the `d_j` are the singular values of
`X`, the `v_j` the right singular vectors). Then

```
Cov(beta_hat) = sigma^2 (X'X)^{-1} = sigma^2 sum_j d_j^{-2} v_j v_j'.
```

A direction `v_j` with a small singular value `d_j` contributes variance proportional to
`1/d_j^2`, which explodes as `d_j -> 0`. Collinearity *is* the presence of such small
singular values: when two covariates are nearly proportional, the design cannot separate
their contributions, and the estimate along that direction is dominated by noise. The
observable symptoms — coefficients with implausible magnitudes, unstable signs, large
standard errors — are diagnostic facts about least squares on real correlated data, not a
defect to be discovered later. In the limit of exact linear dependence, or `p > n`, some
`d_j = 0`, `X'X` is singular, its inverse does not exist, and the normal equations
`X'X beta = X'y` have a whole affine space of solutions — least squares ceases to pin down a
unique `beta` at all.

**Ill-posed inverse problems and stabilization.** Independently of statistics, the theory of
*ill-posed* inverse problems studied exactly this pathology in the abstract: solving `A f = g`
for `f` when `A` has tiny singular values amplifies noise without bound. Tikhonov (1963,
"Solution of incorrectly formulated problems and the regularization method") showed that an
unstable inverse problem can be made well behaved by trading raw data fidelity against a
controlled size / smoothness requirement on the solution. This establishes a useful principle:
when an inverse amplifies noise, accepting a controlled distortion of the solution can produce
a more stable estimate. It does not, by itself, settle whether such a biased move can beat
least squares under regression mean-squared error.

**Shrinkage and the inadmissibility of the unbiased estimator.** A second, decision-theoretic
thread questions whether unbiasedness is even worth its price. Stein (1956) and James & Stein
(1961) showed that for estimating the mean of a `p`-dimensional Gaussian with `p >= 3`, the
obvious unbiased estimator (the sample value itself) is **inadmissible** under summed squared-
error loss: the explicitly biased shrinkage estimator
`(1 - (p-2) sigma^2 / ||x||^2) x`, which pulls the estimate toward the origin, has strictly
lower total mean squared error *everywhere*. The reason it can dodge the Gauss-Markov verdict
is precisely that it is biased — Gauss-Markov only protects the unbiased class. This is the
load-bearing background insight that deliberately trading a little bias for a large reduction
in variance can lower total error in several dimensions at once.

**Mean squared error as the figure of merit.** The natural scalar quality measure for an
estimator `theta_hat` of `theta` is `MSE(theta_hat) = E[(theta_hat - theta)^2] =
Var(theta_hat) + Bias(theta_hat)^2` — the variance-plus-squared-bias decomposition. Least
squares is the all-variance, zero-bias corner of this tradeoff. Whether some point with a
little bias and much less variance achieves a smaller *sum* is the quantity at stake.

## Baselines

**Ordinary least squares (OLS) / maximum likelihood.** Core idea: pick `beta` minimizing the
residual sum of squares; closed form `beta_hat = (X'X)^{-1} X' y`. Unbiased, BLUE by
Gauss-Markov, MLE under Gaussian noise. **Limitation:** requires `X'X` invertible, so it is
undefined when `p > n` or columns are exactly dependent; and when `X'X` is merely
ill-conditioned, the inverse magnifies noise along low-singular-value directions, so the
estimate has huge variance, erratic signs, and implausible magnitudes on nonorthogonal
designs. It pays for zero bias with unbounded variance exactly where the data is least
informative.

**Variable subset selection.** Core idea: drop some covariates entirely (forward / backward /
best-subset selection) so the remaining columns are better conditioned and the model is more
interpretable. **Limitation:** it is a discrete, all-or-nothing operation — a covariate is in
or out — so the chosen model is a high-variance function of the data (small data changes flip
which variables are kept), and discarding a collinear partner throws away the information it
shared rather than balancing the two. It does not smoothly control the conditioning of the
estimate.

**Principal-component regression.** Core idea: project the covariates onto the top principal
components of `X` (the high-singular-value directions) and regress on those, discarding the
low-singular-value directions that caused the instability. **Limitation:** it is again a hard
cut — a component is kept whole or dropped whole — and it discards directions purely by their
variance in `X`, with no reference to how much they help predict `y`; a low-variance direction
that is genuinely predictive is thrown out with the noise.

**Generalized / pseudo-inverse solution.** Core idea: in the rank-deficient case, select the
minimum-norm least-squares solution `(X'X)^+ X' y` via the Moore-Penrose inverse, restoring a
unique answer. **Limitation:** it makes the estimate *defined* but not *stable* — it is still
a pure least-squares fit along the estimable directions, with no damping of the noise
amplification along the small-singular-value directions; it resolves nonuniqueness, not high
variance.

## Evaluation settings

The natural yardsticks are standard multiple-regression datasets where collinearity is real
and where prediction on held-out data is the target:

- **Multiple linear regression on correlated-predictor data** — e.g. the diabetes dataset
  (`n = 442` patients, `p = 10` standardized baseline clinical measurements: age, sex, BMI,
  blood pressure, six serum measurements), predicting a quantitative disease-progression
  score. Several predictors are correlated, so OLS coefficients are unstable. Predictors are
  standardized to zero mean and unit sample variance and the response is centered, so the
  coefficient scale is comparable and the intercept is handled separately.
- **High-dimensional / `p > n` regression** — e.g. gene-expression designs where the number
  of probes exceeds the number of samples, so `X'X` is singular and OLS is undefined; the
  estimator must still produce a fit.
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
optimizer on the squared-error loss. Nothing about *how the fit is stabilized* is settled, so
the scaffold below is just the bare least-squares machinery with one empty hook for the
estimator design.

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
        self.w = np.linalg.solve(A, b)   # fails / is unstable when A is (near-)singular
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

The single empty slot in both substrates is the estimator design that will make the fit
well-defined and stable; the final code fills exactly that hook.
