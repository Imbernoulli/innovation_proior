# Causal Forest (Generalized Random Forests)

A causal forest estimates the heterogeneous treatment effect
`tau(x) = E[Y(1) - Y(0) | X = x]` from observational data under unconfoundedness. It is the
treatment-effect instance of generalized random forests (GRF): read a random forest not as an
average of trees but as an **adaptive kernel** that supplies similarity weights `alpha_i(x)`,
grow the trees with a splitting rule **targeted at heterogeneity in `tau`**, make the trees
**honest** so the forest score noise is mean-zero, and **orthogonalize** (residualize `Y` and
`W` on `X`) so the estimate is robust to confounding. The whole construction admits a central
limit theorem, so it emits asymptotic confidence intervals.

## Problem it solves

CATE estimation from i.i.d. observational `(X_i, W_i, Y_i)`, `W_i in {0,1}`, under
unconfoundedness `{Y(0), Y(1)} _|_ W | X`. Wanted: nonparametric flexibility in high dimension,
adaptive sensitivity to *which* covariates drive the effect, robustness to confounding
(`e(x) = P(W=1|X=x)` depends on `x`), and a valid sampling distribution — since one potential
outcome per unit is never observed, there is no test set, so inference must come from theory.

## Key ideas

1. **Forest as adaptive weights, not an average.** A tree gives weight `1/|L_b(x)|` to each
   training point in the same leaf as `x`; averaging over trees gives
   `alpha_i(x) = (1/B) sum_b 1{X_i in L_b(x)} / |L_b(x)|`, `sum_i alpha_i(x) = 1`. Estimate any
   moment-identified `theta(x)` (here `tau(x)`) by solving **one** weighted estimating equation
   `sum_i alpha_i(x) psi_{theta,nu}(O_i) = 0` rather than averaging per-tree solves — averaging
   would not remove the finite-sample bias of nonlinear leaf solves; pooling weights does.

2. **Heterogeneity splitting via a gradient pseudo-outcome.** The ideal split minimizes child
   MSE `err(C_1,C_2)`. The bias-variance telescoping gives
   `err(C_1,C_2) = K(P) - E[Delta(C_1,C_2)] + o(r^2)`, with
   `Delta(C_1,C_2) = (n_{C_1} n_{C_2}/n_P^2)(theta_hat_{C_1} - theta_hat_{C_2})^2`, the squared
   discrepancy of child estimates, so maximizing `Delta` minimizes the ideal split error to this
   order. Exact `Delta` is too costly, so take one Newton step from the
   parent: pseudo-outcome `rho_i = -xi^T A_P^{-1} psi_{theta_hat_P, nu_hat_P}(O_i)` with
   `A_P = (1/n_P) sum_{i in P} nabla psi`, then run a standard CART regression scan maximizing
   `tDelta(C_1,C_2) = sum_j (1/|C_j|)(sum_{i in C_j} rho_i)^2`. Labeling step (once per node) +
   universal regression step (single pass via cumulative sums). Least squares recovers Breiman.

3. **Honesty.** Each tree splits its subsample into a split-half (chooses splits) and a disjoint
   estimate-half (populates leaves / solves the equation), so `alpha_i(x) _|_ O_i | X_i`.
   Then `Psi - barPsi = sum_i alpha_i(x)(psi_i - M(X_i))` is mean-zero, and localization makes
   `barPsi` shrink at the target point, giving centered intervals.

4. **Orthogonalization / local centering.** Before the forest, residualize out-of-fold:
   `Ytilde_i = Y_i - m_hat^{(-i)}(X_i)`, `Wtilde_i = W_i - e_hat^{(-i)}(X_i)`, with
   `m(x)=E[Y|X=x]`, `e(x)=E[W|X=x]`. By Robinson's partialling-out, the residual-on-residual
   moment identifies `tau` and is Neyman-orthogonal — first-order insensitive to nuisance errors
   — so the estimate stays robust to confounding even without sharp neighborhoods. Cross-fitting
   the nuisances (double machine learning) is what makes this first-order insensitivity bite.

## Treatment-effect instantiation (CAPE / causal forest)

Model `Y_i = W_i b_i + eps_i`, `tau(x) = E[b_i | X_i = x]`. Score
`psi_{beta(x), c(x)}(Y_i, W_i) = (Y_i - beta(x) W_i - c(x))(1, W_i^T)^T`; closed form with forest
weights:

```
tau_hat(x) = ( sum_i alpha_i(x) (W_i - Wbar_a)^2 )^{-1}
             sum_i alpha_i(x) (W_i - Wbar_a)(Y_i - Ybar_a),
             Wbar_a = sum_i alpha_i(x) W_i,  Ybar_a = sum_i alpha_i(x) Y_i.
```

Gradient pseudo-outcome (matches the `grf` relabeling with instrument = treatment):

```
A_P  = (1/n_P) sum_{i in P} (W_i - Wbar_P)^{otimes 2}
rho_i = xi^T A_P^{-1} (W_i - Wbar_P)( Y_i - Ybar_P - (W_i - Wbar_P) beta_hat_P )
```

with `beta_hat_P` the parent OLS slope of `Y` on `W`; in the scalar binary-treatment case,
`xi = 1` and these reduce to scalar products.

## Algorithm

```
GeneralizedRandomForest(examples S, test point x):
  (optional) local centering: replace (Y_i, W_i) by out-of-fold residuals (Ytilde_i, Wtilde_i)
  alpha <- zeros(|S|)
  for b = 1..B:
      I  <- Subsample(S, s)                 # without replacement, s/n -> 0
      J1, J2 <- SplitSample(I)              # honesty: split-half / estimate-half
      tree <- GradientTree(J1)              # labeling (rho_i) + CART regression split
      N <- neighbors of x in tree, drawn from J2
      for e in N: alpha[e] += 1/|N|
  return tau_hat(x) solving  sum_i (alpha_i/B) psi_{tau,nu}(O_i) = 0

GradientTree(J):
  queue <- {root over J}
  while node P <- pop(queue):
      (theta_hat_P, nu_hat_P, A_P) <- SolveEstimatingEquation(P)
      rho_i <- -xi^T A_P^{-1} psi_{theta_hat_P, nu_hat_P}(O_i)        # labeling step
      split <- CART split on (X_i, rho_i) maximizing tDelta          # regression step
      if split ok: add children to queue
```

Implementation defaults differ by library. In R `grf`: `B = 2000` trees, `min.node.size = 5`,
`sample.fraction = 0.5`, `honesty.fraction = 0.5`, `mtry = min(ceil(sqrt(p)) + 20, p)`,
and `ci.group.size = 2`. In `econml.CausalForestDML`: defaults include `n_estimators = 100`,
`criterion = "mse"`, `cv = 2`, `max_features = "auto"`, `max_samples = .45`,
`min_balancedness_tol = .45`, `honest = True`, `subforest_size = 4`, and `inference = True`;
for binary treatment, set `discrete_treatment=True`.

## Inference (CLT + bootstrap of little bags)

`tau_hat(x)` is one weighted moment solution, not a tree average, so linearize to the
pseudo-forest `ttheta*(x) = tau(x) + sum_i alpha_i(x) rho_i*(x)`,
`rho_i*(x) = -xi^T V(x)^{-1} psi_{tau(x),nu(x)}(O_i)` — exactly the output of an (infeasible)
regression forest on outcomes `tau(x) + rho_i*(x)`, hence a U-statistic. With `s = n^beta` and
`beta_min = 1 - (1 + pi^{-1} log(omega^{-1})/log((1 - omega)^{-1}))^{-1} < beta < 1`,
the coupling is

```
sqrt(n/s)(ttheta*(x) - tau_hat(x))
  = o_P(max{s^{-pi/2 * log((1 - omega)^{-1})/log(omega^{-1})}, (s/n)^{1/6}})
  = o_P(1).
```

It transfers Gaussianity:
`(tau_hat_n(x) - tau(x))/sigma_n(x) => N(0,1)`. Estimate
`sigma_hat_n^2(x) = xi^T Vhat^{-1} Hhat Vhat^{-T} xi` with `H_n = Var[sum_i alpha_i psi]` from a
bootstrap of little bags: grow trees in groups of `ell` sharing a half-sample, then recover the
half-sampling variance by subtracting the `(ell - 1)^{-1}` within-bag Monte Carlo term from the
variance of little-bag means. This gives nominal `1 - alpha` intervals
`tau_hat(x) +/- z_{1-alpha/2} sigma_hat_n(x)`.

## No-forest fallback (R-loss)

Where a forest engine is unavailable, the same orthogonalized moment gives the R-learner.
Minimizing the R-loss `mean_i [(Ytilde_i) - (Wtilde_i) tau(X_i)]^2` equals weighted least squares
of pseudo-outcome `Ytilde_i / Wtilde_i` with weight `Wtilde_i^2` (since
`[(Ytilde) - (Wtilde)tau]^2 = (Wtilde)^2 [Ytilde/Wtilde - tau]^2`), the weight downweighting
near-deterministic-treatment points where the pseudo-outcome is uninformative.

## Working code

```python
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.model_selection import KFold


class CausalForest:
    """CATE tau(x) under unconfoundedness via DML residualization (local centering /
    orthogonalization) + a generalized-random-forest causal forest on residuals;
    R-loss weighted-regression fallback when an econml forest engine is absent."""

    def __init__(self, n_folds=3, seed=42):
        self.n_folds, self.seed, self.use_forest = n_folds, seed, True
        try:
            from econml.dml import CausalForestDML
            self._cf = CausalForestDML(
                model_y=GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    min_samples_leaf=20, random_state=seed),       # m_hat = E[Y|X]
                model_t=GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    min_samples_leaf=20, random_state=seed + 1),   # e_hat = E[W|X]
                discrete_treatment=True,                           # binary W
                n_estimators=500, min_samples_leaf=5, max_depth=None,
                max_samples=.45, honest=True, inference=True,
                subforest_size=4, random_state=seed + 2, cv=3)
        except ImportError:
            self.use_forest = False
            self._model_y = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=seed)
            self._model_w = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=seed + 1)
            self._cate = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5,
                max_features="sqrt", random_state=seed + 2)

    def fit(self, X, W, Y):
        if self.use_forest:
            self._cf.fit(Y, W, X=X)               # residualize + grow causal forest
            return self
        # manual DML: out-of-fold residuals -> honest, Neyman-orthogonal
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        Y_res = np.zeros_like(Y, dtype=float)
        W_res = np.zeros_like(W, dtype=float)
        for tr, va in kf.split(X):
            my = clone(self._model_y).fit(X[tr], Y[tr])
            mw = clone(self._model_w).fit(X[tr], W[tr])
            Y_res[va] = Y[va] - my.predict(X[va])
            W_res[va] = W[va] - mw.predict_proba(X[va])[:, 1]
        # R-loss -> weighted regression: pseudo = Ytilde/Wtilde, weight = Wtilde^2
        eps = 0.01
        safe_W = np.where(np.abs(W_res) > eps, W_res,
                          eps * np.where(W_res >= 0, 1.0, -1.0))
        pseudo = Y_res / safe_W
        weights = W_res ** 2
        q = np.percentile(np.abs(pseudo), 95)
        pseudo = np.clip(pseudo, -q, q)
        self._cate.fit(X, pseudo, sample_weight=weights)
        return self

    def predict(self, X):
        return self._cf.effect(X).flatten() if self.use_forest else self._cate.predict(X)
```
