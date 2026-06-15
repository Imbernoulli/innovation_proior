# MICE (Multivariate Imputation by Chained Equations), distilled

MICE — also called Fully Conditional Specification (FCS) — fills missing values in
a multivariate table by modeling each incomplete variable as a regression on all
the others, and cycling through the variables in a round-robin, iterating until the
imputations settle. It sidesteps the need for a single joint distribution over
heterogeneous, mixed-type data: you only ever specify a sensible *conditional*
model per column. Recognized abstractly, the round-robin is a Gibbs sampler over
the per-column conditionals, which (because each column's imputation re-enters only
indirectly through the others) mixes in a handful of sweeps.

## Problem it solves

Impute a table with `NaN`s scattered across several interdependent columns, so the
completed table both borrows strength across correlated variables and (for valid
inference) carries the right uncertainty. The obstacle is that the natural
predictors of any incomplete column are themselves incomplete, and the dependence
is circular — so there is no complete column to regress on.

## Key idea

1. **Conditional, not joint.** Specify `P(Y_j | Y_{-j}, theta_j)` per column (linear
   for continuous, logistic for binary, etc.) instead of a joint `P(Y | theta)`.
   This handles mixed types, bounds, and nonlinearities that no convenient joint
   captures. Price: the conditionals may be mutually *incompatible* (no joint has
   all of them as conditionals); contained in practice by a sensible visit order.
2. **Chained iteration breaks the circularity.** Initialize every column's holes
   crudely (random observed draw, or column mean). Then sweep the columns: for each
   `Y_j`, refit its conditional model on the rows where `Y_j` is observed (using the
   *current* fills of the other columns as predictors) and re-impute `Y_j`'s holes.
   Repeat for a few sweeps. Each column is re-imputed given everyone else's latest
   state, so cross-variable dependence propagates around the cycle.
3. **It is a Gibbs sampler.** If the conditionals were a real joint's conditionals,
   the sweep draws from the correct posterior predictive of the missing data. This
   says each *proper* step should draw parameters then data, and explains the fast
   mixing: `Y_j`'s past imputation affects its next one only through the other
   columns, not directly — so ~5-20 iterations suffice, no long burn-in.

## Per-column proper draw (Bayesian linear regression, the `norm` step)

For a continuous column `y` (observed part, `n1` rows) with predictor matrix `X`
(`q` columns):

```
S         = X_obs' X_obs
V         = (S + kappa * diag(S))^-1            # ridge for stability, kappa = 0.0001
beta_hat  = V X_obs' y_obs
g         ~ chi^2_{n1 - q}                      # draw residual d.o.f.
sigma_dot^2 = (y_obs - X_obs beta_hat)'(y_obs - X_obs beta_hat) / g   # noise draw
beta_dot  = beta_hat + sigma_dot * chol(V) * z1,   z1 ~ N(0, I_q)     # coef draw
y_imp     = X_mis beta_dot + sigma_dot * z2,       z2 ~ N(0, I_{n0})  # + residual
```

Three random pieces, each load-bearing for valid multiple imputation: the
chi-squared draw (noise level unknown), the normal `beta_dot` (coefficients
unknown — *parameter* uncertainty), the additive `sigma_dot * z2` (residual
scatter). Dropping the coefficient draw (stochastic regression with fixed
`beta_hat`) under-covers (~0.908 interval coverage); dropping the noise too (mean /
deterministic regression) sets the between-imputation variance to zero.

## Predictive mean matching (the robust `pmm` step, Little 1988)

Compute `beta_hat` (point) and `beta_dot` (posterior draw). For each missing case,
predicted mean `eta_mis = X_mis beta_dot`; for each observed case, `eta_obs =
X_obs beta_hat`. Form a donor pool of the `d` observed cases (typically `d = 5`)
with `eta_obs` closest to `eta_mis`, pick one at random, and impute its actually
observed `y`. Because every imputed value is a real observed value, it respects
bounds, skew, discreteness, and heteroscedasticity automatically, and is robust to
the linear model being misspecified — at the cost of not extrapolating beyond the
observed range.

## Multiple imputation and pooling (the inference end, Rubin 1987)

Produce `m` completed tables, analyze each, and pool a scalar estimand `Q`:

```
Qbar = (1/m) sum_h Q_hat_h
Ubar = (1/m) sum_h U_h                       # within-imputation variance
B    = (1/(m-1)) sum_h (Q_hat_h - Qbar)^2    # between-imputation variance
T    = Ubar + (1 + 1/m) B                    # total variance
```

`B` is the extra variance from having missing data; it is nonzero only because the
per-column draws inject parameter and residual randomness. The MAR + distinct-
parameters condition (Rubin 1976) is what makes imputing from observed-data
conditionals valid in the first place.

## Point variant for reconstruction / downstream prediction

When the objective is squared-error reconstruction and downstream model accuracy
(not calibrated inference), drop the randomness: at each column impute the
*conditional mean* (which minimizes expected squared error), keeping only the
chained round-robin that handles the multivariate structure. The per-column model
is a self-regularizing Bayesian linear regression whose ridge penalty
`lambda/alpha` is set by evidence maximization (MacKay 1992). On centered data when
an intercept is fitted, posterior-mean coefficients solve
`coef = (lambda/alpha * I + X'X)^-1 X' y`, with `alpha` (noise precision) and
`lambda` (weight precision) iterated via
`gamma = sum_i (alpha d_i)/(lambda + alpha d_i)`. In sklearn's implementation the
updates include tiny Gamma hyperpriors:
`lambda = (gamma + 2*lambda_1)/(sum(coef^2) + 2*lambda_2)` and
`alpha = (n - gamma + 2*alpha_1)/(sse + 2*alpha_2)`, with all four hyperprior
constants defaulting to `1e-6`. So each heterogeneous column regularizes itself
with no manual tuning, while the intercept remains an ordinary centered-data
offset rather than a random coefficient.

## Design choices and why

- **Conditional specification** over joint modeling: no tractable joint exists for
  mixed-type / bounded / nonlinear tabular data; per-column models are easy to pick.
- **Round-robin iteration**: dissolves the circular dependence among incomplete
  predictors; nobody is a fixed complete predictor.
- **Few iterations (5-20)**: fast mixing — a column's imputation re-enters only
  through the other variables.
- **Mean / random-observed initialization**: any complete starting table; the first
  sweep overwrites it.
- **Self-tuning Bayesian ridge per column**: stable on collinear / small-`n`
  columns; no penalty to hand-pick.
- **Ascending visit order (least-missing first)**: the most reliably estimated
  columns are imputed first and serve as better predictors for the harder ones; it
  also stabilizes the otherwise order-dependent incompatible-conditionals case.
- **Early stop on the infinity-norm change**: cheap fixpoint test; chasing exact
  convergence rarely improves the downstream score.

## Working code

The point variant, wrapping the standard iterative-imputer machinery, filling the
`fit` / `transform` slots of the transformer:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """MICE / chained-equations imputation (point variant).

    Each incomplete column is regressed on all the others in a round-robin,
    iterated to a fixpoint. The per-column engine is a self-regularizing
    Bayesian linear regression (evidence-based ridge), and we impute the
    conditional mean -- the squared-error-optimal point fill.
    """

    def __init__(self, random_state=42, max_iter=30):
        self.random_state = random_state
        self.max_iter = max_iter

    def _build(self):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        return IterativeImputer(
            estimator=BayesianRidge(),       # self-tuning ridge per column
            sample_posterior=False,           # point fills: posterior-mean prediction
            max_iter=self.max_iter,          # round-robin sweep cap (early-stops sooner)
            random_state=self.random_state,
            imputation_order="ascending",    # least-missing columns first
            initial_strategy="mean",         # complete starting table = column means
            tol=1e-3,                        # stop when fills stop moving (scaled inf-norm)
        )

    def fit(self, X, y=None):
        self._imputer = self._build()
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        self._imputer = self._build()
        return self._imputer.fit_transform(X)
```

The round-robin, the per-column fit-on-observed / predict-on-missing, the
ascending order, the mean initialization, and the `tol`-based early stop are the
iterative-imputer behaviors used here. With `BayesianRidge` and
`sample_posterior=False`, each fill is the posterior-mean point prediction clipped
to the allowed bounds (default `-inf` to `inf`). For *proper* multiple imputation
feeding Rubin's rules, produce `m` completed tables and use posterior predictive
sampling rather than this deterministic point branch.
