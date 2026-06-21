# Cox Proportional Hazards

## Problem

Given right-censored survival data `(T_i, delta_i, x_i)`, estimate how covariates change the
instantaneous event rate without specifying the underlying baseline hazard over time.

## Model

The hazard for subject `i` is

```text
h_i(t) = h_0(t) exp(x_i^T beta)
```

The baseline hazard `h_0(t)` is left unspecified. For two subjects at the same time,

```text
h(t | x_a) / h(t | x_b) = exp((x_a - x_b)^T beta)
```

so coefficients are log hazard ratios and exponentiated coefficients are hazard ratios.

## Partial Likelihood

At an event time `T_i`, define the risk set

```text
R_i = {j : T_j >= T_i}.
```

Condition on one member of `R_i` failing at that time. Since every subject in the risk set shares
the same baseline factor `h_0(T_i)`, the probability that subject `i` is the one who fails is

```text
exp(x_i^T beta) / sum_{j in R_i} exp(x_j^T beta).
```

Multiplying these terms over observed events gives the partial likelihood

```text
L(beta) = product_{i: delta_i = 1}
          exp(x_i^T beta) / sum_{j in R_i} exp(x_j^T beta).
```

The log partial likelihood is

```text
ell(beta) = sum_{i: delta_i = 1}
            [x_i^T beta - log sum_{j in R_i} exp(x_j^T beta)].
```

Its score is

```text
U(beta) = sum_{i: delta_i = 1} [x_i - xbar_i(beta)],
```

where `xbar_i(beta)` is the risk-score-weighted covariate average over `R_i`. The observed
information is the sum of the weighted covariate covariance matrices over the event risk sets.

## Estimation

Maximize `ell(beta)` by Newton-Raphson. Censored subjects contribute by remaining in risk sets up
to their censoring time and then leaving the risk set. Distinct event times give the clean formula
above; tied event times are usually handled with Breslow or Efron approximations.

After estimating `beta`, a baseline cumulative hazard can be recovered with the Breslow estimator:

```text
dH_0(t) = d(t) / sum_{j in R(t)} exp(x_j^T beta_hat),
```

where `d(t)` is the number of events at time `t`.

## Minimal Implementation

```python
import numpy as np


def cox_partial_objective(beta, time, event, X):
    beta = np.asarray(beta, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    X = np.asarray(X, dtype=float)

    eta = X @ beta
    loglik = 0.0
    score = np.zeros_like(beta)
    observed_info = np.zeros((len(beta), len(beta)))

    for i in np.flatnonzero(event):
        risk = time >= time[i]
        eta_r = eta[risk]
        X_r = X[risk]

        shift = np.max(eta_r)
        w = np.exp(eta_r - shift)
        denom = np.sum(w)
        probs = w / denom

        xbar = probs @ X_r
        centered = X_r - xbar
        cov = (centered.T * probs) @ centered

        loglik += eta[i] - (shift + np.log(denom))
        score += X[i] - xbar
        observed_info += cov

    return -loglik, -score, observed_info


def fit_cox_ph(time, event, X, max_iter=50, tol=1e-8, ridge=1e-9):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    X = np.asarray(X, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(max_iter):
        _, grad, hess = cox_partial_objective(beta, time, event, X)
        step = np.linalg.solve(hess + ridge * np.eye(X.shape[1]), grad)
        beta -= step
        if np.linalg.norm(step) < tol:
            break

    return beta


def hazard_ratios(beta):
    return np.exp(beta)
```
