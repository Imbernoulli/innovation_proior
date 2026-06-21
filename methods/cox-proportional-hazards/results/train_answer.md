Time-to-event data are different from ordinary regression data because not every subject reaches the event during observation. For some subjects we see the exact event time; for others we only know that the event has not occurred up to a censoring time. Treating censored times as events would understate survival, while discarding them would waste the information that those subjects were at risk until censoring and would bias the comparison. The goal is to estimate how covariates change the instantaneous event rate while using both event and censored records correctly.

Earlier approaches each leave something important undone. Kaplan-Meier estimation handles censoring elegantly, but it describes one group at a time rather than providing a joint regression coefficient for many covariates. The log-rank test compares groups through risk-set contrasts, yet it remains a test rather than an estimation framework for continuous predictors and multi-variable adjustment. Parametric models such as exponential or Weibull regression do provide coefficients, but they require a fully specified baseline time distribution, and covariate estimates can be distorted when that distribution is wrong. What is needed is a regression model that adjusts for many covariates, respects censoring, and does not commit to a rigid parametric shape for the baseline event rate over time.

The method is Cox proportional hazards regression. It assumes that each subject's hazard is the product of a common baseline hazard, shared by everyone, and a covariate-driven multiplier. Formally, the hazard for a subject with covariates x is h(t | x) = h_0(t) exp(x^T beta). The baseline hazard h_0(t) is left completely unspecified; it can rise, fall, or change shape for reasons the model does not need to explain. Because the baseline factor is common to every subject at risk at the same time, it cancels when we compare hazards, and the ratio of two hazards depends only on the covariates and beta. The exponentiated coefficients are therefore interpretable as hazard ratios between subjects who are still event-free at the same time.

Estimation proceeds through the partial likelihood. At each observed event time, consider the risk set of all subjects whose follow-up time is at least that long. Condition on the fact that one member of that risk set failed at that time. The probability that the failing subject is the observed one equals her hazard divided by the sum of hazards over the whole risk set. The shared baseline factor and the infinitesimal time interval cancel, leaving only exp(x_i^T beta) divided by the sum of exp(x_j^T beta) over the risk set. Multiplying these conditional probabilities across all event times gives a likelihood for beta alone, with the infinite-dimensional baseline hazard conditioned away. The log partial likelihood is the sum over events of x_i^T beta minus the log of the risk-set weighted sum of exponentials. Its score compares each failing subject's covariates against the risk-score-weighted average covariates of everyone at risk at that time, and its Hessian is the accumulated weighted covariance matrix over those risk sets. Censored subjects enter naturally: they remain in every risk set before their censoring time and drop out afterward, never pretending to fail. Tied event times are usually handled with the Breslow or Efron approximation, both of which preserve the same risk-set logic. After beta is estimated, the baseline cumulative hazard can be recovered afterward, if prediction is needed, by dividing observed event counts at each time by the corresponding sum of risk scores.

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
