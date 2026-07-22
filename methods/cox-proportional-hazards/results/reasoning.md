I need a regression model for survival times, but the data refuse to behave like ordinary outcomes. If a subject has an event at time 8, I know the event time. If another subject is censored at time 8, I only know that the event time exceeds 8. Treating both as events is wrong. Dropping the censored subject is also wrong, because that subject was genuinely at risk up to time 8 and therefore helps define who could have failed before then. So the unit of information is not just an observed time; it is the ordering of event times together with the risk sets that exist at those times.

The target I care about is a covariate comparison. Suppose two patients are both alive just before time `t`, one with covariates `x` and one with covariates `x'`. I want the model to say how their instantaneous event rates compare. But I do not want to write down the whole time pattern of the underlying event rate. In medical follow-up, reliability, and epidemiology, the baseline hazard can have any shape: high early, low later, increasing with age of the device, affected by follow-up practice. If I force it to be exponential or Weibull just to get a likelihood, I may be estimating a covariate effect through the wrong clock.

Let me try the usual full hazard model and see where it gets stuck. If subject `i` has covariates `x_i`, the density contribution for an uncensored event at `t_i` has a hazard term times a survival term, and a censored observation contributes just the survival term. A full likelihood wants something like `h(t_i | x_i)` and `H(t_i | x_i)`. If I specify every piece parametrically, the likelihood is easy, but the cost is exactly the thing I do not trust: the baseline time shape. If I leave the baseline completely free in that full likelihood, I have an infinite-dimensional nuisance object sitting beside the finite coefficient vector I actually want.

The comparison I want suggests a simpler structure. Let the covariates multiply the instantaneous rate, so

`h(t | x) = h_0(t) exp(x^T beta)`.

This says the absolute hazard may be anything through `h_0(t)`, but the ratio of two hazards at the same time is

`h(t | x) / h(t | x') = exp((x - x')^T beta)`.

That is the right estimand: the coefficient does not claim to know the baseline event curve; it claims to know how covariates tilt that curve. The obstruction remains: if `h_0(t)` is arbitrary, how do I estimate `beta` without estimating all of `h_0` first?

Look at one event time instead of the whole likelihood. At time `t_i`, suppose subject `i` is the one who fails, and let `R_i` be the set of subjects still under observation and event-free just before `t_i`. In a tiny interval around `t_i`, subject `j` in the risk set has event probability approximately

`h_0(t_i) exp(x_j^T beta) dt`.

Now condition on the fact that exactly one member of the risk set fails at that time. The conditional probability that the failing subject is `i` is proportional to subject `i`'s instantaneous hazard divided by the sum of all instantaneous hazards in the risk set:

`h_0(t_i) exp(x_i^T beta) dt / sum_{j in R_i} h_0(t_i) exp(x_j^T beta) dt`.

The nuisance pieces vanish. The common `h_0(t_i)` cancels, and so does `dt`. What remains is

`exp(x_i^T beta) / sum_{j in R_i} exp(x_j^T beta)`.

That is the escape route. I do not need to model the baseline hazard to learn the relative hazards. At each observed event time, the data tell me which member of the current risk set failed; the covariates tell me how likely each member was, relative to the others, to be that failure. Multiplying these conditional probabilities over observed events gives an objective for `beta` alone:

`L(beta) = product_{i: event_i = 1} exp(x_i^T beta) / sum_{j in R_i} exp(x_j^T beta)`.

This is not the full likelihood for every unknown in the survival process. It is the likelihood for the ordering identities of the failures, conditional on the risk sets and event times. That is enough for the regression coefficients because the common baseline intensity at each event time has been conditioned away.

Take logs:

`ell(beta) = sum_{i: event_i = 1} [x_i^T beta - log sum_{j in R_i} exp(x_j^T beta)]`.

The score has the form I would hope for:

`U(beta) = sum_{i: event_i = 1} [x_i - xbar_R_i(beta)]`,

where

`xbar_R_i(beta) = sum_{j in R_i} x_j exp(x_j^T beta) / sum_{j in R_i} exp(x_j^T beta)`.

So each event compares the covariates of the subject who failed against the risk-score-weighted average covariates of everyone who could have failed then. If high-risk covariate patterns fail more often than their risk-set availability would predict, the coefficient moves in that direction. Censored subjects enter exactly as they should: they remain in risk sets before their censoring time and disappear afterward. They never pretend to have failed.

The curvature is also clean. Differentiating the score gives the negative of a weighted covariance matrix over each event's risk set:

`dU/dbeta = - sum_{i: event_i = 1} Var_beta(X | R_i)`.

So maximizing the log objective is a Newton problem: add observed covariates, subtract risk-set expected covariates, and use accumulated risk-set covariance as the information matrix. The coefficient estimates become log hazard ratios. Exponentiating a coefficient gives the multiplicative change in hazard for a one-unit covariate change, with all comparisons made among subjects at risk at the same event time.

Ties need a convention because the clean conditional argument assumes one failure at a time. If several subjects fail at the same recorded time, I can approximate the denominator as if the tied failures occur against the same risk set, which gives the Breslow form, or remove fractions of the tied risk from the denominator as the tied failures are allocated, which gives the Efron refinement. The central idea is unchanged: compare failures with the subjects at risk at that time and do not specify the baseline hazard while estimating `beta`.

Once `beta` is estimated, the baseline can come back as a secondary object. At an event time `t`, the observed number of events divided by `sum_{j in R(t)} exp(x_j^T beta_hat)` estimates the jump in the baseline cumulative hazard. Accumulating those jumps gives a baseline survival estimate. That order matters: the regression coefficients are obtained without choosing a parametric baseline; the baseline summary is reconstructed afterward if prediction needs it.

The code I want is therefore just the risk-set conditional log likelihood, its gradient, and its Hessian. Minimizing the negative log objective gives the same coefficient vector as maximizing the partial likelihood.

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

    neg_loglik = -loglik
    grad = -score
    hess = observed_info
    return neg_loglik, grad, hess


def fit_cox_ph(time, event, X, max_iter=50, tol=1e-8, ridge=1e-9):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    X = np.asarray(X, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(max_iter):
        value, grad, hess = cox_partial_objective(beta, time, event, X)
        step = np.linalg.solve(hess + ridge * np.eye(X.shape[1]), grad)
        beta_next = beta - step
        if np.linalg.norm(step) < tol:
            beta = beta_next
            break
        beta = beta_next

    return beta


def hazard_ratio(beta, x_a, x_b):
    return float(np.exp((np.asarray(x_a) - np.asarray(x_b)) @ beta))
```
