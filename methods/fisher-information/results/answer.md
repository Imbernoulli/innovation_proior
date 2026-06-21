# Fisher Information

## Definition

For a regular scalar parametric model `p_theta(x)`, define the score

`S_theta(x) = partial_theta log p_theta(x)`.

The Fisher information in one observation is

`I(theta) = E_theta[S_theta(X)^2]`.

Under the standard regularity conditions that justify differentiating under the integral sign,

`E_theta[S_theta(X)] = 0`

and

`I(theta) = -E_theta[partial_theta^2 log p_theta(X)]`.

Thus the same quantity is both the variance of the score and the expected local curvature of the log-likelihood.

## Additivity

For iid observations `X_1, ..., X_n`,

`I_n(theta) = n I_1(theta)`.

This follows because log-likelihoods add, scores add, and the score contributions have mean zero. Independent evidence therefore contributes additively to local parameter resolution.

## Matrix Form

For `theta = (theta_1, ..., theta_d)`, let

`S_i(x) = partial_{theta_i} log p_theta(x)`.

The information matrix is

`I_ij(theta) = E_theta[S_i(X) S_j(X)]`.

Equivalently, under regularity,

`I_ij(theta) = -E_theta[partial_{theta_i} partial_{theta_j} log p_theta(X)]`.

This matrix is positive semidefinite. When nonsingular, it defines the local quadratic form

`ds^2 = dtheta^T I(theta) dtheta`,

the Fisher-Rao metric on the statistical model.

## Estimation Bound

If `T(X)` is unbiased for `psi(theta)`, then

`Cov_theta(T, S_theta) = psi'(theta)`.

Cauchy-Schwarz gives

`Var_theta(T) >= psi'(theta)^2 / I(theta)`.

For an unbiased estimator of `theta`, this reduces to

`Var_theta(T) >= 1 / I(theta)`.

In vector form, covariance is bounded below by the inverse information matrix in the corresponding directions. Efficient estimators are those whose error is aligned with the score direction, so they waste no first-order variance outside the tangent direction that carries parameter information.

## Core Insight

Fisher information converts "information about a parameter" into a local geometry of probability models. It measures how strongly the distribution changes under an infinitesimal parameter displacement, and its inverse gives the unavoidable local variance scale for regular estimation.
