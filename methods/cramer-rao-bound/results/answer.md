# Cramer-Rao Bound

In a regular one-parameter model with density `p_theta(x)`, define the score and Fisher information for the observed sample by

`S_theta(X) = partial_theta log p_theta(X)`

and

`I(theta) = E_theta[S_theta(X)^2]`,

and assume `0 < I(theta) < infinity`.

If an estimator `T(X)` has mean `E_theta[T] = psi(theta)`, then

`Var_theta(T) >= psi'(theta)^2 / I(theta)`.

For an unbiased estimator of `theta`, this becomes

`Var_theta(T) >= 1 / I(theta)`.

For iid observations, information adds, so if `I_1(theta)` is one-observation information, the bound is

`Var_theta(T) >= 1 / (n I_1(theta))`.

The proof is the score-covariance identity plus Cauchy-Schwarz:

`psi'(theta) = partial_theta E_theta[T] = E_theta[T S_theta] = Cov_theta(T, S_theta)`,

because `E_theta[S_theta] = 0`. Then

`psi'(theta)^2 <= Var_theta(T) Var_theta(S_theta) = Var_theta(T) I(theta)`.

For a vector parameter, with score vector `S_theta`, information matrix `I(theta) = E_theta[S_theta S_theta^T]`, and mean Jacobian `J(theta) = partial E_theta[T] / partial theta^T`,

`Cov_theta(T) >= J(theta) I(theta)^-1 J(theta)^T`

in positive-semidefinite order, when `I(theta)` is nonsingular. Equality requires the centered estimator error to be exactly the score projection: `T - E_theta[T] = J(theta) I(theta)^-1 S_theta`. In the scalar case this is `T - E_theta[T] = (psi'(theta) / I(theta)) S_theta`, so the unbiased parameter case has `T - theta = S_theta / I(theta)`.

The bound applies under the regularity conditions that make the score identity valid, including common support and interchange of differentiation and integration. Non-regular models can escape this exact form.
