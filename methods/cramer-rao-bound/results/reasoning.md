I start with the small-sample question rather than with a favored estimator. Fisher's large-sample language tells me that information measures how sharply the likelihood reacts to the parameter, and Aitken's program tells me to ask for unbiased estimates with minimum sampling variance. But neither point by itself gives a finite-sample obstruction. I can choose a statistic and compute its variance, or I can solve a likelihood equation, yet that only compares constructions after the fact. I need a reason every regular unbiased statistic must pay a variance cost before I know which statistic I will use.

My first temptation is to look harder at likelihood curvature. A sharply curved log likelihood should mean easier estimation, and a flat log likelihood should mean harder estimation. But this is still only an intuition unless I can attach curvature to an arbitrary estimator. The estimator is a function of the data; the likelihood derivative is another function of the data. I need a bridge between them.

The bridge appears when I stop treating unbiasedness as a static averaging condition. If `E_theta[T] = psi(theta)`, then the mean of `T` is not just equal to something; it moves with `theta` at rate `psi'(theta)`. Under the regularity assumptions I can differentiate the expectation under the integral:

`psi'(theta) = partial_theta integral T(x) p_theta(x) dx = integral T(x) partial_theta p_theta(x) dx`.

Now I rewrite the derivative of the density as the density times the derivative of its logarithm. With `S_theta(x) = partial_theta log p_theta(x)`, this becomes `psi'(theta) = E_theta[T S_theta]`. The normalization of the density also differentiates to zero, so `E_theta[S_theta] = 0`. Therefore the same identity is `psi'(theta) = Cov_theta(T, S_theta)`.

This is the decisive turn. Unbiasedness, or more generally a prescribed mean response, forces a covariance with the score. For an unbiased estimator of the parameter itself, the forced covariance is exactly `1`. The statistic cannot be locally indifferent to the score direction and still have the correct mean response.

Once I see a forced covariance, I no longer need to inspect the detailed form of `T`. Cauchy-Schwarz says that a covariance cannot exceed the product of the two standard deviations:

`psi'(theta)^2 <= Var_theta(T) Var_theta(S_theta)`.

The second variance is determined by the model. Since the score has mean zero, `Var_theta(S_theta) = E_theta[S_theta^2]`, the Fisher information in the observed sample. Rearranging gives `Var_theta(T) >= psi'(theta)^2 / I(theta)`, and the unbiased scalar case gives the reciprocal form.

The geometry now becomes clear. In the Hilbert space of centered square-integrable random variables under `P_theta`, the score is the tangent vector of the probability model. Fisher information is the squared length of that tangent. The centered estimator is another vector. Differentiating the estimator's mean fixes its inner product with the tangent. A vector with a fixed projection onto a short tangent must be long; a long estimator vector is a large variance. The inequality is not a clever estimator trick. It is a projection constraint imposed by the model's local motion and the estimator's required local response.

This also explains equality. Cauchy-Schwarz is tight only if the centered estimator lies in the score direction. In the scalar case equality requires `T - E_theta[T] = (psi'(theta) / I(theta)) S_theta`; for an unbiased estimator of the parameter, this is `T - theta = S_theta / I(theta)`. That can happen for simple regular models, such as the normal mean with known variance or the sample proportion in a binomial model, but it is too rigid to expect in every family. Extra components orthogonal to the score add variance without helping the mean move correctly.

The multiparameter case is the same obstruction with several tangent directions. The score becomes a vector, and the information matrix is the Gram matrix of those score components. Differentiating the estimator's mean gives a cross-covariance between estimator error and score equal to the Jacobian of the mean map. Projecting through all directions yields a matrix lower bound in positive-semidefinite order. The scalar reciprocal is only the one-dimensional face of this metric duality.

The regularity assumptions now matter in a concrete way. I have used the identity that the derivative of an expectation equals an expectation against the score. If the support shifts with the parameter or the derivative cannot pass through the integral, that identity may lose boundary terms or fail outright. Then the score vector no longer captures all local parameter information, and the ordinary variance floor is not the right object. That is why the theorem is a regular-model theorem, not a universal statement about every possible family.

So the distinctive scientific insight is this: the lower limit becomes visible only after the estimator's mean constraint is differentiated. The score supplies the local direction and curvature of the model; unbiasedness supplies a required projection onto that direction; Cauchy-Schwarz converts the projection into a length bound. Estimation accuracy is therefore controlled by the geometry of the statistical tangent, not by a search through candidate estimators.
