The central difficulty is measuring how much a sample tells us about a parameter before we commit to a specific estimator. Consistency is not enough: a statistic can converge to the right value while discarding most of the data. Estimator variance, by contrast, only makes sense after we have already chosen a statistic, so it cannot tell us what the best possible variance is. Even the likelihood itself is not a direct answer, because its absolute value has no meaning and can be rescaled arbitrarily. What we really need is an intrinsic, model-level quantity that captures how sharply the probability law separates neighboring parameter values.

A satisfactory measure should be nonnegative, additive across independent observations, invariant under reparameterization, and it should explain the fundamental lower bounds on estimator variance. The answer turns out to be the local geometry of the log-likelihood near the true parameter. By looking at how fast the log-density changes and how curved it is, we obtain a number that belongs to the model itself rather than to any particular procedure.

The method is Fisher information. For a regular scalar parametric model p_theta(x), the score is the derivative of the log-likelihood with respect to theta: S_theta(x) = partial_theta log p_theta(x). This score points in the direction in which the distribution changes most rapidly, but its expected value under the true model is zero, so the signed derivative itself cannot be the information. The right invariant quantity is the expected squared score, I(theta) = E_theta[S_theta(X)^2]. Under the usual regularity conditions that allow differentiation under the integral, this equals the negative expected second derivative of the log-likelihood: I(theta) = -E_theta[partial_theta^2 log p_theta(X)]. Thus Fisher information is simultaneously the variance of the likelihood tangent and the expected local curvature of the log-likelihood.

For independent and identically distributed observations, the log-likelihood is a sum, so the total score is a sum of independent mean-zero contributions. The cross terms vanish in expectation, and the information scales linearly with sample size: I_n(theta) = n I_1(theta). This additivity matches the intuition that two independent observations should contribute twice as much local resolution as one. For a vector parameter theta = (theta_1, ..., theta_d), the score becomes a vector and Fisher information becomes a matrix I_ij(theta) = E_theta[S_i(X) S_j(X)] = -E_theta[partial_{theta_i} partial_{theta_j} log p_theta(X)]. This matrix is positive semidefinite and transforms like a metric tensor, so it defines the Fisher-Rao metric ds^2 = dtheta^T I(theta) dtheta on the statistical manifold.

The same object explains the Cramer-Rao lower bound. If T(X) is an unbiased estimator of psi(theta), then differentiating its expectation shows that Cov_theta(T, S_theta) = psi'(theta). By Cauchy-Schwarz, any estimator with a fixed inner product against a vector of squared length I(theta) must satisfy Var_theta(T) >= psi'(theta)^2 / I(theta). For estimating theta itself this becomes Var_theta(T) >= 1 / I(theta). Efficient estimators are those whose error is aligned with the score direction, so they attain the bound. In this way Fisher information converts the vague notion of "information about a parameter" into a precise local geometry of probability models and the unavoidable variance scale for regular estimation.

Collected into one statement, this is the finished result. For a regular scalar parametric family $p_\theta(x)$, define the score $S_\theta(x) = \partial_\theta \log p_\theta(x)$. The Fisher information in a single observation is

$$I(\theta) \;=\; \mathbb{E}_\theta\!\left[S_\theta(X)^2\right] \;=\; -\,\mathbb{E}_\theta\!\left[\partial_\theta^2 \log p_\theta(X)\right],$$

the second equality holding under the regularity conditions that let differentiation pass under the integral sign. For $n$ independent, identically distributed observations the information adds,

$$I_n(\theta) \;=\; n\,I_1(\theta).$$

For a vector parameter $\theta = (\theta_1,\dots,\theta_d)$ with score components $S_i(x) = \partial_{\theta_i}\log p_\theta(x)$, the information becomes the positive semidefinite matrix

$$I_{ij}(\theta) \;=\; \mathbb{E}_\theta\!\left[S_i(X)\,S_j(X)\right] \;=\; -\,\mathbb{E}_\theta\!\left[\partial_{\theta_i}\partial_{\theta_j}\log p_\theta(X)\right],$$

which, when nonsingular, defines the Fisher-Rao metric $ds^2 = d\theta^{\top} I(\theta)\, d\theta$ on the statistical manifold. Its inverse governs every regular unbiased estimator: for any statistic $T(X)$ with $\mathbb{E}_\theta[T] = \psi(\theta)$,

$$\mathrm{Var}_\theta(T) \;\ge\; \frac{\psi'(\theta)^2}{I(\theta)},$$

and in particular $\mathrm{Var}_\theta(T) \ge 1/I(\theta)$ for an unbiased estimator of $\theta$ itself, with equality exactly when $T$'s deviation from $\theta$ is proportional to the score $S_\theta(X)$ (in the vector case, when the covariance of $T$ meets $I(\theta)^{-1}$ along the directions the score constrains). This is the answer to the reduction problem I opened with: $I(\theta)$ is the intrinsic, additive, reparameterization-covariant measure of how much a sample tells us about $\theta$, and $1/I(\theta)$ is the unavoidable local variance floor for any regular estimator that tries to recover it.
