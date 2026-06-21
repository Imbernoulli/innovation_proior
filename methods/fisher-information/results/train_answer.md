The central difficulty is measuring how much a sample tells us about a parameter before we commit to a specific estimator. Consistency is not enough: a statistic can converge to the right value while discarding most of the data. Estimator variance, by contrast, only makes sense after we have already chosen a statistic, so it cannot tell us what the best possible variance is. Even the likelihood itself is not a direct answer, because its absolute value has no meaning and can be rescaled arbitrarily. What we really need is an intrinsic, model-level quantity that captures how sharply the probability law separates neighboring parameter values.

A satisfactory measure should be nonnegative, additive across independent observations, invariant under reparameterization, and it should explain the fundamental lower bounds on estimator variance. The answer turns out to be the local geometry of the log-likelihood near the true parameter. By looking at how fast the log-density changes and how curved it is, we obtain a number that belongs to the model itself rather than to any particular procedure.

The method is Fisher information. For a regular scalar parametric model p_theta(x), the score is the derivative of the log-likelihood with respect to theta: S_theta(x) = partial_theta log p_theta(x). This score points in the direction in which the distribution changes most rapidly, but its expected value under the true model is zero, so the signed derivative itself cannot be the information. The right invariant quantity is the expected squared score, I(theta) = E_theta[S_theta(X)^2]. Under the usual regularity conditions that allow differentiation under the integral, this equals the negative expected second derivative of the log-likelihood: I(theta) = -E_theta[partial_theta^2 log p_theta(X)]. Thus Fisher information is simultaneously the variance of the likelihood tangent and the expected local curvature of the log-likelihood.

For independent and identically distributed observations, the log-likelihood is a sum, so the total score is a sum of independent mean-zero contributions. The cross terms vanish in expectation, and the information scales linearly with sample size: I_n(theta) = n I_1(theta). This additivity matches the intuition that two independent observations should contribute twice as much local resolution as one. For a vector parameter theta = (theta_1, ..., theta_d), the score becomes a vector and Fisher information becomes a matrix I_ij(theta) = E_theta[S_i(X) S_j(X)] = -E_theta[partial_{theta_i} partial_{theta_j} log p_theta(X)]. This matrix is positive semidefinite and transforms like a metric tensor, so it defines the Fisher-Rao metric ds^2 = dtheta^T I(theta) dtheta on the statistical manifold.

The same object explains the Cramer-Rao lower bound. If T(X) is an unbiased estimator of psi(theta), then differentiating its expectation shows that Cov_theta(T, S_theta) = psi'(theta). By Cauchy-Schwarz, any estimator with a fixed inner product against a vector of squared length I(theta) must satisfy Var_theta(T) >= psi'(theta)^2 / I(theta). For estimating theta itself this becomes Var_theta(T) >= 1 / I(theta). Efficient estimators are those whose error is aligned with the score direction, so they attain the bound. In this way Fisher information converts the vague notion of "information about a parameter" into a precise local geometry of probability models and the unavoidable variance scale for regular estimation.

```python
import numpy as np
from scipy.stats import norm

def fisher_information_gaussian_mu(sigma=1.0):
    """
    Fisher information for the mean parameter of a univariate Gaussian
    with known standard deviation sigma.
    I(mu) = 1 / sigma^2.
    """
    return 1.0 / (sigma ** 2)

def fisher_information_gaussian_sigma(mu=0.0):
    """
    Fisher information for the standard deviation parameter of a univariate
    Gaussian with known mean mu.
    I(sigma) = 2 / sigma^2.
    """
    return 2.0 / (1.0 ** 2)  # evaluated at sigma=1 for illustration

def empirical_fisher_information(samples, theta, h=1e-5):
    """
    Numerically approximate the scalar Fisher information at theta by
    finite differences of the log-likelihood for a parametric density.
    Assumes the model is a N(theta, 1) distribution.
    """
    def log_lik(theta):
        return np.sum(norm.logpdf(samples, loc=theta, scale=1.0))

    I = -(log_lik(theta + h) - 2 * log_lik(theta) + log_lik(theta - h)) / (h ** 2)
    return I / len(samples)

if __name__ == "__main__":
    true_mu = 2.0
    sigma = 1.0
    n = 10000
    samples = np.random.normal(loc=true_mu, scale=sigma, size=n)

    analytic = fisher_information_gaussian_mu(sigma)
    empirical = empirical_fisher_information(samples, true_mu)

    print(f"Analytic I(mu):    {analytic:.6f}")
    print(f"Empirical I(mu):   {empirical:.6f}")
    print(f"Cramer-Rao bound for mu (n={n}): {1.0 / (n * analytic):.6e}")
```
