I will explain the canonical method known as the Cramer-Rao bound, which gives a fundamental lower bound on the variance of any regular estimator in a parametric statistical model. The setting is a regular one-parameter family of distributions with density p_theta(x), where theta is an unknown parameter and we observe a sample X drawn from this family. We want to estimate some function psi(theta), or theta itself, using a statistic T(X). Because T is a function of random data, its quality is judged through its sampling distribution, and in particular through its variance and its expected value under each possible parameter.

The central difficulty before the Cramer-Rao bound was that there was no universal floor on how well any estimator could do. One could compare maximum likelihood, method of moments, least squares, or minimum-variance unbiased estimators, but each comparison was between specific procedures. A better estimator might always exist. What was missing was a local obstruction: a statement that any statistic whose mean responds to the parameter at a certain rate must carry a corresponding amount of sampling variability, regardless of how cleverly it is constructed.

The key insight is to connect the estimator's mean response to the score of the model. The score is defined as S_theta(X) = partial_theta log p_theta(X), the derivative of the log density with respect to theta. Under the standard regularity conditions, the expected score is zero, because the integral of the density is one and differentiation passes through the integral. Now suppose the estimator T has mean E_theta[T] = psi(theta). Differentiating this mean under the integral sign gives psi'(theta) = partial_theta E_theta[T] = integral T(x) partial_theta p_theta(x) dx. Rewriting the derivative of the density as p_theta(x) times the derivative of its logarithm, this becomes psi'(theta) = E_theta[T S_theta]. Since the score has mean zero, this expectation is exactly the covariance between T and the score: psi'(theta) = Cov_theta(T, S_theta).

This covariance identity is the bridge between the estimator and the local geometry of the model. It says that an estimator whose mean moves with theta is forced to correlate with the score direction. Once that covariance is fixed, the Cauchy-Schwarz inequality gives an immediate variance lower bound: psi'(theta)^2 = Cov_theta(T, S_theta)^2 <= Var_theta(T) Var_theta(S_theta). The variance of the score is by definition the Fisher information I(theta) = E_theta[S_theta(X)^2]. Rearranging yields the scalar Cramer-Rao bound: Var_theta(T) >= psi'(theta)^2 / I(theta). For an unbiased estimator of theta itself, psi(theta) = theta and psi'(theta) = 1, so the bound simplifies to Var_theta(T) >= 1 / I(theta). When the data consist of n independent and identically distributed observations, Fisher information adds across samples, giving the familiar form Var_theta(T) >= 1 / (n I_1(theta)), where I_1(theta) is the information in a single observation.

The geometry behind this result is as elegant as the algebra. In the Hilbert space of centered square-integrable random variables under P_theta, the score is the tangent vector to the statistical model at theta. Fisher information is the squared length of that tangent. The mean constraint on the estimator fixes the projection of the estimator onto this tangent direction. A fixed projection onto a short tangent forces the estimator vector to be long, and a long estimator vector is precisely a large variance. Thus the Cramer-Rao bound is not a trick tied to a particular estimator; it is a projection constraint imposed by the model's local motion and the estimator's required local response.

Equality in the bound occurs only when the centered estimator lies exactly along the score direction. In the scalar case this means T - E_theta[T] = (psi'(theta) / I(theta)) S_theta, and for an unbiased estimator of theta this becomes T - theta = S_theta / I(theta). Such exact alignment happens in simple models, such as estimating the mean of a normal distribution with known variance, where the sample mean attains the bound. In most families, however, the estimator has components orthogonal to the score that add variance without improving the mean response, so the bound is strict.

The multiparameter extension follows the same logic with vectors and matrices. Let theta be a vector parameter, S_theta the score vector, I(theta) = E_theta[S_theta S_theta^T] the Fisher information matrix, and J(theta) the Jacobian of the estimator's mean vector with respect to theta. Then differentiating the mean gives a cross-covariance matrix between estimator error and score equal to J(theta), and projecting through all tangent directions yields the matrix inequality Cov_theta(T) >= J(theta) I(theta)^{-1} J(theta)^T in positive-semidefinite order, assuming I(theta) is nonsingular. The scalar reciprocal bound is just the one-dimensional face of this more general metric duality.

The regularity assumptions matter concretely. The proof requires that the support of p_theta does not shift with theta, and that differentiation can be interchanged with integration. If these conditions fail, boundary terms or discontinuities can introduce local information not captured by the ordinary score, and the Cramer-Rao floor may no longer apply. That is why the theorem is properly understood as a result about smooth regular models, not a universal statement for every family of distributions.

The Cramer-Rao bound therefore serves as a benchmark for estimator design. If an estimator achieves the bound, it is efficient in the strong sense of retaining all the Fisher information in the data. If it falls short, the gap measures information lost by the estimator, not by the model. It also explains why maximum likelihood estimators are often asymptotically optimal: under regularity, they concentrate at a rate governed by the inverse Fisher information, matching the bound in the large-sample limit.

```python
import numpy as np
from scipy.stats import norm

def fisher_information_normal_mean(sigma=1.0):
    """One-observation Fisher information for the mean of a N(theta, sigma^2) model."""
    return 1.0 / (sigma ** 2)

def cramer_rao_bound_scalar(psi_prime, fisher_information):
    """Scalar Cramer-Rao lower bound: Var(T) >= psi'(theta)^2 / I(theta)."""
    return (psi_prime ** 2) / fisher_information

def sample_mean_estimator(samples, sigma=1.0):
    """Sample mean estimator for N(theta, sigma^2). It is unbiased and efficient."""
    n = len(samples)
    theta_hat = np.mean(samples)
    variance = sigma ** 2 / n
    return theta_hat, variance

# Demonstration: estimate the mean of a normal distribution with known variance.
np.random.seed(0)
true_theta = 2.0
sigma = 1.5
n = 10000
samples = np.random.normal(loc=true_theta, scale=sigma, size=n)

theta_hat, empirical_variance = sample_mean_estimator(samples, sigma)
i1 = fisher_information_normal_mean(sigma)
crb = 1.0 / (n * i1)

print(f"Estimated theta: {theta_hat:.4f}")
print(f"Empirical variance: {empirical_variance:.6f}")
print(f"Cramer-Rao bound:   {crb:.6f}")
print(f"Ratio (empirical / bound): {empirical_variance / crb:.4f}")

# Multiparameter demonstration: bivariate normal with unknown mean and known covariance.
def cramer_rao_bound_multi(jacobian, fisher_matrix):
    """Matrix Cramer-Rao lower bound: Cov(T) >= J I^{-1} J^T."""
    return jacobian @ np.linalg.inv(fisher_matrix) @ jacobian.T

Sigma = np.array([[sigma ** 2, 0.0], [0.0, sigma ** 2]])
I_multi = n * np.linalg.inv(Sigma)
J = np.eye(2)
CRB_multi = cramer_rao_bound_multi(J, I_multi)
print("Multiparameter Cramer-Rao bound covariance:")
print(CRB_multi)
```
