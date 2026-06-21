The problem is to understand why sums of many small independent random effects converge to the same bell-shaped normal distribution, and to find a criterion that explains this beyond the special binomial case. De Moivre's classical argument for coin flips succeeds because explicit binomial coefficients can be approximated by a Gaussian, but that machinery collapses as soon as the summands have different distributions. Earlier attempts that assume uniformly bounded summands or finite higher moments capture special situations, yet they hide the true boundary: the real obstacle is not the size of individual variables in isolation, but whether any single summand (including its tail) carries a visible fraction of the total variance after normalization.

The right representation turns out to be the characteristic function phi_X(t) = E exp(itX). It always exists, it is bounded, and it is defined for every real t, so no tail integrability beyond a second moment is required. Most importantly, independence converts sums into products of transforms, which means the algebra of the proof happens at the level of multiplying many factors near one rather than convolving many distributions directly. A finite second moment gives the local expansion phi_X(t) = 1 - t^2 sigma^2 / 2 + o(t^2) around zero, and multiplying many expressions of the form 1 - small quadratic naturally produces an exponential of a negative quadratic. That exponential is exactly the characteristic function of the standard normal law, so the target limit is built into the variance structure from the start. The remaining challenge is to control the Taylor remainder uniformly across heterogeneous summands, and this is where a variance-tail condition becomes essential.

The method is the Central Limit Theorem, proved here in the Lindeberg-Feller form using characteristic functions. Consider a triangular array of independent centered random variables X_{n,1}, ..., X_{n,k_n} with variances sigma_{n,j}^2 such that the total variance sum_j sigma_{n,j}^2 tends to 1. The Lindeberg condition requires that for every fixed epsilon > 0, the sum of tail second moments sum_j E[X_{n,j}^2 1{|X_{n,j}| > epsilon}] tends to zero. This condition precisely captures the idea that no individual summand or tail contributes visible variance at the final scale. Under this assumption, the normalized sum sum_j X_{n,j} converges in distribution to the standard normal N(0,1). The Lindeberg condition is not merely sufficient; together with the variance normalization it is essentially the right boundary, because any visible tail variance would produce a non-Gaussian atom in the limit.

The proof works by comparing the true characteristic factor E exp(itX_{n,j}) with the quadratic proxy 1 - t^2 sigma_{n,j}^2 / 2. Inside the threshold |X_{n,j}| <= epsilon, Taylor's theorem bounds the error by a small multiple of X_{n,j}^2, and the Lindeberg condition makes the aggregate tail contribution vanish. Because independence turns the sum's characteristic function into a product of factors, the total error between the true product and the product of quadratic proxies goes to zero. The product of the proxies is prod_j (1 - t^2 sigma_{n,j}^2 / 2), and since the variances sum to one, its logarithm tends to -t^2 / 2. Levy's continuity theorem then converts pointwise convergence of characteristic functions into convergence in distribution to the standard normal.

This framework also explains the classical iid case and Lyapunov's condition as easy corollaries. If X_j are iid with finite variance, scaling by sqrt(n) makes every tail variance vanish, so Lindeberg's condition holds automatically. If a higher moment sum_j E|X_{n,j}|^{2+delta} tends to zero, Markov's inequality forces the Lindeberg tail sum to zero as well, showing that Lyapunov's condition is sufficient but not necessary. The essence of the theorem is therefore not a particular moment of order above two, but the disappearance of visible individual contributions under normalization.

```python
import numpy as np


def empirical_lindeberg_tail(data, epsilons=None):
    """
    Empirical Lindeberg tail sum for a normalized triangular-array row.

    data : list of 1-D arrays, one per centered summand X_{n,j}
    epsilons : thresholds at which to evaluate the tail variance sum

    Returns a dict mapping epsilon to sum_j E[X_{n,j}^2 1{|X_{n,j}| > epsilon}]
    after scaling so that the total variance is 1.
    """
    variances = np.array([np.var(s, ddof=1) for s in data])
    total_var = variances.sum()
    if total_var == 0:
        raise ValueError("Total variance is zero")

    if epsilons is None:
        epsilons = np.linspace(0.05, 1.0, 10)

    results = {}
    for eps in epsilons:
        tail_sum = 0.0
        for s in data:
            x = (s - s.mean()) / np.sqrt(total_var)  # scale to total variance 1
            tail_sum += np.mean(x ** 2 * (np.abs(x) > eps))
        results[eps] = tail_sum
    return results


def empirical_characteristic_function(samples, t_values):
    """
    Compute the empirical characteristic function of a normalized sum.

    samples : 2-D array of shape (n_summands, n_observations), already centered
    t_values : points at which to evaluate the characteristic function
    """
    # Normalize so the total variance is 1
    variances = np.var(samples, axis=1, ddof=1)
    total_var = variances.sum()
    scaled = samples / np.sqrt(total_var)

    # Form n_observations independent row sums
    row_sums = scaled.sum(axis=0)
    char_fn = np.array([np.mean(np.exp(1j * t * row_sums)) for t in t_values])
    return char_fn


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000  # number of summands
    m = 5000  # observations per summand

    # Independent but non-identically distributed summands with small total variance
    scales = rng.uniform(0.5, 1.5, size=n)
    samples = np.array([rng.normal(0, scale, size=m) for scale in scales])

    lindeberg = empirical_lindeberg_tail(samples, epsilons=[0.1, 0.2, 0.5, 1.0])
    print("Empirical Lindeberg tail sums (should be small):")
    for eps, val in lindeberg.items():
        print(f"  epsilon={eps:.2f}, tail_sum={val:.6f}")

    t_values = np.linspace(-3, 3, 61)
    empirical = empirical_characteristic_function(samples, t_values)
    target = np.exp(-t_values ** 2 / 2)
    max_error = np.max(np.abs(empirical - target))
    print(f"\nMax |empirical char fn - exp(-t^2/2)|: {max_error:.4f}")
```
