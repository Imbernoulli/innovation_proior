The core question is how to attach an accuracy statement to a statistic computed from a single sample. Given one draw x = (x₁, …, xₙ) from an unknown distribution F and a statistic θ̂ = t(x), we want its standard error, bias, or a confidence interval. For the sample mean the answer is elementary, but for the median, a ratio of means, a correlation, a discriminant error rate, or a nonlinear regression coefficient there is usually no ready formula. The jackknife offers an automatic recipe, yet it fails for the sample median because deleting one observation probes the statistic at scale O(1/n) while the true sampling fluctuations live at scale O(n^{−1/2}). That diagnostic failure points to a deeper idea: instead of linearizing the statistic, estimate the entire sampling distribution directly.

The bootstrap does this by applying the plug-in principle to the whole sampling operation. Replace the unknown F with the empirical distribution F̂, which places mass 1/n on each observed point. F̂ is the nonparametric maximum-likelihood estimate of F, and it is the distribution the data make most likely. Since we cannot draw fresh samples from F, we draw them from F̂ instead. A bootstrap resample X* = (X₁*, …, Xₙ*) consists of n points drawn with replacement from the original observations. Recomputing the statistic on each resample gives θ̂* = t(X*), and the distribution of θ̂* over resampling randomness estimates the sampling distribution of θ̂. This estimate is exact when F equals F̂, which is Fisher consistency at the most likely distribution, and it is asymptotically correct because n^{1/2}(F̂* − F̂) and n^{1/2}(F̂ − F) converge to the same multinomial normal limit.

Sampling with replacement at the original sample size is essential. Sampling without replacement merely permutes the data and collapses the distribution to a point, while smaller subsamples require ad hoc rescaling. With replacement at size n, X* is literally a sample from F̂, so R(X*, F̂) = t(X*) − θ(F̂) lives on the same scale as the true error R(X, F). For the sample median the bootstrap distribution can even be written in closed form via binomial tail probabilities, and its second moment converges to the correct asymptotic variance 1/(4 f²(θ)), exactly the quantity the jackknife mishandles. More generally the distribution is obtained by Monte Carlo: generate many resamples, recompute the statistic on each, and read the standard error and quantiles off the resulting histogram. A Taylor expansion of the resampled statistic in the multinomial weight vector recovers the infinitesimal jackknife as the linear term, so the jackknife is the delta-method approximation to the bootstrap, while the bootstrap captures the full nonlinear behavior.

From the bootstrap replicates we get the standard error as their Bessel-corrected standard deviation and the bias as their mean minus the original estimate. A percentile interval is formed from the empirical α/2 and 1−α/2 quantiles. The BCa interval improves this by estimating a bias correction z₀ and an acceleration â from the proportion of replicates below θ̂ and from leave-one-out values, then mapping nominal tail probabilities through ndtr(z₀ + (z₀ + zₐ)/(1 − â(z₀ + zₐ))). For regression, residuals rather than whole rows should be resampled so the identical error distribution is reused across design points. Variants include the smoothed bootstrap, the symmetrized bootstrap, and the parametric bootstrap, but the nonparametric empirical bootstrap is the default that assumes nothing beyond the data.

```python
import numpy as np
from scipy.special import ndtr, ndtri


def empirical_distribution(x):
    return np.asarray(x)


def resample(x, n_resamples, rng):
    x = empirical_distribution(x)
    n = x.shape[-1]
    indices = rng.integers(0, n, size=(n_resamples, n))
    return x[..., indices]


def jackknife_resamples(x):
    x = empirical_distribution(x)
    n = x.shape[-1]
    indices = np.broadcast_to(np.arange(n), (n, n))
    keep = indices[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    return x[..., keep]


def percentile_of_score(values, score):
    values = np.asarray(values)
    score = np.expand_dims(score, axis=-1)
    n = values.shape[-1]
    return (
        np.count_nonzero(values < score, axis=-1)
        + np.count_nonzero(values <= score, axis=-1)
    ) / (2 * n)


def adjusted_quantile_levels(x, statistic, resampled_statistics, alpha):
    theta_hat = statistic(x, axis=-1)
    P = percentile_of_score(resampled_statistics, theta_hat)
    z0 = ndtri(P)

    theta_delete = statistic(jackknife_resamples(x), axis=-1)
    theta_dot = np.mean(theta_delete, axis=-1, keepdims=True)
    n = x.shape[-1]
    U = (n - 1) * (theta_dot - theta_delete)
    a_hat = (1.0 / 6.0) * np.sum(U**3, axis=-1) / np.sum(U**2, axis=-1) ** 1.5

    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    lo = ndtr(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    hi = ndtr(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))
    return lo, hi


def sampling_distribution(x, statistic, n_resamples, rng):
    return statistic(resample(x, n_resamples, rng), axis=-1)


def standard_error(resampled_statistics):
    return np.std(resampled_statistics, ddof=1, axis=-1)


def quantile_along_last(values, level):
    values = np.asarray(values)
    level = np.asarray(level)
    if level.ndim == 0:
        return np.quantile(values, level, axis=-1)
    flat_values = values.reshape((-1, values.shape[-1]))
    flat_levels = np.broadcast_to(level, values.shape[:-1]).ravel()
    out = [np.quantile(v, q) for v, q in zip(flat_values, flat_levels)]
    return np.asarray(out).reshape(values.shape[:-1])


def confidence_interval(resampled_statistics, confidence_level, levels=None):
    alpha = (1 - confidence_level) / 2
    if levels is None:
        levels = (alpha, 1 - alpha)
    low, high = levels
    return (
        quantile_along_last(resampled_statistics, low),
        quantile_along_last(resampled_statistics, high),
    )


def bootstrap(x, statistic, n_resamples=9999, confidence_level=0.95,
              method="bca", rng=None):
    rng = np.random.default_rng(rng)
    x = empirical_distribution(x)
    theta_star = sampling_distribution(x, statistic, n_resamples, rng)
    alpha = (1 - confidence_level) / 2

    if method.lower() == "bca":
        levels = adjusted_quantile_levels(x, statistic, theta_star, alpha)
    elif method.lower() == "percentile":
        levels = None
    else:
        raise ValueError('method must be "bca" or "percentile"')

    return {
        "bootstrap_distribution": theta_star,
        "standard_error": standard_error(theta_star),
        "confidence_interval": confidence_interval(
            theta_star, confidence_level, levels
        ),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.standard_normal(13)
    res = bootstrap(data, lambda a, axis: np.median(a, axis=axis), rng=rng)
    print("SE(median) =", res["standard_error"])
    print("95% CI    =", res["confidence_interval"])
```
