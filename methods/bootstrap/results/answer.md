# The Bootstrap

## Problem

Given one random sample x = (x₁, …, xₙ) from an unknown distribution F and any statistic
θ̂ = t(x), estimate the sampling distribution of t(X) — and hence its standard error, bias, and
confidence interval — without a closed-form variance formula and without assuming the form of F.

## Key idea

Replace the unknown F by the empirical distribution F̂ (mass 1/n on each x_i, the nonparametric
MLE of F) and apply the plug-in principle to the *entire* sampling operation. Since you cannot
draw fresh samples from F, draw them from F̂ instead: a **bootstrap resample** X* is n points
drawn **with replacement** from {x₁, …, xₙ}. Recompute the statistic on each resample,
θ̂* = t(X*). The distribution of θ̂* over the resampling randomness (F̂ fixed) estimates the
sampling distribution of θ̂. It is exact when F = F̂ (Fisher consistency), and asymptotically
correct because n^{1/2}(F̂* − F̂) and n^{1/2}(F̂ − F) share the same multinomial limit.

The jackknife is recovered as the first-order (delta-method) approximation to the bootstrap.
Writing R as a function of the resample weight vector P and expanding about P = e/n uses the
multinomial covariance

    E_* P* = e/n,     Cov_* P* = I/n² − e′e/n³.

The homogeneous extension of R gives eU = 0, eV = −nU′, eVe′ = 0, so the leading variance is

    Var_* R* ≈ U(Cov_* P*)U′ = Σᵢ Uᵢ²/n²,

with Uᵢ = ∂R/∂Pᵢ. This is Jaeckel's infinitesimal jackknife, equal to the ordinary jackknife up
to 1 + O(1/n). The full resampling distribution avoids the jackknife's failure on the sample
median because deletions probe at the wrong O(1/n) scale while resampling probes at the correct
O(n^{−1/2}) scale.

## Algorithm (nonparametric bootstrap)

1. From x, form B resamples X*¹, …, X*ᴮ, each n points sampled with replacement.
2. Compute θ̂*ᵇ = t(X*ᵇ) for each — the **bootstrap distribution**.
3. Standard error: SE = sd({θ̂*ᵇ}) (Bessel-corrected). Bias: mean({θ̂*ᵇ}) − θ̂.
4. Percentile interval: the α/2 and 1−α/2 quantiles of {θ̂*ᵇ}.
5. BCa interval (bias-corrected and accelerated): with
   P = (#{θ̂*ᵇ < θ̂} + #{θ̂*ᵇ ≤ θ̂})/(2B), set z0 = ndtri(P). From leave-one-out values
   θ̂₍₋ᵢ₎, set θ̇ = n⁻¹Σᵢθ̂₍₋ᵢ₎, Uᵢ = (n−1)(θ̇ − θ̂₍₋ᵢ₎), and
   â = (1/6)ΣᵢUᵢ³/(ΣᵢUᵢ²)^{3/2}. Apply
   ndtr(z0 + (z0 + zα)/(1 − â(z0 + zα))) at α and 1−α.

Three ways to compute the bootstrap distribution: direct theory when F̂ is simple (e.g. the
median via binomial probabilities), Monte Carlo for any statistic (the general case), or Taylor
expansion (which reproduces the jackknife). Variants: smoothed bootstrap (assume F smooth),
symmetrized bootstrap (assume F symmetric), resample residuals for regression, and the
parametric bootstrap (resample from a fitted parametric MLE — yields 1/Fisher-information for
the variance of the MLE).

## Code

Mirrors the single-sample vectorized core of `scipy.stats.bootstrap`: resample by uniform
integer indices, build the bootstrap distribution, read SE off its spread, and form percentile
or BCa intervals; BCa's acceleration uses the leave-one-out jackknife.

```python
import numpy as np
from scipy.special import ndtr, ndtri   # standard normal cdf and its inverse


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


def resampling_accuracy(x, statistic, n_resamples=9999, confidence_level=0.95,
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


def bootstrap(x, statistic, n_resamples=9999, confidence_level=0.95,
              method="bca", rng=None):
    return resampling_accuracy(
        x, statistic, n_resamples, confidence_level, method, rng
    )


# Example: standard error and 95% CI of the sample median — the case the jackknife botches.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.standard_normal(13)
    res = bootstrap(data, lambda a, axis: np.median(a, axis=axis), rng=rng)
    print("SE(median) =", res["standard_error"])
    print("95% CI    =", res["confidence_interval"])
```
