## Research question

Given a random sample x = (x₁, …, xₙ) drawn from an unknown probability distribution F, and a
statistic θ̂ = t(x) computed from it — the sample median, a ratio of means E_F Y / E_F Z, the
Pearson correlation, the misclassification rate of a discriminant rule, a coefficient of a
nonlinear regression — how accurate is θ̂? Concretely: what is its standard error, its bias, its
whole sampling distribution? The estimate by itself is nearly useless; "54% of voters are
Democrats" means something only once it is "54% plus or minus 3%."

For the sample mean the answer is textbook: SE = s/√n. But for almost everything more
interesting there is no such formula, or deriving one is a separate research problem per
statistic. The goal is a single, automatic, dependable procedure that returns the sampling
distribution — and hence the standard error and confidence interval — of *any* statistic
t(x), without a new theoretical derivation each time, and without modeling assumptions about
the form of F.

## Background

The object of interest is the sampling distribution of a random variable R(X, F) that may
depend on both the data X and the unknown F — most often R(X, F) = t(X) − θ(F), the error of an
estimator. Under repeated sampling of X from F this R has a distribution; its mean is the bias
of t and its variance is the squared standard error. We never see F, only one realization x, so
the distribution of R must be *estimated* from that single sample.

The empirical (sample) distribution function F̂ puts mass 1/n at each observed point x₁, …, xₙ.
It is the nonparametric maximum-likelihood estimate of F: among all distributions, it is the one
the data make most likely, and it is consistent — as n grows it converges to F. The plug-in
principle says: to estimate any functional θ = T(F), evaluate the same functional at the
empirical distribution, T(F̂). For the mean this gives x̄; for the variance it gives the
sample variance. Plug-in turns "estimate a property of the unknown F" into "compute that
property of the known F̂."

The delta method (Taylor / statistical-differentials) handles smooth statistics: write t as a
functional of F, expand to first order in a perturbation of F, and the leading term gives an
asymptotic variance as a sum of squared influence contributions. Gray, Schucany and Watkins
(1975) develop the generalized-jackknife/statistical-differentials connection. The method is
powerful but per-statistic, demands hand calculation, and degrades or fails when t is not
smoothly differentiable in F — the sample median being the canonical hard case, since its
behavior depends erratically on the data near the center.

A diagnostic fact that frames the whole problem: the median exposes where automatic variance
estimators break. The leave-one-out perturbations a deletion-based method uses sit at distance
O(1/n) from the data, but the genuine sampling fluctuations of the median sit at scale
O(n^{−1/2}); a method that probes the statistic only at the O(1/n) scale reads off the wrong
local behavior. Quenouille–Tukey jackknife variance estimates of the median are, in fact, not
consistent (Miller 1974 reviews exactly such successes and failures), whereas the correct
asymptotic squared error of the sample median is (1/4 f²(θ))·(1/n) for a density f.

## Baselines

**Quenouille–Tukey jackknife.** Quenouille (1949, 1956) introduced leave-one-out recomputation
to *reduce bias*: with θ̂₍₋ᵢ₎ the estimate computed after deleting xᵢ, the bias-corrected
estimate removes the O(1/n) bias term, leaving O(1/n²). Tukey (1958) named it the "jackknife,"
and showed the spread of the **pseudo-values** θ̃ᵢ = n·θ̂ − (n−1)·θ̂₍₋ᵢ₎ estimates *variance*
and supports approximate-t confidence intervals. With θ̄₍₋₎ = (1/n)Σ θ̂₍₋ᵢ₎ the standard
estimators are

    bias_jack = (n−1)(θ̄₍₋₎ − θ̂),
    var_jack  = ((n−1)/n) Σᵢ (θ̂₍₋ᵢ₎ − θ̄₍₋₎)².

It is automatic — one recipe for any statistic — which is its appeal. Its gap: it is a *linear*
(first-order) device whose perturbations are O(1/n) deletions, and it can fail for statistics
whose local behavior at that scale is irregular. Miller (1974, "The jackknife — a review")
documents where it works and where it does not; the sample-median variance is a clean failure.

**Jaeckel's infinitesimal jackknife.** Jaeckel (1972) reformulated the jackknife as
differentiation. Write the statistic as a function R(P) of the cell-weight vector P =
(P₁, …, Pₙ), Pᵢ being the mass placed on xᵢ; the ordinary statistic sits at P = (1/n, …, 1/n).
The directional derivatives Uᵢ = ∂R/∂Pᵢ are influence components, and the variance approximation
is Σᵢ Uᵢ²/n². This is the delta method written on the simplex of reweightings; it is the smooth
ideal the ordinary jackknife approximates by finite differences, and it carries the same
smoothness requirement and the same failure mode.

**Hartigan's subsample / replaced-sample methods.** Hartigan (1969, 1971, 1975) used the values
of t on subsets of the data. Drawing subsamples (without replacement) from the 2ⁿ−1 nonempty
subsets, or "replaced samples," yields asymptotically valid confidence statements under fairly
general conditions. The artificial samples are smaller than n (so the variability scale must be
rescaled) and the matching to the true sampling law is only asymptotic.

**Cross-validation (leave-one-out)** for the specific problem of error-rate estimation in
discriminant analysis (Lachenbruch–Mickey 1968): hold out one point, classify it with a rule
trained on the rest, average the misclassifications. Automatic for prediction error, but a
distinct device tied to that one problem.

## Evaluation settings

Natural testbeds, all available beforehand, where an automatic accuracy estimate would be
judged:
- **Sample median**, n odd (e.g. n = 13), sampling X_i ∼ N(0,1) — the standard hard case for
  variance estimation; the true expected absolute/standardized error is known analytically for
  comparison.
- **Ratio estimation** E_F Y / E_F Z from bivariate data (Y, Z) > 0.
- **Pearson correlation** on small bivariate samples, optionally on the tanh⁻¹ (Fisher z)
  scale.
- **Error rate of a linear (Fisher) discriminant** trained on two samples from F and G on Rᵏ,
  measured against the leave-one-out cross-validation estimate as the yardstick.
- **Regression coefficients**, linear and nonlinear, with the residuals as the resampling unit.
Metrics: estimated standard error / variance versus the known or asymptotic truth; coverage and
width of confidence intervals.

## Code framework

The primitives already in hand: a sample as an array, a user-supplied statistic, a uniform
random-integer generator, ordinary array reductions (mean, std, quantile), and the standard
normal cdf/quantile (ndtr/ndtri) for any normal-theory interval. The slot to be filled is a
single general routine that takes *any* statistic and returns its accuracy.

```python
import numpy as np
from scipy.special import ndtr, ndtri

def empirical_distribution(x):
    # Equal mass 1/n on each observed point.
    return np.asarray(x)

def resample(x, n_resamples, rng):
    # TODO: produce n_resamples artificial datasets from the empirical distribution.
    pass

def jackknife_resamples(x):
    # Existing leave-one-out perturbations.
    pass

def percentile_of_score(values, score):
    # TODO: locate the observed statistic inside an artificial sampling distribution.
    pass

def adjusted_quantile_levels(x, statistic, resampled_statistics, alpha):
    # TODO: optionally adjust percentile levels using bias and leave-one-out curvature.
    pass

def sampling_distribution(x, statistic, n_resamples, rng):
    # TODO: estimate the distribution of statistic(X) under repeated sampling.
    pass

def standard_error(resampled_statistics):
    # TODO: read the SE off the estimated sampling distribution.
    pass

def quantile_along_last(values, level):
    # TODO: read a quantile along the artificial-sample axis.
    pass

def confidence_interval(resampled_statistics, confidence_level, levels=None):
    # TODO: read an interval off the estimated sampling distribution.
    pass

def resampling_accuracy(x, statistic, n_resamples, confidence_level, method, rng):
    # TODO: return the distribution estimate, its SE, and an interval.
    pass

# The existing automatic competitor, for reference — leave-one-out:
def jackknife_variance(x, statistic):
    x = np.asarray(x); n = len(x)
    theta_loo = np.array([statistic(np.delete(x, i)) for i in range(n)])
    theta_bar = theta_loo.mean()
    return (n - 1) / n * np.sum((theta_loo - theta_bar) ** 2)
```
