The problem is to turn the qualitative intuition of the law of large numbers and the central limit theorem into an honest, finite-sample guarantee. For independent random variables X_1, ..., X_n, the average X_bar concentrates around its mean mu, but LLN and CLT are asymptotic. When someone fixes n and a deviation t, we need an explicit upper bound on Pr{X_bar - mu >= t}. Chebyshev gives sigma^2/(n t^2), which decays only polynomially in t and is exponentially loose compared with the Gaussian tail the CLT predicts. Bernstein, Bennett, and Prohorov do better using variance information, but they are still framed for sums and require extra moment assumptions. What is missing is a clean, range-only bound that also extends beyond sums to general functions of many independent variables, provided no single coordinate can move the function very much.

The method that fills this gap is McDiarmid's inequality, also called the bounded-differences inequality. It is the capstone of a short chain: a one-dimensional MGF bound for bounded variables (Hoeffding's lemma), applied to sums (Hoeffding's inequality), then lifted to martingales (Azuma-Hoeffding), and finally to arbitrary functions of independent inputs (McDiarmid). The key insight is that the exponential-Markov step only needs the MGF to factor, and factorization survives for conditional-mean-zero increments via the tower property. So independence can be replaced by a martingale-difference structure, and any function of independent variables can be turned into such a martingale by revealing coordinates one at a time.

Hoeffding's lemma is the engine. If X is centered and a <= X <= b almost surely, then E e^{hX} <= exp(h^2 (b-a)^2 / 8). The proof uses convexity of the exponential to trap e^{hx} under its chord on [a,b], which turns the expectation into a linear function of X. After centering, the bound becomes e^{L(u)} with u = h(b-a) and L(u) = -p u + log(1 - p + p e^u), where p = -a/(b-a). A quick second-derivative check shows L''(u) <= 1/4, so Taylor expansion around u = 0 gives L(u) <= u^2/8, which is exactly the Gaussian-looking MGF bound. This says a bounded centered variable is sub-Gaussian with variance proxy (b-a)^2/4.

Feeding this into the Chernoff bound for an independent sum yields Hoeffding's inequality. For X_i in [a_i, b_i] and S = sum_i (X_i - E X_i), we have Pr{S >= t} <= exp(-2 t^2 / sum_i (b_i - a_i)^2). For [0,1] variables this gives Pr{X_bar - mu >= t} <= e^{-2 n t^2}, whose constant matches the fair-coin Gaussian tail. The same proof lifts almost verbatim to martingales. If Z_0, ..., Z_n is a martingale with bounded increments |d_k| <= c_k, then Azuma-Hoeffding gives Pr{Z_n - Z_0 >= t} <= exp(-t^2 / (2 sum_k c_k^2)). The only difference is that the conditional MGF bound is applied one step at a time using the tower property instead of full independence.

McDiarmid's inequality applies this machinery to a general function f. Suppose f satisfies the bounded-differences property: changing only the i-th coordinate changes f by at most c_i. Form the Doob martingale Z_i = E[f(X_1, ..., X_n) | X_1, ..., X_i]. Then Z_0 = E f and Z_n = f. The increment Delta_i = Z_i - Z_{i-1} has conditional mean zero, and bounded differences imply that, conditionally, Delta_i lies in an interval of width at most c_i. Applying Hoeffding's lemma conditionally and peeling the martingale gives E e^{h(f - E f)} <= exp((h^2/8) sum_i c_i^2). Optimizing the Chernoff bound over h yields

    Pr{f(X_1, ..., X_n) - E f >= t} <= exp(-2 t^2 / sum_i c_i^2),

with a factor of 2 for the two-sided version. This is distribution-free beyond independence and the sensitivities c_i; it needs no variance, no boundedness of the individual X_i beyond what controls f, and it recovers Hoeffding's inequality for sums as the special case f = sum_i X_i with c_i = b_i - a_i. That is the concentration-of-measure phenomenon made explicit: a function of many independent variables that is insensitive to any single coordinate is sharply concentrated around its mean.

```python
import numpy as np
from typing import Callable

def hoeffding_lemma_mgf_bound(h: float, a: float, b: float) -> float:
    """Upper bound on E[exp(h X)] for centered X in [a, b]."""
    return np.exp(h**2 * (b - a)**2 / 8.0)

def hoeffding_bound(t: float, ranges: list[tuple[float, float]]) -> float:
    """Pr{sum_i (X_i - E X_i) >= t} <= exp(-2 t^2 / sum_i (b_i - a_i)^2)."""
    if t <= 0:
        return 1.0
    total_range_sq = sum((b - a) ** 2 for a, b in ranges)
    return np.exp(-2.0 * t**2 / total_range_sq)

def mcdiarmid_bound(t: float, sensitivities: list[float]) -> float:
    """Pr{f(X) - E f >= t} <= exp(-2 t^2 / sum_i c_i^2)
    for f with bounded differences c_i."""
    if t <= 0:
        return 1.0
    total_sq = sum(c**2 for c in sensitivities)
    return np.exp(-2.0 * t**2 / total_sq)

def empirical_mean_sensitivity(n: int) -> list[float]:
    """For f(x) = (1/n) sum_i x_i with x_i in [0,1], changing one coordinate
    changes f by at most 1/n."""
    return [1.0 / n] * n

# Example: average of n independent [0,1] variables.
n = 100
t = 0.1
sens = empirical_mean_sensitivity(n)
print("McDiarmid / Hoeffding tail bound:", mcdiarmid_bound(t, sens))
# Direct Hoeffding formulation for the sum deviation n*t:
ranges = [(0.0, 1.0)] * n
print("Hoeffding tail bound (equivalent):", hoeffding_bound(n * t, ranges))

# Example: a general bounded-differences function.
def f(x: np.ndarray) -> float:
    """A toy function with bounded differences 1/n in each coordinate."""
    return float(np.mean(x) + 0.05 * np.sin(2 * np.pi * np.mean(x)))

# Estimate E[f] by Monte Carlo and compare with the bound.
np.random.seed(0)
samples = 100_000
estimates = np.array([f(np.random.rand(n)) for _ in range(samples)])
mean_f = estimates.mean()
deviations = estimates - mean_f
empirical_tail = np.mean(deviations >= t)
print("Empirical P(f - Ef >= t):", empirical_tail)
print("McDiarmid bound:", mcdiarmid_bound(t, [1.06 / n] * n))
```
