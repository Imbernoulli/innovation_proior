# Context

## Research question

I observe a single draw of a $p$-dimensional Gaussian vector $X = (X_1,\dots,X_p)$ whose coordinates are independent, $X_i \sim N(\theta_i, 1)$ (more generally $X \sim N(\theta, \sigma^2 I)$ with $\sigma^2$ known), and I want to estimate the whole mean vector $\theta$. I am graded by **total** squared-error loss

$$L(\theta, d) = \|\theta - d\|^2 = \sum_{i=1}^p (\theta_i - d_i)^2,$$

and an estimator $\delta$ is judged by its risk $R(\theta,\delta) = E_\theta \|\delta(X) - \theta\|^2$. The coordinates carry physically unrelated quantities — there is no assumption that the $\theta_i$ are near each other or share anything.

The obvious estimate is $\delta_0(X) = X$: report each coordinate as itself. Its risk is exactly $p$ in the unit-variance normalization, flat in $\theta$. The question is whether this obvious estimator is the best one can do — precisely, whether it is **admissible**: whether there exists any estimator $\delta$ with $R(\theta,\delta) \le R(\theta,\delta_0)$ for all $\theta$ and strict somewhere. A solution that improves on $X$ would have to lower the *total* risk uniformly in $\theta$, with no prior knowledge tying the coordinates together. It matters because $X$ is the maximum-likelihood estimate, the minimum-variance unbiased estimate, and the answer every classical optimality principle endorses; if it can be uniformly beaten, those principles are not the last word for simultaneous estimation.

## Background

**Gauss and the optimality of the sample mean.** For a normal location model, the least-squares / maximum-likelihood estimate of the mean is the sample average, and Gauss (1823) showed it has lower expected squared error than any other estimator that is a *linear* function of the data and unbiased. By the mid-twentieth century the sample mean / coordinatewise identity map $X$ carried an overwhelming list of credentials: it is unbiased; among unbiased estimators it has minimum variance; it is the MLE; under squared-error loss it is minimax; it is the best *translation-invariant* estimator (the one with lowest constant risk among estimators that commute with shifting the data); and it is the best linear estimator. Every classical notion of "good" pointed at $X$.

**Decision theory: risk, admissibility, minimaxity.** Wald's decision-theoretic framework compares estimators by their risk functions. $\delta$ is *inadmissible* if some other $\delta'$ dominates it — $R(\theta,\delta') \le R(\theta,\delta)$ everywhere, strict somewhere; otherwise $\delta$ is *admissible*. A *minimax* estimator minimizes the worst-case risk. Constant-risk estimators (like $X$, whose risk is $p\sigma^2$ for every $\theta$) are the natural minimax candidates, and an invariant estimator under quadratic loss has constant risk, so the best invariant estimator is simply the one with the smallest constant risk.

**The admissibility of $X$ was firmly believed, and partly proved.** For a single normal mean ($p=1$), Blyth (1951) and Hodges & Lehmann (1951) proved $\delta_0$ admissible; Blyth's method approximates the estimator by a sequence of Bayes rules and takes a limit. Lehmann & Stein (1953) gave an admissibility result in a related one-dimensional testing setting. The Cramér–Rao / information inequality (Hodges & Lehmann 1951) and Bayes/minimax results for quadratic loss (Girshick & Savage 1951) were the standard tools for bounding the risk of a candidate estimator from below. The prevailing wisdom was that $X$ is admissible in every dimension; the open work was to *prove* it generally.

**The diagnostic fact about $\|X\|^2$.** Decompose
$$\|X\|^2 = \|X-\theta\|^2 + \|\theta\|^2 + 2(X-\theta)\cdot\theta.$$
Each of the $p$ coordinates of $X-\theta$ is standard normal, so $E\|X-\theta\|^2 = p$, and the cross term has mean zero. Hence

$$E_\theta\|X\|^2 = \|\theta\|^2 + p,$$

and more sharply $\|X\|^2 = \|\theta\|^2 + p + O_p\!\big(\sqrt{p+\|\theta\|^2}\big)$: as the dimension grows the squared length of $X$ concentrates around $\|\theta\|^2 + p$. The observed vector is therefore *systematically longer* than the truth by an amount that grows with $p$ — its squared length overshoots $\|\theta\|^2$ by about $p$. In high dimension this places $X$, with high confidence, outside the ball where $\theta$ actually sits.

**Random walks and dimension three.** A symmetric random walk on the integer lattice is recurrent in dimensions 1 and 2 — it returns to its start with probability 1 — but transient in dimension $\ge 3$ (Pólya). This makes dimension a plausible structural variable rather than just a count of unrelated coordinates: a multidimensional accumulation of independent fluctuations can behave differently once enough coordinates are present.

## Baselines

**The usual estimator $\delta_0(X) = X$ (MLE / sample mean).** Core idea: report each coordinate as observed. Math: risk $R(\theta,\delta_0) = E\|X-\theta\|^2 = p\sigma^2$ (equal to $p$ when $\sigma^2=1$), constant in $\theta$. It is unbiased, minimum-variance-unbiased, minimax, best invariant, and MLE. Gap it leaves open: all of those properties are about *unbiasedness* or *invariance* or *single-coordinate* optimality; none of them is a statement that the **total** risk over $p$ coordinates cannot be lowered by a biased, coordinate-coupling rule. The credentials guarantee nothing about admissibility once $p$ is large.

**Coordinatewise / one-at-a-time estimation.** Treat the $p$ problems separately and solve each by its own best estimate $X_i$. For a single coordinate this is admissible (Blyth 1951; Hodges–Lehmann 1951). Gap: solving them separately throws away the joint statistic $\|X\|^2$, which (per the diagnostic above) carries information about the common scale of the problem that no single coordinate reveals on its own.

**Bayes / shrinkage-toward-a-point estimators.** Put a prior $\theta_i \sim N(0, A)$. The Bayes estimate under squared-error loss is the posterior mean,
$$\delta^{\text{Bayes}}(X) = \Big(1 - \tfrac{1}{A+1}\Big) X,$$
a fixed fraction shrink toward $0$. Core idea: a prior that says "the means cluster near a point" rationally pulls estimates toward that point. Math/algorithm: linear shrinkage by a constant factor determined by $A$. Gap: it requires knowing the prior variance $A$; if $A$ is wrong the estimator can be worse than $X$; and as a frequentist procedure with $\theta$ fixed and unrelated, there is no prior, so it is unclear whether any *fixed* shrinkage can dominate $X$ for all $\theta$.

**Minimax / invariant estimators.** Restrict to estimators invariant under translation (or under the orthogonal group, for spherically symmetric estimators $\delta(X) = (1 - h(\|X\|^2))X$). Core idea: invariance is a reasonable symmetry to demand, and the best invariant rule has lowest constant risk. Math: spherically symmetric estimators move $X$ along its own ray by a length depending only on $\|X\|$. Gap: invariance is a *constraint*; nothing guarantees the best invariant estimator is admissible against rules that break the symmetry, and the diagnostic about $\|X\|^2$ hints that breaking it (by shrinking) might help.

## Evaluation settings

The natural yardstick is the risk function $R(\theta, \delta) = E_\theta\|\delta(X) - \theta\|^2$ under squared-error loss, plotted against the unknown $\theta$ (or against the scalar $\|\theta\|^2$, by spherical symmetry). The model is $X \sim N(\theta, \sigma^2 I)$ with $\sigma^2$ known (often $\sigma^2 = 1$), one observation, dimension $p$ a free parameter; the comparison is uniform-in-$\theta$ domination. Because the risk of $X$ is the constant $p\sigma^2$ (equal to $p$ when $\sigma^2=1$), the test of any candidate is whether its risk curve lies weakly below that constant everywhere and strictly below somewhere. Natural examples include parallel rate-estimation or signal-estimation problems across many independent units, where the per-coordinate variances are comparable. Risk is evaluated either in closed form, when the distribution of the relevant statistic is tractable, or by Monte-Carlo averaging over draws of $X$ at fixed $\theta$.

## Code framework

A Gaussian-mean simulation harness needs a data model, the obvious estimator, and risk evaluators. The estimator rule and any analytic risk calculation remain open slots.

```python
import numpy as np

def sample(theta, sigma=1.0, rng=None):
    """One draw X ~ N(theta, sigma^2 I)."""
    rng = rng or np.random.default_rng()
    return theta + sigma * rng.standard_normal(theta.shape)

def usual_estimator(X, sigma=1.0):
    """The obvious estimate: report X itself. Risk is exactly p*sigma^2."""
    return X

def candidate_estimator(X, sigma=1.0):
    """Placeholder for a rule that maps X to an estimate of theta."""
    # TODO: define the estimator.
    raise NotImplementedError

def risk_montecarlo(estimator, theta, sigma=1.0, n=200_000, rng=None):
    """Estimate E||estimator(X) - theta||^2 by averaging over draws."""
    rng = rng or np.random.default_rng()
    total = 0.0
    for _ in range(n):
        X = sample(theta, sigma, rng)
        d = estimator(X, sigma=sigma)
        total += np.sum((d - theta) ** 2)
    return total / n

def analytic_risk(estimator, theta, sigma=1.0):
    # TODO: risk formula for estimator, when one is available.
    raise NotImplementedError
```

The same harness compares any candidate rule against `usual_estimator` by Monte-Carlo risk, with `analytic_risk` available when a formula can be derived.
