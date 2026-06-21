## Research question

We are given only unlabelled observations `x_1, ..., x_l` sampled i.i.d. from an
unknown distribution `P` on an input space `X`, often with `X` a high-dimensional
subset of `R^N`. The goal is to learn a simple region `S` that contains most of the
probability mass, so that a future point falling outside `S` can be treated as novel,
faulty, or anomalous. Condition-monitoring and novelty-detection settings frequently
come with a prior belief about the contamination fraction in training data.

The question is how to estimate such a region directly, without a parametric density
model and with tractable computation in high dimension.

## Background

Kernel support-vector methods are already a mature supervised-learning tool. A
feature map `Phi: X -> F` sends inputs into an inner-product space, and a kernel
computes inner products there:

```text
k(x, y) = <Phi(x), Phi(y)>.
```

The Gaussian kernel,

```text
k(x, y) = exp(-||x - y||^2 / c) = exp(-gamma ||x - y||^2),
```

is localized and translation-invariant, with `gamma = 1/c` controlling the length
scale. For this kernel `k(x, x) = 1`, so all mapped observations have the same
feature-space norm.

The supervised soft-margin SVM solves a maximum-margin separation problem between
two labelled classes:

```text
min_{w,b,xi}  1/2 ||w||^2 + C sum_i xi_i
s.t.          y_i(<w, Phi(x_i)> + b) >= 1 - xi_i,   xi_i >= 0.
```

The regularizer `1/2 ||w||^2` controls the flatness of the decision function in feature
space. Lagrangian duality gives a kernel expansion

```text
w = sum_i alpha_i y_i Phi(x_i),
f(x) = sgn(sum_i alpha_i y_i k(x_i, x) + b),
```

where only support vectors with `alpha_i > 0` appear.

A separate statistical line describes the target region itself. Einmahl and Mason's
multidimensional quantile framework fixes a class `C` of measurable sets and a size
functional `lambda`, then defines

```text
U(alpha) = inf { lambda(C) : P(C) >= alpha, C in C }.
```

With `lambda` equal to Lebesgue measure, the minimizer is a minimum-volume set
containing at least an `alpha` fraction of the probability mass; for `alpha = 1`, this is
the support when the density exists, and the set remains meaningful even when it does
not.

The `nu` reparameterization of SVMs is also available. In the supervised `nu`-SVM,
one replaces an opaque penalty constant by `nu in (0, 1]`, introduces a margin
variable `rho`, and proves that `nu` upper-bounds the fraction of margin errors while
lower-bounding the fraction of support vectors.

## Baselines

**Density estimation and thresholding.** Parzen windows, Gaussian mixtures, and
related density estimators first estimate `p`, for example

```text
p_hat(x) = (1/l) sum_i k(x_i, x),
```

then accept points whose estimated density exceeds a threshold, yielding a level-set
rule.

**Minimum-volume-set estimators.** Sager, Hartigan, Nolan, Polonik, and Tsybakov
study estimators of the minimum-size set carrying probability mass `alpha`, with
results on consistency, rates, VC-type richness, bracketing entropy, and particular set
families such as convex sets, ellipsoids, or piecewise-polynomial approximations.

**Feature-space balls and data-domain descriptions.** The smallest-enclosing-sphere
idea appears in capacity bounds and in data-domain description methods. Its soft
version finds a center `a` and radius `R` by solving

```text
min_{R,a,xi}  R^2 + C sum_i xi_i
s.t.          ||Phi(x_i) - a||^2 <= R^2 + xi_i,   xi_i >= 0.
```

The dual has the form

```text
max_alpha  sum_i alpha_i k(x_i, x_i)
           - sum_{ij} alpha_i alpha_j k(x_i, x_j)
s.t.       sum_i alpha_i = 1,   0 <= alpha_i <= C,
```

with center `a = sum_i alpha_i Phi(x_i)`. A point is accepted when its
feature-space distance from the center is at most `R`.

**Neural-network boundaries and synthetic outliers.** One can train a network to
produce a boundary around the target data, or synthesize artificial outliers and train
a two-class classifier.

## Evaluation settings

Natural evaluation settings for an unlabelled region estimator include low-dimensional
toy distributions whose learned regions can be plotted; standard digit-image data such
as USPS, trained on one digit class as normal and scored against held-out or atypical
examples; and condition-monitoring streams where the training set represents normal
operation and later observations are ranked for novelty.

Relevant diagnostics are the fraction of training points rejected by the learned region,
the fraction of support vectors in the fitted kernel expansion, sensitivity to the kernel
width, and the cost of fitting as the sample size grows. These are settings and
quantities to inspect, not outcome claims.

## Code framework

The available software substrate is a kernel-method estimator with a standard `fit` /
`decision_function` interface. One object fits on unlabelled features, delegates hard
optimization to a kernel solver, and returns larger values for more unusual points.

```python
class SupportRegionDetector:
    """Unlabelled kernel-region scorer with a fit/score interface."""

    def __init__(self, nu=0.5, gamma="auto", kernel="rbf"):
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = None

    def fit(self, X):
        # TODO: define and fit the unlabelled kernel-region rule.
        pass

    def decision_function(self, X):
        # TODO: convert the fitted signed score into an anomaly score.
        pass
```
