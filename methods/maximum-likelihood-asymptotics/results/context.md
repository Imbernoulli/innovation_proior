## A Random Criterion

Suppose data are generated from a distribution inside a parametric family indexed by an unknown vector `theta_0`. A natural estimator can be defined by choosing the parameter value that makes the observed sample look most compatible with the model. This creates a random objective function: every new sample changes its surface, its peaks, and sometimes even whether a useful peak exists.

The scientific problem is not only to compute an optimizing parameter. The problem is to explain when an optimizer exists, stays measurable and meaningful, and behaves like a stable statistical estimate as the sample size grows.

## Global Separation

Before any local uncertainty calculation is meaningful, the random objective must point toward the true parameter at a coarse scale. The population version of the criterion needs a unique optimum at `theta_0`, and the sample criterion needs to approximate that population shape uniformly enough that distant false peaks lose.

This step depends on identifiability and on a law of large numbers strong enough for the criterion function. Without it, a formal local calculation can describe the wrong neighborhood.

## Local Fluctuation

After localization, the remaining uncertainty lives on a shrinking scale around `theta_0`. On that scale, the first derivative of the criterion is a sum of many small random contributions. If those contributions are centered and have stable variance, the central limit theorem can turn the local first-order push into approximately Gaussian noise.

This random push suggests a natural local coordinate system, but it does not by itself determine whether an optimizer has a stable limit.

## Curvature As Scale

Noise alone does not produce a parameter estimate. The local objective must also have stable curvature, so that a push in a parameter direction has a predictable cost. A sharply curved direction produces small displacement; a flat direction produces large displacement or instability.

There is also a coordinate problem. Ordinary quadratic distances can change their behavior under reparametrization, near boundaries, or when covariance varies rapidly. A useful local scale has to come from the likelihood geometry itself, not merely from a convenient Euclidean chart.

## Failure Signals

The preceding requirements are substantive. Peaks can be created by individual observations, likelihood suprema can sit on shifted supports, parameters can lie on boundaries, curvature can vanish, nuisance parameters can grow with sample size, and a change of parametrization can expose misleading quadratic approximations.

Any theory for the optimizer must therefore separate the global step that finds the right neighborhood from the local step that describes uncertainty inside that neighborhood.
