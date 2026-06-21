## Product-Form Posteriors

The inference object is a posterior or unnormalized graphical-model distribution that can be written as a product of local factors. The product can be computationally hostile: one mixture likelihood, one nonconjugate observation, or one hybrid discrete-continuous dependency turns a small local model into an exact belief state with many components.

The desired approximation is deterministic and keeps useful summaries such as marginals, means, variances, and sometimes covariances. The goal is a compact tractable distribution that can be reused for evidence estimates, prediction, and downstream Bayesian decisions.

## Sequential Projection

A known strategy processes factors one at a time. Starting from a tractable belief state, it multiplies in the next exact factor, computes the resulting one-step posterior as well as possible, and then replaces that posterior by a member of a tractable family. In exponential families this replacement is naturally expressed as preserving selected expectations.

This gives online inference and includes the factored-belief approximation used for complex stochastic processes. It processes data in a single pass, in the order presented.

## Iterative Local Messages

Belief propagation supplies a different template. On a tree, local messages are enough to compute exact marginals. On graphs with loops, the same recursions can be iterated as an approximation; earlier empirical work showed that these beliefs often converge to good marginal estimates, while also documenting oscillation and inaccurate convergence in harder regimes.

The idea is that a global inference problem can be attacked through local consistency. Exact messages can be arbitrary functions of variables.

## Variational Baselines

Laplace approximation uses a mode and curvature, and is often cheap. Mean-field variational inference chooses a tractable family and optimizes a global objective, commonly a `KL(q || p)` free energy.

## Test Regime

The strongest tests are models where exact inference is just out of reach globally but one-factor updates remain manageable: cluttered Gaussian observations, Bayes point machine classifiers, Gaussian latent variable models, Gaussian-process classification, and hybrid belief networks. Relevant baselines include Laplace, mean-field or variational Bayes, one-pass filtering, loopy message passing when applicable, and Monte Carlo.

Important measurements include posterior mean error, variance or marginal calibration, approximate evidence, computational cost per pass, sensitivity to update order, convergence behavior, and behavior on multimodal posteriors.
