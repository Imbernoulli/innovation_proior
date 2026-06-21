## Product-Form Posteriors

The inference object is a posterior or unnormalized graphical-model distribution that can be written as a product of local factors. The notation is compact, but the product can be computationally hostile: one mixture likelihood, one nonconjugate observation, or one hybrid discrete-continuous dependency can turn a small local model into an exact belief state with exponentially many components.

The desired approximation is deterministic and keeps useful summaries such as marginals, means, variances, and sometimes covariances. A local mode is not enough, because uncertainty and posterior mass matter. A pure sample cloud is also not the target, because the goal is a compact tractable distribution that can be reused for evidence estimates, prediction, and downstream Bayesian decisions.

## Sequential Projection

A known strategy processes factors one at a time. Starting from a tractable belief state, it multiplies in the next exact factor, computes the resulting one-step posterior as well as possible, and then replaces that posterior by a member of a tractable family. In exponential families this replacement is naturally expressed as preserving selected expectations.

This gives useful online inference and includes the factored-belief approximation used for complex stochastic processes. Its weakness is the direction of time. Early projections are made before later evidence has shaped the context, so information discarded in an early step cannot be reconsidered. In a batch problem, dependence on the order of data presentation is a defect rather than a feature.

## Iterative Local Messages

Belief propagation supplies a different template. On a tree, local messages are enough to compute exact marginals. On graphs with loops, the same recursions can be iterated as an approximation; earlier empirical work showed that these beliefs often converge to good marginal estimates, while also documenting oscillation and inaccurate convergence in harder regimes.

The attractive part is not the specific discrete formula, but the idea that a global inference problem can be attacked through local consistency. The limitation is that exact messages can be arbitrary functions of variables. This is too narrow for hybrid networks and continuous nonconjugate models unless the local messages themselves can be compressed.

## Variational Baselines

Laplace approximation uses a mode and curvature. It is often cheap, but it can ignore mass away from the chosen mode. Mean-field variational inference chooses a tractable family and optimizes a global objective, commonly a `KL(q || p)` free energy. It is systematic, but the usual direction can prefer a distribution that avoids regions where the true posterior has mass, which often leads to underestimated uncertainty.

The comparison point is therefore clear: a useful method should keep the factorization as a computational handle, preserve selected expectations rather than only a mode, avoid permanent one-pass decisions, and remain applicable when exact local messages are too rich to store.

## Test Regime

The strongest tests are models where exact inference is just out of reach globally but one-factor updates remain manageable: cluttered Gaussian observations, Bayes point machine classifiers, Gaussian latent variable models, Gaussian-process classification, and hybrid belief networks. Relevant baselines include Laplace, mean-field or variational Bayes, one-pass filtering, loopy message passing when applicable, and Monte Carlo.

Important measurements include posterior mean error, variance or marginal calibration, approximate evidence, computational cost per pass, sensitivity to update order, convergence behavior, and failure on multimodal posteriors. A fair account must include both successful average-case behavior and cases where a compact moment summary hides essential posterior structure.
