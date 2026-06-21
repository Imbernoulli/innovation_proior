# Context: black-box hyperparameter optimization under a fixed resource budget (circa 2015-2016)

## Research question

Given a machine learning algorithm with a handful of free knobs — a hyperparameter
configuration `x` drawn from a configuration space `X` (continuous, integer, categorical, and
possibly conditional axes) — find a configuration that minimizes validation error, using as
little total computation as possible. Each evaluation is one full or partial training run plus
a validation pass, and that is the expensive unit: a single configuration can take hours of
GPU time. The total compute is capped at some budget `B`; the question is how to spend `B` so
that the returned configuration is as close as possible to the best one in `X`.

The map from `x` to validation error is a *black box*: non-convex, of unknown smoothness,
high-dimensional, and observed only through noisy, expensive evaluations. It is not known in
advance how validation error varies across `x` for a fixed amount of training, nor how quickly
a single configuration's loss improves as it is given more training.

## Background

By the mid-2010s, machine learning models had grown in expressivity and in cost, and their
performance had become critically sensitive to hyperparameters — learning rates, regularization
strengths, architectural widths — that interact in poorly understood ways. Two broad families of
search methods existed, and they attack different halves of the problem.

The first family is **configuration selection**: be clever about *which* configuration to try
next. Grid search and manual tuning are the brute-force end of this. The Bayesian-optimization
methods refine it by building a probabilistic model `p(y | x)` of validation performance `y`
given a configuration `x` and using it to pick promising configurations adaptively. Each chosen
configuration is trained to completion before its loss is observed.

The second family is **configuration evaluation**: be clever about *how much* resource to spend
evaluating a configuration before judging it. The resource can be number of training iterations,
size of a training-data subsample, or number of features. The idea is adaptive computation —
pour resource into configurations that look promising and quickly cut off the rest — so that a
search can examine far more candidates than a uniform "train everyone to the end" scheme.

A useful way to reason about configuration evaluation, due to the bandit literature, is to think
of each configuration as an *arm* and a unit of training as a *pull*. Pulling arm `i` a total of
`k` times produces a validation loss `ell_{i,k}`. A standing empirical fact about iteratively
trained models grounds everything that follows: as `k` grows, `ell_{i,k}` converges to a
terminal value `nu_i = lim_{k->inf} ell_{i,k}`; the *rate* of that convergence varies across
configurations, and the intermediate loss curves can be non-monotone and jagged (Jamieson &
Talwalkar 2016 show exactly such curves for kernel-SVM models trained by SGD). One can summarize
the convergence with an *envelope*: let `gamma(j)` be the smallest non-increasing function with
`sup_i |ell_{i,j} - nu_i| <= gamma(j)`, so `gamma(j)` bounds how far a partially trained loss can
be from its terminal value after `j` units of resource. `gamma` is guaranteed to exist whenever
the limits exist. Two configurations with terminal losses `nu_1 < nu_i` can be told apart once
their envelopes stop overlapping, i.e. once the resource `j` is large enough that
`gamma(j) <= (nu_i - nu_1)/2`. So `gamma^{-1}((nu_i-nu_1)/2)` is the resource needed to separate
configuration `i` from the best one — small when the curves converge fast or the gap is large,
large when convergence is slow or the gap is tiny.

A second standing fact concerns *which* configurations even exist to be found. If configurations
are drawn i.i.d. and `nu_i` has cumulative distribution `F` with infimum `nu_*`, then the chance
that `n` random draws all miss the near-optimal region is `(1 - F(nu_* + Delta))^n ≈
exp(-n F(nu_* + Delta))`. So `E[min_i nu_i - nu_*] ≈ F^{-1}(1/n) - nu_*`: `n` must be large
enough that a good terminal value is *sampled at all*, and how large depends on how rare good
configurations are under `F`.

There is also a structural reason random sampling is a respectable starting point. Bergstra &
Bengio (2012) showed that across many datasets only a few hyperparameters actually matter, and
*which* few differ by dataset; random sampling spreads its trials across the important axes far
more effectively than a grid, which spends trials on combinations of irrelevant axes. Random
search also converges to the optimum asymptotically by a simple covering argument, regardless of
the smoothness of the objective, which makes it a clean, assumption-free foundation to build on.

## Baselines

**Random search (Bergstra & Bengio, JMLR 2012).** Draw `n` configurations i.i.d. from a
distribution over `X` (uniform over a hypercube of per-hyperparameter ranges, in the default
case), train each to the full resource `R`, and return the one with the lowest validation loss.
Simple, embarrassingly parallel, and a strong baseline in high dimensions because it covers the
few axes that matter. It spends the maximum resource `R` on every configuration; under a fixed
total budget `B`, the number of configurations it examines is `B/R`.

**Bayesian optimization for configuration selection (TPE — Bergstra et al. 2011; SMAC — Hutter
et al. 2011; Spearmint — Snoek et al. 2012).** Fit a model of `p(y | x)` — Parzen-window density
estimators (TPE), random forests (SMAC), or Gaussian processes (Spearmint) — and use an
acquisition function to choose the next configuration to evaluate, then train it to completion
and update the model. Empirically these beat random search on low-dimensional benchmarks. The GP
variant costs `O(n^3)` to fit its posterior. These methods optimize *which* configuration to
evaluate, with each selected configuration trained all the way to completion before its loss is
extracted.

**Successive elimination / halving for best-arm identification (Karnin, Koren & Somekh, ICML
2013; brought to non-stochastic hyperparameter optimization by Jamieson & Talwalkar, AISTATS
2016).** Treat configurations as bandit arms whose loss sequences converge to terminal values
`nu_i`, and run a sequential-elimination procedure: with a budget `B` and `n` arms, proceed in
rounds; within each round pull every surviving arm the same number of times, then drop a fixed
fraction of the worst arms; repeat until one remains. Halving in particular keeps the better half
each round, which means survivors receive geometrically more resource while the budget per round
stays roughly constant. The non-stochastic analysis shows that if `B` exceeds a problem-dependent
quantity
`z = 2 ceil(log2 n) max_{i>=2} i (1 + gamma^{-1}((nu_i - nu_1)/2))`,
the best arm is returned, and that this is essentially the resource an oracle would need merely
to *verify* each arm's ordering against the best, which can be far smaller than the uniform
strategy's worst-case necessary budget `n · gamma^{-1}((nu_2 - nu_1)/2)`. The procedure requires
the number of arms `n` as an input. For a fixed total budget `B`, the average resource per arm is
`B/n`: large `n` means many arms with little resource each and aggressive early cutoff, while
small `n` means few arms with much resource each. The setting that performs best for a given
problem depends on `gamma` and `F`.

**Hybrid early-stopping methods (Swersky et al. 2014; Domhan et al. 2015; György & Kégl 2011;
Agarwal et al. 2012; Sparks et al. 2015).** Various combinations of adaptive selection with
adaptive evaluation, and learning-curve extrapolation that stops training when a configuration is
predicted to be unpromising, typically by fitting a parametric model of the convergence behavior
of training or by applying heuristics with user-defined safety margins.

## Evaluation settings

The natural yardsticks already in use for hyperparameter optimization at the time, against which
any new method would be measured:

- **Iterations as the resource, deep nets.** Tuning a convolutional network (e.g. on CIFAR-10,
  rotated-MNIST-with-background, SVHN) where one unit of resource is a fixed number of
  mini-batch SGD iterations, with `R` the maximum number of iterations per configuration; search
  spaces of roughly 6-8 hyperparameters (learning rate and its schedule, momentum, weight decay,
  response-normalization parameters), batch size fixed. Also the LeNet/MNIST setting with
  learning rate, batch size, and per-layer kernel counts.
- **Data-set subsampling as the resource.** A black-box batch learner where the resource is the
  size of a random training subset, `R` the full data-set size; e.g. the 117-OpenML-dataset
  AutoML framework of Feurer et al. (2015) spanning ~110 hyperparameters across many classifiers
  and preprocessors, and kernel methods (regularized least squares / SVM on CIFAR-10) whose
  superlinear training cost makes subsampling especially attractive.
- **Features as the resource.** Random-feature kernel approximations, where the resource is the
  number of random features used to approximate an RBF kernel feeding a ridge classifier; `R` the
  largest feature count that fits in memory.
- **Metrics and protocol.** Validation/test error of the best configuration found, plotted
  against total resource expended (so that early-anytime quality is visible); average rank across
  many datasets; multiple independent trials per searcher per benchmark with means reported.
  Standard reference baselines: random search, and a doubled-budget "random 2×".

## Code framework

A search harness already exists. The substrate is: a way to *sample* a configuration from the
search space; a way to *train a configuration for a chosen amount of resource and return its
validation loss*; a way to *rank* configurations by loss; and a wrapper that must return a
configuration under a fixed resource cap. What is *not* settled is the search rule inside that
wrapper.

```python
import numpy as np


def sample_configuration(space, rng):
    """Draw one configuration i.i.d. from the search space (e.g. uniform over the
    per-hyperparameter ranges). Already available."""
    return space.sample_uniform(rng)


def run_then_return_val_loss(config, resource):
    """Train `config` using `resource` units of the chosen resource type (iterations,
    data-subsample size, or features) and return its validation loss. Already available;
    this is the single expensive operation."""
    ...


def top_k(configs, losses, k):
    """Return the k configurations with the smallest losses. Already available."""
    order = np.argsort(losses)
    return [configs[i] for i in order[:k]]


def search(space, max_resource, rng):
    """Return a configuration using the primitives above.

    Everything above already exists. What is missing is the search rule that decides how
    to call the sampler and the expensive training primitive before returning a config."""
    # TODO: the search rule we will design.
    pass
```

The wrapper owns the resource cap and the three primitives; `search` is the empty slot.
