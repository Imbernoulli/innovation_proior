## Research question

Two problems that look unrelated keep turning out to have the same shape. The first is *rare-event estimation*: a complicated stochastic system — a queueing network, a reliability model, a telecommunications switch — has an input vector X drawn from a known law f(·; u), a real-valued performance S(X), and a level γ, and the quantity wanted is

    ℓ = P_u(S(X) ≥ γ) = E_u[ 1_{S(X)≥γ} ],

a probability so small (10⁻⁵ and below) that the event essentially never happens in a simulation of feasible length. The second is *optimization*: a real-valued objective S over a (combinatorial or continuous) space X, and the goal is

    γ* = max_{x∈X} S(x)

together with a maximizer x*. The combinatorial cases — travelling salesman, max-cut, the quadratic assignment problem — are NP-hard; the continuous cases are multi-extremal, with several local optima.

In both, the *interesting* configurations are rare under naive sampling. In estimation, the failure set {S ≥ γ} is almost never visited; in optimization, the near-optimal states {S ≥ γ*−ε} are a tiny fraction of X. The common question is how to construct a sampling density concentrated on the set {S ≥ γ}, when that set cannot be reached by chance in feasible time.

## Background

**Crude Monte-Carlo and its relative error.** To estimate ℓ = E_u[1_{S≥γ}], draw X₁,…,X_N i.i.d. from f(·; u) and average the indicator. The estimator is unbiased with variance ℓ(1−ℓ)/N, so its *relative* error is ≈ √((1−ℓ)/(Nℓ)) ≈ 1/√(Nℓ). Pinning that to 1% relative error needs N ≈ 10⁴/ℓ draws; for ℓ = 10⁻⁵ that is 10⁹ runs.

**Importance sampling and the likelihood ratio.** The same expectation can be taken under a different sampling density g, as long as each sample is reweighted. If g(x) = 0 only where 1_{S(x)≥γ} f(x; u) = 0, then

    ℓ = ∫ 1_{S(x)≥γ} f(x; u) dx = ∫ 1_{S(x)≥γ} [f(x; u)/g(x)] g(x) dx = E_g[ 1_{S(X)≥γ} W(X) ],

with the *likelihood ratio* W(x) = f(x; u)/g(x). The estimator ℓ̂ = (1/N) Σ 1_{S(X_i)≥γ} W(X_i), X_i ∼ g, is unbiased for every admissible g; g only changes the variance. Restricting g to the *same parametric family* f(·; v) makes W a closed-form ratio W(x; u, v) = f(x; u)/f(x; v) parameterized by a single *reference (tilting) parameter* v.

**The optimal, zero-variance importance density.** Among all g, the variance-minimizing choice (by Cauchy–Schwarz, or by direct minimization of ∫(1_{S≥γ} f − ℓ g)²/g) is

    g*(x) = 1_{S(x)≥γ} f(x; u) / ℓ,

i.e. the nominal law restricted to the rare set and renormalized. Under g* the integrand 1_{S≥γ} f/g* equals ℓ identically, so the estimator has *zero* variance and one sample suffices. Its normalizer is the unknown ℓ.

**Kullback–Leibler divergence (cross-entropy).** A standard measure of dissimilarity between two densities g and h is

    D(g, h) = E_g[ ln(g(X)/h(X)) ] = ∫ g ln g dx − ∫ g ln h dx.

It is non-negative, zero iff g = h, and not symmetric. The second term, −∫ g ln h, is called the *cross-entropy* of h relative to g.

**Maximum likelihood and exponential families.** Given data x₁,…,x_N modelled as i.i.d. f(·; v), the maximum-likelihood estimate is v̂ = argmax_v Σ_i ln f(x_i; v). For a natural exponential family (Bernoulli, Gaussian, exponential, …) the stationary equation ∇_v Σ ln f(x_i; v) = 0 has a closed form: the fitted parameter is a sample moment of the data. For Bernoulli f(x; p) = p^x(1−p)^{1−x}, ∂_p ln f = (x − p)/[p(1−p)], so p̂ = (1/N) Σ x_i, the fraction of ones. For a Gaussian N(μ, σ²), ∂_μ ln f = (x − μ)/σ² and ∂_{σ²} ln f = −1/(2σ²) + (x − μ)²/(2σ⁴), so μ̂ = mean and σ̂² = variance of the data. For an exponential mean-parameterized density, v̂ = sample mean.

**Exponential tilting / change of measure for light tails.** For light-tailed laws, the way to make a tail event typical without distorting the exponential decay rate is to reweight f by an exponential and renormalize — an exponential family of tilts. For a random walk that must climb to a far level, large-deviations theory points to an exponential tilt that reverses the drift and makes the crossing typical under the new measure. These tilts form an exponential family, so a single reference parameter indexes the candidate change of measure.

**The recurring structure that ties the two problems together.** A maximization can be turned into a sequence of rare-event estimation problems: introduce the family of indicators {1_{S(x)≥γ}} and the *associated stochastic problem* ℓ(γ) = E_u[1_{S(X)≥γ}]. When γ approaches γ*, the set {S ≥ γ} shrinks to the optimizer(s), and a density that puts its mass on {S ≥ γ} therefore concentrates on x*. Both faces reduce to the same question: how to find a sampling density concentrated on {S ≥ γ}.

## Baselines

**Crude Monte-Carlo (estimation).** Draw from f(·; u), count, divide. Unbiased, with relative error ≈ 1/√(Nℓ); certifying a 10⁻⁵ probability to 1% costs ~10⁹ runs, and a few-percent estimate costs ~10⁸ runs.

**Static importance sampling with a hand-picked tilt.** Fix a single reference parameter v by analysis (large-deviations, a guessed shift), sample from f(·; v), reweight by W. The variance reduction depends on the chosen v.

**Variance minimization of the reference parameter.** Choose v to directly minimize the second moment of the IS estimator,

    v = argmin_v E_w[ 1_{S(X)≥γ} W(X; u, v) W(X; u, w) ],

estimated from a sample under w. This is a numerical optimization over v for a given family.

**Randomized search heuristics for optimization.** Simulated annealing (accept-with-Boltzmann-probability local moves), tabu search (memory-guided local search), genetic algorithms and ant-colony optimization (population-based recombination / pheromone reinforcement) all attack max S(x) by maintaining and perturbing candidate solutions. Each carries problem-specific operators and schedules (temperature cooling, mutation/crossover rates, pheromone evaporation) set by the user.

## Evaluation settings

The natural test beds combine an estimation harness and an optimization harness over standard families:

- **Rare-event estimation.** A stochastic network — e.g. a small graph with independent exponentially-distributed edge weights f(x; u) = Π_j (1/u_j) e^{−x_j/u_j}, performance S(X) = length of the shortest A→B path, and the target P_u(S(X) ≥ γ) at a level γ deep in the tail. Ground truth where available is an extremely long crude-MC run at a feasible level; the figure of merit for an *estimator* is its relative error (equivalently, runs to a target accuracy) at fixed effort, and whether it stays unbiased.
- **Combinatorial optimization.** Binary problems coded as cut/selection vectors — max-cut on a weighted graph (cut vector x ∈ {0,1}ⁿ, S = cut weight), the travelling salesman tour-length problem, the quadratic assignment problem. Sampling families: independent Bernoulli per node (stochastic-node networks) or per-edge transition matrices (stochastic-edge networks). The yardstick is solution quality versus a known optimum on synthetic instances constructed to have a planted optimum, and number of objective evaluations to reach it.
- **Continuous optimization.** Multi-extremal scalar/vector objectives S: ℝᵈ → ℝ with several local maxima. Sampling family: a (diagonal or full) Gaussian. The yardstick is convergence to the global optimum and the number of function evaluations.

## Code framework

A generic adaptive-sampling harness: a performance function S, a parametric sampler, a likelihood-ratio routine for the estimation setting, and loops that draw a sample, score it, and replace the sampling parameter. The unresolved pieces are the level choice and the refit from a scored sample.

```python
import numpy as np

def shortest_path(x):
    """Performance S for the five-edge bridge network."""
    pass

def exponential_log_likelihood_ratio(X, u, v):
    """Log f(X; u) - log f(X; v) for independent exponential means."""
    pass

def choose_level(scores, target, rho):
    """Choose the current working level from a scored sample."""
    pass

def refit_exponential_means(X, scores, level, u, v):
    """Choose the next exponential mean vector from the scored sample."""
    pass

def adaptive_rare_event(u, gamma, rho, N, N_final, rng):
    v = u
    while True:
        X = rng.exponential(scale=v, size=(N, len(v)))
        scores = np.array([shortest_path(x) for x in X])
        level, reached = choose_level(scores, gamma, rho)          # TODO
        v = refit_exponential_means(X, scores, level, u, v)        # TODO
        if reached:
            break
    # TODO: draw a final sample under v and form the reweighted tail estimate.
    pass

def refit_gaussian(X, scores, rho):
    """Choose the next Gaussian location and scale from the scored sample."""
    pass

def adaptive_optimize(objective, mu, sigma, rho, N, alpha, n_iter, tol, rng):
    for _ in range(n_iter):
        X = mu + sigma * rng.standard_normal((N, len(mu)))
        scores = np.array([objective(x) for x in X])
        mu_hat, sigma_hat = refit_gaussian(X, scores, rho)         # TODO
        # TODO: update the sampling parameters from the refit and decide when to stop.
        pass
```
