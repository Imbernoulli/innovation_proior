## Research question

Two problems that look unrelated keep turning out to have the same shape. The first is *rare-event estimation*: a complicated stochastic system — a queueing network, a reliability model, a telecommunications switch — has an input vector X drawn from a known law f(·; u), a real-valued performance S(X), and a level γ, and the quantity wanted is

    ℓ = P_u(S(X) ≥ γ) = E_u[ 1_{S(X)≥γ} ],

a probability so small (10⁻⁵ and below) that the event essentially never happens in a simulation of feasible length. The second is *optimization*: a real-valued objective S over a (combinatorial or continuous) space X, and the goal is

    γ* = max_{x∈X} S(x)

together with a maximizer x*. The combinatorial cases — travelling salesman, max-cut, the quadratic assignment problem — are NP-hard; the continuous cases are multi-extremal, riddled with local optima.

The pain point common to both is that the *interesting* configurations are vanishingly rare under any naive sampling law. In estimation, the failure set {S ≥ γ} is almost never visited, so a draw-and-count estimator reports zero or pure noise. In optimization, the near-optimal states {S ≥ γ*−ε} are a microscopic fraction of X, so blind random search almost never lands on them. A solution would have to *learn where the good mass lives* and steer sampling there, while staying honest — unbiased for the estimation problem, and convergent to a true optimum for the optimization problem.

## Background

**Crude Monte-Carlo and its relative error.** To estimate ℓ = E_u[1_{S≥γ}], draw X₁,…,X_N i.i.d. from f(·; u) and average the indicator. The estimator is unbiased with variance ℓ(1−ℓ)/N, so its *relative* error is ≈ √((1−ℓ)/(Nℓ)) ≈ 1/√(Nℓ). Pinning that to a fixed accuracy needs N on the order of 1/ℓ — about 10⁵/ℓ draws for 1% relative error. For ℓ = 10⁻⁵ that is already 10¹⁰ runs; the scaling is the wall.

**Importance sampling and the likelihood ratio.** The same expectation can be taken under a different sampling density g, as long as each sample is reweighted. If g(x) = 0 only where 1_{S(x)≥γ} f(x; u) = 0, then

    ℓ = ∫ 1_{S(x)≥γ} f(x; u) dx = ∫ 1_{S(x)≥γ} [f(x; u)/g(x)] g(x) dx = E_g[ 1_{S(X)≥γ} W(X) ],

with the *likelihood ratio* W(x) = f(x; u)/g(x). The estimator ℓ̂ = (1/N) Σ 1_{S(X_i)≥γ} W(X_i), X_i ∼ g, is unbiased for every admissible g; g only changes the variance. Restricting g to the *same parametric family* f(·; v) makes W a closed-form ratio W(x; u, v) = f(x; u)/f(x; v) parameterized by a single *reference (tilting) parameter* v.

**The optimal, zero-variance importance density.** Among all g, the variance-minimizing choice (by Cauchy–Schwarz, or by direct minimization of ∫(1_{S≥γ} f − ℓ g)²/g) is

    g*(x) = 1_{S(x)≥γ} f(x; u) / ℓ,

i.e. the nominal law restricted to the rare set and renormalized. Under g* the integrand 1_{S≥γ} f/g* equals ℓ identically, so the estimator has *zero* variance and one sample suffices. But g* contains the unknown ℓ in its normalizer — it presupposes the answer. It is unusable as a recipe, yet it states exactly what a good sampler should look like: live on {S ≥ γ}, and there be shaped like f.

**Kullback–Leibler divergence (cross-entropy).** A standard measure of dissimilarity between two densities g and h is

    D(g, h) = E_g[ ln(g(X)/h(X)) ] = ∫ g ln g dx − ∫ g ln h dx.

It is non-negative, zero iff g = h, and not symmetric. The second term, −∫ g ln h, is the *cross-entropy* of h relative to g; only this term depends on h, so minimizing D over a parametric h means maximizing ∫ g ln h.

**Maximum likelihood and exponential families.** Given data x₁,…,x_N modelled as i.i.d. f(·; v), the maximum-likelihood estimate is v̂ = argmax_v Σ_i ln f(x_i; v). For a natural exponential family (Bernoulli, Gaussian, exponential, …) the stationary equation ∇_v Σ ln f(x_i; v) = 0 has a closed form: the fitted parameter is a sample moment of the data. For Bernoulli f(x; p) = p^x(1−p)^{1−x}, ∂_p ln f = (x − p)/[p(1−p)], so p̂ = (1/N) Σ x_i, the fraction of ones. For a Gaussian N(μ, σ²), ∂_μ ln f = (x − μ)/σ² and ∂_{σ²} ln f = −1/(2σ²) + (x − μ)²/(2σ⁴), so μ̂ = mean and σ̂² = variance of the data. For an exponential mean-parameterized density, v̂ = sample mean.

**Exponential tilting / change of measure for light tails.** For light-tailed laws, the way to make a tail event typical without distorting the exponential decay rate is to reweight f by an exponential and renormalize — an exponential family of tilts. For a random walk that must climb to a far level, the drift-reversing tilt (the Lundberg root of E[e^{γΔ}] = 1) makes the rare crossing a sure event under the new measure; large-deviations theory identifies this tilt as the asymptotically efficient sampler. So a *parametric reference v* shifting the nominal law toward the rare set is the natural class of importance densities — the open question is which v.

**The recurring structure that ties the two problems together.** A maximization can be turned into a sequence of rare-event estimation problems: introduce the family of indicators {1_{S(x)≥γ}} and the *associated stochastic problem* ℓ(γ) = E_u[1_{S(X)≥γ}]. When γ approaches γ*, the set {S ≥ γ} shrinks to the optimizer(s), and a density that puts its mass on {S ≥ γ} therefore concentrates on x*. Both faces reduce to the same question: *how do you find a sampling density concentrated on {S ≥ γ}, when you cannot afford to wait for that set to be hit by chance?*

## Baselines

**Crude Monte-Carlo (estimation).** Draw from f(·; u), count, divide. Unbiased and correct, but relative error ≈ 1/√(Nℓ); certifying a 10⁻⁵ probability to a few percent costs ~10¹⁰ runs. The gap it leaves: it never exploits *where* the rare mass concentrates.

**Static importance sampling with a hand-picked tilt.** Fix a single reference parameter v by analysis (large-deviations, a guessed shift), sample from f(·; v), reweight by W. This can cut variance by orders of magnitude *when* the tilt is right, but the optimal v is generally very hard to obtain in closed form for a complex system, and a poorly chosen v can make the variance *worse* than crude MC — the few relevant samples carry exploding weights. The gap: no constructive, self-tuning way to find a good v.

**Variance minimization of the reference parameter.** Choose v to directly minimize the second moment of the IS estimator,

    v = argmin_v E_w[ 1_{S(X)≥γ} W(X; u, v) W(X; u, w) ],

estimated from a sample under w. This targets exactly the right objective, but the program is a generally non-convex numerical optimization with no closed-form solution even for nice families; it is awkward to run inside an adaptive loop. The gap: tractability — there is no sample-moment update.

**Randomized search heuristics for optimization.** Simulated annealing (accept-with-Boltzmann-probability local moves), tabu search (memory-guided local search), genetic algorithms and ant-colony optimization (population-based recombination / pheromone reinforcement) all attack max S(x) by maintaining and perturbing candidate solutions. They are general and often effective, but each is a bundle of problem-specific operators and schedules (temperature cooling, mutation/crossover rates, pheromone evaporation) tuned by hand; none defines a single mathematical objective whose optimum *is* the parameter update. The gap: a principled, self-tuning updating rule derived from the sampling distribution itself, rather than hand-designed move operators.

## Evaluation settings

The natural test beds combine an estimation harness and an optimization harness over standard families:

- **Rare-event estimation.** A stochastic network — e.g. a small graph with independent exponentially-distributed edge weights f(x; u) = Π_j (1/u_j) e^{−x_j/u_j}, performance S(X) = length of the shortest A→B path, and the target P_u(S(X) ≥ γ) at a level γ deep in the tail. Ground truth where available is an extremely long crude-MC run at a feasible level; the figure of merit for an *estimator* is its relative error (equivalently, runs to a target accuracy) at fixed effort, and whether it stays unbiased.
- **Combinatorial optimization.** Binary problems coded as cut/selection vectors — max-cut on a weighted graph (cut vector x ∈ {0,1}ⁿ, S = cut weight), the travelling salesman tour-length problem, the quadratic assignment problem. Sampling families: independent Bernoulli per node (stochastic-node networks) or per-edge transition matrices (stochastic-edge networks). The yardstick is solution quality versus a known optimum on synthetic instances constructed to have a planted optimum, and number of objective evaluations to reach it.
- **Continuous optimization.** Multi-extremal scalar/vector objectives S: ℝᵈ → ℝ with several local maxima. Sampling family: a (diagonal or full) Gaussian. The yardstick is convergence to the global optimum and the number of function evaluations.

## Code framework

A generic adaptive-sampling harness: a parametric family from which one can sample and whose log-density gradient is known, a performance function S, and a loop that draws a sample, scores it, and refits the sampling parameter. The unresolved piece is the *refitting rule* — how to turn a scored sample into a better sampling parameter — and the *level schedule* that keeps the interesting set from being empty.

```python
import numpy as np

# --- the parametric sampling family (sample + log-density), problem-specific ---
class SamplingFamily:
    """A family f(.; v): draw samples, evaluate the likelihood ratio f(.;u)/f(.;v)."""
    def sample(self, v, N, rng):
        raise NotImplementedError
    def log_likelihood_ratio(self, x, u, v):
        # log f(x; u) - log f(x; v); needed only for the estimation face
        raise NotImplementedError

def performance(x):
    """S(x): shortest-path length, cut weight, or a continuous objective."""
    raise NotImplementedError

# --- the unresolved core: turn a scored sample into a better sampling parameter ---
def refit_parameter(X, scores, gamma, u, v_prev, family):
    """Choose the next reference parameter from the current sample.

    TODO: pick v to bring the sampling density close to 'f restricted to {S >= gamma}'.
    The elite-selection rule and the closed-form update both go here.
    """
    pass

def level_update(scores, gamma_target, rho):
    """Pick the working level gamma_t so {S >= gamma_t} is not empty.

    TODO: keep the interesting set at a manageable rarity (~rho) and march it
    toward gamma_target / the optimum.
    """
    pass

def adaptive_loop(family, u, gamma_target, rho, N, rng):
    v = u
    while True:
        X = family.sample(v, N, rng)
        scores = np.array([performance(x) for x in X])
        gamma_t = level_update(scores, gamma_target, rho)   # TODO
        v = refit_parameter(X, scores, gamma_t, u, v, family)  # TODO
        # TODO: stopping rule; for estimation, a final reweighted average
        break
    return v
```
