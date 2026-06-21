The two problems on the table, rare-event estimation and hard optimization, turn out to be the same puzzle in different clothes. In rare-event estimation we want ℓ = P_u(S(X) ≥ γ) for a performance S and a level γ so high that the event essentially never occurs in a feasible simulation; crude Monte Carlo needs on the order of 1/ℓ samples to keep its relative error small, which is already billions of runs for probabilities near 10⁻⁵. In optimization we want γ* = max_x S(x), but the near-optimal states are a microscopic fraction of a combinatorial or multi-extremal continuous space, so blind random search almost never lands on them. The common failure is that the configurations that matter are vanishingly rare under any naive sampling law. Existing fixes have their own weaknesses. Static importance sampling with a hand-picked tilt can cut variance when the tilt is right, but finding that tilt analytically is usually impossible and a bad tilt can make variance worse than crude Monte Carlo. Direct variance minimization of the reference parameter targets the right objective, yet it is a non-convex numerical program with no closed-form update. Randomized search heuristics such as simulated annealing or genetic algorithms are useful bundles of operators, but their update rules are hand-designed rather than derived from a single principle.

The method that addresses this cleanly is the Cross-Entropy Method. It is an adaptive sampling scheme that learns a parametric distribution f(·; v) whose mass concentrates on the important set {S ≥ γ}. The starting observation is that the variance-minimizing importance-sampling density is g*(x) = 1_{S(x)≥γ} f(x; u)/ℓ, the nominal law restricted to the rare set and renormalized. This density has zero variance, but it is unusable because its normalizer is exactly the unknown ℓ. So it is treated not as a recipe but as a target shape: a good sampler should live on {S ≥ γ} and look like f there. The Cross-Entropy Method reaches this shape by choosing v to minimize the Kullback-Leibler divergence, equivalently the cross-entropy, from g* to f(·; v). The entropy term of the KL divergence does not involve v, so minimizing cross-entropy reduces to maximizing E_u[1_{S(X)≥γ} ln f(X; v)], which is a weighted maximum-likelihood fit of f(·; v) to the elite samples that satisfy S(X) ≥ γ. For natural exponential families this weighted score equation has a closed-form solution: the new parameter is an elite-weighted sample moment.

Because the target level γ is too extreme for a single fit, the algorithm uses a multilevel bootstrap. At each iteration it draws a batch from the current law f(·; v_{t-1}), orders the observed scores, and sets the working level γ_t to the (1 − ρ)-quantile of those scores, capped at the final target γ in estimation. This guarantees that the elite set has probability approximately ρ under the current sampler and is therefore never empty. It then refits v_t to the elite samples. For rare-event estimation the likelihood ratio W(x; u, v_{t-1}) = f(x; u)/f(x; v_{t-1}) must remain in the weighted update and in the final estimate, because the tail probability being reported belongs to the fixed nominal law f(·; u). For optimization there is no privileged nominal law; the associated stochastic problem is redefined around f(·; v_{t-1}) each iteration, so the likelihood ratio becomes identically one and the update is pure elite maximum likelihood. The same two-phase loop, sample then refit to the quantile-defined elite, serves both problems. Closed-form cases are particularly useful: independent exponentials update each mean to the elite-weighted average of that coordinate; independent Bernoullis update each success probability to the fraction of elite samples with a one in that coordinate; and a diagonal Gaussian updates its mean and variance to the elite-sample mean and variance.

One practical concern is premature collapse. A Gaussian variance that shrinks too fast can freeze the search at a local optimum, and a Bernoulli probability that snaps to zero or one can lock a coordinate forever. The standard remedy is smoothing: after computing the raw elite-MLE update ŵ_t, set v_t = α ŵ_t + (1 − α) v_{t-1} with α in roughly [0.4, 0.9]. This keeps the distribution from degenerating too early while still allowing convergence. Typical settings use ρ between 0.01 and 0.1 and a sample size N that is a small multiple of the number of parameters. A steep weighting function over the elite can be used but is usually avoided because a few lucky samples can dominate and increase local-optimum risk.

```python
import numpy as np


def shortest_path(x):
    """Tiny bridge network with 5 edges; S is the length of the shortest A->B route."""
    return min(
        x[0] + x[3],
        x[1] + x[4],
        x[0] + x[2] + x[4],
        x[1] + x[2] + x[3],
    )


def exponential_log_likelihood_ratio(X, u, v):
    """log f(X; u) - log f(X; v) for independent exponential means u and v."""
    X = np.asarray(X, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return np.sum((1.0 / v - 1.0 / u) * X - np.log(u / v), axis=1)


def choose_level(scores, target, rho):
    """Set working level to the (1-rho)-quantile, capped at the target."""
    scores = np.asarray(scores, float)
    idx = int(np.ceil((1.0 - rho) * len(scores))) - 1
    idx = int(np.clip(idx, 0, len(scores) - 1))
    level = np.sort(scores)[idx]
    if target is not None and level >= target:
        return target, True
    return level, False


def refit_exponential_means(X, scores, level, u, v):
    """Weighted MLE: v_j = sum elite W X_j / sum elite W."""
    elite = scores >= level
    weights = elite * np.exp(exponential_log_likelihood_ratio(X, u, v))
    total = weights.sum()
    if total == 0.0:
        return np.asarray(v, float).copy()
    return (weights[:, None] * X).sum(axis=0) / total


def adaptive_rare_event(u, gamma, rho=0.1, N=2000, N_final=100_000, rng=None):
    """Estimate P_u(S(X) >= gamma) for the bridge network."""
    rng = np.random.default_rng(0) if rng is None else rng
    u = np.asarray(u, float)
    v = u.copy()
    while True:
        X = rng.exponential(scale=v, size=(N, len(v)))
        scores = np.array([shortest_path(x) for x in X])
        level, reached = choose_level(scores, gamma, rho)
        v = refit_exponential_means(X, scores, level, u, v)
        if reached:
            break
    X = rng.exponential(scale=v, size=(N_final, len(v)))
    scores = np.array([shortest_path(x) for x in X])
    ell = np.mean((scores >= gamma) * np.exp(exponential_log_likelihood_ratio(X, u, v)))
    return ell, v


def refit_gaussian(X, scores, rho):
    """Fit a diagonal Gaussian to the elite samples."""
    level, _ = choose_level(scores, target=None, rho=rho)
    elite = X[scores >= level]
    return elite.mean(axis=0), elite.std(axis=0)


def adaptive_optimize(objective, mu, sigma, rho=0.1, N=100, alpha=0.7,
                      n_iter=100, tol=1e-8, rng=None):
    """Maximize a continuous objective via the cross-entropy method."""
    rng = np.random.default_rng(0) if rng is None else rng
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    best_score = -np.inf
    for _ in range(n_iter):
        X = mu + sigma * rng.standard_normal((N, len(mu)))
        scores = np.array([objective(x) for x in X])
        best_score = max(best_score, float(scores.max()))
        mu_hat, sig_hat = refit_gaussian(X, scores, rho)
        mu = alpha * mu_hat + (1 - alpha) * mu
        sigma = alpha * sig_hat + (1 - alpha) * sigma
        if np.max(sigma) < tol:
            break
    return mu, best_score


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ell, v = adaptive_rare_event(
        u=[0.25, 0.4, 0.1, 0.3, 0.2], gamma=2.0, rng=rng
    )
    print(f"P(S >= 2) ~ {ell:.3e}   learned tilt v = {np.round(v, 3)}")

    f = lambda x: np.exp(-(x[0] - 2.0) ** 2) + 0.8 * np.exp(-(x[0] + 2.0) ** 2)
    mu, best = adaptive_optimize(f, mu=[0.0], sigma=[5.0], rng=rng)
    print(f"argmax ~ {mu[0]:.4f}   S ~ {best:.4f}")
```
