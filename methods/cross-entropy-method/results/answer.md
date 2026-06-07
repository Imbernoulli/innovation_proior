# The Cross-Entropy (CE) Method

## Problem

One adaptive scheme for two problems that share a structure:

- **Rare-event estimation.** Estimate ℓ = P_u(S(X) ≥ γ) = E_u[1_{S(X)≥γ}], where X ∼ f(·; u), when ℓ is tiny (10⁻⁵ or below) so the event {S ≥ γ} almost never occurs in a feasible simulation. Crude Monte-Carlo has relative error ≈ 1/√(Nℓ), forcing N ∼ 1/ℓ trials.
- **Optimization.** Find γ* = max_{x∈X} S(x) and a maximizer x*, when the near-optimal states are a microscopic fraction of X — combinatorial (TSP, max-cut, QAP) or continuous multi-extremal.

Both reduce to the same question: construct a sampling density concentrated on the rare/good set {S ≥ γ}, without waiting for that set to be hit by chance.

## Key idea

Importance sampling rewrites ℓ = E_g[1_{S≥γ} f(·;u)/g] for any sampling density g, unbiased for every g; g only affects variance. The variance-minimizing choice is the **zero-variance density**

    g*(x) = 1_{S(x)≥γ} f(x; u) / ℓ,

f restricted to {S ≥ γ} and renormalized — unusable because its normalizer is the unknown ℓ, but it dictates the target *shape*: live on {S ≥ γ}, look like f there.

Reach that shape inside a parametric family f(·; v) by choosing v to **minimize the Kullback–Leibler cross-entropy** D(g*, f(·;v)) = ∫ g* ln g* − ∫ g* ln f(·;v). The first term is constant in v, so

    min_v D(g*, f(·;v))  ⟺  max_v ∫ g*(x) ln f(x;v) dx  ⟺  max_v E_u[ 1_{S(X)≥γ} ln f(X;v) ].

This is a **weighted maximum-likelihood fit of f(·;v) to the elite samples** {S(X) ≥ γ}. Evaluated under any reference w and as a sample average,

    v̂ = argmax_v (1/N) Σ_i 1_{S(X_i)≥γ} W(X_i;u,w) ln f(X_i;v),   W(x;u,w) = f(x;u)/f(x;w),   X_i ∼ f(·;w),

whose stationarity Σ_i 1_{S(X_i)≥γ} W_i ∇_v ln f(X_i;v) = 0 has a closed form for natural exponential families (the parameter equals the elite-weighted sample moment).

Rarity breaks a one-shot fit, so iterate a **two-phase multilevel loop**: set the working level γ_t to the (1−ρ)-sample-quantile of the current performances (so the elite has probability ≈ ρ and is never empty), refit v_t to that elite, and move γ_t toward γ (estimation) or toward the optimum (optimization). The **same scheme is both faces**: the likelihood ratio W stays for estimation (a specific law's tail must be reported); for optimization the associated problem is redefined around f(·;v_{t-1}) each step, so W = 1 and the update is plain elite-MLE.

**Closed forms by family** (elite set E_t = {i : S(X_i) ≥ γ_t}):
- Exponential f(x;v)=Π_j v_j⁻¹e^{−x_j/v_j}: v_{t,j} = Σ_i 1_{S≥γ_t} W_i X_{ij} / Σ_i 1_{S≥γ_t} W_i.
- Bernoulli f(x;p)=Π_j p_j^{x_j}(1−p_j)^{1−x_j}: p_{t,j} = Σ_i 1_{S≥γ_t} 1_{X_{ij}=1} / Σ_i 1_{S≥γ_t} (elite fraction of ones).
- Gaussian N(μ,σ²): μ_{t,j} = mean of elite X_{·j}, σ²_{t,j} = variance of elite X_{·j} (elite-sample mean and variance).

**Stabilizers.** Smoothed update v_t = α ŵ_t + (1−α) v_{t-1}, α ∈ [0.4, 0.9], prevents the distribution from snapping to a degenerate point (a Bernoulli p_j locking at 0/1, a Gaussian σ collapsing) too early. Sample size N = c·(#parameters) with c > 1; ρ ∈ [0.01, 0.1].

## Algorithm

Two-phase CE loop (v̂₀ = u, t = 1):
1. **Sample.** Draw X₁,…,X_N ∼ f(·; v̂_{t-1}); compute S(X_i).
2. **Level.** γ̂_t = (1−ρ)-quantile = S_{(⌈(1−ρ)N⌉)}, capped at the target γ (estimation) or left free (optimization).
3. **Refit (CE / weighted MLE over the elite {S ≥ γ̂_t}).**
   - estimation: v̂_t solves Σ_i 1_{S(X_i)≥γ̂_t} W(X_i;u,v̂_{t-1}) ∇_v ln f(X_i;v) = 0;
   - optimization: same with W = 1; optionally smooth v̂_t ← α ŵ_t + (1−α) v̂_{t-1}.
4. **Iterate / stop.** Estimation: until γ̂_t = γ, then report ℓ̂ = (1/N₁) Σ_i 1_{S(X_i)≥γ} W(X_i;u,v̂_T). Optimization: until γ̂_t stalls for d≈5 iterations; the converged f(·;v̂_T) concentrates on the maximizer.

## Code

```python
import numpy as np

def shortest_path(x):
    # Tiny bridge network, 5 edges: 0,1 upper; 2 cross; 3,4 lower. S = min route.
    return min(x[0] + x[3], x[1] + x[4], x[0] + x[2] + x[4], x[1] + x[2] + x[3])

def exponential_log_likelihood_ratio(X, u, v):
    # log f(X; u) - log f(X; v) for independent exponentials with mean vectors u and v.
    X = np.asarray(X, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return np.sum((1.0 / v - 1.0 / u) * X - np.log(u / v), axis=1)

def choose_level(scores, target, rho):
    # Algorithmic (1-rho) order statistic, capped at the prescribed rare-event level.
    scores = np.asarray(scores, float)
    idx = int(np.ceil((1.0 - rho) * len(scores))) - 1
    idx = int(np.clip(idx, 0, len(scores) - 1))
    level = np.sort(scores)[idx]
    if target is not None and level >= target:
        return target, True
    return level, False

def refit_exponential_means(X, scores, level, u, v):
    # Weighted MLE: v_j = sum I{S>=level} W X_j / sum I{S>=level} W.
    elite = scores >= level
    weights = elite * np.exp(exponential_log_likelihood_ratio(X, u, v))
    total = weights.sum()
    if total == 0.0:
        return np.asarray(v, float).copy()
    return (weights[:, None] * X).sum(axis=0) / total

def adaptive_rare_event(u, gamma, rho=0.1, N=2000, N_final=100_000, rng=None):
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
    level, _ = choose_level(scores, target=None, rho=rho)
    elite = X[scores >= level]
    return elite.mean(axis=0), elite.std(axis=0)

def adaptive_optimize(objective, mu, sigma, rho=0.1, N=100, alpha=0.7,
                      n_iter=100, tol=1e-8, rng=None):
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
    ell, v = adaptive_rare_event(u=[0.25, 0.4, 0.1, 0.3, 0.2], gamma=2.0, rng=rng)
    print(f"P(S >= 2) ~ {ell:.3e}   learned tilt v = {np.round(v, 3)}")

    f = lambda x: np.exp(-(x[0] - 2.0) ** 2) + 0.8 * np.exp(-(x[0] + 2.0) ** 2)
    mu, best = adaptive_optimize(f, mu=[0.0], sigma=[5.0], rng=rng)
    print(f"argmax ~ {mu[0]:.4f}   S ~ {best:.4f}")
```

The CE method turns "find the optimal importance-sampling density" — known in closed form (g* ∝ 1_{S≥γ} f) but intractable — into "minimize KL cross-entropy to it within a parametric family," which is weighted maximum likelihood over elite samples; iterating with a (1−ρ)-quantile level schedule makes rare-event estimation and optimization the same adaptive distribution-learning scheme.
