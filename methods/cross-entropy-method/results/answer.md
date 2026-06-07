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

Rarity breaks a one-shot fit, so iterate a **two-phase multilevel loop**: set the working level γ_t to the (1−ρ)-sample-quantile of the current performances (so the elite has probability ≈ ρ and is never empty), refit v_t to that elite, and let γ_t ratchet up — toward γ (estimation) or toward the optimum (optimization). The **same scheme is both faces**: the likelihood ratio W stays for estimation (a specific law's tail must be reported); for optimization the associated problem is redefined around f(·;v_{t-1}) each step, so W = 1 and the update is plain elite-MLE.

**Closed forms by family** (elite set E_t = {i : S(X_i) ≥ γ_t}):
- Exponential f(x;v)=Π_j v_j⁻¹e^{−x_j/v_j}: v_{t,j} = Σ_i 1_{S≥γ_t} W_i X_{ij} / Σ_i 1_{S≥γ_t} W_i.
- Bernoulli f(x;p)=Π_j p_j^{x_j}(1−p_j)^{1−x_j}: p_{t,j} = Σ_i 1_{S≥γ_t} 1_{X_{ij}=1} / Σ_i 1_{S≥γ_t} (elite fraction of ones).
- Gaussian N(μ,σ²): μ_{t,j} = mean of elite X_{·j}, σ²_{t,j} = variance of elite X_{·j} (elite-sample mean and variance).

**Stabilizers.** Smoothed update v_t = α ŵ_t + (1−α) v_{t-1}, α ∈ [0.4, 0.9], prevents the distribution from snapping to a degenerate point (a Bernoulli p_j locking at 0/1, a Gaussian σ collapsing) too early. Sample size N = c·(#parameters) with c > 1; ρ ∈ [0.01, 0.1]. As a sibling estimation-of-distribution method, the Gaussian form is the cross-entropy cousin of CMA-ES: both keep elite samples and refit a Gaussian, but CE refits by the KL/MLE moment projection rather than an evolution-path covariance rule.

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

# ============================================================================
# (i) RARE-EVENT ESTIMATION: iterated-threshold tilt with likelihood ratios.
#     Family: independent exponential edge weights f(x; v) = prod_j (1/v_j) e^{-x_j/v_j}.
#     Target: ell = P_u(S(X) >= gamma), S = shortest A->B path length.
# ============================================================================

def shortest_path(x):
    # tiny bridge network, 5 edges: 0,1 upper; 2 cross; 3,4 lower. S = min route.
    return min(x[0] + x[3], x[1] + x[4], x[0] + x[2] + x[4], x[1] + x[2] + x[3])

def ce_rare_event(u, gamma, rho=0.1, N=2000, N_final=100_000, rng=None):
    rng = rng or np.random.default_rng(0)
    u = np.asarray(u, float)
    v = u.copy()                                            # v_0 := u (nominal law)
    while True:
        X = rng.exponential(scale=v, size=(N, len(v)))      # sample f(.; v_{t-1})
        S = np.array([shortest_path(x) for x in X])
        gamma_t = np.quantile(S, 1.0 - rho)                 # (1-rho)-quantile level
        reached = gamma_t >= gamma
        if reached:
            gamma_t = gamma
        elite = S >= gamma_t                                # the top-rho elite set
        # likelihood ratio W = f(.;u)/f(.;v): log W = sum (1/v-1/u) x - ln(u/v)
        logW = np.sum((1.0 / v - 1.0 / u) * X - np.log(u / v), axis=1)
        w = elite * np.exp(logW)                            # elite-and-LR weights
        v = (w[:, None] * X).sum(0) / w.sum()               # CE update: weighted elite mean
        if reached:
            break
    # final honest LR estimate under the learned tilt v_T
    X = rng.exponential(scale=v, size=(N_final, len(v)))
    S = np.array([shortest_path(x) for x in X])
    logW = np.sum((1.0 / v - 1.0 / u) * X - np.log(u / v), axis=1)
    ell = np.mean((S >= gamma) * np.exp(logW))              # (1/N) sum 1_{S>=gamma} W
    return ell, v

# ============================================================================
# (ii) CONTINUOUS OPTIMIZATION: Gaussian family, elite mean/variance refit.
#     max_x S(x); no privileged law, so W = 1 and the update is pure elite-MLE.
# ============================================================================

def ce_optimize(objective, mu, sigma, rho=0.1, N=100, alpha=0.7,
                n_iter=100, tol=1e-8, rng=None):
    rng = rng or np.random.default_rng(0)
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    n_elite = max(2, int(round(rho * N)))
    for _ in range(n_iter):
        X = mu + sigma * rng.standard_normal((N, len(mu)))  # sample f(.; mu, sigma)
        S = np.array([objective(x) for x in X])
        elite = X[np.argsort(S)[-n_elite:]]                 # top-rho by performance
        mu_hat = elite.mean(0)                              # elite-sample MEAN  (mu MLE)
        sig_hat = elite.std(0)                              # elite-sample STD   (sigma MLE)
        mu = alpha * mu_hat + (1 - alpha) * mu              # smoothed update
        sigma = alpha * sig_hat + (1 - alpha) * sigma
        if np.max(sigma) < tol:                             # Gaussian has collapsed
            break
    return mu, S.max()

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ell, v = ce_rare_event(u=[0.25, 0.4, 0.1, 0.3, 0.2], gamma=2.0, rng=rng)
    print(f"P(S >= 2) ~ {ell:.3e}   learned tilt v = {np.round(v, 3)}")

    f = lambda x: np.exp(-(x[0] - 2.0) ** 2) + 0.8 * np.exp(-(x[0] + 2.0) ** 2)
    mu, best = ce_optimize(f, mu=[0.0], sigma=[5.0], rng=rng)
    print(f"argmax ~ {mu[0]:.4f}   S ~ {best:.4f}")
```

The CE method turns "find the optimal importance-sampling density" — known in closed form (g* ∝ 1_{S≥γ} f) but intractable — into "minimize KL cross-entropy to it within a parametric family," which is weighted maximum likelihood over elite samples; iterating with a (1−ρ)-quantile level schedule makes rare-event estimation and optimization the same adaptive distribution-learning scheme.
