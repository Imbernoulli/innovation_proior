I have two problems on my desk that look like they belong to different worlds, and I want to understand why they keep rhyming. One is estimation: a system with input X ∼ f(·; u), a performance S(X), a level γ, and I need ℓ = P_u(S(X) ≥ γ) = E_u[1_{S(X)≥γ}], a probability down at 10⁻⁵ or below. The other is optimization: maximize S over some space X, find γ* = max_x S(x) and a maximizer x*. The estimation one is hard because the event {S ≥ γ} never happens in a feasible run; the optimization one is hard because the near-optimal states are a microscopic fraction of X and blind search never lands on them. Let me start from the estimation problem, because there the failure of the naive approach is something I can write down exactly, and then see whether anything I learn transfers.

Crude Monte-Carlo: draw X₁,…,X_N from f(·; u), average the indicator 1_{S(X_i)≥γ}. Unbiased, variance ℓ(1−ℓ)/N. But the absolute error of a 10⁻⁵ number is meaningless; what I care about is relative error, which is √(ℓ(1−ℓ)/N)/ℓ ≈ 1/√(Nℓ). To hold that at 1% I need N ≈ 10⁴/ℓ — for ℓ = 10⁻⁵ that is 10⁹ runs, and it only gets worse as the event gets rarer. Growing N buys me √N and nothing more. I cannot out-sample a 1/ℓ problem. The waste is obvious: almost every draw lands outside the rare set and contributes a zero. I'm spending all my computation confirming that nothing happened. I need to sample *more often* in the region that matters and correct for it.

The correction is exact, and it's the only lever I have. ℓ = ∫ 1_{S(x)≥γ} f(x; u) dx. Multiply and divide by any other density g that doesn't vanish where the integrand is nonzero:

  ℓ = ∫ 1_{S(x)≥γ} [f(x; u)/g(x)] g(x) dx = E_g[ 1_{S(X)≥γ} W(X) ],  W(x) = f(x; u)/g(x).

So if I draw from g and reweight each sample by the likelihood ratio W = f/g, the average ℓ̂ = (1/N) Σ 1_{S(X_i)≥γ} W(X_i) is unbiased for *any* admissible g — g cancels in expectation, valid because wherever the indicator times f is nonzero I required g > 0. The choice of g changes nothing about the bias; it changes only the variance. So the entire game is: pick g to crush the variance.

Which g? Let me write the second moment and let it tell me. Under g, each term is T = 1_{S≥γ} f/g, and E_g[T²] = ∫_{S≥γ} f²/g dx. So Var_g(T) = ∫_{S≥γ} f²/g − ℓ². I can complete the square: ∫_{S≥γ} f²/g − ℓ² = ∫ (1_{S≥γ} f − ℓ g)²/g dx (expand the numerator: (1_{S≥γ}f)² − 2ℓ g 1_{S≥γ}f + ℓ²g², over g and integrated, gives ∫_{S≥γ} f²/g − 2ℓ·ℓ + ℓ²). The integrand is a square over g, non-negative, and exactly zero when 1_{S(x)≥γ} f(x; u) − ℓ g(x) = 0 for all x, i.e. g(x) ∝ 1_{S(x)≥γ} f(x; u). Normalizing — the integral of 1_{S≥γ}f is ℓ itself — the constant is 1/ℓ:

  g*(x) = 1_{S(x)≥γ} f(x; u) / ℓ.

That's f restricted to the rare set and renormalized, and it gives zero variance: under g*, 1_{S≥γ} f/g* = ℓ identically, one sample computes ℓ exactly. Beautiful, and immediately useless, because to normalize g* I divided by ℓ — the very number I'm trying to find. g* presupposes the answer. Circular.

But it is not useless as a *shape*. It tells me precisely what a good sampler wants to be: it lives on {S ≥ γ}, and within that set it is shaped like f. So I want a density I *can* sample from that imitates this shape. The cheapest move is to stay inside the same parametric family — instead of the nominal f(·; u), use f(·; v) for some other reference parameter v, so that the likelihood ratio W(x; u, v) = f(x; u)/f(x; v) is a closed-form function of v and the family is one I already know how to draw from. Now the question sharpens to: which v makes f(·; v) closest to g*?

And here "closest" has to be made precise. I want a measure of distance between g* and f(·; v) that I can actually minimize. The Kullback–Leibler divergence is the natural one when one of the two densities is the thing I'm fitting:

  D(g*, f(·; v)) = E_{g*}[ ln(g*(X)/f(X; v)) ] = ∫ g* ln g* dx − ∫ g* ln f(·; v) dx.

The first term doesn't involve v at all — it's the entropy of g*, a constant as far as the fit is concerned. So minimizing D over v is the same as *maximizing* the second term, the cross-entropy piece:

  max_v ∫ g*(x) ln f(x; v) dx.

Now substitute g* = 1_{S≥γ} f(·; u)/ℓ. The 1/ℓ is a positive constant, irrelevant to the argmax, so it drops:

  max_v ∫ 1_{S(x)≥γ} f(x; u) ln f(x; v) dx = max_v E_u[ 1_{S(X)≥γ} ln f(X; v) ].

Stare at that. ∫ over x, weighted by f(·; u), of 1_{S≥γ} times ln f(x; v). This is a *weighted maximum-likelihood problem*: I am maximizing the expected log-likelihood ln f(X; v), but only over the part of the population where S(X) ≥ γ — the "elite" states. The KL-to-the-optimal-density objective has collapsed into "fit f(·; v) by maximum likelihood to the elite samples." Choosing the importance-sampling parameter to minimize cross-entropy to the unreachable zero-variance density is the same as doing maximum likelihood on the subpopulation that hits the rare set.

Why does that feel right? Maximum likelihood pulls the fitted density toward where the data it's fitted on lives. Restricting the data to {S ≥ γ} pulls f(·; v) toward the rare set — exactly the "live on {S ≥ γ}" half of what g* wanted — while keeping it inside the family that has f's shape, which is the other half. So the KL projection of g* onto the family does, by construction, the two things the shape demanded.

Let me make it operational. The expectation E_u is itself under the nominal law, which is the thing I can't sample the rare set from — so I import importance sampling again, one level up. For any reference w,

  max_v E_w[ 1_{S(X)≥γ} W(X; u, w) ln f(X; v) ],  W(x; u, w) = f(x; u)/f(x; w),

and its sample version with X₁,…,X_N ∼ f(·; w),

  max_v D̂(v) = max_v (1/N) Σ_i 1_{S(X_i)≥γ} W(X_i; u, w) ln f(X_i; v).

Differentiate in v and set to zero:

  (1/N) Σ_i 1_{S(X_i)≥γ} W(X_i; u, w) ∇_v ln f(X_i; v) = 0.

So the update is a *weighted score equation*: each elite sample contributes its score ∇_v ln f, weighted by its likelihood ratio. The whole appeal of doing this in the KL / maximum-likelihood form rather than minimizing the IS variance directly is right here: for nice families this equation has a closed-form solution, whereas the variance-minimization program min_v E_w[1_{S≥γ} W(X;u,v) W(X;u,w)] is a non-convex numerical optimization with no sample-moment update. Same goal — concentrate the sampler on the rare set — but the cross-entropy route gives an analytic update.

Let me see the closed form fall out for a concrete family. Take the network with independent exponential edge weights, f(x; v) = Π_j (1/v_j) e^{−x_j/v_j}, so ln f(x; v) = Σ_j (−x_j/v_j − ln v_j) and ∂_{v_j} ln f = x_j/v_j² − 1/v_j. The j-th score equation becomes

  Σ_i 1_{S(X_i)≥γ} W(X_i; u, w) (X_{ij}/v_j² − 1/v_j) = 0
  ⟹  v_j = [ Σ_i 1_{S(X_i)≥γ} W(X_i; u, w) X_{ij} ] / [ Σ_i 1_{S(X_i)≥γ} W(X_i; u, w) ].

A weighted average of the j-th coordinate over the elite samples. The new mean of edge j is just the elite-and-likelihood-weighted empirical mean of edge j. That's the update — no inner optimization, a single pass over the sample. And the same thing happens for any natural exponential family: the score is linear in a sufficient statistic, so the stationary equation is "set the model's parameter to the elite-weighted sample moment."

There's still the wall I haven't dealt with: rarity. If I set w = u and γ to the true rare level, almost every indicator 1_{S(X_i)≥γ} is zero, so both D̂(v) and its gradient are essentially zero — the update is computed from a handful of, or zero, elite samples and is meaningless. The very rareness that motivated all this breaks the single-shot fit.

So don't aim at γ in one shot. Bootstrap the level. I'll build two sequences at once — reference parameters v₀, v₁, … and working levels γ₁, γ₂, … — and march the level toward γ while the parameter chases it. Pick a rarity fraction ρ, not too small, say 0.01 to 0.1. At iteration t, having v_{t-1}, draw a sample from f(·; v_{t-1}) and set the working level γ_t to the (1−ρ)-quantile of the sampled performances: order S_{(1)} ≤ … ≤ S_{(N)} and take γ_t = S_{(⌈(1−ρ)N⌉)} (capped at γ — once the quantile would exceed γ, use γ). By construction {S ≥ γ_t} has probability about ρ under the current law, so the top ρ fraction of this very sample *are* the elite — never empty, never void. Then refit v_t by the weighted-MLE update above, using the same sample, with the likelihood ratio W(X_i; u, v_{t-1}) because I'm sampling under v_{t-1} but still estimating the property of the nominal law f(·; u).

Why should the level move upward? Because v_t is fitted to push mass onto {S ≥ γ_t}, the next sample drawn from f(·; v_t) should have its performance distribution shifted upward, so its (1−ρ)-quantile tends to advance even though finite samples can wiggle. Each iteration the elite set is a moving ρ-quantile aimed toward γ. When γ_t finally reaches γ, the loop has a reference parameter v_T under which {S ≥ γ} is no longer rare, and I spend a final batch on the honest LR estimate

  ℓ̂ = (1/N₁) Σ_i 1_{S(X_i)≥γ} W(X_i; u, v_T),  X_i ∼ f(·; v_T).

So the rare-event estimate comes out of the *last* distribution; the earlier iterations exist only to *find* a good v_T by chasing the level up. The variance-minimization origin of all this — picking v to minimize the IS estimator's second moment — is the same destination, but the cross-entropy detour is what made each step a closed-form weighted mean instead of a numerical search.

Now back to the optimization problem, and the rhyme I started with. Maximizing S over X — turn it into a family of rare-event problems. Define, for each level γ, the associated stochastic problem ℓ(γ) = E_u[1_{S(X)≥γ}] under some initial law f(·; u). As γ rises toward γ* = max_x S(x), the set {S ≥ γ} shrinks down onto the optimizer(s) x*, and the density g* ∝ 1_{S≥γ} f, the one that puts all its mass there, becomes concentrated on x*. So *finding the optimum is finding the limiting concentrating density* — and I already have a machine that keeps refitting toward {S ≥ γ_t} while choosing γ_t from the current upper quantile. Run the very same loop, but now don't stop at a prescribed γ; keep pushing the quantile until it stalls, and read off the state where f(·; v_t) has piled up its mass.

One thing genuinely changes, and it's worth being careful about. In the estimation problem, u was sacred — it was the law whose tail probability I had to report, so the likelihood ratio W(·; u, v_{t-1}) had to stay in the update to keep me honest about *that* law. In the optimization problem there is no privileged law: I only care about *where* S is large, not about any particular f. So the associated stochastic problem can be redefined every iteration around the current f(·; v_{t-1}) rather than a fixed f(·; u). When I do that, the "nominal" law for iteration t *is* f(·; v_{t-1}), the same law I'm sampling from, so W(·; v_{t-1}, v_{t-1}) = 1 and the likelihood ratio drops out entirely. The initial u becomes arbitrary — just a starting distribution — and the update is pure elite-sample maximum likelihood:

  v_t = argmax_v Σ_{i : S(X_i) ≥ γ_t} ln f(X_i; v).

That's it: at each step, fit the family by ordinary MLE to the elite samples, the top ρ fraction by performance. The estimation and optimization algorithms are the same two-phase loop — update γ_t to the (1−ρ)-quantile, update v_t by weighted MLE over the elite — differing only in whether the likelihood-ratio weight W is present (estimation) or equals one (optimization).

Let me get the closed forms for the families I'd actually optimize over. Binary problems first — max-cut, subset selection, anything coded as x ∈ {0,1}ⁿ. The natural family is independent Bernoulli per coordinate, f(x; p) = Π_j p_j^{x_j}(1−p_j)^{1−x_j}, so ∂_{p_j} ln f = (x_j − p_j)/[p_j(1−p_j)]. The elite-MLE stationary equation Σ_{elite} (X_{ij} − p_j) = 0 gives

  p_{t,j} = [ Σ_i 1_{S(X_i)≥γ_t} 1_{X_{ij}=1} ] / [ Σ_i 1_{S(X_i)≥γ_t} ],

the fraction of elite samples whose j-th bit is 1. Dead simple: to update the success probability of coordinate j, count how many of the high-performing samples had a 1 there, and normalize. Repeat, and the p_j's drift toward 0 or 1 — the distribution degenerates onto a single binary vector, the candidate optimum.

Now continuous optimization, which is where I want the Gaussian. Sample X ∈ ℝᵈ from a (say diagonal) Gaussian, coordinate j ∼ N(μ_j, σ_j²). The log-density of one coordinate is ln f = −(x − μ)²/(2σ²) − ½ ln(2πσ²), with scores ∂_μ ln f = (x − μ)/σ² and ∂_{σ²} ln f = −1/(2σ²) + (x − μ)²/(2σ⁴). Elite-MLE: setting Σ_{elite}(X_i − μ)/σ² = 0 gives

  μ_{t,j} = (1/|E_t|) Σ_{i ∈ E_t} X_{ij},

the elite-sample mean; and setting Σ_{elite}[−1/(2σ²) + (X_i − μ)²/(2σ⁴)] = 0, i.e. Σ_{elite}(X_i − μ)² = |E_t| σ², gives

  σ²_{t,j} = (1/|E_t|) Σ_{i ∈ E_t} (X_{ij} − μ_{t,j})²,

the elite-sample variance, where E_t is the elite set {i : S(X_i) ≥ γ_t}. So the continuous CE update is exactly: take the top-ρ samples by objective value, set the new sampling Gaussian's mean and variance to the *mean and variance of those elite samples*. Sample, keep the best, refit a Gaussian to the best, repeat. The mean walks toward the optimum and the variance shrinks; in the limit the Gaussian collapses to a point mass at the maximizer. Each refit is the KL projection of "f restricted to {S ≥ γ_t}" onto the Gaussian family — which is why moment-matching the elite is the *right* refit and not just a plausible heuristic: it is literally the cross-entropy-minimizing parameter.

This is the same lineage as the population-and-Gaussian search methods — maintain a sampling distribution, draw a population, keep the fittest, move the distribution toward them. What the cross-entropy derivation supplies that a bare heuristic doesn't is the *reason* the update is mean-and-variance-of-the-elite specifically: it's the maximum-likelihood / minimum-cross-entropy fit to the optimal concentrating density, not a tuned recombination rule.

Two practical dangers I should guard against before I write code. First, the Gaussian variance can collapse too fast: a couple of lucky elite samples bunched together set σ² tiny, the sampler stops exploring, and the search freezes at a local maximum — the same way the Bernoulli p_j's can snap to 0 or 1 and stay stuck there forever, since once p_j = 0 no sample ever again has a 1 in coordinate j to pull it back. The cure is to not take the raw refit but to *smooth* it: blend the new fit with the old parameter,

  v_t = α ŵ_t + (1 − α) v_{t-1},

with ŵ_t the elite-MLE update and α a smoothing constant somewhere around 0.4 to 0.9. With α < 1 the parameters approach 0/1 (or σ → 0) only in the limit, never snapping, so a coordinate that looks decided early can still recover. For α = 1 this is the raw rule.

Second danger, on the estimation side: a badly chosen reference v can make the likelihood ratio W = f(·; u)/f(·; v) explode on the few samples that matter, blowing up the variance — the classic importance-sampling failure where the sampling density has lighter effective tails than the nominal one where the integrand lives. The multilevel ρ-quantile schedule is itself the safeguard: by only ever moving v to fit a set that currently has probability ≈ ρ under v_{t-1}, I never leap the distribution so far that it stops covering the region it just came from. Each step is a moderate change of measure, and the weights stay tame.

One more knob to set deliberately rather than by accident: ρ and the sample size N. ρ governs how aggressively the level climbs — too large (close to 1) and γ_t barely moves each iteration, wasting iterations; too small and the elite set is a handful of samples and the refit is noisy. Around 0.01–0.1 is the working range, larger ρ when the parameter vector is short. N has to be large enough to estimate the number of parameters reliably — for d parameters, N = c·d with c a small constant above 1 — because each refit is fitting that many numbers from the elite fraction ρN of the sample. I can replace the indicator 1_{S≥γ} by a gentle positive weight ψ(S)·1_{S≥γ} that favors better elite samples when scores are positive, but very steep monotone weights can let a few lucky samples dominate and increase local-optimum risk, so the plain indicator is the safer default.

Let me assemble the whole thing as one machine with two faces.

```python
import numpy as np

def shortest_path(x):
    # tiny bridge network with 5 edges; S = min over the two A->B routes.
    # edges: 0,1 (upper), 2 (cross), 3,4 (lower)
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
    # rare event: probability the shortest path exceeds gamma, nominal means u
    ell, v = adaptive_rare_event(u=[0.25, 0.4, 0.1, 0.3, 0.2], gamma=2.0, rng=rng)
    print(f"P(S >= 2) ~ {ell:.3e}   learned tilt v = {np.round(v, 3)}")

    # continuous optimization of a two-bump objective; global max near x = +2
    f = lambda x: np.exp(-(x[0] - 2.0) ** 2) + 0.8 * np.exp(-(x[0] + 2.0) ** 2)
    mu, best = adaptive_optimize(f, mu=[0.0], sigma=[5.0], rng=rng)
    print(f"argmax ~ {mu[0]:.4f}   S ~ {best:.4f}")
```

The causal chain, end to end: the quantity I need — a tail probability, or the location of a maximum — lives in a set that is far too rare for naive sampling to reach, so I rewrite the tail integral under a chosen sampling density f(·; v) and reweight by the likelihood ratio W = f(·; u)/f(·; v), unbiased for any v so v only affects variance; the variance-minimizing density g* ∝ 1_{S≥γ} f has zero variance but its normalizer is the unknown answer, so it's only a *target shape* — live on the rare set, look like f there — and I reach for it by minimizing the Kullback–Leibler cross-entropy from g* to f(·; v), which collapses into maximum-likelihood fitting of f(·; v) to the elite samples {S ≥ γ}, a closed-form elite-weighted moment update for any exponential family; the rarity that breaks a one-shot fit is handled by a two-phase multilevel loop that sets the working level γ_t to the (1−ρ)-quantile of the current sample so the elite is never empty and refits v to that elite, moving γ_t toward γ (estimation) or toward the optimum (optimization); the same loop serves both faces, with the likelihood-ratio weight W present when a particular law's tail must be reported and equal to one when only the location of the maximum matters — Bernoulli giving the elite-fraction-of-ones update, Gaussian giving the elite mean and variance — smoothed by v_t = α ŵ_t + (1−α) v_{t-1} to keep the distribution from collapsing onto a local optimum too soon.
