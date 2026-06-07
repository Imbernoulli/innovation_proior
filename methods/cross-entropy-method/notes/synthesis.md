# CE method — synthesis

## Pain point / problem
- Two faces, one machine:
  (i) rare-event estimation: ℓ = P_u(S(X) ≥ γ) = E_u[1_{S(X)≥γ}], ℓ tiny (≤1e-5).
      Crude MC relative error ≈ 1/√(Nℓ) → ~100/ℓ trials. Need adaptive IS.
  (ii) optimization: max_x S(x) = γ*. Cast as ASP: ℓ(γ)=P_u(S(X)≥γ); as γ→γ*, the
      optimal IS density concentrates on the optimizer(s).
- Both reduce to: find a sampling density that concentrates on {S≥γ}.

## Background / ancestors (cite these in-frame as prior art)
- Importance sampling change of measure: ℓ = E_g[1_{S≥γ} f(X;u)/g(X)], W=f/g likelihood ratio.
  Unbiased for any g with g>0 on {1·f≠0}.
- Optimal / zero-variance IS density: g*(x) = 1_{S(x)≥γ} f(x;u)/ℓ. Cauchy–Schwarz / variational
  minimization of Var. Gives Var=0. Intractable: normalizer = ℓ (the unknown).
- Kullback–Leibler divergence D(g,h)=E_g[ln g/h] = ∫g ln g − ∫g ln h. Not symmetric.
  "Cross-entropy" = the −∫ g ln h term (the part depending on h).
- MLE / natural exponential families (NEF): argmax_v Σ ln f(x_i;v) → for NEF, sample-mean
  type closed form. Gaussian: μ=mean, σ²=var. Bernoulli: p=fraction of 1s. Exponential: v=mean.

## The intellectual move (the heart)
1. Want g close to g*. Pick parametric family {f(·;v)} (same family as nominal f(·;u)).
2. Choose v minimizing D(g*, f(·;v)) = KL cross-entropy to the optimal density.
   min_v D = min_v [−∫ g* ln f(·;v)] (the ∫g* ln g* term is const in v)
   = max_v ∫ g*(x) ln f(x;v) dx
   = max_v ∫ [1_{S(x)≥γ} f(x;u)/ℓ] ln f(x;v) dx        (sub g*)
   = max_v E_u[1_{S(X)≥γ} ln f(X;v)]                    (drop 1/ℓ, const>0)   ... (D(v))
3. This is a WEIGHTED MAXIMUM-LIKELIHOOD: maximize log-likelihood ln f(X;v) over the
   sub-population {S(X)≥γ} (the "elite"). The CE-optimal v is the MLE fit to the elite set.
4. IS to evaluate the expectation under any w: max_v E_w[1_{S≥γ} W(X;u,w) ln f(X;v)],
   W=f(·;u)/f(·;w). Stochastic counterpart:
   max_v (1/N) Σ 1_{S(X_i)≥γ} W(X_i;u,w) ln f(X_i;v),  X_i~f(·;w).
   Solve ∇_v: Σ 1_{S(X_i)≥γ} W_i ∇_v ln f(X_i;v) = 0.
5. NEF closed form: for exponential f, the stationary equation is a W-and-indicator-weighted
   mean. E.g. exponential-means example: v_j = Σ 1·W·X_ij / Σ 1·W. (eq 24)

## The wall (rarity) → multilevel
- With w=u, if γ is the true rare level, almost all indicators are 0 → (16)/(17) void.
- Fix: bootstrap γ. Build sequences {v_t},{γ_t}. Each iter:
  (a) γ_t = sample (1−ρ)-quantile of S under f(·;v_{t-1}) (ρ ~ 0.01–0.1), capped at γ.
      So {S≥γ_t} has prob ≈ ρ, never void.
  (b) v_t = CE/weighted-MLE solution of (22) using same sample, with W=f(·;u)/f(·;v_{t-1}).
- Raise γ_t toward γ; when γ_t reaches γ, stop; final LR estimate
  ℓ̂ = (1/N1) Σ 1_{S≥γ} W(X_i;u,v_T).

## Optimization face (the same scheme, W dropped)
- ASP redefined each iter around f(·;v_{t-1}), not f(·;u) → W=1 (Remark 2.4). Initial u arbitrary.
- Algorithm 2.3: sample → γ_t=(1−ρ)-quantile → elite={S(X_i)≥γ_t} → v_t = MLE over elite.
- Stop when γ_t stalls for d iters (d~5). f(·;v_t) → Dirac at optimizer.
- Smoothed update v_t = α ŵ_t + (1−α) v_{t-1}, α∈[0.4,0.9] (prevents premature 0/1 lock-in
  for discrete; for continuous σ→0 too fast).

## Concrete families (closed forms — verify each)
- Bernoulli (binary COP, max-cut): p_{t,j} = Σ 1_{S≥γ_t} 1_{X_ij=1} / Σ 1_{S≥γ_t}
  = fraction of elite with bit j = 1.  (∂_p ln f = (X_j−p)/[(1−p)p])
- Gaussian continuous opt: f=N(μ,σ²). MLE over elite:
  ∂_μ ln f=(x−μ)/σ² → μ_t = (1/|E|) Σ_{i∈E} X_i  (elite mean)
  ∂_{σ²} ln f = −1/(2σ²)+(x−μ)²/(2σ⁴) → σ²_t = (1/|E|) Σ_{i∈E}(X_i−μ_t)²  (elite variance)
  (with W-weights for the rare-event Gaussian-tail case; W=1 for opt)
- Exponential (rare-event toy, shortest-path): v_j = Σ 1·W·X_ij / Σ 1·W (elite W-weighted mean).

## Alternative objectives / variance-min comparison
- Variance minimization (Rubinstein 1997 origin): min_v E_w[1_{S≥γ} W(·;u,v) W(·;u,w)] (eq 27-29)
  — needs numerical opt, no closed form. CE (KL) gives closed-form NEF updates → preferred.
- ϕ(s;γ)=ψ(s)1_{s≥γ}, ψ(s)=s mild speedup; high powers/Boltzmann → local minima.

## CMA-ES / EDA kinship (third-party, for answer.md context, in-frame)
- CEM is an estimation-of-distribution algorithm: maintain Gaussian, sample, keep elite,
  refit mean+(co)variance. CMA-ES is the sibling that adds evolution-path covariance adaptation.
  CEM = elite-sample mean/cov refit (the KL/MLE projection); same family of methods.

## Sources retrieved this run
- PRIMARY surrogate (same authors, full canonical reference): De Boer, Kroese, Mannor,
  Rubinstein 2005 "A Tutorial on the CE Method" (Kroese's site PDF). Covers Rubinstein 1997
  (rare-event/variance-min origin) and 1999 (combinatorial+continuous opt) — both cited inside.
- Rubinstein 1997 (EJOR 99:89-112) and 1999 (Methodol Comput Appl Probab 1:127-190):
  PAYWALLED (Elsevier/Springer), full text not obtained — flagged. Content reconstructed from
  the 2005 tutorial which is by the same authors and reproduces their equations.
- Background: IS/zero-var density, KL, MLE/NEF — from tutorial §2.1 + standard.
- Third-party: Wikipedia CE method (Gaussian elite mean/var pseudocode); CEM/CMA-ES/EDA
  kinship (GACEM arXiv 2002.07236 summary, simulation-opt review).
