# EDM synthesis notes

## Pain point / research question
Diffusion-based generative models circa 2021–2022 (DDPM, NCSN/SMLD, score-SDE VP/VE, iDDPM, ADM, DDIM) are each derived from a different theoretical scaffold (discrete Markov chain ELBO, score matching at multiple noise scales, continuous SDEs). The derivations couple together choices that are logically independent: the noise schedule σ(t), the scaling s(t), the time discretization {t_i}, the ODE/SDE solver, the network's input/output parameterization, the loss weighting, and the noise distribution during training. As written, a method "looks like" a tightly bound package and you can't change one knob without seeming to break the theory. Goal: strip everything to ONE continuous-time object (the probability-flow ODE with σ as the time variable), expose every degree of freedom, then ask component-by-component what is actually optimal — independently of which "framework" a model was born in.

## Load-bearing ancestors
- **Sohl-Dickstein 2015 / Ho 2020 (DDPM):** forward Markov chain q(x_t|x_{t-1})=N(√(1−β_t)x_{t-1}, β_t I); closed form q(x_t|x_0)=N(√ᾱ_t x_0,(1−ᾱ_t)I), ᾱ_t=∏(1−β_s). Train ε-prediction with L_simple = ||ε_θ(x_t,t)−ε||². Variance-preserving: signal is scaled down as noise is added so total variance ≈ const. Sampling = ancestral, T=1000 steps. Gap: hundreds–thousands of NFE; schedule baked into the chain.
- **Song & Ermon 2019 (NCSN/SMLD):** estimate ∇log p_σ(x) at a geometric ladder of σ's via denoising score matching; sample with annealed Langevin. Variance-exploding: data unscaled, just add noise of growing σ. Gap: many Langevin steps, tuning.
- **Song et al. 2021 (score-SDE):** unifies both as SDEs dx=f(t)x dt + g(t)dω. VP and VE are choices of f,g. Perturbation kernel N(s(t)x_0, s(t)²σ(t)²I) with s(t)=exp∫f, σ(t)=√∫(g²/s²). Reverse-time SDE (Anderson 1982) and the **probability-flow ODE** dx=[f x − ½g²∇log p_t]dt share the same marginals p_t. Score modeled by a net. Gap: f,g are the "first-class" objects but are practically uninteresting; the marginals are what matter. Schedule chosen for theoretical convenience (σ∝√t = heat diffusion).
- **Vincent 2011 (denoising score matching):** the optimal L2 denoiser D*(x;σ)=E[y|x] (posterior mean), and ∇log p(x;σ)=(D(x;σ)−x)/σ². This is THE bridge: score ≡ denoiser. For a finite dataset, D*(x;σ)=Σ_i N(x;y_i,σ²I)y_i / Σ_i N(x;y_i,σ²I) (softmax-weighted average of data points).
- **Song et al. 2020 (DDIM):** deterministic non-Markovian sampler; reinterpreted as Euler integration of an ODE. Uses σ(t)=t, s(t)=1 implicitly. Fewer steps than DDPM. Gap: still Euler (1st order); schedule from the discrete chain.
- **Jolicoeur-Martineau 2021:** explored higher-order/adaptive SDE solvers and Heun for diffusion. Karras2020-ADA: non-leaking augmentation for GANs.

## Unified ODE (Phase-1 derivation, in-frame)
Marginal p_t(x)=∫p_{0t}(x|x_0)p_data(x_0)dx_0 with kernel N(s x_0, s²σ²I). Factor: p_t(x)=s(t)^{−d} p(x/s(t); σ(t)) where p(x;σ)=p_data * N(0,σ²I) is the mollified data density — depends only on σ, not on the s,f,g machinery. Sub into Song's ODE; the s^{−d} constant has zero gradient and drops:
- f(t)=ṡ/s, g(t)=s√(2σ̇σ) (invert s=exp∫f, σ²=∫g²/s²).
- General scaled ODE: dx = [ (ṡ/s) x − s²σ̇σ ∇_x log p(x/s; σ) ] dt.
- With s=1: **dx = −σ̇σ ∇log p(x;σ) dt**. With σ(t)=t: σ̇=1, so dx = −t ∇log p(x;t) dt = (x − D(x;t))/t dt (using ∇log p=(D−x)/σ²).
- VE: σ(t)=√t reproduces Song's reverse-diffusion VE predictor exactly (verified in app by substituting into Heun line 5). VP: σ(t)=√(e^{½β_d t²+β_min t}−1), s=1/√(σ²+1).

## Why σ(t)=t, s(t)=1 (curvature argument)
With σ=t,s=1 the ODE tangent dx/dt=(x−D(x;t))/t always points from x straight at the current denoiser output D(x;t). A single Euler step to t=0 lands exactly on D(x;t). Since D changes only slowly with σ outside a narrow mid-σ band, trajectories are nearly straight at both large and small σ → low curvature → low truncation error → fewer steps. This is the simplest schedule and it minimizes solver work. (DDIM already uses it.)

## Heun / ρ-schedule (truncation error, app:truncationerror)
- Euler: LTE = O(h²) per step (1st order). Heun (trapezoidal, "improved Euler"): take Euler predictor, re-evaluate derivative at the endpoint, average the two slopes → LTE=O(h³) at cost of ONE extra D eval per step. At fixed NFE this dominates Euler. Revert to Euler on the final step to σ=0 (else d=(x−D)/0).
- α-family of 2-stage RK: x_{i+1}=x_i+h[(1−1/2α)d_i + (1/2α)f(x_i+αh d_i)]. α=1 → Heun, α=½ midpoint, α=⅔ Ralston. α=1 is ~optimal AND uniquely evaluates the correction at exactly t_{i+1} (lets you use nets trained only at discrete σ). Fix α=1.
- Discretization: want step size monotonically decreasing as σ→0 (error concentrated at low σ). Parameterize σ_i = (A i + B)^ρ; set σ_0=σ_max, σ_{N−1}=σ_min → **σ_i = (σ_max^{1/ρ} + i/(N−1)(σ_min^{1/ρ}−σ_max^{1/ρ}))^ρ, σ_N=0.** ρ=1 uniform; ρ→∞ geometric (Song's VE). ρ=3 nearly equalizes per-step LTE (RMSE flat ~0.003); but FID is better at ρ=5–10 because errors near σ_min matter more perceptually and σ_max is somewhat arbitrary (overshooting there is cheap). Use **ρ=7**. σ_min=0.002, σ_max=80.

## Preconditioning (THE heart, app:ourprecond)
Don't train D_θ directly: input x=y+n has variance σ_data²+σ² that explodes with σ. Don't do ε-prediction D=x−σF either: at large σ the net must finely cancel n and its error is amplified ×σ. Instead a σ-dependent skip lets the net choose signal-vs-noise prediction:
- **D_θ(x;σ) = c_skip(σ) x + c_out(σ) F_θ(c_in(σ) x; c_noise(σ)).**
- Loss E_{σ,y,n}[λ(σ)||D_θ(y+n;σ)−y||²]. Rewrite w.r.t. F_θ: effective weight λc_out², effective target F_target = (1/c_out)(y − c_skip(y+n)).
- **c_in:** require Var[c_in(y+n)]=1. Var[y+n]=σ_data²+σ² (y,n independent). ⇒ c_in = 1/√(σ²+σ_data²).
- **c_out & c_skip:** require Var[F_target]=1 ⇒ c_out² = Var[(1−c_skip)y + c_skip n] = (1−c_skip)²σ_data² + c_skip²σ². Then choose c_skip to minimize c_out (minimize error amplification): d/dc_skip[c_out²]=0 ⇒ σ_data²(2c_skip−2)+σ²(2c_skip)=0 ⇒ (σ²+σ_data²)c_skip=σ_data² ⇒ **c_skip = σ_data²/(σ²+σ_data²).** Sub back: c_out²=(σ²σ_data)²/(σ²+σ_data²)² + (σ_data²σ)²/(σ²+σ_data²)² = (σσ_data)²(σ²+σ_data²)/(σ²+σ_data²)² = (σσ_data)²/(σ²+σ_data²). ⇒ **c_out = σ·σ_data/√(σ²+σ_data²).**
- **λ(σ):** require effective weight uniform: λc_out²=1 ⇒ **λ(σ)=1/c_out²=(σ²+σ_data²)/(σ·σ_data)².** (At init F=0, expected per-σ loss = 1 exactly — verified algebraically.)
- **c_noise = ln(σ)/4** — empirical (no derivation; just a convenient input transform of σ for the net's conditioning).
- Limiting checks: σ→0: c_skip→1, c_out→0, c_in→1/σ_data → D≈x (return input, sensible since noise vanishes). σ→∞: c_skip→0, c_out→σ_data, c_in→1/σ → D≈c_out F (predict from scratch, since input is pure noise). The skip smoothly interpolates ε-pred-like and x-pred-like regimes.

## Training noise distribution p_train(σ)
After fixing λ to equalize *initial* loss, the *post-training* per-σ loss is still U-shaped: low σ irrelevant (tiny noise trivially removed, no gradient signal worth spending) and high σ hopeless (target → dataset mean). Concentrate samples at intermediate σ: **ln σ ~ N(P_mean, P_std²)** with P_mean=−1.2, P_std=1.2, σ_data=0.5.

## Stochastic sampler (churn)
SDE = prob-flow ODE ± Langevin (deterministic score decay + noise injection that cancel in marginal). β(t) sets rate of replacing old noise with new. Song's β=σ̇/σ has no special status → tune empirically. Sampler: each step add noise to go t_i → t̂_i = t_i+γ_i t_i (γ_i = S_churn/N clamped to ≤√2−1), with new-noise std S_noise (slightly >1 to counter L2-denoiser's regression-to-mean detail loss), then one Heun ODE step from t̂_i down to t_{i+1}. Only churn within σ∈[S_min,S_max] (oversaturation at extreme σ from non-conservative learned field). x_hat noise magnitude = √(t̂²−t_cur²)·S_noise.

## Canonical code (NVlabs/edm)
- EDMPrecond.forward: exactly the c_* formulas above, c_noise=sigma.log()/4. EDMLoss: sigma=exp(randn*P_std+P_mean), weight=(σ²+σ_data²)/(σσ_data)², loss=weight*(D(y+n,σ)−y)². sigma_data=0.5, P_mean=−1.2, P_std=1.2.
- edm_sampler: t_steps via ρ-schedule (num_steps=18, σ_min=.002, σ_max=80, ρ=7), x init = latents*t_steps[0]; per step churn gamma=min(S_churn/N,√2−1) if S_min≤t_cur≤S_max; t_hat=t_cur+γt_cur; x_hat = x_cur+√(t_hat²−t_cur²)·S_noise·randn; Euler d_cur=(x_hat−denoised)/t_hat, x_next=x_hat+(t_next−t_hat)d_cur; if i<N−1 Heun correction d_prime=(x_next−denoised')/t_next, x_next=x_hat+(t_next−t_hat)(½d_cur+½d_prime). Deterministic = S_churn=0.
