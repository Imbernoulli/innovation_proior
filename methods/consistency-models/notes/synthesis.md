# Synthesis — Consistency Models

## Pain point
Diffusion / score-based models sample by numerically integrating the PF ODE (or SDE) from noise to data. This is many NFEs (10–2000). GANs/VAEs/flows are 1 step but GANs are adversarial/unstable, others lower quality. Goal: ONE network eval to map noise→data, but keep the diffusion advantages (compute↔quality tradeoff, zero-shot editing). No adversarial training, minor architectural constraints.

## Ancestors (load-bearing)
- **Score-based SDE / PF ODE (Song et al. 2021)**: forward SDE dx=μ dt+σ dw; there is a deterministic PF ODE dx=[μ - ½σ²∇log p_t]dt with same marginals p_t. Score model s_φ≈∇log p_t via score matching, plug in, integrate. Sampling = solving ODE backward, slow.
- **EDM (Karras et al. 2022)**: choose μ=0, σ(t)=√(2t) ⇒ p_t = p_data ⊗ N(0,t²I), σ(t)=t is the noise std. Empirical PF ODE becomes dx/dt = -t s_φ(x,t). π=N(0,T²I). T=80, ε=0.002. Heun 2nd-order solver. Preconditioning: D_θ(x,σ)=c_skip x + c_out F_θ(c_in x, c_noise), c_skip=σ_data²/(σ²+σ_data²), c_out=σσ_data/√(σ²+σ_data²), c_in=1/√(σ²+σ_data²), σ_data=0.5, c_noise=¼ln σ. ρ=7 time discretization t_i=(ε^{1/ρ}+ (i-1)/(N-1)(T^{1/ρ}-ε^{1/ρ}))^ρ.
- **Progressive distillation (Salimans & Ho 2022)**: halve the number of sampling steps repeatedly; student matches teacher's 2-step DDIM output in 1 step; repeated halving. Doesn't need a sample dataset (unlike Luhman, Zheng). Baseline.
- **Knowledge distillation (Luhman 2021), Zheng 2022**: collect big dataset of diffusion samples then regress one-step. Expensive offline data generation.

## The idea
PF ODE trajectory {x_t} smoothly connects x_ε (≈data) to x_T (≈noise). Define **consistency function** f(x_t,t)=x_ε for any t. **Self-consistency**: f(x_t,t)=f(x_{t'},t') on same trajectory. **Boundary**: f(x_ε,ε)=x_ε (identity at ε). Single-step sample = f(x_T,T).

## Parameterization (boundary for free)
- Option 1: f_θ = x if t=ε else F_θ(x,t). (discontinuous-ish; not differentiable.)
- Option 2 (used): f_θ(x,t)=c_skip(t) x + c_out(t) F_θ(x,t), with c_skip(ε)=1, c_out(ε)=0 ⇒ boundary exact and f differentiable (needed for continuous-time).
- Concrete (modify EDM): c_skip(t)=σ_data²/((t-ε)²+σ_data²), c_out(t)=σ_data(t-ε)/√(σ_data²+t²). Check: at t=ε, c_skip=σ_data²/σ_data²=1, c_out=0. Good.

## Consistency Distillation (CD)
Discretize [ε,T] into t_1=ε<...<t_N=T (ρ=7 EDM grid). Sample x~data, x_{t_{n+1}}~N(x,t_{n+1}²I). One ODE solver step backward to estimate x̂^φ_{t_n}=x_{t_{n+1}}+(t_n-t_{n+1})Φ(x_{t_{n+1}},t_{n+1};φ). Euler: Φ=-t s_φ ⇒ x̂=x_{t_{n+1}}-(t_n-t_{n+1})t_{n+1}s_φ. Loss:
L_CD = E[λ(t_n) d(f_θ(x_{t_{n+1}},t_{n+1}), f_{θ⁻}(x̂^φ_{t_n},t_n))], θ⁻=EMA stopgrad target. λ≡1, d∈{ℓ2,ℓ1,LPIPS}.
EMA: θ⁻←stopgrad(μθ⁻+(1-μ)θ). Stabilizes (target/online like RL target net, BYOL).

### Theorem 1 (CD consistency)
If L_CD^N(θ,θ;φ)=0 and f_θ Lipschitz-L, solver local error O(Δt^{p+1}), then sup||f_θ(·,t_n)-f(·,t_n;φ)||=O(Δt^p). Proof: zero loss ⇒ f_θ(x_{t_{n+1}},t_{n+1})=f_θ(x̂_{t_n},t_n). Define e_n=f_θ(x_{t_n},t_n)-f(x_{t_n},t_n;φ). Recursion e_{n+1}=[f_θ(x̂_{t_n},t_n)-f_θ(x_{t_n},t_n)]+e_n (using consistency of true f along trajectory + zero-loss identity). ||e_{n+1}||≤||e_n||+L·O((t_{n+1}-t_n)^{p+1}). e_1=0 because boundary f_θ(x_{t_1},t_1)=x_{t_1}=f(x_{t_1},t_1;φ). Telescope ⇒ O(Δt^p). Boundary precludes trivial f≡0.

## Consistency Training (CT)
Lemma: ∇log p_t(x_t) = -E[(x_t-x)/t² | x_t], x~data, x_t~N(x,t²I). (Tweedie/score-of-Gaussian-convolution; prove via differentiating log of convolution + Bayes.) So -(x_t-x)/t² is an unbiased one-sample estimate of the score. Replaces s_φ in the Euler step.

### Theorem 2 (CD↔CT)
With Euler solver and s_φ=∇log p_t exactly,
L_CD^N(θ,θ⁻;φ) = L_CT^N(θ,θ⁻) + o(Δt),
where L_CT^N = E[λ(t_n) d(f_θ(x+t_{n+1}z,t_{n+1}), f_{θ⁻}(x+t_n z,t_n))], z~N(0,I).
Derivation: Euler target x̂_{t_n} = x_{t_{n+1}} + (t_{n+1}-t_n)t_{n+1}∇log p_{t_{n+1}}(x_{t_{n+1}}) [since -t s, and (t_n-t_{n+1})·(-t_{n+1})s = (t_{n+1}-t_n)t_{n+1}s]. Taylor-expand f_{θ⁻} and d to first order around x_{t_{n+1}}. Apply law of total expectation to replace ∇log p_{t_{n+1}}(x_{t_{n+1}}) inside E by the conditional unbiased estimator -(x_{t_{n+1}}-x)/t_{n+1}². Then (t_n-t_{n+1})t_{n+1}·(x_{t_{n+1}}-x)/t_{n+1}² = (t_n-t_{n+1})(x_{t_{n+1}}-x)/t_{n+1} = (t_n-t_{n+1})z with z=(x_{t_{n+1}}-x)/t_{n+1}~N(0,I). Reverse-Taylor: argument becomes x_{t_{n+1}}+(t_n-t_{n+1})z = x+t_{n+1}z+(t_n-t_{n+1})z = x+t_n z. Gives L_CT. As Δt→0, CT→CD. Also L_CT≥O(Δt) if inf L_CD>0, so the leading term dominates the o(Δt) remainder ⇒ minimizing CT ≈ minimizing CD. CT needs NO pretrained model. Schedule N(k) increasing (small N=low variance high bias early; large N late), μ(k)=exp(s_0 ln μ_0 / N(k)).

## Sampling
- 1-step: x=f_θ(x_T,T), x_T~N(0,T²I).
- Multistep (Alg 2): x=f_θ(x̂_T,T); for n: z~N(0,I), x̂_{τ_n}=x+√(τ_n²-ε²)z, x=f_θ(x̂_{τ_n},τ_n). τ via greedy ternary search on FID.

## Continuous-time (N→∞), appendix
- θ⁻=θ (no stopgrad), ℓ2: L_CD^∞ = E[ λ/((τ⁻¹)')² ||∂f/∂t - t (∂f/∂x) s_φ||² ]. From 2nd-order Taylor of d (G=Hessian); uses (eq ctcd1) f_θ(x̂_{t_n},t_n)-f_θ(x_{t_{n+1}},t_{n+1}) = -(∂f/∂t - t ∂f/∂x s_φ)τ'Δu + O(Δu²). Remark: =0 iff f matches true consistency fn, since d/dt f(x_t,t)=0 ⇒ ∂f/∂t + ∂f/∂x·(dx/dt)=0 and dx/dt=-t s.
- θ⁻=stopgrad(θ): pseudo-objective (gradient only).
- CT^∞ pseudo-objective, replace s_φ by -(x_t-x)/t². No bias (Δt→0).

## Code mapping (openai/consistency_models, cm/karras_diffusion.py)
- get_scalings_for_boundary_condition(sigma) → c_skip,c_out,c_in with (sigma-sigma_min).
- denoise(): rescaled_t=1000*0.25*ln(sigma); model_output=model(c_in*x_t, rescaled_t); denoised=c_out*model_output+c_skip*x_t = f_θ.
- consistency_losses(): sample t (index) and t2 (index+1) on EDM ρ grid; x_t=x_start+noise*t; distiller=denoise_fn(x_t,t) [online @ t_{n+1}]; x_t2 = euler_solver (CT, denoiser=x0) or heun_solver (CD, teacher) → target point at t2=t_n; distiller_target=target_denoise_fn(x_t2,t2) [EMA target]; loss d(distiller,distiller_target)*weights. NOTE in code t=index→larger sigma is t_{n+1}, t2=index+1→ careful: actually grid is descending sigma with index, t corresponds to higher sigma. Online at noisier point, target at less-noisy point. weight schedule get_weightings.
- euler_solver with teacher=None uses denoiser=x0 (=data x_start), d=(x-x0)/t — this is exactly the score estimate -(x_t-x)/t²·... → CT. heun_solver with teacher → CD.
- sample_onestep: distiller(x, sigma_max). stochastic_iterative_sampler: multistep add-noise/denoise.
- target EMA in train_util _update_target_ema; ema_scale_fn returns (target_ema μ(k), num_scales N(k)).
