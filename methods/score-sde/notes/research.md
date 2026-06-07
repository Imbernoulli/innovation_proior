# Research notes — ancestors, field state, design rationale

## Field state at the time (2020)
Two families of "corrupt-then-learn-to-reverse" generative models had emerged and were strong:
- **SMLD / NCSN** (Song & Ermon 2019, 2020): estimate the score (∇_x log p) of data perturbed
  at a discrete geometric ladder of Gaussian noise scales σ_1<...<σ_N, then sample by annealed
  Langevin dynamics from the largest scale down. Trained by denoising score matching.
- **DDPM** (Sohl-Dickstein 2015; Ho et al. 2020): a discrete Markov forward chain
  x_i = sqrt(1-β_i) x_{i-1} + sqrt(β_i) z; learn a variational reverse chain trained with a
  reweighted ELBO (L_simple). Ho et al. showed it implicitly is denoising score matching.

By 2020 DDPM had pulled ahead of NCSN on FID. Both used a *finite, discrete* set of noise scales.
GANs (BigGAN, StyleGAN2-ADA) were the FID leaders; flows (Glow, FFJORD, Flow++) led likelihood.

## Why score-based at all (Yang Song's blog, in-frame knowable facts)
- EBMs p_θ = e^{-f_θ}/Z_θ need intractable Z_θ. The *score* s_θ = ∇_x log p_θ = -∇_x f_θ kills
  ∇ log Z_θ = 0. So modeling the score frees the network architecture (no normalizability constraint).
- Fisher divergence E_{p(x)} ||∇log p - s_θ||² weights by p(x): scores are inaccurate in
  low-density regions where there's little data. In high-d, Langevin starts in low-density regions
  → diverges; also slow mixing between modes. Multiple noise scales fix this: big noise populates
  empty regions (accurate score there), small noise preserves fidelity.

## Key ancestors (verify against primary paper — done)
- **Vincent 2011 (denoising score matching, DSM).** Identity: minimizing
  E_{p_σ(x̃)} ||s_θ(x̃) - ∇_x̃ log p_σ(x̃)||² is equivalent (up to a const indep of θ) to
  E_{p_data(x)} E_{p_σ(x̃|x)} ||s_θ(x̃) - ∇_x̃ log p_σ(x̃|x)||². The conditional score of a
  Gaussian kernel N(x̃; x, σ²I) is ∇_x̃ log p = -(x̃-x)/σ² = -z/σ, which is trivial. This is the
  reason the intractable marginal score in the objective can be replaced by a tractable target.
- **Hyvärinen 2005 (score matching).** Original SM via integration by parts → trace of Jacobian;
  expensive in high-d. Motivates DSM and sliced SM (Song et al. 2019) as scalable variants.
- **Anderson 1982 (reverse-time diffusion).** For dx = f dt + g dw (Itô), the time reversal is
  also a diffusion: dx = [f - g² ∇_x log p_t(x)] dt + g dw̄, running backward, dw̄ reverse Wiener,
  dt negative. General matrix-G form: drift gets -∇·[GGᵀ] - GGᵀ ∇log p_t, diffusion stays G.
  Derived via Fokker-Planck + Bayes/Girsanov; paper just cites it and uses it.
- **Fokker-Planck / Kolmogorov forward equation** (Øksendal). Governs ∂_t p_t. Used to derive the
  probability-flow ODE: rewrite the FP equation as a continuity (Liouville) equation ∂_t p = -∇·(f̃ p)
  with f̃ = f - ½∇·[GGᵀ] - ½ GGᵀ ∇log p, i.e. the same marginals follow a deterministic ODE dx = f̃ dt.
- **Maoutsa et al. 2020.** Source of the probability-flow ODE idea (simplified case).
- **Chen et al. 2018 (neural ODE).** Instantaneous change of variables:
  d log p_t/dt = -∇·f̃; integrate to get exact log-likelihood. Use Skilling-Hutchinson trace
  estimator E_v[vᵀ ∇f̃ v] (FFJORD, Grathwohl 2018) to make the divergence cheap (one vjp).
- **Särkkä & Solin 2019 (Applied SDEs).** Eqs 5.50/5.51: for affine drift, transition kernel is
  Gaussian with closed-form mean/variance ODEs. This is what makes VE/VP/sub-VP kernels closed-form.

## VE / VP / sub-VP as continuous limits (derivations confirmed from appendix A,B)
- **VE (= NCSN limit).** NCSN kernel p_σi(x̃|x)=N(x,σi²I). Equivalent Markov chain
  x_i = x_{i-1} + sqrt(σi²-σ_{i-1}²) z (so Var accumulates to σ²). Limit N→∞ with σ(t):
  x(t+Δt) = x(t) + sqrt(σ²(t+Δt)-σ²(t)) z ≈ x(t)+ sqrt(d[σ²]/dt · Δt) z
  → dx = sqrt(d[σ²(t)]/dt) dw. No drift → Var(x(t)) = σ²(t) explodes as t grows. "Variance Exploding".
  Geometric σ(t)=σ_min (σ_max/σ_min)^t ⇒ g(t)=σ(t) sqrt(2 log(σ_max/σ_min)).
- **VP (= DDPM limit).** chain x_i = sqrt(1-β_i) x_{i-1} + sqrt(β_i) z. Set β̄_i = N β_i, β(t) cts.
  x(t+Δt)=sqrt(1-β(t+Δt)Δt) x(t)+sqrt(β(t+Δt)Δt) z ≈ x(t) - ½β(t)Δt x(t) + sqrt(β(t)Δt) z
  → dx = -½β(t) x dt + sqrt(β(t)) dw. Variance ODE dΣ/dt = β(t)(I-Σ) ⇒ Σ(t)=I+e^{-∫β}(Σ0-I);
  bounded, ≡ I if Σ0=I. "Variance Preserving". β(t)=β_min + t(β_max-β_min).
- **sub-VP (new).** dx = -½β x dt + sqrt(β(t)(1-e^{-2∫β})) dw. Same mean as VP; variance
  Σ_subVP(t)=I + e^{-2∫β} I + e^{-∫β}(Σ0 - 2I); with Σ0=I gives [1-e^{-∫β}]² I, always ≤ VP variance,
  → I as t→∞. Lower variance ⇒ smaller noise injected ⇒ better likelihoods empirically.

## Continuous training objective
J(θ)=E_t{λ(t) E_{x(0)} E_{x(t)|x(0)} ||s_θ(x(t),t) - ∇ log p_{0t}(x(t)|x(0))||²}.
For affine-drift SDE the kernel is Gaussian N(x(t); μ_t x(0), σ_t² I), so the target is
∇ log p = -(x(t)-μ_t x(0))/σ_t² = -z/σ_t with x(t)=μ_t x(0)+σ_t z. Choosing λ(t) ∝ 1/E||∇log p||²
≈ σ_t² makes the weighted loss ≈ ||σ_t s_θ + z||² (the "non-likelihood" weighting in code).

## Samplers (appendix)
- **Reverse diffusion predictor.** Discretize forward as x_{i+1}=x_i+f_i+G_i z; mirror it for the
  reverse SDE: x_i = x_{i+1} - f_{i+1} + G_{i+1}G_{i+1}ᵀ s(x_{i+1},i+1) + G_{i+1} z.
- **Ancestral sampling = special discretization of reverse VP SDE** (Taylor expand 1/sqrt(1-β),
  shown in appendix; matches as β→0).
- **PC sampler.** Predictor = numerical reverse-SDE step; Corrector = score-based MCMC (Langevin)
  that pushes the sample back onto the correct marginal p_t. Langevin step size from SNR:
  ε = 2(α)(r ||z||/||s||)². NCSN sampling = corrector-only (identity predictor); DDPM = predictor-only.
- **Probability flow ODE sampler** = integrate dx=[f - ½g² s]dt deterministically; black-box RK45;
  enables exact likelihood + identifiable latent encoding.
- **Denoising final step** via Tweedie (one predictor step w/o noise) — matters for FID.

## Controllable generation
Conditional reverse SDE: replace ∇log p_t(x) with ∇log p_t(x|y) = ∇log p_t(x)+∇log p_t(y|x).
Train a time-dependent classifier p_t(y|x(t)) for class-conditional; imputation/colorization are
special cases using only the unconditional model.

## Canonical code (yang-song/score_sde_pytorch)
- sde_lib.py: SDE abstract base with sde(), marginal_prob(), prior_sampling(), discretize(),
  reverse() (builds RSDE subclass that subtracts g²·score (×0.5 for ODE) from drift, zeros diffusion
  for ODE). VPSDE/subVPSDE/VESDE concrete classes with closed-form marginal_prob.
- losses.py: get_sde_loss_fn — sample t~U(eps,T), z~N(0,I), perturbed = mean+std·z,
  loss = ||score·std + z||² (non-likelihood weighting) or ||score+z/std||²·g² (likelihood weighting).
- sampling.py: Predictor/Corrector ABCs; EulerMaruyama, ReverseDiffusion, AncestralSampling predictors;
  Langevin corrector; get_pc_sampler (corrector then predictor each step); get_ode_sampler (RK45).
- models/utils.py get_score_fn: for VP, network predicts noise; score = -model/std (sign flip + scale).
- likelihood.py: instantaneous change of variables + Hutchinson trace.
</content>
