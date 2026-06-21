# VDM — Synthesis notes (Phase 1.5)

## Pain point / research question
Diffusion models (DDPM, score-SDE, NCSN) produce perceptually great samples but lose to autoregressive
models on *likelihood* (bits/dim) benchmarks. Can a diffusion model be a great *likelihood* model?
Two sub-problems:
1. The discrete-time diffusion ELBO is messy, requires fp64, depends on hand-tuned noise schedules.
2. We want to *learn* the noise schedule and understand what the loss actually depends on.

## Load-bearing ancestors
- **Sohl-Dickstein et al. 2015 (DPM)**: forward Gaussian diffusion that gradually destroys structure;
  reverse process learned; trained by VLB. ELBO decomposes into prior + reconstruction + sum of KLs (the L_t).
  Limitation: complicated, weak samples, not competitive likelihood.
- **Ho et al. 2020 (DDPM)**: parameterize reverse mean via a NOISE-prediction net ε̂; show VLB ≈ weighted
  denoising score matching ("simple objective" drops the weights, sets w=1 on ε-MSE). Fixed β-linear schedule.
  Needs fp64 for the discrete loss. Optimizes sample quality (FID), not likelihood; "simple" objective is NOT the VLB.
- **Song & Ermon 2019 / 2020 (NCSN / NCSNv2)**: multi-scale denoising score matching; variance-exploding
  noise (α=1, σ geometric). Score model s≈∇log q(z_t).
- **Song et al. 2020 (score-SDE)**: continuous-time diffusion as forward SDE; reverse SDE/probability-flow ODE;
  unifies VP (DDPM) and VE (NCSN). Gives likelihood via the ODE but the bound/loss connection is via SDE machinery.
- **Kingma & Welling 2013 / Rezende 2014 (VAE)**: ELBO = reconstruction − KL; reparameterization trick.
  A diffusion model = a VAE with a FIXED Gaussian inference chain and many (∞) layers.
- **Vincent 2011 (DSM)**: denoising score matching identity — matching score of q(z_t|x) equals matching
  score of marginal q(z_t) up to a constant. Gives consistency.
- **Nichol & Dhariwal 2021 (IDDPM)**: cosine schedule, learn variances, better LL.
- **Townsend 2018 (BB-ANS) / Hinton 1993 (bits-back) / Kingma 2019 (Bit-Swap)**: VLB of a hierarchical
  latent model = expected codelength of bits-back coding → lossless compression.

## The derivation chain (the heart)
1. View diffusion as a deep VAE: q(z_t|x)=N(α_t x, σ_t² I), SNR(t)=α_t²/σ_t² strictly decreasing.
   Generative model inverts it. ELBO: −log p(x) ≤ prior KL + recon + diffusion loss L_T.
2. Markov forward → q(z_t|z_s)=N(α_{t|s} z_s, σ²_{t|s} I), α_{t|s}=α_t/α_s, σ²_{t|s}=σ_t²−α_{t|s}²σ_s².
   Posterior q(z_s|z_t,x) Gaussian by Bayes (Gaussian prior × linear-Gaussian likelihood).
   σ_Q²=σ²_{t|s}σ_s²/σ_t², μ_Q = (α_{t|s}σ_s²/σ_t²) z_t + (α_s σ²_{t|s}/σ_t²) x.
3. p(z_s|z_t)=q(z_s|z_t, x=x̂_θ(z_t,t)) → same form, x→x̂. Equal variances ⇒ KL = (1/2σ_Q²)‖μ_Q−μ_θ‖².
4. KEY ALGEBRA: μ_Q−μ_θ = (α_s σ²_{t|s}/σ_t²)(x−x̂). Plug in:
   KL = (1/2)(α_s²/σ_s² − α_t²/σ_t²)‖x−x̂‖² = (1/2)(SNR(s)−SNR(t))‖x−x̂‖².  [each term cancels beautifully]
   So L_T = (1/2) E_ε Σ_i (SNR(s)−SNR(t))‖x−x̂‖²; unbiased MC: (T/2)E[(SNR(s)−SNR(t))‖x−x̂‖²].
5. More steps better: L_2T − L_T < 0 if model good (Riemann sum of decreasing function). ⇒ take T→∞.
6. Continuous limit: write L_T with (SNR(t−τ)−SNR(t))/τ; τ→0 →
   L_∞ = −(1/2)E_ε ∫₀¹ SNR'(t)‖x−x̂‖² dt = −(1/2)E_{ε,t}[SNR'(t)‖x−x̂‖²].
7. SNR-ENDPOINT INVARIANCE: change of variable v=SNR(t) (invertible, monotone). dv = SNR'(t)dt.
   L_∞ = (1/2)E_ε ∫_{SNR_min}^{SNR_max} ‖x−x̃(z_v,v)‖² dv.  (−SNR' and reversed limits give +.)
   Integrand depends only on v through z_v = α_v x + σ_v ε = α_v(x + ε/√v) — only v matters, NOT α,σ separately.
   ⇒ L_∞ depends on schedule ONLY through endpoints SNR_min=SNR(1), SNR_max=SNR(0).
   Invariant to the shape between. Also p(x) itself invariant (equivalence of VP/VE etc.) up to rescaling
   of latents x̃^B(z,v)=x̃^A((α_v^A/α_v^B)z,v).
8. v-/noise-/x-/score equivalence: x̂=(z−σ ε̂)/α; ε̂=(z−α x̂)/σ; s=∇log q(z_t|x)=−ε/σ. score view:
   s_θ(z,t)=(α x̂−z)/σ². Noise-pred plugged in: L_∞ = (1/2)E[γ'(t)‖ε−ε̂‖²], discrete (T/2)E[(exp(γ(t)−γ(s))−1)‖ε−ε̂‖²].
9. Weighted loss L_∞(x,w)=(1/2)E∫ w(v)‖x−x̃‖² dv captures DDPM/NCSN. w=1 = VLB. DDPM "simple" ⇒ w=1/γ'(t).
   NCSN's geometric σ ⇒ γ'(t) const ⇒ w const ⇒ consistent with VLB. DDPM/IDDPM put more weight on low SNR.
10. Variance minimization: because L_∞ is invariant to SCHEDULE SHAPE (only endpoints affect VLB),
    we are FREE to choose the shape to MINIMIZE the VARIANCE of the MC estimator. Parameterize γ_η(t)
    monotone NN; endpoints γ0,γ1 trained w.r.t. VLB; interior η trained by SGD on the SQUARED loss
    L^MC² (since E[L^MC²]=L²+Var, and L² is η-independent, ∇_η E[L^MC²]=∇_η Var). Compute via d/dSNR of loss
    times d SNR/dη — no second backprop. Also low-discrepancy/antithetic sampling of t.
11. Fourier features: recon model p(x|z_0) is weak; fine-scale detail burden on x̂. At low noise q(z_t)
    sharply peaked (discrete 8-bit data). Paper formula appends sin/cos(2^n π z) channels for n=7,8.
    Public code uses Base2FourierFeatures(start=6, stop=8) with multiplier 2π, giving the same two frequencies
    2^7π and 2^8π. Lets SNR_max go high (log-SNR ≈ 13.3 vs DDPM ≈ 8). Helps likelihood a lot.
12. Numerical stability: discrete loss has exp(γ(t)−γ(s))−1 → use expm1; σ²_{t|s} via −expm1(softplus diff).
    Allows fp32 where DDPM needed fp64.
13. Reconstruction loss: p(x_i|z_0,i) ∝ q(z_0,i|x_i) normalized over 256 values; closed-form discretized Gaussian.
14. Bits-back: discrete-T model is hierarchical latent var model; −VLB = expected bits-back codelength → lossless compression.

## Design decisions → why
- Parameterize marginal q(z_t|x) directly (not steps q(z_{t+ε}|z_t)) → lets us learn schedule & take T→∞ cleanly.
- Noise-prediction parameterization ε̂ (not x̂ directly) → matches DDPM, simple objective, well-conditioned at high noise.
- VP (α²=1−σ²) chosen because of the equivalence result — any spec equivalent, VP is convenient/stable.
- Learn schedule via monotone NN γ_η(t)=l1(t)+l3(σ(l2(l1(t)))) with positive weights → guarantees monotone SNR.
- Minimize variance (not loss) for interior schedule → loss value is invariant so only variance is free; faster training.
- w=1 (unweighted) → exactly the VLB / likelihood; weighted only for FID.
- No up/down-sampling in U-Net, deeper, drop attention except middle → likelihood-tuned, less overfit.

## Code grounding
google-research/vdm (JAX/Flax), commit dc27b98a554f65cdc654b800da5aa1846545d41b. model_vdm.py:
VDM.__call__ computes loss_recon, loss_klz, loss_diff (T==0 continuous via jvp of gamma → γ'(t)·MSE/2;
T>0 discrete via expm1(g_t−g_s)). It samples z_t as sqrt(1-sigmoid(g_t))*f + sqrt(sigmoid(g_t))*eps.
NoiseSchedule_NNet = DenseMonotone with abs(weights); public configs also use learnable_scalar. EncDec.decode =
discretized-Gaussian logits over vocab. Base2FourierFeatures start=6 stop=8 with 2π multiplier. sample():
ancestral with c=−expm1(g_s−g_t). Public train loop optimizes summed BPD and does not implement a separate
squared-loss variance-gradient path for the schedule interior.
