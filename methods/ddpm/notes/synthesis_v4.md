# Synthesis (V4) — composing the deliverables FROM this file

This is the pre-Phase-2 understanding dump. Everything in context/reasoning/answer is transcribed from here. Method = a Gaussian forward-noising / learned-reverse-denoising latent-variable generative model trained by variational inference, with an ε-prediction parameterization and a simplified (unweighted) MSE loss. Name reserved for answer.md only; context.md must NOT name it.

## The pain point (research question, in-frame, mid-2020)
Want a generative model of images that (a) produces high-quality samples, (b) is straightforward to define and stable/efficient to train, (c) is likelihood-based (admits a proper bound on NLL), and (d) does not require adversarial training. The four families on the table each fail one of these:
- GANs: best samples, but adversarial min-max → unstable, mode-dropping, no likelihood.
- Autoregressive (PixelCNN/Sparse Transformer): great likelihood, but sequential O(D) sampling, and a fixed coordinate ordering is an arbitrary inductive bias.
- Normalizing flows: exact likelihood + fast sampling, but the invertibility/tractable-Jacobian constraint caps expressiveness and forces equal latent dim.
- VAEs: clean ELBO, fast, but posterior collapse + blurry samples from a single-step amortized posterior.
- Score/EBM (NCSN, EBMs): samples approaching GANs, but sampler hyperparameters set by hand post-hoc, no proper likelihood, training not directly optimizing the sampler.

So: is there a latent-variable model with a *parameter-free* inference path (no posterior to collapse), trained by a real variational bound (likelihood-based, non-adversarial), that still samples at GAN quality?

## Load-bearing ancestors (with the precise gap each leaves)

### Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli 2015 — diffusion via nonequilibrium thermodynamics
Idea: define a fixed forward Markov chain q that slowly adds noise over T steps, gradually destroying structure until x_T is pure noise. Then *learn* the reverse chain p_θ that undoes it. Key fact: if each forward step adds a small enough amount of Gaussian noise, the reverse conditional is also approximately Gaussian, so a Gaussian reverse model is expressive enough. Trained by a variational bound on NLL. The forward process has NO learnable parameters → no inference network to collapse. THIS IS THE SKELETON the method inherits exactly (appendix even says the ELBO-reduction derivation "is from Sohl-Dickstein"). Gap: never demonstrated high-quality images — CIFAR10 NLL ≤ 5.40 bits/dim and visually weak. Left open: the right reverse-process *parameterization*, variance choices, schedule, network.

### Kingma & Welling 2013 — VAE / reparameterization
ELBO = E_q[log p(x|z)] − KL(q(z|x)‖p(z)). Reparameterization trick: z = μ + σ⊙ε, ε~N(0,I), turns a sampling step into a differentiable deterministic function of params + fixed noise → low-variance pathwise gradients. The diffusion model is, structurally, a T-layer hierarchical VAE whose encoder q is fixed and parameter-free. Reparameterization is what gives the closed-form x_t(x_0,ε) and the low-variance SGD on random ELBO terms.

### Song & Ermon 2019 — NCSN (score matching + annealed Langevin)
Learn the score ∇_x log p(x) via denoising score matching at multiple Gaussian noise scales σ_1>...>σ_L, with one noise-conditional network s_θ(x,σ). Sample by annealed Langevin dynamics: x ← x + (δ/2)·s_θ(x,σ) + √δ·z, walking σ from large to small. Loss = Σ_σ λ(σ)·E‖s_θ(x̃,σ) − ∇ log q_σ(x̃|x)‖². Gaps the method fixes: (1) the σ schedule, step sizes, and noise scales of the sampler are hand-tuned post-hoc, not learned; (2) NCSN does NOT rescale x by √(1−β) each step, so the variance of the perturbed data grows — inputs to the net are inconsistently scaled; (3) the perturbation kernel does NOT drive the data to a fixed known prior (forward doesn't truly "destroy signal" to N(0,I)); (4) it is not trained as a latent-variable model by a variational bound — no likelihood, no guarantee the sampler optimizes a quality metric. The method's ε-loss turns out to BE NCSN-style weighted DSM, but now derived from an ELBO, with the sampler coefficients rigorously fixed by the β schedule.

### Vincent 2011 — denoising score matching = score matching
For a Gaussian corruption q_σ(x̃|x) = N(x̃; x, σ²I), the DSM objective E‖s_θ(x̃) − ∇_{x̃} log q_σ(x̃|x)‖² equals (explicit) score matching up to a constant. And ∇_{x̃} log N(x̃;x,σ²I) = (x − x̃)/σ² = −(x̃−x)/σ² ∝ −(the noise that was added). THIS is the bridge: predicting the added noise ε is, up to a positive scale, predicting the score of the noised data. Concretely for our marginal q(x_t|x_0)=N(√ᾱ_t x_0,(1−ᾱ_t)I): ∇_{x_t} log q(x_t|x_0) = −(x_t−√ᾱ_t x_0)/(1−ᾱ_t) = −ε/√(1−ᾱ_t). So a network predicting ε is a learned (scaled) score.

### Langevin dynamics (background sampler)
x ← x + (δ/2)∇ log p(x) + √δ z. The reverse step the method lands on, x_{t-1} = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε_θ) + σ_t z, has exactly this shape with ε_θ as a learned scaled gradient → "aha, the sampler IS Langevin, but with coefficients pinned by β_t instead of hand-tuned."

## The full derivation chain (every step that must be lived in reasoning.md)

1. **Setup.** Latent-variable model p_θ(x_0)=∫p_θ(x_{0:T})dx_{1:T}, x_1..x_T same dim as x_0. Reverse process = Markov chain with learned Gaussian transitions, fixed prior p(x_T)=N(0,I):
   p_θ(x_{0:T}) = p(x_T)∏ p_θ(x_{t-1}|x_t), p_θ(x_{t-1}|x_t)=N(x_{t-1};μ_θ(x_t,t),Σ_θ(x_t,t)).

2. **Forward process FIXED (design decision).** q(x_{1:T}|x_0)=∏ q(x_t|x_{t-1}), q(x_t|x_{t-1})=N(√(1−β_t) x_{t-1}, β_t I). WHY fix β_t (not learn them)? In a VAE you'd learn the encoder, but here learning the forward variances by reparameterization is possible yet (a) unnecessary — fixing them makes q parameter-free so there's no posterior to collapse and L_T becomes a constant droppable from training; (b) the √(1−β_t) scaling is chosen so the marginal variance stays bounded (≈1) instead of growing (fixes NCSN gap #2). WHY √(1−β_t) and not just +noise? so Var stays ≈ constant: if x_0 has unit var, Var(x_t)=(1−β_t)Var(x_{t-1})+β_t → stays 1. Consistent input scale to one shared net.

3. **Closed-form marginal q(x_t|x_0) by induction (must derive).** α_t:=1−β_t, ᾱ_t:=∏_{s≤t}α_s. x_t=√α_t x_{t-1}+√β_t ε_{t-1}. Substitute recursively: x_t=√(α_t α_{t-1}) x_{t-2} + √(α_t β_{t-1}) ε' + √β_t ε''. Two independent Gaussians with variances α_t(1−α_{t-1})... merge: sum of N(0,α_t(1−α_{t-1})) and N(0,β_t)=N(0,1−α_tα_{t-1}). Induct → q(x_t|x_0)=N(√ᾱ_t x_0,(1−ᾱ_t)I). So x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε, ε~N(0,I) (reparam). This is the engine: sample any t directly, no chain walk → SGD on random terms.

4. **ELBO and the variance-reduction rewrite (full appendix-A telescoping, must derive).**
   Start: E[−log p_θ(x_0)] ≤ E_q[−log p_θ(x_{0:T})/q(x_{1:T}|x_0)] = E_q[−log p(x_T) − Σ_{t≥1} log p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1})] =: L.
   Naive: each term needs MC over the forward noise; q(x_t|x_{t-1}) has huge variance.
   Trick: condition the forward posterior on x_0. By Bayes, q(x_t|x_{t-1}) = q(x_{t-1}|x_t,x_0)·q(x_t|x_0)/q(x_{t-1}|x_0). Split t=1 out, substitute for t>1; the q(x_{t-1}|x_0)/q(x_t|x_0) ratios telescope, leaving p(x_T)/q(x_T|x_0). Land on:
   L = E_q[ KL(q(x_T|x_0)‖p(x_T))  +  Σ_{t>1} KL(q(x_{t-1}|x_t,x_0)‖p_θ(x_{t-1}|x_t))  −log p_θ(x_0|x_1) ]
       =      L_T (const)                 Σ L_{t-1}                                         L_0.
   WHY this matters: every L_{t-1} is now a KL between two GAUSSIANS → closed form (Rao-Blackwellized), not high-variance MC. THIS is the payoff of conditioning on x_0.

5. **Tractable forward posterior (complete the square, must derive).** q(x_{t-1}|x_t,x_0) ∝ q(x_t|x_{t-1})q(x_{t-1}|x_0). Both Gaussian in x_{t-1}; product is Gaussian. Collect quadratic and linear terms in x_{t-1}:
   precision = α_t/β_t + 1/(1−ᾱ_{t-1}); → β̃_t = (1−ᾱ_{t-1})/(1−ᾱ_t)·β_t.
   mean μ̃_t(x_t,x_0) = (√ᾱ_{t-1}β_t/(1−ᾱ_t)) x_0 + (√α_t(1−ᾱ_{t-1})/(1−ᾱ_t)) x_t.
   So q(x_{t-1}|x_t,x_0)=N(μ̃_t, β̃_t I).

6. **Fix Σ_θ = σ_t² I (design decision).** Two natural choices both work empirically: σ_t²=β_t (optimal if x_0~N(0,I)) and σ_t²=β̃_t (optimal if x_0 is a single point) — upper/lower entropy bounds (Sohl-Dickstein). WHY untrained const, not learned? KL is dominated by the means when variances match; learning a diagonal Σ_θ in the bound was unstable and worse (later confirmed in ablation). So set variance to a schedule constant, only learn the mean.

7. **L_{t-1} reduces to a mean-matching MSE.** With matched isotropic variances, KL(N(μ̃,σ²I)‖N(μ_θ,σ²I)) = ‖μ̃−μ_θ‖²/(2σ²) + const. So L_{t-1} = E_q[ (1/2σ_t²)‖μ̃_t(x_t,x_0) − μ_θ(x_t,t)‖² ] + C. Most direct param: μ_θ predicts μ̃_t.

8. **ε-prediction parameterization (the algebra + the aha).** Plug x_0 = (x_t − √(1−ᾱ_t)ε)/√ᾱ_t (invert the reparam) into μ̃_t. After simplification:
   μ̃_t expressed via (x_t,ε) = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε).
   Since x_t is the net's input, choose μ_θ(x_t,t) = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t)) — i.e. predict ε instead of μ̃. Substituting back, L_{t-1}−C becomes
   E_{x_0,ε}[ β_t²/(2σ_t² α_t (1−ᾱ_t)) · ‖ε − ε_θ(√ᾱ_t x_0+√(1−ᾱ_t)ε, t)‖² ].
   AHA: this is exactly NCSN-style denoising score matching summed over noise scales t (Vincent 2011 bridge), and the reverse step x_{t-1}=μ_θ+σ_t z is Langevin with ε_θ as a learned scaled score. So fitting the ELBO by predicting ε = doing DSM = training a Langevin sampler — three views, one objective. (x_0-prediction is a third option but gave worse early samples.)

9. **Discrete decoder for L_0 (data scaling).** Images are {0..255}→[−1,1]. Last reverse step p_θ(x_0|x_1)= product over pixels of the integral of N(x;μ_θ(x_1,1),σ_1²) over the [x−1/255, x+1/255] bin (discretized Gaussian, à la PixelCNN++/improved-VAE). WHY: makes the bound a true lossless codelength over 8-bit data without dequantization noise or scaling-Jacobian terms. At sampling end, display μ_θ(x_1,1) noiselessly.

10. **Simplified objective (drop the weighting) — design decision + why it trains better.**
    L_simple(θ) = E_{t,x_0,ε} ‖ε − ε_θ(√ᾱ_t x_0+√(1−ᾱ_t)ε, t)‖². i.e. set the per-t weight β_t²/(2σ_t²α_t(1−ᾱ_t)) → 1.
    WHY drop it: (i) simpler to implement; (ii) the true weight is large at small t (low noise) — those terms train the net to remove tiny amounts of noise, an easy task; the unit weight DOWN-weights small t relative to the true bound, letting the net spend capacity on harder large-t (high-noise) denoising → better samples. It is still a (re-)weighted variational bound (β-VAE-style emphasis reweighting), so it's principled, just emphasizes perceptually relevant scales. Empirically gives best sample quality (worse codelength, fine — we want samples).

11. **Architecture choices (designed, with why).**
    - Backbone: U-Net (Ronneberger 2015) shaped like unmasked PixelCNN++ (Salimans 2017) over Wide-ResNet blocks. WHY U-Net: denoiser input/output are same-resolution images; skip connections let high-freq detail bypass the bottleneck (a pure bottleneck would destroy the fine detail a denoiser must restore).
    - One SHARED net across all t. WHY: ᾱ_t makes the task a smooth family indexed by t; sharing params is far cheaper than T separate nets and lets the net interpolate across noise levels (NCSN's noise-conditioning idea).
    - Timestep conditioning: Transformer sinusoidal embedding of integer t → small MLP → injected into EVERY residual block (vs NCSNv1 only-in-norm or v2 only-at-output). WHY sinusoidal: a smooth multi-frequency code so nearby t map to nearby embeddings; injecting throughout lets every layer modulate to the current noise scale.
    - Self-attention at 16×16. WHY there: attention is O(n²) in #positions, affordable only at coarse resolution; 16×16 is where global layout/symmetry can be coordinated without the quadratic cost exploding; convolutions handle local texture at fine resolutions.
    - Group norm (Wu 2018) replacing weight norm. WHY: batch/noise-level-independent normalization, consistent across all t and the small high-res batches; simpler than weight norm.
    - Dropout 0.1 (CIFAR): without it, PixelCNN++-style overfitting artifacts.

12. **T and β schedule.** T=1000 (matches prior work's NN-eval count). β linear from 1e-4 to 0.02. WHY small β: keeps reverse≈forward functional form (Gaussian reverse valid) and keeps SNR at x_T tiny so L_T=KL(q(x_T|x_0)‖N(0,I))≈1e-5 bits/dim ≈ 0 → forward truly destroys signal, prior matches aggregate posterior (fixes NCSN gaps #1,#3,#4). WHY large T: many small steps keep each reverse conditional near-Gaussian; T need not equal data dim (can be shorter for speed, longer for expressiveness).

13. **Optimization substrate.** Adam, lr 2e-4 (2e-5 at 256² — large rate unstable), EMA decay 0.9999, batch 128 (64 at high-res), random horizontal flips. All standard, transferred from a CIFAR sweep.

14. **Algorithms (the landing).**
    Training: repeat{ x_0~data; t~Unif{1..T}; ε~N(0,I); grad step on ‖ε − ε_θ(√ᾱ_t x_0+√(1−ᾱ_t)ε, t)‖² }.
    Sampling: x_T~N(0,I); for t=T..1: z~N(0,I) if t>1 else 0; x_{t-1}=(1/√α_t)(x_t−((1−α_t)/√(1−ᾱ_t))ε_θ(x_t,t))+σ_t z; return x_0.

## Connections worth surfacing (appendix material, no experiments)
- Alternate ELBO form: L = KL(q(x_T)‖p(x_T)) + Σ E_q KL(q(x_{t-1}|x_t)‖p_θ(x_{t-1}|x_t)) + H(x_0). Not tractable, but: set T=dim, forward = mask the t-th coordinate, p(x_T)=blank, fully-expressive p_θ → this IS an autoregressive model. So Gaussian diffusion = autoregressive model with a generalized (non-coordinate) "bit ordering"; Gaussian noise may be a more natural ordering for images than masking, and T need not equal dim.
- Progressive lossy coding: treat L_1+..+L_T as rate, L_0 as distortion; the reverse process is a progressive decoder; x̂_0=(x_t−√(1−ᾱ_t)ε_θ)/√ᾱ_t is the running estimate. Most bits go to imperceptible detail → diffusion is an excellent lossy compressor (conceptual-compression hint). (Mention as forward-looking interpretation, no numbers.)

## Code grounding (final code mirrors these)
- lucidrains GaussianDiffusion (PyTorch): buffers betas, alphas_cumprod, alphas_cumprod_prev, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1/2; methods q_sample, q_posterior, predict_start_from_noise, model_predictions, p_mean_variance, p_sample, p_sample_loop, p_losses (pred_noise → target=noise, MSE), forward (normalize + random t). Unet with SinusoidalPosEmb→MLP, ResnetBlock(time scale/shift), Attention at inner res, group/RMS norm, dim_mults. Trainer: Adam, EMA, cycle(dl), save/sample. extract() gather helper. linear_beta_schedule scale 0.0001→0.02.
- hojonathanho official (TF) confirms identical buffers/coeffs, fixedlarge=betas/fixedsmall=posterior_variance, mse loss target=noise, p_sample nonzero_mask at t=0.

## Code-framework SCAFFOLD (pre-method, what reasoning/answer FILL IN) — corresponds piece-for-piece
Presuppose NOTHING about Gaussian chains / ε / betas. Bare latent-variable generative training harness only:
- data pipeline: images → [−1,1], flips, batches (KNOWN).
- `class DenoiseNet(nn.Module): # TODO` — generic image→image net w/ optional scalar conditioning input; same-shape in/out (the U-Net-style backbone exists generically).
- `def training_loss(model, x0): # TODO` — generic per-example scalar loss for a latent-variable generative model.
- `def sample(model, shape): # TODO` — generic draw from the trained generative model.
- optimizer = Adam; ema = EMA(model); training loop over batches calling training_loss, opt.step, ema.update (KNOWN).
NO names: no GaussianDiffusion, q_sample, q_posterior, beta/alpha, epsilon, posterior_*, schedule. NO "reference implementation"/"official repo". The reasoning DISCOVERS forward/reverse Gaussian chain, the ELBO terms, ε-prediction, betas; the final code fills training_loss (= ε-MSE on √ᾱ x0+√(1-ᾱ)ε), sample (= reverse Langevin loop), DenoiseNet (= time-conditioned U-Net), and adds the schedule-buffer object.
