# Synthesis — Latent Diffusion Models (LDM)

## The pain point (what existed, where it hurt)

By ~2021, high-resolution image synthesis was split among three model families, each with a fatal flaw:
- **GANs** (Goodfellow 2014; BigGAN/Brock 2019; StyleGAN/Karras 2019): fast single-pass sampling, sharp images, but adversarial training is unstable (mode collapse, careful tuning), and they don't cover the full data distribution — poor recall.
- **Likelihood models** — VAEs (Kingma 2013), normalizing flows (Dinh 2017), autoregressive models (PixelCNN/PixelRNN, van den Oord 2016): well-behaved density estimation but either lower sample quality (VAE/flow) or, for ARMs, sequential per-pixel sampling that is hopelessly slow at high resolution.
- **Diffusion models** (Sohl-Dickstein 2015; DDPM Ho 2020; ADM Dhariwal & Nichol 2021; score-based Song 2021): new SOTA in density estimation AND sample quality; no mode collapse; a UNet backbone fits the spatial inductive bias of images; even unconditional DMs can do inpainting/colorization at test time. BUT they run entirely in **pixel space**, so training and sampling require repeated UNet forward+backward passes over full-resolution RGB tensors. Training a strong DM takes 150–1000 V100-days; generating 50k samples ~5 days on one A100; sampling needs 25–1000 sequential steps.

**The diagnostic finding that sets up everything:** likelihood-based models are mode-covering, so they spend capacity proportional to the bits in the data — and pixel data is dominated by **imperceptible high-frequency detail** (Dieleman 2020 typicality; Salimans 2017 PixelCNN++). A rate-distortion view of a *trained* pixel DM shows two regimes: a first regime where a large fraction of the bits buys aggressive **perceptual compression** (removing high-frequency detail) with almost no semantic change, and a second regime where the remaining bits do the actual **semantic/conceptual** modeling. DDPM's reweighted objective already tries to undersample the high-SNR (near-clean) steps to focus on semantics, but the network still *evaluates in pixel space at every step*, so the compute is still dominated by perceptual detail.

**Goal a solution must hit:** drastically cut training+inference compute of DMs **without** losing their quality/flexibility, and keep the guiding/conditioning tricks. Democratize: make it trainable on a single GPU.

## The core idea (insight before method)

If the first regime (perceptual compression) is nearly semantics-free and eats most of the compute, then **don't make the diffusion model do it**. Hand perceptual compression to a cheap, train-once autoencoder that maps images to a lower-dimensional, *perceptually equivalent* latent space; run the (expensive, sequential) diffusion model only in that latent space, where every spatial location already carries semantic content. The DM keeps the same UNet, same objective — it just operates on `z = E(x)` instead of `x`.

Why this saves compute concretely: cost of one UNet pass scales with the spatial size of its input. Downsample by factor f spatially (H/f × W/f), and the dominant conv/attention work drops by ~f² per layer (attention by ~f⁴ at the finest level). With f=4 or 8 that's a 16–64× reduction *per step*, multiplied over T steps and over training. Because the latent keeps 2D structure, the UNet's convolutional inductive bias still applies — unlike VQGAN/DALL·E which flatten the latent to a 1D sequence for an autoregressive transformer.

## Why NOT the obvious alternatives (walls hit)

1. **Just sample faster (DDIM, etc.)** — addresses inference step count but every step is still a full pixel-space UNet pass; training cost untouched. Insufficient.
2. **Two-stage with autoregressive transformer over a discrete latent (VQ-VAE-2 Razavi 2019; VQGAN/Taming Esser 2020; DALL·E Ramesh 2021)** — exactly the right "compress then model" decomposition, but the second-stage model is an AR transformer whose cost is quadratic in sequence length, so it *forces extreme compression* (e.g. f=16) to keep the token sequence short → throws away detail, needs billions of parameters, and the 1D raster ordering ignores the latent's 2D structure. The compression level is dictated by the transformer's appetite, not by what preserves quality. LDM's fix: replace the AR transformer with a **convolutional DM**, which scales gently in spatial dimension, so we're free to choose a *mild* compression (f=4/8) that keeps high-fidelity reconstructions.
3. **Jointly learn encoder/decoder + score prior (LSGM, Vahdat 2021)** — learns the latent space and the diffusion prior together, but then you must delicately balance reconstruction quality against prior-fit; the latent space keeps shifting under the DM. LDM's fix: **freeze** the autoencoder, train the DM in a *fixed* latent space → no reconstruction-vs-prior weighting, faithful reconstructions, and one universal autoencoder reusable across many DM trainings/tasks.

## The method, end to end

### Stage 1 — perceptual autoencoder (train once, then freeze)
Encoder E downsamples x∈R^{H×W×3} by f=2^m to z∈R^{h×w×c}; decoder D reconstructs. Trained with **(a)** pixel L1 + **(b)** LPIPS perceptual loss (Zhang 2018) + **(c)** patch-based adversarial loss (PatchGAN, Isola 2017) with a discriminator. (b)+(c) keep reconstructions on the image manifold and avoid the blur that pure L2/L1 cause (L2 minimizes mean pixel error → spatial averaging → blur). Adaptive GAN weight: d_weight = ||∇nll||/||∇g|| balances the reconstruction-NLL gradient against the adversarial gradient at the last decoder layer.

**Regularization of the latent (two variants):**
- **KL-reg**: encoder outputs mean+logvar, a VAE-style diagonal Gaussian posterior, with a *very* small KL pull toward N(0,1) (weight ~1e-6). Just enough to keep the latent zero-centered and low-variance, not enough to hurt reconstruction. z is sampled: z = μ + σ·ε.
- **VQ-reg**: a vector-quantization layer (van den Oord 2017) with a large codebook; the quantization is *absorbed into the decoder* (z taken before quantization), so it behaves like a VQGAN with the quantizer as D's first layer. High codebook dim → mild regularization.
Why mild either way: heavy regularization buys a nicer prior-friendly latent but costs reconstruction fidelity, which becomes a hard ceiling on final image quality. Since the DM (not an AR transformer) models the prior, we don't *need* a heavily-regularized latent, so we keep regularization tiny and reconstruction near-perfect.

### Stage 2 — diffusion in latent space
DDPM machinery, verbatim, on z instead of x.

Forward: q(z_t|z_0)=N(α_t z_0, σ_t² I), reparam z_t = √(ᾱ_t) z_0 + √(1−ᾱ_t) ε. (code uses α_t²↔ᾱ_t convention: sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod.)

ELBO (Appendix B) decomposes over t into KL terms; parameterizing p(z_{t−1}|z_t) via the true posterior q(z_{t−1}|z_t,z_0) with z_0 replaced by an estimate, the reconstruction terms become Σ_t ½(SNR(t−1)−SNR(t))||z_0 − z_θ(z_t,t)||². Reparameterizing to ε-prediction, ε_θ=(z_t−α_t z_θ)/σ_t, gives ||z_0−z_θ||² = (σ_t²/α_t²)||ε−ε_θ||². **Reweighting** = drop the SNR-difference weight, weight all t equally → simplified objective:
L = E_{E(x),ε,t} ||ε − ε_θ(z_t, t)||². This reweighting is exactly what makes DMs spend effort on perceptually meaningful (mid-SNR) steps rather than the trivially-denoisable high-SNR ones — and mirrors denoising score matching.

Backbone ε_θ is a **time-conditional UNet** (Ronneberger 2015), built mostly from 2D convs (uses the latent's 2D structure), with self-attention at low-resolution levels, timestep embedding injected via FiLM-like scale/shift in residual blocks.

### Conditioning — cross-attention
To model p(z|y) for general y (text, layout, class, low-res image, semantic map): a domain-specific encoder τ_θ maps y to a sequence τ_θ(y)∈R^{M×d_τ} (for text: an unmasked transformer with token+positional embeddings, BERT tokenizer; for class: a single learnable embedding; for spatially-aligned y like low-res/segmentation: just **concatenate** to the UNet input, τ = identity). The UNet's intermediate feature map φ_i(z_t)∈R^{N×d} is the query side; conditioning is the key/value side:
Attention(Q,K,V)=softmax(QKᵀ/√d)·V, with Q=W_Q·φ_i(z_t), K=W_K·τ_θ(y), V=W_V·τ_θ(y).
This drops a cross-attention layer into each transformer block of the UNet (alternating self-attn, cross-attn, position-wise MLP — without cross-attn+MLP it's exactly the ADM "ablated UNet"). Conditional loss: L = E_{E(x),y,ε,t} ||ε − ε_θ(z_t,t,τ_θ(y))||², τ_θ and ε_θ jointly optimized.
Why cross-attention not concat for non-aligned modalities: text/class have no spatial correspondence to the image grid, so you can't concatenate; attention lets every spatial query pull from the whole conditioning sequence, modality-agnostic.

### Guidance (Appendix D + classifier-free)
- Post-hoc **image guiding**: ε̂ ← ε_θ(z_t,t) + √(1−α_t²) ∇_{z_t} log p_Φ(y|z_t) (Dhariwal & Nichol's classifier-guidance formula), reinterpreted with p_Φ(y|T(D(z_0(z_t)))) = a general image-to-image guider, e.g. Gaussian → L2 regression −½||y−T(D(z_0(z_t)))||², or LPIPS for perceptual super-res.
- **Classifier-free guidance** (Ho & Salimans 2021): train ε_θ with conditioning randomly dropped; at sampling ε̂ = ε_θ(z,c) + s·(ε_θ(z,c) − ε_θ(z,∅)); trades diversity for fidelity, no separate classifier. Latent classifiers (for classifier guidance) are also trained cheaply in latent space.

### The scale_factor detail
SNR of the latent ∝ Var(z)/σ_t². A KL latent can have large variance → high SNR → DM allocates detail too early in the reverse process, and convolutional sampling at >256² breaks. Fix: rescale z by component-wise std estimated from the first batch so the latent has unit variance: z ← z/σ̂ (in code: scale_factor; for VQ latents var≈1 already so no rescale).

## Design-decision → why table (with rejected alternatives)

| Decision | Why / what breaks otherwise |
|---|---|
| Diffuse in learned latent, not pixels | Perceptual-compression regime is ~semantics-free yet eats most compute; moving it out of the DM cuts cost ~f² per step with negligible quality loss. |
| Separate, frozen autoencoder (vs LSGM joint) | Joint training forces a delicate reconstruction-vs-prior weighting and a moving target; frozen space = stable, faithful, reusable across tasks. |
| Convolutional DM (vs AR transformer over latent, VQGAN/DALL·E) | AR transformer cost is quadratic in sequence length → forces extreme compression (f=16) → detail loss + billions of params + 1D ordering ignores 2D structure. Conv DM scales gently → choose mild f=4/8, keep fidelity. |
| Mild compression f=4–8 | f=1,2 → DM still does perceptual compression, slow training; f≥16 → first stage loses information, quality ceiling. f=4/8 is the sweet spot (FID gap of 38 between LDM-1 and LDM-8 at 2M steps). |
| LPIPS + PatchGAN first-stage loss (not L2/L1) | L2 minimizes mean pixel error → spatial averaging → blur. Perceptual+adversarial keep reconstructions on the image manifold, sharp. |
| Tiny regularization (KL ~1e-6 / large codebook) | Heavy reg helps prior but caps reconstruction fidelity = hard ceiling on samples; DM (not AR) models prior, so heavy reg unnecessary. |
| KL-reg vs VQ-reg | KL: continuous Gaussian latent, slightly better reconstruction. VQ: discrete codebook absorbed into decoder; empirically sometimes better LDM samples despite slightly worse reconstruction. Offer both. |
| ε-prediction + reweighted (simple) L2 objective | Reweighting drops SNR-difference weights → equal weight per step → focuses capacity on perceptually relevant mid-SNR steps; ε-param matches denoising score matching, stable. |
| Time-conditional UNet backbone, 2D convs | Convolutional inductive bias matches the latent's spatial structure (which AR-over-1D throws away); attention only at coarse levels to bound cost. |
| Cross-attention conditioning | General/multi-modal; non-spatial conditions (text, class) can't be concatenated to the grid; attention is modality-agnostic, every query attends to whole condition sequence. |
| τ_θ = identity + concat for spatial conditions | Low-res/segmentation maps are spatially aligned to the latent grid → concatenation is the natural, cheap injection; no attention needed. |
| scale_factor (unit-variance latent) | Latent variance sets SNR; un-normalized KL latent → SNR too high → detail allocated too early, convolutional >256² sampling degrades. |
| QKᵀ/√d scaling in attention | Standard: keeps softmax logits O(1) as d grows; without it large-d dot products saturate softmax → vanishing gradients. |
| Adaptive GAN weight ||∇nll||/||∇g|| | Auto-balances reconstruction vs adversarial gradient magnitudes at the last decoder layer so neither term dominates training. |

## Load-bearing ancestors (for context.md baselines)
- DDPM (Ho, Jain, Abbeel 2020): forward/reverse Markov chain, ε-param, simplified L2; pixel-space, expensive.
- DPM (Sohl-Dickstein 2015): original diffusion/thermodynamics formulation.
- ADM "Diffusion beats GANs" (Dhariwal & Nichol 2021): SOTA class-conditional pixel DM; ablated UNet; classifier guidance; 150–1000 V100-days cost = the headline pain.
- Score-based SDE (Song 2021): score matching ↔ diffusion, test-time conditioning.
- VQ-VAE (van den Oord 2017): discrete latent + codebook.
- VQ-VAE-2 (Razavi 2019), VQGAN/Taming (Esser 2020), DALL·E (Ramesh 2021): two-stage compress-then-AR-transformer; the direct foil.
- LSGM (Vahdat 2021): joint latent+score; the joint-training foil.
- VAE (Kingma & Welling 2013): the KL-reg ancestor.
- U-Net (Ronneberger 2015): backbone.
- Transformer/attention (Vaswani 2017): cross-attention mechanism.
- LPIPS (Zhang 2018), PatchGAN (Isola 2017): first-stage losses.
- Classifier-free guidance (Ho & Salimans 2021).

## Code grounding (files read)
- ddpm.py: DDPM base (register_schedule, q_sample, p_losses, ε-target) + LatentDiffusion (get_input → encode_first_stage → get_first_stage_encoding scale_factor, apply_model, p_losses with cross-attn cond, DiffusionWrapper concat/crossattn).
- autoencoder.py: AutoencoderKL (encode→DiagonalGaussianDistribution, quant_conv 2*z_channels, post_quant_conv, decode), VQModel.
- distributions.py: DiagonalGaussianDistribution (kl, sample = μ+σε).
- attention.py: CrossAttention (scale=dim_head**-0.5, to_q/k/v, softmax), BasicTransformerBlock (self-attn, cross-attn, FF), SpatialTransformer.
- contperceptual.py: LPIPSWithDiscriminator (L1 + LPIPS + adaptive-weight PatchGAN + KL).
