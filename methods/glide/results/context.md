## Research question

How can a diffusion model be turned into a photorealistic **text-to-image** generator and editor — one that takes a free-form natural-language caption and produces a realistic image, and that can also **edit** existing images (inpaint a masked region) conditioned on text? Diffusion models had just become a strong approach for *class-conditional* image synthesis, matching or beating GANs on standard benchmarks. The question is how to condition such a model on arbitrary text — a far richer signal than a class label — and how to guide sampling so that samples are photorealistic, diverse, and faithful to the caption.

## Background

**Diffusion models** (Sohl-Dickstein et al. 2015; Ho et al. 2020, DDPM; Nichol & Dhariwal 2021, improved DDPM). A forward process gradually corrupts a data sample `x_0 ∼ q(x_0)` into noise through a Markov chain
`q(x_t | x_{t-1}) = N(x_t; √α_t x_{t-1}, (1−α_t) I)`.
If each step's noise `1−α_t` is small, the true reverse posterior `q(x_{t-1}|x_t)` is approximately a diagonal Gaussian; if the total noise is large, `x_T ≈ N(0, I)`. A model `p_θ(x_{t-1}|x_t) = N(μ_θ(x_t), Σ_θ(x_t))` is trained to invert the chain, so sampling starts from `x_T ∼ N(0,I)` and denoises down to `x_0`. Rather than the variational lower bound directly, a re-weighted surrogate is used: noise `x_0` to `x_t ∼ q(x_t|x_0)` with `ε ∼ N(0,I)` and regress the model's noise prediction,
`L_simple = E_{t, x_0, ε}[ ‖ε − ε_θ(x_t, t)‖² ]`.
DDPM derive `μ_θ` from `ε_θ` and fix `Σ_θ` constant; they also show the score connection `∇_{x_t} log p(x_t) ≈ -ε_θ(x_t,t)/σ_t`, with `σ_t = √(1-\bar α_t)` (denoising score matching). Nichol & Dhariwal *learn* `Σ_θ`, yielding high quality with fewer sampling steps. Diffusion also handles **super-resolution** (SR3): the reverse model `p_θ(y_{t-1}|y_t, x)` conditions on a low-res image `x` by concatenating its bicubic upsample in the channel dimension.

**Classifier guidance** (Dhariwal & Nichol 2021, ADM). For a class-conditional model, sample quality improves by perturbing the reverse-process mean with the gradient of a separately-trained classifier's log-probability:
`μ̂_θ(x_t|y) = μ_θ(x_t|y) + s · Σ_θ(x_t|y) ∇_{x_t} log p_φ(y|x_t)`.
The **guidance scale** `s` trades quality for diversity — larger `s` sharpens samples toward the class but reduces variety. The classifier `p_φ` is trained on *noised* images, since it is evaluated on the noisy intermediate `x_t`.

**Classifier-free guidance** (Ho & Salimans 2021). Avoids a separate classifier: during training, the conditioning label `y` is replaced by a null token `∅` with some fixed probability, so the same network learns both a conditional `ε_θ(x_t|y)` and an unconditional `ε_θ(x_t|∅)`. At sampling, the prediction is extrapolated along the conditional direction:
`ε̂_θ(x_t|y) = ε_θ(x_t|∅) + s · (ε_θ(x_t|y) − ε_θ(x_t|∅))`, `s ≥ 1`.
This form follows from an implicit classifier `p^i(y|x_t) ∝ p(x_t|y)/p(x_t)`. Its score is `∇_{x_t} log p^i = ∇ log p(x_t|y) − ∇ log p(x_t) ≈ -(ε*(x_t|y) − ε*(x_t|∅))/σ_t`; converting the guided score back into the `ε` parameterization gives the extrapolation above. The effect is the same mean-perturbing guidance as classifier guidance, using the model's own conditional vs. unconditional scores.

**CLIP** (Radford et al. 2021). A joint text–image model with an image encoder `f(x)` and caption encoder `g(c)`, trained with a contrastive loss that pushes the dot product `f(x)·g(c)` high for matched `(x,c)` pairs and low for mismatched ones. Several works steer GANs and diffusion models toward a caption by following the gradient of this similarity.

**Text-to-image generation prior art.** GAN approaches (AttnGAN, DM-GAN, DF-GAN, XMC-GAN) train conditional GANs on captioned datasets. DALL-E (Ramesh et al. 2021) builds on VQ-VAE: train an autoregressive model over discrete image latent codes conditioned on text, then rank many samples with CLIP. Concurrent work trains text-conditional *discrete* diffusion over latent codes.

**Image editing / inpainting with diffusion.** Diffusion models can inpaint without task-specific training by sampling normally but overwriting the known region with a noised sample `q(x_t|x_0)` after each step (Sohl-Dickstein 2015; SDEdit); here the model sees a noised version of the context. Palette trains *directly* on the inpainting task to produce completions.

## Baselines

- **DALL-E** (autoregressive over discrete VQ-VAE codes + CLIP reranking): text-to-image via autoregressive sampling over many tokens and best-of-N CLIP reranking.
- **Text-conditional GANs** (AttnGAN, XMC-GAN, …): fast feed-forward sampling on open-domain captions.
- **Class-conditional diffusion with classifier guidance** (ADM, Dhariwal & Nichol): diffusion synthesis for a fixed label set, using a separately-trained noised classifier.
- **CLIP-guided diffusion** (community notebooks): steer a pretrained diffusion model with gradients from the *public* CLIP, using augmentations and perceptual losses on the noisy intermediate `x_t`.
- **Discrete text-conditional diffusion** (concurrent, over latent codes): operates on discrete latents rather than pixels.

## Evaluation settings

- **Datasets:** large captioned image data of the kind used by DALL-E is the natural training source for the generator and upsampler; a noised CLIP guide needs image-text pairs from both CLIP-scale data and the captioned generator data. Comparison points are `64×64` base images and `256×256` images after super-resolution.
- **Benchmarks/metrics:** MS-COCO prompts for `64×64` comparisons — Precision/Recall, Inception Score / FID, and CLIP-score vs. FID to chart the diversity–fidelity trade-off across guidance scales. Human pairwise evaluations scored by Elo for **photorealism** and **caption similarity** at `256×256`. Automated metrics computed against the COCO validation set.
- **Tasks:** zero-shot text-to-image; text-driven inpainting; SDEdit sketch-to-image; iterative scene construction by repeated inpainting.

## Code framework

A Gaussian-diffusion trainer/sampler already supplies an `ε`-prediction objective, learned-variance reverse steps, a `p_sample_loop` accepting model kwargs, optional `cond_fn` gradients, optional `denoised_fn` projections, and a separate low-resolution-to-high-resolution sampler. CLIP-style image/caption encoders can supply an external differentiable score. The base denoiser returns `6` channels for an RGB image: `3` for predicted noise `ε` and `3` for variance interpolation. A minimal harness keeps the text, guidance, editing, and upsampling hooks generic.

```python
import torch as th

def build_text_kwargs(model, prompt, batch_size, text_ctx, device, include_null=False):
    # TODO: encode text conditioning for the sampler.
    pass

def model_fn(x_t, ts, **kwargs):
    # TODO: call the denoiser and return cat([eps(3ch), variance(3ch)], dim=1).
    pass

def cond_fn(x, t, **kwargs):
    # TODO: optional external-score gradient for mean perturbation.
    # The sampler will use it as μ̂ = μ + Σ·cond_fn(x_t,t).
    pass

def denoised_fn(x_start):
    # TODO: optional projection on the predicted clean image.
    pass

def build_upsample_kwargs(model_up, prompt, low_res, batch_size, text_ctx, device):
    # TODO: pack low-resolution conditioning and text conditioning for the upsampler.
    pass

# samples = diffusion.p_sample_loop(
#     model_fn, shape, device=device, clip_denoised=True,
#     model_kwargs=model_kwargs, cond_fn=cond_fn, denoised_fn=denoised_fn)
# up_samples = diffusion_up.ddim_sample_loop(
#     model_up, up_shape, noise=noise, device=device, clip_denoised=True,
#     model_kwargs=upsample_kwargs, cond_fn=None)
```
