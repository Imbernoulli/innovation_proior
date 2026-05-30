# Context

## Research question

How can a diffusion model be turned into a photorealistic **text-to-image** generator and editor — one that takes a free-form natural-language caption and produces a realistic image, and that can also **edit** existing images (inpaint a masked region) conditioned on text? Diffusion models had just become the strongest approach for *class-conditional* image synthesis, beating GANs on standard benchmarks; the open question is how to condition them on arbitrary text (a far richer, harder-to-classify signal than a class label) and, critically, **which guidance strategy** best trades off photorealism, sample diversity, and faithfulness to the caption. Two candidates are on the table — guidance from a CLIP image-text model, and classifier-free guidance — and it is not known which produces better text-to-image samples.

## Background

**Diffusion models** (Sohl-Dickstein et al. 2015; Ho et al. 2020, DDPM; Nichol & Dhariwal 2021, improved DDPM). A forward process gradually corrupts a data sample `x_0 ∼ q(x_0)` into noise through a Markov chain
`q(x_t | x_{t-1}) = N(x_t; √α_t x_{t-1}, (1−α_t) I)`.
If each step's noise `1−α_t` is small, the true reverse posterior `q(x_{t-1}|x_t)` is approximately a diagonal Gaussian; if the total noise is large, `x_T ≈ N(0, I)`. A model `p_θ(x_{t-1}|x_t) = N(μ_θ(x_t), Σ_θ(x_t))` is trained to invert the chain, so sampling starts from `x_T ∼ N(0,I)` and denoises down to `x_0`. Rather than the variational lower bound directly, a re-weighted surrogate works better: noise `x_0` to `x_t ∼ q(x_t|x_0)` with `ε ∼ N(0,I)` and regress the model's noise prediction,
`L_simple = E_{t, x_0, ε}[ ‖ε − ε_θ(x_t, t)‖² ]`.
DDPM derive `μ_θ` from `ε_θ` and fix `Σ_θ` constant; they also show the score connection `∇_{x_t} log p(x_t) ∝ ε_θ(x_t, t)` (denoising score matching). Nichol & Dhariwal showed how to *learn* `Σ_θ`, yielding high quality with fewer sampling steps. Diffusion also handles **super-resolution** (SR3): the reverse model `p_θ(y_{t-1}|y_t, x)` conditions on a low-res image `x` by concatenating its bicubic upsample in the channel dimension.

**Classifier guidance** (Dhariwal & Nichol 2021, ADM). For a class-conditional model, sample quality improves by perturbing the reverse-process mean with the gradient of a separately-trained classifier's log-probability:
`μ̂_θ(x_t|y) = μ_θ(x_t|y) + s · Σ_θ(x_t|y) ∇_{x_t} log p_φ(y|x_t)`.
The **guidance scale** `s` trades quality for diversity — larger `s` sharpens samples toward the class but reduces variety. The classifier `p_φ` must itself be trained on *noised* images, since it is evaluated on the noisy intermediate `x_t`.

**Classifier-free guidance** (Ho & Salimans 2021). Avoids a separate classifier: during training, the conditioning label `y` is replaced by a null token `∅` with some fixed probability, so the same network learns both a conditional `ε_θ(x_t|y)` and an unconditional `ε_θ(x_t|∅)`. At sampling, the prediction is extrapolated along the conditional direction:
`ε̂_θ(x_t|y) = ε_θ(x_t|∅) + s · (ε_θ(x_t|y) − ε_θ(x_t|∅))`, `s ≥ 1`.
This form follows from an implicit classifier `p^i(y|x_t) ∝ p(x_t|y)/p(x_t)`, whose score is `∇_{x_t} log p^i ∝ ∇ log p(x_t|y) − ∇ log p(x_t) ∝ ε*(x_t|y) − ε*(x_t)` — i.e. the same mean-perturbing effect as classifier guidance, but using the model's own conditional vs. unconditional scores. Its appeal: the model leverages its own knowledge instead of a separate (often smaller) classifier, and it conditions easily on signals that are hard to classify.

**CLIP** (Radford et al. 2021). A joint text–image model with an image encoder `f(x)` and caption encoder `g(c)`, trained with a contrastive loss that pushes the dot product `f(x)·g(c)` high for matched `(x,c)` pairs and low for mismatched ones. Several works steer GANs and diffusion models toward a caption by following the gradient of this similarity.

**Text-to-image generation prior art.** GAN approaches (AttnGAN, DM-GAN, DF-GAN, XMC-GAN) train conditional GANs on captioned datasets. DALL-E (Ramesh et al. 2021) builds on VQ-VAE: train an autoregressive model over discrete image latent codes conditioned on text, then rank many samples with CLIP. Concurrent work trains text-conditional *discrete* diffusion over latent codes.

**Image editing / inpainting with diffusion.** Diffusion models can inpaint without task-specific training by sampling normally but overwriting the known region with a noised sample `q(x_t|x_0)` after each step (Sohl-Dickstein 2015; SDEdit) — but the model only ever sees a noised version of the context, which causes edge artifacts. Palette shows that training *directly* on the inpainting task produces seamless completions.

## Baselines

- **DALL-E** (autoregressive over discrete VQ-VAE codes + CLIP reranking): strong text-to-image, but autoregressive sampling over many tokens and reliance on expensive best-of-N CLIP reranking; no native editing. Gap: not a single end-to-end conditional sampler, expensive reranking.
- **Text-conditional GANs** (AttnGAN, XMC-GAN, …): fast sampling but weaker photorealism and compositional binding on open-domain captions. Gap: fidelity / compositionality.
- **Class-conditional diffusion with classifier guidance** (ADM, Dhariwal & Nichol): the diffusion SOTA for *labels*, but requires a separate noised classifier and is built for a fixed label set, not free-form text. Gap: needs a separate classifier; text is hard to classify.
- **Unnoised CLIP-guided diffusion** (community notebooks): steer a pretrained diffusion model with gradients from the *public* CLIP. Because public CLIP never saw noised images, the noisy intermediate `x_t` are out-of-distribution, so it needs hand-engineered augmentations and perceptual losses to yield recognizable samples. Gap: CLIP is not noise-aware.
- **Discrete text-conditional diffusion** (concurrent, over latent codes): competitive samples but operates on discrete latents rather than pixels.

## Evaluation settings

- **Datasets:** trained on the same large captioned image dataset as DALL-E; the upsampling and CLIP components use the CLIP dataset as well. Base generation at `64×64`, upsampled to `256×256`.
- **Benchmarks/metrics:** MS-COCO prompts for `64×64` comparisons — Precision/Recall, Inception Score / FID, and CLIP-score vs. FID to chart the diversity–fidelity trade-off across guidance scales. Human evaluations (pairwise, scored by Elo) for **photorealism** and **caption similarity** at `256×256`, used to pick guidance scales and to compare CLIP guidance against classifier-free guidance. Automated metrics computed against the COCO validation set.
- **Tasks:** zero-shot text-to-image; text-driven inpainting; SDEdit sketch-to-image; iterative scene construction by repeated inpainting.

## Code framework

The pieces that already exist: a Gaussian-diffusion trainer/sampler with an `ε`-prediction objective and a learned-variance reverse model (ADM-style UNet), a `p_sample_loop` that supports a mean-perturbing `cond_fn` hook, a Transformer text encoder, and CLIP-style image/caption encoders. The base UNet outputs `6` channels for an RGB image: `3` for the predicted noise `ε` and `3` for the variance interpolation. What is *not* yet decided: how text conditions the denoiser, and — the crux — how to steer the reverse process toward a caption. The slots below are empty.

```python
import torch as th

class TextConditionalUNet(th.nn.Module):
    """ADM-style denoiser. Predicts noise + variance for x_t at step t. How the caption
    enters the network is to be designed."""
    def forward(self, x_t, t, tokens=None, mask=None):
        # TODO: inject text conditioning; return cat([eps(3ch), var(3ch)], dim=1)
        pass

def guided_prediction(model, x_t, t, model_kwargs, guidance_scale):
    """Steer the reverse process toward the caption. The mechanism — gradient of an
    external scorer, or extrapolating the model's own conditional/unconditional
    predictions — is exactly the open design choice."""
    # TODO
    pass

def cond_fn(x, t, **kwargs):
    """Optional mean-perturbing hook used by p_sample_loop:
    μ̂ = μ + s·Σ·(this gradient). What it differentiates is to be decided."""
    # TODO
    pass

# Provided by the diffusion library:
# samples = diffusion.p_sample_loop(
#     model_fn, (batch, 3, H, W), clip_denoised=True,
#     model_kwargs=model_kwargs, cond_fn=cond_fn)
```
