# Context: text-guided diffusion sampling and its guidance scale (circa 2022-2024)

## Research question

Modern text-to-image diffusion models are trained so that a single noise-prediction network
`eps_theta(x_t, c)` can be queried with a text condition `c` or with the null condition
`c = empty`. Generating an image that follows a prompt can be posed as solving the conditional
probability-flow ODE with the conditional score. In practice, querying the conditional score
alone produces images that follow the prompt only weakly. The standard approach is
classifier-free guidance, which amplifies the conditional direction by a scale `w`, and the
regime used for high-quality, well-aligned samples is a "moderately high" scale, typically
`w` in `[5, 30]` (and orders of magnitude larger for score distillation). In that high-`w`
regime the sampler exhibits mode collapse and reduced sample diversity, over-saturated colors,
an accumulation of error across sampling steps, and the breakdown of DDIM inversion, so that
the inversion-based route to image editing (invert a real image to its latent, then resample
with an edited prompt) does not hold once `w > 1`.

These behaviors are commonly treated as inherent properties of diffusion models. The question
this study takes up is how guidance is injected into the sampler: how the conditional and
unconditional predictions should be combined, and which predicted noise should drive each half
of the reverse step. Each existing piece below supplies part of the machinery.

## Background

By this time diffusion models are the dominant image generators. A model learns to reverse a
forward Gaussian noising process; sampling solves the reverse SDE or the equivalent
probability-flow ODE. Under the variance-preserving (VP) convention with cumulative signal
rate `bar_alpha_t`, the network is trained as a noise predictor `eps_theta`, and the two
load-bearing identities are:

- **Tweedie's formula** (Efron 2011; Robbins): the posterior mean of the clean signal given a
  noisy `x_t` is recoverable in closed form from the predicted noise,

  ```
  x_hat_null(x_t) = E[x_0 | x_t, empty] = (x_t - sqrt(1 - bar_alpha_t) * eps_null(x_t)) / sqrt(bar_alpha_t),
  ```

  i.e. one network call lands a *denoised estimate* on the clean data manifold `M`. Here
  `eps_null := eps_theta(x_t, empty)` and, for the conditional case,
  `eps_c := eps_theta(x_t, c)`.

- **DDIM** (Song et al. 2020): a non-Markovian, deterministic sampler whose single reverse
  step is "denoise, then renoise." For the unconditional process,

  ```
  x_hat_null = (x_t - sqrt(1 - bar_alpha_t) * eps_null) / sqrt(bar_alpha_t)          # denoise to M
  x_{t-1}    = sqrt(bar_alpha_{t-1}) * x_hat_null + sqrt(1 - bar_alpha_{t-1}) * eps_null   # renoise to M_{t-1}
  ```

  iterated for `t = T ... 1`. Because the renoise reuses the same predicted noise, the
  unconditional DDIM map is approximately invertible (`x_t` recoverable from `x_{t-1}` when
  `eps_null(x_t) ~ eps_null(x_{t-1})`), which is what makes inversion-based editing possible.

The geometric picture that organizes the rest: a sample `x_t` lives near a *noisy* manifold
`M_t`; Tweedie's denoise maps it to the clean manifold `M`; the renoise carries it to the
next noisy manifold `M_{t-1}`. A useful approximation in the inverse-problem literature is
that `M` is locally **piecewise linear**, so the tangent at a denoised point `x_hat` is a good
local model of `M`, and short moves *along* segments between nearby denoised points stay on
`M`, while moves that leave those segments leave `M`.

There is a parallel line on **diffusion model-based inverse problem solvers (DIS)**: given a
loss `l(x)` (usually a data-likelihood term), solve `min_{x in M} l(x)` by interleaving a
data-consistency gradient step with the diffusion sampler, navigating so as to reduce cost
while staying on the correct clean manifold. The diagnostic phenomenon documented at the time
is empirical: at the high `w` used for quality, the per-step denoised estimate under guidance
shows a sudden shift and intense color saturation in the early (high-noise) phase of sampling,
and DDIM inversion under `w > 1` produces visibly distorted reconstructions.

## Baselines

**Classifier guidance (Dhariwal & Nichol 2021).** Steer the unconditional score by the
gradient of a separately trained, noise-aware classifier `p(c | x_t)`:
`score <- score + s * grad_x log p(c | x_t)`, with `s ~ 10` for strong conditioning. Requires
training an extra classifier on noisy data.

**Classifier-free guidance, CFG (Ho & Salimans 2022).** Remove the classifier by sampling
from a *sharpened* posterior `p^w(x | c) ∝ p(x) * p(c | x)^w`. Taking the score and
parameterizing with the noise predictor gives a linear mix of the conditional and
unconditional predictions,

```
eps_cfg(x_t) = eps_null(x_t) + w * (eps_c(x_t) - eps_null(x_t)),
```

with `w > 1` for strong alignment. In a DDIM step this `eps_cfg` is substituted for
`eps_null` in **both** the Tweedie denoise and the renoise. The useful regime is the
moderately-high band `w` in `[5, 30]`. The conditional direction
`delta(x_t) := eps_c(x_t) - eps_null(x_t)` is non-negligible, so under guidance the
inversion-stability quantity `eps_cfg(x_t) ~ eps_cfg(x_{t-1})` varies with `w`.

**Patches around CFG inversion (Mokady et al. 2023, null-text inversion; Wallace et al. 2023,
EDICT).** To make inversion-based editing usable, null-text optimization re-fits the null
embedding per image, and EDICT runs coupled, exactly-invertible transforms. Both keep the same
`eps_cfg`-driven trajectory.

**Schedule-based guidance adjustments (Kynkaanniemi et al. 2024; SD-community schedules).**
Keep the CFG trajectory but vary the strength over time — e.g. apply guidance only in a
limited middle interval of sampling, on the observation that early high-noise guidance hurts
diversity and late guidance does little. These reshape *how much* guidance is applied along
the same trajectory.

**Diffusion inverse-problem machinery.** Diffusion posterior sampling, DPS (Chung et al. 2023),
enforces data consistency with the *manifold-constrained gradient*
`grad^mcg_{x_t} l(x_t) := grad_{x_t} l(x_hat_t)` evaluated at the Tweedie-denoised estimate,
keeping the update on the noisy manifold under a linear-manifold assumption; it differentiates
through the score network and targets measurement likelihoods. Decomposed diffusion sampling,
DDS (Chung et al. 2024), shows that the DPS+MCG step is equivalent to a one-step projected
gradient on the tangent space at the denoised estimate, so the gradient may be taken with
respect to the denoised estimate `x_hat` directly — bypassing the score Jacobian. Score
distillation, SDS (Poole et al. 2022), independently found that the gradient of the diffusion
training loss is a noise residual `(eps_hat - eps)` times the U-Net Jacobian times a generator
Jacobian, and that *omitting the U-Net Jacobian* still yields an effective optimization
direction. These supply a stable, Jacobian-free way to take a loss-reducing step on a denoised
estimate; they have been deployed for measurement inverse problems and 3D distillation.

## Evaluation settings

Natural yardsticks already in use at the time:

- **Text-to-image generation.** Stable Diffusion v1.5 and SDXL with frozen weights; DDIM
  sampling at 50 neural function evaluations (NFE), and accelerated/distilled regimes
  (SDXL-Turbo, SDXL-Lightning at ~4-6 NFE; DPM++ 2M at ~20 NFE). Prompts drawn from COCO
  captions (10k images) and a fixed prompt set. Metrics: **FID** against a reference image set
  (lower better) and **CLIP score** (image-text cosine similarity, higher better);
  ImageReward where used.
- **DDIM inversion / editing.** Reconstruct real images (COCO-2014, FFHQ 512x512) after
  inversion; metrics PSNR and RMSE (and LPIPS for matching scales). Editing swaps a source
  concept in the prompt and resamples from the inverted latent.
- **Text-conditioned linear inverse problems.** Super-resolution (x8), motion and Gaussian
  deblurring (61x61 kernels), inpainting (free-form masks) on FFHQ, with a latent
  inverse-problem solver (PSLD) as the base; metrics FID, LPIPS, PSNR.
- Protocol for comparing two guidance rules whose scales live in different ranges: match the
  scales by generating from the same seed and pairing the values that minimize LPIPS distance,
  so the comparison is at equal effective guidance strength.

## Code framework

A guidance rule plugs into the standard DDIM text-to-image harness already used for the
baselines. What exists before the method: a frozen latent-diffusion backbone with a U-Net
noise predictor and a VAE decoder; a tokenizer/text-encoder producing the conditional and null
embeddings; a noise scheduler exposing `bar_alpha_t`; and the DDIM "denoise then renoise"
loop. Exactly two predictions per step are available — the unconditional `eps_null` and the
conditional `eps_c` from one batched network call — and the loop is `denoise to the clean
manifold, then renoise to the next noisy manifold`. How the conditional and unconditional
predictions are combined, and which predicted noise drives each half of the DDIM step, is the
open slot.

```python
import torch


class DiffusionSampler:
    """Frozen latent-diffusion T2I sampler. Owns the U-Net noise predictor, the VAE,
    the text encoder, and the DDIM scheduler. The per-step guidance rule is not fixed."""

    def get_text_embed(self, null_prompt, prompt):
        """Encode the null prompt and the text prompt -> (uc, c)."""
        ...

    def initialize_latent(self):
        """Sample x_T ~ N(0, I) in latent space."""
        ...

    def predict_noise(self, x_t, t, uc, c):
        """One batched U-Net call: returns (eps_null, eps_c) for this step.
        Exactly two predictions are available per step (no extra NFE budget)."""
        ...

    def alpha(self, t):
        """Cumulative signal rate bar_alpha_t from the scheduler."""
        ...

    def decode(self, z0):
        """VAE-decode a clean latent estimate to an image."""
        ...

    @torch.no_grad()
    def sample(self, guidance_scale, prompt=["", ""], **kwargs):
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        x_t = self.initialize_latent()

        for t in self.scheduler.timesteps:
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)
            eps_null, eps_c = self.predict_noise(x_t, t, uc, c)

            # TODO: the guidance rule we will design. Given (eps_null, eps_c, guidance_scale),
            #       form the denoised estimate (Tweedie) and the renoised next latent x_{t-1},
            #       i.e. fill the denoise and renoise halves of the DDIM step.
            #   x0_hat = (x_t - sqrt(1-at) * <eps for denoise>) / sqrt(at)
            #   x_t    = sqrt(at_prev) * x0_hat + sqrt(1-at_prev) * <eps for renoise>
            raise NotImplementedError

        img = self.decode(x0_hat)
        return (img / 2 + 0.5).clamp(0, 1).detach().cpu()
```

The outer loop supplies `eps_null` and `eps_c` each step; the marked slot is where the
combination and the two halves of the DDIM step remain undecided.
