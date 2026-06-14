## Research question

Text-to-image diffusion follows a prompt only weakly if I sample the conditional model alone, so
every system uses *guidance*: at each reverse step I have two predictions from one frozen U-Net —
the unconditional noise `eps_uc = eps_theta(z_t, empty)` and the conditional `eps_c =
eps_theta(z_t, c)` — and I combine them to push the trajectory toward the prompt. The single thing
being designed is **the guidance rule**: given those two predictions per step, how to form the
denoised estimate and how to renoise to the next latent. Everything else — the frozen SD v1.5 /
SD v2 / SDXL weights, the DDIM step count, the prompt set, the VAE decode, the metrics — is fixed.
The score I am judged on is per-variant **FID** (lower is better), with **CLIP score** (higher is
better) reported alongside; a good rule should improve image quality without giving up the
prompt-following that guidance buys.

## Prior art before the first rung (the guidance lineage)

The first rung reacts to standard classifier-free guidance and its discontents; that is the
lineage to hold in mind.

- **Classifier guidance (Dhariwal & Nichol 2021, arXiv:2105.05233).** Steer the unconditional
  score with the input-gradient of a separately trained, noise-aware classifier `p(c|z_t)`:
  `score <- score + s * grad_z log p(c|z_t)`, with `s ~ 10` for strong conditioning. *Gap:* needs a
  second network trained on noisy latents, and the update direction is an adversarial-attack-like
  classifier gradient rather than a clean generative direction.
- **Classifier-free guidance, CFG (Ho & Salimans 2022, arXiv:2207.12598).** Drop the classifier:
  a single network trained with the condition randomly nulled produces both `eps_uc` and `eps_c`,
  and the guided noise is the linear mix `eps_g = eps_uc + w (eps_c - eps_uc)`. In a DDIM step this
  `eps_g` is substituted for the noise in **both** the Tweedie denoise and the renoise. *Gap:* the
  useful regime is a moderately-high scale `w` in roughly `[5, 30]`, and in that band the sampler
  over-saturates colors, collapses modes, accumulates trajectory error, and breaks DDIM
  invertibility — pathologies widely treated as inherent to diffusion.
- **The geometric reading that the first rung needs.** A DDIM step is "denoise (Tweedie to the
  clean manifold `M`), then renoise (lift to the next noisy manifold)." The clean manifold is
  locally piecewise-linear, so the segment between two nearby denoised points stays on `M`. With
  `eps_g`, the denoised estimate is `x_hat_g = (1-w) x_hat_uc + w x_hat_c`; for `w > 1` this is an
  *extrapolation* past the `[x_hat_uc, x_hat_c]` segment, off `M`. And the renoise reuses the same
  guided `eps_g`, injecting an off-manifold noise direction. These two off-manifold sources are the
  opening the first rung exploits.

## The fixed substrate

A frozen latent-diffusion text-to-image stack is the substrate and must not be touched: the U-Net
noise predictor and VAE for each variant (SD v1.5, SD v2-base, SDXL, frozen weights), the
tokenizer/text-encoder producing the null and conditional embeddings, the DDIM noise scheduler
exposing the cumulative signal rate `bar_alpha_t`, and the "denoise then renoise" reverse loop run
for a fixed step budget. Exactly two predictions are available per step from one batched network
call — `eps_uc` and `eps_c` — and that budget is fixed (no extra neural function evaluations
allowed). The harness also fixes the prompt set (COCO captions), the seed, the guidance-scale
argument it passes in, and all evaluation code (FID against a reference Inception-stats set, CLIP
score with ViT-B/32).

## The editable interface

Exactly one region per variant is editable: the `BaseDDIMCFGpp` solver class registered as
`ddim_cfg++`. For SD v1.5 / SD v2 it is the `sample()` method of `BaseDDIMCFGpp(StableDiffusion)`
in `latent_diffusion.py` (the marked block, lines 621-679); for SDXL it is the `reverse_process()`
method of `BaseDDIMCFGpp(SDXL)` in `latent_sdxl.py` (lines 713-755). Every method on the ladder is
a fill of this same contract, with these helpers provided by the parent class:

- `self.get_text_embed(null_prompt, prompt)` -> `(uc, c)` (SD path);
- `self.initialize_latent()` / `self.initialize_latent(size=...)` -> `z_T ~ N(0, I)`;
- `self.predict_noise(zt, t, uc, c)` -> `(eps_uc, eps_c)` — the two predictions, one batched call;
- `self.alpha(t)` -> `bar_alpha_t` (SD path); `self.scheduler.alphas_cumprod[t]` (SDXL path);
- `self.scheduler.timesteps`, `self.skip`, `self.decode(z)`, `self.vae_scale_factor`.

The contract: combine `eps_uc` and `eps_c` into a denoised estimate `z0t` (Tweedie) and a renoised
next latent `zt`, iterate over `self.scheduler.timesteps`, and return the decoded clean estimate.
The rule may change how the two predictions are combined, which prediction drives the renoise, or
how guidance varies with time — but not the prompt set, the weights, the NFE budget, or the
evaluation code. The starting point is the scaffold default: an unfilled stub that raises
`NotImplementedError`. The first rung replaces it with the CFG++ fill below; later rungs replace
exactly this block and nothing else.

```python
# EDITABLE region of CFGpp-main/latent_diffusion.py (lines 621-679) — scaffold DEFAULT (stub)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    # The baseline CFG++ uses unconditional noise (noise_uc) for renoising to keep
    # the trajectory on the data manifold. Improve upon this approach.
    #
    # Helpers from the parent class:
    #   self.get_text_embed(null_prompt, prompt) -> (uc, c)
    #   self.initialize_latent()                 -> z_T ~ N(0, I)
    #   self.predict_noise(zt, t, uc, c)         -> (noise_uc, noise_c)
    #   self.alpha(t)                            -> bar_alpha_t
    #   self.decode(z)                           -> image
    #   self.scheduler.timesteps, self.skip

    def __init__(self,
                 solver_config: Dict,
                 model_key: str = "runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device] = None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["", ""],
               callback_fn=None,
               **kwargs):
        # TODO: implement the guidance rule here.
        #   x0_hat = (zt - sqrt(1-at) * <eps for denoise>) / sqrt(at)
        #   zt     = sqrt(at_prev) * x0_hat + sqrt(1-at_prev) * <eps for renoise>
        raise NotImplementedError("You need to implement the sample method")
```

The SDXL block is the identical contract in `reverse_process()`, indexing
`self.scheduler.alphas_cumprod[t]` directly and threading the dual text embeddings; the same
denoise/renoise slot is the open region.

## Evaluation settings

Three frozen variants — **SD v1.5**, **SD v2-base**, **SDXL** — each scored independently. The
harness generates images from a shared COCO-caption prompt set at a single seed (42) under the
fixed DDIM step budget and the guidance-scale argument the harness passes in, decodes with the
frozen VAE, then computes two metrics per variant: **FID** against a precomputed COCO reference
Inception-stats set (lower is better; this is the task score) and **CLIP score** (ViT-B/32
image-text cosine, higher is better). The denoiser-evaluation budget, prompt set, weights, and
metric code are identical across every rung; only the editable guidance block changes.
