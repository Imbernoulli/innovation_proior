## Research question

Text-to-image diffusion sampling is only weakly prompt-aligned when using the conditional model alone, so practical systems rely on guidance. At each reverse step the frozen U-Net gives two predictions: the unconditional noise `eps_uc = eps_theta(z_t, empty)` and the conditional `eps_c = eps_theta(z_t, c)`. The design question is the guidance rule: given those two predictions per step, how do we form the denoised estimate and how do we renoise to the next latent? Everything else — the frozen SD v1.5 / SD v2 / SDXL weights, the DDIM step count, the prompt set, the VAE decode, the metrics — is fixed. The judged score is per-variant **FID** (lower is better), with **CLIP score** (higher is better) reported alongside; a good rule should improve image quality without sacrificing prompt-following.

## Prior art / Background / Baselines

- **Classifier guidance (Dhariwal & Nichol 2021).** It steers the unconditional score with the gradient of a separately trained, noise-aware classifier `p(c|z_t)`: `score <- score + s * grad_z log p(c|z_t)`, with `s` around 10 for strong conditioning. *Gap:* it requires an extra network trained on noisy latents, and the update direction is a classifier gradient rather than a native generative direction.
- **Classifier-free guidance, CFG (Ho & Salimans 2022).** A single network trained with random null conditioning produces both `eps_uc` and `eps_c`; the guided noise is the linear mix `eps_g = eps_uc + w (eps_c - eps_uc)`, and in a DDIM step this `eps_g` is used for both the Tweedie denoise and the renoise. *Gap:* at the moderately-high scales `w` in roughly `[5, 30]` that give strong prompt adherence, the sampler tends to oversaturate colors, collapse modes, accumulate trajectory error, and lose DDIM invertibility.

## Fixed substrate / Code framework

The substrate is a frozen latent-diffusion text-to-image stack: the U-Net noise predictor and VAE weights for each variant (SD v1.5, SD v2-base, SDXL), the tokenizer/text-encoder producing null and conditional embeddings, the DDIM noise scheduler exposing the cumulative signal rate `bar_alpha_t`, and the denoise-then-renoise reverse loop run for a fixed step budget. Exactly two predictions are available per step from one batched network call — `eps_uc` and `eps_c` — and no extra neural function evaluations are allowed. The harness also fixes the COCO-caption prompt set, the seed, the guidance-scale argument it passes in, and all evaluation code (FID against a precomputed COCO reference Inception-stats set, CLIP score with ViT-B/32).

## Editable interface

Exactly one region per variant is editable: the `BaseDDIMCFGpp` solver class registered as `ddim_cfg++`. For SD v1.5 / SD v2 it is the `sample()` method of `BaseDDIMCFGpp(StableDiffusion)` in `latent_diffusion.py` (the marked block, lines 621-679); for SDXL it is the `reverse_process()` method of `BaseDDIMCFGpp(SDXL)` in `latent_sdxl.py` (lines 713-755). Every method fills this same contract, with these helpers provided by the parent class:

- `self.get_text_embed(null_prompt, prompt)` -> `(uc, c)` (SD path);
- `self.initialize_latent()` / `self.initialize_latent(size=...)` -> `z_T ~ N(0, I)`;
- `self.predict_noise(zt, t, uc, c)` -> `(eps_uc, eps_c)` — the two predictions, one batched call;
- `self.alpha(t)` -> `bar_alpha_t` (SD path); `self.scheduler.alphas_cumprod[t]` (SDXL path);
- `self.scheduler.timesteps`, `self.skip`, `self.decode(z)`, `self.vae_scale_factor`.

The contract: combine `eps_uc` and `eps_c` into a denoised estimate `z0t` (Tweedie) and a renoised next latent `zt`, iterate over `self.scheduler.timesteps`, and return the decoded clean estimate. The rule may change how the two predictions are combined, which prediction drives the renoise, or how guidance varies with time — but not the prompt set, the weights, the NFE budget, or the evaluation code. The starting point is an unfilled stub that raises `NotImplementedError`.

```python
# EDITABLE region of CFGpp-main/latent_diffusion.py (lines 621-679) — scaffold DEFAULT (stub)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
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

The SDXL block is the identical contract in `reverse_process()`, indexing `self.scheduler.alphas_cumprod[t]` directly and threading the dual text embeddings; the same denoise/renoise slot is the open region.

## Evaluation settings

Three frozen variants — **SD v1.5**, **SD v2-base**, **SDXL** — each scored independently. The harness generates images from a shared COCO-caption prompt set at a single seed (42) under the fixed DDIM step budget and the guidance-scale argument the harness passes in, decodes with the frozen VAE, then computes two metrics per variant: **FID** against a precomputed COCO reference Inception-stats set (lower is better; this is the task score) and **CLIP score** (ViT-B/32 image-text cosine, higher is better). The denoiser-evaluation budget, prompt set, weights, and metric code are identical across every run; only the editable guidance block changes.
