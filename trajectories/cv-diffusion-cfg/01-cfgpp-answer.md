**Problem.** Text-to-image diffusion needs guidance to follow a prompt, but standard CFG renoises
the DDIM step with the guided noise `eps_g = eps_uc + w(eps_c - eps_uc)`, and at the moderately-high
`w` it needs, that pushes the trajectory off the data manifold twice over: the denoised estimate
`(1-w)x_hat_uc + w x_hat_c` extrapolates past the conditional endpoint, and the renoise injects the
guided, off-manifold noise. The result is over-saturation, mode collapse, accumulating error, and
broken DDIM inversion.

**Key idea (the baseline fill).** CFG++ keeps the same two predictions per step but reframes
guidance as an optimization on the manifold: run the well-behaved *unconditional* sampler and let
the text enter only through a data-consistency nudge on the denoised estimate. The
decomposed/Jacobian-free gradient step turns the denoise into an **interpolation**
`x_hat_cfgpp = (1-lambda)x_hat_uc + lambda x_hat_c` (scale in `[0,1]`, stays on the segment, hence
on `M`) and forces the **renoise to use the unconditional noise `eps_uc`** instead of the guided
noise. Versus CFG, exactly one line of the DDIM step changes — the renoise noise, `eps_g -> eps_uc`
— at zero extra network cost.

**Why it works.** `lambda <= 1` makes the denoise a convex combination of two on-manifold endpoints
(no extrapolation); the `eps_uc` renoise is the same clean direction as the unconditional sampler
(no guided off-manifold offset); and the guidance-sensitive part of the DDIM inversion error shrinks
from the large `w` to the small interpolation scale, restoring invertibility.

**Step-1 edit / hyperparameters.** Fill the `BaseDDIMCFGpp.sample()` stub (SD path) and
`reverse_process()` (SDXL path). Form `noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)`,
Tweedie-denoise with `noise_pred`, **renoise with `noise_uc`**. The mix expression and the
`cfg_guidance` argument are kept exactly as the scaffold defines them — the small interpretable CFG++
scale is supplied at run time by the evaluation harness's `cfg_guidance` value (the signature default
`7.5` is unused when the harness passes its own); the only design choice this fill makes is which
predicted noise drives the renoise. No extra NFE, no retraining, no per-image optimization.

**What to watch.** CFG++ is on-manifold and artifact-free by construction, but it interpolates
gently and renoises unconditionally, so on FID-scored COCO it may sit slightly farther from the
reference statistics than a sampler that renoises with the harder-pushing guided noise. If it trails,
the next rung is to put the guided noise back into the renoise (standard CFG).

```python
# EDITABLE region of CFGpp-main/latent_diffusion.py (lines 621-679) — step 1: CFG++ (SD v1.5 / v2)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG++.
    """
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise - CFG++: use noise_uc to stay on manifold
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```

```python
# EDITABLE region of CFGpp-main/latent_sdxl.py (lines 713-755) — step 1: CFG++ (SDXL)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # CFG++: use noise_uc to stay on manifold
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        return z0t
```
