**Problem.** Frozen text-to-image diffusion model, fixed budget NFE = 20, fixed prompts; design only
the sampler update rule so the images are as good as possible (low FID, high CLIP) inside those twenty
denoiser calls. The floor: the simplest correct update.

**Key idea.** DDIM is the first-order deterministic ODE step. The network was trained by the
unweighted $\epsilon$-MSE, which only fixes the marginals $q(x_t|x_0)$, so a short 20-step sub-grid of
the training chain is solved by the same frozen network. At each step form the guided prediction
$\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$, read the clean latent by Tweedie
$z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$, and step deterministically to
the next level renoising with the *unconditional* $\epsilon_{uc}$ (CFG++):
$z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_{uc}$.

**Why.** First order means no derivative estimate, no history, no intermediate call — so one
`predict_noise` per step exactly fills the budget, and there is no high-order term for a large
guidance scale to corrupt (DDIM is the stable fallback). But it ignores the curvature of the denoiser
trajectory between adjacent levels, so at twenty steps it is the least step-efficient sensible choice —
the floor by construction.

**Hyperparameters.** `cfg_guidance = 7.5`; NFE = 20 (one call per timestep, stride `self.skip`);
$\eta = 0$ (deterministic); CFG++ renoising with `noise_uc`; VAE latent space (no thresholding).

```python
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM sampler with CFG++.
    First-order ODE solver - simple and deterministic.
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
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Tweedie: estimate clean image
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # DDIM update: first-order, use unconditional noise for CFG++
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # Decode final latent
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```
