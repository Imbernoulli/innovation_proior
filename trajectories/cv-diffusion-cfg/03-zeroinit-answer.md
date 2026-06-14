**Problem.** Standard CFG (rung 2) beat CFG++ on FID but only by a tenth to six-tenths of a point,
and it bought that with off-manifold distortion — the guided renoise both sharpens (helps FID) and
distorts (hurts FID), nearly cancelling. Both prior rungs apply guidance at full strength on the
first, highest-noise steps, where the predictions `eps_uc`, `eps_c` are least reliable, so a guided
move there can be a *worse* estimate of the right direction than no move at all.

**Key idea (zero-init).** Make the high-noise prefix inert. In a controlled Gaussian flow where the
optimal step is known, the guided first-step move can be farther from the optimum than the zero move
(`||v_guided(t=0) - v_0^*||^2 >= ||0 - v_0^*||^2`) when the model is unreliable. So skip the first
`K = 2` steps entirely — no denoise, no renoise — leaving the latent at its initialization, and run
the **CFG++ manifold-safe** step (mix denoise, renoise with `noise_uc`) on every step afterward. The
prefix removes the regime where even the clean renoise is under-committed *and* unreliable; the CFG++
renoise keeps the manifold guarantee for the informative steps.

**Why it works.** It composes the two prior diagnoses: keep CFG++'s clean `noise_uc` renoise (rung 1)
and remove its under-committed worst steps, so once the prefix ends every remaining step is
on-manifold — clearing the bar standard CFG set by sharpening (rung 2) without paying its
off-manifold cost. `K` stays small because the inert-prefix argument is about the unreliable source
end only; skipping more would throw away budget-limited informative steps and lose conditional signal.

**Step-3 edit / hyperparameters.** Start from the rung-1 CFG++ fill (mix with `cfg_guidance`,
Tweedie-denoise with the mix, renoise with `noise_uc`) and add one guard at the top of the loop:
`K = 2`, then `if step < K: continue`. `K = 2` is the scaffold's choice (skip the two highest-noise
steps); not swept (seed and budget pinned). No optimized-scale projection in this baseline — the mix
keeps the plain CFG++ form. No extra NFE, no retraining.

**What to watch.** FID should come in under *both* priors on every variant — below standard CFG's
23.65 / 24.29 / 25.74 and CFG++'s 23.99 / 24.89 / 25.88. If skipping two steps instead costs too much
conditional signal at this tight budget, FID regresses toward the no-skip CFG++ numbers. Either way,
the optimized-scale projection `s* = <eps_c, eps_uc>/||eps_uc||^2` is the explicit lever left on the
table — the move past this strongest baseline.

```python
# EDITABLE region of CFGpp-main/latent_diffusion.py (lines 621-679) — step 3: CFG++ + zero-init (SD v1.5 / v2)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG++ and Zero-init.
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

        # Zero-init parameter
        K = 2  # Skip first K steps

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            # Zero-init: skip first K steps
            if step < K:
                continue

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
# EDITABLE region of CFGpp-main/latent_sdxl.py (lines 713-755) — step 3: CFG++ + zero-init (SDXL)
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

        K = 2  # Skip first K steps

        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            if step < K:
                continue

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
