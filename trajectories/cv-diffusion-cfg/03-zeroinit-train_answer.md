Standard CFG came in at FID $23.65/24.29/25.74$, beating CFG++'s $23.99/24.89/25.88$ on every variant, so the rung-2 hypothesis held: the harder push with the guided noise driving the renoise pulled the distribution onto the sharp reference modes and lowered FID. But the gains are *small* — a tenth to six-tenths of a point — and they came with everything CFG++ was built to avoid: over-saturation, off-manifold drift, broken inversion. The narrow margin is the clue. If the guided renoise were unambiguously better I would expect a larger gap; that it is only marginally ahead says the guided renoise is *both* helping (sharpening toward the reference) and hurting (off-manifold distortion), and the two effects nearly cancel. That reframes the question: rather than choosing between two rules that each mix help and harm, can I keep the helpful part of one and drop the harmful part? The answer is in the *timing*. The early steps of a DDIM trajectory are the highest-noise steps: $z_t$ is almost pure noise, the two predictions $\varepsilon_{uc}$ and $\varepsilon_c$ are at their least reliable, and yet both prior rungs let guidance push at full strength there. This is exactly where a guided move is most likely to be a *worse* estimate of the right direction than no move at all, because the conditional signal $\varepsilon_c - \varepsilon_{uc}$ is dominated by prediction error when the latent is near-pure noise. The untouched lever is not which noise renoises, but whether to apply the guided update *at all* during the first, least-trustworthy steps.

I propose CFG++ with zero-init: make the high-noise prefix inert by skipping the first $K$ steps entirely, run on top of CFG++'s manifold-safe renoise. The diagnostic that makes "skip" principled rather than a hack comes from a controlled Gaussian flow where the optimal step is known in closed form. Take $x_0\sim N(0,I)$, $x_1\sim N(\mu,I)$, and the linear path $x_t = (1-t)x_0 + t x_1$. The optimal velocity is the conditional mean $E[x_1 - x_0\mid x_t]$, and because $(x_t, x_1-x_0)$ is jointly Gaussian with $\mathrm{Cov}(x_t, x_1-x_0) = (2t-1)I$ and $\mathrm{Var}(x_t) = ((1-t)^2 + t^2)I$, it has the closed form

$$v_t^*(x) = \frac{2t-1}{(1-t)^2 + t^2}\,(x - t\mu) + \mu.$$

This lets me compare a learned, guided step against the true step at any $t$, and the source end ($t$ near the start) is the dangerous place: the sample is still close to noise while guidance is already pushing hard. The trouble is that the same scale which strengthens the conditional signal also strengthens the conditional *error* — if $\varepsilon_{uc} = \varepsilon_{uc}^* + e_{uc}$ and $\varepsilon_c = \varepsilon_c^* + e_c$, the guided prediction carries an error term that grows with the scale, and at high noise the model cannot tell semantic control from a wrong prediction. So I ask the sharp question directly: in the underfitted regime, is the guided first-step move closer to the optimal first-step move than the zero move is? It is not — the diagnostic can satisfy $\|v_{\text{guided}}(t{=}0) - v_0^*\|^2 \ge \|0 - v_0^*\|^2$. Read as a decision rule, taking the guided step injects the largest wrong direction exactly when the trajectory carries the least semantic information; doing nothing for that step is provably no worse, and often better. So the high-noise prefix should be inert: zero the update for the first $K$ steps and let guidance act only once the predictions become informative.

The fill is concrete. Start from the rung-1 CFG++ step — mix with `cfg_guidance`, Tweedie-denoise with the mix, renoise with `noise_uc` — and wrap the loop with a guard that skips the first $K=2$ iterations entirely: no denoise, no renoise, no callback, so the latent stays at its initialization through the two highest-noise steps, and the manifold-safe CFG++ update runs on every step after. Two design choices carry the weight. First, why build on the CFG++ renoise rather than the standard-CFG renoise that just beat it? Because the two diagnoses *compose*. Standard CFG's edge was small and bought with off-manifold distortion; CFG++'s renoise is the clean one. If the reason the early steps hurt is that guidance is unreliable at high noise, the right base for the inert prefix is the clean-renoise sampler, so that once the prefix ends every remaining step is on-manifold. Skipping the worst two steps removes the regime where even the clean renoise is under-committed *and* unreliable; running CFG++ on the rest keeps the manifold guarantee for the informative steps. The bet is that "manifold-safe renoise, but only after the noise has come down enough for the predictions to mean something" beats both plain CFG++ (clean renoise applied even at the useless start) and standard CFG (hard guided renoise applied everywhere). Second, $K$ must stay small. The inert-prefix argument is about the *source end* only, not a license to skip arbitrarily many steps; once the velocity field becomes informative a few steps in, continuing to do nothing would throw away budget-limited solver steps and lose the conditional signal I need. $K=2$ skips the two worst steps and keeps the rest; I do not sweep it, since the harness pins the seed and budget.

One deliberate note on scope. The method this rung is named for has *two* parts, and this baseline uses only one. The full version pairs the inert prefix with a per-sample *optimized scale* on the unconditional prediction — replace $\varepsilon_{uc}$ in the mix by $s^*\varepsilon_{uc}$ where $s^* = \langle\varepsilon_c, \varepsilon_{uc}\rangle/\|\varepsilon_{uc}\|^2$ is the least-squares projection, so guidance amplifies only the conditional residual the unconditional prediction cannot already explain. This fill omits that: it keeps the plain CFG++ mix `noise_pred = noise_uc + cfg_guidance*(noise_c - noise_uc)` and renoise with `noise_uc`, and adds *only* the inert prefix, `if step < K: continue` with `K = 2`. The optimized scale is the obvious lever left on the table, and it is where the ladder goes next. The SDXL fill is identical in `reverse_process()`: the same guard, then the CFG++ step indexing `alphas_cumprod[t]` and renoising with `noise_uc`. I expect FID under *both* priors on every variant — below standard CFG's $23.65/24.29/25.74$ and CFG++'s $23.99/24.89/25.88$ — with the failure mode being that two skipped steps cost too much conditional signal at this tight budget, in which case FID regresses toward the no-skip CFG++ numbers.

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
