Zero-init — CFG++'s manifold-safe renoise with the first $K=2$ highest-noise steps made inert — came in at FID $22.76/23.31/25.49$, under both priors on every variant. The rung-3 hypothesis held cleanly: keeping CFG++'s clean `noise_uc` renoise while deleting its under-committed, unreliable high-noise prefix beat the hard guided renoise *and* the no-skip clean renoise, without paying standard CFG's off-manifold cost. But the gains were lopsided — $1.23$ on SD v1.5, $0.98$ on SD v2-base, and only $0.25$ on SDXL, the variant that has stubbornly resisted every move ($25.88\to25.74\to25.49$). SDXL is the signal: a high-resolution model whose FID barely budges as I swap renoise rules and skip steps, which says the remaining error there is not in *when* I guide or *which noise renoises*, but in *how the two predictions are mixed* at the steps I do guide. And that is exactly the lever I flagged and set aside at rung 3. The zero-init baseline implemented only half of the method it is named after — it took the inert prefix but kept the *plain* CFG++ mix, leaving the coefficient on `noise_uc` fixed at $1$. Every rung so far has subtracted a full unit of the raw unconditional prediction inside $\varepsilon_c - \varepsilon_{uc}$, but there is no reason the best baseline to subtract is exactly one times that prediction: if at this particular $(z_t, t)$ the unconditional prediction is too large, too small, or partly mis-aligned with the conditional one, subtracting a full unit pollutes the guidance direction with a component the model did not intend.

I propose CFG-Zero\* (Fan, Zheng, Yeh, Liu, 2025), the full method the strongest baseline only half-implemented. It adds the missing half: a per-sample, per-step *optimized scale* on the unconditional prediction. I want guidance to amplify only the part of the conditional prediction the unconditional one does not already explain — the conditional *residual* — so introduce a scalar $s$ on the unconditional prediction and write the guided noise as $s\,\varepsilon_{uc} + w(\varepsilon_c - s\,\varepsilon_{uc})$, which collapses to $(1-w)s\,\varepsilon_{uc} + w\,\varepsilon_c$; at $s=1$ this is exactly the mix every prior rung used. How to pick $s$? If I could see the true guided noise I would minimize $\|v_s - v^*\|^2$ over $s$, but $v^*$ is invisible — that is the wall. The way through: write $\delta = w - 1$ so $v_s = v_c + \delta(v_c - s\,v_{uc})$, then the unavailable loss is $\|(v_c - v^*) + \delta(v_c - s\,v_{uc})\|^2$. For any positive $\lambda$, Young's inequality bounds this above by $(1+\lambda)\|v_c - v^*\|^2 + (1 + 1/\lambda)\delta^2\|v_c - s\,v_{uc}\|^2$. The first term still hides the truth but does *not* depend on $s$; the only $s$-dependent part of the bound is a positive constant times $\|v_c - s\,v_{uc}\|^2$ — in noise units, $\|\varepsilon_c - s\,\varepsilon_{uc}\|^2$, built entirely from the two predictions I already have. So minimizing the bound replaces the impossible "match the invisible truth" with a solvable projection. Setting the derivative $-2\,\varepsilon_{uc}^\top(\varepsilon_c - s\,\varepsilon_{uc})$ to zero gives

$$s^* = \frac{\varepsilon_c^\top \varepsilon_{uc}}{\|\varepsilon_{uc}\|^2},$$

and the second derivative $2\|\varepsilon_{uc}\|^2$ is positive, so this is the minimizer whenever $\varepsilon_{uc}\neq 0$. Geometrically $s^*\varepsilon_{uc}$ is the orthogonal projection of $\varepsilon_c$ onto the line spanned by $\varepsilon_{uc}$, and $\varepsilon_c - s^*\varepsilon_{uc}$ is the residual the unconditional prediction cannot explain — exactly the direction I want guidance to push, instead of the whole conditional vector measured against a raw, possibly mis-scaled baseline. The optimized-scale mix is therefore $s^*\varepsilon_{uc} + w(\varepsilon_c - s^*\varepsilon_{uc})$: rescale the unconditional baseline to its best least-squares match, then amplify only the leftover conditional residual.

This is genuinely a new degree of freedom, not a relabeled guidance scale. $w$ is a single global scalar fixed for the whole run, applied identically to every sample and step; $s^*$ is computed per sample and per step from the actual geometry of that step's two predictions. They act on different objects — $w$ sets *how hard* to push along the residual, $s^*$ sets *what the residual is* by fixing the baseline it is measured against. In the implicit-classifier language of rung 2, where the guided direction was the difference $\varepsilon_c - \varepsilon_{uc}$ with a unit coefficient on the unconditional term, the optimized scale says that unit coefficient was an unexamined default: the implicit classifier is sharper once the unconditional baseline is projected onto the conditional prediction, so the difference isolates only the conditional-specific component and not a mis-scaled chunk of the shared image prior. The limits confirm it is the right scalar, not a tuned one: if the two predictions are collinear the residual $\varepsilon_c - s^*\varepsilon_{uc}$ vanishes and the mix collapses to the strongest baseline's exactly (with $s^* = 1$ when they are equal), so the finale can only differ where there is a real off-axis component to correct; if $\varepsilon_{uc}$ were orthogonal to $\varepsilon_c$, then $s^* = 0$ and guidance pushes along the full conditional vector.

The finale composes with everything the strongest baseline already does — that is the whole point. Three pieces, each fixing a distinct failure I diagnosed climbing the ladder: the `noise_uc` renoise removes the off-manifold drift (rung 1), the inert prefix drops the unreliable high-noise steps (rung 3), and the optimized scale corrects the per-step mis-scaling of the unconditional baseline that every prior rung fixed at $1$. The first two are already in the strongest baseline; the third is the genuinely new ingredient it omitted. So I keep the prefix (`if step < K: continue`, `K = 2`) and CFG++'s manifold-safe renoise with `noise_uc` exactly as they were, and replace *only* the denoise mix, swapping `noise_pred = noise_uc + cfg_guidance*(noise_c - noise_uc)` for `noise_pred = noise_uc * alpha + cfg_guidance*(noise_c - noise_uc * alpha)`. A few implementation cautions keep the projection faithful: the dot product and squared norm are taken per sample over the flattened non-batch dimensions, then reshaped to broadcast over the channel and spatial axes — a single scalar per image, not a global scalar across the batch nor a per-element one; the denominator gets a small floor (`+1e-8`) so a near-zero unconditional prediction does not blow up the scale; the scale is cast back to the prediction's dtype (the SD path runs under fp16 autocast); and the whole thing is computed from the *detached* predictions inside the existing `no_grad` block, since it is a sampling-time correction with nothing trained. It costs one dot product and one squared norm per step — no extra network evaluation. The SDXL fill is identical in `reverse_process()`, indexing `alphas_cumprod[t]`. The bar to clear is zero-init's $22.76/23.31/25.49$, and the sharpest expected gain is on SDXL — the variant whose FID barely moved across rungs 1–3 is exactly where a per-step correction to the mix has the most untapped room — with a CLIP score that does not fall, since the optimized scale amplifies the conditional *residual* rather than trading alignment for distribution-matching.

```python
# EDITABLE region of CFGpp-main/latent_diffusion.py (lines 621-679) — finale: CFG-Zero* (SD v1.5 / v2)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG-Zero* (optimized scale + zero-init), CFG++ renoise.
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

                # CFG-Zero*: per-sample optimized scale on the unconditional prediction.
                # s* = <noise_c, noise_uc> / ||noise_uc||^2  (projection onto noise_uc)
                bsz = noise_c.shape[0]
                c_flat = noise_c.reshape(bsz, -1)
                uc_flat = noise_uc.reshape(bsz, -1)
                dot = (c_flat * uc_flat).sum(dim=1, keepdim=True)
                sq_norm = (uc_flat ** 2).sum(dim=1, keepdim=True) + 1e-8
                alpha = (dot / sq_norm).reshape(bsz, *([1] * (noise_c.dim() - 1)))
                alpha = alpha.to(noise_c.dtype)

                # mix on the rescaled unconditional baseline
                noise_pred = noise_uc * alpha + cfg_guidance * (noise_c - noise_uc * alpha)

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
# EDITABLE region of CFGpp-main/latent_sdxl.py (lines 713-755) — finale: CFG-Zero* (SDXL)
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

                # CFG-Zero*: per-sample optimized scale on the unconditional prediction.
                bsz = noise_c.shape[0]
                c_flat = noise_c.reshape(bsz, -1)
                uc_flat = noise_uc.reshape(bsz, -1)
                dot = (c_flat * uc_flat).sum(dim=1, keepdim=True)
                sq_norm = (uc_flat ** 2).sum(dim=1, keepdim=True) + 1e-8
                alpha = (dot / sq_norm).reshape(bsz, *([1] * (noise_c.dim() - 1)))
                alpha = alpha.to(noise_c.dtype)

                noise_pred = noise_uc * alpha + cfg_guidance * (noise_c - noise_uc * alpha)

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
