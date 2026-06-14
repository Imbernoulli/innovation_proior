**Problem.** The strongest baseline, zero-init (CFG++ renoise + skip first `K = 2` steps), reached
FID 22.76 / 23.31 / 25.49 but implemented only *half* of the method it is named after: it took the
inert prefix and kept the *plain* CFG++ mix `noise_uc + w(noise_c - noise_uc)`, leaving the
unconditional prediction at a fixed coefficient of 1. SDXL barely moved across all three rungs
(25.88 -> 25.74 -> 25.49), which says the remaining error there is in *how the two predictions are
mixed*, not in the renoise or the timing.

**Key idea (CFG-Zero\*, the full method).** Add the missing half: a per-sample, per-step **optimized
scale** on the unconditional prediction. Subtracting a full unit of `noise_uc` pollutes the guidance
direction when `noise_uc` is mis-scaled relative to `noise_c`. Instead rescale it to its least-squares
match `s* = <noise_c, noise_uc>/||noise_uc||^2` (the projection of `noise_c` onto `noise_uc`), so the
mix amplifies only the conditional *residual* `noise_c - s* noise_uc` that the unconditional
prediction cannot explain. Keep CFG++'s `noise_uc` renoise and the `K = 2` inert prefix unchanged;
replace only the denoise mix.

**Why it works.** Three pieces, each fixing a distinct failure diagnosed on the way up: `noise_uc`
renoise removes off-manifold drift (rung 1), the inert prefix drops the unreliable high-noise steps
(rung 3), and the optimized scale corrects the per-step mis-scaling of the unconditional baseline that
every prior rung fixed at 1. When the two predictions are collinear the residual `noise_c - s* noise_uc`
vanishes and the mix collapses to plain CFG (with `s* = 1` in the equal-predictions case), so the finale
never underperforms the baseline mix and helps wherever they are not — most likely on the model that
resisted every other lever. The minimizer is the only `s`-dependent term in a Young's-inequality
upper bound on the (invisible) `||guided - truth||^2`, so it is principled, not tuned.

**Reference.** CFG-Zero\* (Fan, Zheng, Yeh, Liu, 2025, arXiv:2503.18886). The optimized-scale +
zero-init guidance rule; the eps-space transplant here matches the official per-sample
`optimized_scale` (dot-product over squared-norm, flattened per batch element) and the
`noise_uc * alpha + w*(noise_c - noise_uc * alpha)` mix.

**Edit / hyperparameters.** Start from the rung-3 zero-init fill. Add, inside the existing `no_grad`
block, a per-sample projection: flatten `noise_c`, `noise_uc` to `(B, -1)`, `alpha = sum(c*uc) /
(sum(uc^2) + 1e-8)`, reshape `alpha` to broadcast over the latent dims, cast to the prediction dtype.
Replace the mix with `noise_pred = noise_uc * alpha + cfg_guidance * (noise_c - noise_uc * alpha)`.
Renoise stays `noise_uc`; the `if step < K: continue` guard with `K = 2` stays. One dot product and
one norm per step — no extra NFE, no retraining.

**Bar to clear / what to validate.** Must beat zero-init's 22.76 / 23.31 / 25.49 on FID; sharpest
expected gain on SDXL (most untapped room), smaller gains on the already-strong SD v1.5 / v2-base.
Validate with the seed and prompt set pinned as the baselines ran: all three FID at or below zero-init,
largest drop on SDXL, and CLIP score not falling (the optimized scale amplifies the conditional
residual, so a CLIP drop would mean it traded alignment for distribution-matching).

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
