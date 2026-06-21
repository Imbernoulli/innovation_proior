Text-to-image diffusion follows a prompt only weakly if I sample the conditional model alone, so every system relies on guidance. The scaffold here hands me one frozen U-Net I can call two ways at each reverse step — the unconditional prediction $\varepsilon_{uc} = \varepsilon_\theta(z_t, \varnothing)$ and the conditional $\varepsilon_c = \varepsilon_\theta(z_t, c)$ — and the whole task is deciding how to combine them into the DDIM "denoise then renoise" update. The default move, standard classifier-free guidance (Ho & Salimans 2022), forms the guided noise $\varepsilon_g = \varepsilon_{uc} + w(\varepsilon_c - \varepsilon_{uc})$ and substitutes it into *both* halves of the step. That is exactly where the damage lives. Writing $a_t$ for the cumulative signal rate $\bar\alpha_t$, the Tweedie denoise gives $\hat x_g = (z_t - \sqrt{1-a_t}\,\varepsilon_g)/\sqrt{a_t}$, and by linearity this collapses to $\hat x_g = (1-w)\hat x_{uc} + w\hat x_c$. The clean data manifold $M$ is locally piecewise-linear, so the segment between the two on-manifold endpoints $\hat x_{uc}$ and $\hat x_c$ stays on $M$ — but the useful CFG regime is $w\in[5,30]$, so $1-w$ is hugely negative and this is an *extrapolation* shot far past $\hat x_c$, off $M$. That is the early-step color saturation. Then the renoise reuses the same guided $\varepsilon_g$ to lift back to the next noisy level, injecting a second off-manifold direction. Two distinct off-manifold sources, and they are what produce the over-saturation, mode collapse, accumulating trajectory error, and broken DDIM inversion usually treated as inherent to diffusion.

I propose CFG++, a manifold-constrained guidance rule that removes both sources with a single change to the DDIM step. The reframe that makes it work is to stop thinking of guidance as "amplify the conditional score" — which is precisely what produced the $w>1$ extrapolation — and instead run the well-behaved *unconditional* sampler while letting the text enter only through a data-consistency nudge on the denoised estimate. Formally this is an optimization on the manifold, $\min_{x\in M}\ell(x)$, where $\ell$ measures how poorly a clean image matches the prompt. The natural choice is the text-conditioned score-matching loss the network was already trained on: noise $x$ to level $t$, ask the conditional network to predict the noise, penalize the residual. Reducing it with the forward relation and the conditional Tweedie estimate gives a clean quadratic, $\ell(x) = \frac{a_t}{1-a_t}\|x - \hat x_c\|^2$. Its gradient at $x = \hat x_{uc}$ is $\frac{2a_t}{1-a_t}(\hat x_{uc} - \hat x_c)$. Following the decomposed-diffusion-sampling view, I take this gradient with respect to the denoised estimate itself rather than backpropagating the badly-conditioned U-Net Jacobian, so the correction costs nothing. One projected-gradient step on the denoised estimate is then

$$\hat x_{uc} - \gamma_t\,\tfrac{2a_t}{1-a_t}(\hat x_{uc} - \hat x_c) = (1-\lambda)\hat x_{uc} + \lambda\,\hat x_c,\qquad \lambda := \tfrac{2a_t\gamma_t}{1-a_t}.$$

This has the same affine *shape* as CFG, but the coefficient is now $\lambda$, a step size kept in $[0,1]$. With $\lambda\in[0,1]$ it is an honest convex combination of two on-manifold endpoints — a point on the segment, hence on $M$ — so the denoised estimate can never shoot off the manifold, and $\lambda$ now means something interpretable ($0$ = unconditional, $1$ = conditional). That fixes source one.

Source two is fixed by where the optimization-on-the-manifold view puts the renoise. Because the template solves the *unconditional* PF-ODE and the conditioning enters only through the data-consistency gradient on the denoised estimate, the renoise term is $\varepsilon_{uc}$ — the same clean direction the unconditional sampler uses — not anything guided. The text never touches the renoise. This is not a heuristic swap I picked; it is what the derivation forces. Re-expressing the denoise in noise-mixing notation shows how small the actual edit is: define $\varepsilon_{\text{cfgpp}} = \varepsilon_{uc} + \lambda(\varepsilon_c - \varepsilon_{uc})$, identical to CFG but with $\lambda$, and the interpolated denoised estimate is exactly the Tweedie estimate of this mixed noise. So the step is: denoise with the *mixed* noise, renoise with the *unconditional* noise. Side by side, CFG forms $\varepsilon_g$, denoises with $\varepsilon_g$, renoises with $\varepsilon_g$; CFG++ forms $\varepsilon_{\text{cfgpp}}$, denoises with it, renoises with $\varepsilon_{uc}$. The single line that changed is the renoise noise, $\varepsilon_g \to \varepsilon_{uc}$, and the scale moved from $w>1$ to $\lambda\in[0,1]$. It costs no extra network evaluation — I already compute both predictions every step — yet removes both off-manifold sources at once and shrinks the guidance-sensitive part of the DDIM inversion error from the large $w$ to the small interpolation scale, restoring invertibility.

One implementation honesty about the scaffold. The clean CFG++ story runs at a small interpretable scale (around $0.6$), but the editable signature here is `sample(self, cfg_guidance=7.5, ...)` and reuses the same `noise_uc + cfg_guidance*(noise_c - noise_uc)` mix expression — the *value* of the scale at run time is whatever the evaluation harness passes in, and it passes the small CFG++ scale, not the $7.5$ default in the signature. The contract does not let me rename or rescale that argument; the only design choice this fill makes is which predicted noise drives the renoise. So my fill keeps the exact mix expression and changes exactly the renoise to `noise_uc`. No per-image null-text optimization, no coupled invertible transforms, no time-varying schedule — just the renoise swap. The SDXL variant is the identical contract in `reverse_process()`, indexing `alphas_cumprod[t]` directly and threading the dual prompt embeddings. By construction CFG++ is conservative and on-manifold; the one thing to watch is that on FID-scored COCO, the gentle interpolation plus unconditional renoise may sit slightly *farther* from the sharp reference statistics than a sampler that renoises with the harder-pushing guided noise — and if it trails, the obvious next move is to put the guided noise back into the renoise.

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
