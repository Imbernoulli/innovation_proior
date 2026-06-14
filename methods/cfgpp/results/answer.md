# CFG++, distilled

CFG++ is a manifold-constrained replacement for classifier-free guidance (CFG) in diffusion
sampling. It keeps the exact same two network evaluations per step (`eps_null`, `eps_c`) and
changes one line of the reverse step: the **denoise** half mixes the two predictions as an
*interpolation* with a small scale `lambda` in `[0,1]`, while the **renoise** half uses the
*unconditional* noise `eps_null` instead of the guided noise. This keeps the sampling
trajectory on the data manifold and makes the guidance scale an interpretable interpolation
parameter, with zero extra network evaluations.

## Problem it solves

Text-to-image diffusion needs CFG to follow prompts, but the useful regime forces a high
guidance scale `w` in `[5, 30]`, which causes mode collapse, reduced diversity,
over-saturation, accumulating sampling error, and broken DDIM inversion (so editing fails).
These were widely assumed to be inherent diffusion-model limitations. They are instead an
**off-manifold artifact of how CFG injects guidance** into the sampler.

## Key idea

A DDIM step is "denoise (Tweedie) then renoise." Two distinct off-manifold sources in CFG:

1. With the guided noise `eps_cfg = eps_null + w(eps_c - eps_null)`, the Tweedie denoised
   estimate is `x_hat_cfg = (1-w) x_hat_null + w x_hat_c`. For `w > 1` this is an
   **extrapolation** past the `[x_hat_null, x_hat_c]` segment, which leaves the (locally
   piecewise-linear) clean manifold `M`.
2. CFG renoises with `eps_cfg`, adding a guided, off-manifold noise direction even on the
   transition between noisy manifolds.

CFG++ reframes text guidance as an optimization on the manifold, `min_{x in M} l(x)`, with the
text-conditioned score-matching (SDS) loss
`l_sds(x) = || eps_theta(sqrt(a_t)x + sqrt(1-a_t)eps, c) - eps ||^2`, solved by a
decomposed-diffusion / Jacobian-free gradient step on the *denoised estimate*. Because that
template solves the **unconditional** PF-ODE and lets the text enter only through the
data-consistency gradient, the renoise automatically uses `eps_null` (fixes source 2), and the
gradient step turns the denoised estimate into an **interpolation**
`x_hat_cfgpp = (1-lambda) x_hat_null + lambda x_hat_c` with `lambda` in `[0,1]` (fixes
source 1 — convex combination stays on the segment, hence on `M`).

## Derivation (the load-bearing steps)

With `x_t = sqrt(a_t) x + sqrt(1-a_t) eps` and the conditional Tweedie estimate
`x_hat_c = (x_t - sqrt(1-a_t) eps_c)/sqrt(a_t)`:

```
eps_c - eps = sqrt(a_t)/sqrt(1-a_t) * (x - x_hat_c)
=> l_sds(x) = || eps_c - eps ||^2 = (a_t/(1-a_t)) || x - x_hat_c ||^2
=> grad_{x_hat} l_sds (x_hat_null) = (2 a_t/(1-a_t)) (x_hat_null - x_hat_c)
```

One DDS step on the denoised estimate (gradient w.r.t. `x_hat`, no U-Net Jacobian):

```
x_hat_null - gamma_t * grad l_sds = x_hat_null + lambda (x_hat_c - x_hat_null),
        lambda := 2 a_t gamma_t / (1 - a_t)  in [0, 1].
```

In noise-mixing form, `eps_cfgpp = eps_null + lambda(eps_c - eps_null)`, and
`x_hat_null + lambda(x_hat_c - x_hat_null) = (x_t - sqrt(1-a_t) eps_cfgpp)/sqrt(a_t)`.

## Final algorithm (one step), CFG vs CFG++

```
CFG:    eps_g  = eps_null + w * (eps_c - eps_null)            # w > 1
        x_hat  = (x_t - sqrt(1-a_t) * eps_g) / sqrt(a_t)
        x_{t-1}= sqrt(a_{t-1}) * x_hat + sqrt(1-a_{t-1}) * eps_g

CFG++:  eps_g  = eps_null + lambda * (eps_c - eps_null)       # lambda in [0,1]
        x_hat  = (x_t - sqrt(1-a_t) * eps_g) / sqrt(a_t)
        x_{t-1}= sqrt(a_{t-1}) * x_hat + sqrt(1-a_{t-1}) * eps_null
```

Only the renoise noise changes (`eps_g -> eps_null`) and the scale moves from `w > 1` to
`lambda` in `[0,1]`.

## Why it works

- **On-manifold denoise.** `lambda <= 1` makes `x_hat_cfgpp` a convex combination of two
  on-manifold endpoints, so it stays on the piecewise-linear `M` instead of extrapolating off.
- **On-manifold renoise.** The renoise uses `eps_null`, the same clean direction as the
  unconditional sampler, removing the guided off-manifold offset.
- **Invertibility.** DDIM inversion needs `eps(x_t) ~ eps(x_{t-1})`. With
  `delta := eps_c - eps_null`, CFG's guidance-sensitive error is
  `e_cfg ~= w(delta(x_t)-delta(x_{t-1}))`. Under the usual unconditional DDIM approximation,
  CFG++'s guidance-sensitive error satisfies
  `||e_cfgpp|| = lambda ||delta(x_t)-delta(x_{t-1})|| < ||e_cfg||` for
  `0 <= lambda <= 1 < w`.
- **Smooth trajectory.** Tracking the denoised estimate (`d z := z(x_t) - z(x_{t+1})`,
  `Delta := x_hat_c - x_hat_null`):

  ```
  d x_hat_cfg   = sqrt(1-a_t)/sqrt(a_t) * d eps_null + w (Delta(x_t) - Delta(x_{t+1}))   # oscillatory
  d x_hat_cfgpp = sqrt(1-a_t)/sqrt(a_t) * d eps_null + lambda Delta(x_t)                 # single smooth nudge
  ```

  CFG relies on a large `w Delta(x_t)` overshoot that is partly cancelled by `-w Delta(x_{t+1})`
  from the prior step; CFG++ adds one small non-oscillatory nudge toward the current condition.

## Extension to higher-order / distilled solvers

For any solver whose single-step unconditional update is
`x_i = x_hat_null(x_{i-1}) + a_i x_hat_null(x_{i-1}) + b_i x_hat_null(x_{i-2}) + c_i x_{i-1} + d_i eps`,
the CFG++ rule is: replace **only the leading denoising term** by `x_hat_cfgpp`, and keep all
renoising Tweedie terms unconditional:
`x_i = x_hat_cfgpp(x_{i-1}) + a_i x_hat_null(x_{i-1}) + b_i x_hat_null(x_{i-2}) + c_i x_{i-1} + d_i eps`.
This gives Euler, Euler-ancestral, DPM++ 2M, and DPM++ 2S CFG++ variants. For distilled
few-step models (SDXL-Turbo/Lightning, which bake a fixed guidance into the conditional score),
set `lambda = 1` and still take the rest of the renoising components unconditionally.

## Working code

Filling the guidance slot of the DDIM sampler (SD v1.5). `cfg_guidance` is `lambda`;
`noise_uc = eps_null`, `noise_c = eps_c`; `z0t` is the denoised estimate, `zt` the renoised
latent.

```python
import torch
from tqdm import tqdm


class BaseDDIMCFGpp(StableDiffusion):
    """DDIM sampler with CFG++ for SD v1.5: interpolating guidance on the denoise,
    unconditional noise on the renoise. Same two NFE/step as CFG."""

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance=0.6, prompt=["", ""], callback_fn=None, **kwargs):
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        zt = self.initialize_latent()                    # x_T ~ N(0, I)
        zt = zt.requires_grad_()

        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)                           # bar_alpha_t
            at_prev = self.alpha(t - self.skip)          # bar_alpha_{t-1}

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)        # eps_null, eps_c
                # mixed noise: eps_null + lambda (eps_c - eps_null)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # denoise: Tweedie with the mixed noise
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # renoise with the unconditional noise eps_null
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(), 'zt': zt.detach(), 'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # last step: return the clean denoised estimate, no renoise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```

SDXL variant — identical step using the scheduler's `alphas_cumprod[t]` and dual text
embeddings:

```python
import torch
from tqdm import tqdm


class BaseDDIMCFGpp(SDXL):
    """DDIM sampler with CFG++ for SDXL."""

    def reverse_process(self, null_prompt_embeds, prompt_embeds, cfg_guidance,
                        add_cond_kwargs, shape=(1024, 1024), callback_fn=None, **kwargs):
        zt = self.initialize_latent(
            size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(
                    zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()        # denoise (mixed)
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_uc  # renoise (uncond)

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(), 'zt': zt.detach(), 'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        return z0t
```
