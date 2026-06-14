**Problem.** DDIM at twenty first-order steps left FID high — 34.23 / 28.41 / 51.52 on SD v1.5 / v2.0 /
SDXL — with the worst gap on SDXL, the variant with the most structure to resolve. The error is
step-efficiency: a first-order step ignores the curvature of the denoiser trajectory between noise
levels. Capture that curvature without asking for more calls.

**Key idea.** A second-order singlestep (Heun) update. At each step take a provisional DDIM (Euler)
step to the next level, evaluate the model *again* at that predicted point to get the clean-image
estimate at the interval endpoint, and average the two clean estimates before taking the real step:
$\bar z_0=\tfrac12(z_{0|t}+z_{0|t}^{(2)})$, then
$z_{t-\text{next}}=\sqrt{\bar\alpha_{\text{next}}}\,\bar z_0+\sqrt{1-\bar\alpha_{\text{next}}}\,\epsilon_{uc}$.
The trapezoidal average of the endpoint slopes follows the bend the chord missed.

**Why.** Two model calls per step means the NFE budget only allows half as many steps, so the grid is
halved (`timesteps[::2]`, stride `2*self.skip`): ten second-order steps at twenty NFE instead of twenty
first-order ones. The bet is that $O(h^2)$ accuracy over ten coarse steps beats $O(h)$ over twenty fine
ones. This is the trapezoidal-Heun fill in the substrate's $(\bar\alpha, z_0, \epsilon_{uc})$
vocabulary — not the log-SNR singlestep with a tuned midpoint $r_1$ and $\mathrm{expm1}$ coefficients;
both are $O(h^2)$, this one has a slightly larger error constant but is the natural fill given
`self.alpha` and Tweedie. The final step degrades to first-order DDIM (no endpoint to correct toward,
and the trajectory is nearly straight at low noise).

**Hyperparameters.** `cfg_guidance = 7.5`; NFE = 20 → 10 steps, 2 calls per interior step + 1 for the
last; CFG++ renoising with `noise_uc`; equal $\tfrac12/\tfrac12$ Heun averaging in $z_0$ space;
lower-order final step.

```python
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DPM++ 2S sampler with CFG++.
    Second-order singlestep (Heun's method) - higher quality per step.
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

        # Halve timesteps: 2 model evals per step to stay within NFE budget
        timesteps = self.scheduler.timesteps[::2]
        double_skip = 2 * self.skip

        # Sampling
        pbar = tqdm(timesteps, desc="DPM++2S")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - double_skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Tweedie: estimate clean image
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # First prediction (Euler step to next timestep)
            zt_euler = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            # DPM++ 2S: Heun's method for second-order accuracy
            if step < len(timesteps) - 1:
                # Evaluate at the predicted point (endpoint of Euler step)
                with torch.no_grad():
                    noise_uc_2, noise_c_2 = self.predict_noise(zt_euler, t - double_skip, uc, c)
                    noise_pred_2 = noise_uc_2 + cfg_guidance * (noise_c_2 - noise_uc_2)

                z0t_2 = (zt_euler - (1-at_prev).sqrt() * noise_pred_2) / at_prev.sqrt()

                # Average the two estimates (Heun's method)
                z0t_avg = 0.5 * (z0t + z0t_2)
                zt = at_prev.sqrt() * z0t_avg + (1-at_prev).sqrt() * noise_uc
            else:
                # Last step: just use first-order
                zt = zt_euler

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
