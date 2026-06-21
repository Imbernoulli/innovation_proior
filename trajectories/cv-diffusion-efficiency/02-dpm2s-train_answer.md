The first-order run told me exactly where the budget is being wasted, and it told me in FID. DDIM came in at 34.23 on SD v1.5, 28.41 on SD v2.0, and 51.52 on SDXL — usable, never-diverging images on every variant, but high FIDs, and the *shape* of the gap is the tell: SD v2.0 lowest, SDXL worst by nearly double. That is the signature of a first-order solver, whose error scales with the curvature of the denoiser trajectory between adjacent noise levels; where there is more fine structure to resolve, twenty straight-line steps leave the most on the table. So the failure is not instability and not guidance — it is *step-efficiency*. Each DDIM step holds the clean-image estimate $z_{0|t}$ constant across the interval and moves on a straight chord while the true trajectory bends, and I am paying the full first-order truncation error twenty times over. The fix is not to ask for calls I do not have, but to extract the *bend* of the trajectory from the calls I already spend — to go to second order.

I propose **DPM-Solver++(2S)**: a second-order singlestep (Heun) update. The diffusion ODE, written on the clean-image (data) prediction, is semi-linear — the exact step from $s$ to $t$ carries a linear factor exactly and only ever approximates an integral of $z_\theta$ in the half-log-SNR variable $\lambda$. First order freezes $z_\theta$ at the left endpoint (that is DDIM, the $k=1$ member). To reach $k=2$ I need an estimate of the first derivative of $z_\theta$ along the trajectory, and a derivative needs a second sample. There are two ways to get it: the *multistep* way reuses the previous step's $z_\theta$ for free but inherits an error constant set by past spacing; the *singlestep* way takes a fresh evaluation at an intermediate point strictly inside the current interval and uses the finite difference of the two clean-image predictions as the derivative — it costs an extra call but stands alone, so each step's error constant is governed entirely by the within-step geometry. I build the singlestep version here because at this rung I want the cleanest, most history-independent second-order step, and the substrate already hands me at each timestep the two ingredients a singlestep Heun step needs: a clean-image estimate at the current level and the machinery to take a provisional step and re-evaluate.

The budget is the constraint DDIM did not feel. A singlestep second-order step makes *two* model calls per step — one at the current timestep, one at the provisional endpoint — so at twenty steps I would spend forty NFE. The only way it fits in twenty is to **halve the number of steps**: walk every other timestep on the harness grid (`timesteps[::2]`, stride `2*self.skip`), ten boundaries, two calls each, twenty NFE. The trade is explicit and not free — twenty first-order chords become ten second-order ones, each spanning a *double* interval, so the per-step $h$ doubles and a first-order solver on this coarse grid would be much worse than the twenty-step DDIM I measured. The bet is that $O(h^2)$ accuracy over ten coarse steps beats $O(h)$ accuracy over twenty fine ones — that the curvature correction more than pays back the coarser grid. That is exactly what the FID will answer, and it is a genuine question, not a foregone one.

The update itself I derive in the substrate's own vocabulary — $(\bar\alpha, z_0, \epsilon_{uc})$, not $(\lambda, h, \phi)$ — because `self.alpha(t)` hands me $\bar\alpha$ directly and Tweedie hands me $z_0$ directly. At the current timestep $t$ I form the guided prediction $\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$ and the clean-image estimate by Tweedie, $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$ — the slope at the *start* of the interval. The DDIM (Euler) move would commit to it; instead I treat it as a *predictor* and take a provisional step,

$$z^{\text{euler}}=\sqrt{\bar\alpha_{t-2s}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-2s}}\,\epsilon_{uc},$$

renoised in the CFG++ style with the unconditional noise. A second model call at $z^{\text{euler}}$, at timestep $t-2s$, gives $\tilde\epsilon_2$ and a clean-image estimate at the *endpoint*, $z_{0|t}^{(2)}=(z^{\text{euler}}-\sqrt{1-\bar\alpha_{t-2s}}\,\tilde\epsilon_2)/\sqrt{\bar\alpha_{t-2s}}$. Now I have a clean-image estimate at both ends of the interval, and the corrected step averages them — Heun's method, the trapezoidal rule for the slope: $\bar z_0=\tfrac12(z_{0|t}+z_{0|t}^{(2)})$, and the actual step uses this averaged estimate, $z_{t-2s}=\sqrt{\bar\alpha_{t-2s}}\,\bar z_0+\sqrt{1-\bar\alpha_{t-2s}}\,\epsilon_{uc}$, again renoised with the unconditional noise to keep the CFG++ convention.

This is deliberately simpler than the fully general singlestep solver, and the simplification is honest. The general construction places the intermediate point at a tunable fraction $r_1$ of the interval in $\lambda$ and weights the two predictions by $1/(2r_1)$ with exponential-integrator $\mathrm{expm1}$ coefficients $e^{-h}-1$. Here the intermediate point *is* the endpoint ($r_1=1$, trapezoidal rather than midpoint), the two clean estimates are weighted equally $\tfrac12/\tfrac12$, and the coefficients are the plain $\sqrt{\bar\alpha}$ ones rather than $\lambda$-space $\mathrm{expm1}$ factors. The arithmetic mean of two endpoint $z_0$ estimates *is* the trapezoidal second-order correction in data space; trapezoidal and midpoint rules are both $O(h^2)$, so what I forgo by not tuning $r_1$ or importing the $\mathrm{expm1}$ constants is a slightly larger error *constant*, not the order — an acceptable trade for a clean ten-step fill. One subtlety the doubled grid forces is the last step: with ten boundaries the final one lands at the end of the chain with no further point to evaluate a slope at, and a second call there would overrun the budget or have nowhere to correct toward, so it degrades gracefully to first-order DDIM, $z=z^{\text{euler}}$, no averaging. This is the standard lower-order-final stabilizer and it costs nothing — at the very end the trajectory is nearly straight (low noise, the clean estimate barely moves), so a first-order step there is almost lossless. Nine interior steps at two calls plus one first-order final lands at nineteen calls, at or just under the NFE = 20 ceiling.

So the delta from DDIM is one structural change with a cost attached: twenty straight-line steps become ten steps that each evaluate the model twice — once to predict, once at the predicted endpoint — and average the two clean-image estimates so the step follows the bend instead of a chord. Renoising stays CFG++ (`noise_uc`), the clean estimate stays guided, everything else untouched. I expect the curvature correction to pull FID down on all three variants, with the largest absolute improvement on SDXL where DDIM was worst (51.52) precisely because first-order coarseness costs most where there is the most structure to resolve. The risk is the halved grid: if the doubled $h$ is large enough that $O(h^2)$ over ten coarse steps does not beat $O(h)$ over twenty fine ones, FID could stall or regress where it bites most — and if SDXL in particular does not move, the next rung is already named: get the high-order correction *without* paying for it in steps, via a multistep solver that reuses past predictions at one call per step.

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
