Text-to-image diffusion models are trained as noise predictors that accept either a text condition or a null prompt. Solving the conditional probability-flow ODE with the conditional prediction alone yields weak prompt alignment, so the field relies on classifier-free guidance to amplify the conditional direction. The standard CFG rule mixes unconditional and conditional noise predictions with a scale w greater than one, then substitutes that guided noise into both the denoise and renoise halves of a DDIM step. That works in the sense that high alignment appears once w is pushed into the moderately high regime of roughly 5 to 30, but it also produces mode collapse, reduced diversity, over-saturated colors, accumulating sampling error, and the collapse of DDIM inversion as soon as w exceeds one. These pathologies are often treated as inherent to diffusion models, yet the same backbone sampled unconditionally behaves cleanly and inverts well, which suggests the damage is in how guidance is injected rather than in the model itself.

The core issue is that CFG distorts the geometry of a single DDIM step in two ways. First, replacing eps_null with eps_cfg = eps_null + w (eps_c - eps_null) inside Tweedie's formula turns the denoised estimate into (1 - w) x_hat_null + w x_hat_c. For w larger than one this is an extrapolation past the conditional endpoint, leaving the locally piecewise-linear clean manifold. Second, the renoise half of the step also uses eps_cfg, so the noise added to lift the point back to the next noisy manifold is itself guided and off-manifold. Existing patches either add computation, such as null-text optimization or coupled invertible transforms, or merely reschedule the same bad trajectory, so they do not remove both off-manifold sources at zero extra network cost.

The method I propose is CFG++. It reframes text guidance not as posterior sharpening but as a small optimization on the clean data manifold. Concretely, it runs the well-behaved unconditional sampler and lets the prompt enter only through a data-consistency nudge applied to the denoised estimate. The natural loss is the text-conditioned score-matching loss used in score distillation, l_sds(x) = || eps_theta(sqrt(a_t) x + sqrt(1 - a_t) eps, c) - eps ||^2. Minimizing this pushes a clean image toward outputs the conditional model finds typical for the prompt. The decomposed-diffusion-sampling view lets us take the gradient with respect to the denoised estimate itself rather than differentiating through the score network, which avoids the expensive and ill-conditioned U-Net Jacobian. Reducing the loss using the forward noising relation and the conditional Tweedie estimate gives l_sds(x) proportional to || x - x_hat_c ||^2, whose gradient at x_hat_null points toward x_hat_c. One projected gradient step therefore yields the interpolation x_hat_cfgpp = (1 - lambda) x_hat_null + lambda x_hat_c with lambda in [0, 1]. Because lambda is at most one, the denoised estimate stays on the segment between the two on-manifold endpoints and therefore remains on the manifold. Equivalently, in noise-mixing form, eps_cfgpp = eps_null + lambda (eps_c - eps_null), and x_hat_cfgpp is exactly the Tweedie estimate obtained from this mixed noise.

The crucial remaining choice is which predicted noise drives the renoise. Because the optimization template solves the unconditional probability-flow ODE and adds conditioning only as a correction on the denoised estimate, the renoise automatically uses the unconditional noise eps_null. So one CFG++ DDIM step is: form eps_cfgpp = eps_null + lambda (eps_c - eps_null), Tweedie-denoise with eps_cfgpp, and renoise with eps_null. Compared with CFG, only the renoise noise changes, from the guided eps_cfg to the unconditional eps_null, and the scale moves from a large w to a small interpolation parameter lambda. This costs nothing extra, since eps_null and eps_c are already computed in one batched U-Net call. The same principle extends to higher-order solvers: replace only the leading denoising Tweedie term with x_hat_cfgpp and keep all renoising Tweedie terms unconditional.

The consequences line up with the design goals. The denoised estimate is a convex combination of on-manifold points, eliminating the extrapolation-induced saturation and sudden early shifts. The renoise uses the clean unconditional direction, removing the guided off-manifold offset. DDIM inversion needs the predicted noise to vary little between adjacent steps; CFG++'s guidance-sensitive inversion error is scaled by lambda instead of w, so it is strictly smaller whenever 0 <= lambda <= 1 < w, which restores invertibility without extra optimization. The posterior-mean evolution also becomes smoother: CFG advances by a large oscillatory difference w (Delta_t - Delta_{t+1}) that mostly cancels across steps, whereas CFG++ advances by a single small nudge lambda Delta_t at each step.

```python
import torch
from tqdm import tqdm


class BaseDDIMCFGpp(StableDiffusion):
    """DDIM sampler with CFG++ for SD v1.5.

    Uses the same two network evaluations per step as standard CFG:
    - denoise with the interpolated noise eps_cfgpp
    - renoise with the unconditional noise eps_null
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance=0.6, prompt=["", ""], callback_fn=None, **kwargs):
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        zt = self.initialize_latent()          # x_T ~ N(0, I)
        zt = zt.requires_grad_()

        for step, t in enumerate(tqdm(self.scheduler.timesteps, desc="SD")):
            at = self.alpha(t)                 # bar_alpha_t
            at_prev = self.alpha(t - self.skip) # bar_alpha_{t-1}

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                # interpolated guidance noise: eps_null + lambda (eps_c - eps_null)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # denoise using the interpolated noise
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # renoise using the unconditional noise (the CFG++ change)
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {
                    'z0t': z0t.detach(),
                    'zt': zt.detach(),
                    'decode': self.decode,
                }
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```
