**Problem.** DPM++(2S) beat DDIM on every variant (29.01 / 23.89 / 42.83), but it has two costs: it is
singlestep, so the throwaway intermediate call forces a halved ten-step grid; and it is a deterministic
march, so discretization error compounds along the trajectory. Both leave FID on the table — especially
on SDXL, still the worst variant.

**Key idea.** A multistep, stochastic, third-order exponential-integrator solver. Reuse the last two
clean-image predictions to estimate the first *and* second $\lambda$-derivatives of $x_\theta$ via a
Newton divided difference — one model call per step, all twenty steps kept fine. Solve the reverse SDE
(not the ODE) with a single knob $\eta$: signal contracts by $e^{-h_\eta}$, $h_\eta=h(\eta+1)$, the
clean prediction takes $1-e^{-h_\eta}$, and Langevin noise of std $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$
is re-injected so the next denoising step washes out accumulated drift. Curvature correction
$+\phi_2 d_1+\phi_3 d_2$, with $\phi_2=(e^{-h_\eta}-1)/h_\eta+1$, $\phi_3=\phi_2/h_\eta-0.5$.

**Why.** Multistep over singlestep: at fixed NFE, twenty fine steps (reused derivative) beat ten coarse
ones (fresh derivative) because smaller $h$ shrinks truncation error. SDE/$\eta=1.2$ over deterministic
ODE: re-noising cancels the drift a pure march accumulates, which often beats the deterministic solver
perceptually. Karras $\rho=7$ schedule concentrates steps at low $\sigma$ where error lives; `expm1`
keeps small-argument exponentials stable; the SD path uses `kdiffusion_x_to_denoised` (CFG++ guided
Tweedie). Note this fill uses the off-canonical $r_1=h_2/h_1$ scaling and the $+\phi_3 d_2$ sign (the
canonical k-diffusion scheme uses $r_1=h_2/h$ and $-\phi_3 d_2$), and re-injects plain
`torch.randn_like` (not a Brownian-tree sampler) — correctly scaled but not Brownian-correlated. So this
is a practical third-order multistep step, not an exact $O(h^4)$ scheme.

**Hyperparameters.** `eta = 1.2`; `rho = 7` Karras schedule; NFE = 20 (one call per step); CFG++ guided
prediction; final step at $\sigma=0$ returns `denoised`; orders rise with available history (1 → 2 → 3).

```python
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """DPM-Solver++(3M) SDE with Karras schedule."""

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
        t_fn = lambda sigma: sigma.log().neg()

        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)

        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)

        eta = 1.2
        denoised_1, denoised_2 = None, None
        h_1, h_2 = None, None

        pbar = tqdm(self.scheduler.timesteps, desc="DPM++3M-SDE")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)

            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            if sigmas[i + 1] == 0:
                x = denoised
            else:
                t, s = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = s - t
                h_eta = h * (eta + 1)

                x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

                if denoised_1 is not None:
                    phi_2 = h_eta.neg().expm1() / h_eta + 1

                    if denoised_2 is None:
                        r = h_1 / h
                        d = (denoised - denoised_1) / r
                        x = x + phi_2 * d
                    else:
                        r0 = h_1 / h
                        r1 = h_2 / h_1
                        d1_0 = (denoised - denoised_1) / r0
                        d1_1 = (denoised_1 - denoised_2) / r1
                        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                        d2 = (d1_0 - d1_1) / (r0 + r1)
                        phi_3 = phi_2 / h_eta - 0.5
                        x = x + phi_2 * d1 + phi_3 * d2

                if eta > 0:
                    noise = torch.randn_like(x)
                    x = x + noise * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()

                denoised_2 = denoised_1
                denoised_1 = denoised
                h_2 = h_1
                h_1 = h

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                x = callback_kwargs["zt"]

        z0t = x
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```
