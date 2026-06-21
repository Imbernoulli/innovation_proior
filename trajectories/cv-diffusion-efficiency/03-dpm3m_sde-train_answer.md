The second-order run paid off exactly where I bet it would. DPM++(2S) pulled every variant down from DDIM — SD v1.5 from 34.23 to 29.01, SD v2.0 from 28.41 to 23.89, and SDXL, the one I called as having the most to gain, from 51.52 to 42.83, an 8.7-point drop and the largest of the three. The curvature correction was real and the halved grid did not cost me the win. But looking at *how* I bought that improvement, two weaknesses sit in the 2S construction that FID is quietly paying for, and both point the same way. First, 2S is *singlestep*: two model calls per step force the halved ten-step grid, and the intermediate evaluation I spend to estimate the slope is thrown away after one use — pure overhead, never a sample on the trajectory. If I could get the same first-derivative information without the throwaway call I could keep all twenty steps fine, and smaller $h$ directly shrinks the truncation error. Second, 2S is a pure deterministic march: every step is an ODE step, error accumulates along the trajectory and nothing washes it back out, so the state drifts off the data manifold and stays off — exactly the texture and global-structure error FID is sensitive to.

I propose **DPM-Solver++(3M) SDE**: a multistep, stochastic, third-order exponential-integrator solver. Both weaknesses dissolve in it. For the singlestep overhead, I reuse the clean-image prediction I already computed at the *previous* step as the second sample for the finite difference — one new call per step, twenty fine steps kept, the derivative for free; the error constant is now set by past spacing rather than within-step geometry, but on a tight budget the step-count win dominates. And reusing *two* past predictions lets me fit a quadratic through three points and pick up the *second* derivative too, reaching third order at no extra call. For the deterministic drift, the known antidote is a Langevin correction: re-inject a controlled amount of Gaussian noise each step and let the next denoising step remove it — the noise-and-denoise cycle pulls the state back toward where the model thinks data lives, cancelling accumulated error, and fully deterministic sampling is well documented to be perceptually worse at the same step count. So I solve the reverse *SDE* with the same exponential-integrator machinery and keep a knob for how much noise to put back.

I derive it in the noise-level variable the substrate now hands me, $\sigma=\sqrt{1-\bar\alpha}/\sqrt{\bar\alpha}$, with the k-diffusion helpers: `get_sigmas_karras` builds the schedule, `self.timestep(sigma)` converts back, and `self.kdiffusion_x_to_denoised` returns the clean-image estimate ("denoised") at a given $\sigma$. Working in the VE convention where $\alpha=1$ and the half-log-SNR is $\lambda=-\log\sigma$, the reverse SDE on the data prediction is $dx=[-(1+\alpha^2)x+2\alpha\,x_\theta]\,d\lambda+\sqrt2\,\sigma\,dw$; solving the linear part exactly by variation of constants and the data term by the exponential integrator gives the exact step, with $h=\lambda_{\text{next}}-\lambda$. The single knob $\eta$ is what makes this one method instead of two: set $h_\eta=h(\eta+1)$, and the signal contracts by $e^{-h_\eta}$, the clean prediction takes the complement $1-e^{-h_\eta}$ (a convex split), and the Itô integral contributes Gaussian noise of standard deviation $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$. At $\eta=0$ the contraction is $e^{-h}$ and the noise vanishes — the deterministic ODE step, the multistep deterministic limit of what 2S was doing; at $\eta=1$ it is the full reverse SDE; and $\eta>1$ injects extra Langevin stochasticity beyond it. Since my whole second motivation is to add re-noising against the diagnosed drift, I want $\eta$ a touch above one — the working default is $\eta=1.2$, buying a bit more self-correction than the bare SDE at the cost of a bit more noise to clean up, the right posture at a tight budget where drift is the bigger enemy.

For the multistep order I keep the last two clean predictions `denoised_1`, `denoised_2` and the last two $\lambda$-steps `h_1`, `h_2`. The data integral's Taylor expansion in $\lambda$ brings in the first and second derivatives of $x_\theta$, weighted by the exponential-integrator $\phi$ functions at $h_\eta$: $\phi_2(h_\eta)=(e^{-h_\eta}-1)/h_\eta+1\approx h_\eta/2$ (first-derivative weight) and $\phi_3(h_\eta)=\phi_2/h_\eta-0.5\approx-h_\eta/6$ (second-derivative weight). The derivatives come from a Newton divided difference through the three points, and here I am explicit that the fill this harness records makes two coefficient choices that are *not* the canonical k-diffusion ones — the trajectory's job is to land *this* implementation. The canonical third-order multistep scales the older interval as $r_1=h_2/h$ and applies the curvature correction with a minus sign, $x+\phi_2 d_1-\phi_3 d_2$; this fill instead uses $r_1=h_2/h_1$ (older interval relative to the *previous* step, not the current one) and a plus sign, $x+\phi_2 d_1+\phi_3 d_2$. So with scaled spacings $r_0=h_1/h$ and $r_1=h_2/h_1$, the divided differences are $d_{1,0}=(\text{denoised}-\text{denoised}_1)/r_0$ and $d_{1,1}=(\text{denoised}_1-\text{denoised}_2)/r_1$; the endpoint first derivative is $d_1=d_{1,0}+(d_{1,0}-d_{1,1})\,r_0/(r_0+r_1)$, the second is $d_2=(d_{1,0}-d_{1,1})/(r_0+r_1)$, and the step is

$$x=e^{-h_\eta}x+(1-e^{-h_\eta})\,\text{denoised}+\phi_2 d_1+\phi_3 d_2.$$

The $r_1=h_2/h_1$ scaling reweights the older sample, and the $+\phi_3 d_2$ sign (with $\phi_3<0$) *subtracts* a curvature contribution where the canonical scheme adds it, so this is best read as a *practical* third-order multistep correction with a slightly off-tuned constant rather than an exact $O(h^4)$ scheme — acceptable, because the dominant gain over 2S is the step count (twenty vs ten) and the Langevin re-noising, not the exact third-order constant. The order ramps with available history: when only one past value exists (the first correction step) I drop $d_2$ and use the two-point estimate $d=(\text{denoised}-\text{denoised}_1)/r$ with $r=h_1/h$, the second-order step $x+\phi_2 d$; when no past value exists yet (the first step) it is the plain constant-data step plus the noise.

A few implementation realities I keep because the anchor does. The schedule is the Karras power grid, `get_sigmas_karras(N, sigma_min, sigma_max, rho=7)`, which concentrates steps at low $\sigma$ where per-step truncation error is largest — the right way to spend a tight budget. The exponentials all appear as $e^x-1$ with small $x$, where naive subtraction cancels catastrophically, so they go through `expm1`: $(1-e^{-h_\eta})$ is `(-h_eta).expm1().neg()` and the noise std $\sqrt{1-e^{-2h\eta}}$ is `(-2*h*eta).expm1().neg().sqrt()`. The last step lands at $\sigma=0$ with nothing left to denoise toward and no noise to add, so it returns `denoised` itself. And the re-injected noise is a plain `torch.randn_like(x)` scaled by $\sigma_{\text{next}}\sqrt{1-e^{-2h\eta}}$ — correctly *scaled* but not Brownian-tree *correlated*; the canonical implementation draws Brownian-consistent increments from a tree keyed to the noise levels, while this fill uses fresh independent Gaussian noise each step, trading exact noise-path reproducibility for a simpler draw, which is defensible for single-seed FID at this budget. The SD path forms `denoised` through `self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)`, which bakes in the CFG++ guided prediction and Tweedie; the SDXL path does it explicitly with the VE scalings $c_{\text{in}}=1/\sqrt{\sigma^2+1}$, $c_{\text{out}}=-\sigma$, and `denoised = x + c_out * noise_pred`.

So the delta from 2S is two structural changes bundled into one method: where 2S spent a throwaway call per step for a fresh derivative on a halved grid, 3M reuses two past predictions for both first and second derivatives at one call per step, keeping all twenty steps fine; and where 2S marched the deterministic ODE with error compounding, 3M solves the reverse SDE at $\eta=1.2$, re-injecting Langevin noise the next denoising step washes out. I expect both the finer grid and the re-noising to pull FID down on every variant, with SDXL — 42.83 under 2S, still worst — the largest absolute drop, landing well below 40 for the first time. The risk I carry is the $\eta=1.2$ choice and the simplified noise: too much injected stochasticity at twenty steps can leave residual noise a short chain cannot clean, and the off-tuned $r_1=h_2/h_1$, $+\phi_3 d_2$ constants mean the curvature correction could under- or over-shoot. If FID fails to beat 2S on some variant, the likely culprit is exactly those off-canonical coefficients, pointing the next move at the kdfix correction that restores $r_1=h_2/h$ and $-\phi_3 d_2$.

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
