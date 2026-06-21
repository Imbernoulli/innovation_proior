The third-order multistep SDE paid off across the board and landed where I called it. DPM++(3M) SDE pulled every variant below 2S — SD v1.5 from 29.01 to 27.98, SD v2.0 from 23.89 to 23.45, and SDXL, again the one I said had the most to gain, from 42.83 to 39.66, the first time any sampler put SDXL under 40. Both diagnoses from the 2S rung held: keeping all twenty steps fine beat the halved singlestep grid, and the Langevin re-noising at $\eta=1.2$ bought back the global structure a deterministic march drifts away from. But every sampler I have run — DDIM, 2S, 3M — is a **predictor**: at each step it forms an estimate of the next latent from information it already has and then *commits*, with the leading truncation error of that step baked in and never refined. 3M's third-order curvature term shrinks that error and the SDE noise washes out some of what remains — but notice the noise is a *statistical* correction for accumulated drift, not a correction of the *individual step's* deterministic error; the $\eta=1.2$ stochasticity is partly compensating for the very thing a better-corrected step would not produce. So the question is whether I can correct each step's leading error *directly*, within the loop, rather than letting it accumulate and dousing it with noise.

The classical answer is a predictor-corrector pair: take the predictor step, evaluate the right-hand side at the predicted point, and use that fresh evaluation to refine the step, gaining an order. The catch in a generic ODE is that the corrector's evaluation is extra cost, and at NFE = 20 I cannot afford a single extra model call — 3M already spends all twenty on one call per step, so a corrector looks disqualified on budget. But in a *multistep* loop, look at what the next step already does: at the start of step $i+1$, the predictor's base evaluation is the network applied at the latent step $i$ just predicted — *exactly* the evaluation a corrector for step $i$ would need. The corrector's "extra" call is already being taken by the next step's predictor. So in a multistep diffusion sampler the corrector is **free**: I reuse step $i+1$'s evaluation to correct step $i$, and the total call count stays at twenty. This is the lever none of the three rungs pulled — each spent its evaluations purely on prediction; I can spend the same evaluations on prediction *and* correction, raising the realized order by one at no budget cost.

If this were the whole story, 3M would already be a predictor-corrector. The real obstacle is derivation cost: a corrector of order $p$ is a *different* formula from the predictor of order $p$, and each order is a separate hand-derivation of exponential-integrator coefficients, so people derive a second- and third-order predictor and stop. What is missing is a *unified* form yielding both predictor and corrector at arbitrary order from one template — then "add a corrector" is the same one-line change as "go one order higher."

I propose **UniPC**, a unified predictor-corrector, and I derive it in the data-prediction face because that is the one that stays stable under the large CFG++ guidance the substrate uses, and because it lets me work in the clean-image estimate the harness's Tweedie step already produces. The exact data-prediction step from the current level $s_0$ to the next $t$ approximates an integral of $x_\theta(\lambda)$ in half-log-SNR. Holding $x_\theta$ at its current value $m_0$ gives the first-order base step; the higher-order terms are the Taylor derivatives of $x_\theta$, estimated from finite differences of past data predictions. Keep the recent predictions $m_0, m_1, m_2,\dots$ at $\lambda$-spacings, define the ratios $r_k=(\lambda_{s_k}-\lambda_{s_0})/h$ with $h=\lambda_t-\lambda_{s_0}$ (and $hh=-h$ in this convention, the sign the implementation uses for $\mathrm{predict\_x0}$), and the scaled differences $D1_k=(m_k-m_0)/r_k$. A linear combination $\sum_k\rho_k D1_k$ reproduces the Taylor correction, and the coefficients $\rho_k$ are exactly the solution of a small linear system $R\rho=b$, where $R$ has rows $r_k^{\,i-1}$ (a Vandermonde structure in the ratios) and $b$ is the $\phi$-derived sequence $h\phi_k\cdot i!/B(h)$, advanced by $h\phi_k\leftarrow h\phi_k/hh-1/(i+1)!$. Matching the system to the exact integral's Taylor expansion to order $p$ makes the method order-$p$ for *any* $p$ — that single linear solve is the whole unification, with no per-order algebra.

Predictor and corrector are then the *same* update with different amounts of information. The predictor (UniP) does not yet have an evaluation at the new point $t$, so it solves the *reduced* system: for order $p$ it uses $\rho_p=\mathrm{solve}(R[:\!-1,:\!-1],b[:\!-1])$ (order 2 collapses to $\rho_p=0.5$), and the step is $x_{\text{base}}-\alpha_t B(h)\,(\sum_k\rho_{p,k}D1_k)$, where $x_{\text{base}}=(\sigma_t/\sigma_{s_0})x-\alpha_t\,h\phi_1\,m_0$. The corrector (UniC) runs *after* the predictor has stepped and the next iteration has evaluated the network at the predicted point, giving $m_t=x_\theta$ there. Now there is one *more* usable evaluation, so the corrector solves the *full* $p$-dimensional system $\rho_c=\mathrm{solve}(R,b)$ (order 1 gives $\rho_c=0.5$) and refines the same base step with the extra difference $D1_t=m_t-m_0$:

$$x=x_{\text{base}}-\alpha_t B(h)\Big(\textstyle\sum_k\rho_{c,k}D1_k+\rho_{c,\text{last}}D1_t\Big).$$

The free scalar $B(h)$ sets the error constant; I take $B(h)=e^{hh}-1$ (the "bh2" choice), which tracks the exponential weight better at the large $h$ a twenty-step budget forces, over the simpler $B(h)=hh$ ("bh1"). The loop is therefore: at each step make the one model call, first run the corrector on the *previous* step using that call, then run the predictor for the current step — one evaluation doing double duty.

In this task's edit surface the substrate is the same `sample` body I have been filling, so I express UniPC in its vocabulary. The loop walks `self.scheduler.timesteps` exactly as DDIM did — twenty steps, one `predict_noise` per step, NFE stays at twenty. At each timestep I form the guided prediction $\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$ and the data prediction by Tweedie, $m_t=z_0=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$ — that single evaluation is both the corrector input for the step I took last iteration and the predictor base for this one. I keep $\bar\alpha$ via `self.alpha(t)` and compute $\lambda=\tfrac12(\log\bar\alpha-\log(1-\bar\alpha))$ from it, so I never leave the substrate's $(\bar\alpha,z_0)$ language for a separate $\sigma$ schedule. The $R\rho=b$ machinery, the reduced-vs-full solves, the `expm1` for the $h\phi$ factors, and the order ramp (1 → 2 → 3 as history fills, dropping on the final step where there is no future evaluation to correct with) all live in the body, and the final step returns the clean data prediction $z_0$ itself, exactly as 3M's $\sigma=0$ step does. Two things I deliberately do *not* import: there is no thresholding (latent space, no $[-1,1]$ bound), and there is no SDE noise term — UniPC is a deterministic predictor-corrector, and the whole point is that *correcting* each step's leading error is a more direct fix than 3M's stochastic compensation for uncorrected error. The CFG++ convention persists in that the data prediction uses the guided $\tilde\epsilon$ while the unconditional stream stays available for the base; the renoising 2S/DDIM did with $\epsilon_{uc}$ is subsumed into the data-prediction step.

The bar this has to clear is 3M's measured numbers, and the claim is falsifiable: correcting each step's leading truncation error within the loop, at the same twenty calls with no extra evaluation, beats predicting-and-renoising. So I expect UniPC below 27.98 on SD v1.5, below 23.45 on SD v2.0, and below 39.66 on SDXL, with the largest absolute improvement again on SDXL where uncorrected per-step error has been the dominant cost at every rung — and a deterministic corrected step should not need the $\eta=1.2$ noise 3M spent to mask its drift. The clean way this is falsified: if UniPC fails to beat 39.66 on SDXL, then at this budget the corrector's extra order is not worth more than the stochastic correction it replaces, and the right next move would be a *stochastic* predictor-corrector rather than abandoning the corrector. But the expectation I commit to is that the unified predictor-corrector, correcting where every previous rung only predicted, is the strongest sampler of the four, and shows it most on SDXL.

```python
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    UniPC sampler with CFG++.
    Unified predictor-corrector (bh2, data-prediction); +1 order per step at no extra NFE.
    """
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    def _lam(self, at):
        # half log-SNR lambda = log(alpha/sigma) = 0.5*log(at/(1-at))
        return 0.5 * (at.log() - (1 - at).log())

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               max_order=3,
               **kwargs):

        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        timesteps = list(self.scheduler.timesteps)
        N = len(timesteps)

        def R_b(rks, hh, B_h, order):
            h_phi_1 = torch.expm1(hh)
            R, b = [], []
            h_phi_k = h_phi_1 / hh - 1
            factorial_i = 1
            for i in range(1, order + 1):
                R.append(torch.pow(rks, i - 1))
                b.append(h_phi_k * factorial_i / B_h)
                factorial_i *= i + 1
                h_phi_k = h_phi_k / hh - 1 / factorial_i
            return torch.stack(R), torch.stack(b), h_phi_1

        def ratios(m_list, lam_list, lam_s0, h, order, device):
            rks, D1s = [], []
            m0 = m_list[-1]
            for i in range(1, order):
                rk = (lam_list[-(i + 1)] - lam_s0) / h
                rks.append(rk)
                D1s.append((m_list[-(i + 1)] - m0) / rk)
            rks.append(torch.ones((), device=device))
            return torch.stack(rks), D1s

        def step_core(x, at_s0, at_t, m_list, lam_list, order, model_t=None):
            # shared predictor/corrector body in the data-prediction face (bh2)
            lam_s0, lam_t = self._lam(at_s0), self._lam(at_t)
            h = lam_t - lam_s0
            hh = -h                                          # predict_x0
            B_h = torch.expm1(hh)                            # bh2
            m0 = m_list[-1]
            rks, D1s = ratios(m_list, lam_list, lam_s0, h, order, x.device)
            R, b, h_phi_1 = R_b(rks, hh, B_h, order)
            sigma_s0, sigma_t = (1 - at_s0).sqrt(), (1 - at_t).sqrt()
            x_base = (sigma_t / sigma_s0) * x - at_t.sqrt() * h_phi_1 * m0
            if model_t is None:                              # predictor (UniP)
                if D1s:
                    D1s = torch.stack(D1s, dim=1)
                    if order == 2:
                        rhos = torch.tensor([0.5], dtype=x.dtype, device=x.device)
                    else:
                        rhos = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(x.dtype)
                    res = torch.einsum("k,bk...->b...", rhos, D1s)
                else:
                    res = 0
                return x_base - at_t.sqrt() * B_h * res
            else:                                            # corrector (UniC)
                if order == 1:
                    rhos = torch.tensor([0.5], dtype=x.dtype, device=x.device)
                else:
                    rhos = torch.linalg.solve(R, b).to(x.dtype)
                D1_t = model_t - m0
                if D1s:
                    D1s = torch.stack(D1s, dim=1)
                    corr = torch.einsum("k,bk...->b...", rhos[:-1], D1s)
                else:
                    corr = 0
                return x_base - at_t.sqrt() * B_h * (corr + rhos[-1] * D1_t)

        m_list, lam_list = [], []
        x_prev, at_prev_s0 = None, None
        z0t = None
        pbar = tqdm(range(N), desc="UniPC")
        for i in pbar:
            t = timesteps[i]
            at = self.alpha(t)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            model_t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()   # data prediction (z0)
            z0t = model_t

            if x_prev is not None:                            # correct the previous predictor step
                order_c = min(max_order, len(m_list))         # past points available for the corrector
                zt = step_core(x_prev, at_prev_s0, at, m_list, lam_list, order_c, model_t=model_t)

            m_list.append(model_t)
            lam_list.append(self._lam(at))

            if i == N - 1:
                z0t = m_list[-1]
                break

            order_p = min(max_order, len(m_list), N - i)
            x_prev, at_prev_s0 = zt, at
            at_next = self.alpha(t - self.skip)
            zt = step_core(zt, at, at_next, m_list, lam_list, order_p)

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```
