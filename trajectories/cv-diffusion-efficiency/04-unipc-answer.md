**Problem.** DPM-Solver++(3M) SDE is the strongest baseline (FID 27.98 / 23.45 / 39.66 on SD v1.5 /
v2.0 / SDXL), but it is a *predictor-only* solver: each multistep step extrapolates the history and
commits, with its leading truncation error uncorrected, and it spends stochasticity to paper over the
accumulated drift. At twenty steps that uncorrected per-step error is exactly what still separates it
from the model's quality ceiling.

**Key idea.** A unified **predictor-corrector** (UniPC). The predictor (UniP) is the same kind of
multistep data-prediction step. The corrector (UniC) refines the *previous* step using the model
evaluation the current step already computes — so it raises the realized order by one at **no extra
NFE**. Both are one analytical update whose high-order coefficients come from solving a tiny linear
system `R rho = b` in the half-log-SNR ratios `r_k = (lambda_{s_k}-lambda_{s0})/h`, with the predictor
solving the reduced `(p-1)`-system and the corrector the full `p`-system. `B(h) = e^{hh}-1` (bh2),
`hh = -h` (data-prediction convention).

**Why.** One network call per step, same budget as 3M, but each step's leading error is corrected
within the loop instead of left to accumulate — which is precisely the few-step regime where it pays.
The unified linear-system form means the corrector is finally cheap to have at arbitrary order. Order
ramps with history (1 → 2 → 3) and drops on the final step (no future evaluation to correct with);
latent space, so no thresholding. Renoising stays in the CFG++ convention via the guided/unconditional
predictions inside the data estimate.

**Hyperparameters.** `max_order = 3`; `solver_type = "bh2"`; `predict_x0 = True`; `lower_order_final`;
NFE = 20 (one `predict_noise` per timestep); `cfg_guidance = 7.5`.

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
