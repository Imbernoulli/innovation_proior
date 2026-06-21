The high-order solver did move DIODE for the first time — and the pattern of where it helped and where it barely did is the clue. Edges→Handbags edged from 5.180 to **4.988**, ImageNet inpainting from 6.070 to **5.528**, and DIODE, frozen at 15, finally dropped from 15.002 to **14.306**. The DIODE bottleneck really was solver order, and the second-order exponential integrator paid it down — but the *magnitude* of that gain is small, and DIODE is still an order of magnitude worse than the easy translation. The high-order solver squeezed the deterministic family about as far as it goes. So the remaining error on hard translation is not about taking more accurate steps along the same trajectory — it is about *which* trajectory I am on, and the one degree of freedom this whole family never had is *how stochastic* the sampling is. DBIM and its high-order solver are built deterministic, with the only noise being the single boot. On a genuinely one-to-many task, scheduled stochasticity through the trajectory may be exactly what carves the conditional sample sharp, and the deterministic family structurally cannot reach it. EDM's lesson applies: the path you train on and the sampler you run are separate design problems — and I have explored neither the path nor the stochasticity level.

I propose **ECSI**, endpoint-conditioned stochastic interpolants. Take the path first. The Doob-bridge kernel's three coefficients $a_t, b_t, c_t$ are all functions of the *same* two schedule functions $\alpha_t, \sigma_t$ — braided together, so I cannot change how much noise the path carries in the middle without dragging the interpolation weights around. That coupling is an artifact of pinning a reference SDE, not a law. The fix is to build the bridge *directly* as a flow map $x_t = \alpha_t x_0 + \beta_t x_T + \gamma_t z$, $z \sim \mathcal N(0,I)$, asking only for boundary conditions that land the ends ($\alpha_0 = \beta_T = 1$, $\alpha_T = \beta_0 = \gamma_0 = \gamma_T = 0$). Now $\alpha_t, \beta_t, \gamma_t$ are three *independent* functions; the kernel is the same Gaussian $\mathcal N(\alpha_t x_0 + \beta_t x_T, \gamma_t^2 I)$, and DBIM's coupled kernel is one particular choice, so I lose nothing and only un-cramp the space. (In the harness names, code `a_t` multiplies $x_T$ and is my $\beta$, code `b_t` multiplies $x_0$ and is my $\alpha$, code `c_t` is my $\gamma$.) Because this flow map is unconditional, I condition everything on the observed $x_T$ and keep the network a denoiser $\hat x_0 = E[x_0 \mid x_t, x_T]$, so it stays a one-model bridge. Realizing the kernel as a linear SDE $dX_t = (f_t X_t + s_t x_T)\,dt + g_t\,dW_t$ and matching the mean and variance evolution gives $f_t = \dot\alpha_t/\alpha_t$, $s_t = \dot\beta_t - (\dot\alpha_t/\alpha_t)\beta_t$, $g_t^2 = 2(\gamma_t\dot\gamma_t - (\dot\alpha_t/\alpha_t)\gamma_t^2)$, and the EDM reparameterization yields the clean affine score $\nabla\log p_t(x_t \mid x_T) = (\alpha_t \hat x_0 + \beta_t x_T - x_t)/\gamma_t^2$ — pointing from the current state toward the predicted clean image, scaled by $1/\gamma^2$.

The heart of the move is the sampler degree of freedom the deterministic family never used. Take any deterministic flow $dX_t = u_t\,dt$ with density $p_t$ and add drift-plus-diffusion: $dX_t = (u_t + \varepsilon_t \nabla\log p_t)\,dt + \sqrt{2\varepsilon_t}\,dW_t$. In its Fokker–Planck equation the added drift contributes $-\varepsilon_t \nabla\cdot[(\nabla\log p_t)p]$, and since $(\nabla\log p_t)p = \nabla p_t$ that is $-\varepsilon_t \nabla^2 p_t$, which exactly cancels the $+\varepsilon_t \nabla^2 p$ from diffusion. What is left is the Fokker–Planck of the *original* ODE — the marginals are untouched for *any* non-negative $\varepsilon_t$. So the noise level along the path is a genuine free function $\varepsilon_t$ sitting on top of $\alpha, \beta, \gamma$, changing nothing about the distributions I sample, only how the trajectory wanders between them. Plugging the reparameterized score into the reverse SDE collapses the drift to
$$b = \dot\alpha_t \hat x_0 + \dot\beta_t x_T + \big(\dot\gamma_t + \varepsilon_t/\gamma_t\big)\hat z_t, \qquad \hat z_t = \frac{X_t - \alpha_t \hat x_0 - \beta_t x_T}{\gamma_t},$$
the normalized residual — move the clean estimate at rate $\dot\alpha$, the endpoint at rate $\dot\beta$, and along the predicted noise direction at rate $\dot\gamma + \varepsilon/\gamma$ — with diffusion $\sqrt{2\varepsilon}$. Setting $\varepsilon_t = 0$ is a pure ODE; setting $\varepsilon_t = \gamma_t\dot\gamma_t - (\dot\alpha_t/\alpha_t)\gamma_t^2 = \tfrac12 g_t^2$ recovers DDBM's reverse SDE. DDBM was using one specific $\varepsilon_t$ and calling it "the" SDE; I parameterize $\varepsilon_t = \eta\,(\gamma_t\dot\gamma_t - (\dot\alpha_t/\alpha_t)\gamma_t^2)$, $\eta \in [0,1]$, one scalar dialing from pure ODE to full DDBM-strength noise.

Why this reaches where the high-order solver could not comes down to *how* I discretize. Euler on the SDE is $x_{t-h} \approx x_t - b(t)h + \sqrt{2\varepsilon_t h}\,\bar z$. Rearranging it in the regime $\gamma_{t-h}^2 - 2\varepsilon_t h > 0$ recovers *exactly* the DBIM update with $\rho^2 = 2\varepsilon_t h$ — so DBIM is my family restricted to that positivity condition, and that restriction is the cap I have been fighting: when I want aggressive noise, $\gamma_{t-h}^2 - 2\varepsilon_t h$ goes negative, the $\sqrt{\gamma^2 - \rho^2}$ in the closed DBIM form turns imaginary, and the update is undefined. The Euler form $x_t - bh + \sqrt{2\varepsilon_t h}\,\bar z$ has no positivity requirement — it is well-defined for any $\varepsilon_t \ge 0$. So I ship the Euler-SDE form precisely to run the strong stochasticity the previous two rungs structurally could not, attacking DIODE's residual error, the part deterministic sampling cannot reach.

Strong noise *everywhere* is wrong, and this is the single most important sampler decision — worth more than the exact $\eta$. Following the Euler-SDE all the way to $t = 0$ keeps dumping fresh noise into the state right when the image should crystallize, smearing high-frequency detail — the same endpoint-blur the previous rungs avoided by dropping fresh noise on the final step, only sharper. So for the **last two steps** I set $\varepsilon_t = 0$ and take the deterministic DBIM transition $x_{t-h} = \alpha_{t-h}\hat x_0 + \beta_{t-h} x_T + \gamma_{t-h}\hat z_t$, well-defined ($\gamma^2 > 0$ always) and committing cleanly. The sampler is therefore two-phase: Euler-SDE with the $\eta$-dialed $\varepsilon_t$ for early/middle steps where stochasticity builds detail, then deterministic for the final two to sharpen the endpoint. The schedule choices fall out of arguments, not sweeps. With only five calls I bunch steps where the trajectory changes fastest and where sharpness is decided, near the small-$t$ endpoint — EDM's $\rho$-ramp, but with $\rho$ taken *below* one (the edit uses `rho_k = 0.6`), the opposite of EDM's $\rho = 7$ for unconditional generation, because here the hard part is the sharp endpoint, not the noisy end. And the derivatives $\dot\alpha, \dot\beta, \dot\gamma$ are read off *analytically* from the VP schedule via `get_f_g2` ($f = (\log\alpha)'$, $g^2$), giving $\dot\alpha = \alpha f$, $\dot\rho = \tfrac12(\rho^2 + 1)g^2/\rho$ and then the chain rule — finite-differencing near the boundary where $\gamma \sim O(10^{-2})$ would lose accuracy and destabilize the strong-noise drift.

The authority is the edit, not the generic construction, and it makes three substantive trims. First, this `sample_dbim` *ignores* the caller's `eta` and `ts`: it hardcodes stochasticity at `churn = 0.3` (not 1.0, not the caller's value) and builds its own Karras `rho_k = 0.6` schedule from `sigma_min = 0.15` and `sigma_max_offset = 5e-4`, overriding the passed `ts` — task-local edges2handbags sweep values. Second, the full method has a third knob — a base-distribution diversity fix $\pi_T = \pi_{\text{cond}} * \mathcal N(0, b^2 I)$ that lossy-compresses the input to restore conditional diversity, since the stochasticity lemma proves more sampling noise alone cannot widen the conditional — and this edit **does not implement it at all**: it starts from the source $x_T$ with no base smoothing. Third, the edit re-applies the mask *after every update* ($x = x\cdot\text{mask} + x_T\cdot(1-\text{mask})$), not just inside the denoiser blend, because for inpainting the unmasked border pixels must stay pinned or the SDE noise accumulates on the known region and FID explodes; `eps` is also clamped non-negative for numerical safety. One denoiser call per step, five NFE.

```python
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs,
):
    """
    ECSI (Endpoint-Conditioned Stochastic Interpolants) sampler.
    Paper: Zhang et al. arXiv:2410.21553
    ('Exploring the Design Space of Diffusion Bridge Models').
    Code: https://github.com/szhan311/ECSI  (sibm/sampling.py: sample_stoch).

    Task-local ECSI-inspired sampler settings:
      * pred_mode = "vp"  (already the dbim-codebase e2h default)
      * sigma_min is set below from the local e2h sweep
      * churn_step_ratio = 0.3
      * rho = 0.6
      * NFE = steps (5 for e2h)

    Convention mapping ECSI(alpha,beta,gamma) -> dbim-codebase(b_t,a_t,c_t):
    dbim's x_t = a_t*x_T + b_t*x_0 + c_t*noise, so ECSI's alpha (x_0 coef)
    = dbim's b_t, beta = a_t, gamma = c_t. Derivatives are computed
    analytically from the VP schedule (dbim-codebase exposes f_fn = -(ln alpha)'
    and g2_fn = (rho^2 + 1)' / (rho^2 + 1), which give us alpha'(t) and rho'(t)
    exactly — finite differences would lose ~1e-4 accuracy near the t_max
    boundary where c(t) ~ O(1e-2) itself).
    """
    churn = 0.3
    rho_k = 0.6
    sigma_min_ecsi = 0.15   # task-local e2h sweep value
    sigma_max_offset = 5e-4 # paired task-local sweep value
    t_max = diffusion.t_max
    ns = diffusion.noise_schedule
    alpha_T = float(ns.alpha_T)
    rho_T = float(ns.rho_T)
    rho_T2 = rho_T * rho_T

    # --- Karras rho=0.6 schedule (ECSI's native setup for e2h) ------------
    n = len(ts)
    t_lo = sigma_min_ecsi
    t_hi = t_max - sigma_max_offset
    min_inv_rho = t_lo ** (1.0 / rho_k)
    max_inv_rho = t_hi ** (1.0 / rho_k)
    ramp = torch.linspace(0.0, 1.0, n, device=x.device, dtype=torch.float64)
    ts_k = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho_k
    ts = torch.cat([ts_k, torch.tensor([float(diffusion.t_min)], device=x.device, dtype=ts_k.dtype)])

    x_T = x
    path = [x.detach().cpu()]
    pred_x0 = []
    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    generator = BatchedSeedGenerator(seed)
    first_noise = generator.randn_like(x)  # return-contract compatibility

    def _abc_and_deriv(t_scalar):
        """Analytical (a, b, c) and their t-derivatives at scalar t for VP.

        VP formulas (see dbim-codebase/ddbm/karras_diffusion.py VPNoiseSchedule):
            alpha(t)   = exp(-0.5 β_min t - 0.25 β_d t^2)
            alpha'(t)  = alpha(t) * f_fn(t)               f_fn = -0.5*(β_min+β_d*t)
            rho(t)     = sqrt(exp(β_min t + 0.5 β_d t^2) - 1)
            rho'(t)    = 0.5 * (rho^2 + 1) * g2_fn(t) / rho     g2_fn = β_min + β_d*t
            a(t) = α_bar * ρ^2 / ρ_T^2, α_bar = α/α_T
            b(t) = α * ρ_bar^2 / ρ_T^2, ρ_bar^2 = ρ_T^2 - ρ^2
            c(t) = α * ρ_bar * ρ / ρ_T
        """
        t_clamped = t_scalar.clamp(min=1e-6, max=t_max - 1e-6)
        t = t_clamped * ones
        alpha, alpha_bar, rho, rho_bar = ns.get_alpha_rho(t)
        alpha = append_dims(alpha, x.ndim)
        alpha_bar = append_dims(alpha_bar, x.ndim)
        rho = append_dims(rho, x.ndim)
        rho_bar = append_dims(rho_bar, x.ndim)

        f_t, g2_t = ns.get_f_g2(t)
        f_t = append_dims(f_t, x.ndim)
        g2_t = append_dims(g2_t, x.ndim)

        alpha_d = alpha * f_t
        rho_d = 0.5 * (rho**2 + 1.0) * g2_t / rho

        rho_sq = rho * rho
        rho_bar_sq = rho_bar * rho_bar
        a = alpha_bar * rho_sq / rho_T2
        b = alpha * rho_bar_sq / rho_T2
        c = alpha * rho_bar * rho / rho_T

        alpha_bar_d = alpha_d / alpha_T
        rho_bar_sq_d = -2.0 * rho * rho_d
        rho_bar_d = -rho * rho_d / rho_bar

        a_d = (alpha_bar_d * rho_sq + alpha_bar * 2.0 * rho * rho_d) / rho_T2
        b_d = (alpha_d * rho_bar_sq + alpha * rho_bar_sq_d) / rho_T2
        c_d = (alpha_d * rho_bar * rho + alpha * rho_bar_d * rho + alpha * rho_bar * rho_d) / rho_T

        return (a, b, c), (a_d, b_d, c_d)

    nfe = 0
    n_steps = len(ts) - 1
    for step_idx, i in enumerate(indices):
        s = ts[i]
        t_next = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        (a_s, b_s, c_s), (a_d, b_d, c_d) = _abc_and_deriv(s)

        if step_idx >= n_steps - 2:
            # Last 2 iterations: DBIM deterministic transition.
            a_t, b_t, c_t = [append_dims(v, x.ndim) for v in ns.get_abc(t_next * ones)]
            x = b_t * x0_hat + a_t * x_T + (c_t / c_s) * (x - b_s * x0_hat - a_s * x_T)
        else:
            # Euler-SDE step (ECSI).
            eps = churn * (c_s * c_d - (b_d / b_s) * c_s**2)
            eps = eps.clamp(min=0)

            z_hat = (x - b_s * x0_hat - a_s * x_T) / c_s
            drift = b_d * x0_hat + a_d * x_T + (c_d + eps / c_s) * z_hat
            diff_coef = (2.0 * eps).sqrt()

            dt = t_next - s
            step_noise = generator.randn_like(x)
            x = x + drift * dt + diff_coef * step_noise * dt.abs().sqrt()

        if mask is not None:
            # Preserve the known (mask=0) region at x_T after every update.
            # For inpainting, unmasked border pixels must stay put; without
            # this the SDE noise accumulates on the known region and FID
            # explodes in the local inpainting harness.
            x = x * mask + x_T * (1 - mask)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
```
