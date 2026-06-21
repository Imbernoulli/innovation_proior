A sampler steps something, and before I can be clever about a transition rule under a hard five-call budget I need a process to march and a reference for what "good but slow" even is. So I start by pinning down the floor: the original bridge sampler, the one that gets the quality right by brute force, pays many denoiser calls, and makes no attempt at frugality. It answers the only question that frames the whole task — what FID is reachable when NFE is no object — and every budgeted sampler I build afterward has to approach that number under five calls instead of fifty.

The object I am sampling is a pretrained diffusion bridge: a stochastic process pinned to start at the clean target $x_0$ and arrive almost surely at the informative endpoint $x_T = y$ — a sketch, a degraded photo, a masked image. It is built by Doob's $h$-transform of an ordinary diffusion $dx = f x\,dt + g\,dw$: adding the drift $g^2 \nabla\log p(x_T = y \mid x_t)$ forces the process to hit $y$, and pinning the other end too gives a bridge whose doubly-conditioned forward kernel is an analytic Gaussian,
$$q(x_t \mid x_0, x_T) = \mathcal N\!\big(a_t x_T + b_t x_0,\; c_t^2 I\big),$$
with $a_t = (\alpha_t/\alpha_T)(\mathrm{SNR}_T/\mathrm{SNR}_t)$, $b_t = \alpha_t(1 - \mathrm{SNR}_T/\mathrm{SNR}_t)$, $c_t^2 = \sigma_t^2(1 - \mathrm{SNR}_T/\mathrm{SNR}_t)$, and $\mathrm{SNR}_t = \alpha_t^2/\sigma_t^2$. The ends check out: as $t \to t_{\max}$ the ratio $\mathrm{SNR}_T/\mathrm{SNR}_t \to 1$, so $b_t, c_t \to 0$, $a_t \to 1$, and the kernel collapses to a Dirac at $x_T$ — the pin; as $t \to 0$ the ratio $\to 0$, so $a_t \to 0$, $b_t \to \alpha_t$, $c_t \to \sigma_t$, and the kernel is the ordinary diffusion marginal $\mathcal N(\alpha_t x_0, \sigma_t^2 I)$ around the target. The bridge mean is thus a weighted interpolation between the two endpoints, tight at both ends and fat in the middle — exactly right for translation, where source and target are already close in pixel space. The harness hands me this kernel through `get_abc(t)` and a data predictor `denoiser(x, t)` $\to \hat x_0$.

I propose, as this floor, the **DDBM** sampler: generate by simulating the bridge's *own* reverse dynamics with a generic high-order solver. The bridge is itself a diffusion with drift $f + g^2 h$, where $h = \nabla\log p(x_T \mid x_t)$ is the $h$-transform term I can write in closed form. Anderson's reversal gives a reverse SDE and Song's continuity identity a probability-flow ODE with the same marginals; written through the learned bridge score $s = \nabla\log q(x_t \mid x_T)$ and the analytic $h$, both reduce to one per-step drift with a single switch,
$$d = f\,x - g^2\big(\kappa\, s - h\big),$$
where $s = -(x - a_t x_T - b_t \hat x_0)/c_t^2$, $h = -\big(x - (\alpha_t/\alpha_T) x_T\big)/(\alpha_t^2 \bar\rho_t^2)$ with $\bar\rho_t^2 = \rho_T^2 - \rho_t^2$, and $\kappa = 1$ for a stochastic step, $\kappa = \tfrac12$ for a deterministic one. The factor of one-half is the load-bearing detail and is easy to get wrong: only the *learned score* $s$ is halved in the ODE — the $h$-transform term is part of the bridge's defining forward drift, not the thing the SDE-to-ODE conversion splits, so $h$ stays at full strength in both cases. Collapsing the whole reverse drift into this one switched object is what lets a single helper serve both the stochastic and deterministic moves.

What forces stochasticity here, and what makes a pure fast ODE the wrong instinct, is that a bridge has a fixed, given starting point $x_T = y$ — real data, not a fresh noise draw. Integrating a deterministic ODE backward from one fixed point produces exactly *one* trajectory, the conditional-mean path; but translation is genuinely one-to-many — one edge map admits many handbags, one mask many completions — so $p(x_0 \mid x_T)$ is not a point mass, and the conditional mean of a multimodal distribution is a blurry average. Determinism plus a pinned start equals a washed-out image. So I put noise back predictor-corrector style, using EDM's "churn": each step, briefly add a controlled bit of noise (a short stochastic Euler move at $\kappa=1$ that bumps the noise level up), then take an accurate deterministic Heun move ($\kappa=\tfrac12$) back down. The churn supplies the diversity the bridge needs and corrects accumulated discretization error; the Heun step does the heavy lifting cheaply. I discretize time on EDM's power-law grid $t_i = \big(t_{\max}^{1/\rho} + (i/N)(t_{\min}^{1/\rho} - t_{\max}^{1/\rho})\big)^\rho$ with $\rho = 7$, which equalizes truncation error across the trajectory and is image-friendly, and I append a trailing zero so the last iteration lands exactly on $t = 0$.

The call count is the entire reason this is the floor and not the endpoint. At each interior step I spend one denoiser call on the churn-Euler move and two on the Heun predictor-corrector — three per Heun iteration. On the final interval, where $t_{i+1} = 0$, there is no valid second evaluation at $t = 0$, so I take a single churn-Euler step (one churn call plus one Euler call = two), which also saves a call. With the reference setting — `churn_step_ratio = 0.33`, $\rho = 7$, and 17 iterations — sixteen Heun iterations at three calls plus one terminal iteration at two calls total **50 NFE**, ten times the agent's budget. The structure of that cost is the diagnosis I will carry forward: this sampler treats the whole reverse drift as one opaque vector field and pays small-step discretization error on *every* part of it, including the linear part a smarter sampler could integrate in closed form. It is a generic ODE/SDE discretizer borrowed wholesale from diffusion models, carrying no bridge-specific large-jump update — it buys quality with calls, full stop.

One grounding detail separates this reference from the budgeted rungs to come: the editable `sample_dbim` body here does *not* honor the caller's `ts` or `eta`. It overrides `churn_step_ratio = 0.33` internally, builds its own EDM $\rho=7$ 17-step grid with the trailing zero, and routes every transition through a shared `ddbm_simulate` helper that computes the one-line drift $f x - g^2(\kappa s - h)$ (with $\kappa$ from the `stochastic` flag) and applies either an Euler or a Heun update. It returns `None` in the sixth tuple slot — there is no booting-noise latent, since here the diversity comes from the per-step churn injections rather than a single seeded draw. So this row's numbers are a 50-NFE, churn-0.33, Heun-on-EDM-$\rho{=}7$ sampler, and the budget it used is not available to anything I write next.

```python
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    churn_step_ratio=0.0,
    **kwargs,
):
    x_T = x
    path = []
    pred_x0 = []

    # DDBM reference baseline: 50-NFE gold-standard reference.
    # Each iteration costs:
    #   * churn euler step (stochastic): 1 denoiser call
    #   * Heun 2nd-order step: 2 denoiser calls (or 1 if ts[i+1]==0)
    # With churn_step_ratio>0: 16 Heun-iters (3 NFE ea.) + 1 final Euler-iter
    # (1 churn + 1 Euler = 2 NFE) = 48 + 2 = 50 NFE total.
    # Terminal ts=0 so the last iteration takes the Euler branch.
    #
    # Agent baselines stay at NFE=5 (caller's default). DDBM at 50 NFE is
    # the upper-bound reference agents should try to approach with 10x
    # less compute.
    churn_step_ratio = 0.33
    # EDM/Karras-style rho=7 schedule for this reference sampler.
    _rho = 7.0
    _n = 17  # 17 iters: 16 Heun (3 NFE) + 1 final Euler (2 NFE) = 50 NFE
    _t_min = float(diffusion.t_min)
    _t_max = float(diffusion.t_max - 1e-4)
    _ramp = torch.linspace(0.0, 1.0, _n, device=x.device, dtype=torch.float64)
    _min_inv = _t_min ** (1.0 / _rho)
    _max_inv = _t_max ** (1.0 / _rho)
    _ts_k = (_max_inv + _ramp * (_min_inv - _max_inv)) ** _rho
    # append_zero so last iter takes Euler branch
    ts = torch.cat([_ts_k, torch.zeros(1, device=x.device, dtype=torch.float64)])
    indices = range(len(ts) - 1)

    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):

        if churn_step_ratio > 0:
            # 1 step euler
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                ts[i],
                t_hat,
                stochastic=True,
            )
            nfe += 1
            path.append(x.detach().cpu())
            pred_x0.append(_pred_x0.detach().cpu())
        else:
            t_hat = ts[i]

        # heun step
        if ts[i + 1] == 0:
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
            )
            nfe += 1
        else:
            # Heun's method
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
                second_order=True,
            )
            nfe += 2

        path.append(x.detach().cpu())
        pred_x0.append(_pred_x0.detach().cpu())

    return x, path, nfe, pred_x0, ts, None
```
