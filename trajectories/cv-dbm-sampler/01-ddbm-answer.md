**Problem.** Sample a pretrained diffusion bridge `q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)`
for image-to-image translation. A bridge is genuinely one-to-many under a fixed source `x_T`, so a
pure deterministic ODE from the pinned start yields a blurry conditional mean — stochasticity is
required. The reference question for the whole task: what FID is reachable when NFE is no object?

**Key idea (the floor).** Generate by simulating the bridge's own reverse dynamics with a generic
high-order solver. The per-step drift is one object with a single switch:
`d = f·x − g²·(κ·s − h)`, where the learned bridge score `s = −(x − a_t x_T − b_t x̂_0)/c_t²`, the
analytic h-transform `h = −(x − (α_t/α_T) x_T)/(α_t² ρ̄_t²)`, and `κ = 1` for a stochastic step,
`κ = ½` for a deterministic step (only the *learned score* is halved in the ODE; `h` stays full
strength). Each step churns — a short stochastic Euler move (κ=1) injects noise — then takes an
accurate deterministic Heun move (κ=½) back down, on EDM's ρ=7 power-law time grid.

**Why it is the weakest rung.** It treats the reverse dynamics as a black box and pays small-step
discretization error on every part, including the linear part a bridge-specific sampler integrates in
closed form. So it buys quality with calls: `churn_step_ratio = 0.33`, ρ=7, 17 iterations = 16 Heun
(3 calls each) + 1 terminal Euler (2 calls) = **50 NFE**, ≈10× the agent's budget and ≈8× the
wall-clock. Its FID is good in absolute terms but the bridge-specific few-step samplers beat it.

**Hyperparameters.** `churn_step_ratio = 0.33`; EDM grid `ρ = 7`, `n = 17`, `t_min`, `t_max − 1e-4`,
trailing zero appended so the last iteration is the terminal Euler branch; sixth return slot is
`None` (no booting-noise latent — diversity comes from per-step churn).

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
