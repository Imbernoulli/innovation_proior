**Problem.** Sample the same pretrained bridge under the hard NFE = 5 budget, without retraining.
DDBM reached its FID only by spending 50 calls on a generic black-box discretizer. The task is to
beat that with a fifth the calls by exploiting the bridge's analytic structure.

**Key idea (DBIM, the bridge DDIM).** The training loss depends on the model only through the
per-time marginals, so replace DDBM's joint with a non-Markovian family that *preserves* those
marginals, indexed by a per-step injected std `ρ_n` (forced boundary `ρ_{N-1} = c_{t_{N-1}}`). The
reverse conditional mean is "bridge mean at `t_n` + recycled noise direction + fresh noise," giving
a closed-form large-jump update:
`x_{t_n} = a_t x̂_0 + a_t·(coeff) x_T + √(c_t² − ρ_n²)/c_s · x_{t_{n+1}} + ρ_n ε`. Marginal
preservation is proved by backward induction; the ELBO reduces to the same score-matching minimizer,
so the network is reused as-is.

**Why it beats the floor.** One denoiser call per step (5 NFE) takes clean analytic jumps instead of
many small churned Heun steps. `η` (the `eta` arg) dials stochasticity: `η = 1` Markovian/DDPM-like,
`η = 0` deterministic implicit (sharp, invertible). The first deterministic step divides by
`c_{t_max} = 0`; the **booting noise** (Markovian boundary at step 1) defuses it and is the diversity
latent (sixth return value). Fresh noise is dropped on the final step for endpoint sharpness.

**Hyperparameters.** `eta = 1.0` (caller default); booting noise seeded by `BatchedSeedGenerator`;
injected std `ω_st = η·(α_t ρ_t)·√(1 − ρ_t²/ρ_s²)` (schedule `rho` = σ/α, not the injected ρ); mask
re-blended each call for inpainting. The harness honors the caller's `eta` and `ts` directly.

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
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

        _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [
            append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(t * ones)
        ]

        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)

        x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
```
