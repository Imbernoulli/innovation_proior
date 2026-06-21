Image-to-image translation with diffusion bridges is the problem of transporting a source image distribution to a target image distribution along a stochastic path whose two endpoints are fixed. The source and target are already close in pixel space, so the bridge path is short and should be cheap to traverse. Existing bridge families such as DDBM and I2SB achieve good quality, but their transition kernels are built from a pinned reference diffusion, which couples the interpolation weights and the noise level into a small set of parameters. The fast bridge sampler DBIM inherits that coupling and, more importantly, is derived under a positivity condition that forbids strong stochasticity schedules. At small NFE budgets these methods either collapse conditional diversity in one-to-many tasks or leave hard translation workloads with large residual error. What is needed is a path family whose parameters can be tuned independently, a sampler that is valid for any noise level, and a way to sharpen the endpoint without smearing detail.

The method I propose is ECSI, Endpoint-Conditioned Stochastic Interpolants. It builds the bridge directly as a flow map x_t = α_t x_0 + β_t x_T + γ_t z with z ~ N(0, I), where α, β, and γ are independent functions of time subject only to the boundary conditions that pin the endpoints. Because the coefficients are no longer braided together through a shared reference diffusion, the path design space is strictly larger than that of h-transform bridges; DDBM-VP, DDBM-VE, and EDM all emerge as special choices of the three functions. To make the construction usable for paired translation, every quantity is conditioned on the observed terminal image x_T, and the network is trained as a denoiser x̂_0 = E[x_0 | x_t, x_T]. The score is then reparameterized as (α_t x̂_0 + β_t x_T - x_t) / γ_t², which keeps the singular 1/γ² factor out of the network and stabilizes training near the endpoints.

The key insight is that the stochasticity level along the path is a free knob, not a property of the marginals. For any non-negative ε_t, adding a drift ε_t ∇ log p_t with diffusion sqrt(2 ε_t) leaves every marginal unchanged, because the Fokker-Planck diffusion term is cancelled by the divergence of (∇ log p) p. The reverse SDE therefore has the clean drift b = α̇_t x̂_0 + β̇_t x_T + (γ̇_t + ε_t/γ_t) ẑ_t, where ẑ_t is the normalized residual (x_t - α_t x̂_0 - β_t x_T)/γ_t. Setting ε_t = 0 recovers a deterministic ODE, while setting ε_t equal to γ_t γ̇_t - (α̇_t/α_t) γ_t² recovers DDBM's reverse SDE; every intermediate value is reachable. I choose ε_t = η (γ_t γ̇_t - (α̇_t/α_t) γ_t²) with η ∈ [0, 1], giving a single scalar that interpolates from pure ODE to full DDBM-strength stochasticity.

For discretization I use an Euler-SDE step rather than the DBIM closed-form update. The DBIM update contains sqrt(γ² - ρ²), which becomes imaginary when the injected noise is too strong; the Euler form has no such positivity constraint and is valid for any ε_t ≥ 0. To avoid smearing the final image, the last two steps are made deterministic by setting ε_t = 0 and applying the exact DBIM transition that reconstructs the interpolant at the new time from the carried-over noise direction. The path itself is taken to be the straight interpolant α_t = 1 - t, β_t = t, because any invertible reparameterization is equivalent and the line is simplest. The noise envelope is the symmetric arch γ_t = 2 γ_max sqrt(t(1 - t)), which balances detail-building across the path, with γ_max around 0.125 to 0.25. Time-steps under a five-call budget are placed with an EDM-style rho-ramp using rho < 1 to concentrate steps near the sharp target endpoint. Finally, conditional diversity is restored not by increasing sampling noise, which cannot widen the conditional marginal, but by smoothing the base distribution with a small Gaussian convolution π_T = π_cond * N(0, b² I), trading a small amount of input information for genuine output variation.

```python
import torch as th


def linear_route(gamma_max):
    """ECSI straight-line path: alpha (x0 weight), beta (xT weight), gamma (noise)."""
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: -th.ones_like(t)
    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: gamma_max * 2 * (t * (1 - t)) ** 0.5
    gamma_deriv = lambda t: gamma_max * 2 * (1 - 2 * t) / (2 * (t * (1 - t)) ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


def get_sigmas_karras(n, t_min, t_max, rho, device="cpu"):
    """EDM-style rho-ramp; rho < 1 concentrates steps near the sharp endpoint."""
    ramp = th.linspace(0, 1, n, device=device)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


def to_d_stoch(x, x0_hat, x_T, alpha, alpha_deriv, beta, beta_deriv,
               gamma, gamma_deriv, epsilon):
    """ECSI Euler-SDE drift and diffusion coefficient."""
    z_hat = (x - alpha * x0_hat - beta * x_T) / gamma
    drift = (alpha_deriv * x0_hat + beta_deriv * x_T
             + (gamma_deriv + epsilon / gamma) * z_hat)
    diffusion = (2 * epsilon) ** 0.5
    return drift, diffusion


@th.no_grad()
def sample_ecsi(denoiser, x, sigmas, route, eta=1.0, smooth=0.0):
    """
    ECSI sampler. eta=0 gives the deterministic ODE; eta=1 gives full DDBM-strength noise.
    The last two steps are deterministic to sharpen the endpoint.
    smooth>0 convolves the source with Gaussian noise to restore conditional diversity.
    """
    x_T = x
    x = x + smooth * th.randn_like(x)      # π_T = π_cond * N(0, smooth² I)
    x_T_s = x
    s_in = x.new_ones([x.shape[0]])
    alpha, alpha_d, beta, beta_d, gamma, gamma_d = route
    epsilon = lambda t: eta * (
        gamma(t) * gamma_d(t) - alpha_d(t) / alpha(t) * gamma(t) ** 2)

    path = [x.detach().cpu()]
    pred_x0 = [x.detach().cpu()]

    for i in range(len(sigmas) - 1):
        x0_hat = denoiser(x, sigmas[i] * s_in, x_T)     # one budgeted denoiser call
        dt = sigmas[i + 1] - sigmas[i]                  # negative: sigmas decrease to 0

        if i >= len(sigmas) - 3:
            # Last two steps: deterministic DBIM transition for a sharp endpoint.
            x = (alpha(sigmas[i + 1]) * x0_hat
                 + beta(sigmas[i + 1]) * x_T_s
                 + (gamma(sigmas[i + 1]) / gamma(sigmas[i]))
                 * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s))
        else:
            # Early/middle steps: Euler-SDE, valid for any epsilon >= 0.
            drift, diffusion = to_d_stoch(
                x, x0_hat, x_T_s,
                alpha(sigmas[i]), alpha_d(sigmas[i]),
                beta(sigmas[i]), beta_d(sigmas[i]),
                gamma(sigmas[i]), gamma_d(sigmas[i]), epsilon(sigmas[i]))
            x = x + drift * dt + th.randn_like(x) * (dt.abs() ** 0.5) * diffusion

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, pred_x0
```
