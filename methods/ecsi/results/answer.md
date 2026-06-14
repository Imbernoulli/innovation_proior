# ECSI — Endpoint-Conditioned Stochastic Interpolants

## Problem

Image-to-image translation with diffusion bridges. The best-quality bridge families (DDBM, I2SB)
are slow (hundreds of denoiser calls) and built on a transition kernel whose path parameters are
coupled, restricting the design space; the fast sampler (DBIM) inherits that restriction and, via
a positivity condition, cannot realize strong noise schedules; and bridges collapse conditional
diversity in one-to-many tasks. ECSI wants a decoupled bridge family plus a sampler that is fast
(small NFE), can use any stochasticity level, and produces sharp endpoints — with a knob to
restore diversity.

## Key ideas

1. **Decoupled path.** Build the bridge as a stochastic interpolant flow map
   `x_t = α_t x_0 + β_t x_T + γ_t z`, `z ~ N(0, I)`, with `α, β, γ` independent functions
   (boundary conditions `α_0 = β_T = 1`, `α_T = β_0 = γ_0 = γ_T = 0`). Kernel
   `p_{t|0,T}(x_t|x_0,x_T) = N(α_t x_0 + β_t x_T, γ_t² I)`. DDBM-VP/VE, I2SB, EDM are special cases.

2. **Endpoint conditioning + denoiser score reparameterization.** Condition on the observed `x_T`,
   train one denoiser `x̂_0 = E[x_0 | x_t, x_T]` (EDM-style preconditioning keeps `1/γ²` out of the
   network), and use
   `∇_{x_t} log p_t(x_t|x_T) = (α_t x̂_0 + β_t x_T − x_t)/γ_t²`.

3. **Stochasticity as a free knob.** The kernel is the marginal of the linear SDE
   `dX_t = (f_t X_t + s_t x_T) dt + g_t dW_t` with `f_t = α̇_t/α_t`,
   `s_t = β̇_t − (α̇_t/α_t)β_t`, `g_t² = 2(γ_t γ̇_t − (α̇_t/α_t)γ_t²)`. For *any* `ε_t ≥ 0`, adding
   the forward drift `+ε_t ∇log p_t` with diffusion `√(2ε_t)` preserves the ODE marginals because
   `−ε_t∇·[(∇log p)p] = −ε_t∇²p` cancels the `+ε_t∇²p` Fokker–Planck diffusion term. The backward
   sign gives the analogous cancellation for the reverse-time equation. So `ε_t` is an extra
   sampler degree of freedom, decoupled from the marginals.

4. **Clean reverse SDE.** Plugging the reparameterized score in and simplifying:
   ```
   dX_t = b(t, X_t, x_T) dt + √(2 ε_t) dW_t,
   b = α̇_t x̂_0 + β̇_t x_T + (γ̇_t + ε_t/γ_t) ẑ_t,   ẑ_t = (X_t − α_t x̂_0 − β_t x_T)/γ_t.
   ```
   `ε_t = 0` → ODE; `ε_t = γ_t γ̇_t − (α̇_t/α_t)γ_t²` → DDBM's reverse SDE.

5. **Sampler: Euler-SDE, deterministic at the endpoint.** Discretize by Euler (no positivity
   constraint, unlike the DBIM closed form which needs `γ_{t-h}² − 2ε_t h > 0`) with
   `ε_t = η(γ_t γ̇_t − (α̇_t/α_t)γ_t²)`, `η ∈ [0,1]`, dialing ODE → DDBM-strength SDE. For the
   **last two steps** set `ε_t = 0` and take the deterministic transition
   `x_{t-h} = α_{t-h} x̂_0 + β_{t-h} x_T + γ_{t-h} ẑ_t` to sharpen the endpoint.

6. **Schedules.** Linear path `α_t = 1−t, β_t = t` (any invertible `β_t` is reparameterization-
   equivalent, so pick the line); symmetric noise arch `γ_t = 2 γ_max √(t(1−t))` (`k=1`),
   `γ_max ∈ {0.125, 0.25}`; EDM time-step ramp
   `t_i = (t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ`, `t_min=0.001`, `t_max=1−10⁻⁴`,
   `ρ ≈ 0.6` (concentrate steps near the sharp endpoint).

7. **Conditional diversity.** More sampling noise cannot help (it does not change the conditional
   marginal). Instead modify the base distribution `π_T = π_cond * N(0, b² I)`: lossy-compressing
   the input trades a little input information for genuine output variation (a VAE-style
   information-bottleneck effect), interpolating between a pure bridge (`b=0`) and a diffusion
   model (`b→∞`).

## Algorithm (ECSI sampler)

```
Input: denoiser D_θ(x_t, x_T, t); time-steps {t_j}; base noise b; schedule α,β,γ,γ̇; η.
Sample x_T ~ π_cond, n_0 ~ N(0, b²I); set x_N = x_T + n_0.
for i = N down to 1:
    x̂_0 = D_θ(x_i, x_T, t_i);  ẑ_i = (x_i − α_{t_i} x̂_0 − β_{t_i} x_N) / γ_{t_i}
    if i > 2 (not the last two steps):
        ε = η (γ_{t_i} γ̇_{t_i} − (α̇_{t_i}/α_{t_i}) γ_{t_i}²)
        b_i = α̇_{t_i} x̂_0 + β̇_{t_i} x_N + (γ̇_{t_i} + ε/γ_{t_i}) ẑ_i;   sample z̄ ~ N(0,I)
        h = t_i − t_{i-1}
        x_{i-1} = x_i − b_i h + √(2 ε h) z̄
    else:
        x_{i-1} = α_{t_{i-1}} x̂_0 + β_{t_{i-1}} x_N + γ_{t_{i-1}} ẑ_i
```

## Linear-path sampler code

```python
import torch as th


def linear_route(gamma_max):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: -th.ones_like(t)
    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: gamma_max * 2 * (t * (1 - t)) ** 0.5
    gamma_deriv = lambda t: gamma_max * 2 * (1 - 2 * t) / (2 * (t * (1 - t)) ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


def get_sigmas_karras(n, t_min, t_max, rho, device="cpu"):
    ramp = th.linspace(0, 1, n, device=device)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


def to_d_stoch(x, x0_hat, x_T, alpha, alpha_deriv, beta, beta_deriv,
               gamma, gamma_deriv, epsilon):
    z_hat = (x - alpha * x0_hat - beta * x_T) / gamma           # normalized residual ẑ
    drift = alpha_deriv * x0_hat + beta_deriv * x_T + (gamma_deriv + epsilon / gamma) * z_hat
    diffusion = (2 * epsilon) ** 0.5
    return drift, diffusion


@th.no_grad()
def sample_stoch(
    denoiser, x, sigmas, route, progress=False, callback=None,
    churn_step_ratio=0.0, route_scaling=0, smooth=0.0
):
    x_T = x
    x = x + smooth * th.randn_like(x)        # π_T = π_cond * N(0, b²I), b = smooth
    x_T_s = x
    s_in = x.new_ones([x.shape[0]])
    alpha, alpha_d, beta, beta_d, gamma, gamma_d = route
    epsilon = lambda t: churn_step_ratio * (
        gamma(t) * gamma_d(t) - alpha_d(t) / alpha(t) * gamma(t) ** 2)

    path, x0_est = [x.detach().cpu()], [x.detach().cpu()]
    indices = range(len(sigmas) - 1)
    for i in indices:
        x0_hat = denoiser(x, sigmas[i] * s_in, x_T)            # one budgeted denoiser call
        x0_est.append(x0_hat.detach().cpu())
        dt = sigmas[i + 1] - sigmas[i]                         # negative: schedule decreases
        if i >= len(indices) - 2:                              # last two steps: ε=0, sharp endpoint
            x = (alpha(sigmas[i + 1]) * x0_hat + beta(sigmas[i + 1]) * x_T_s
                 + (gamma(sigmas[i + 1]) / gamma(sigmas[i]))
                 * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s))
        else:                                                  # Euler-SDE (any ε ≥ 0)
            drift, diffusion = to_d_stoch(
                x, x0_hat, x_T_s,
                alpha(sigmas[i]), alpha_d(sigmas[i]),
                beta(sigmas[i]), beta_d(sigmas[i]),
                gamma(sigmas[i]), gamma_d(sigmas[i]), epsilon(sigmas[i]))
            x = x + drift * dt + th.randn_like(x) * (dt.abs() ** 0.5) * diffusion
        path.append(x.detach().cpu())
    return x, path, x0_est
```

## DBIM-codebase coefficient mapping

For code paths using `get_abc(t)`, the coefficient names are reversed relative to the math
notation: code `a_t` multiplies `x_T` and is formula `β_t`; code `b_t` multiplies `x_0` and is
formula `α_t`; code `c_t` is formula `γ_t`. With the VP schedule
`alpha_fn(t)=exp(-0.5 beta_min t - 0.25 beta_d t^2)`,
`rho_fn(t)=sqrt(exp(beta_min t + 0.5 beta_d t^2)-1)`,
`f(t)=-0.5(beta_min+beta_d t)`, `g2(t)=beta_min+beta_d t`, the analytic derivatives are:

```python
alpha_dot = alpha * f
alpha_bar_dot = alpha_dot / alpha_T
rho_dot = 0.5 * (rho ** 2 + 1.0) * g2 / rho
rho_bar_dot = -rho * rho_dot / rho_bar

a_dot = (alpha_bar_dot * rho ** 2 + alpha_bar * 2 * rho * rho_dot) / rho_T ** 2
b_dot = (alpha_dot * rho_bar ** 2 - alpha * 2 * rho * rho_dot) / rho_T ** 2
c_dot = (alpha_dot * rho_bar * rho + alpha * rho_bar_dot * rho
         + alpha * rho_bar * rho_dot) / rho_T
```

The sampler then uses
`eps = eta * (c * c_dot - (b_dot / b) * c**2)`, residual
`z_hat = (x - b*x0_hat - a*x_T)/c`, drift
`b_dot*x0_hat + a_dot*x_T + (c_dot + eps/c)*z_hat`, and the deterministic endpoint update
`b_next*x0_hat + a_next*x_T + (c_next/c)*(x - b*x0_hat - a*x_T)`.

## Training

One denoiser, L2 regression `∫ E[‖x̂_0(t, x_t, x_T) − x_0‖²] dt` with `x_t` from the kernel.
EDM-style preconditioning `D_θ = c_skip x_t + c_out F_θ(c_in x_t, c_noise)` with
`c_in = 1/√(α_t² σ_0² + β_t² σ_T² + 2 α_t β_t σ_{0T} + γ_t²)`,
`c_skip = (α_t σ_0² + β_t σ_{0T}) c_in²`,
`c_out = √(β_t² σ_0² σ_T² − β_t² σ_{0T}² + γ_t² σ_0²) c_in`, `λ = 1/c_out²`, `c_noise = ¼ log t`.
