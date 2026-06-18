# DPM-Solver

## Problem

Sampling a diffusion probabilistic model means solving the probability-flow ODE from a high-noise time `T`
to a small end time `epsilon`, calling a trained noise-prediction network `epsilon_theta(x, t)` at each
evaluation. Generic ODE solvers and ancestral samplers spend tens to thousands of network evaluations. The
goal is a training-free solver that uses the existing model and reaches good quality in roughly 10 to 20
network evaluations.

## Core derivation

The diffusion ODE is

  dx_t/dt = f(t)x_t + (g^2(t)/(2 sigma_t)) epsilon_theta(x_t,t),
  f(t)=d log alpha_t/dt,
  g^2(t)=d sigma_t^2/dt - 2(d log alpha_t/dt)sigma_t^2.

It is semi-linear: the `f(t)x_t` part is exactly solvable, while the neural network part is not. Variation of
constants gives, for `t < s`,

  x_t = (alpha_t/alpha_s)x_s + int_s^t (alpha_t/alpha_tau)(g^2(tau)/(2 sigma_tau))
        epsilon_theta(x_tau,tau) d tau.

Define the half-log-SNR

  lambda_t = log(alpha_t/sigma_t).

Since the SNR decreases with forward time, `lambda_t` is strictly decreasing in `t` and is invertible. Also

  g^2(t) = -2 sigma_t^2 d lambda_t/dt.

Substituting this identity and changing variables from `tau` to `lambda` yields the exact solution

  x_t = (alpha_t/alpha_s)x_s - alpha_t int_{lambda_s}^{lambda_t}
        e^{-lambda} eps_hat_theta(x_hat_lambda,lambda) d lambda.

All schedule-dependent coefficients have collapsed into the known exponential weight and endpoint factors.
The solver only approximates the network inside this weighted integral.

Taylor-expand `eps_hat_theta` in `lambda` at the left endpoint. With

  h = lambda_t - lambda_s > 0,
  phi_k(z) = int_0^1 exp((1-delta)z) delta^{k-1}/(k-1)! d delta,

the exact step expansion is

  x_t = (alpha_t/alpha_s)x_s
        - sigma_t sum_{j=0}^n h^{j+1} phi_{j+1}(h)
          eps_hat_theta^{(j)}(x_hat_{lambda_s},lambda_s)
        + O(h^{n+2}),

where `phi_1(h)=(e^h-1)/h`, `phi_2(h)=(e^h-h-1)/h^2`, and
`phi_3(h)=(e^h-h^2/2-h-1)/h^3`.

## Single-step updates

Let `s` be the current, noisier time and `t` the next, cleaner time.

**Order 1.**

  x_t = (alpha_t/alpha_s)x_s - sigma_t(e^h-1)epsilon_theta(x_s,s).

This is exactly DDIM after substituting `sigma/alpha = exp(-lambda)`.

**Order 2.** For `r1 in (0,1)`, set `s1 = t_lambda(lambda_s + r1 h)` and

  u = (alpha_s1/alpha_s)x_s - sigma_s1(e^{r1 h}-1)epsilon_theta(x_s,s).

The recommended single-step branch is

  x_t = (alpha_t/alpha_s)x_s
        - sigma_t(e^h-1)epsilon_theta(x_s,s)
        - (sigma_t/(2r1))(e^h-1)(epsilon_theta(u,s1)-epsilon_theta(x_s,s)).

For the paper default `r1=1/2`, this is equivalently

  x_t = (alpha_t/alpha_s)x_s - sigma_t(e^h-1)epsilon_theta(u,s1).

**Order 3.** With `r1=1/3`, `r2=2/3`,

  s1 = t_lambda(lambda_s + r1 h),   s2 = t_lambda(lambda_s + r2 h),
  u1 = (alpha_s1/alpha_s)x_s - sigma_s1(e^{r1 h}-1)epsilon_theta(x_s,s),
  D1 = epsilon_theta(u1,s1) - epsilon_theta(x_s,s),
  u2 = (alpha_s2/alpha_s)x_s - sigma_s2(e^{r2 h}-1)epsilon_theta(x_s,s)
       - (sigma_s2 r2/r1)((e^{r2 h}-1)/(r2 h)-1)D1,
  D2 = epsilon_theta(u2,s2) - epsilon_theta(x_s,s),
  x_t = (alpha_t/alpha_s)x_s - sigma_t(e^h-1)epsilon_theta(x_s,s)
        - (sigma_t/r2)((e^h-1)/h - 1)D2.

Under the paper's smoothness and Lipschitz assumptions, the global error is `O(h_max^k)` for orders
`k=1,2,3`.

## Implementation

The canonical implementation is `LuChengTHU/dpm-solver/dpm_solver_pytorch.py`. The original paper branch is
`algorithm_type="dpmsolver"`; the later `dpmsolver++` branch uses data prediction, `sigma_t/sigma_s`, and
`expm1(-h)`, so it is not the branch below.

```python
import torch


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule, algorithm_type="dpmsolver"):
        assert algorithm_type == "dpmsolver"
        self.model_fn = model_fn              # noise prediction epsilon_theta(x, t)
        self.noise_schedule = noise_schedule  # marginal_log_mean_coeff/std/lambda/inverse_lambda

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            lambdas = torch.linspace(lambda_T.item(), lambda_0.item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(lambdas)
        if skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        if skip_type == "time_quadratic":
            return torch.linspace(t_T ** 0.5, t_0 ** 0.5, N + 1).pow(2).to(device)
        raise ValueError("skip_type must be logSNR, time_uniform, or time_quadratic")

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
        elif order == 2:
            K = steps // 2 if steps % 2 == 0 else steps // 2 + 1
            orders = [2] * K if steps % 2 == 0 else [2] * (K - 1) + [1]
        elif order == 1:
            K, orders = steps, [1] * steps
        else:
            raise ValueError("order must be 1, 2, or 3")

        if skip_type == "logSNR":
            timesteps = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            base = self.get_time_steps(skip_type, t_T, t_0, steps, device)
            timesteps = base[torch.cumsum(torch.tensor([0] + orders), 0).to(device)]
        return timesteps, orders

    def dpm_solver_first_update(self, x, s, t, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)
        if model_s is None:
            model_s = self.model_fn(x, s)
        phi_1 = torch.expm1(h)
        return torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        s1 = ns.inverse_lambda(lambda_s + r1 * h)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)
        if model_s is None:
            model_s = self.model_fn(x, s)
        phi_11, phi_1 = torch.expm1(r1 * h), torch.expm1(h)
        x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model_fn(x_s1, s1)
        return (torch.exp(log_alpha_t - log_alpha_s) * x
                - sigma_t * phi_1 * model_s
                - (0.5 / r1) * sigma_t * phi_1 * (model_s1 - model_s))

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        s1 = ns.inverse_lambda(lambda_s + r1 * h)
        s2 = ns.inverse_lambda(lambda_s + r2 * h)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_s2 = ns.marginal_log_mean_coeff(s2)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        if model_s is None:
            model_s = self.model_fn(x, s)
        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
        phi_2 = phi_1 / h - 1.0

        x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model_fn(x_s1, s1)
        x_s2 = (torch.exp(log_alpha_s2 - log_alpha_s) * x
                - sigma_s2 * phi_12 * model_s
                - (r2 / r1) * sigma_s2 * phi_22 * (model_s1 - model_s))
        model_s2 = self.model_fn(x_s2, s2)
        return (torch.exp(log_alpha_t - log_alpha_s) * x
                - sigma_t * phi_1 * model_s
                - (1.0 / r2) * sigma_t * phi_2 * (model_s2 - model_s))
```

Use `torch.expm1` for every `e^h-1` term. For classifier guidance, the noise prediction function is replaced by

  epsilon_theta(x,t,y) = epsilon_theta(x,t) - scale * sigma_t * grad_x log p_t(y|x).

For discrete-time models, the implementation first wraps the model so continuous `t` is mapped to the time label
expected by the trained network.
