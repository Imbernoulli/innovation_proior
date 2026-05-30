# DPM-Solver

## Problem

Sampling from a diffusion probabilistic model (DPM) means numerically solving its probability-flow ODE
from t=T to t=0, calling a large network ε_θ once per evaluation. Generic ODE solvers need ~50–1000
evaluations; the goal is a training-free solver that reaches comparable quality in around 10. DPM-Solver
is a dedicated solver that exploits the structure of the diffusion ODE to do exactly that.

## Key idea

The diffusion ODE

  dx_t/dt = f(t) x_t + (g²(t) / 2σ_t) ε_θ(x_t, t),   f(t)=d log α_t/dt,   g²(t)=dσ_t²/dt − 2(d log α_t/dt)σ_t²

is *semi-linear*: an exactly-solvable linear term plus a nonlinear network term. Generic solvers discretize
both and waste accuracy on the linear part (whose exact solution is exponential, so its discretization error
can blow up). DPM-Solver solves the linear part exactly via variation of constants and approximates only the
network term.

1. **Variation of constants** gives the exact solution from time s to t:
   x_t = (α_t/α_s) x_s + ∫_s^t (α_t/α_τ)(g²(τ)/2σ_τ) ε_θ(x_τ,τ) dτ.

2. **Change of variable to the half-log-SNR** λ_t := log(α_t/σ_t) (strictly decreasing in t). Using
   g²(t) = −2σ_t² dλ_t/dt and σ_τ/α_τ = e^{−λ}, the solution collapses to an exponentially weighted integral
   of the network alone:

   **x_t = (α_t/α_s) x_s − α_t ∫_{λ_s}^{λ_t} e^{−λ} ε̂_θ(x̂_λ, λ) dλ.**

   All noise-schedule dependence becomes the analytic factor e^{−λ}; only the integral of ε̂_θ is approximated.

3. **Taylor-expand** ε̂_θ in λ around the left endpoint and integrate term by term. The scalar integrals are
   the exponential-integrator φ-functions:
   φ_1(h)=(e^h−1)/h, φ_2(h)=(e^h−h−1)/h², φ_3(h)=(e^h−h²/2−h−1)/h³, with h = λ_t − λ_s. Keeping k terms gives
   a k-th order method, DPM-Solver-k (k network evaluations per step). DPM-Solver-k is provably order k for
   k=1,2,3: error at t=0 is O(h_max^k).

**DPM-Solver-1 = DDIM.** With h_i = λ_{t_i} − λ_{t_{i-1}}:
  x̃_{t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − σ_{t_i}(e^{h_i} − 1) ε_θ(x̃_{t_{i-1}}, t_{i-1}).
Substituting σ_t/α_t = e^{−λ_t} into the DDIM update recovers this exactly, explaining why DDIM (alone among
first-order ODE samplers) makes full use of the semi-linearity.

## Algorithm

Steps are placed uniformly in λ (not in t): λ_{t_i} = λ_T + (i/M)(λ_0 − λ_T), inverting λ to recover t for the
network input (closed form for linear/cosine VP schedules). For a fixed budget K, take as many DPM-Solver-3
steps as fit and finish with one order-2 or order-1 step. With h = λ_t − λ_s:

- **Order 1:** x_t = (α_t/α_s)x_s − σ_t(e^h−1)ε_θ(x_s,s).
- **Order 2** (r_1=1/2): intermediate node s_1 = t_λ(λ_s + r_1 h);
  u = (α_{s_1}/α_s)x_s − σ_{s_1}(e^{r_1 h}−1)ε_θ(x_s,s);
  x_t = (α_t/α_s)x_s − σ_t(e^h−1)ε_θ(x_s,s) − (σ_t/(2r_1))(e^h−1)(ε_θ(u,s_1) − ε_θ(x_s,s)).
- **Order 3** (r_1=1/3, r_2=2/3): two nodes; D_1 = ε_θ(u_1,s_1)−ε_θ(x_s,s),
  u_2 = (α_{s_2}/α_s)x_s − σ_{s_2}(e^{r_2 h}−1)ε_θ(x_s,s) − (σ_{s_2}r_2/r_1)((e^{r_2 h}−1)/(r_2 h) − 1)D_1,
  D_2 = ε_θ(u_2,s_2)−ε_θ(x_s,s),
  x_t = (α_t/α_s)x_s − σ_t(e^h−1)ε_θ(x_s,s) − (σ_t/r_2)((e^h−1)/h − 1)D_2.

Use expm1(h) for e^h−1. Discrete-time models are wrapped to continuous time. Classifier guidance: replace
ε_θ(x,t) by ε_θ(x,t) − s·σ_t ∇_x log p_t(y|x).

## Code

```python
import torch

class NoiseScheduleVP:
    """alpha_t, sigma_t, lambda_t = log(alpha_t) - log(sigma_t), and the inverse t_lambda
    for a variance-preserving linear schedule."""
    def __init__(self, beta_0=0.1, beta_1=20.):
        self.beta_0, self.beta_1 = beta_0, beta_1

    def marginal_log_mean_coeff(self, t):   # log(alpha_t)
        return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):              # sigma_t = sqrt(1 - alpha_t^2)
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):           # half log-SNR (the integration variable)
        log_mean = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean))
        return log_mean - log_std

    def inverse_lambda(self, lamb):         # t as a function of lambda (closed form)
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)


class DPM_Solver:
    def __init__(self, eps_theta, noise_schedule):
        self.model = eps_theta          # eps_theta(x, t) -> predicted noise
        self.ns = noise_schedule

    def get_time_steps(self, t_T, t_0, N, device):
        lam_T = self.ns.marginal_lambda(torch.tensor(t_T).to(device))
        lam_0 = self.ns.marginal_lambda(torch.tensor(t_0).to(device))
        lam = torch.linspace(lam_T.item(), lam_0.item(), N + 1).to(device)   # uniform in lambda
        return self.ns.inverse_lambda(lam)

    def first_update(self, x, s, t, model_s=None):                            # DPM-Solver-1 (= DDIM)
        ns = self.ns
        lam_s, lam_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lam_t - lam_s
        log_a_s, log_a_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)
        if model_s is None:
            model_s = self.model(x, s)
        return torch.exp(log_a_t - log_a_s) * x - sigma_t * phi_1 * model_s

    def second_update(self, x, s, t, r1=0.5, model_s=None):                   # DPM-Solver-2
        ns = self.ns
        lam_s, lam_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lam_t - lam_s
        s1 = ns.inverse_lambda(lam_s + r1 * h)
        log_a_s, log_a_s1, log_a_t = (ns.marginal_log_mean_coeff(s),
                                      ns.marginal_log_mean_coeff(s1),
                                      ns.marginal_log_mean_coeff(t))
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)
        phi_11, phi_1 = torch.expm1(r1 * h), torch.expm1(h)
        if model_s is None:
            model_s = self.model(x, s)
        x_s1 = torch.exp(log_a_s1 - log_a_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model(x_s1, s1)
        return (torch.exp(log_a_t - log_a_s) * x
                - sigma_t * phi_1 * model_s
                - (0.5 / r1) * sigma_t * phi_1 * (model_s1 - model_s))

    def third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None):        # DPM-Solver-3
        ns = self.ns
        lam_s, lam_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lam_t - lam_s
        s1, s2 = ns.inverse_lambda(lam_s + r1 * h), ns.inverse_lambda(lam_s + r2 * h)
        log_a_s, log_a_s1, log_a_s2, log_a_t = (ns.marginal_log_mean_coeff(s),
                                                ns.marginal_log_mean_coeff(s1),
                                                ns.marginal_log_mean_coeff(s2),
                                                ns.marginal_log_mean_coeff(t))
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        phi_11, phi_12, phi_1 = torch.expm1(r1 * h), torch.expm1(r2 * h), torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = phi_1 / h - 1.
        if model_s is None:
            model_s = self.model(x, s)
        x_s1 = torch.exp(log_a_s1 - log_a_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model(x_s1, s1)
        D1 = model_s1 - model_s
        x_s2 = (torch.exp(log_a_s2 - log_a_s) * x
                - sigma_s2 * phi_12 * model_s
                - (r2 / r1) * sigma_s2 * phi_22 * D1)
        model_s2 = self.model(x_s2, s2)
        D2 = model_s2 - model_s
        return (torch.exp(log_a_t - log_a_s) * x
                - sigma_t * phi_1 * model_s
                - (1. / r2) * sigma_t * phi_2 * D2)

    def sample(self, x_T, steps, t_T=1.0, t_0=1e-3, order=3, device='cuda'):
        x = x_T
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:   orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1: orders = [3] * (K - 1) + [1]
            else:                orders = [3] * (K - 1) + [2]
        elif order == 2:
            orders = [2] * (steps // 2) + ([1] if steps % 2 else [])
            K = len(orders)
        else:
            K, orders = steps, [1] * steps
        ts = self.get_time_steps(t_T, t_0, K, device)
        for i, o in enumerate(orders):
            s, t = ts[i], ts[i + 1]
            x = (self.first_update if o == 1 else
                 self.second_update if o == 2 else
                 self.third_update)(x, s, t)
        return x
```
