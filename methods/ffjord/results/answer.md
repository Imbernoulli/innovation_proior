# FFJORD (Free-form Jacobian of Reversible Dynamics)

## Problem

A change-of-variables (flow) generative model gives exact likelihoods and one-pass sampling, but the change-of-variables formula `log p_x(x) = log p_z(z) − log|det(∂f/∂z)|` needs the Jacobian log-determinant, which is `O(D³)` for a general `f`. Prior flows make it cheap only by constraining the architecture (partitioned/triangular Jacobians, rank-one layers, autoregressive orderings). Goal: keep exact likelihood and one-pass sampling with a completely **free-form** neural-net transformation, at a cost that scales gracefully with `D`.

## Key idea

Define the transformation by an ODE and use the **instantaneous change of variables** (Chen et al. 2018), which replaces the determinant with a **trace**:
`∂ log p(z(t))/∂t = − Tr(∂f/∂z(t))`, so `log p(z(t₁)) = log p(z(t₀)) − ∫_{t₀}^{t₁} Tr(∂f/∂z) dt`.
Trace is linear and needs no Jacobian structure, so `f` can be any (Lipschitz) net. Then estimate the trace cheaply and unbiasedly with **Hutchinson's estimator**:
`Tr(A) = E_{p(ε)}[εᵀ A ε]` for any `ε` with `E[ε]=0`, `Cov(ε)=I`.
A vector-Jacobian product `εᵀ(∂f/∂z)` costs ≈ one evaluation of `f` (reverse-mode autodiff), so `εᵀ(∂f/∂z)ε` is an `O(D)` unbiased estimate of the trace — removing the last architectural constraint. Cost drops from `O((DH+D³)L)` (discrete flow) to `O((DH+D²)L̂)` (exact-trace CNF) to `O((DH+D)L̂)` (FFJORD).

## Final method

- **Log-density.** Sample one `ε` and hold it fixed for the whole ODE solve (deterministic RHS for the adaptive solver; unbiased by pulling `E` outside the time integral via Fubini):
  `log p(z(t₁)) = log p(z(t₀)) − E_{p(ε)}[ ∫_{t₀}^{t₁} εᵀ (∂f/∂z(t)) ε dt ]`. `ε ∼` standard Gaussian or Rademacher (`±1`).
- **Likelihood of data `x`.** Integrate the augmented state `[z, Δlogp]` backward from `t₁` to `t₀` with init `[x, 0]`; then `log p̂(x) = log p_{z0}(z₀) − Δlogp`.
- **Sampling.** Draw `z₀ ∼ p_{z0}`, integrate forward to `x = z(t₁)` — one pass.
- **Training.** Maximize the estimated log-likelihood; backprop through the solver via the **adjoint method** (`O(1)` memory ⇒ very large batches). Train with the estimator; report test likelihood with the exact trace where feasible.
- **Bottleneck trick (variance reduction).** Variance of Hutchinson `∼ ‖A‖_F²`. If `f = g∘h` with hidden width `H<D`, the cyclic property gives `Tr(∂f/∂z) = Tr((∂h/∂z)(∂g/∂h))` (an `H×H` matrix), estimated by `E[εᵀ(∂h/∂z)(∂g/∂h)ε]`, `ε ∈ R^H` — smaller norm, lower variance (helps Gaussian `ε` more than Rademacher).
- **Architecture.** `f(z,t)` concatenates `t` onto every layer's input; smooth Lipschitz activations (softplus/tanh) for ODE existence/uniqueness and non-stiffness.

**Defaults (canonical implementation):** Dormand–Prince `dopri5` adaptive solver, `atol = rtol = 1e-5` (tabular `atol 1e-8`, `rtol 1e-6`); Adam, lr `1e-3` decayed to `1e-4`; batches up to `10,000` (tabular) / `900` (images); softplus activations; NFE grows during training but converges to a value independent of `D` (depends on distribution complexity). VAE variant: encoder emits a low-rank weight update plus input-dependent bias, `layer(h;x)=σ((W+Û(x)V̂(x)ᵀ)h + b + b̂(x))`.

## Code

```python
import torch, torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

def divergence_bf(dx, y):                      # exact Tr(∂f/∂z), O(D²)
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0][:, i]
    return sum_diag

def divergence_approx(f, y, e):                # Hutchinson εᵀ(∂f/∂z)ε, O(D), unbiased
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    return (e_dzdx * e).view(y.shape[0], -1).sum(dim=1)

def sample_rademacher_like(y):
    return torch.randint(0, 2, y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)


class ODEfunc(nn.Module):
    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        super().__init__()
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher
        self.divergence_fn = divergence_approx if divergence_fn == "approximate" else divergence_bf
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        self._num_evals += 1
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]
        if self._e is None:                    # sample ε once, fix for the solve
            self._e = sample_rademacher_like(y) if self.rademacher else sample_gaussian_like(y)
        with torch.set_grad_enabled(True):
            y.requires_grad_(True); t.requires_grad_(True)
            dy = self.diffeq(t, y)
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * float(y.shape[1])
        return (dy, -divergence)


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.odefunc = odefunc
        self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))
        self.solver, self.atol, self.rtol = solver, atol, rtol

    def forward(self, z, logpz=None, reverse=False):
        _logpz = torch.zeros(z.shape[0], 1).to(z) if logpz is None else logpz
        integration_times = torch.tensor([0.0, self.sqrt_end_time ** 2]).to(z)
        if reverse:
            integration_times = integration_times.flip(0)
        self.odefunc.before_odeint()
        z_t, logpz_t = odeint(self.odefunc, (z, _logpz), integration_times,
                              atol=self.atol, rtol=self.rtol, method=self.solver)
        z_t, logpz_t = z_t[-1], logpz_t[-1]
        return (z_t, logpz_t) if logpz is not None else z_t


# log-likelihood:  z0, dlogp = cnf(x, logpz=zeros, reverse=True);  logp = base_logp(z0) - dlogp
# sample:          x = cnf(z0 ~ base)              (one forward pass)
```
