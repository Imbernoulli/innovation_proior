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
- **Likelihood of data `x`.** Integrate the augmented state `[z, Δlogp]` backward from `t₁` to `t₀` with init `[x, 0]`; the accumulator becomes `Δlogp = log p_{z0}(z₀) − log p(x)`, so `log p̂(x) = log p_{z0}(z₀) − Δlogp`.
- **Sampling.** Draw `z₀ ∼ p_{z0}`, integrate forward to `x = z(t₁)` — one pass.
- **Training.** Maximize the estimated log-likelihood; backprop through the solver via the **adjoint method** (`O(1)` memory ⇒ very large batches). Train with the estimator; report test likelihood with the exact trace where feasible.
- **Bottleneck trick (variance reduction).** Variance of Hutchinson `∼ ‖A‖_F²`. If `f = g∘h` with hidden width `H<D`, the cyclic property gives `Tr(∂f/∂z) = Tr((∂h/∂z)(∂g/∂h))` (an `H×H` matrix), estimated by `E[εᵀ(∂h/∂z)(∂g/∂h)ε]`, `ε ∈ R^H` — smaller matrix norm, lower estimator variance.
- **Architecture.** `f(z,t)` concatenates `t` onto every layer's input; smooth Lipschitz activations (softplus/tanh) for ODE existence/uniqueness and non-stiffness.

**Implementation defaults:** Dormand-Prince `dopri5` adaptive solver, `atol = rtol = 1e-5` (tabular `atol 1e-8`, `rtol 1e-6`); Adam, lr `1e-3` decayed to `1e-4`; batches up to `10,000` (tabular) / `900` (images); softplus activations. VAE variant: encoder emits a low-rank weight update plus input-dependent bias, `layer(h;x)=σ((W+Û(x)V̂(x)ᵀ)h + b + b̂(x))`.

## Code

```python
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from . import diffeq_layers
from .squeeze import squeeze, unsqueeze


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": nn.Identity(),
}


class ODEnet(nn.Module):
    """Dynamics f(y, t): time-conditioned layers with smooth activations."""
    def __init__(self, hidden_dims, input_shape, strides=None, conv=False,
                 layer_type="concat", nonlinearity="softplus", num_squeeze=0):
        super().__init__()
        self.num_squeeze = num_squeeze

        if conv:
            assert strides is not None and len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "concat_v2": diffeq_layers.ConcatConv2d_v2,
                "concatsquash": diffeq_layers.ConcatSquashConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "concat_v2": diffeq_layers.ConcatLinear_v2,
                "concatsquash": diffeq_layers.ConcatSquashLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        layers, activation_fns = [], []
        hidden_shape = list(input_shape)
        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError("Unsupported stride: {}".format(stride))

            layers.append(base_layer(hidden_shape[0], dim_out, **layer_kwargs))
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(hidden_shape)
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


def divergence_bf(dx, y, **unused_kwargs):
    # Exact trace: sum diagonal Jacobian entries for small states.
    sum_diag = 0.
    for i in range(y.shape[1]):
        grad_i = torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0]
        sum_diag += grad_i.contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    # Hutchinson trace estimate: e^T (df/dy) e.
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    return (e_dzdx * e).view(y.shape[0], -1).sum(dim=1)


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class ODEfunc(nn.Module):
    def __init__(self, diffeq, divergence_fn="approximate",
                 residual=False, rademacher=False):
        super().__init__()
        assert divergence_fn in ("brute_force", "approximate")
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher
        self.divergence_fn = divergence_bf if divergence_fn == "brute_force" else divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        self._num_evals += 1
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        if self._e is None:
            self._e = sample_rademacher_like(y) if self.rademacher else sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)

        if self.residual:
            dy = dy - y
            dim = torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32).to(divergence)
            divergence -= torch.ones_like(divergence) * dim

        zeros = [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        return tuple([dy, -divergence] + zeros)


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False,
                 solver="dopri5", atol=1e-5, rtol=1e-5):
        super().__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))
        self.odefunc = odefunc
        self.solver, self.atol, self.rtol = solver, atol, rtol
        self.test_solver, self.test_atol, self.test_rtol = solver, atol, rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        _logpz = torch.zeros(z.shape[0], 1).to(z) if logpz is None else logpz
        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time ** 2]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        self.odefunc.before_odeint()
        if self.training:
            state_t = odeint(self.odefunc, (z, _logpz), integration_times.to(z),
                             atol=self.atol, rtol=self.rtol,
                             method=self.solver, options=self.solver_options)
        else:
            state_t = odeint(self.odefunc, (z, _logpz), integration_times.to(z),
                             atol=self.test_atol, rtol=self.test_rtol,
                             method=self.test_solver)

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
        z_t, logpz_t = state_t[:2]
        return (z_t, logpz_t) if logpz is not None else z_t


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


# likelihood: z0, dlogp = cnf(x, logpz=zeros, reverse=True); logp = base_logp(z0) - dlogp
# sample:     x = cnf(z0)
```
