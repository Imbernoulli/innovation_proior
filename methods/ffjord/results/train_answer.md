The problem is to build a generative model that gives exact log-likelihoods, samples in one pass, and uses a free-form neural network for its transformation. Ordinary flow models satisfy the first two properties only by constraining the invertible map so that the Jacobian log-determinant is cheap to compute. Planar and Sylvester flows restrict the functional form so that a determinant identity applies, but each transformation has little capacity and no convenient inverse. Autoregressive flows make the Jacobian triangular, which permits exact likelihood evaluation, yet sampling requires D sequential inversions and is therefore slow. Coupling layers such as those in Real NVP and Glow split the dimensions and transform one block conditioned on another, giving cheap forward, inverse, and determinant computations, but the architectural restriction limits expressiveness and often demands many stacked layers. The shared cost of tractability is a hand-engineered transformation.

Continuous normalizing flows offer a different path. Instead of a discrete composition of maps, the transformation is defined by the solution of an ordinary differential equation. The instantaneous change-of-variables formula shows that, in continuous time, the log-density evolves according to the trace of the Jacobian of the dynamics rather than its determinant. The trace is linear and structure-agnostic, so the dynamics network need not be triangular, partitioned, or rank-one. However, computing the trace exactly still costs O(D^2), because each diagonal entry is a separate derivative. This remaining cost is enough to discourage the use of large, free-form networks.

The method that removes this last obstacle is FFJORD, which stands for Free-form Jacobian of Reversible Dynamics. FFJORD estimates the trace with Hutchinson's stochastic trace estimator. For any matrix A and a random vector epsilon with zero mean and identity covariance, the expectation of epsilon^T A epsilon equals the trace of A. In practice this means a single reverse-mode vector-Jacobian product, epsilon^T times the Jacobian of the dynamics, is dotted with epsilon to obtain an unbiased O(D) estimate of the trace. Because the estimate is unbiased, the expectation can be pulled outside the time integral, so one fixed epsilon per ODE solve yields an unbiased estimate of the whole log-likelihood. The random right-hand side is held constant within a solve to avoid confusing the adaptive stepper. This drops the per-step cost from O(D^2) to O(D), leaving no remaining reason to constrain the neural network architecture.

In FFJORD, the generative process begins by sampling a base variable z0 from a simple density such as a standard Gaussian. The sample is then propagated forward through time by the neural ODE to obtain a data-space sample x at the terminal time. The probability is computed by running the augmented ODE backward from the data point with an initial log-density accumulator of zero. The accumulator ends at the difference between the base log-density at z0 and the data log-density, so the likelihood is obtained by adding the base log-density and subtracting the accumulator. Training maximizes this estimated log-likelihood, and gradients are computed through the ODE solver with the adjoint method, which uses O(1) memory independent of the solver's internal steps.

A few design details keep the method stable and efficient. The dynamics network f(z, t) concatenates the scalar time t onto the input of every layer, which is simpler and more integrable than hypernetwork-style conditioning. Smooth Lipschitz activations such as softplus or tanh ensure existence and uniqueness of ODE solutions and keep the dynamics non-stiff. For small two-dimensional problems the exact trace can be used at test time for deterministic evaluation, while the stochastic estimate is used during training. A variance-reduction trick is available when the network has a hidden bottleneck: by the cyclic property of trace, the trace of the D-by-D Jacobian equals the trace of a smaller H-by-H product of Jacobians, so the Hutchinson probe can be sampled in the lower-dimensional hidden space instead of the data space. For residual vector fields, the trace is adjusted by subtracting the data dimension so that the log-density update remains correct.

The following code implements the core pieces: a time-conditioned dynamics network, exact and stochastic divergence functions, the augmented ODE right-hand side, and a continuous normalizing flow wrapper that can run forward for sampling or backward for likelihood evaluation.

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
```
