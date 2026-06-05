Let me lay out what I actually want from a generative model and where the tension is. I want three things at once: exact log-likelihoods, so I can train by maximum likelihood and report held-out likelihood honestly; cheap one-pass sampling; and freedom to use whatever neural network I like for the transformation. Flow models — warp a simple base `p_z` through an invertible `f: R^D → R^D` — give me the first two through the change-of-variables formula. For `z ∼ p_z` and `x = f(z)`,

`log p_x(x) = log p_z(z) − log |det( ∂f/∂z )|`.

The whole game is that determinant. For a general `f` it's `O(D³)`, which is hopeless at scale. Existing flows all get out of that cost by narrowing the map. Planar and Sylvester-style flows restrict the functional form so a determinant identity applies; the planar case is basically a one-layer network with a single hidden unit per step, so each transform has very little capacity and no convenient inverse for data-to-sample use. Autoregressive flows force an ordering, which makes the Jacobian triangular and the determinant just the product of the diagonal, but then sampling needs `D` sequential inversions. NICE, Real NVP, and Glow split the variables and transform one block conditioned on the other, so the inverse and determinant are cheap, but every layer is still a hand-shaped triangular object. The architectural restriction is the price of the tractable determinant. I want to stop paying it. The question is whether there is a formulation where the expensive object is not a determinant at all.

A discrete flow is a finite composition of maps. What if the transformation is the solution of an ODE instead — continuous-time dynamics `∂z(t)/∂t = f(z(t), t; θ)`, with `z(t_0) ∼ p_{z0}` the base sample and `z(t_1) = x` the data? Then I need to know how the log-density evolves along the trajectory. Let me derive it rather than guess. Over an infinitesimal step `Δt`, the map is `z(t+Δt) = z(t) + Δt · f(z(t),t) + O(Δt²)`, so its Jacobian is `I + Δt · ∂f/∂z + O(Δt²)`. The discrete change of variables says the log-density picks up `− log|det(Jacobian)|`. For a matrix close to the identity,
`log det( I + Δt · ∂f/∂z ) = Tr( Δt · ∂f/∂z ) + O(Δt²) = Δt · Tr(∂f/∂z) + O(Δt²)`,
using `log det(I + εM) = ε Tr(M) + O(ε²)`. So
`log p(z(t+Δt)) − log p(z(t)) = − Δt · Tr(∂f/∂z) + O(Δt²)`,
and dividing by `Δt` and letting `Δt → 0`,

`∂ log p(z(t))/∂t = − Tr( ∂f/∂z(t) )`.

This is the instantaneous change of variables. Stare at what just happened: in continuous time the *determinant* collapses to a *trace*. That matters enormously, because a determinant needs structure in the Jacobian to be cheap, while a trace is just a sum of diagonal entries — it's *linear* in the matrix and asks nothing of its structure. Integrating along the trajectory,

`log p(z(t_1)) = log p(z(t_0)) − ∫_{t_0}^{t_1} Tr(∂f/∂z(t)) dt`.

And to go from a data point `x` to its likelihood, I run the dynamics — both `z` and the accumulated log-density change — backward from `t_1` to `t_0`. Stack the two into one augmented state and initialize it at `[z(t_1), Δlogp] = [x, 0]`. If the second component follows `dΔlogp/dt = −Tr(∂f/∂z)`, then after integrating backward it equals

`Δlogp = ∫_{t_1}^{t_0} − Tr(∂f/∂z(t)) dt = log p_{z0}(z_0) − log p(x)`,

so `log p̂(x) = log p_{z0}(z_0) − Δlogp`. (For this IVP to have a unique solution I need `f` and its first derivatives Lipschitz, which I get by using smooth Lipschitz activations like tanh or softplus.) To train I maximize this likelihood; differentiating through a black-box ODE solve is exactly what the adjoint method handles — solve a second ODE backward for the adjoint `a(t) = −∂L/∂z(t)` and accumulate `dL/dθ = −∫_{t_1}^{t_0} (∂L/∂z)ᵀ (∂f/∂θ) dt`, with `O(1)` memory because I never store intermediate activations.

So this continuous formulation already frees `f` somewhat and drops the cost. But how much? Computing `Tr(∂f/∂z)` exactly means getting every diagonal entry `∂f_i/∂z_i`, and each one is a separate derivative of `f` — `D` of them, so `O(D²)` per solver step, roughly `D` evaluations of `f`. Better than `O(D³)`, but still scaling with `D²`, and that's enough to keep people using restricted, low-width `f` (planar CNF widens the single hidden unit to many, but stays cautious). I haven't actually removed the architectural pressure; I've just lowered it. The trace is still the bottleneck.

Let me think about what a trace really costs if I'm willing to estimate it. Reverse-mode autodiff gives me a *vector-Jacobian product* `vᵀ(∂f/∂z)` for about the price of a single evaluation of `f` — I never have to materialize the `D×D` Jacobian, I just push a cotangent vector through. An unbiased trace estimate can be written using exactly that product. Take any `D×D` matrix `A` and a random vector `ε` with `E[ε] = 0` and `Cov(ε) = I`. Then

`E[ εᵀ A ε ] = E[ Σ_{i,j} ε_i A_{ij} ε_j ] = Σ_{i,j} A_{ij} E[ε_i ε_j] = Σ_{i,j} A_{ij} δ_{ij} = Σ_i A_{ii} = Tr(A)`,

because `E[ε_i ε_j] = δ_{ij}` is exactly the covariance condition. So `εᵀ A ε` is an unbiased estimator of `Tr(A)` — Hutchinson's estimator. Apply it to `A = ∂f/∂z`: I form `εᵀ(∂f/∂z)` as one vector-Jacobian product (cost ≈ one `f` eval, `O(DH)`), then take its dot with `ε` (cost `O(D)`). That's an unbiased estimate of `Tr(∂f/∂z)` in `O(D)` — no `D` separate derivatives, no Jacobian matrix, and, crucially, *nothing required of the Jacobian's structure*. The last reason to constrain `f` is gone. `f` can be any Lipschitz neural net.

I have to be a little careful about where the randomness lives, though. The log-density is a *time integral* of the trace, and I'm going to feed the dynamics to an adaptive ODE solver that wants a deterministic right-hand side within a solve. If I resampled `ε` at every function evaluation, the integrand would jitter and the adaptive stepper would chase noise. The fix: sample one `ε` and *fix it for the whole solve*. Does fixing it bias anything? No — by Fubini I can pull the expectation outside the time integral:

`log p(z(t_1)) = log p(z(t_0)) − ∫_{t_0}^{t_1} E_{p(ε)}[ εᵀ (∂f/∂z(t)) ε ] dt = log p(z(t_0)) − E_{p(ε)}[ ∫_{t_0}^{t_1} εᵀ (∂f/∂z(t)) ε \, dt ]`,

so a single sampled `ε` held constant across the solve gives an unbiased estimate of the whole log-density. For `p(ε)` I just need mean zero, identity covariance: a standard Gaussian works, and so does Rademacher (each entry `±1` with probability one-half, which has identity covariance and unit entries). Now the cost of the likelihood is `O((DH + D) L̂)` for `L̂` solver evaluations — versus `O((DH + D²) L̂)` for the exact-trace CNF and `O((DH + D³) L)` for a discrete flow. Free-form Jacobian of reversible dynamics.

One more thing I can squeeze: variance. Hutchinson's estimator is unbiased but noisy, and its variance for `Tr(A)` grows like `‖A‖_F²`, the squared Frobenius norm. If my dynamics net has a hidden bottleneck — write `f = g ∘ h` where `h` maps into a hidden space of width `H < D` — I can shrink the matrix the noise vector probes. The Jacobian factors by the chain rule, `∂f/∂z = (∂g/∂h)(∂h/∂z)`, and the cyclic property of trace lets me reorder:

`Tr(∂f/∂z) = Tr( (∂g/∂h)(∂h/∂z) ) = Tr( (∂h/∂z)(∂g/∂h) )`,

where the first product is `D×D` but the *reordered* one is `H×H`. So I can estimate `Tr` with an `ε ∈ R^H` against the smaller matrix: `E_{p(ε)}[ εᵀ (∂h/∂z)(∂g/∂h) ε ]`, computed as two chained vector-Jacobian products. Smaller matrix, smaller Frobenius norm, lower variance — pick `H` to be the smallest hidden width. I would expect this to matter most when the probe distribution still has avoidable variance; Rademacher already removes some diagonal noise by fixing `ε_i² = 1`, so the extra room may be smaller there than with a Gaussian probe.

A couple of architectural choices follow from making the ODE actually integrable. The dynamics `f` takes both `z(t)` and `t`. I could feed `t` through a hypernetwork that generates weights, but that tends to produce dynamics that are nasty to integrate; the simple, stable choice is to *concatenate* `t` onto the layer input at every layer. And the activations should be smooth — softplus or tanh — both because existence/uniqueness wants Lipschitz `f` and because smooth dynamics stay non-stiff so a general-purpose solver doesn't blow up its evaluation count.

There is one small implementation case I should keep straight. If I use a residual vector field `f(y,t) - y`, the Jacobian trace is `Tr(∂f/∂y) - D`, or more generally `Tr(∂f/∂y) - prod(y.shape[1:])` for tensor states. The code has to subtract that same dimension from the divergence before returning `-divergence`; otherwise the log-density sign is wrong.

Now the code lands on those choices: time-conditioned dynamics, exact and stochastic trace paths, one fixed probe per solve, the optional residual trace correction, and an ODE wrapper that flips the time interval for inversion.

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

To get a likelihood I run `CNF` in reverse from a data batch `x` with `logpz = 0`, recovering `z_0` and the accumulated `Δlogp`, then add the base log-density: `log p̂(x) = log p_{z0}(z_0) − Δlogp`. To sample I draw `z_0 ∼ p_{z0}` and run forward to `x = z(t_1)` — one pass, no sequential inversion.

Tracing the chain back: I wanted exact-likelihood, one-pass-sampling generative modeling with an unrestricted `f`, but the change-of-variables determinant is `O(D³)` and the usual fix is to constrain the architecture. Moving to continuous-time dynamics turned that determinant into a *trace* (the instantaneous change of variables), which is linear and structure-agnostic and brought the cost to `O(D²)` — but exact trace still scales with `D²` and kept `f` cautious. Hutchinson's identity `Tr(A) = E[εᵀAε]`, evaluated with a single reverse-mode vector-Jacobian product and `ε` held fixed across each solve (unbiased by Fubini), drops the trace to an `O(D)` unbiased estimate and removes the last reason to constrain `f`. The cyclic-trace bottleneck trick shrinks the probed matrix to cut variance, the adjoint method backpropagates through the solver in `O(1)` memory, and concatenated-time, smooth-activation dynamics keep the ODE integrable — together, a continuous reversible generative model with a genuinely free-form Jacobian.
