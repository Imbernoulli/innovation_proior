# Neural Ordinary Differential Equations

## The problem

Deep residual networks, recurrent decoders, and normalizing flows build complicated maps by composing many small hidden-state transformations:

  h_{k+1} = h_k + f(h_k, θ_k).

This is exactly a forward-Euler step. The discrete-stack form has three costs: reverse-mode training stores the intermediate activations, so memory grows with depth; depth is a fixed hand-chosen integer applied uniformly to every input; and the numerical integrator is the lowest-order fixed-step method.

## The key idea

Parameterize the derivative of the hidden state and let a black-box ODE solver compute the layer output:

  dh(t)/dt = f(h(t), t, θ),
  h(t₁) = ODESolve(h(t₀), f, t₀, t₁, θ).

The solver chooses the number of function evaluations needed to satisfy a tolerance. The same trained model can be evaluated faster or more accurately by changing that tolerance. The training problem is solved with the adjoint sensitivity method: a second augmented ODE is solved backward in time, so the forward trajectory does not have to be stored.

## Adjoint Equations

Let z(t₁) = ODESolve(z(t₀), f, t₀, t₁, θ), and let the row adjoint be a(t) = ∂L/∂z(t). The continuous chain rule gives

  da(t)/dt = -a(t) ∂f(z(t),t,θ)/∂z.

In column convention this is

  da(t)/dt = -(∂f/∂z)ᵀ a(t).

Augmenting θ as a constant state gives

  dL/dθ = -∫[t₁→t₀] a(t) ∂f/∂θ dt
        =  ∫[t₀→t₁] a(t) ∂f/∂θ dt.

Endpoint gradients require the boundary contribution:

  dL/dt₁ = a(t₁) f(z(t₁), t₁, θ).

The backward time-adjoint is initialized with -dL/dt₁ and evolves by da_t/dt = -a(t) ∂f/∂t; in the simple two-endpoint case with no direct loss on z(t₀), this reduces to dL/dt₀ = -a(t₀) f(z(t₀), t₀, θ).

Implementation-wise, the augmented backward state is `(vjp_t, z, a, *vjp_params)`. A single

```python
torch.autograd.grad(f_eval, (t, z) + params, -a)
```

gives the `-a ∂f/∂t`, `-a ∂f/∂z`, and `-a ∂f/∂θ` integrands at once; the `-a` seed is the minus sign in the adjoint equations. Each requested output interval is solved backward, the reconstructed state is reset to the saved forward value at the previous output time, and any local observation gradient ∂L/∂z(t_i) is added before continuing.

## Continuous Normalizing Flows

For a continuous transformation dz/dt = f(z(t),t), the infinitesimal change-of-variables theorem is

  ∂ log p(z(t))/∂t = -tr(∂f/∂z).

The finite-flow log-determinant is replaced by a trace. For a sum of hidden-unit dynamics, the sign stays negative:

  dz/dt = Σ_n f_n(z),
  d log p/dt = -Σ_n tr(∂f_n/∂z).

For the planar continuous flow f(z)=u h(wᵀz+b),

  ∂f/∂z = u (∂h/∂z)ᵀ,
  d log p/dt = -uᵀ ∂h/∂z.

The flow map is invertible under the usual uniqueness conditions for the ODE, so the dynamics do not need a triangular Jacobian, a dimension partition, or a hand-designed inverse.

## Reference-Faithful PyTorch

```python
import torch
import torch.nn as nn


# ---------- black-box ODE solver (func(t, y) -> dy/dt; y a tensor OR a tuple) ----------
def rk4_step(func, t, dt, y):
    # classic 4th-order Runge-Kutta step (not Euler — that is the whole point)
    def axpy(a, xs, ys):                       # a*xs + ys, elementwise over a tuple/tensor
        if isinstance(ys, tuple):
            return tuple(a * x + yy for x, yy in zip(xs, ys))
        return a * xs + ys
    k1 = func(t, y)
    k2 = func(t + dt / 2, axpy(dt / 2, k1, y))
    k3 = func(t + dt / 2, axpy(dt / 2, k2, y))
    k4 = func(t + dt, axpy(dt, k3, y))
    def combine(yy, a, b, c, d):
        return yy + dt / 6 * (a + 2 * b + 2 * c + d)
    if isinstance(y, tuple):
        return tuple(combine(*parts) for parts in zip(y, k1, k2, k3, k4))
    return combine(y, k1, k2, k3, k4)


def odeint(func, y0, t, step=None):
    # Integrate dy/dt = func(t, y) from t[0] to each value in t, returning the state at
    # every requested time. Fixed-grid here for transparency; a real solver adapts the
    # step to a tolerance and decides how many times to evaluate func per input.
    is_tuple = isinstance(y0, tuple)
    sol = [y0]
    y = y0
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        n = 1 if step is None else max(1, int(((t1 - t0).abs() / step).item()))
        dt = (t1 - t0) / n
        for _ in range(n):
            y = rk4_step(func, t0, dt, y)
            t0 = t0 + dt
        sol.append(y)
    if is_tuple:
        return tuple(torch.stack([s[j] for s in sol]) for j in range(len(y0)))
    return torch.stack(sol)


class ODEF(nn.Module):
    # parameterizes dh/dt = f(h, t); time fed in as an extra input so f can vary with t
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),   # Lipschitz ⇒ Picard uniqueness
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.nfe = 0                                  # count function evaluations

    def forward(self, t, h):
        self.nfe += 1
        tt = torch.ones_like(h[..., :1]) * t
        return self.net(torch.cat([tt, h], dim=-1))


class ODEBlock(nn.Module):
    # drop-in replacement for a stack of residual blocks
    def __init__(self, odef):
        super().__init__()
        self.odef = odef

    def forward(self, h0, integration_time=None):
        if integration_time is None:
            integration_time = torch.tensor([0.0, 1.0], device=h0.device, dtype=h0.dtype)
        return odeint(self.odef, h0, integration_time)[-1]

    @property
    def nfe(self):
        return self.odef.nfe


class PlanarCNF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.1)
        self.w = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, t, state):
        z, _ = state
        with torch.enable_grad():
            z = z.requires_grad_(True)
            a = z @ self.w + self.b
            h = torch.tanh(a)
            dz_dt = h[..., None] * self.u
            dh_dz = (1 - h ** 2)[..., None] * self.w
            trace = (self.u * dh_dz).sum(-1)
        return dz_dt, -trace[..., None]


def cnf_logprob(cnf, x, base_logprob, t0=0.0, t1=1.0):
    # integrate data (at t1) backward to the base (at t0), accumulating Δ log p
    logp_diff_t1 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    t = torch.tensor([t1, t0], device=x.device, dtype=x.dtype)
    z_t, logp_diff_t = odeint(cnf, (x, logp_diff_t1), t)
    z_base = z_t[-1]
    logp_diff = logp_diff_t[-1].view(-1)
    return base_logprob(z_base) - logp_diff   # log p_data = log p_base − ∫ tr(∂f/∂z) dt
```

`ODEBlock` is the continuous-depth replacement for a residual stack. `PlanarCNF` carries the tuple state `(z, logp_diff)` through the same solver. The likelihood sign is the one that survives the boundary check: integrating data backward to the base distribution returns log p_base − log p_data, so the accumulated log-density difference is *subtracted* from the base log-probability, not added.
