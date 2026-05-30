# Neural Ordinary Differential Equations

## The problem

Deep models (residual networks, RNN decoders, normalizing flows) build complicated maps by composing many small transformations of a hidden state:

  h_{t+1} = h_t + f(h_t, θ_t).

This residual update is exactly one step of forward Euler for an ODE. It comes with three costs: backprop must **store every intermediate activation** (memory O(L) in depth), depth is a **fixed, discrete** hyperparameter applied uniformly to every input, and Euler is the crudest possible integrator (fixed step size 1).

## The key idea

Take the continuous-depth limit. Parameterize the *derivative* of the hidden state with a neural network and let a **black-box adaptive ODE solver** produce the output:

  dh(t)/dt = f(h(t), t, θ),  h(t₁) = ODESolve(h(t₀), f, t₀, t₁, θ).

The solver chooses how many times to evaluate f (the "implicit depth" L̃), monitors truncation error against a tolerance, and adapts per input — you can even lower the tolerance at test time to trade accuracy for speed. Parameters are tied across all of continuous depth (one f).

The one hard part is gradients: differentiating through the solver's internal steps would store everything (killing the memory win) and inject the solver's internal error. Instead use the **adjoint sensitivity method** — solve a second, augmented ODE **backwards in time**, recomputing the state as you go, so training costs **O(1) memory** in depth and works for *any* solver.

## The adjoint method (final equations)

Loss L(z(t₁)) with z(t₁) = ODESolve(z(t₀), f, t₀, t₁, θ). Define the adjoint a(t) = ∂L/∂z(t). Then:

- **State adjoint** (instantaneous chain rule), integrated backward from a(t₁) = ∂L/∂z(t₁):

  da(t)/dt = −a(t)ᵀ ∂f(z(t),t,θ)/∂z

- **Parameter gradient** (augment θ as a state with dθ/dt = 0, terminal a_θ(t₁) = 0):

  dL/dθ = −∫_{t₁}^{t₀} a(t)ᵀ ∂f/∂θ dt

- **Time-endpoint gradients**:

  dL/dt₁ = a(t₁)ᵀ f(z(t₁),t₁,θ),  dL/dt₀ = −∫_{t₁}^{t₀} a(t)ᵀ ∂f/∂t dt

All of these run in **one** backward ODE solve over the augmented state [z, a, ∂L/∂θ (, a_t)]. The products a(t)ᵀ∂f/∂z, a(t)ᵀ∂f/∂θ, a(t)ᵀ∂f/∂t are **vector-Jacobian products**, computed by one reverse-mode autodiff pass through f (cost ≈ one evaluation of f) — never forming the D×D Jacobian. The state z(t) needed along the way is **recomputed backward** from z(t₁) instead of stored. If the loss touches multiple times t₁…t_N, break the backward solve into intervals and add ∂L/∂z(t_i) to the adjoint at each.

## Continuous Normalizing Flows (instantaneous change of variables)

For a continuous transformation dz/dt = f(z(t),t), the log-density obeys

  ∂ log p(z(t))/∂t = −tr(∂f/∂z).

The finite change-of-variables formula needs the **log-determinant** of the Jacobian (O(D³)), which forces standard flows into triangular-Jacobian / single-unit-layer designs and partitioning of dimensions. The continuous version needs only the **trace**, which is **linear**: tr(Σ_n J_n) = Σ_n tr(J_n). A wide flow dz/dt = Σ_{n=1}^M f_n(z) then costs O(M) in hidden units instead of O(M³), so a CNF can be **wide instead of deep**. f need not be bijective — Picard uniqueness makes the flow map invertible automatically, so the partition/ordering machinery is dropped. The planar instance dz/dt = u·h(wᵀz+b) has a rank-1 Jacobian, so the trace is the inner product −uᵀ∂h/∂z. Integrate [z, log p] jointly; reverse the flow to sample.

## Reference algorithm (reverse-mode through an ODE solve)

```
Input: θ, t₀, t₁, final state z(t₁), loss gradient ∂L/∂z(t₁)
s₀ = [ z(t₁), ∂L/∂z(t₁), 0_{|θ|} ]                 # augmented initial state at t₁
def aug_dynamics([z, a, ·], t, θ):
    return [ f(z,t,θ), −aᵀ ∂f/∂z, −aᵀ ∂f/∂θ ]      # all three via ONE vjp through f
[ z(t₀), ∂L/∂z(t₀), ∂L/∂θ ] = ODESolve(s₀, aug_dynamics, t₁, t₀, θ)   # solve BACKWARD
return ∂L/∂z(t₀), ∂L/∂θ
```

## Working code (PyTorch)

```python
import torch
import torch.nn as nn


# ---------- black-box ODE solvers ----------
# func has signature func(t, y) -> dy/dt, where y is a tensor (or tuple of tensors).

def rk4_step(func, t, dt, y):
    # classic 4th-order Runge-Kutta step (one of 120+ years of solvers,
    # vastly better than the Euler step a ResNet block implicitly takes)
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt / 2 * k1)
    k3 = func(t + dt / 2, y + dt / 2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint(func, y0, t, method="rk4", step=None):
    # Integrate dy/dt = func(t, y) from t[0] to each value in t.
    # Fixed-grid solver here for transparency; a real solver (dopri5) adapts
    # the step size to a tolerance and chooses how many times to evaluate func.
    solution = [y0]
    y = y0
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        n = 1 if step is None else max(1, int(((t1 - t0).abs() / step).item()))
        dt = (t1 - t0) / n
        for _ in range(n):
            y = rk4_step(func, t0, dt, y)
            t0 = t0 + dt
        solution.append(y)
    return torch.stack(solution)


# ---------- adjoint method: O(1)-memory backprop through the solver ----------
class OdeintAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, y0, t, *params):
        with torch.no_grad():
            ans = odeint(func, y0, t)               # forward solve; store nothing extra
        ctx.func = func
        ctx.save_for_backward(t, ans, *params)
        return ans

    @staticmethod
    def backward(ctx, grad_out):                    # grad_out = ∂L/∂z at each output time
        func = ctx.func
        t, ans, *params = ctx.saved_tensors
        params = tuple(params)

        # augmented backward dynamics:
        #   d z /dt = f
        #   d a /dt = -aᵀ ∂f/∂z         (the adjoint ODE)
        #   d(∂L/∂θ)/dt = -aᵀ ∂f/∂θ     (parameter-gradient integrand)
        # all three are vector-Jacobian products from ONE autograd.grad on f.
        def aug_dynamics(t, aug):
            z, a = aug[0], aug[1]
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                t_ = t.detach().requires_grad_(True)
                f_eval = func(t_, z)
                vjp_z, *vjp_params = torch.autograd.grad(
                    f_eval, (z,) + params, -a,        # the minus sign is the −aᵀ in da/dt
                    allow_unused=True, retain_graph=True,
                )
            vjp_z = torch.zeros_like(z) if vjp_z is None else vjp_z
            vjp_params = [torch.zeros_like(p) if g is None else g
                          for p, g in zip(params, vjp_params)]
            return (f_eval, vjp_z, *vjp_params)

        # initialize the adjoint at the final time
        adj_z = grad_out[-1]
        adj_params = [torch.zeros_like(p) for p in params]
        T = len(t)
        # walk backward over output intervals, injecting ∂L/∂z(t_i) at each observation
        for i in range(T - 1, 0, -1):
            z_i = ans[i]
            aug0 = (z_i, adj_z, *adj_params)
            aug = odeint(aug_dynamics, aug0, torch.stack([t[i], t[i - 1]]))
            _, adj_z, *adj_params = [a[-1] for a in aug]
            adj_z = adj_z + grad_out[i - 1]           # add gradient from this output time
        return (None, adj_z, None, *adj_params)


def odeint_adjoint(func, y0, t):
    params = tuple(func.parameters())
    return OdeintAdjoint.apply(func, y0, t, *params)


# ---------- a continuous-depth "residual block" ----------
class ODEF(nn.Module):
    # parameterizes dh/dt = f(h, t); time is fed in as an extra input so f can vary with t
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),   # Lipschitz nonlinearity ⇒ Picard uniqueness
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, h):
        tt = torch.ones_like(h[..., :1]) * t
        return self.net(torch.cat([tt, h], dim=-1))


class ODEBlock(nn.Module):
    # drop-in replacement for a stack of residual blocks
    def __init__(self, odef):
        super().__init__()
        self.odef = odef

    def forward(self, h0, integration_time=None, use_adjoint=True):
        if integration_time is None:
            integration_time = torch.tensor([0.0, 1.0]).to(h0)
        solve = odeint_adjoint if use_adjoint else odeint
        return solve(self.odef, h0, integration_time)[-1]


# ---------- continuous normalizing flow (planar, exact trace) ----------
class PlanarCNF(nn.Module):
    # dz/dt = u h(wᵀz + b);  d log p/dt = -tr(∂f/∂z) = -uᵀ ∂h/∂z   (rank-1 ⇒ inner product)
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.1)
        self.w = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, t, state):
        z, _ = state                                 # state = (z, log p)
        with torch.enable_grad():
            z = z.requires_grad_(True)
            a = z @ self.w + self.b                  # wᵀz + b
            h = torch.tanh(a)
            f = h[..., None] * self.u                 # u h(wᵀz+b)
            # ∂h/∂z = (1 - tanh²(a)) w ; tr(∂f/∂z) = uᵀ ∂h/∂z
            dh = (1 - h ** 2)[..., None] * self.w
            trace = (self.u * dh).sum(-1)
        return (f, -trace)                            # (dz/dt, d log p/dt)


def cnf_logprob(cnf, z1, base_logprob, t0=1.0, t1=0.0):
    # run the flow from data (t0) back to base (t1), accumulating Δ log p
    logp0 = torch.zeros(z1.shape[:-1], device=z1.device)
    t = torch.tensor([t0, t1]).to(z1)
    z, dlogp = odeint(cnf, (z1, logp0), t)
    z_base, delta_logp = z[-1], dlogp[-1]
    return base_logprob(z_base) + delta_logp          # log q(z1) by change of variables
```

The `ODEBlock` replaces a residual stack: forward is one `odeint` from t=0 to t=1; backward is the adjoint solve in `OdeintAdjoint`, costing constant memory in depth. The `PlanarCNF` carries the density along the trajectory through the trace term, trained by maximum likelihood and reversible at the same cost as the forward pass.
