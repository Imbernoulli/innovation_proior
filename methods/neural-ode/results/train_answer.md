Deep models build expressive maps by stacking many small transformations of a hidden state, and the workhorse primitive is the residual block $h_{t+1} = h_t + f(h_t, \theta_t)$. The "$+h_t$" is usually read as "learn a residual correction," but written as $h_{t+1} - h_t = f(h_t,\theta_t)$ it is transparently a finite-difference step: put a step size on the layer index, $h_{t+\Delta t} - h_t = \Delta t \, f(h_t,\theta_t)$, divide by $\Delta t$ and let $\Delta t \to 0$, and we get $dh/dt = f$. A residual network is exactly forward Euler with step size hard-wired to $1$, run for a fixed number of layers — the crudest integrator anyone is taught, ignoring a century of better ones. That framing exposes three structural costs at once. Reverse-mode differentiation must store every intermediate activation $h_0,\dots,h_L$ to compute gradients, so training memory grows linearly with depth, $O(L)$ — the wall one hits when trying to make models deeper. Depth $L$ is a discrete hyperparameter chosen by hand, and every input gets exactly $L$ steps regardless of difficulty, with no principled way to spend more compute on harder inputs or to trade accuracy for speed after training. And the integrator itself is the worst available, with no error control and no adaptivity. The existing fixes each address one cost in isolation: reversible residual networks recompute activations to get $O(1)$ memory but only for a restricted partitioned/coupled architecture; learned adaptive-computation schemes train a secondary halting network at extra parameter and runtime cost; and differentiating through a solver's operations either stores the whole trajectory or, via forward sensitivity, costs $O(D^2)$ in the state size. A separate but parallel pain lives in density modeling, where the change-of-variables formula $\log p(z_1) = \log p(z_0) - \log\lvert\det \partial f/\partial z_0\rvert$ forces an $O(D^3)$ log-determinant, and the entire flow literature contorts its architectures — triangular couplings with hand-chosen partitions, or rank-one planar layers stacked many deep — just to keep that determinant cheap.

I propose Neural Ordinary Differential Equations: take the limit seriously and let the model *be* the ODE rather than a fixed stack of Euler steps. Parameterize the derivative of the hidden state with a single shared network $f$ and a single shared $\theta$ across all of continuous "depth," $$\frac{dh(t)}{dt} = f(h(t), t, \theta),$$ and define the layer output as the solution of the initial value problem at the end time, $h(t_1) = h(t_0) + \int_{t_0}^{t_1} f(h(t),t,\theta)\,dt = \mathrm{ODESolve}(h(t_0), f, t_0, t_1, \theta)$, handing the integration to a real, modern, adaptive black-box solver. The solver decides how many times to evaluate $f$ — call that the implicit depth — shrinking or growing its step to keep local error under a requested tolerance, so easy inputs get few evaluations and hard inputs get more, and the *same trained model* can be run faster or more accurately afterward just by changing the tolerance. That is adaptive computation for free, borrowed from numerical analysis rather than paid for with a separate halting network. Existence and uniqueness are not a concern: Picard's theorem guarantees a unique solution whenever $f$ is uniformly Lipschitz in $h$ and continuous in $t$, which a net with finite weights and a Lipschitz nonlinearity like $\tanh$ satisfies.

The real obstacle is training: given a loss $L(h(t_1))$, I need $dL/d\theta$ without backpropagating through the solver's internals — doing that would store every internal operation (destroying the memory advantage), leak the solver's discretization error into the gradient, and tie the method to one solver, becoming hopeless for implicit methods whose Newton iterations resist differentiation. What I want is an object that treats the solver as a black box, costs memory independent of depth, and is linear, not quadratic, in the state size. The right object is the adjoint $a(t) := \partial L/\partial z(t)$ (I switch the state to $z$ since it gets reused for densities), the continuous analog of the per-layer gradient in backprop. To find its law, note that over an infinitesimal step the state map is $z(t+\varepsilon) = T_\varepsilon(z(t))$ and the chain rule connects adjoints exactly as backprop connects adjacent layers, $a(t) = a(t+\varepsilon)\,\partial T_\varepsilon/\partial z$ (treating $a$ as a row vector). Taking $da/dt = \lim_{\varepsilon\to 0^+}[a(t+\varepsilon)-a(t)]/\varepsilon$, substituting that chain rule, and Taylor-expanding the map as $T_\varepsilon(z) = z + \varepsilon f + O(\varepsilon^2)$ so that $\partial T_\varepsilon/\partial z = I + \varepsilon\,\partial f/\partial z + O(\varepsilon^2)$, the identity terms cancel and the surviving first-order term gives the instantaneous chain rule $$\frac{da(t)}{dt} = -a(t)\,\frac{\partial f(z(t),t,\theta)}{\partial z},$$ or $da/dt = -(\partial f/\partial z)^\top a(t)$ in column convention. This runs *backwards*: $a(t_1) = \partial L/\partial z(t_1)$ is the one gradient the loss hands over directly, and integrating the adjoint ODE from $t_1$ down to $t_0$ recovers $a(t_0)$, just as backprop flows gradients output-to-input, now continuous.

A snag almost slips by: the adjoint dynamics contain $\partial f/\partial z$ evaluated along the trajectory $z(t)$, which I deliberately did not store. But the dynamics are reversible by construction — I already hold $z(t_1)$, so I append $dz/dt = f$ to the backward solve and recompute $z(t)$ as I go, integrating it backward alongside $a(t)$. Nothing from the forward pass is stored; memory is $O(1)$ in depth. (If unstable dynamics ever make the reconstruction drift, a handful of checkpointed $z$ values can be re-integrated between, but at sensible tolerances the drift is negligible, so the core method does not carry that complication.) To get the parameter gradient I fold $\theta$ into the state with trivial dynamics $d\theta/dt = 0$, and likewise treat $t$ as a state with $dt/dt = 1$ to recover endpoint gradients, forming an augmented state $s = [z,\theta,t]$ with $f_{\mathrm{aug}} = [f, 0, 1]$. Because $\theta$ and $t$ have constant dynamics their Jacobian rows vanish, and the same adjoint law $da_{\mathrm{aug}}/dt = -a_{\mathrm{aug}}\,\partial f_{\mathrm{aug}}/\partial s$ multiplies out component-wise into $da/dt = -a\,\partial f/\partial z$, $da_\theta/dt = -a\,\partial f/\partial\theta$, and $da_t/dt = -a\,\partial f/\partial t$. Since $\theta$ is constant, its total gradient is the integral of the middle equation with terminal condition $a_\theta(t_1) = 0$ (the loss touches $\theta$ only through the trajectory, so its gradient starts at zero and builds up as the solve runs backward), giving $$\frac{dL}{d\theta} = a_\theta(t_0) = -\int_{t_1}^{t_0} a(t)\,\frac{\partial f}{\partial\theta}\,dt = \int_{t_0}^{t_1} a(t)\,\frac{\partial f}{\partial\theta}\,dt,$$ where flipping the limits cancels the minus sign. The endpoint gradients follow from the boundary term $dL/dt_1 = a(t_1)\,f(z(t_1),t_1,\theta)$, initializing the backward time-adjoint with $-dL/dt_1$ and reading out $a_t(t_0)$, which in the simple two-endpoint case collapses to $dL/dt_0 = -a(t_0)\,f(z(t_0),t_0,\theta)$. So every gradient — w.r.t. the input, the parameters, and both time endpoints — comes from concatenating $[z, a, \text{the } \partial L/\partial\theta \text{ integral}, a_t]$ into one augmented vector and making a single backward call to the same black-box solver, whose dynamics return $[f, -a\,\partial f/\partial z, -a\,\partial f/\partial\theta, -a\,\partial f/\partial t]$.

What makes this linear rather than quadratic is that each integrand is a *row-vector-times-Jacobian* — a vector-Jacobian product — which reverse-mode autodiff computes natively in one backward pass through $f$ at a cost comparable to a single evaluation, never forming the $D\times D$ Jacobian that forward sensitivity would. Better still, one VJP seeded with $-a$ produces the products against all of $f$'s inputs — $z$, $\theta$, and $t$ — simultaneously, since they are just different input slots of the same graph, so one backward pass per solver step yields all three integrand terms. Nothing scales with $D^2$, and because the backward pass is itself an adaptive solve of a possibly smoother system it can even need fewer evaluations than the forward pass. When the loss depends on $z$ at several observation times $t_1,\dots,t_N$, the adjoint is integrated backward interval by interval, adding the local $\partial L/\partial z(t_i)$ onto the running adjoint at each observation before continuing — valid superposition because everything is linear in $a$.

The same continuous limit pays a second dividend in density modeling. Pushing the change-of-variables formula to the infinitesimal limit, $\log p(z(t+\varepsilon)) = \log p(z(t)) - \log\lvert\det\partial T_\varepsilon/\partial z\rvert$, gives $\partial \log p/\partial t = -\lim_{\varepsilon\to 0^+}\log\lvert\det\partial T_\varepsilon/\partial z\rvert/\varepsilon$, a $0/0$ form since at $\varepsilon=0$ the map is the identity and the determinant is $1$. Applying L'Hôpital in $\varepsilon$, the $1/\lvert\det\rvert$ prefactor goes to $1$ and drops, leaving $-\lim \partial_\varepsilon\lvert\det\partial T_\varepsilon/\partial z\rvert$; Jacobi's formula $\partial_\varepsilon\det A = \mathrm{tr}(\mathrm{adj}(A)\,\partial_\varepsilon A)$ with $A=I$ and $\mathrm{adj}(I)=I$ reduces this to a trace, and expanding $\partial T_\varepsilon/\partial z = I + \varepsilon\,\partial f/\partial z + O(\varepsilon^2)$ leaves $$\frac{\partial \log p(z(t))}{\partial t} = -\,\mathrm{tr}\!\left(\frac{\partial f}{\partial z}\right).$$ The cubic-cost log-*determinant* has become a *trace* — the linear part of the determinant's variation at the identity. Decisively, the trace is linear: for $dz/dt = \sum_{n=1}^M f_n(z)$, $d\log p/dt = -\sum_n \mathrm{tr}(\partial f_n/\partial z)$, an $M$-term sum, so a *wide* continuous flow with $M$ hidden units costs $O(M)$ where a standard wide flow layer would cost $O(M^3)$. The flow can be wide instead of deep, and a second gift comes free: $f$ need not be designed bijective, because Picard uniqueness already makes the flow map from $z(t_0)$ to $z(t_1)$ a bijection — trajectories cannot cross — so any Lipschitz $f$ is invertible and the whole partition/coupling/ordering apparatus is discarded. The simplest instance is the continuous planar flow $dz/dt = u\,h(w^\top z + b)$, whose Jacobian $\partial f/\partial z = u\,(\partial h/\partial z)^\top$ is rank one, and since the trace of an outer product is the inner product, $\partial \log p/\partial t = -u^\top\,\partial h/\partial z$, no matrix-determinant lemma needed. I solve the $(D{+}1)$-dimensional ODE for $[z,\log p]$ jointly with that trace as the extra coordinate's velocity; running data backward from the data time to the base time with the log-density difference initialized at zero returns $\log p_{\text{base}}(z_{\text{base}}) - \log p_{\text{data}}(x)$, so the data log-likelihood is $\log p_{\text{base}}(z_{\text{base}})$ *minus* the accumulated difference, and the flow runs forward to sample. One ODE engine thus powers continuous-depth supervised models, continuous normalizing flows, and continuous-time latent series alike.

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
