Let me lay out what I actually want from a generative model and where the tension is. I want three things at once: exact log-likelihoods, so I can train by maximum likelihood and report held-out likelihood honestly; cheap one-pass sampling; and freedom to use whatever neural network I like for the transformation. Flow models — warp a simple base `p_z` through an invertible `f: R^D → R^D` — give me the first two through the change-of-variables formula. For `z ∼ p_z` and `x = f(z)`,

`log p_x(x) = log p_z(z) − log |det( ∂f/∂z )|`.

The whole game is that determinant. For a general `f` it's `O(D³)`, which is hopeless at scale. Every existing flow buys a cheap determinant by *constraining* `f`:

- Restrict the functional form so a determinant identity applies — planar flows are literally a one-layer net with a *single* hidden unit per step, almost no capacity, and they don't have a usable inverse so you can't sample from data with them.
- Force an autoregressive ordering so the Jacobian is triangular and the determinant is the product of its diagonal — great densities, but inverting to sample takes `D` *sequential* passes.
- Partition the dimensions and affinely transform one block conditioned on the other (NICE, Real NVP, Glow) — cheap triangular determinant, inverse as cheap as forward, but each layer is constrained, so you stack many of them and still can't represent an arbitrary Jacobian per step.

So the architectural restriction *is* the price of the tractable determinant. I want to stop paying it. The question is whether there's a formulation where the expensive object isn't a determinant at all.

Here's the lever. A discrete flow is a finite composition of maps. What if the transformation is the solution of an ODE instead — continuous-time dynamics `∂z(t)/∂t = f(z(t), t; θ)`, with `z(t_0) ∼ p_{z0}` the base sample and `z(t_1) = x` the data? Then I need to know how the log-density evolves along the trajectory. Let me derive it rather than guess. Over an infinitesimal step `Δt`, the map is `z(t+Δt) = z(t) + Δt · f(z(t),t) + O(Δt²)`, so its Jacobian is `I + Δt · ∂f/∂z + O(Δt²)`. The discrete change of variables says the log-density picks up `− log|det(Jacobian)|`. For a matrix close to the identity,
`log det( I + Δt · ∂f/∂z ) = Tr( Δt · ∂f/∂z ) + O(Δt²) = Δt · Tr(∂f/∂z) + O(Δt²)`,
using `log det(I + εM) = ε Tr(M) + O(ε²)`. So
`log p(z(t+Δt)) − log p(z(t)) = − Δt · Tr(∂f/∂z) + O(Δt²)`,
and dividing by `Δt` and letting `Δt → 0`,

`∂ log p(z(t))/∂t = − Tr( ∂f/∂z(t) )`.

This is the instantaneous change of variables. Stare at what just happened: in continuous time the *determinant* collapses to a *trace*. That matters enormously, because a determinant needs structure in the Jacobian to be cheap, while a trace is just a sum of diagonal entries — it's *linear* in the matrix and asks nothing of its structure. Integrating along the trajectory,

`log p(z(t_1)) = log p(z(t_0)) − ∫_{t_0}^{t_1} Tr(∂f/∂z(t)) dt`.

And to go from a data point `x` to its likelihood, I run the dynamics — both `z` and the accumulated log-density change — backward from `t_1` to `t_0`. Stack the two into one augmented state and solve the initial value problem with `[z(t_1), Δlogp] = [x, 0]`:

`[ z_0 ; log p(x) − log p(z(t_1)) ] = ∫_{t_1}^{t_0} [ f(z(t),t;θ) ; − Tr(∂f/∂z) ] dt`,

then `log p̂(x) = log p_{z0}(z_0) − Δlogp`. (For this IVP to have a unique solution I need `f` and its first derivatives Lipschitz, which I get by using smooth Lipschitz activations like tanh or softplus.) To train I maximize this likelihood; differentiating through a black-box ODE solve is exactly what the adjoint method handles — solve a second ODE backward for the adjoint `a(t) = −∂L/∂z(t)` and accumulate `dL/dθ = −∫_{t_1}^{t_0} (∂L/∂z)ᵀ (∂f/∂θ) dt`, with `O(1)` memory because I never store intermediate activations.

So this continuous formulation already frees `f` somewhat and drops the cost. But how much? Computing `Tr(∂f/∂z)` exactly means getting every diagonal entry `∂f_i/∂z_i`, and each one is a separate derivative of `f` — `D` of them, so `O(D²)` per solver step, roughly `D` evaluations of `f`. Better than `O(D³)`, but still scaling with `D²`, and that's enough to keep people using restricted, low-width `f` (planar CNF widens the single hidden unit to many, but stays cautious). I haven't actually removed the architectural pressure; I've just lowered it. The trace is still the bottleneck.

Let me think about what a trace really costs if I'm willing to estimate it. Two things are true. First, reverse-mode autodiff gives me a *vector-Jacobian product* `vᵀ(∂f/∂z)` for about the price of a single evaluation of `f` — I never have to materialize the `D×D` Jacobian, I just push a cotangent vector through. Second — and this is the unlock — there's an unbiased stochastic estimate of a trace that only needs such products. Take any `D×D` matrix `A` and a random vector `ε` with `E[ε] = 0` and `Cov(ε) = I`. Then

`E[ εᵀ A ε ] = E[ Σ_{i,j} ε_i A_{ij} ε_j ] = Σ_{i,j} A_{ij} E[ε_i ε_j] = Σ_{i,j} A_{ij} δ_{ij} = Σ_i A_{ii} = Tr(A)`,

because `E[ε_i ε_j] = δ_{ij}` is exactly the covariance condition. So `εᵀ A ε` is an unbiased estimator of `Tr(A)` — Hutchinson's estimator. Apply it to `A = ∂f/∂z`: I form `εᵀ(∂f/∂z)` as one vector-Jacobian product (cost ≈ one `f` eval, `O(DH)`), then take its dot with `ε` (cost `O(D)`). That's an unbiased estimate of `Tr(∂f/∂z)` in `O(D)` — no `D` separate derivatives, no Jacobian matrix, and, crucially, *nothing required of the Jacobian's structure*. The last reason to constrain `f` is gone. `f` can be any Lipschitz neural net.

I have to be a little careful about where the randomness lives, though. The log-density is a *time integral* of the trace, and I'm going to feed the dynamics to an adaptive ODE solver that wants a deterministic right-hand side within a solve. If I resampled `ε` at every function evaluation, the integrand would jitter and the adaptive stepper would chase noise. The fix: sample one `ε` and *fix it for the whole solve*. Does fixing it bias anything? No — by Fubini I can pull the expectation outside the time integral:

`log p(z(t_1)) = log p(z(t_0)) − ∫_{t_0}^{t_1} E_{p(ε)}[ εᵀ (∂f/∂z(t)) ε ] dt = log p(z(t_0)) − E_{p(ε)}[ ∫_{t_0}^{t_1} εᵀ (∂f/∂z(t)) ε \, dt ]`,

so a single sampled `ε` held constant across the solve gives an unbiased estimate of the whole log-density. For `p(ε)` I just need mean zero, identity covariance: a standard Gaussian works, and so does Rademacher (each entry `±1` with probability one-half, which has identity covariance and unit entries). Now the cost of the likelihood is `O((DH + D) L̂)` for `L̂` solver evaluations — versus `O((DH + D²) L̂)` for the exact-trace CNF and `O((DH + D³) L)` for a discrete flow. Free-form Jacobian of reversible dynamics.

One more thing I can squeeze: variance. Hutchinson's estimator is unbiased but noisy, and its variance for `Tr(A)` grows like `‖A‖_F²`, the squared Frobenius norm. If my dynamics net has a hidden bottleneck — write `f = g ∘ h` where `h` maps into a hidden space of width `H < D` — I can shrink the matrix the noise vector probes. The Jacobian factors by the chain rule, `∂f/∂z = (∂g/∂h)(∂h/∂z)`, and the cyclic property of trace lets me reorder:

`Tr(∂f/∂z) = Tr( (∂g/∂h)(∂h/∂z) ) = Tr( (∂h/∂z)(∂g/∂h) )`,

where the first product is `D×D` but the *reordered* one is `H×H`. So I can estimate `Tr` with an `ε ∈ R^H` against the smaller matrix: `E_{p(ε)}[ εᵀ (∂h/∂z)(∂g/∂h) ε ]`, computed as two chained vector-Jacobian products. Smaller matrix, smaller Frobenius norm, lower variance — pick `H` to be the smallest hidden width. (Empirically this helps Gaussian `ε` more than Rademacher.)

A couple of architectural choices follow from making the ODE actually integrable. The dynamics `f` takes both `z(t)` and `t`. I could feed `t` through a hypernetwork that generates weights, but that tends to produce dynamics that are nasty to integrate; the simple, stable choice is to *concatenate* `t` onto the layer input at every layer. And the activations should be smooth — softplus or tanh — both because existence/uniqueness wants Lipschitz `f` and because smooth dynamics stay non-stiff so a general-purpose solver doesn't blow up its evaluation count.

Now the code. The dynamics net stacks layers that each get `t` concatenated, with a smooth nonlinearity:

```python
import torch, torch.nn as nn

class ODEnet(nn.Module):
    """The dynamics f(z(t), t): each layer gets t concatenated; smooth activations."""
    def __init__(self, hidden_dims, input_shape, layer_type="concat", nonlinearity="softplus"):
        super().__init__()
        base_layer = ConcatLinear            # concatenates t onto the layer input
        layers, activations = [], []
        in_dim = input_shape[0]
        for dim_out in hidden_dims + (input_shape[0],):
            layers.append(base_layer(in_dim, dim_out))
            activations.append(NONLINEARITIES[nonlinearity])   # softplus / tanh / swish
            in_dim = dim_out
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activations[:-1])   # none after the last layer

    def forward(self, t, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx
```

The trace, two ways — the brute-force exact diagonal (loop the `D` derivatives, used at test or for tiny `D`), and the Hutchinson estimate (one vector-Jacobian product then a dot with `ε`):

```python
def divergence_bf(dx, y):
    # exact Tr(∂f/∂z): sum the D diagonal entries, one derivative each — O(D²)
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0][:, i]
    return sum_diag

def divergence_approx(f, y, e):
    # Hutchinson: εᵀ(∂f/∂z)ε — one vJP, then dot with ε — O(D), unbiased
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]    # εᵀ ∂f/∂z
    return (e_dzdx * e).view(y.shape[0], -1).sum(dim=1)            # · ε

def sample_rademacher_like(y):
    return torch.randint(0, 2, y.shape).to(y) * 2 - 1              # ±1 entries, Cov = I

def sample_gaussian_like(y):
    return torch.randn_like(y)                                    # mean 0, Cov = I
```

The augmented dynamics — state `z` integrated together with its log-density change `−Tr̃`, with `ε` sampled once and held fixed for the solve:

```python
class ODEfunc(nn.Module):
    def __init__(self, diffeq, divergence_fn="approximate", rademacher=False):
        super().__init__()
        self.diffeq = diffeq
        self.rademacher = rademacher
        self.divergence_fn = divergence_approx if divergence_fn == "approximate" else divergence_bf
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e                          # reset; ε will be fixed for the whole solve
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]                        # log-density change is the second state
        self._num_evals += 1
        t = torch.tensor(t).type_as(y)
        # sample ε ONCE and keep it fixed across the solve (deterministic RHS, still unbiased)
        if self._e is None:
            self._e = sample_rademacher_like(y) if self.rademacher else sample_gaussian_like(y)
        with torch.set_grad_enabled(True):
            y.requires_grad_(True); t.requires_grad_(True)
            dy = self.diffeq(t, y)           # dz/dt = f(z,t)
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(y.shape[0], 1)        # exact for 2D / test
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(y.shape[0], 1)
        return (dy, -divergence)             # d/dt [z, logp] = [f, -Tr(∂f/∂z)]
```

And the flow itself — solve the augmented IVP from `t_0` to `t_1` (flip the times to invert), carrying the running log-density as the second state:

```python
from torchdiffeq import odeint_adjoint as odeint

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
            integration_times = _flip(integration_times, 0)
        self.odefunc.before_odeint()         # fixes a fresh ε for this solve
        z_t, logpz_t = odeint(self.odefunc, (z, _logpz), integration_times,
                              atol=self.atol, rtol=self.rtol, method=self.solver)
        z_t, logpz_t = z_t[-1], logpz_t[-1]
        return (z_t, logpz_t) if logpz is not None else z_t
```

To get a likelihood I run `CNF` in reverse from a data batch `x` with `logpz = 0`, recovering `z_0` and the accumulated `Δlogp`, then add the base log-density: `log p̂(x) = log p_{z0}(z_0) − Δlogp`. To sample I draw `z_0 ∼ p_{z0}` and run forward to `x = z(t_1)` — one pass, no sequential inversion.

Tracing the chain back: I wanted exact-likelihood, one-pass-sampling generative modeling with an unrestricted `f`, but the change-of-variables determinant is `O(D³)` and the usual fix is to constrain the architecture. Moving to continuous-time dynamics turned that determinant into a *trace* (the instantaneous change of variables), which is linear and structure-agnostic and brought the cost to `O(D²)` — but exact trace still scales with `D²` and kept `f` cautious. Hutchinson's identity `Tr(A) = E[εᵀAε]`, evaluated with a single reverse-mode vector-Jacobian product and `ε` held fixed across each solve (unbiased by Fubini), drops the trace to an `O(D)` unbiased estimate and removes the last reason to constrain `f`. The cyclic-trace bottleneck trick shrinks the probed matrix to cut variance, the adjoint method backpropagates through the solver in `O(1)` memory, and concatenated-time, smooth-activation dynamics keep the ODE integrable — together, a continuous reversible generative model with a genuinely free-form Jacobian.
