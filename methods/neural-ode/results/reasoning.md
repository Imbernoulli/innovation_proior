Let me start from the thing that has been bugging me about residual networks. The update is

  h_{t+1} = h_t + f(h_t, θ_t).

I stare at that "+h_t". Everyone reads it as "learn a residual correction," which is fine. But written this way it's also, transparently, a finite-difference step:

  h_{t+1} − h_t = f(h_t, θ_t).

If I imagine the layer index t as a time and put a step size on it, h_{t+Δt} − h_t = Δt·f(h_t, θ_t), then dividing by Δt and letting Δt → 0 gives dh/dt = f. A ResNet is forward Euler with step size 1. That's not a metaphor, it's literally what the recurrence computes: Euler's method, the very first and crudest integrator anyone is taught, run for exactly L steps of size 1.

So three things bother me at once. First, Euler at step size 1 is the worst possible integrator — there have been a hundred-plus years of better ones (Runge, Kutta, the adaptive embedded methods, the implicit Adams/BDF solvers) that monitor their own error and pick their own step sizes, and a ResNet uses none of that; it takes L crude steps whether the input is easy or hard. Second, the depth L is a discrete hyperparameter I pick by hand, and every input gets exactly L steps regardless of difficulty. Third, and this is the one that actually hurts in practice: to backprop, I have to store every intermediate h_0, …, h_L, because reverse-mode needs the forward activations. Memory is O(L). That's the wall I keep hitting when I try to make models deeper — I run out of memory long before depth stops helping.

So here's the question that won't leave me alone. What if I take the limit seriously? Don't approximate an ODE with a fixed stack of Euler steps — let the model *be* the ODE,

  dh(t)/dt = f(h(t), t, θ),

with one shared f and one shared θ across all of continuous "depth," and hand the actual integration to a real, modern, adaptive solver. The output is just the solution of the initial value problem at the end time:

  h(t₁) = h(t₀) + ∫_{t₀}^{t₁} f(h(t), t, θ) dt = ODESolve(h(t₀), f, t₀, t₁, θ).

Now the solver decides how many times to evaluate f. Call that number L̃ — that's my "implicit depth," and the solver sets it adaptively per input to hit whatever error tolerance I ask for. Easy inputs, few evaluations; hard inputs, more. And after training I can loosen the tolerance to go faster, or tighten it for accuracy, with the *same* trained model. That's adaptive computation for free — I don't have to train a separate halting network the way the adaptive-computation-time people do, with their extra parameters and extra overhead. The solver already does error control; I'm just borrowing a century of numerical analysis.

Before I get carried away — does the solution even exist and is it unique? Picard's theorem says yes, as long as f is uniformly Lipschitz in h and continuous in t. A net with finite weights and a Lipschitz nonlinearity like tanh or relu satisfies that. Good, no landmine there.

But now I hit the real wall, and it's a bad one. How do I get gradients? I have a loss L(h(t₁)) and I need dL/dθ to train. The naive thing is: the solver is just a sequence of arithmetic operations, so let autodiff backprop through all of them. But that's a disaster on two counts. It stores every internal operation of the solver — which throws away the entire memory advantage I was chasing; I'm back to O(number of solver steps), and worse, I don't even control how many steps an adaptive solver takes. And it differentiates the solver's *internal* approximation, so the solver's discretization error leaks straight into my gradients. On top of that, if I ever want to use an implicit method (Adams/BDF), each step internally solves a nonlinear system by Newton iteration — backpropagating through those iterations is a nightmare, and it would tie me to one specific solver forever.

The libraries that do this — backpropagating through the operations of the forward solver — exist, and they pay exactly these costs: full-trajectory memory, solver-specific code, internal error in the gradient. And the alternative people use, forward sensitivity analysis (carry ∂h/∂θ along with h), is quadratic in the number of variables, because you're propagating a whole Jacobian-sized object forward. For a big hidden state that's hopeless.

So I want something that (a) treats the solver as a black box — works for *any* solver, never touches its internals — (b) costs memory independent of depth, and (c) is linear, not quadratic, in the state size. Let me think about what object I'd actually need.

The thing I care about at each instant is: how sensitive is the final loss to the state right now? Define

  a(t) := ∂L/∂z(t)

(I'll switch to calling the state z, since this is going to get reused for densities too). In a discrete net, the gradient at layer t comes from the gradient at layer t+1 by the chain rule: dL/dh_t = (dL/dh_{t+1})(dh_{t+1}/dh_t). That's just backprop. I want the continuous analog of that — a law for how a(t) itself evolves. If a(t) follows its own differential equation, then I can get the gradient by *solving that equation*, again with a black-box solver, and never store anything.

Let me derive it. Over an infinitesimal step ε, the state map is

  z(t+ε) = z(t) + ∫_t^{t+ε} f dτ = T_ε(z(t)),

and the chain rule connects the adjoints at t and t+ε exactly like backprop connects adjacent layers:

  a(t) = a(t+ε) · ∂T_ε(z(t))/∂z(t).

(I'll treat a as a row vector here, the way it falls out of "gradient times Jacobian"; I'll restore transposes at the end.) Now just take the time derivative from the definition:

  da(t)/dt = lim_{ε→0⁺} [a(t+ε) − a(t)] / ε.

Substitute the chain rule for a(t):

  = lim_{ε→0⁺} [a(t+ε) − a(t+ε) ∂T_ε/∂z] / ε.

I need ∂T_ε/∂z. Taylor-expand the map: T_ε(z) = z + ε f(z,t,θ) + O(ε²), so

  ∂T_ε/∂z = I + ε ∂f/∂z + O(ε²).

Plug it in:

  = lim_{ε→0⁺} [a(t+ε) − a(t+ε)(I + ε ∂f/∂z + O(ε²))] / ε
  = lim_{ε→0⁺} [a(t+ε) − a(t+ε) − ε a(t+ε) ∂f/∂z + O(ε²)] / ε
  = lim_{ε→0⁺} [ − ε a(t+ε) ∂f/∂z + O(ε²)] / ε
  = lim_{ε→0⁺} [ − a(t+ε) ∂f/∂z + O(ε)]
  = − a(t) ∂f/∂z.

There it is:

  da(t)/dt = − a(t) ∂f(z(t),t,θ)/∂z,

or, if I store gradients as columns, da/dt = −(∂f/∂z)ᵀ a(t). I will keep the row-vector notation for the derivation, because it lines up with a vector-Jacobian product. This is the instantaneous chain rule. And notice it runs *backwards*: I know a at the end, a(t₁) = ∂L/∂z(t₁), because that's the one gradient the loss hands me directly, and I integrate this ODE from t₁ down to t₀ to get a(t₀) = ∂L/∂z(t₀). Just like backprop flows gradients from the output back to the input — same direction, now continuous.

There's an immediate snag and I almost miss it. The adjoint ODE has ∂f/∂z(z(t)) in it, so to integrate it backward I need z(t) along the whole trajectory. But I deliberately threw z(t) away — not storing it was the whole point! For a second this feels like it defeats the idea. Then: the dynamics are reversible by construction. I know z(t₁) (it's the output I already have), and I can just append dz/dt = f to the backward solve and recompute z(t) as I go, integrating it backward right alongside a(t). So the augmented backward state carries (z, a), starts at (z(t₁), ∂L/∂z(t₁)), and runs to t₀. Nothing from the forward pass needs to be stored. Memory O(1) in depth. The wall is gone.

(One honest worry: reconstructing z backward can drift if the dynamics are unstable, since I'm re-integrating rather than reading back exact saved values. If that ever bites, I can checkpoint a handful of z values on the forward pass and re-integrate between them. In practice at sensible tolerances the drift is negligible, so I won't complicate the core method with it.)

Now the gradient I actually want is dL/dθ, not dL/dz(t₀). How do I fold θ in? Here's the trick: treat θ as part of the state with trivial dynamics. θ doesn't change in time, so dθ/dt = 0. And while I'm at it, treat t itself as a state with dt/dt = 1, so I'll get endpoint gradients too. Stack everything into an augmented state s = [z, θ, t] with augmented dynamics

  f_aug([z,θ,t]) = [ f(z,t,θ), 0, 1 ].

Its Jacobian is the block matrix

  ∂f_aug/∂[z,θ,t] = [ [∂f/∂z, ∂f/∂θ, ∂f/∂t], [0, 0, 0], [0, 0, 0] ],

because θ and t have constant dynamics (their rows are zero). The augmented adjoint is a_aug = [a, a_θ, a_t] where a_θ = dL/dθ and a_t = dL/dt. The law I just derived applies unchanged to the augmented system: da_aug/dt = − a_aug · ∂f_aug/∂s. Multiplying out the row vector by the block matrix, component by component:

  da/dt   = − a ∂f/∂z      (the state adjoint, as before),
  da_θ/dt = − a ∂f/∂θ,
  da_t/dt = − a ∂f/∂t.

The middle equation is what I wanted. θ is constant in t, so its *total* gradient is the integral of da_θ/dt, with the terminal condition a_θ(t₁) = 0 (the loss doesn't depend on θ "through the end-time state directly" — θ's contribution is entirely accumulated along the trajectory, so it starts at zero at t₁ and builds up as I integrate backward):

  dL/dθ = a_θ(t₀) = − ∫_{t₁}^{t₀} a(t) ∂f/∂θ dt.

Let me sanity-check the sign and the direction, because this is exactly where it's easy to slip. The integral runs from t₁ down to t₀ (backward), and there's an explicit minus sign from da_θ/dt = −a ∂f/∂θ. If I prefer to write it as a forward integral I flip the limits and the sign cancels: dL/dθ = +∫_{t₀}^{t₁} a(t) ∂f/∂θ dt. Both say the same thing. The endpoint gradients need one more boundary check. Moving the final time changes the loss at rate dL/dt₁ = a(t₁) f(z(t₁),t₁,θ). For the start time I initialize the backward time-adjoint with −dL/dt₁, integrate da_t/dt = −a ∂f/∂t from t₁ to t₀, and read out a_t(t₀). In the simple two-endpoint case with no direct loss on z(t₀), this collapses to dL/dt₀ = −a(t₀) f(z(t₀),t₀,θ), which is the boundary-term sanity check I want.

So I can get *every* gradient — w.r.t. the input z(t₀), the parameters θ, and both time endpoints — by concatenating [z, a, the running ∂L/∂θ integral (and a_t)] into one big augmented vector and making a single backward call to the same black-box ODE solver, from t₁ to t₀. The augmented dynamics function returns

  [ f(z,t,θ), − a(t)∂f/∂z, − a(t)∂f/∂θ, − a(t)∂f/∂t ],

initialized at t₁ with [z(t₁), ∂L/∂z(t₁), 0, −dL/dt₁]. Out the other end at t₀ comes [z(t₀), ∂L/∂z(t₀), ∂L/∂θ, ∂L/∂t₀]. Beautiful — gradients of an ODE solve computed by another ODE solve.

Now the cost. Each of those terms is a(t) times a Jacobian of f — a(t)∂f/∂z, a(t)∂f/∂θ, a(t)∂f/∂t. The crucial thing is I never have to *form* those Jacobians. ∂f/∂z is D×D; materializing it would be O(D²) memory and forward-mode would make the whole thing quadratic — exactly the forward-sensitivity cost I was trying to escape. But a row-vector-times-Jacobian is precisely a **vector-Jacobian product**, and that is what reverse-mode autodiff computes natively in a single backward pass through f, at a cost comparable to one evaluation of f. Even better: one VJP through f, seeded with the vector −a, simultaneously produces the products against *all* of f's inputs — z, θ, and t — because they're just different input slots of the same computation graph. So one backward pass through f per solver step gives me all three integrand terms at once. That's why the whole method is linear in the state size, not quadratic. The solver calls f and its VJP a handful of times per step; nothing scales with D².

And there's a small empirical surprise lurking here that I'd want to confirm: because the backward pass is itself an adaptive solve of a (possibly smoother) augmented system, it can need *fewer* function evaluations than the forward pass — whereas backprop-through-the-solver would, by construction, have to differentiate *every single* forward evaluation. So the adjoint isn't just more memory-efficient, it can be more compute-efficient too.

One more case to handle. If the loss depends on z at several times — say a time series with observations at t₁, t₂, …, t_N — then I can't do one clean backward sweep, because each observation injects its own gradient ∂L/∂z(t_i) into the adjoint. The fix is natural: integrate the adjoint backward interval by interval, [t_N, t_{N-1}], [t_{N-1}, t_{N-2}], …, and at each observation time add the corresponding ∂L/∂z(t_i) onto the running adjoint a before continuing. That's just superposition of the gradient contributions, which is fine because everything is linear in a.

Let me write the supervised piece down to make sure it's real code and not just equations. f is a small network; I feed time in as an extra input so it can be genuinely time-varying. The forward of a "block" is one odeint from t=0 to t=1; the backward is the augmented adjoint solve.

```python
import torch, torch.nn as nn

def _axpy(a, xs, ys):                           # a*xs + ys, over a tensor or a tuple
    if isinstance(ys, tuple):
        return tuple(a*x + yy for x, yy in zip(xs, ys))
    return a*xs + ys

def rk4_step(func, t, dt, y):                  # one 4th-order Runge-Kutta step
    k1 = func(t, y)
    k2 = func(t + dt/2, _axpy(dt/2, k1, y))
    k3 = func(t + dt/2, _axpy(dt/2, k2, y))
    k4 = func(t + dt,   _axpy(dt,   k3, y))
    comb = lambda yy,a,b,c,d: yy + dt/6 * (a + 2*b + 2*c + d)
    if isinstance(y, tuple):
        return tuple(comb(*p) for p in zip(y, k1, k2, k3, k4))
    return comb(y, k1, k2, k3, k4)             # not Euler — this is the point

def odeint(func, y0, t, step=None):            # black-box solver; y a tensor OR a tuple
    sol, y = [y0], y0
    for i in range(len(t)-1):
        t0, t1 = t[i], t[i+1]
        n = 1 if step is None else max(1, int(((t1-t0).abs()/step).item()))
        dt = (t1 - t0)/n
        for _ in range(n):
            y = rk4_step(func, t0, dt, y); t0 = t0 + dt
        sol.append(y)
    if isinstance(y0, tuple):
        return tuple(torch.stack([s[j] for s in sol]) for j in range(len(y0)))
    return torch.stack(sol)
```

And the adjoint, as a custom autograd Function so the forward stores nothing and the backward runs the augmented solve:

```python
class OdeintAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, y0, t, *params):
        with torch.no_grad():
            ans = odeint(func, y0, t)          # forward solve, no graph kept
        ctx.func = func
        ctx.save_for_backward(t, ans, *params)
        return ans

    @staticmethod
    def backward(ctx, grad_out):               # grad_out = dL/dz at each output time
        func = ctx.func
        t, ans, *params = ctx.saved_tensors
        params = tuple(params)

        def aug_dynamics(t, aug):
            vjp_t, z, a = aug[0], aug[1], aug[2]  # canonical state: (vjp_t, z, a, dL/dθ)
            with torch.enable_grad():
                z  = z.detach().requires_grad_(True)
                t_ = t.detach().requires_grad_(True)
                f_eval = func(t_, z)
                # one VJP gives the -a products for time, state, and parameters
                vjp_t_step, vjp_z, *vjp_p = torch.autograd.grad(
                    f_eval, (t_, z)+params, -a, allow_unused=True, retain_graph=True)
            vjp_t_step = torch.zeros_like(t_) if vjp_t_step is None else vjp_t_step
            vjp_z = torch.zeros_like(z) if vjp_z is None else vjp_z
            vjp_p = [torch.zeros_like(p) if g is None else g for p,g in zip(params, vjp_p)]
            return (vjp_t_step, f_eval, vjp_z, *vjp_p)

        vjp_t = torch.zeros((), dtype=t.dtype, device=t.device)
        time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
        adj_z = grad_out[-1]                    # terminal condition a(t₁)=∂L/∂z(t₁)
        adj_p = [torch.zeros_like(p) for p in params]   # a_θ(t₁)=0
        for i in range(len(t)-1, 0, -1):       # integrate backward, interval by interval
            dLd_cur_t = func(t[i], ans[i]).reshape(-1).dot(grad_out[i].reshape(-1))
            vjp_t = vjp_t - dLd_cur_t
            time_vjps[i] = dLd_cur_t
            aug0 = (vjp_t, ans[i], adj_z, *adj_p)
            aug  = odeint(aug_dynamics, aug0, torch.stack([t[i], t[i-1]]))
            vjp_t, _, adj_z, *adj_p = [a[-1] for a in aug]
            adj_z = adj_z + grad_out[i-1]      # inject ∂L/∂z(t_i) at each observation
        time_vjps[0] = vjp_t
        return (None, adj_z, time_vjps, *adj_p)
```

That's the engine. Switch hats to density modeling. The change-of-variables formula for an invertible map z₁ = f(z₀) is

  log p(z₁) = log p(z₀) − log|det ∂f/∂z₀|.

The whole pain of normalizing flows lives in that log-determinant: O(D³) in general. Every flow design is a contortion to make it cheap. Coupling layers (NICE, RealNVP) split the dimensions and transform half conditioned on the other, forcing the Jacobian triangular so the determinant is the product of the diagonal — but then each layer only touches half the state, and you have to choose a partition and an ordering of the dimensions. Planar flows (Rezende & Mohamed) do z + u·h(wᵀz+b), a rank-one perturbation whose determinant is |1 + uᵀ∂h/∂z| by the matrix-determinant lemma — cheap, but each layer has a single hidden unit, so you only get expressiveness by stacking many of them. A *wide* layer with M hidden units would cost O(M³) again, so the whole literature is stuck building deep stacks of one-unit layers.

But I now have a continuous transformation, dz/dt = f(z,t). What does the density do as z flows? Let me push the discrete formula to the infinitesimal limit, the same move that gave me the ODE in the first place. Over a step ε, with T_ε the state map,

  log p(z(t+ε)) = log p(z(t)) − log|det ∂T_ε/∂z|,

so

  ∂ log p(z(t))/∂t = lim_{ε→0⁺} [log p(z(t)) − log|det ∂T_ε/∂z| − log p(z(t))] / ε
                   = − lim_{ε→0⁺} log|det ∂T_ε/∂z| / ε.

At ε = 0 the map is the identity, T_0 = z, so ∂T_0/∂z = I and det = 1, and log 1 = 0 — the numerator and denominator both vanish. That's a 0/0, so L'Hôpital in ε:

  = − lim_{ε→0⁺} [ ∂/∂ε log|det ∂T_ε/∂z| ] / [ ∂/∂ε ε ]
  = − lim_{ε→0⁺} [ (1/|det ∂T_ε/∂z|) · ∂/∂ε |det ∂T_ε/∂z| ].

As ε → 0 the determinant → 1, so that leading 1/|det| factor → 1 and drops out:

  = − lim_{ε→0⁺} ∂/∂ε |det ∂T_ε/∂z|.

Now I need the derivative of a determinant. Jacobi's formula: ∂/∂ε det A(ε) = tr( adj(A) · ∂A/∂ε ), where adj is the adjugate. Here A = ∂T_ε/∂z, and at ε = 0, A = I, so adj(I) = I:

  = − tr( lim_{ε→0⁺} ∂/∂ε ∂T_ε/∂z ).

Finally expand the map: T_ε = z + ε f(z,t) + O(ε²), so ∂T_ε/∂z = I + ε ∂f/∂z + O(ε²), and ∂/∂ε of that is ∂f/∂z + O(ε), which → ∂f/∂z as ε → 0. Therefore

  ∂ log p(z(t))/∂t = − tr( ∂f/∂z ).

Stare at that. The cubic-cost log-**determinant** has become a **trace**. The determinant is what was killing me — its derivative is a product over a permutation, fundamentally nonlinear and O(D³). But in the continuous limit the determinant evaluates at the identity, where Jacobi's formula and the adjugate of I leave behind only the linear part of the change, and the linear part of a determinant's variation is exactly the trace. A trace is just the sum of the diagonal of ∂f/∂z — and, decisively, it's *linear*: tr(Σ_n J_n) = Σ_n tr(J_n).

That linearity is the whole game. If my dynamics is a sum of M simple functions, dz/dt = Σ_{n=1}^M f_n(z), then d log p/dt = − Σ_n tr(∂f_n/∂z), an M-term sum. So a *wide* continuous flow with M hidden units costs O(M), where a standard wide flow layer would cost O(M³). The reason every discrete flow was a deep stack of single-unit layers — the cubic determinant — simply doesn't apply here. I can be wide instead of deep.

And there's a second gift I almost overlook: f doesn't have to be designed bijective. In the discrete world I had to engineer invertible coupling maps and pick partitions and orderings, all to guarantee a bijection with a cheap determinant. But here, Picard's uniqueness theorem already guarantees that the *flow map* from z(t₀) to z(t₁) is a bijection — distinct initial conditions can't cross, because that would violate uniqueness of solutions. So any Lipschitz f gives an invertible transformation automatically. I can throw out the entire partition/coupling/ordering apparatus.

Let me write the simplest concrete instance, the continuous analog of the planar flow: dz/dt = u·h(wᵀz + b). Its Jacobian is ∂f/∂z = u (∂h/∂z)ᵀ, an outer product of u with the gradient of the scalar h — rank one. And the trace of an outer product is the inner product, tr(u vᵀ) = uᵀv, so

  ∂ log p/∂t = − tr( u (∂h/∂z)ᵀ ) = − uᵀ ∂h/∂z,

which is trivial to compute, no determinant lemma needed. I solve the (D+1)-dimensional ODE for [z, log p] jointly, with that trace as the extra coordinate's velocity. Because the flow is reversible at essentially the same cost forward and backward, I can train it by maximum likelihood. The sign is worth checking carefully. If I run data backward from the data time to the base time with d log p/dt = −tr(∂f/∂z) and initialize the accumulated log-density difference at zero, the backward solve returns log p_base(z_base) − log p_data(x). So the data log-likelihood is log p_base(z_base) − logp_diff, not plus. Then I can run the flow forward to sample. (And if I want the dynamics to turn on and off over the flow, I can make the parameters depend on t via a small hypernetwork and gate each unit with a σ_n(t) ∈ (0,1); that's free, it doesn't touch the trace cost.)

```python
class PlanarCNF(nn.Module):
    # dz/dt = u h(wᵀz+b);  d log p/dt = -tr(∂f/∂z) = -uᵀ ∂h/∂z
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim)*0.1)
        self.w = nn.Parameter(torch.randn(dim)*0.1)
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, t, state):
        z, _ = state
        with torch.enable_grad():
            z = z.requires_grad_(True)
            a = z @ self.w + self.b
            h = torch.tanh(a)
            f = h[..., None] * self.u
            dhdz = (1 - h**2)[..., None] * self.w     # ∂h/∂z
            trace = (self.u * dhdz).sum(-1)           # uᵀ ∂h/∂z
        return (f, -trace[..., None])                 # tuple state: (dz/dt, d log p/dt)

def cnf_logprob(cnf, x, base_logprob, t0=0.0, t1=1.0):
    logp_diff_t1 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    t = torch.tensor([t1, t0], device=x.device, dtype=x.dtype)   # data → base
    z_t, logp_diff_t = odeint(cnf, (x, logp_diff_t1), t)         # tuple-state solve
    z_base, logp_diff = z_t[-1], logp_diff_t[-1].view(-1)
    return base_logprob(z_base) - logp_diff   # log p_base − ∫ tr(∂f/∂z) dt
```

And the same ODE machinery serves a third model with almost no extra work: a generative time-series model. Put an ODE-defined latent trajectory inside a VAE. An RNN reads the (possibly irregularly sampled) observations and outputs a posterior q(z(t₀)); sample z(t₀); a single ODESolve produces z at every observation time t₁…t_N on a *continuous* timeline — no binning, so irregular sampling is native and I can extrapolate forward or backward past the data; a decoder emits each x_{t_i} from z_{t_i}; train by the ELBO. Because f is time-invariant, the whole latent trajectory is fixed by z(t₀). Binned RNNs had to discretize time and choke on missing data; here time is just the integration variable.

So the causal chain, end to end: a residual block is one Euler step of an ODE; take the continuous limit and let an adaptive black-box solver integrate dz/dt = f(z,t,θ), buying error control and per-input adaptive depth; to train without storing the trajectory or differentiating the solver's internals, define the row adjoint a(t)=∂L/∂z(t), derive its backward ODE da/dt = −a∂f/∂z from the instantaneous chain rule, augment with θ and t to get dL/dθ = −∫_{t₁}^{t₀} a∂f/∂θ dt and the endpoint gradients, compute every integrand as a single vector-Jacobian product through f, and recompute z backward alongside a so the whole thing is O(1) memory and linear time, black-box in the solver; then notice that the same continuous limit turns the change-of-variables log-determinant into a trace, ∂ log p/∂t = −tr(∂f/∂z), which is linear and lets flows be wide instead of deep and drops the bijectivity/partition constraints — and the one ODE engine powers continuous-depth supervised models, continuous normalizing flows, and continuous-time latent series alike.
