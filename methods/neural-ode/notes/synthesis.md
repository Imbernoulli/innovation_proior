# Neural ODE — synthesis notes (Phase 1.5)

## The pain point at the time (2017–2018)

Deep nets build complicated maps by **composing many small transformations** on a hidden state. The dominant primitive is the **residual block** (He et al. 2016):

  h_{t+1} = h_t + f(h_t, θ_t),  t ∈ {0…T}.

This works spectacularly (you can train 100+ layer nets), but it has structural costs:
- **Memory grows with depth.** Reverse-mode autodiff (backprop) must store every intermediate activation h_0…h_T to compute gradients. Memory is O(L) in the number of layers L. This is *the* bottleneck for training very deep models — you run out of GPU memory long before you run out of useful depth.
- **Depth is a discrete, hand-chosen hyperparameter.** You pick the number of blocks. Every input gets exactly the same amount of computation, regardless of how hard it is.
- **Each layer has its own parameters θ_t.** Adjacent layers in a ResNet are doing *almost the same thing* (small residual updates), yet their weights are untied.

Several groups had already noticed (Lu et al. 2017 "Beyond Finite Layer NN"; Haber & Ruthotto 2017 "Stable architectures"; Ruthotto & Haber 2018) that the residual update **looks exactly like one step of the forward Euler method** for an ODE:

  h_{t+1} = h_t + Δt · f(h_t, θ_t),  with Δt = 1.

That is the discretization of dh/dt = f(h, t, θ). So a ResNet is "a crude ODE solver with a fixed step size of 1, run for L steps." This reframing was in the air, used mostly to *analyze stability* and design reversible architectures. It had not been pushed to its logical end: **what if the model literally IS the ODE, and we hand the integration to a real, modern, adaptive solver?**

Goal: replace the discrete stack of layers with a continuous-depth model dh/dt = f(h,t,θ) whose forward pass is a black-box ODE solve — getting (a) **O(1) memory** training independent of "depth", (b) **adaptive computation** (the solver picks how many evaluations each input needs, and you can dial accuracy vs. speed at test time), (c) parameters tied across all of continuous depth. The central technical obstacle: **how do you backpropagate through a black-box adaptive ODE solver** without storing its internal steps (which would throw away the whole memory advantage and inject the solver's internal numerical error into the gradient)?

## Load-bearing ancestors

### Residual networks as Euler discretization (He et al. 2016; Lu et al. 2017; Haber & Ruthotto 2017)
- Core: h_{t+1} = h_t + f(h_t, θ_t). The "+h_t" identity skip is what lets gradients flow through very deep nets.
- The ODE reading: identify the layer index t with continuous time and the residual with a step: h_{t+1} − h_t = f(h_t,θ_t) ≈ dh/dt at step size 1. As you add more layers and shrink the step, the trajectory → the solution of dh/dt = f(h(t),t,θ).
- **Gap it leaves:** backprop still stores all activations (O(L) memory), depth is fixed and discrete, step size is hard-coded to 1 (Euler is the *worst*, lowest-order solver — 120+ years of numerical analysis since Runge 1895 / Kutta 1901 built far better adaptive high-order methods, none of which a fixed ResNet uses).

### Reversible residual nets (Gomez et al. 2017 RevNet; Chang et al. 2017; Haber & Ruthotto 2017)
- Motivation is the SAME memory problem: they make blocks **analytically invertible** so activations can be *recomputed* on the backward pass instead of stored → O(1) memory in depth.
- Mechanism: partition hidden units into two groups (x1,x2); y1 = x1 + F(x2); y2 = x2 + G(y1); invertible in closed form.
- **Gap:** requires a **restricted architecture** (the partition + coupling structure), constraining what f can be. We want O(1) memory for an *arbitrary* f with no architectural restriction.

### Adaptive ODE solvers (Runge 1895, Kutta 1901, Dormand–Prince; Hairer et al. 1987)
- Modern solvers (explicit Runge–Kutta like RK4/Dopri5, implicit Adams/BDF in LSODE/VODE) **monitor local truncation error** (e.g. compare a 4th- and 5th-order step, "embedded RK") and **adapt the step size** to keep the error under a user tolerance (rtol, atol). They evaluate f only where needed.
- This is exactly "adaptive computation" for free, with rigorous error guarantees — something the adaptive-computation-time line (Graves 2016; Figurnov et al. 2017) tried to learn with extra networks and extra parameters.
- **Why we treat the solver as a black box:** implicit methods (Adams/BDF) solve a nonlinear system at each step; direct backprop through those internal Newton iterations is a nightmare and ties you to one solver. We want gradients that work for *any* solver.

### Adjoint sensitivity method (Pontryagin 1962; LeCun 1988; Pearlmutter 1995)
- Classical optimal-control tool for differentiating the solution of an ODE w.r.t. its parameters/initial conditions by solving a **second ("adjoint") ODE backwards in time**. LeCun (1988) and Pearlmutter (1995) had *proposed* it for continuous-time nets but never demonstrated it practically at scale.
- **Why it's the right tool:** it gives gradients in O(state size) memory and linear time, treating the forward solver as a black box. Contrast with **forward sensitivity analysis** (used by Stan, Carpenter et al. 2015): forward sensitivities propagate ∂z/∂θ alongside z, which is **quadratic** in the number of variables; adjoint is **linear** (Zhang & Sandu 2014 FATODE).
- Contrast with **dolfin / dolfin-adjoint** (Farrell et al. 2013) and Stan: those compute adjoints by backpropagating through the *individual operations of the solver* — exactly what we want to avoid (kills the memory win, injects solver-internal error).

### Normalizing flows: NICE / RealNVP / planar flows (Dinh 2014; Dinh et al. 2016; Rezende & Mohamed 2015)
- Change of variables: z1 = f(z0), bijective ⇒ log p(z1) = log p(z0) − log|det ∂f/∂z0|.
- The **bottleneck is the log-determinant of the D×D Jacobian — O(D³)** in general. The whole flow literature is a fight against this cost:
  - **NICE / RealNVP coupling layers:** make the Jacobian triangular so det = product of diagonal (O(D)), but at the cost of only transforming half the dimensions per layer and needing careful partitioning/ordering of data dimensions.
  - **Planar flow** (Rezende & Mohamed 2015): z(t+1) = z(t) + u·h(wᵀz(t)+b), a single-hidden-unit perturbation, with det via the matrix-determinant lemma: |1 + uᵀ ∂h/∂z|. Cheap, but each layer is a **rank-1 / single-unit bottleneck** — to get expressiveness you must stack *many* one-unit layers (depth K). You can't make a planar layer "wide" (M hidden units) cheaply because a general M-unit layer's det is O(M³).
- **Gap:** flows are forced into the triangular-Jacobian / single-unit straitjacket, plus the artificial requirement of **partitioning or ordering dimensions**, purely to keep the determinant cheap.

## The continuous-depth limit (the starting move)

Take the residual recurrence and insert a step size Δt = 1/N, run N steps over t∈[0,1]:
  h_{t+Δt} = h_t + Δt · f(h_t, t, θ).
As N→∞ this is the forward-Euler discretization of the **initial value problem (IVP)**
  dh(t)/dt = f(h(t), t, θ),  h(t_0) = input.
The output is the solution at t_1: h(t_1) = h(t_0) + ∫_{t_0}^{t_1} f(h(t),t,θ) dt = ODESolve(h(t_0), f, t_0, t_1, θ).
Now hand the integral to a black-box adaptive solver (Dopri5, Adams/BDF). The number of f-evaluations the solver chooses, call it L̃, is the "implicit depth." Parameters θ are **shared across all of continuous time** (one f, not one per layer). Picard's existence/uniqueness theorem (Coddington & Levinson 1955) guarantees a unique solution when f is uniformly Lipschitz in h and continuous in t — true for finite-weight nets with tanh/relu.

## Core derivation 1 — the adjoint method (the heart)

Setup: loss L(z(t_1)) = L(ODESolve(z(t_0), f, t_0, t_1, θ)). Need dL/dθ, dL/dz(t_0), and (bonus) dL/dt_0, dL/dt_1.

Define the **adjoint** a(t) := ∂L/∂z(t) — sensitivity of the loss to the state at instant t.

**Claim: da(t)/dt = −a(t)ᵀ ∂f/∂z (column convention; row convention drops the transpose).**

Proof (instantaneous analog of the chain rule). In a discrete net, dL/dh_t = dL/dh_{t+1} · dh_{t+1}/dh_t. For the continuous state, the map over an ε step is
  z(t+ε) = z(t) + ∫_t^{t+ε} f dt = T_ε(z(t)).
Chain rule: a(t) = a(t+ε) ∂T_ε(z(t))/∂z(t). Then by definition of derivative (use row vectors to match the appendix; transpose for column form):
  da/dt = lim_{ε→0+} [a(t+ε) − a(t)]/ε
        = lim [a(t+ε) − a(t+ε) ∂T_ε/∂z]/ε                    (chain rule)
        = lim [a(t+ε) − a(t+ε)(I + ε ∂f/∂z + O(ε²))]/ε        (Taylor: T_ε = z + εf + O(ε²) ⇒ ∂T_ε/∂z = I + ε∂f/∂z + O(ε²))
        = lim [ −ε a(t+ε) ∂f/∂z + O(ε²)]/ε
        = lim [ −a(t+ε) ∂f/∂z + O(ε)]
        = −a(t) ∂f/∂z.
So **da/dt = −a(t)ᵀ ∂f/∂z**. Like backprop, this runs **backwards in time**, with terminal condition a(t_1) = ∂L/∂z(t_1) (the only thing we're handed). Integrate from t_1 down to t_0 to get a(t_0) = ∂L/∂z(t_0).

One snag: the adjoint ODE needs z(t) along the whole trajectory, but we threw it away (that was the point — O(1) memory). Fix: **recompute z(t) backwards in time** by appending dz/dt = f to the backward solve, starting from the known final z(t_1). So the augmented backward state carries (z, a) and we never stored anything.

**Parameter gradient.** Treat θ as part of the state with trivial dynamics dθ/dt = 0, and t likewise with dt/dt = 1. Augmented state s = [z, θ, t]; augmented dynamics f_aug = [f, 0, 1]. Its Jacobian is the block matrix
  ∂f_aug/∂[z,θ,t] = [[∂f/∂z, ∂f/∂θ, ∂f/∂t],[0,0,0],[0,0,0]].
The augmented adjoint a_aug = [a, a_θ, a_t] with a_θ = dL/dθ, a_t = dL/dt obeys the same law da_aug/dt = −a_aug ∂f_aug/∂s, which componentwise gives:
  da/dt   = −a ∂f/∂z   (recovers the state adjoint),
  da_θ/dt = −a ∂f/∂θ,
  da_t/dt = −a ∂f/∂t.
θ is constant in t, so its **total** gradient is the integral of a_θ's dynamics with terminal condition a_θ(t_1)=0:
  **dL/dθ = a_θ(t_0) = −∫_{t_1}^{t_0} a(t)ᵀ ∂f/∂θ dt** = +∫_{t_0}^{t_1} a(t)ᵀ ∂f/∂θ dt (note the reversed limits flip the sign).
Time endpoints: dL/dt_1 = a(t_1)ᵀ f(z(t_1),t_1,θ); dL/dt_0 = a_t(t_0) = −∫_{t_1}^{t_0} a(t)ᵀ ∂f/∂t dt.

**Putting it in one solver call (Algorithm 1).** Concatenate [z, a, ∂L/∂θ-accumulator (, a_t)] into ONE augmented vector and integrate it backward from t_1 to t_0 in a single ODESolve. The augmented dynamics returns
  [ f(z,t,θ),  −a(t)ᵀ∂f/∂z,  −a(t)ᵀ∂f/∂θ (, −a(t)ᵀ∂f/∂t) ].
Initial augmented state at t_1: [z(t_1), ∂L/∂z(t_1), 0_{|θ|}]. Out comes [z(t_0), dL/dz(t_0), dL/dθ].

**Why the vector-Jacobian products are cheap.** We never form ∂f/∂z (D×D) or ∂f/∂θ explicitly. The dynamics only needs the *row-vector × Jacobian* products aᵀ∂f/∂z and aᵀ∂f/∂θ — these are exactly **vector-Jacobian products**, which reverse-mode autodiff computes in one backward pass through f at cost ~ one evaluation of f. So one VJP through f gives ALL of aᵀ∂f/∂z, aᵀ∂f/∂θ, aᵀ∂f/∂t at once (different "input" slots of the same VJP). This is why the method is linear-time, not quadratic like forward sensitivity.

**Multiple observation times.** If L depends on z at several times t_1…t_N (e.g. a time series), break the backward solve into intervals [t_i, t_{i-1}], and at each observation **add** ∂L/∂z(t_i) into the adjoint before continuing.

**Memory & a subtlety.** Memory is O(1) in depth (nothing stored from the forward pass; z recomputed backward). Risk: reverse-time reconstruction of z can drift if the trajectory is unstable; fix by **checkpointing** a few z values on the forward pass and re-integrating between them. In practice (default tolerances) the drift was negligible. Empirically the backward solve used ~half the function evaluations of the forward — so adjoint is both more memory- and compute-efficient than backprop-through-solver-steps (which must differentiate every forward evaluation).

## Core derivation 2 — instantaneous change of variables (CNF)

For a continuous transformation dz/dt = f(z(t),t), the log-density obeys an ODE too, and the determinant collapses to a **trace**:

**∂ log p(z(t))/∂t = −tr(∂f/∂z).**

Proof (infinitesimal limit of the discrete change-of-variables). Let T_ε(z(t)) = z(t+ε). Discrete CoV: log p(z(t+ε)) = log p(z(t)) − log|det ∂T_ε/∂z|. So
  ∂log p/∂t = lim_{ε→0+} [log p(z(t)) − log|det ∂T_ε/∂z| − log p(z(t))]/ε
            = −lim log|det ∂T_ε/∂z| / ε.
At ε=0, T_0 = identity ⇒ det ∂T_0/∂z = 1 ⇒ the numerator → 0, denominator → 0: apply **L'Hôpital in ε**:
            = −lim [ ∂/∂ε log|det ∂T_ε/∂z| ] / [∂/∂ε ε]
            = −lim [ (∂/∂ε |det ∂T_ε/∂z|) / |det ∂T_ε/∂z| ].
As ε→0, |det ∂T_ε/∂z| → 1, so the denominator → 1. By **Jacobi's formula** ∂/∂ε det A(ε) = tr(adj(A) ∂A/∂ε), and adj(∂T_0/∂z) = adj(I) = I:
            = −tr( lim_{ε→0} ∂/∂ε ∂T_ε/∂z ).
Now Taylor-expand T_ε = z + εf(z,t) + O(ε²) ⇒ ∂T_ε/∂z = I + ε ∂f/∂z + O(ε²) ⇒ ∂/∂ε of that = ∂f/∂z + O(ε) → ∂f/∂z. Hence
  **∂ log p(z(t))/∂t = −tr(∂f/∂z).**

**Why this is huge.** The finite-flow cost was the **O(D³) log-determinant**; here it's a **trace** of the Jacobian, which is **O(D²) to assemble (or cheaper)** and, crucially, **linear**: tr(Σ_n J_n) = Σ_n tr(J_n). So a "wide" continuous flow dz/dt = Σ_{n=1}^M f_n(z) costs only **O(M)** in the number of hidden units, vs O(M³) for a standard wide NF layer — which is exactly why standard NFs were forced into stacks of single-unit (planar) layers. Continuous flows can be **wide instead of deep**.

**No bijectivity constraint needed.** f need not be designed bijective: Picard uniqueness already makes the *flow map* a bijection automatically. So we drop the partition/ordering-of-dimensions machinery (the coupling layers) entirely.

**Planar CNF (the concrete instance used).** dz/dt = u·h(wᵀz + b). Then ∂f/∂z = u (∂h/∂z)ᵀ is an outer product, and tr(outer product) = inner product:
  ∂log p/∂t = −tr(u (∂h/∂z)ᵀ) = −uᵀ ∂h/∂z.
Time-dependent params via a small hypernetwork θ(t) (Ha et al. 2016) and a per-unit gate σ_n(t)∈(0,1): dz/dt = Σ_n σ_n(t) f_n(z). Solve the (D+1)-dim ODE for [z, log p] jointly. Train by maximum likelihood (the flow is reversible at ~equal forward/backward cost, unlike discrete NFs), or by minimizing KL to a target.

Connection to PDEs (sanity check): the instantaneous CoV is the **Liouville equation** (zero-diffusion Fokker–Planck) followed along a particle's trajectory. The total derivative dp(z(t),t)/dt has a "fixed-point" Liouville term and the transport term; following the particle cancels the transport piece and leaves −Σ_i ∂f_i/∂z_i · p, i.e. ∂log p/∂t = −Σ_i ∂f_i/∂z_i = −tr(∂f/∂z). The win over solving Liouville directly: Liouville needs a grid exponential in D; the trace-ODE needs only D extra state following one trajectory.

## Other instantiation — latent ODE for time series
VAE with: RNN encoder (run backwards over the series) → q(z_{t_0}) → sample z_{t_0} → ODESolve gives z_{t_1…t_N} on a *continuous* timeline (handles irregular sampling natively, unlike binned RNNs) → decoder p(x_{t_i}|z_{t_i}). f is time-invariant so the whole trajectory is determined by z_{t_0}; can extrapolate forward/backward. Optional inhomogeneous-Poisson-process likelihood on event *times*: log p(t_1…t_N) = Σ log λ(z(t_i)) − ∫ λ(z(t)) dt, computed in the same ODE solve. Train by ELBO.

## Design-decision → why table

| Decision | Why this | Rejected alternative & failure |
|---|---|---|
| Continuous limit dh/dt=f(h,t,θ) of the residual recurrence | ResNet update = Euler step at Δt=1; shrink the step → ODE | keep discrete stack: O(L) memory, fixed depth, Euler is lowest-order |
| Hand the forward pass to a black-box adaptive solver | error-controlled, adapts evals per input, trade speed/accuracy at test time | fixed Euler (ResNet): no error control, no adaptivity |
| Treat the solver as a black box (don't backprop its steps) | works for ANY solver incl. implicit Adams/BDF; keeps O(1) memory; no solver-internal error in grad | dolfin/Stan-style backprop-through-operations: O(L) memory, solver-specific, injects internal error |
| Adjoint sensitivity method for gradients | O(1) memory in depth; linear time; black-box | forward sensitivity (Stan): quadratic in #vars; reversible nets: restricted architecture |
| Recompute z(t) backward alongside the adjoint | don't store forward activations → constant memory | store all z(t): O(L) memory, the thing we're killing |
| Augment state with θ (dθ/dt=0) and t (dt/dt=1) | one uniform adjoint law yields dL/dθ, dL/dt_0, dL/dt_1 together | separate bespoke derivations per input |
| Concatenate [z,a,∂L/∂θ] into one backward ODESolve | all gradients in a single solver call | multiple solves, more bookkeeping |
| Use vector-Jacobian products (reverse-mode AD on f) | never form D×D Jacobian; aᵀ∂f/∂{z,θ,t} all from one VJP ≈ cost of one f-eval | materialize ∂f/∂z: O(D²) memory, forward-mode = quadratic time |
| Terminal condition a(t_1)=∂L/∂z(t_1), a_θ(t_1)=0, integrate backward | matches backprop's "gradient flows from the output back" | forward integration of adjoint: wrong direction, needs unknown a(t_0) |
| Checkpoint z occasionally if reverse drift | bound reconstruction error on unstable trajectories | none needed at default tol in practice |
| Continuous normalizing flow: trace instead of det | tr(∂f/∂z) replaces O(D³) log|det|; trace is linear → wide flows O(M) | discrete NF: O(D³) det forces triangular Jacobians / single-unit layers |
| Drop the coupling/partition-ordering machinery | Picard uniqueness makes the flow map bijective for free; f need not be bijective | NICE/RealNVP: must partition & order dims to keep det cheap |
| Planar CNF f = u·h(wᵀz+b) | ∂f/∂z is rank-1 outer product ⇒ tr = uᵀ∂h/∂z, trivial | a general wide layer's det/trace assembly is costlier; planar is the cleanest instance |
| Time-dependent f via hypernetwork + per-unit gates σ_n(t) | let the flow turn dynamics on/off over t; more expressive without more det cost | static f: less expressive over the time axis |
| Latent ODE: ODESolve between observation times in a VAE | native continuous time → irregular sampling, extrapolation | binned RNN: discretization artifacts, missing-data trouble |
| Tolerances rtol/atol as the knob | one principled dial for compute vs. accuracy, incl. lower tol at test time | learned adaptive-computation nets (Graves/Figurnov): extra params, train+test overhead |

## Code grounding (canonical: rtqichen/torchdiffeq)
- `odeint(func, y0, t, rtol, atol, method)` — black-box solver; `func` is an `nn.Module` with signature `forward(t, y)` returning dy/dt. Fixed-grid `Euler/Midpoint/RK4` and adaptive `dopri5`.
- `odeint_adjoint` = `OdeintAdjointMethod(torch.autograd.Function)`: forward runs `odeint` under `no_grad` and saves only (t, y, params); backward builds `augmented_dynamics(t, y_aug)` where `y_aug=[vjp_t, y, vjp_y, *vjp_params]`, computes the VJPs with `torch.autograd.grad(func_eval, (t,y)+params, -adj_y)`, and integrates the augmented system **backward** between consecutive output times, adding `grad_y[i-1]` to the state-adjoint at each observation. Matches Algorithm 1/2 exactly (vjp_t ↔ a_t, vjp_y ↔ a, vjp_params ↔ a_θ; the `−adj_y` is the minus sign in da/dt).
- Supervised model (`examples/odenet_mnist.py`): `ODEfunc(t,x)` (groupnorm→relu→ConcatConv2d(t)→… ), wrapped in `ODEBlock` that calls `odeint(self.odefunc, x, [0,1], rtol=tol, atol=tol)` and returns `out[1]`; `nfe` counter shows adaptive depth. `ConcatConv2d` appends t as a channel so f depends on time.
- Toy dynamics (`examples/ode_demo.py`): `ODEFunc` = small MLP on `y**3`, trained with `pred=odeint(func,y0,t)`, L1 loss, RMSprop.
- Planar CNF: integrate the (D+1) state [z, logp] with dz/dt = u·h(wᵀz+b), dlogp/dt = −uᵀ∂h/∂z; for MLE reverse the flow to sample. (FFJORD-style trace; here exact trace via the rank-1 planar form.)
- Final answer/reasoning code: clean PyTorch mirroring these — an `ODEF` nn.Module, an RK4 + adaptive `odeint`, an `odeint_adjoint` autograd.Function doing the augmented backward solve with VJPs, an `ODEBlock`, and a planar CNF with the trace term.
</content>
