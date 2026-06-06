OK, let me think this through from scratch. I have a metal AM process — a laser tracing a toolpath, melting and fusing powder layer by layer — and I have a thermal simulation of it that, given the process parameters, predicts the temperature at every node and every time step. The thermal history is the thing that matters: cooling rates set microstructure, the integrated thermal load sets residual stress and warping, and the melt-pool depth sets whether I'm fusing properly and getting the geometry right. So far so good. But I don't want to *predict* the thermal history for a given recipe — I want to *design* the recipe to produce a thermal history I choose. That's the inverse problem, and that's where it gets hard.

Let me be concrete about what "the recipe" is. In the easy version it's a handful of scalars: a constant laser power, beam radius, maybe some material properties I don't know exactly and want to calibrate. In the hard version it's a *function of time* — the laser power P(t) over the whole build. And the build, with an explicit thermal solver stepping at a small Δt, is tens of thousands of steps long. If I want P(t) to be freely shaped over the build — turn it down when heat is piling up at the thin neck of the part, turn it up over the thick sections — then I have on the order of 20,000 free numbers. The design space is enormous and temporal.

So how would I optimize that? I have a loss: mean-squared error between the thermal history the current recipe produces and the target thermal history I want. Minimize it. The obvious thing is gradient descent. But to do gradient descent I need ∂(loss)/∂(each parameter). How do I get that gradient from a simulator?

The brute-force answer is finite differences: nudge one parameter by δ, re-run the *entire* simulation, see how the loss changed, divide by δ. That gives me one component of the gradient. For the next component I nudge a different parameter and re-run again. So the cost of one full gradient is (number of parameters) × (cost of one simulation). And one simulation of a realistic part is *minutes*. For the static case with five scalars, fine, that's five extra sims per step, annoying but survivable. For the time-series case with 20,000 numbers, that's 20,000 full simulations to get a single gradient. At minutes apiece that's not an optimization loop, it's a geological process. Dead on arrival. And it's not a quirk of finite differences — any derivative-free optimizer (pattern search, evolutionary, Bayesian) needs a swarm of forward evaluations and chokes the same way once I'm in thousands of dimensions. The wall is: the cost of *information about the gradient* scales with the number of parameters, and I have far too many parameters.

Let me stare at that scaling, because the whole problem lives there. The cost of FD is O(P) simulations for P parameters. Why O(P)? Because I'm probing the function one input direction at a time. Is there any way to get the sensitivity to *all* inputs without probing each one separately?

There is, and I already use it every day to train neural networks: backpropagation. Backprop computes the gradient of one scalar loss with respect to *all* the millions of weights in a single backward pass, not one pass per weight. That's not magic, it's reverse-mode automatic differentiation, and the reason it's cheap is worth being precise about. Think of the computation as a graph of elementary operations, each with a known local derivative. Forward-mode AD pushes a perturbation in *one input* forward through the graph — to get the gradient w.r.t. P inputs I'd need P forward sweeps, exactly the O(P) cost again, the same wall as finite differences but exact. Reverse-mode instead starts from the *one output* and pulls its sensitivity *backward*, accumulating for every intermediate variable an adjoint — the derivative of the loss with respect to that variable. One backward traversal touches every parameter on the way and hands me the *entire* gradient. The cost is a small constant times one forward evaluation, and — this is the key — it's *independent of the number of inputs*.

So which mode do I want? My loss is one scalar; my parameters number in the thousands. That's a function from R^P to R with P huge and the output dimension 1. Forward mode costs O(P); reverse mode costs O(1) in the parameter count. Reverse mode wins by a factor of ~P. The map from "many inputs, few outputs" to "reverse mode is the cheap one" is exact, and my problem sits squarely in the regime where reverse mode is cheap. That's the lever.

The simulator is *also* just a composition of elementary operations — assemble matrices, multiply, add fluxes, step forward in time. There's nothing special about it that exempts it from AD. If I implement the thermal solver inside an automatic-differentiation framework — every field a differentiable tensor, every operation one whose local derivative is known — then reverse-mode AD will give me ∂(loss)/∂(all parameters) in essentially one backward pass *through the simulation itself*, no matter how many parameters there are. I don't optimize against a black box anymore; I make the box transparent. Make the *simulator itself* differentiable.

And notice what reverse-mode AD through the time-stepping solver actually *is*, mathematically. The explicit update is a recurrence, T^{n+1} = T^n + Δt M⁻¹[R_G − R_F − R_C − R_R − K T^n]. Differentiating the final loss back through that recurrence, step by step from the last step to the first, accumulating sensitivities — that's precisely the discrete adjoint of the solver. People derive adjoint equations of PDEs by hand to get exactly this O(1)-in-parameters gradient. But reverse-mode AD *constructs the discrete adjoint automatically*, for whatever objective and whatever parameterization I write down, with no bespoke derivation. I get the adjoint for free, and I get it for arbitrary downstream computation — if I put a neural net in front of the laser power, or compute some derived feature like melt-pool depth and put the loss on *that*, the chain rule just flows through it. That composability is the thing the hand-derived continuous adjoint can't give me cheaply, because every change of objective or parameterization means re-deriving by hand.

Good. But "make the time-stepping solver differentiable and backprop through it" has a cost I shouldn't wave away, and it's not compute — it's memory. The backward pass evaluates each operation's local derivative *at the value that operation produced on the forward pass*. For a recurrence of N time steps, the adjoint of step n needs the state T^n that the forward pass computed. So I have to keep all of them around — every temperature field at every one of ~20,000 steps. That's O(N) memory, and for a mesh of any size it's a lot. This is the same backprop-through-time memory problem as training a very deep recurrent net.

Is O(N) memory acceptable, or do I need to fight it? Two thoughts. First, if I'm careful about *how* the AD framework tapes things, I can avoid taping intermediate buffers I don't need: if all the state already lives in named global tensors, the "tape" only has to record which kernels ran and their scalar arguments, and replay their gradient kernels in reverse — it doesn't need to snapshot every intermediate array, because the arrays are already there. That keeps the bookkeeping light, though the temperature history itself is still O(N). Second, if O(N) on the history is too much, there's the standard trade: checkpointing. Don't store every step; store the state only at segment boundaries S steps apart, and on the backward pass, when I need the states inside a segment, recompute them forward from the boundary checkpoint. Store O(N/S) checkpoints plus O(S) live states within the segment I'm currently reversing → O(S + N/S) memory, minimized at S = √N → O(√N), and the extra cost is just one re-simulation, so time stays O(N). That gives me a known release valve, and for the test part the history fits.

There's a sharper problem than memory, and it's about whether the gradients even *exist and carry signal*. Gradient-based optimization needs the loss to be a smooth function of the parameters with non-degenerate gradients. An AM simulation is full of things that are *not* smooth.

The worst offenders are the discontinuities from material deposition. Material is added by *element birth*: an element (and its newly-exposed surfaces) pops into the active mesh at a discrete time. The set of active elements, and the set of free surfaces exchanging heat with the air, jumps from one step to the next. If I tried to differentiate the loss with respect to *when* an element is born, or with respect to a geometric quantity that controls that birth, I'd be differentiating across a discrete jump — the gradient is either zero or undefined, and useless. Same with the toolpath geometry: it's a discrete, hard-coded pattern.

So I have a choice to make about *what* I let the gradient flow through. I freeze the geometric and boundary-related discontinuities. Precompute the element-birth and surface-birth schedule from the toolpath once, hold it fixed, and gate every assembly with a simple "is this element/surface born yet?" test — `if birth ≤ t·Δt`. That test is a discrete condition, but I'm *not optimizing through it*; it's a fixed mask. The gradient then only ever flows through the *continuous* quantities: the material properties (c_p, conductivity, convection coefficient) and the process parameters (laser power, beam radius). My claim is that, with the discrete scaffolding frozen, those continuous parameters are the real drivers of the thermal response, and they're exactly the ones I want to design anyway. The toolpath and geometry stay fixed; I optimize the power schedule and properties. That's the deal that makes the whole thing differentiable in practice.

There's a second, subtler gradient hazard, and it's about step-like *functions* even when nothing discontinuous is happening in the mesh. Take the most natural definition of melt-pool depth: scan downward and report the deepest node whose temperature exceeds the melting point. As a function of the laser power, that depth is *piecewise constant* — nudge the power a little and the deepest-melted node usually doesn't change at all, then suddenly jumps to the next node down. It's differentiable almost everywhere, sure, but its derivative is *zero* almost everywhere and undefined at the jumps. Backprop through it would deliver a gradient of zero and the optimizer would sit dead in the water. A valid-but-useless gradient is its own kind of wall.

So if I want to optimize melt-pool depth, I can't use the discrete "deepest molten node" definition; I need a *continuous* surrogate for depth that moves smoothly when the temperature field moves. Let me build one. The melt pool is where the temperature crosses the melting isotherm, so depth is really "how far down does the melting-temperature crossing go, directly under the laser." Make that continuous: at several height levels below the build surface, find the temperature *right under the beam* by interpolating from a small local stencil fixed by the mesh and toolpath — the nine nearby nodes around the laser footprint give weights for the value at the beam location. Now I have a smooth temperature-versus-depth profile under the laser. Then find the depth at which that profile crosses the melting temperature by a *pairwise linear* interpolation between adjacent levels — a linear blend, which is continuous and has a well-defined nonzero derivative. That gives a melt-pool depth that responds smoothly to power and produces a usable gradient. Same physical quantity, a parameterization chosen specifically so the gradient survives.

One more saturation worry while I'm here. I'm going to want to parameterize a *time-series* control, and the natural tool is a small neural network mapping time → laser power, with tanh nonlinearities. tanh has fine gradients in isolation, but stacked and driven hard, repeated tanh saturates — outputs pin near ±1 where the slope is ~0 — and the chain of local derivatives collapses to nearly zero, the vanishing-gradient problem. I'll keep the network shallow and lean on an optimizer with momentum and per-coordinate step scaling so that even small, uneven gradients still make progress. And there's a hard temperature clamp in the integrator (cap T at some max so a runaway element can't blow up the sim); above the cap the gradient is killed, which I'll accept as a deliberate stability measure rather than fight.

Alright, the physics I'm differentiating through. The transient heat equation is

  ρ c_p ∂T/∂t − ∇·(k∇T) − s = 0,

with the boundary conditions I care about: a fixed-temperature Dirichlet condition at the base, a prescribed laser flux on the top surface, convection q = −h(T − T_amb), and radiation q = −εσ(T⁴ − T_amb⁴) on the exposed surfaces. Take the weak form, discretize over hexahedral elements with shape functions N and their spatial derivatives B, assemble the global capacitance matrix [M] and conduction matrix [K] from per-element contributions (each integrated over the element's Gauss points with the Jacobian of the isoparametric map), and assemble the load vectors: the laser flux R_F, convection R_C, radiation R_R, and any internal generation R_G. The semi-discrete system M Ṫ + K T = R, integrated *explicitly* in time, gives

  {T^{n+1}} = {T^n} + Δt [M]⁻¹ [ {R_G} − {R_F} − {R_C} − {R_R} − [K]{T^n} ].

Why explicit, and why lump the mass matrix? Because [M]⁻¹ is the expensive part of any thermal step — a linear solve. If I *lump* the capacitance matrix, row-summing it to a diagonal, then [M]⁻¹ is just elementwise division by the nodal masses: the whole update is local, embarrassingly parallel, and — what I care about most here — a trivially differentiable sequence of multiplies, adds, and divides with no implicit solve hiding inside. An implicit step would bury a linear-solve-of-a-changing-matrix in the graph, which is differentiable but a headache to make efficient; the explicit lumped step keeps the entire forward pass as plain arithmetic that AD eats happily. The price is a small Δt for stability, which is why I have ~20,000 steps — and that's exactly why I needed the O(1)-in-parameters reverse-mode gradient in the first place. It all hangs together.

How does the laser enter? As the flux R_F on the active top surfaces. I model it as a moving surface Gaussian centered at the current beam location: at a surface integration point a distance d from the beam center, the flux is

  q_s(x,t) = (3 Q P(t) on(t)) / (π r_b²) · exp( −3 ||x − x_L(t)||² / r_b² ),

where Q is the nominal power scale, P(t) the normalized commanded power from my control, on(t) the laser on/off state from the toolpath, x_L(t) the beam center, and r_b the beam radius. The factor 3 in the exponent concentrates about 95% of the energy within the spot radius, and 3/(πr_b²) makes the surface integral equal to QP(t)on(t). In the weak-form sign convention R_F is an outward prescribed-flux vector; the incoming laser has outward flux −q_s, so subtracting R_F in the update becomes a positive heat-input contribution in the assembled right-hand side. Every term here is smooth in Q, P, and r_b, so the gradient flows straight back through the heat input into my control parameters. That's the whole point: P(t) is differentiable, and it's where my design knob lives.

Convection and radiation are evaluated on the exposed surfaces from the interpolated surface temperature: q_conv = −h(T_ip − T_amb), q_rad = −εσ(T_ip⁴ − T_amb⁴), scattered back to the surface nodes through the shape functions and surface Jacobian. The T⁴ in radiation is smooth — good, AD handles it. And I'll fold latent heat into the specific heat: in the mushy band between solidus and liquidus, add L/(T_liq − T_solid) to c_p, so melting absorbs energy over that interval. That's a temperature-dependent c_p; the dependence is mild and continuous across the band, so it differentiates fine — though I have to be careful that the *branch* selecting "am I in the mushy band" doesn't itself become a gradient-stopping step; it's a property-magnitude change, not a control I optimize through, so I treat it as a forward-only condition.

Now, *which* AD framework do I build this in? This matters more than I'd like, because FE assembly is not what mainstream deep-learning AD libraries are built for. Their fast path is dense, regular, element-wise tensor algebra — convolutions, big matmuls. But FE assembly is the opposite: I gather a handful of nodal values per element by *arbitrary indexing* (the element's node list), do small dense element computations, and *scatter-accumulate* the results into global vectors with *atomic adds* — and I do this over a *dynamically changing* active-element set. Random indexing, large-scale atomics, dynamic domains. Those are exactly the operations the array-programming libraries express awkwardly and execute slowly, because each becomes a separate kernel launch and a separate buffer allocation rather than fusing.

So the choice should turn on the data-access pattern, not on which library is most familiar. A tensor library can express many pieces, but the assembly pattern wants arbitrary indexing and scatter accumulation rather than dense batched algebra; a tracing-JIT functional library is awkward for the same reason; and a framework without flexible matrix-assembly indexing cannot express the solver naturally. What I want is a system designed around *imperative, parallel, flexibly-indexed* kernels with native atomic accumulation — one that fuses a whole simulation stage into a single kernel so arithmetic intensity stays high, *and* that does reverse-mode AD by source-transforming those kernels and replaying their adjoints over a lightweight tape. The deciding requirement is the data-access pattern of FE assembly, and it points away from the DL-array libraries and toward an imperative differentiable-programming substrate. There's a subtlety it imposes that I have to respect: to make AD well-defined under in-place writes, a global tensor element, once written, may only be written again by atomic *accumulation*, and isn't read until its accumulation is done. That's not a burden — it's literally the assembly pattern (zero the global vectors, then atomic-add every element's contribution, then read for the time step). So my forward code naturally satisfies it.

Let me now think about the parameterizations for the three things I actually want to do, because each tests a different part of this machine.

First, parameter inference from partial data. I treat a handful of static scalars — heat capacity, conductivity, convection coefficient, a static laser power, beam radius — as differentiable leaves, run the sim, and put an MSE loss between the simulated and a *target* thermal response, but only on the nodes I can "see," say the top build layer (as if an IR camera observed only the surface). Backprop gives ∂loss/∂(each scalar) in one pass; Adam updates them. This is calibration: infer hidden material/process parameters from a partial observation. I expect it to work, but I should be honest that the inverse problem is ill-posed — a lower heat capacity and a higher laser power can produce nearly the same thermal response, so the parameters are coupled and individually non-identifiable. What I should care about is that they all move in the right *direction* to collectively drive the partial-observation loss toward zero, not that each scalar lands exactly on its true value. And this case is a stress test of the AD plumbing: those five scalars touch matrix assembly, mass lumping, the Gaussian distribution calculation, the temporal mapping — so if the gradients are clean here, the machinery is sound across all the critical FE operations.

Second, designing the *time-series* thermal history. Now I want the full power schedule P(t) over the build, ~20,000 values. I could make each step's power an independent free variable, but instead I'll parameterize P(t) by a small fully-connected network: input the (normalized) time, two hidden layers of ~50 tanh units, output a scalar squeezed to [0,1] and scaled to the power range (0–1000 W). Why a network instead of 20,000 free knobs? Because the network is a compact, smooth function approximator — it produces a *coherent* schedule with far fewer effective degrees of freedom, regularizes the control to be reasonably smooth in time, and folds the whole time-series into a few thousand weights that AD handles as one object. It's also the clean way to *integrate* a data-driven model with the physics simulator: the network's output feeds the differentiable sim, the sim's loss backpropagates into the network's weights, and the chain rule handles the whole composition directly. The loss is MSE between the achieved thermal history and a target one (I generate the target by running the sim once with some deliberately complex power pattern, then throw that pattern away and ask the optimizer to recover a matching history from scratch). Adam on the network weights.

Third, melt-pool stabilization — the real prize. Same network-parameterized P(t), but now the loss penalizes the *continuous melt-pool depth* (the smooth surrogate I built above) deviating from a target depth at every step. Physically this is the useful control: as the laser builds up the hourglass and reaches the thin neck, heat accumulates and the pool would deepen uncontrollably; I want the schedule to back the power off through the neck and ramp it up again over the thick top where fresh cold material is being added. Because the depth surrogate is differentiable, the loss backpropagates through the depth extraction, through the thermal recurrence, into the network. This is the case that would have been hopeless with finite differences — thousands of time-series degrees of freedom — and is routine once the simulator is differentiable.

Let me put the loop together. Allocate every state and parameter as a differentiable global field. The forward simulation, per step, is: clear the global mass/rhs vectors; evaluate the control network to get this step's power; update the (latent-heat-dependent) c_p; assemble the lumped mass over active elements; assemble conduction and accumulate −K·T^n into the rhs; add the moving-Gaussian laser flux; add convection and radiation; explicit-integrate to get T^{n+1}. After the last step, compute the scalar loss. Wrap the whole forward simulation in the reverse-mode tape so that, on exit, the adjoints of every parameter are populated; then Adam-step the parameters. Repeat.

```python
import numpy as np
import taichi as ti

# ---- scales / properties (stainless steel) ----
ambient_init = 300.0
density      = 0.03
cp_init      = 0.5
cond_init    = 0.01
Qin_init     = 250.0
r_beam_init  = 1.0
h_conv_init  = 0.00005
h_rad        = 0.2
solidus, liquidus = 1533.15, 1609.15
latent_cp = 272.0 / (liquidus - solidus)  # L/(T_liq - T_sol) folded into c_p
max_temp = 2000.0
pi = 3.141592653589793

# ---- control network: time -> normalized laser power in [0,1] ----
n_input, n_hidden_1, n_hidden_2, n_hidden_3 = 3, 50, 50, 1
learning_rate, beta_1, beta_2, epsilon = 1e-2, 0.9, 0.999, 1e-7
loss_kind = "temperature_history"   # or "melt_pool" for the smooth-depth objective

ti.init()
# every state/parameter is a differentiable field; reverse-mode fills .grad
temperature = ti.field(float, (steps, nn), needs_grad=True)   # full history -> O(N) memory (checkpointable)
m_vec = ti.field(float, nn, needs_grad=True)                   # lumped capacitance (diagonal)
rhs   = ti.field(float, nn, needs_grad=True)
# static differentiable parameters (case I): leaves of the graph
ambient = ti.field(float, (), needs_grad=True)
cp   = ti.Vector.field(8, float, nel, needs_grad=True)
cond = ti.Vector.field(8, float, nel, needs_grad=True)
Qin  = ti.field(float, (), needs_grad=True)
r_beam = ti.field(float, (), needs_grad=True)
h_conv = ti.field(float, (), needs_grad=True)
loss = ti.field(float, (), needs_grad=True)
# control-network weights (cases II/III): also differentiable leaves
weight1 = ti.field(float, (n_hidden_1, n_input),  needs_grad=True); bias1 = ti.field(float, n_hidden_1, needs_grad=True)
weight2 = ti.field(float, (n_hidden_2, n_hidden_1), needs_grad=True); bias2 = ti.field(float, n_hidden_2, needs_grad=True)
weight3 = ti.field(float, (n_hidden_3, n_hidden_2), needs_grad=True); bias3 = ti.field(float, n_hidden_3, needs_grad=True)
output1 = ti.field(float, (steps, n_hidden_1), needs_grad=True)
output2 = ti.field(float, (steps, n_hidden_2), needs_grad=True)
output3 = ti.field(float, (steps, n_hidden_3), needs_grad=True)   # normalized power P(t)
melt_depth = ti.field(float, steps, needs_grad=True)              # smooth depth proxy

# ---- the control network: P(t) from a tiny MLP of (normalized) time ----
@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden_1):
        act = 0.0
        for j in ti.static(range(n_input)):
            act += weight1[i, j] * (((t / steps) - 0.5))**j   # time features, centered
        output1[t, i] = ti.tanh(act + bias1[i])
@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_hidden_2):
        act = 0.0
        for j in ti.static(range(n_hidden_1)):
            act += weight2[i, j] * output1[t, j]
        output2[t, i] = ti.tanh(act + bias2[i])
@ti.kernel
def nn3(t: ti.i32):
    for i in range(1):
        act = 0.0
        for j in ti.static(range(n_hidden_2)):
            act += weight3[i, j] * output2[t, j]
        output3[t, i] = (ti.tanh(act + bias3[i]) + 1) / 2.0     # squeeze to [0,1]

# ---- temperature-dependent c_p: latent heat across the mushy band ----
@ti.func
def calc_cp(t, el_id, solidus, liquidus, latent_cp):
    cp_el = ti.Vector.zero(float, 8)
    for ip in ti.static(range(8)):
        theta_ip = 0.0
        N = Nip_element[ip]
        for i in ti.static(range(8)):
            theta_ip += N[i] * temperature[t - 1, elements_node_ids[el_id][i]]
        cp_el[ip] = cp_init
        if theta_ip > solidus and theta_ip < liquidus:   # forward-only branch; not an optimized control
            cp_el[ip] += latent_cp
    return cp_el
@ti.kernel
def update_matprop(t: ti.i32):
    for el_id in range(nel):
        cp[el_id] = calc_cp(t, el_id, solidus, liquidus, latent_cp)

# ---- lumped capacitance over ACTIVE elements (M^-1 becomes a divide) ----
@ti.func
def calc_mass(t, el_id, density):
    mass = ti.Matrix.zero(float, 8, 8); lump_mass = ti.Vector.zero(float, 8)
    nodes_pos = ti.Matrix.rows([node_position[elements_node_ids[el_id][k]] for k in range(8)])
    for ip in ti.static(range(8)):
        N = Nip_element[ip]; B = Bip_element[ip]
        detJac = (B @ nodes_pos).determinant()
        mass += density * cp[el_id][ip] * detJac * N @ N.transpose()
    for i in ti.static(range(8)):
        for j in ti.static(range(8)):
            lump_mass[i] += mass[i, j]          # row-sum lumping -> diagonal
    return lump_mass
@ti.kernel
def update_mvec(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:        # frozen discrete mask, NOT differentiated through
            lump_mass = calc_mass(t, el_id, density)
            for i in ti.static(range(8)):
                m_vec[elements_node_ids[el_id][i]] += lump_mass[i]   # atomic accumulation

# ---- conduction: accumulate -K T^n into rhs ----
@ti.func
def calc_stiffness(el_id):
    stiffness = ti.Matrix.zero(float, 8, 8)
    nodes_pos = ti.Matrix.rows([node_position[elements_node_ids[el_id][k]] for k in range(8)])
    for ip in ti.static(range(8)):
        B = Bip_element[ip]; Jac = B @ nodes_pos
        gradN = Jac.inverse() @ B
        stiffness += cond[el_id][ip] * Jac.determinant() * gradN.transpose() @ gradN
    return stiffness
@ti.kernel
def update_stiffness(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            stiffness = calc_stiffness(el_id)
            temperature_nodes = ti.Matrix.rows([[temperature[t-1, elements_node_ids[el_id][k]]] for k in range(8)])
            stiff_temp = stiffness @ temperature_nodes
            for i in ti.static(range(8)):
                rhs[elements_node_ids[el_id][i]] -= stiff_temp[i]

# ---- moving Gaussian surface heat source: where P(t) enters ----
@ti.kernel
def update_fluxes_m(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            for sur_id in ti.static(range(6)):
                if surface_birth[el_id][sur_id, 0] <= t*dt and surface_birth[el_id][sur_id, 1] > t*dt:
                    for ip in ti.static(range(4)):
                        N = Nip_surface[ip]
                        # interpolate this surface IP's position, then distance to the beam
                        ip_pos = ...   # N . surface-node positions
                        r2 = (ip_pos - laser_loc[t]).norm_sqr()
                        qmov = 3.0 * Qin[None] * output3[t, 0] * laser_on[t] \
                               / (pi * r_beam[None]**2) * ti.exp(-3.0 * r2 / (r_beam[None]**2))
                        for i in ti.static(range(4)):
                            rhs[element_surface_ids[el_id][sur_id, i]] += N[i] * qmov * surface_jac[el_id][sur_id, ip]

# ---- convection + radiation on exposed surfaces (T^4 is smooth) ----
@ti.kernel
def update_fluxes_cr(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            for sur_id in ti.static(range(6)):
                if surface_birth[el_id][sur_id, 0] <= t*dt and surface_birth[el_id][sur_id, 1] > t*dt:
                    for ip in ti.static(range(4)):
                        N = Nip_surface[ip]
                        temperature_ip = N.dot(surface_temperature_nodes)
                        qconv = -1 * h_conv[None] * (temperature_ip - ambient[None])
                        qrad  = -1 * 5.67e-14 * h_rad * (temperature_ip**4 - ambient[None]**4)
                        for i in ti.static(range(4)):
                            rhs[element_surface_ids[el_id][sur_id, i]] += N[i] * (qconv + qrad) * surface_jac[el_id][sur_id, ip]

# ---- explicit lumped update: T^{n+1} = T^n + dt * rhs / m_vec ----
@ti.kernel
def time_integrate(t: ti.i32, dt: ti.f32):
    for i in range(nn):
        if node_birth[i] <= t*dt:
            temperature[t, i] = ti.min(temperature[t-1, i] + dt * rhs[i] / m_vec[i], max_temp)  # cap = deliberate gradient stop

# ---- smooth melt-pool depth: fixed local interpolation + linear isotherm crossing ----
@ti.func
def interp_under_beam(t, level):
    value = 0.0
    for j in ti.static(range(9)):
        node = beam_interp_node[t, level, j]     # precomputed from fixed mesh/toolpath
        value += beam_interp_weight[t, level, j] * temperature[t, node]
    return value
@ti.kernel
def update_melt_pool_depth(t: ti.i32):
    depth = 0.0
    for level in range(n_depth_levels - 1):
        t0 = interp_under_beam(t, level)
        t1 = interp_under_beam(t, level + 1)
        if t0 >= liquidus and t1 < liquidus:
            alpha = (t0 - liquidus) / (t0 - t1 + 1e-7)
            depth = depth_level[level] + alpha * (depth_level[level + 1] - depth_level[level])
    melt_depth[t] = depth

@ti.kernel
def compute_temperature_loss():
    for t, i in ti.ndrange(steps, nn):
        ti.atomic_add(loss[None], (temperature[t, i] - target[t, i])**2)   # MSE (partial-observation variant masks i)
@ti.kernel
def compute_melt_pool_loss():
    for t in range(1, steps):
        ti.atomic_add(loss[None], (melt_depth[t] - target_depth[t])**2)

def compute_loss():
    if loss_kind == "melt_pool":
        compute_melt_pool_loss()
    else:
        compute_temperature_loss()

def simulate():
    nn1(0); nn2(0); nn3(0)
    for time_step in range(1, steps):
        clear_vectors()
        nn1(time_step); nn2(time_step); nn3(time_step)     # P(t)
        update_matprop(time_step)
        update_mvec(time_step, dt)
        update_stiffness(time_step, dt)
        update_fluxes_m(time_step, dt)                     # laser injects P(t)
        update_fluxes_cr(time_step, dt)
        time_integrate(time_step, dt)
        update_melt_pool_depth(time_step)
    compute_loss()

# ---- optimize: one tape = the whole recurrence; one backward = full gradient ----
for it in range(iterations):
    loss[None] = 0.0
    with ti.Tape(loss):        # records kernel launches; on exit replays adjoints in reverse (the discrete adjoint)
        simulate()
    update_weights_adam(it + 1)   # Adam on whichever parameters are the leaves (scalars or net weights)
```

AM quality lives in the transient thermal field; that field is set by process parameters, and in the hard case the parameter is a time-series power schedule with thousands of degrees of freedom. Optimizing that against a simulation-based loss needs gradients, and the only affordable way to get a gradient whose cost doesn't scale with the parameter count is reverse-mode AD — because the loss is one scalar and the parameters are many, the backward pass delivers the whole gradient at roughly the cost of one forward run. So I make the time-stepping thermal solver itself differentiable; backprop through the explicit recurrence *is* the discrete adjoint of the solver, obtained automatically for whatever loss and parameterization I write, at the price of storing the state history (O(N) memory, reducible to O(√N) by checkpointing). To keep the gradients real I freeze the discrete machinery — element birth, surface birth, toolpath — behind fixed masks so the gradient only flows through the continuous power and properties, and I replace any step-like quantity (melt-pool depth) with a smooth surrogate that carries a nonzero gradient. I host it in an imperative differentiable-programming framework whose flexible indexing and atomic accumulation make FE assembly fast and whose lightweight tape makes the reverse pass cheap. Then a tiny network maps time to power, the differentiable sim turns that into a thermal history, the MSE loss backpropagates into the network, and Adam shapes the schedule — to calibrate hidden parameters from partial observations, to hit an arbitrary target thermal history, and to hold the melt-pool depth steady through the geometry. The simulator stopped being a black box to probe and became a function to differentiate.
