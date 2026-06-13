# Context

## Research question

In metal additive manufacturing (AM) — directed energy deposition, powder-bed fusion — the quality of the finished part is governed almost entirely by its *thermal history*. The transient temperature field at every point, as the laser traverses the build, sets the local cooling rate (hence microstructure and phase), the accumulated residual stress and distortion, and the size and depth of the melt pool (hence geometric accuracy and porosity). That thermal field is in turn controlled by the *process parameters*: laser power, scan speed, beam radius, the toolpath, and the material properties of the deposited powder.

The practical goal is the inverse of simulation: given a *target* thermal or melt-pool behavior, find the process parameters that produce it. The hard version of this is high-dimensional and temporal — not "what constant laser power should I use," but "what laser-power *schedule* P(t), varying over the tens of thousands of time steps of a build, keeps the melt-pool depth constant as the geometry changes from a thick base to a thin neck." A solution method has to (i) handle thousands of coupled design variables, (ii) respect the known transient heat-transfer physics over an arbitrary part geometry, and (iii) do so at a cost that does not explode with the number of parameters. The pain point is that the simulation that maps parameters → thermal field is itself expensive, so any optimizer that needs many simulation evaluations is dead on arrival for the high-dimensional case.

## Background

**Transient heat transfer in AM, by finite elements.** The governing physics is the transient heat-conduction PDE with a moving source,

  ρ c_p ∂T/∂t − ∇·(k ∇T) − s = 0,

subject to a Dirichlet condition T = T₁ on Γ₁, a prescribed-flux Neumann condition q = −q_s on Γ₂, a convection condition q = −h(T − T_amb) on Γ₃, and a radiation condition q = −εσ(T⁴ − T_amb⁴) on Γ₄. Here ρ is density, c_p specific heat, k conductivity, s volumetric generation, h the convection coefficient, ε emissivity, and σ the Stefan–Boltzmann constant. Discretizing the weak form over the meshed part with shape functions Nᵉ and their derivatives Bᵉ gives global matrices — a capacitance (mass) matrix [M] and a conduction (stiffness) matrix [K] — assembled from per-element contributions, plus load vectors for the laser flux, convection, and radiation.

**The moving heat source.** Rosenthal's analytical moving-point-source solution and its successors model the laser as a concentrated traveling source; Goldak's double-ellipsoid and the simpler surface-Gaussian models spread that energy over a finite spot with a Gaussian density so the temperature field is finite and physical. A circular surface Gaussian deposits incoming flux

  q_s(x,t) = 3 Q P(t) on(t) / (π r_b²) · exp(−3||x − x_L(t)||² / r_b²),

with beam radius r_b, beam center x_L(t), commanded normalized power P(t), and toolpath on/off state on(t). The coefficient integrates to QP(t)on(t) over the plane and puts about 95% of the energy inside radius r_b, which is the standard Goldak-style surface source used to inject laser power into the thermal FE model.

**Material deposition by element birth.** AM adds material as the build proceeds, so the simulation domain grows in time. The standard device is *element birth* (quiet/active element activation): every element and every exposed surface is tagged with a birth time computed from the toolpath, and only elements/surfaces whose birth time has passed are assembled into the global system at a given step. This makes the active mesh, and the set of free surfaces exchanging heat with the environment, change discretely from step to step.

**Explicit time integration.** For these problems an explicit scheme with a *lumped* (row-summed, diagonalized) capacitance matrix is attractive:

  T^{n+1} = T^n + Δt M⁻¹[ R_G − R_F − R_C − R_R − K T^n ].

Here R_G is internal generation, R_F is the prescribed boundary-flux vector under the outward-flux convention, R_C and R_R are convection and radiation losses, and K T^n is conductive diffusion. With a lumped M, M⁻¹ is just elementwise division by nodal heat capacities, so the step needs no linear solve and parallelizes naturally on GPUs. Prior work established GPU-accelerated explicit FE for powder-based AM, assembling element contributions with atomic scatter operations and stepping through tens of thousands of small Δt steps (Mozaffar, Ndip-Agbor, Lin, Wagner, Ehmann, Cao 2019, *Acceleration strategies for explicit finite element analysis of metal powder-based additive manufacturing processes using graphical processing units*, Comput. Mech.). The cost of one forward simulation of a realistic part is minutes.

**The cost of derivative-free / finite-difference optimization.** To optimize parameters against a simulation-based loss, the cheapest gradient estimate is finite differences: perturb one parameter, re-run the whole simulation, take the difference. That costs one extra full simulation *per parameter*. For the static case (a handful of scalars) this is merely expensive; for a time-series laser power discretized over ~20,000 steps it is thousands of full simulations per gradient step — completely infeasible. Derivative-free optimizers (Nelder–Mead, evolutionary, Bayesian) likewise need many forward evaluations and scale poorly into thousands of dimensions. The cost of getting a gradient scales with the *number of parameters*, and that is the barrier to high-dimensional temporal design.

**Reverse-mode automatic differentiation.** AD treats a numerical program as a composition of elementary operations with known local derivatives, and computes exact derivatives (to machine precision) by the chain rule — not by finite differencing. It comes in two modes. Forward mode propagates a derivative *with respect to one input* through the program; computing the full gradient of P inputs costs O(P) forward sweeps. Reverse mode propagates the sensitivity *of one output* backward through the program, accumulating an adjoint for every intermediate; it computes the gradient with respect to *all* inputs at once, at a cost of a small constant times one forward evaluation, **independent of the number of inputs**. This is exactly the backpropagation used to train neural networks (Baydin, Pearlmutter, Radul, Siskind 2018, *Automatic differentiation in machine learning: a survey*). The catch is memory: reverse mode must store (or recompute) the intermediate values produced on the forward pass, because the backward pass evaluates local derivatives at those values. For a recurrent computation of N time steps naively storing every state costs O(N) memory.

**Differentiable physics as a substrate.** Outside manufacturing, differentiable simulators had begun to appear in robotics and graphics — differentiable rigid-body, cloth, and particle simulators used to train controllers and solve inverse problems by gradient descent through the simulation, converging in tens of iterations where model-free methods need orders of magnitude more (Hu et al. 2019, ChainQueen; Heiden et al. 2019; Liang, Lin, Koltun 2019; Holl, Koltun, Thuerey 2020; Qiao et al. 2020). Tooling to *build* such simulators efficiently also matured: a differentiable programming system can record a lightweight tape of the operations performed during a forward simulation and replay their adjoints in reverse for end-to-end backpropagation, with source-code-transformation AD applied to imperative, parallel, flexibly-indexed kernels (Hu, Anderson, Li, Sun, Carr, Ragan-Kelley, Durand 2019, *DiffTaichi*). To keep differentiation well-defined under in-place imperative updates, such systems impose global-data-access rules: a tensor element written more than once must be written by atomic accumulation, and it is not read until accumulation is done.

## Baselines

**Forward thermal FE simulation (no gradients).** The established workhorse: build the mesh, generate the toolpath and element-birth schedule, step the explicit lumped-mass thermal solver, read out the temperature history. Core algorithm is the per-step assembly of M, K and the flux/convection/radiation vectors over active elements and the explicit update. *Gap:* it answers only the forward question ("given these parameters, what is the thermal field"); it gives no way to *choose* parameters to hit a target except by trial and error.

**Finite-difference / derivative-free optimization of the simulator.** Wrap the forward simulator in an outer optimizer and estimate gradients by perturb-and-resimulate, or use a gradient-free search. Core idea: treat the simulator as a black box J(θ); approximate ∂J/∂θᵢ ≈ (J(θ+δeᵢ) − J(θ))/δ. *Gap:* O(P) full simulations per gradient — fine for a few scalars, impossible for thousands of time-series values; also FD step-size error and noise.

**Pure data-driven surrogates.** Replace the physics with a learned regressor mapping parameters → response, then optimize the cheap surrogate. Core idea: fit a neural net / Gaussian process to simulation or experimental data. *Gap:* extrapolates poorly outside its training domain, does not conserve the known heat-transfer physics over arbitrary geometries, and can violate physical laws exactly where the design pushes into novel regimes.

**Hand-derived continuous adjoint of the PDE.** For specific objectives one can derive an adjoint PDE by hand, solve it once backward in time, and read off the gradient — O(1) solves regardless of parameter count, like reverse-mode AD in spirit. *Gap:* the derivation is bespoke per objective and per boundary-condition set, error-prone, and must be re-done by hand whenever the loss, the parameterization (e.g. a neural-net-parameterized control), or the discrete bookkeeping (element birth, melt-pool extraction) changes — it does not compose automatically with arbitrary downstream differentiable computations.

## Evaluation settings

A representative metal-AM test part: an hourglass geometry (thick base and top, thin neck) defined in CAD, meshed into hexahedral (Hex8) elements with substrate, and a toolpath generated by slicing the geometry at fixed vertical intervals and filling each 2-D section by hard-coded strategies (inward-from-boundary, zig-zag), with laser-on / laser-off segments. Inputs are the mesh, the toolpath, and the per-element birth file. Stainless-steel thermal properties (density, c_p, conductivity, convection/radiation coefficients, solidus 1533.15 K / liquidus 1609.15 K with a latent-heat band) and a beam radius / nominal power set the scale; the explicit step Δt is small enough that a build spans ~10⁴–2·10⁴ steps. The natural design tasks are: (i) infer static material/process scalars from a *partially observed* thermal response (e.g. only top-layer node temperatures, as an IR camera would see); (ii) match a full target thermal-history time series by shaping the time-series laser power; (iii) hold a target melt-pool depth throughout the build by shaping the time-series laser power. The metric in each case is mean-squared error between the achieved and target response.

## Code framework

A minimal scaffold starts from the forward explicit thermal FE program and leaves the process input, response extraction, gradient source, and scalar objective as open slots.

```python
import numpy as np

# ----- pre-existing FE primitives -----
def shape_fnc_element(parCoord):      # Hex8 N at a Gauss point
    pass

def derivate_shape_fnc_element(parCoord):  # Hex8 dN/dxi (B)
    pass

def load_inputfile(filename):
    # mesh nodes/elements, node & element birth times, exposed surfaces & surface birth
    pass
    return nodes, birth_list_node, elements, birth_list_element, element_surface, element_surface_birth

def load_toolpath(filename, dt):
    # laser location + on/off state, resampled at dt
    pass
    return toolpath, state, endTime

# ----- forward-simulation storage -----
temperature = np.zeros((steps, nn))
m_vec = np.zeros(nn)
rhs = np.zeros(nn)
loss = 0.0

# ----- process input slot -----
def process_input(t):
    # TODO: produce the laser/material/process input used at step t.
    pass

# ----- per-step thermal FE kernels -----
def update_matprop(t):    # temperature-dependent c_p incl. latent-heat band
    pass                  # TODO
def update_mvec(t, dt):   # assemble lumped capacitance over ACTIVE elements
    pass                  # TODO
def update_stiffness(t, dt):  # assemble conduction, accumulate -K T^n into rhs
    pass                  # TODO
def update_fluxes_laser(t, dt):   # moving Gaussian surface source over active surfaces
    pass                  # TODO: inject process_input(t) here
def update_fluxes_conv_rad(t, dt):  # convection + radiation on active surfaces
    pass
def time_integrate(t, dt):    # explicit lumped update T^{n+1} = T^n + dt * rhs / m
    pass

# ----- response feature slot -----
def response_feature(t):
    # TODO: extract temperature, partial observation, or a smooth melt-pool-depth proxy
    pass

# ----- loss: TODO -----
def compute_loss():
    # TODO: MSE between achieved response (temperature history / melt-pool depth)
    #       and a target, possibly under partial observation
    pass

def compute_gradient(loss, parameters):
    # TODO: choose a gradient source for the simulator-defined objective
    pass

def simulate():
    for t in range(1, steps):
        clear_vectors()
        process_input(t)
        update_matprop(t)
        update_mvec(t, dt)
        update_stiffness(t, dt)
        update_fluxes_laser(t, dt)
        update_fluxes_conv_rad(t, dt)
        time_integrate(t, dt)
        response_feature(t)
    compute_loss()

# ----- optimization loop -----
def optimize(iterations, lr):
    for it in range(iterations):
        simulate()
        grads = compute_gradient(loss, parameters)
        # TODO: update the design parameters from grads
        pass
```
