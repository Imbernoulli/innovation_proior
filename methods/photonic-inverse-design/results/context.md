# Context: designing nanophotonic devices by optimizing the permittivity

## Research question

Silicon photonics patterns sub-wavelength silicon waveguides on a chip, so that
optical functions which once needed bulky separate components can be integrated
alongside CMOS electronics, cutting cost, energy and size. Straight silicon
waveguides transport light with very low loss. At functional elements such as
splitters, waveguide crossings, multimode interferometers, and fiber-to-chip
couplers the mode is reshaped, and evanescent fields interact with silicon/oxide
interfaces.

The concrete goal: given a device footprint (an input waveguide, an output
waveguide, and a small region of chip in between where the silicon pattern is
free to be anything), choose the permittivity distribution ε(r) over that region
so as to maximize a figure of merit F — typically the optical power coupled into
a specified mode at the output port. The design space has one free variable per
pixel of the grid, and the forward model is Maxwell's equations, whose single
evaluation is a full electromagnetic simulation.

## Background

**Maxwell's equations and frequency-domain simulation.** A time-harmonic field
at frequency ω in a non-magnetic dielectric obeys ∇×∇×E − ω²/c² ε_r(r) E =
−iωμ₀ J, driven by a current source J. Discretizing space (a Yee grid in
finite-difference frequency domain, FDFD, or time stepping in FDTD) turns this
into a large sparse linear system A(ε) e = b, where the operator A carries the
curl-curl term and the permittivity, e is the stacked field, and b is
proportional to the source. Solving it once yields the fields everywhere — this
is the forward simulation, and it is the expensive primitive. In two dimensions
with transverse-magnetic polarization only the out-of-plane electric component is
nonzero (the in-plane magnetic components remain), which keeps illustrative
problems tractable by reducing each field to a single scalar unknown per cell. Established solvers (FDTD
packages, both commercial and open) provide this forward solve as a black box.

**The figure of merit lives at a port.** Device performance is read out as a
scalar F computed from the simulated field on a monitor plane at the output. For
a waveguide device this is the power transmitted into a particular guided mode:
the field is projected onto the target mode profile (E_m, H_m) through an overlap
integral built from Poynting fluxes, and the coupling efficiency is the squared,
mode-normalized overlap. So F is a real, smooth function of the field on a
surface, ultimately a function of ε through the field.

**Why high-dimensional design is the hard part.** A continuous, point-by-point
permittivity gives essentially unlimited design freedom — recent structural
topology-optimization work has handled upward of a billion variables. With N free
pixels, a finite-difference gradient — perturb pixel i, re-simulate, read the
change in F — needs one extra simulation per pixel, i.e. N+1 forward solves for a
single gradient. With N from thousands to millions and each solve a full Maxwell
problem, this is a significant cost.

**Diagnostic facts about naive pixel optimization.** Two phenomena are well
documented for density-based topology optimization. First, an unregularized pixel
grid develops single-pixel and checkerboard features — a solid/void checkerboard
is a numerical artifact that low-order finite elements report as artificially
stiff/efficient, so an optimizer left unconstrained will exploit it; such patterns
are also unmanufacturable. Second, the optimizer tends to settle on intermediate,
"gray" permittivity values ε between the two real materials, because a continuum
of index lets it tune phase locally. Gray material is not a fabricable material.
Simply thresholding a converged gray design at the midpoint to recover a
black/white pattern can collapse the figure of merit (in a reported metalens
example the merit fell from ≈18.2 to ≈4.7), because the thresholded structure is
a different device.

**Sensitivity analysis and shape representations in neighboring fields.** Shape
and topology optimization in mechanical engineering and fluid mechanics, and
sensitivity analysis for quantum electronic device design, are mature fields that
routinely optimize objectives constrained by a PDE. Perturbation theory for
Maxwell with shifting material boundaries, and geometry-projection techniques,
have been worked out as tools for relating a change in material layout to a
change in the fields. The level-set method of Osher and Sethian represents a
shape implicitly as the zero contour of a function and evolves it under a
velocity field, admitting a very large number of boundary degrees of freedom.

## Baselines

**Heuristic / stochastic global optimization (genetic algorithms, particle
swarm).** The prevailing approach to photonic device shaping. A modest set of
shape parameters is chosen, a population of random parameter sets is simulated,
and the population is evolved using the collected figure-of-merit values until a
satisfactory member appears. Particle swarm optimization, for instance, drove the
prior state-of-the-art silicon Y-splitter (Zhang et al. 2013), reaching a low
insertion loss after on the order of a thousand-plus full simulations. These
methods only sample F and use a limited parameterization to stay affordable.

**Finite-difference gradient descent.** Conceptually one could compute a gradient
and descend. Estimating ∂F/∂ε_i by perturbing each pixel and re-simulating
costs one Maxwell solve per pixel. For a pixelated permittivity this is N+1
solves per gradient step.

**Continuous-permittivity penalization (e.g. Seliger/Levi-type aperiodic
dielectric optimization).** Optimizes a continuously variable permittivity
directly. It can use gradients but yields gray, intermediate-index structures
that are not directly manufacturable in a two-material process.

## Evaluation settings

The natural testbed is a silicon-photonics passive component at telecom
wavelength λ = 1550 nm, in a silicon-core / silicon-dioxide-cladding material
system, with a 220 nm silicon thickness (the standard for the platform). A
canonical benchmark is a Y-junction splitter occupying a small central design
region (e.g. 2 µm × 2 µm), with the input/output waveguide stubs fixed, and a
manufacturing constraint such as a minimum radius of curvature (e.g. 200 nm) so
the result is fabricable. The figure of merit is the coupling efficiency /
insertion loss into the fundamental mode of the output waveguides, measured by
mode overlap from the simulated fields; the operating bandwidth (e.g. 1.5–1.6 µm)
indicates robustness. The forward model is an FDTD/FDFD solver; a cheap 2D
effective-index simulation (silicon assigned an effective index that mimics the
3D in-plane propagation constant) can stand in for a full 3D solve during early
iterations. Cost is counted in number of Maxwell simulations.

## Code framework

The forward simulation and the readout primitives exist; the parameterization of
the permittivity, the gradient, and the optimization loop remain to be built.

```python
import numpy as np
import meep as mp          # FDTD forward solver (the expensive primitive)

WVL = 1.55
FREQ = 1.0 / WVL
DESIGN_SIZE = mp.Vector3(2.0, 2.0, 0)   # central designable region (um)
NX, NY = 61, 61                          # design pixels
N = NX * NY

# --- parameterize the permittivity over the design region --------------------
def map_design(rho):
    """Map a vector of design variables rho to the permittivity weights that
    the forward solver places in the design region."""
    pass  # TODO

# --- build the forward problem and a figure-of-merit readout -----------------
def build_simulation(design_weights):
    """Place the input/output silicon waveguides and a design region whose
    permittivity is set by design_weights; launch the input mode; set up a
    monitor at the output port. Returns a solver ready to run."""
    # geometry: input WG + design region + output WG ; EigenModeSource at input
    # output mode-overlap monitor -> scalar figure of merit F
    pass  # TODO

def forward_F(design_weights):
    """One Maxwell solve -> scalar figure of merit F (power into target mode)."""
    pass  # TODO

# --- the gradient of F w.r.t. EVERY design variable --------------------------
def grad_F(design_weights):
    """Return dF/d(design variable) for all variables at once."""
    pass  # TODO

# --- optimization loop -------------------------------------------------------
def main():
    rho = 0.5 * np.ones(N)               # initial design
    for it in range(num_iters):          # gradient-based update
        w = map_design(rho)
        F = forward_F(w)
        g = grad_F(w)
        rho = update(rho, g)             # ascent + bounds/constraints
        pass  # TODO

if __name__ == "__main__":
    main()
```
