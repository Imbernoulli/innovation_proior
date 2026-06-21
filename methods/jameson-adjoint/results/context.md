# Context: Gradient-based aerodynamic shape design with CFD

## Research question

Modern aircraft surfaces — wings, nacelles, wing–body fairings — are defined by computational
simulation: a flow solver evaluates a candidate shape, an engineer inspects the result, and the
shape is nudged and re-evaluated. The goal is to close the loop and let the simulation itself
drive the shape toward the true optimum.

Formally: parameterize a shape by a control — a finite set of design variables α (weights on bump
functions, spline control points), or, in the limit, a free boundary surface F(x) constrained only
to be smooth. Pick a scalar cost I — drag at fixed lift, lift-to-drag ratio, or a target-pressure
mismatch I = ½∫(p − p_d)² for inverse design. To improve the shape we need the gradient dI/dα so we
can step downhill. The question is how to obtain this gradient affordably when the number of design
variables is large — a real wing wants hundreds, and the genuinely optimal shape lives in an
infinite-dimensional space of surfaces.

## Background

The flow model is the compressible Euler (later Navier–Stokes) equations for transonic flow over the
surface, written as a steady conservation law ∂fᵢ/∂xᵢ = 0 in a domain D, with w = (ρ, ρu₁, ρu₂, ρu₃,
ρE) the conserved state, fᵢ the inviscid fluxes, and p = (γ−1)ρ(E − ½uᵢ²) the closure. These are
nonlinear hyperbolic PDEs admitting shocks; solving them for a single shape is itself expensive. By
the late 1980s fast solvers existed and were the enabling substrate: the JST finite-volume scheme
(central differencing with blended second/fourth-difference artificial dissipation), explicit
multistage time-marching to steady state, and multigrid acceleration drove the cost of one transonic
flow solution down to something a design loop could call repeatedly.

The dominant, load-bearing empirical fact of the era: **a single transonic flow solve is the unit of
cost**, and everything is measured in multiples of it. A finite-difference gradient perturbs each of
N design variables in turn and re-solves the flow — so it costs N+1 flow solves per gradient. The
finite-difference estimate ∂I/∂αᵢ ≈ (I(αᵢ+δαᵢ) − I(αᵢ))/δαᵢ also has a step-size tradeoff: too
large and the difference is contaminated by curvature; too small and it is affected by the flow
solver's iterative-convergence noise.

A second, more subtle phenomenon shapes the formulation. If one tries to compute the sensitivity of
the **pressure at a fixed point** to a shape change, a small shape modification can sweep a shock
across that point, so the pointwise pressure sensitivity ∂p/∂F can become unbounded. Yet the shock
*moves continuously* with the shape, so an **integrated** quantity — drag, or the pressure-mismatch
integral — depends continuously and smoothly on the shape even where the pointwise sensitivity blows
up. Any gradient method that targets integrated functionals rather than pointwise sensitivities
remains well-posed through shocks.

A third phenomenon concerns the descent step itself. In the calculus of variations, minimizing
I = ∫F(y, y′)dx gives the gradient g = ∂F/∂y − (d/dx)(∂F/∂y′), which contains a derivative of the
trajectory; stepping y ← y − λg therefore differentiates y and reduces its smoothness by two
classes per step.

A fourth, physical fact: induced (vortex) drag is a large fraction of total drag and falls if lift
is simply shed or span increased. The natural cost is drag **at a fixed target lift coefficient**,
with planform and minimum thickness also held, so the optimizer works under realistic aerodynamic
constraints.

## Baselines

**Finite-difference gradient + gradient descent (Hicks & Henne 1978, *Wing Design by Numerical
Optimization*, J. Aircraft 15:407).** The first serious CFD-plus-optimization wing design: a
full-potential flow solver coupled to a conjugate-gradient optimizer, with the gradient of the cost
w.r.t. each shape parameter obtained by finite differences. Core idea: treat I as a black-box
function of α, estimate ∇I component-by-component, descend. Gradient cost scales as N+1 flow
solves, and the FD estimate carries step-size sensitivity.

**Optimal control of PDEs (Lions 1971, *Optimal Control of Systems Governed by Partial Differential
Equations*).** The abstract machinery for minimizing a functional subject to a PDE constraint:
introduce an adjoint (costate) field as a Lagrange multiplier defined over the domain, and obtain the
functional's gradient from the adjoint without separately perturbing each control. Core idea: the
Lagrangian I + ⟨ψ, R⟩ with R the state PDE; stationarity in the state yields an adjoint PDE for ψ;
the remaining variation gives the gradient. The framework is stated abstractly for general PDEs.

**Adjoint/costate gradients in trajectory optimization (Bryson & Ho 1975, *Applied Optimal
Control*).** For optimal control of an ODE system, the gradient of a terminal cost w.r.t. *all*
controls is obtained by integrating a costate equation **backward in time** in a single sweep —
one backward integration regardless of how many control parameters there are. Core idea: the costate
carries the sensitivity information backward so the gradient assembles cheaply. This template of
"one extra solve gives the whole gradient" comes from time-dependent ODE control.

**Control-theoretic optimal shape design for elliptic flows (Pironneau 1973/1974, *On optimum
profiles in Stokes flow*; *On optimum design in fluid mechanics*, JFM 64:97–110).** The first
application of control theory / adjoint optimal-shape-design to fluid mechanics: minimal-drag
profiles in Stokes and low-Reynolds flow, derived via an adjoint state for an **elliptic** system.
Core idea: the shape gradient comes from an adjoint solution rather than from N separate flow
perturbations. The regime is Stokes / elliptic / low-speed, smooth and shock-free.

**Direct (tangent/forward) differentiation.** An alternative to FD that is also exact: differentiate
the converged flow constraint R(w,α)=0 to get the linear *forward* sensitivity equation
(∂R/∂w)(∂w/∂αᵢ) = −∂R/∂αᵢ, solve it once per design variable for ∂w/∂αᵢ, then form dI/dαᵢ. Core idea:
propagate each input perturbation forward through the linearized flow. It removes FD's step-size
sensitivity and provides exact sensitivities, with cost proportional to N linear solves.

## Evaluation settings

The natural model problem is the **quasi-one-dimensional Euler nozzle**: conserved state w = (ρ, ρu,
e) per cell, a duct area distribution S(x) as the control parameterized by a few design variables
(sine bumps or spline control points), the flow found by time-marching the residual R(w,S)=0 to
steady state with a Roe / Steger–Warming flux. The canonical cost is inverse design,
I = ½ Σᵢ (pᵢ/p_t − p_{d,i})² Δxᵢ, recovering a prescribed target pressure distribution p_d (so the
known optimum is the shape that generated p_d, which lets a gradient be checked against a known
answer). The natural gradient checks are forward differentiation and central finite differences, both
evaluated on the same converged discrete residual.

The full-scale settings are three-dimensional transonic wing and wing–body configurations under the
Euler and Reynolds-Averaged Navier–Stokes equations, with cost = drag coefficient at a fixed target
C_L (held by trimming angle of attack each flow solve), planform and minimum thickness constrained,
and the surface parameterized by mesh-point displacements or bump functions on the wing. Standard
test geometries (e.g. the NACA 0012 airfoil for inviscid airfoil studies) and structured / overset /
unstructured meshes are the yardstick. The metrics are the cost itself and the per-design-cycle
computational cost expressed in flow-solve units.

## Code framework

The primitives that already exist: a quasi-1D Euler flow solver that marches R(w,S)=0 to steady
state, a shape parameterization mapping design variables to the area distribution, a cost evaluation,
and a generic descent-with-line-search optimizer. The one empty slot is the gradient of the cost
w.r.t. the design variables — everything below routes through `getGradient`, whose interesting branch
is a `# TODO`.

```cpp
// --- existing: flow solver, parameterization, cost ---
void quasiOneD(const std::vector<double>& x, const std::vector<double>& area,
               const Flow_options& flo, Flow_data<double>* flow);          // marches R(w,S)=0 to steady state
std::vector<double> evalS(const Design<double>& design,
                          const std::vector<double>& x,
                          const std::vector<double>& dx);                  // design vars -> area distribution S(x)
double evalFitness(const std::vector<double>& dx, const Flow_options& flo,
                   const std::vector<double>& W, const Optimization_options<double>& o);
                                                                            // cost I = 1/2 sum (p/p_t - p_d)^2 dx

// --- gradient: the slot the method fills ---
VectorXd gradient_FD(/* perturb each design var, re-solve flow */);        // baseline: N+1 flow solves, noisy

VectorXd getGradient(int gradient_type, /* x, dx, area, flow, opt, design */) {
    // gradient_type < 0 : finite differences (baseline above)
    // TODO: fill in the gradient evaluation
    pass;
}

// --- existing: generic descent loop with a line search ---
void optimizer(/* ... */) {
    quasiOneD(x, area, flo, &flow);
    double cost = evalFitness(dx, flo, flow.W, opt);
    VectorXd g  = getGradient(opt.gradient_type, /* ... */);
    while (g.norm() > opt.opt_tol && it < opt.opt_maxit) {
        VectorXd pk = /* TODO: descent direction from the gradient (and any conditioning of it) */;
        cost = linesearch_backtrack(/* Armijo: new_cost <= cost + alpha*c1*g.dot(pk) */);
        area = evalS(current_design, x, dx);
        g    = getGradient(opt.gradient_type, /* ... */);
    }
}
```
