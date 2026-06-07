# Context — Predictability of a Forced Dissipative Flow, and Whether Small Errors Stay Small

## Research question

The atmosphere is a deterministic fluid: its motion is fixed, in principle, by the hydrodynamic
equations together with the present state. The practical question that organizes a generation of
meteorology is whether that determinism can be turned into forecasting — and how far ahead. The tacit
working assumption, inherited from celestial mechanics, is that approximate knowledge of the present
state buys approximate knowledge of the future: refine the observations a little and the forecast
improves a little, smoothly, without limit. If that assumption holds, long-range prediction is merely
an engineering problem of better data and bigger machines.

But the atmosphere is conspicuously *nonperiodic*: weather varies in an irregular, seemingly
haphazard way and rarely repeats its past closely. Many simple deterministic systems, by contrast,
settle to a steady state or to a periodic cycle. So the precise question is sharper than "how good can
forecasts get?" It is: can a finite system of deterministic equations, with constant forcing, produce
sustained nonperiodic behavior at all — and if it can, does a small uncertainty in the initial state
stay small as the system evolves, or can it grow until the forecast is worthless? Writing the state of
a system as a point moving in phase space, the question becomes whether two trajectories that start
imperceptibly close remain close as `t → ∞`, or whether they can diverge to the full extent of the
system's range. The stakes are the legitimacy of long-range weather prediction itself: if small errors
can amplify without bound, then no improvement in observation, short of perfection, suffices, and
prediction of the distant future is impossible in principle rather than in practice.

## Background

**Determinism and the smoothness assumption.** Classical mechanics teaches that the present state plus
the forces determine all future states. In Laplace's formulation, an intelligence that knew the
positions and velocities of all particles and all forces "would embrace in the same formula the
movements of the greatest bodies of the universe and those of the lightest atom; for it, nothing would
be uncertain and the future, as the past, would be present to its eyes." Carried into practice, this
became the assumption that *approximate* initial knowledge yields *approximate* prediction — supported
by the spectacular success of predicting eclipses, planetary positions, and comet returns, where small
observational errors do stay small.

**A nineteenth-century crack.** Henri Poincaré, studying the three-body problem in celestial
mechanics, found trajectories that fit none of the standard pictures — not a fixed point, not a closed
loop, not a torus. He observed that "small differences in the initial conditions produce very great
ones in the final phenomena... prediction becomes impossible," and that the curves "fold upon
themselves in very complex fashion." Hadamard found related instability in geodesic flows on negatively
curved surfaces. These results were largely set aside as pathologies — a "gallery of monsters" — rather
than as a general feature of dynamics, and lay mostly dormant for decades.

**Phase space and the behavior of trajectories.** A system of `M` variables `X₁,…,X_M` governed by
`dXᵢ/dt = Fᵢ(X₁,…,X_M)` is studied as a point moving along a trajectory in an `M`-dimensional phase
space. If the `Fᵢ` have continuous partial derivatives, a unique trajectory passes through each point.
The prevailing expectation, drawn from linear systems and from conservative mechanics, is that a
trajectory left alone eventually approaches a steady state (a point), a periodic state (a closed loop),
or a quasi-periodic state with several incommensurate frequencies (a torus). In a *conservative* system
some quantity `Q` (e.g. energy) is invariant, so each trajectory is pinned to a surface `Q = const`.

**Forced dissipative systems.** Real hydrodynamic systems are not conservative: viscosity always
dissipates kinetic energy and thermal conduction always dissipates temperature differences, while
external heating continually supplies energy. For determinism the forcing must itself follow a fixed
rule, so one studies *forced dissipative* systems of the form
```
dXᵢ/dt = Σ aᵢⱼₖ Xⱼ Xₖ − Σ bᵢⱼ Xⱼ + cᵢ,
```
where the quadratic terms (advection — transport of a fluid property by the motion of the fluid
itself) conserve `Q = ½ Σ Xᵢ²` (the term `Σ aᵢⱼₖ Xⱼ Xₖ Xᵢ` vanishes identically), the linear terms
represent dissipation (`Σ bᵢⱼ XᵢXⱼ` positive definite), and the constants `cᵢ` the forcing. Along a
trajectory, `dQ/dt = −Σ bᵢⱼXᵢXⱼ + Σ cᵢXᵢ`; the negative definite quadratic term dominates the linear
forcing term outside a bounded ellipsoid, so `dQ/dt < 0` there. Every trajectory is eventually trapped
inside a bounded region `R` of phase space and cannot escape to infinity. The presence of the nonlinear
advective terms is essential: for *linear* dissipative systems, constant forcing leads inexorably to a
constant (steady) response, so any nonperiodicity would have to be blamed on nonperiodic forcing. With
nonlinearity, constant forcing can produce a variable response.

**Convection as the simplest testbed.** The cleanest physical system with these ingredients is a layer
of fluid of depth `H` heated from below, with a fixed temperature difference `ΔT` between the lower and
upper surfaces (Rayleigh–Bénard convection). There is a motionless steady solution in which
temperature falls linearly with height. Rayleigh (1916) showed this rest state goes unstable, and
convection sets in, once the dimensionless Rayleigh number
```
Ra = g α H³ ΔT / (ν κ)
```
exceeds a critical value `Rc = π⁴ a⁻² (1+a²)³`, minimized at `27π⁴/4` when `a² = ½` (here `g` is
gravity, `α` the thermal expansion coefficient, `ν` the kinematic viscosity, `κ` the thermal
conductivity, and `a` sets the horizontal scale of the convection cells). For two-dimensional roll
convection with stress-free boundaries, the governing equations for the stream function `ψ` and the
temperature departure `θ` from the no-convection profile are
```
∂(∇²ψ)/∂t = −∂(ψ, ∇²ψ)/∂(x,z) + ν ∇⁴ψ + g α ∂θ/∂x,
∂θ/∂t    = −∂(ψ, θ)/∂(x,z) + (ΔT/H) ∂ψ/∂x + κ ∇²θ.
```

**Spectral reduction of the convection equations (Saltzman, 1962).** Saltzman attacked finite-amplitude
convection as an initial-value problem by expanding `ψ` and `θ` in a double Fourier series in `x` and
`z`, with coefficients that are functions of time alone. Substituting these series into the convection
equations turns them into an infinite set of coupled ordinary differential equations in the
time-dependent coefficients, the nonlinear advection appearing as quadratic interactions between
coefficients. Truncating to a finite set of modes and integrating numerically, Saltzman found that in
certain cases all but a few of the coefficients eventually died away, while the surviving ones varied
in an irregular, apparently nonperiodic manner — a numerical hint that constant-forced convection,
suitably reduced, can sustain nonperiodic motion. This is the diagnostic observation that makes the
whole question concrete: there exists a small deterministic ODE system, born from a real fluid, that
appears not to settle down.

**The forecasting backdrop.** Operational weather prediction at the time was largely *linear* and
*statistical*: tomorrow's weather was modeled as a linear combination of features of today's, fitted to
historical data. A rival school of *dynamic* meteorologists held that integrating the governing fluid
equations forward would do better. Small digital computers had just become available to individual
researchers — a Royal McBee LGP-30, with 4096 thirty-two-bit words of memory and roughly one second per
arithmetic step, was about a thousand times faster than a desk calculator — making it feasible to
integrate a modest dynamical model over months of simulated weather and compare the two approaches. A
natural way to stress the linear-statistical method is to feed it *nonperiodic* model output, since a
scheme built on repeating patterns should struggle most where the data never repeat.

## Baselines

**Steady-state and periodic dynamics (the expected outcomes).** Core idea: a deterministic system left
alone approaches a point (steady state), a closed loop (periodic), or a torus (quasi-periodic). For a
linear dissipative system with constant forcing the response is provably a constant; for a conservative
system trajectories lie on invariant surfaces `Q = const`. Gap: none of these can be reconciled with a
*bounded, dissipative, deterministic* system whose trajectories never repeat — and the atmosphere, and
Saltzman's reduced convection, both look nonperiodic. The standard catalogue of long-term behaviors has
no slot for sustained aperiodicity, so either the catalogue is incomplete or the observed
nonperiodicity is an artifact of forcing or noise.

**Nonperiodicity blamed on external causes.** Core idea: since linear dissipative systems with constant
forcing cannot be nonperiodic, irregular flow is attributed to nonperiodic or random forcing. Gap: the
rotating-basin convection experiments of Fultz et al. (1959) and Hide (1958) produce irregular
nonperiodic flow under *constant* thermal forcing, within experimental control. So the irregularity is
generated internally; the external-forcing explanation fails for the very systems of interest, and the
nonlinear (advective) terms must be the source.

**Linear stability theory around a steady state.** Core idea: linearize the equations about a steady
solution and examine whether small perturbations grow; the onset of convection is exactly the parameter
value at which the rest state's perturbations first grow (Rayleigh's `Ra = Rc`). Algorithm: form the
Jacobian at the fixed point, read off the eigenvalues, find where one crosses into the right half-plane.
Gap: linear theory predicts only the *onset* of instability and the immediate growth of infinitesimal
perturbations; it says nothing about what the motion does once the perturbations become finite — whether
it saturates into steady convection, oscillates periodically, or does something else. The finite-
amplitude regime is exactly where the interesting question lives, and it is inaccessible to linearization.

**Spectral truncation + numerical integration (Saltzman's approach).** Core idea: expand the fields in
orthogonal functions, truncate to finitely many time-dependent coefficients, integrate the resulting
ODEs numerically. Algorithm: project the PDEs onto a chosen mode set; advance the coefficient vector
with a finite-difference time scheme. Gap: Saltzman observed irregular nonperiodic coefficient behavior
but the system was still relatively large and the nonperiodicity was a reported observation, not a
property pinned down or explained; whether it was genuine, what minimal system would exhibit it, and
what it implied for predictability were all open.

## Evaluation settings

The object of study is a finite system of ordinary differential equations of the forced-dissipative
form above, integrated forward from chosen initial conditions over long stretches of dimensionless
time. The natural diagnostics are: whether trajectories remain bounded (they must, if `Q` is to be
trapped in `R`); whether the long-term motion is steady, periodic, quasi-periodic, or nonperiodic;
whether two trajectories started from nearby initial states stay close or diverge, measured by the
phase-space distance between them as a function of time; and how a small phase-space volume evolves,
read from the divergence of the flow field `Σ ∂(dXᵢ/dt)/∂Xᵢ`. Integration is by a finite-difference
scheme with a fixed small time step on a digital computer of the era (single-precision arithmetic,
printed output truncated to a few decimal places to save space). The convection model carries the
dimensionless parameters: a Prandtl number `σ = ν/κ`, a relative Rayleigh number `r = Ra/Rc`, and a
geometric constant `b = 4/(1+a²)`. The yardstick for "nonperiodic" is the failure of the trajectory,
or of a derived sequence such as successive maxima of one coordinate, to repeat; the yardstick for
"sensitive dependence" is the growth of the separation between neighboring trajectories.

## Code framework

The primitives already available are floating-point arithmetic, a routine to evaluate the right-hand
side of an ODE system at a state, a stable multi-stage time-stepping rule to advance the state, and a
loop to integrate over many steps and record the trajectory. The open slot is the finite ODE itself:
the variables, the right-hand side, the parameter values, and the diagnostics that reveal how its
trajectories behave.

```python
def deriv(s):
    # TODO: choose the state variables and fill in the right-hand side F(s).
    pass

def rk4_step(s, dt):
    k1 = deriv(s)
    k2 = deriv(tuple(s[i] + 0.5*dt*k1[i] for i in range(len(s))))
    k3 = deriv(tuple(s[i] + 0.5*dt*k2[i] for i in range(len(s))))
    k4 = deriv(tuple(s[i] + dt*k3[i] for i in range(len(s))))
    return tuple(s[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                 for i in range(len(s)))

def run(s0, dt=0.01, n=4000):
    s = s0
    traj = [s]
    for _ in range(n):
        s = rk4_step(s, dt)
        traj.append(s)
    return traj

def diagnostics(traj, other=None):
    # TODO: measure boundedness, contraction, recurrence/nonperiodicity,
    # and separation from a nearby trajectory.
    pass

# TODO: choose constants, initial states, and diagnostics for boundedness,
# periodicity, volume contraction, and separation of nearby trajectories.
```
