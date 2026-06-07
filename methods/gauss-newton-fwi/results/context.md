# Context: Reconstructing a subsurface velocity model from recorded seismograms

## Research question

A seismic survey fires controlled sources at the surface, and an array of receivers records the
ground motion — the seismograms — as the elastic/acoustic waves propagate down, reflect and refract
off the layering and heterogeneity of the earth, and come back up. Encoded in the precise arrival
times, amplitudes, and waveform shapes of those traces is the spatial distribution of the wave speed
in the subsurface, c(x). The goal is to recover that velocity model — a field defined on a fine
computational grid, so millions of unknown numbers — directly from the recorded waveforms, by
*fitting* the data: find the model whose simulated seismograms match the observed ones.

Formally: pick a model parameter field m(x) (the squared slowness m = 1/c², or the velocity c
itself), let u_s(x,t) be the wavefield that the acoustic wave equation produces for shot s in that
model, and sample it at the receiver positions. Minimize the least-squares waveform misfit between
predicted and observed traces,

    J(m) = ½ Σ_s ∫₀ᵀ ‖ S u_s(·,t) − d_s(t) ‖² dt,

over m. To descend this misfit we need its gradient ∂J/∂m at every grid point. The hard requirement
is that obtaining this gradient must stay affordable **even though m has millions of components** and
each evaluation of J already costs a full wave-propagation simulation per shot. A method that costs
one wave simulation *per model parameter* is hopeless — that is the wall. A second requirement is
that the descent actually reach the right basin: the misfit over an oscillatory wavefield is
violently non-convex, and a naive local optimizer can converge to a wrong model that fits a shifted
copy of the data.

## Background

The forward model is the (constant-density) acoustic wave equation. With m = 1/c² the squared
slowness, a source f_s(x,t) at the shot location, and homogeneous initial conditions,

    m ∂²u_s/∂t² − Δu_s = f_s,    u_s(·,0) = 0,  ∂_t u_s(·,0) = 0,

solved by finite differences on a grid (an explicit leapfrog time march, second/fourth-order spatial
Laplacian, absorbing/PML layers at the artificial boundaries). The predicted trace at receiver r is
the wavefield sampled there, S_{s,r} u_s. By the time large surveys were being inverted, this forward
solve was the established, well-understood unit of computation — but it *is* the unit: everything in
a model-fitting loop is counted in multiples of one forward wave simulation per shot, and a survey
has thousands of shots.

The dominant, load-bearing fact of the setting: a finite-difference gradient would perturb each of
the M model parameters in turn and re-simulate the wavefield, costing M+1 forward solves per gradient.
With M in the millions this is astronomically prohibitive — it is *the* obstacle. The same scaling
defeats forward (tangent) differentiation: differentiating the wave-equation constraint gives a
linear sensitivity equation that must be solved once per model parameter, again M solves. Whichever
way one pushes perturbations *forward* through the simulation, the count of model parameters sits in
the expensive part.

The misfit's non-convexity is the other load-bearing phenomenon, and it has a precise, knowable
cause. Seismic signals are oscillatory — they are wave trains with a dominant period. When the
predicted trace for a candidate model is shifted in time relative to the observed trace by more than
**half a period**, the least-squares misfit prefers to line up the predicted wiggle with the
*neighbouring* cycle of the observed wiggle rather than the correct one. The gradient then points
toward matching the wrong cycle, and a local optimizer descends into a spurious minimum. This is
**cycle skipping**, and the half-period criterion is sharp: matching the data at the receivers to
within half a period is a necessary condition for the local optimization to head toward the true
model. A direct consequence: the *lower* the frequency, the *longer* the period, so the wider the
window of model error that stays within half a period — low-frequency data are far less prone to
cycle skipping than high-frequency data. The misfit surface for the smooth, long-wavelength part of
the model, probed by low frequencies, is comparatively benign; the high-frequency detail is what
makes it a minefield.

A third fact concerns conditioning. The waves spread geometrically as they propagate, so their
amplitude — and hence the raw gradient's amplitude — decays with depth and distance from the
sources; near-source grid points are illuminated far more strongly than deep ones. A raw
steepest-descent step therefore over-updates the shallow model and barely touches the deep model. The
second-order information that would correct this is the Hessian of the misfit; its dominant,
geometrical-spreading part is what a usable method must somehow account for, exactly or approximately,
to get a well-scaled update.

## Baselines

**Finite-difference / perturbational gradient.** Treat J(m) as a black box, perturb each model
parameter, re-simulate, difference. Core idea: estimate ∂J/∂m_i ≈ (J(m+δ e_i) − J(m))/δ
component-by-component. Exact in the limit, conceptually trivial, and the obvious thing to try. Its
decisive limitation is cost: M+1 forward wave simulations per gradient, with M in the millions —
unusable at field scale — and it inherits a step-size dilemma (too large contaminates with
curvature, too small drowns in the solver's discretization noise).

**Forward (tangent) sensitivity.** Differentiate the converged forward constraint to propagate each
model perturbation forward through the linearized wave equation, solving (∂F/∂u)(∂u/∂m_i) = −∂F/∂m_i
once per parameter, then assembling ∂J/∂m_i. Core idea: exact directional derivatives without
finite-difference noise. It removes the step-size problem but **keeps the M-solves scaling** — one
linear wave solve per model parameter. It is on the right side of the trade only when outputs vastly
outnumber inputs; here there is a single scalar misfit and millions of inputs, so it is on the wrong
side.

**Optimal control of PDEs (Lions 1971, *Optimal Control of Systems Governed by Partial Differential
Equations*).** The abstract machinery for minimizing a functional under a PDE constraint: introduce
an adjoint (costate) field as a Lagrange multiplier defined over the domain; stationarity in the
state yields an adjoint PDE for the costate; the remaining variation gives the gradient of the
functional with respect to the control, without perturbing each control separately. Provides the
theory but stated abstractly — not tied to the hyperbolic wave equation, and giving no recipe for the
adjoint's source and final/boundary conditions that a time-dependent wave problem needs.

**Costate gradients in trajectory optimization (Bryson & Ho 1975, *Applied Optimal Control*).** For
control of an ODE system, the gradient of a cost with respect to *all* controls is obtained by
integrating a costate equation **backward in time** in a single sweep — one backward integration
regardless of the number of controls. The concrete template of "one extra (backward) solve gives the
whole gradient," but living in time-dependent ODE control, not in a spatial PDE field inversion.

**Adjoint-state method in inverse problems (Chavent 1974; the migration interpretation, Lailly
1983).** The adjoint-state idea carried into geophysical inverse problems: the gradient of the
least-squares waveform misfit is obtained by **back-propagating the data residual** into the earth
and correlating it with the forward (source) wavefield — the gradient *is* a migration / imaging
operator. Core idea: the residual, injected at the receivers and propagated by the adjoint of the
wave operator, plays the role of the costate field; its correlation with the incident wavefield gives
the model update. This is the imaging principle (Claerbout) read as a gradient: the source wavefield
and a receiver-side wavefield correlated at zero time lag light up the reflectors. The open thing is
to assemble this into a *full* iterative waveform-fitting scheme — the exact gradient with the right
time-derivative weighting, second-order (Gauss–Newton) scaling, and the cure for the non-convexity.

**Gauss–Newton / Newton in frequency-space inversion (Pratt, Shin & Hicks 1998, *Gauss–Newton and
full Newton methods in frequency-space seismic waveform inversion*).** The least-squares structure
J = ½‖r(m)‖² admits a Gauss–Newton model: linearize the residual r about the current model via its
Fréchet (Born) derivative J = ∂r/∂m, approximate the Hessian by H_GN = Re(JᴴJ), and take the step
δm = −H_GN⁻¹ ∇J. Core idea: the Gauss–Newton Hessian deconvolves the gradient by the geometrical
spreading and illumination, giving a far better-scaled update than steepest descent; its diagonal is
the zero-lag autocorrelation of the partial-derivative wavefields. Its limitation at field scale is
that H_GN is enormous and dense — forming or inverting it is itself the bottleneck — so practical
schemes need a cheap surrogate (a diagonal pseudo-Hessian / source-illumination preconditioner, or a
matrix-free conjugate-gradient solve of the Gauss–Newton system).

## Evaluation settings

The natural model problem is a **2D acoustic transmission / reflection experiment** on a small grid:
a known "true" velocity model (a smooth background with a localized anomaly — e.g. a circular
high-velocity inclusion, or a layered model with a salt-like body), a line of sources and a line of
receivers, a Ricker (second-derivative-of-Gaussian) source wavelet of a chosen peak frequency. The
"observed" data are generated by running the forward solver in the true model; inversion starts from
a smooth or constant initial model and tries to recover the true one. Because the true answer is
known, the gradient can be checked two ways: an **adjoint dot-test** (forward and adjoint operators
must be exact transposes to machine precision) and a **finite-difference gradient check** (the
adjoint gradient must match a directional finite difference, and the misfit must decrease to first
order along −∇J). The full-scale settings are 2D/3D field surveys with thousands of shots, inverted
under the acoustic, then elastic, equations, with the velocity model parameterized on the simulation
grid and the misfit measured per shot. The metrics are the misfit itself and the per-iteration cost
expressed in forward-solve units.

## Code framework

The primitives that already exist: a finite-difference acoustic wave-equation forward solver that,
given a model and a source, marches the wavefield in time and samples it at receivers to produce a
shot record; a routine that forms the data residual against observed traces and evaluates the
least-squares misfit; and a generic gradient-descent loop with a line search. The one empty slot is
the gradient of the misfit with respect to the millions of model parameters — everything routes
through `gradient`, whose interesting branch is a `# TODO`.

```python
import numpy as np

# --- existing: forward acoustic FD modeler ---
def forward(model, src_wavelet, src_pos, rec_pos, geom, save_wavefield=False):
    """March  m u_tt - Lap(u) = f  in time; return the shot record at rec_pos
    (and optionally the full forward wavefield u[t,x,z] for later reuse)."""
    # ... leapfrog time loop, absorbing boundaries, source injection, receiver sampling ...
    pass

def residual_and_misfit(d_pred, d_obs):
    """r = d_pred - d_obs ;  J = 1/2 ||r||^2 (summed over receivers and time)."""
    r = d_pred - d_obs
    J = 0.5 * np.sum(r * r)
    return r, J

# --- gradient: the slot the method fills ---
def gradient_FD(model, ...):
    """Baseline: perturb each model point, re-run forward, difference. M+1 solves. Hopeless at scale."""
    pass

def gradient(model, src_wavelet, src_list, rec_pos, geom, d_obs):
    # TODO: a gradient of J w.r.t. ALL model parameters at the cost of ~one extra
    #       wave simulation per shot, independent of the number of model parameters.
    pass

# --- existing: generic descent loop with a line search ---
def invert(model0, src_wavelet, src_list, rec_pos, geom, d_obs, n_iter):
    model = model0.copy()
    for k in range(n_iter):
        g = gradient(model, src_wavelet, src_list, rec_pos, geom, d_obs)
        # TODO: descent direction from the gradient (and any conditioning / Hessian scaling of it)
        step = line_search(...)               # backtracking on the misfit
        model = model + step                  # update the velocity model
    return model
```
