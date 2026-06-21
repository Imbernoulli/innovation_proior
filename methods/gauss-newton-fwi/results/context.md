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

over m. To descend this misfit one needs its gradient ∂J/∂m at every grid point, where m has
millions of components and each evaluation of J already costs a full wave-propagation simulation per
shot, with a survey holding thousands of shots. The question is how to compute that gradient and
descend the misfit, which over an oscillatory wavefield is strongly non-convex.

## Background

The forward model is the (constant-density) acoustic wave equation. With m = 1/c² the squared
slowness, a source f_s(x,t) at the shot location, and homogeneous initial conditions,

    m ∂²u_s/∂t² − Δu_s = f_s,    u_s(·,0) = 0,  ∂_t u_s(·,0) = 0,

solved by finite differences on a grid (an explicit leapfrog time march, second/fourth-order spatial
Laplacian, absorbing/PML layers at the artificial boundaries). The predicted trace at receiver r is
the wavefield sampled there, S_{s,r} u_s. This forward solve is the established, well-understood unit
of computation, and the cost of a model-fitting loop is counted in multiples of one forward wave
simulation per shot.

A finite-difference gradient perturbs each of the M model parameters in turn and re-simulates the
wavefield, costing M+1 forward solves per gradient. Forward (tangent) differentiation has the same
character: differentiating the wave-equation constraint gives a linear sensitivity equation solved
once per model parameter, again M solves. Pushing perturbations *forward* through the simulation
places the count of model parameters in the expensive part.

The misfit's non-convexity has a precise, knowable cause. Seismic signals are oscillatory — they are
wave trains with a dominant period. When the predicted trace for a candidate model is shifted in time
relative to the observed trace by more than **half a period**, the least-squares misfit prefers to
line up the predicted wiggle with the *neighbouring* cycle of the observed wiggle rather than the
correct one. The gradient then points toward matching the wrong cycle, and a local optimizer descends
into a spurious minimum. This is **cycle skipping**, and the half-period criterion is sharp: matching
the data at the receivers to within half a period is a necessary condition for the local optimization
to head toward the true model. A direct consequence: the *lower* the frequency, the *longer* the
period, so the wider the window of model error that stays within half a period — low-frequency data
are far less prone to cycle skipping than high-frequency data. The misfit surface for the smooth,
long-wavelength part of the model, probed by low frequencies, is comparatively benign.

A third fact concerns conditioning. The waves spread geometrically as they propagate, so their
amplitude — and hence the raw gradient's amplitude — decays with depth and distance from the
sources; near-source grid points are illuminated far more strongly than deep ones. A raw
steepest-descent step over-updates the shallow model and lightly touches the deep model, the step
inheriting the same depth imbalance as the illumination.

## Baselines

**Finite-difference / perturbational gradient.** Treat J(m) as a black box, perturb each model
parameter, re-simulate, difference. Core idea: estimate ∂J/∂m_i ≈ (J(m+δ e_i) − J(m))/δ
component-by-component. Exact in the limit, conceptually trivial, and the obvious thing to try. It
costs M+1 forward wave simulations per gradient and carries a step-size choice (too large contaminates
with curvature, too small drowns in the solver's discretization noise).

**Forward (tangent) sensitivity.** Differentiate the converged forward constraint to propagate each
model perturbation forward through the linearized wave equation, solving (∂F/∂u)(∂u/∂m_i) = −∂F/∂m_i
once per parameter, then assembling ∂J/∂m_i. Core idea: exact directional derivatives without
finite-difference noise — one linear wave solve per model parameter. It is suited to the regime where
outputs vastly outnumber inputs.

**Optimal control of PDEs (Lions 1971, *Optimal Control of Systems Governed by Partial Differential
Equations*).** The abstract machinery for minimizing a functional under a PDE constraint: introduce
an adjoint (costate) field as a Lagrange multiplier defined over the domain; stationarity in the
state yields an adjoint PDE for the costate; the remaining variation gives the gradient of the
functional with respect to the control, without perturbing each control separately. The theory is
stated abstractly, not tied to the hyperbolic wave equation.

**Costate gradients in trajectory optimization (Bryson & Ho 1975, *Applied Optimal Control*).** For
control of an ODE system, the gradient of a cost with respect to *all* controls is obtained by
integrating a costate equation **backward in time** in a single sweep — one backward integration
regardless of the number of controls. The setting is time-dependent ODE control.

**Adjoint-state method in inverse problems (Chavent 1974; the migration interpretation, Lailly
1983).** The adjoint-state idea in geophysical inverse problems: the gradient of the least-squares
waveform misfit is obtained by **back-propagating the data residual** into the earth and correlating
it with the forward (source) wavefield — the gradient *is* a migration / imaging operator. Core idea:
the residual, injected at the receivers and propagated by the adjoint of the wave operator, plays the
role of the costate field; its correlation with the incident wavefield gives the model update. This is
the imaging principle (Claerbout) read as a gradient: the source wavefield and a receiver-side
wavefield correlated at zero time lag light up the reflectors. In the migration literature it is
presented as a stand-alone imaging step.

**Gauss–Newton / Newton in frequency-space inversion (Pratt, Shin & Hicks 1998, *Gauss–Newton and
full Newton methods in frequency-space seismic waveform inversion*).** The least-squares structure
J = ½‖r(m)‖² admits a Gauss–Newton model: linearize the residual r about the current model via its
Fréchet (Born) derivative G = ∂r/∂m, approximate the Hessian by H_GN = Re(GᴴG), and take the step
δm = −H_GN⁻¹ ∇J. Core idea: the Gauss–Newton Hessian deconvolves the gradient by the geometrical
spreading and illumination, giving a better-scaled update than steepest descent; its diagonal is the
zero-lag autocorrelation of the partial-derivative wavefields. At field scale with millions of model
parameters, H_GN is enormous and dense.

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

The primitives that already exist: a finite-difference acoustic wave-equation stencil, sparse source
injection and receiver interpolation, a forward operator that can save the wavefield, a routine that
forms the data residual against observed traces and evaluates the least-squares misfit, and a generic
descent loop with a line search. The empty computational slot is the model-space derivative of that
misfit — how to turn the residual into an update of every grid-cell parameter.

```python
from devito import Function

def acoustic_stencil(field, model, kernel="OT2", forward=True, q=0):
    """Finite-difference update for model.m * u.dt2 - H(u) - q + damping = 0."""
    # existing time-stepping stencil, including absorbing-boundary damping
    pass

def forward_operator(model, geometry, space_order=4, save=False, kernel="OT2"):
    """Inject the physical source, march the wavefield, and interpolate receiver traces."""
    pass

def residual_misfit(d_pred, d_obs):
    r = d_pred - d_obs
    return r, 0.5 * float((r * r).sum())

class AcousticWaveSolver:
    def __init__(self, model, geometry, kernel="OT2", space_order=4):
        self.model = model
        self.geometry = geometry
        self.kernel = kernel
        self.space_order = space_order

    def forward(self, src=None, rec=None, u=None, save=None, **kwargs):
        """Run the existing forward operator; optionally save the wavefield u."""
        pass

    def model_gradient(self, residual, u, grad=None, **kwargs):
        # TODO: derivative of the misfit with respect to every grid-cell parameter.
        pass

def descent_direction(g, solver, u_by_shot=None):
    # TODO: map the raw gradient to a descent direction.
    pass

def invert(model, solver, observed_data, n_iter):
    for k in range(n_iter):
        g = Function(name="grad", grid=model.grid)
        saved_wavefields = []
        objective = 0.0
        for shot, d_obs in observed_data:
            d_pred, u, _ = solver.forward(src=shot, save=True)
            r, value = residual_misfit(d_pred.data, d_obs)
            objective += value
            saved_wavefields.append(u)
            solver.model_gradient(r, u, grad=g)
        p = descent_direction(g, solver, saved_wavefields)
        alpha = line_search(model, p, objective)
        model = apply_model_update(model, -alpha * p)
    return model
```
