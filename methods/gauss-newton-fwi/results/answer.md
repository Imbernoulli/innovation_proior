# Gauss-Newton Full-Waveform Inversion

## Problem

Reconstruct a spatial wave-speed model of the subsurface, c(x) (equivalently the squared slowness
m = 1/c²), from recorded seismograms. Sources are fired at the surface and an array of receivers
records the waveforms. Fit the data in the least-squares sense: minimize the L2 waveform misfit
between predicted and observed traces over the model,

    J(m) = ½ Σ_s ∫₀ᵀ ‖ S u_s(·,t) − d_s(t) ‖² dt,

where, for each shot s, u_s solves the acoustic wave equation in model m and S samples the wavefield
at the receivers. The model has millions of grid-point parameters and one scalar misfit; the central
difficulty is computing ∂J/∂m affordably, and reaching the correct minimum despite a highly
non-convex misfit.

## Key idea

Treat the wave equation as a constraint F(u,m)=0 and apply the adjoint-state method. The gradient of
the waveform misfit with respect to *all* model parameters is obtained from **one forward wavefield
plus one adjoint wavefield**, independent of the number of parameters, instead of one simulation per
parameter. The adjoint of the wave operator is the wave equation run **backward in time, sourced by
the data residual injected at the receiver positions**. For the absorbing-boundary damping term, the
adjoint stencil must transpose the first time derivative, so its sign flips in the reverse-time
operator. With the change of variable q(t)=λ(T−t), the lossless part is just another forward solve.
Because the model parameter m=1/c² multiplies ∂²u/∂t² in the wave
operator, the gradient is the **zero-lag time correlation of the forward wavefield's second time
derivative with the back-propagated residual wavefield**:

    ∂J/∂m(x) = − Σ_s ∫₀ᵀ λ_s(x,t) ∂²u_s(x,t)/∂t² dt   (m = 1/c²),

with the adjoint field λ_s solving the *same* wave operator,

    m ∂²λ_s/∂t² − Δλ_s = Σ_r S^T_{s,r}(S_{s,r} u_s − d_{s,r}),   λ_s(T)=0, ∂_t λ_s(T)=0.

This is the imaging/migration principle read as an exact derivative. For a velocity
parameterization the chain rule m=1/c² (∂m/∂c=−2/c³) gives

    ∂J/∂c(x) = (2/c³) Σ_s ∫₀ᵀ λ_s(x,t) ∂²u_s(x,t)/∂t² dt.

Steepest descent on this gradient is mis-scaled by the geometrical spreading of the waves
(over-updating shallow, under-updating deep). The **Gauss–Newton** model J=½‖r‖² with residual
Jacobian G=∂r/∂m and Hessian H_GN = Re(GᴴG) (dropping the multiple-scattering second-derivative term)
corrects this: its diagonal
is the zero-lag autocorrelation of the partial-derivative wavefields — a source-illumination
preconditioner — and the full step δm = −H_GN⁻¹∇J is solved matrix-free by conjugate gradients
(each H_GN-vector product = one Born forward-sensitivity solve + one adjoint solve).

The L2 misfit over oscillatory seismograms is severely non-convex: **cycle skipping** occurs when a
predicted arrival is more than half a period off its observed twin, trapping a local optimizer in a
spurious minimum that matches the wrong cycle. Since lower frequencies have longer periods (a wider
half-period tolerance), the remedy is **multiscale frequency continuation**: invert the lowest
frequencies first to recover the smooth background velocity, then progressively add higher-frequency
bands, so each scale starts within half a period of the data.

## Algorithm

Per shot s, per outer iteration:
1. **Forward solve.** March m ∂²u_s/∂t² − Δu_s = f_s forward in time; record the trace S u_s at
   receivers and store the wavefield u_s.
2. **Residual.** r_s = S u_s − d_s^obs; accumulate J += ½‖r_s‖².
3. **Adjoint solve.** March the transposed wave stencil backward in time, sourced by the residual
   injected at the receivers.
4. **Gradient (zero-lag correlation).** Accumulate g(x) += −∫ λ_s(x,t) ∂²u_s(x,t)/∂t² dt over time and
   shots; in the Devito reverse loop this is implemented equivalently as `grad += -u * v.dt2`.
   (Multiply g by −2/c³ if updating velocity, so ∂J/∂c = (2/c³)∫λ∂²u/∂t² dt.)
5. **Precondition.** Divide g by a diagonal pseudo-Hessian based on virtual-source illumination
   such as Σ_s∫(∂²u_s/∂t²)²dt, or take a matrix-free Gauss–Newton step via CG.
6. **Update.** Line-search a step α and set m ← m − α·g_pc.

Wrap steps 1–6 in a multiscale loop: low-pass the wavelet/data to the lowest band, iterate, raise the
corner frequency, repeat.

## Code

The core Devito implementation is the acoustic stencil, forward operator, adjoint-gradient operator,
and Born operator used for Gauss–Newton products. The residual stored in `rec` is `d_pred - d_obs`,
matching the derivative of the least-squares objective.

```python
from devito import Eq, Function, Inc, Operator, TimeFunction, solve
from devito.tools import memoized_meth

def laplacian(field, model, kernel):
    """Spatial operator H(u); OT4 adds the double-laplacian correction."""
    if kernel not in ("OT2", "OT4"):
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1 / model.m) if kernel == "OT4" else 0
    return field.laplace + s**2 / 12.0 * biharmonic

def iso_stencil(field, model, kernel, forward=True, q=0):
    """model.m * u.dt2 - H(u) - q + damp*u.dt = 0, or its reverse-time adjoint."""
    unext = field.forward if forward else field.backward
    udt = field.dt if forward else field.dt.T
    eq_time = solve(model.m * field.dt2 - laplacian(field, model, kernel)
                    - q + model.damp * udt, unext)
    return [Eq(unext, eq_time, subdomain=model.grid.subdomains["physdomain"])]

def ForwardOperator(model, geometry, space_order=4, save=False, kernel="OT2", **kwargs):
    m = model.m
    u = TimeFunction(name="u", grid=model.grid, save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec
    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel, forward=True)
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)
    rec_term = rec.interpolate(expr=u)
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name="Forward", **kwargs)

def GradientOperator(model, geometry, space_order=4, save=True, kernel="OT2", **kwargs):
    m = model.m
    grad = Function(name="grad", grid=model.grid)
    u = TimeFunction(name="u", grid=model.grid, save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    v = TimeFunction(name="v", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = geometry.rec
    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    if kernel == "OT2":
        gradient_update = Inc(grad, -u * v.dt2)
    elif kernel == "OT4":
        gradient_update = Inc(grad, -u * v.dt2
                              - s**2 / 12.0 * u.biharmonic(m**(-2)) * v)
    else:
        raise ValueError("unknown acoustic kernel")

    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)
    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                    name="Gradient", **kwargs)

def BornOperator(model, geometry, space_order=4, kernel="OT2", **kwargs):
    """Apply the residual Jacobian G to a model perturbation dm."""
    m = model.m
    src = geometry.src
    rec = geometry.rec
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)
    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel, forward=True)
    eqn2 = iso_stencil(U, model, kernel, forward=True, q=-dm * u.dt2)
    source = src.inject(field=u.forward, expr=src * s**2 / m)
    receivers = rec.interpolate(expr=U)
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name="Born", **kwargs)

class AcousticWaveSolver:
    def __init__(self, model, geometry, kernel="OT2", space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry
        self.kernel = kernel
        self.space_order = space_order
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.dtype(1.73 * self.model.critical_dt) if self.kernel == "OT4" else self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        return ForwardOperator(self.model, self.geometry, save=save, kernel=self.kernel,
                               space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        return GradientOperator(self.model, self.geometry, save=save, kernel=self.kernel,
                                space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_born(self):
        return BornOperator(self.model, self.geometry, kernel=self.kernel,
                            space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, save=None, model=None, **kwargs):
        src = src or self.geometry.src
        rec = rec or self.geometry.rec
        u = u or TimeFunction(name="u", grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)
        model = model or self.model
        kwargs.update(model.physical_params(**kwargs))
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u,
                                          dt=kwargs.pop("dt", self.dt), **kwargs)
        return rec, u, summary

    def gradient(self, rec, u, v=None, grad=None, model=None, **kwargs):
        grad = grad or Function(name="grad", grid=self.model.grid)
        v = v or TimeFunction(name="v", grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        model = model or self.model
        kwargs.update(model.physical_params(**kwargs))
        summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u,
                                       dt=kwargs.pop("dt", self.dt), **kwargs)
        return grad, summary

    def jacobian(self, dm, src=None, rec=None, u=None, U=None, model=None, **kwargs):
        src = src or self.geometry.src
        rec = rec or self.geometry.rec
        u = u or TimeFunction(name="u", grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name="U", grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        model = model or self.model
        kwargs.update(model.physical_params(**kwargs))
        summary = self.op_born().apply(dm=dm, u=u, U=U, src=src, rec=rec,
                                       dt=kwargs.pop("dt", self.dt), **kwargs)
        return rec, u, U, summary

def fwi_objective_and_gradient(solver, observed_rec):
    predicted_rec, u, _ = solver.forward(save=True)
    residual_rec = predicted_rec
    residual_rec.data[:] = predicted_rec.data - observed_rec.data
    phi = 0.5 * float((residual_rec.data ** 2).sum())
    grad, _ = solver.gradient(rec=residual_rec, u=u)
    return phi, grad, u

def gauss_newton_matvec(solver, dm, u_bg=None):
    # G^T G dm. The adjoint Jacobian (G^T) needs the FULL saved background
    # wavefield for the -u*v.dt2 correlation, so build it once with save=True
    # and cache it across CG iterations. The Born forward (G) propagates its
    # own rolling background internally, so it does not take u_bg.
    if u_bg is None:
        u_bg = TimeFunction(name="u", grid=solver.model.grid,
                            save=solver.geometry.nt, time_order=2,
                            space_order=solver.space_order)
        solver.forward(u=u_bg, save=True)
    scattered_rec, _, _, _ = solver.jacobian(dm)
    Hg, _ = solver.gradient(rec=scattered_rec, u=u_bg)
    return Hg
```
