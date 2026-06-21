We need to recover a subsurface wave-speed model from recorded seismograms. The forward problem is straightforward: given a velocity model, we can march the acoustic wave equation and predict the traces at the receivers. The inverse problem is to find the model whose predicted traces match the observed ones in the least-squares sense. The brute-force approach, perturbing each grid point and re-running the forward simulation, is impossible because the model has millions of parameters and each wave solve is expensive. The same cost wall blocks tangent-sensitivity methods, which still require one linearized wave solve per model parameter. The central obstacle is therefore not the physics but the cost of obtaining the full gradient.

The waveform misfit is also violently non-convex. Seismic traces are oscillatory, and if a predicted arrival is shifted by more than half a period relative to the observed arrival, least-squares prefers to match the wrong cycle, dragging the optimizer into a spurious basin. A method that simply descends the local gradient will fail unless the starting model is already very close. The lower the frequency, the longer the period and the wider the basin of convergence, so the only reliable path is to invert low frequencies first and progressively add higher frequencies once the smooth background is recovered.

The method is Gauss–Newton Full-Waveform Inversion. It treats the wave equation as a constraint and uses the adjoint-state method to compute the gradient of the waveform misfit with respect to every model parameter from just two wave solves per shot: one forward solve and one adjoint solve. The adjoint equation is the transposed wave equation, run backward in time and sourced by the data residual injected at the receivers. Because the squared slowness multiplies the second time derivative in the wave operator, the gradient is the zero-lag time correlation between the forward wavefield and the back-propagated residual. This removes the parameter count from the expensive part of the computation entirely.

To fix the poor conditioning caused by geometrical spreading, the method uses the Gauss–Newton Hessian. Writing the misfit as one-half the squared residual norm, the Gauss–Newton Hessian is the real part of the Jacobian-adjoint-Jacobian product. Its diagonal acts as an illumination preconditioner that rescales deep, poorly lit parts of the model, while the full step can be applied matrix-free by conjugate gradients using the Born operator for Jacobian products. The outer loop proceeds through a frequency continuation schedule, starting with a low-passed wavelet and data, fitting the long-wavelength model, then raising the cutoff and refining the details while staying within the half-period basin.

```python
from devito import Eq, Function, Inc, Operator, TimeFunction, solve
from devito.tools import memoized_meth

def laplacian(field, model, kernel):
    if kernel not in ("OT2", "OT4"):
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1 / model.m) if kernel == "OT4" else 0
    return field.laplace + s**2 / 12.0 * biharmonic

def iso_stencil(field, model, kernel, forward=True, q=0):
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
    m = model.m
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)
    src = geometry.src
    rec = geometry.rec
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
        return (self.model.dtype(1.73 * self.model.critical_dt)
                if self.kernel == "OT4" else self.model.critical_dt)

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
    if u_bg is None:
        u_bg = TimeFunction(name="u", grid=solver.model.grid,
                            save=solver.geometry.nt, time_order=2,
                            space_order=solver.space_order)
        solver.forward(u=u_bg, save=True)
    scattered_rec, _, _, _ = solver.jacobian(dm)
    Hg, _ = solver.gradient(rec=scattered_rec, u=u_bg)
    return Hg
```
