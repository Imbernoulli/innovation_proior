A lens is an ordered stack of refracting surfaces, each one a curvature, a glass after it, and an axial distance to the next, and the task is to take a starting design that more or less works but images badly and automatically nudge the curvatures and spacings (and sometimes the glasses) until the image is as sharp as the application demands. The forward direction is easy: pick a parameter set, trace rays through it by geometric optics — Snell's law $n_1\sin\theta_1 = n_2\sin\theta_2$ at every surface, straight-line travel by the thickness between them — and read off how far each ray lands from the ideal image point. The inverse problem is the hard half, and it is hard for three reasons at once. The map from parameters $x$ (curvatures, thicknesses, indices) to image defects is strongly nonlinear, so the ray trace cannot be inverted analytically and the procedure must be iterative. There are far more quality requirements than parameters — a spread of rays across the pupil, at several field angles, at several wavelengths because the index is wavelength-dependent, plus first-order constraints like a fixed focal length read off the paraxial trace $n_2 u' = n_1 u - y(n_2-n_1)/R$ — so the conditions are wildly overdetermined and no setting of the knobs drives every defect to zero. And the parameters act on the defects in strongly correlated, very unequally-scaled ways: bending a lens, or splitting power between two nearby curvatures, or trading a conic constant against a fourth-order aspheric term, all move several aberrations together in nearly the same way.

The honest way to handle an overdetermined set of defects is least squares. I define a residual for each defect operand, $f_i(x) = w_i\,(a_i(x) - \tilde a_i)$, the weighted deviation of operand $a_i$ (a transverse ray aberration, a wavefront/OPD error, a Seidel coefficient, or a paraxial quantity like focal length) from its target $\tilde a_i$, and minimize the scalar merit function $f(x) = \sum_i f_i(x)^2$. Squares are right because a ray landing $10\,\mu$m high and one landing $10\,\mu$m low are equally harmful, so sign should not matter, and because near a minimum a sum of squares is a paraboloid with a well-defined curvature to exploit. The crudest minimizer, steepest descent — step along $-\nabla f$ and line-search — is safe but crawls, because the correlated variables make the merit contours long thin valleys and the gradient method zig-zags across them instead of running down. Plain Gauss–Newton goes to the other extreme and fails outright, as I will show, so neither baseline alone is acceptable.

The method is Damped Least Squares, which is the Levenberg–Marquardt algorithm specialized to lens design. The starting point is to use the sum-of-squares structure to get curvature for free. Linearize the residual vector about the current $x$, $f(x+\Delta x) \approx f(x) + J\,\Delta x$ with Jacobian $J_{ij} = \partial f_i/\partial x_j$, and the local objective $\|f + J\Delta x\|^2$ is a linear least-squares problem in the step. Expanding $\|f + J\Delta x\|^2 = f^\top f + 2(J\Delta x)^\top f + \Delta x^\top J^\top J\,\Delta x$ and setting the gradient in $\Delta x$ to zero gives the normal equations and the Gauss–Newton step
$$J^\top J\,\Delta x = -J^\top f \quad\Longrightarrow\quad \Delta x = -(J^\top J)^{-1} J^\top f.$$
The true Hessian of $\sum f_i^2$ is $2J^\top J + 2\sum_i f_i \nabla^2 f_i$; Gauss–Newton drops the second term, which carries the operands' second derivatives — expensive and noisy to estimate, and negligible whenever the residuals are small (near a good solution) or the operands are mildly curved. So $J^\top J$, costing nothing beyond $J$ itself, stands in as a free positive-semidefinite curvature model. The Jacobian itself comes from finite differences, since the ray trace is a black box: perturb one variable by a small $\delta$, retrace the whole ray set, take $(f(x+\delta e_j) - f(x))/\delta$ for column $j$ — one extra trace pass per variable, affordable because there are only a handful of variables.

Run undamped Gauss–Newton on a real design and it falls over: the step comes out enormous, the merit gets worse instead of better, and sometimes a ray totally-internally-reflects or misses a surface so the trace fails and the merit is undefined. The pathology lives in $(J^\top J)^{-1}$, and the cause is exactly the correlation that makes the valley thin. When two variables push the aberrations in nearly the same direction, two columns of $J$ are nearly parallel, $J^\top J$ has an eigenvalue near zero, and its inverse divides by it. The singular value decomposition $J = U\Sigma V^\top$ makes this clean: the step is $\Delta x = -\sum_k (u_k^\top f / \sigma_k)\,v_k$, so each parameter-space direction $v_k$ gets a coefficient with $\sigma_k$ in the denominator. For a near-redundant combination $\sigma_k$ is tiny, $1/\sigma_k$ is gigantic, and the step takes a colossal excursion along a direction the aberrations barely respond to — far outside the neighborhood where the linear model was any good, where the true merit goes up or the rays stop tracing.

So I leash the step length directly: rather than the step that minimizes $\|f + J\Delta x\|^2$ over all of parameter space, I want the one that minimizes it subject to $\|\Delta x\|^2 \le h^2$, staying inside the trust region where the linearization holds. A Lagrange multiplier turns the constrained quadratic into the unconstrained $\|f + J\Delta x\|^2 + \lambda\|\Delta x\|^2$ — equivalently, replace the merit by a damped version $f_D = f + \lambda|\Delta x|^2$ that punishes large steps. Setting the gradient of this to zero gives the damped normal equations
$$(J^\top J + \lambda I)\,\Delta x = -J^\top f \quad\Longrightarrow\quad \Delta x = -(J^\top J + \lambda I)^{-1} J^\top f.$$
That single $+\lambda I$ is the whole fix, and the SVD certifies it: now $J^\top J + \lambda I = V(\Sigma^2 + \lambda I)V^\top$ and the per-direction coefficient changes from $1/\sigma_k$ to $\sigma_k/(\sigma_k^2 + \lambda)$. Where the direction is well-determined, $\sigma_k^2 \gg \lambda$ and this is $\approx 1/\sigma_k$, the full Gauss–Newton step untouched, so I keep fast convergence in the trustworthy directions; where the direction is soft, $\sigma_k^2 \ll \lambda$ and this is $\approx \sigma_k/\lambda \to 0$, so the exploding directions are suppressed instead of amplified. An exactly singular direction ($\sigma_k = 0$) now gets coefficient $0$, not infinity, and the damping tames precisely the directions that were blowing the step up while leaving the healthy ones alone.

The two limits of $\lambda$ are exactly the two methods I was torn between, which is why a single dial resolves the tension. As $\lambda \to 0$, $(J^\top J + \lambda I)^{-1} \to (J^\top J)^{-1}$ and I recover Gauss–Newton — fast. As $\lambda \to \infty$, $J^\top J$ becomes negligible and $\Delta x \to -(1/\lambda)J^\top f$; since $J^\top f$ is the merit gradient ($\nabla(\tfrac12\|f\|^2) = J^\top f$), a huge $\lambda$ gives a tiny step straight downhill — steepest descent with step size $1/\lambda$. So $\lambda$ slides continuously between the safe-slow gradient method and the fast-fragile Gauss–Newton one, per iteration, by one number. I pick $\lambda$ by trust measured from results: choose a $\lambda$, solve, apply $\Delta x$, retrace, compare the new merit to the old. If the merit fell, the step was trustworthy — accept it and shrink $\lambda$ to lean toward Gauss–Newton next time; if it rose or the trace failed, the step over-reached — reject it, grow $\lambda$ (e.g. $\times 10$), and re-solve the same iteration with more damping, which both shortens the step and rotates it toward the gradient. Because $J^\top J + \lambda I$ is positive-definite for any $\lambda > 0$, the solve never fails for being singular no matter how degenerate $J$ is, so growing $\lambda$ is guaranteed to eventually produce a short enough, downhill enough step that the merit decreases unless I am already at a minimum.

One refinement matters for optics specifically. A single scalar $\lambda$ on $I$ damps every variable the same, but a curvature, an air space, and a glass index live on wildly different scales — penalizing all of them with the same $\lambda\|\Delta x\|^2$ uses a round ball where the natural trust region is a long thin ellipsoid, over-damping insensitive variables and under-damping sensitive ones, which just transplants the conditioning problem into the damping. The cure is to measure the penalized length in a metric that respects each variable's sensitivity: nondimensionalize the variables up front by a per-variable linear scaling (e.g. curvatures in units of $\sim\!1/100\ \mathrm{mm}^{-1}$) so $\lambda I$ is roughly isotropic in the right metric, or equivalently damp with $\lambda\,\mathrm{diag}(J^\top J)$ so each variable's damping is proportional to its own sensitivity. Either way the trust region becomes an ellipsoid aligned with the problem's scales and convergence speeds up markedly; I keep both levers, normalizing the variables and applying the damping in those normalized units. A second optical guard: when a ray fails and an operand returns a non-finite value, I must not let it poison the merit comparison, so any non-finite residual evaluation is replaced by a large finite penalty — a wall the optimizer feels and backs away from, which in the $\lambda$ loop simply reads as "merit went up, reject, increase damping." Finally, an honest caveat that sets expectations rather than changing the step: the whole construction is local, descending to the minimum in the starting design's basin, and the optical merit landscape has many minima for different lens forms, so the starting configuration and the staging of which variables are freed still decide which basin I land in. The step rule's job is only to reach the bottom of that basin reliably and fast without diverging — and the damping is what buys that.

```python
import numpy as np
from scipy import optimize


class Operand:
    """One image-defect condition. residual = weight * (value - target)."""
    def __init__(self, kind, target, weight, input_data):
        self.kind, self.target, self.weight = kind, target, weight
        self.input_data = input_data

    def value(self):
        # real-ray transverse aberration / OPD, a Seidel coefficient,
        # or a paraxial quantity (focal length, image location), via ray trace
        return evaluate_operand(self.kind, self.input_data)

    def delta(self):
        return self.value() - self.target

    def residual(self):
        return self.weight * self.delta()              # f_i = w_i (a_i - target_i)


class Variable:
    """A freed parameter, kept in normalized units so damping is isotropic."""
    def __init__(self, optic, kind, surface_number, scale=1 / 100.0, offset=-1.0):
        self.optic, self.kind = optic, kind
        self.surface_number = surface_number
        self.scale, self.offset = scale, offset        # per-variable normalization

    def get(self):
        raw = read_parameter(self.optic, self.kind, self.surface_number)
        return raw * self.scale + self.offset

    def update(self, scaled_value):
        raw = (scaled_value - self.offset) / self.scale
        write_parameter(self.optic, self.kind, self.surface_number, raw)


class OptimizationProblem:
    def __init__(self):
        self.operands = []                              # residuals f_i  (m of them)
        self.variables = []                             # parameters x_j (n; m >> n)

    def add_operand(self, kind, target, weight, input_data):
        self.operands.append(Operand(kind, target, weight, input_data))

    def add_variable(self, optic, kind, surface_number, **kw):
        self.variables.append(Variable(optic, kind, surface_number, **kw))

    def residual_vector(self):
        return np.array([op.residual() for op in self.operands])   # f(x), unsquared

    def sum_squared(self):
        r = self.residual_vector()
        return float(r @ r)                             # merit f(x) = sum f_i^2

    def update_optics(self):
        for optic in {v.optic for v in self.variables}:
            optic.update()                              # propagate solves / pickups


class DampedLeastSquares:
    """DLS / Levenberg-Marquardt: (J^T J + lambda I) dx = -J^T f, lambda adapted by trust."""
    def __init__(self, problem):
        self.problem = problem

    def _residuals(self, x):
        for xi, var in zip(x, self.problem.variables):
            var.update(xi)
        self.problem.update_optics()
        f = self.problem.residual_vector()
        if not np.all(np.isfinite(f)):                  # ray failure -> finite penalty wall
            big = np.sqrt(1e10 / max(len(f), 1))
            return np.full(len(f), big)
        return f

    def optimize(self, maxiter=None, tol=1e-3):
        x0 = np.array([v.get() for v in self.problem.variables])
        # method='lm' = damped least squares: finite-difference Jacobian J,
        # damped normal equations (J^T J + lambda I) dx = -J^T f, and the
        # lambda up/down trust loop. Needs m >= n (many operands, few variables).
        result = optimize.least_squares(
            self._residuals, x0, method="lm", max_nfev=maxiter, ftol=tol,
        )
        for xi, var in zip(result.x, self.problem.variables):
            var.update(xi)
        self.problem.update_optics()
        return result
```
