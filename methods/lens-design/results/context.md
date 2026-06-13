# Context: Automatic correction of an optical system by least-squares minimization of image defects

## Research question

Given a starting optical system — an ordered stack of refracting (and possibly reflecting) surfaces, each with a curvature, a separation to the next surface, and a glass between them — find values of the chosen system parameters that make the imaging as good as the application demands.

Evaluating a *fixed* system is easy: trace rays through it by geometric optics and read off how badly they miss the ideal image point. The hard part is the inverse problem. The relation between the parameters (curvatures, thicknesses, glass indices) and the image defects is strongly nonlinear, a real system has dozens of parameters, and there are far more quality requirements than there are parameters — many rays at many field angles and wavelengths, each of which would ideally land exactly on the ideal image point, plus first-order requirements such as a prescribed focal length. No setting of the parameters drives every defect to zero simultaneously. So the goal a usable method has to achieve is: from an arbitrary tolerable starting design, automatically and repeatably move the parameters to a configuration that minimizes the image defects in a sensible *collective* sense, doing so in a way that does not blow up when the local linear picture of the system is wrong — which, for optical systems, it routinely is, because the parameters have strongly correlated and very unequally-scaled effects on the defects.

## Background

**Geometric optics and ray tracing.** The forward model is geometric optics. A ray is a position and a direction. At each surface it is refracted by Snell's law: with surface normal **n**, incident index n₁ and exit index n₂, the refracted direction follows n₁ sinθ₁ = n₂ sinθ₂, implemented in vector form so the exit direction is a function of the incident direction, the surface normal, and n₂/n₁. Between surfaces the ray travels in a straight line by the separation (thickness) along the axis. Tracing a ray is therefore a deterministic composition of these per-surface refractions; the final image-plane intersection is a smooth (but nonlinear) function of all the curvatures, thicknesses and indices it passed through. In the paraxial limit the refraction linearizes to n₂u′ = n₁u − y(n₂−n₁)/R for marginal ray height y and angle u — this gives first-order quantities (focal length, image location, pupil positions) used as constraints.

**Aberrations as the things to be driven to zero.** A perfect system images a point to a point. A real one does not; the deviation of an actual ray's image-plane intersection from the ideal is a *transverse ray aberration*. Equivalently, the wavefront departs from a sphere (wavefront aberration). For a centered system the lowest-order monochromatic aberrations are the five Seidel sums (spherical aberration, coma, astigmatism, field curvature, distortion), each computable from the paraxial trace and the surface data; chromatic aberrations come from the index varying with wavelength. These give a small set of analytic *aberration-coefficient* targets that do not require tracing many real rays and do not suffer ray failure, which makes them cheap and robust early in a design.

**The error/merit function.** The quality of a system is summarized as a weighted sum of squares of defects: f(x) = Σᵢ wᵢ (aᵢ(x) − ãᵢ)² (often normalized by Σwᵢ), where each operand aᵢ is a defect quantity — a transverse ray aberration of one traced ray, a wavefront aberration, a Seidel coefficient, or a lens parameter to be held in range — and ãᵢ is its target. Squares are used because positive and negative departures are equally harmful, and because the resulting function is smooth and, near a minimum, well approximated by a quadratic. There are typically many more operands than variables, because a faithful merit function traces many rays over field and wavelength: the operands form an overdetermined system that cannot be solved exactly, only in a least-squares sense.

**Nonlinear least squares and the Gauss–Newton idea.** A sum of squared residuals fᵢ(x) is minimized by the standard machinery of nonlinear least squares. Linearizing the residual vector about the current x, f(x+Δx) ≈ f(x) + J Δx with Jacobian Jᵢⱼ = ∂fᵢ/∂xⱼ, and minimizing ‖f + JΔx‖² gives the normal equations JᵀJ Δx = −Jᵀf, i.e. the Gauss–Newton step Δx = −(JᵀJ)⁻¹Jᵀf. Gauss–Newton drops the second-derivative term of the full Hessian, which is cheap and works well when residuals or curvature are small. The Jacobian here is obtained numerically: because a ray trace is a black box, ∂aᵢ/∂xⱼ is estimated by finite differences — perturb one variable, retrace, divide — costing one extra pass of the ray set per variable.

**Where the field stood and the pain point.** Steepest descent on the merit function was the first automatic method used. It is robust but converges very slowly when the equimagnitude contours of the merit function are elongated or irregular — which is exactly the optical case, because different parameters act on the aberrations in strongly correlated ways. The decisive empirical fact about optical systems, established by the structure of the aberration relations, is that the parameters' effects are highly correlated and very unequally scaled: two curvatures of a single thick element, or "bending" a lens, or a conic constant versus a fourth-order aspheric term, can produce *nearly the same* change in the aberrations (the Seidel formulas show the conic and the fourth-order deformation term enter identically, so one is redundant given the other). This near-redundancy makes the Jacobian columns nearly linearly dependent, so JᵀJ is ill-conditioned — nearly singular. A naive Gauss–Newton step then has a huge component along the near-null directions: the predicted Δx leaves the region where the linearization is valid, the true merit function rises instead of falling, and rays can miss surfaces or totally internally reflect so the trace fails outright (the merit function is undefined). The error landscape is also riddled with multiple local minima, so the result depends on the starting configuration and on the algorithm's own details.

## Baselines

**Steepest descent.** Move along −∇f, line-search to the minimum along that direction, repeat. Core idea: the gradient is the locally best descent direction; it is always perpendicular to the equimagnitude contours. Math: x ← x − α∇f(x) with α from a 1-D search. Gap: convergence is very slow whenever the contours are elongated (an anisotropic, ill-conditioned problem), which is the norm in lens design; it ignores the curvature of f and so zig-zags down long narrow valleys.

**Plain least squares / Gauss–Newton.** Treat the operands as locally linear, build J by finite differences, and solve the normal equations JᵀJ Δx = −Jᵀf for the full step in one shot. Core idea: if the operands were linear, the least-squares optimum is reached in a single linear solve; iterate to handle the nonlinearity. Math: Δx = −(JᵀJ)⁻¹Jᵀf. Gap: on the ill-conditioned optical Jacobian, JᵀJ is nearly singular, its inverse amplifies the near-null directions, and the resulting step is far too large — the linear approximation breaks, the merit function increases, and the iteration frequently diverges or hits ray failure. It also presupposes JᵀJ is invertible, which fails when columns are exactly redundant.

**Newton's method.** Use the full Hessian of f. Core idea: second-order model of f gives quadratic local convergence. Gap: the full Hessian needs second derivatives of every operand (expensive and noisy under finite differencing), it is not guaranteed positive-definite away from a minimum (so the step can point uphill), and it offers no remedy for the ill-conditioning that is the real problem here.

**Aberration-coefficient (analytic) correction.** Set up equations to null the primary (Seidel) aberration sums directly from first-order data. Core idea: a few analytic targets, no real-ray tracing, no ray failure, fast. Gap: primary coefficients alone do not capture the full real-ray performance of a fast, wide-field, polychromatic system; balancing higher-order and chromatic defects still requires minimizing many real-ray residuals.

## Evaluation settings

The natural yardsticks are standard imaging-system design tasks defined by a specification: an aperture (entrance pupil diameter or f-number), a set of field points (object/image angles or heights), and a set of wavelengths with one taken as primary. A starting configuration is given as a surface prescription (radii, thicknesses, glasses); a subset of parameters is freed as variables (commonly surface curvatures/radii, air spaces, sometimes glass characteristics within bounds and aspheric coefficients), while constraints such as a fixed effective focal length or non-negative edge/center thicknesses bound the search. The defect operands are transverse ray aberrations or wavefront/OPD errors of rays sampled over field and wavelength (e.g. by a quadrature distribution across the pupil), and/or Seidel coefficients, with per-operand and per-field/per-wavelength weights. Representative starting systems range from a single plano-convex or biconvex singlet to a three-element (triplet) objective. Quality is judged after the fact by ray fans, spot diagrams, OPD maps, and (near the diffraction limit) RMS wavefront error; first-order quantities (focal length, image location) are checked against the specification.

## Code framework

```python
import numpy as np
from scipy import optimize

# --- Geometric-optics ray trace through a sequence of surfaces (already exists) ---
class Surface:
    def __init__(self, radius, thickness, material):
        self.radius = radius          # R; curvature c = 1/R
        self.thickness = thickness    # separation to next surface
        self.material = material      # provides index n(wavelength)

class Optic:
    """An ordered stack of surfaces, an aperture, fields, wavelengths."""
    def trace(self, Hx, Hy, wavelength, pupil_samples):
        """Refract rays (Snell) surface by surface; return image-plane data."""
        ...  # forward model; differentiable-in-principle but treated as a black box

# --- Defect quantities ("operands"): each returns one residual value ---
class Operand:
    def __init__(self, kind, target, weight, input_data):
        self.kind, self.target, self.weight = kind, target, weight
        self.input_data = input_data
    def value(self):
        # transverse ray aberration / OPD / Seidel coeff / paraxial f, via trace
        raise NotImplementedError
    def delta(self):
        return self.value() - self.target
    def residual(self):
        return self.weight * self.delta()           # f_i = w_i (a_i - target_i)

# --- A freed system parameter ("variable") with its own normalization ---
class Variable:
    def __init__(self, optic, kind, surface_number, scale):
        self.optic, self.kind = optic, kind
        self.surface_number, self.scale = surface_number, scale
    def get(self): ...     # read parameter (in scaled units)
    def update(self, v): ...  # write parameter back into the optic

# --- The optimization problem: residual vector + merit = sum of squares ---
class OptimizationProblem:
    def __init__(self):
        self.operands = []   # the f_i
        self.variables = []  # the x_j
    def residual_vector(self):
        return np.array([op.residual() for op in self.operands])  # f(x)
    def sum_squared(self):
        r = self.residual_vector()
        return float(r @ r)                                       # merit f(x)

# --- The local optimizer: take the residual vector + variables and step x ---
class Optimizer:
    def __init__(self, problem):
        self.problem = problem
    def _residuals(self, x):
        for xi, var in zip(x, self.problem.variables):
            var.update(xi)
        self.problem.update_optics()
        return self.problem.residual_vector()
    def optimize(self, maxiter):
        # TODO: iteratively step the variables x to reduce sum_squared().
        pass
```
