# Automatic lens design by Damped Least Squares (Levenberg–Marquardt)

## Problem

Given a starting optical system (an ordered stack of refracting surfaces, each with a curvature, an axial thickness, and a glass), automatically adjust a chosen subset of parameters so the image defects fall within tolerance. Evaluating a fixed system is a forward ray trace; the design task is the inverse problem, and it is hard because (a) the map from parameters to defects is strongly nonlinear, (b) there are far more quality requirements (rays over field and wavelength, plus first-order constraints) than parameters, so no exact solution exists, and (c) the parameters act on the defects in strongly correlated, very unequally-scaled ways.

## Key idea

Express image quality as a vector of residuals fᵢ(x) = wᵢ(aᵢ(x) − ãᵢ) — weighted deviations of each defect operand (transverse ray aberration, OPD/wavefront error, Seidel coefficient, paraxial quantity like focal length) from its target — and minimize the merit function f(x) = Σᵢ fᵢ(x)² by nonlinear least squares.

Linearizing the residuals about the current x with Jacobian Jᵢⱼ = ∂fᵢ/∂xⱼ gives the Gauss–Newton step from the normal equations

  JᵀJ Δx = −Jᵀf  ⟹  Δx = −(JᵀJ)⁻¹Jᵀf.

The optical Jacobian is ill-conditioned: correlated variables (two curvatures of a thick element, lens bending, conic vs. aspheric coefficient) make columns of J nearly parallel, so JᵀJ is near-singular and (JᵀJ)⁻¹ explodes along the soft directions. The Gauss–Newton step then overshoots out of the region where the linear model holds — the merit rises, or rays fail to trace — and the iteration diverges.

The fix is **damping**: leash the step to a trust region ‖Δx‖² ≤ h², equivalently add a penalty λ‖Δx‖² to the linearized objective. The damped normal equations are

  (JᵀJ + λI) Δx = −Jᵀf  ⟹  Δx = −(JᵀJ + λI)⁻¹Jᵀf.

In the SVD J = UΣVᵀ, the per-direction coefficient changes from 1/σₖ (which blows up as σₖ→0) to σₖ/(σₖ²+λ): well-determined directions (σₖ²≫λ) keep the full Gauss–Newton step ≈1/σₖ, while soft directions (σₖ²≪λ) are suppressed toward 0, so the singular directions can no longer run away. The limits recover the two classical methods: λ→0 gives Gauss–Newton (fast); λ→∞ gives Δx → −(1/λ)Jᵀf, a small steepest-descent step (safe, since Jᵀf is the merit gradient). Because JᵀJ + λI is positive-definite for any λ>0, the solve never fails even when J is rank-deficient.

λ is adapted by trust: solve, step, retrace, compare merit — if it decreased, accept and shrink λ (lean Gauss–Newton); if it increased or the trace failed, reject and grow λ (lean steepest descent), re-solving the same iteration. Damping is applied in normalized variable units (per-variable scaling, or equivalently λ·diag(JᵀJ)) so each parameter is damped in proportion to its own sensitivity, since curvatures, thicknesses, and indices live on very different scales.

This is the Levenberg–Marquardt algorithm. The method is local: it descends to the minimum in the starting design's basin, and the merit landscape has many local minima, so the starting configuration and the staging of which variables are freed still determine the final lens form.

## Algorithm

1. Choose variables x (curvatures/radii, air spaces, possibly aspheric coefficients, glasses within bounds), in normalized units.
2. Build operands: trace rays over field and wavelength for ray/OPD aberrations, plus Seidel and paraxial (e.g. focal-length) operands; form the residual vector f.
3. Estimate J by finite differences (one extra trace pass per variable).
4. Solve (JᵀJ + λI)Δx = −Jᵀf.
5. Apply Δx, retrace, evaluate the merit (ray failure → large finite penalty).
6. If merit decreased: accept, decrease λ. Else: reject, increase λ, go to 4.
7. Iterate until the merit change / step is below tolerance.

## Code

Grounded in the optiland optimization framework (`OptimizationProblem` with operands + variables; the residual vector feeding `scipy.optimize.least_squares(method="lm")`, which is the damped-least-squares / Levenberg–Marquardt solver).

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

Example setup (a triplet objective): operands are a focal-length target plus the five Seidel sums driven to zero; variables are the surface radii; symmetry is imposed with pickups, then the damped-least-squares loop runs.

```python
problem = OptimizationProblem()
problem.add_operand("f2", target=50, weight=1, input_data={"optic": lens})
for i in range(1, 6):                                    # Seidel S1..S5 -> 0
    problem.add_operand("seidel", target=0, weight=10,
                        input_data={"optic": lens, "seidel_number": i})
for s in (1, 2, 3):
    problem.add_variable(lens, "radius", surface_number=s)

result = DampedLeastSquares(problem).optimize()
```
