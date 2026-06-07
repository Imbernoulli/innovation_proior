OK, let me think this through from scratch. I have a lens — an ordered stack of refracting surfaces, each one a curvature, a glass after it, and an axial distance to the next — and I have a starting design that more or less works but images badly. I want a procedure that, on its own, nudges the curvatures and spacings (and maybe the glasses) until the image is as sharp as the job needs. The forward direction is the easy half: pick a parameter set, trace rays through it by geometric optics, look at how badly they miss. The hard half is the inverse: given "I want the image defects below tolerance," what parameters get me there? I can't invert the ray trace analytically — it's a deeply nonlinear composition. So this has to be iterative: measure how bad I am now, take a step in parameter space that should make me less bad, repeat.

Let me first pin down what "bad" even is, quantitatively, because the whole method is going to hang on the shape of that function. Trace a ray from a field point through the system; at each surface it bends by Snell's law, n₁ sinθ₁ = n₂ sinθ₂, so its outgoing direction is some smooth function of the incident direction, the local surface normal (which depends on the curvature), and the index ratio; between surfaces it just runs straight by the thickness. Compose all of that and the ray lands somewhere on the image plane. For a perfect system every ray from one object point would pile up at one image point. It doesn't. The miss distance — where this ray actually lands minus where it ideally should — is a transverse ray aberration. That's one number. And I have lots of rays: a spread across the pupil, at several field angles, at several wavelengths, because color matters and the index is wavelength-dependent. Plus I have first-order requirements I refuse to give up, like "the focal length must be 50 mm," which I get from the paraxial trace, n₂u′ = n₁u − y(n₂−n₁)/R marched surface to surface.

So I don't have *one* defect, I have a whole vector of them. Call them a₁(x), a₂(x), …, where x is the vector of freed parameters (curvatures, thicknesses, …). Each one has a target — the aberrations want to be zero, the focal length wants to be 50. I want all of them at target at once. Count them: rays times fields times wavelengths is easily dozens or hundreds of operands, and I might only be freeing a handful of curvatures. That's the key structural fact — far more conditions than knobs. The system aᵢ(x) = ãᵢ is wildly overdetermined. There is no x that satisfies all of it exactly; a point image at every field and color through a real glass stack simply doesn't exist. So "solve" is the wrong verb. I want the *best collective* near-miss.

The natural way to say "best collective near-miss" for an overdetermined system is least squares: minimize the sum of the squared misses. Define residuals fᵢ(x) = wᵢ(aᵢ(x) − ãᵢ) with weights wᵢ to say which defects I care about more, and minimize f(x) = Σᵢ fᵢ(x)². Why squares, specifically? Because a ray landing 10 µm high and one landing 10 µm low are equally bad — sign shouldn't matter — and squaring both penalizes the magnitude symmetrically and smoothly. And because, near a minimum, a sum of squares looks like a paraboloid, which is the friendliest possible thing to optimize: it has a well-defined curvature I can exploit. So my objective is a smooth scalar Σfᵢ(x)², built from a vector of residuals, with way more residuals than variables. Good. That's a nonlinear least-squares problem, and I should reach for nonlinear-least-squares machinery rather than treating it as a generic black-box scalar to minimize.

The crudest thing I could do is steepest descent: compute the gradient ∇f, step along −∇f, line-search, repeat. It's safe — downhill is downhill — and it always moves perpendicular to the contours of f. But I can already smell the trouble. Steepest descent crawls when the contours of f are long thin valleys, because it keeps zig-zagging across the valley instead of running down it. And are the contours long and thin here? Almost certainly, because the parameters of a lens don't act independently on the aberrations — bending a lens, or splitting a power between two nearby curvatures, moves several aberrations together in nearly the same way. Correlated variables make an elongated, anisotropic error landscape. So steepest descent will technically converge but take forever. I want to use the *curvature* of f, not just its slope.

Here's where the sum-of-squares structure pays off, because I don't have to estimate the full Hessian by brute force — the residual vector hands me most of it. Linearize the residuals about my current x: f(x+Δx) ≈ f(x) + J Δx, where J is the Jacobian, Jᵢⱼ = ∂fᵢ/∂xⱼ. Now the objective, in this local linear model, is ‖f + JΔx‖². That's just a *linear* least-squares problem in the step Δx — and a linear least-squares problem I can solve in one shot. Expand: ‖f + JΔx‖² = fᵀf + 2(JΔx)ᵀf + ΔxᵀJᵀJΔx. Set the gradient with respect to Δx to zero: 2Jᵀf + 2JᵀJΔx = 0, so

  JᵀJ Δx = −Jᵀf,  Δx = −(JᵀJ)⁻¹Jᵀf.

These are the normal equations, and that's the Gauss–Newton step. Notice what just happened to the Hessian: the true Hessian of Σfᵢ² is 2JᵀJ + 2Σfᵢ∇²fᵢ, and I've quietly dropped the second term. That's fine — it carries the second derivatives of every operand, which would be expensive and noisy to get, and it's small whenever the residuals fᵢ are small (near a good solution) or the operands are mildly curved. So JᵀJ, which costs me nothing beyond J itself, stands in for the curvature. This is exactly why the least-squares framing is the right one: the structure gives me a free, positive-semidefinite curvature model.

I still need J. The ray trace is a black box — Snell's law composed many times — and I'm not about to differentiate it by hand through every surface. Finite differences: perturb one variable by a small δ, retrace the whole ray set, take (f(x+δeⱼ) − f(x))/δ for column j. One extra trace pass per variable. Since I have only a handful of variables and the expensive part is tracing many rays, that's an affordable price for the Jacobian.

So: build J by finite differences, solve JᵀJ Δx = −Jᵀf, step, repeat. Let me just run this in my head on a real design and watch it. … And it falls over. The step Δx comes out enormous — curvatures swinging wildly — and after I apply it the merit function is *worse* than before, not better. Sometimes a ray now misses a surface entirely or totally-internally-reflects, the trace fails, and the merit function isn't even defined. The iteration diverges.

Why? Stare at (JᵀJ)⁻¹. The catastrophe is in that inverse, and the cause is the very correlation I noticed makes the valley thin. The columns of J are the sensitivity vectors ∂f/∂xⱼ — how the whole defect vector responds to wiggling variable j. If two variables push the aberrations in nearly the same direction — and they do; two curvatures of one thick element, or a conic constant versus a fourth-order aspheric coefficient, change the aberrations almost identically — then two columns of J are nearly parallel. Nearly parallel columns mean JᵀJ has an eigenvalue near zero: it's ill-conditioned, nearly singular, sometimes exactly singular if two variables are perfectly redundant. And (JᵀJ)⁻¹ does the worst possible thing with a near-zero eigenvalue: it divides by it. Look through the singular value decomposition to see it cleanly. Write J = UΣVᵀ with singular values σ₁ ≥ … ≥ σₙ. Then JᵀJ = VΣ²Vᵀ and the Gauss–Newton step is

  Δx = −V Σ⁻¹ Uᵀ f  =  −Σₖ (uₖᵀf / σₖ) vₖ.

Each direction vₖ in parameter space gets a coefficient with σₖ in the *denominator*. For the well-determined directions σₖ is healthy and 1/σₖ is fine. But for a near-redundant combination of variables σₖ is tiny, so 1/σₖ is gigantic, and the step takes a colossal excursion along vₖ — a direction the aberrations barely respond to. That's exactly the pathology: the optimizer says "the aberrations are almost indifferent to this combination of curvatures, so I'll move it a mile to squeeze out the last drop of linear improvement." But "a mile" is far outside the little neighborhood where my linear model f + JΔx was any good. The real, nonlinear merit function does something completely different out there — it goes up, or the rays stop tracing. The undamped least-squares step trusts the linearization globally when it's only valid locally, and the ill-conditioning is what lets it run away.

So I have two failure modes pulling against each other. Steepest descent is safe but pathologically slow on the thin valley. Gauss–Newton is fast but unsafe — it sprints along the near-null directions and leaves the region where its own model is true. I want the long step where the model is trustworthy and a short, safe step where it isn't. The problem is purely that I let |Δx| become huge along the soft directions.

So put a leash on |Δx| directly. I don't want the step that minimizes ‖f + JΔx‖² over all of parameter space; I want the step that minimizes it *subject to the step being small enough that the linearization still holds*. Constrain ‖Δx‖² ≤ h² for some trust radius h, and minimize the linear model inside that ball. The way to handle a quadratic objective under a quadratic constraint is a Lagrange multiplier: form

  ‖f + JΔx‖² + λ‖Δx‖²

and minimize unconstrained over Δx, where λ ≥ 0 is the multiplier for the active constraint ‖Δx‖² = h². Equivalently — and this is the same thing said in the optimizer's own language — replace the merit function f by a damped version that adds a penalty growing with the square of the step length: f_D = f + λ|Δx|². The minima of the true f and of f_D coincide (at a true minimum Δx vanishes and the penalty is zero), but away from a minimum f_D refuses to take big steps because the penalty punishes them. λ is a *damping factor*: crank it up and large Δx becomes expensive, so the step shrinks.

Now redo the linear solve with the penalty. Minimize fᵀf + 2(JΔx)ᵀf + ΔxᵀJᵀJΔx + λΔxᵀΔx. Gradient in Δx to zero: 2Jᵀf + 2JᵀJΔx + 2λΔx = 0, so

  (JᵀJ + λI) Δx = −Jᵀf,  Δx = −(JᵀJ + λI)⁻¹Jᵀf.

That single +λI is the whole fix, and it's beautiful to check against the SVD. Now JᵀJ + λI = V(Σ² + λI)Vᵀ, and the step becomes

  Δx = −Σₖ (σₖ uₖᵀf / (σₖ² + λ)) vₖ.

Compare the coefficient on direction vₖ: it went from 1/σₖ (which exploded as σₖ→0) to σₖ/(σₖ²+λ). Look at the two regimes. Where the direction is well-determined, σₖ² ≫ λ, this is ≈ σₖ/σₖ² = 1/σₖ — the full Gauss–Newton step, untouched, so I keep the fast convergence in the directions I can trust. Where the direction is soft, σₖ² ≪ λ, this is ≈ σₖ/λ → 0 as σₖ→0 — the step in that direction is *suppressed* instead of amplified. The denominator can no longer vanish: even an exactly singular direction (σₖ = 0) now gets coefficient 0, not infinity. The damping tames precisely the singular directions that were blowing the undamped step up, and leaves the healthy ones alone. The leash only tightens where the model can't be trusted.

And the two extremes of λ are exactly the two methods I was torn between. As λ → 0, (JᵀJ + λI)⁻¹ → (JᵀJ)⁻¹ and I recover Gauss–Newton — fast, when I can afford it. As λ → ∞, JᵀJ becomes negligible next to λI, so Δx → −(1/λ)Jᵀf. But Jᵀf is the gradient of the merit function (∇(½‖f‖²) = Jᵀf), so a huge λ gives me a tiny step straight down the gradient — steepest descent, with step size 1/λ. So the damping factor is a dial that slides continuously between the safe-but-slow gradient method and the fast-but-fragile Gauss–Newton method. Small λ = trust the curvature model, take the big informed step; large λ = distrust it, take a small safe downhill step. That's the resolution of the tension: I no longer have to choose between the two methods globally, I interpolate between them, per iteration, by one number.

Which leaves: how do I pick λ? It's a trust statement, so let the trust be measured by results. Pick a λ, solve for Δx, apply it, retrace, and compare the new merit to the old. If the merit went *down*, the step was trustworthy — accept it, and loosen the leash for next time (shrink λ, lean toward Gauss–Newton, go faster). If the merit went *up*, or the trace failed, the step over-reached — reject it, tighten the leash (grow λ, e.g. ×10), and re-solve the *same* iteration with the bigger damping, which both shortens the step and rotates it toward the safe gradient direction. Because (JᵀJ + λI) is positive definite for any λ > 0 — I've added a positive multiple of the identity to a positive-semidefinite matrix — the solve never fails for being singular, no matter how degenerate J is. So increasing λ is guaranteed to eventually produce a short enough, gradient-aligned enough step that the merit decreases (in the limit it's an infinitesimal step straight downhill, which must decrease f unless I'm already at a minimum). The adaptation always finds a productive step or certifies I'm done.

I should double back on one thing: is a single scalar λ on I really the right leash? I added λI — the same damping to every variable. But my variables are wildly different beasts. A curvature is in units of inverse length and a tiny change moves the aberrations a lot; an air space is in length and a millimeter barely registers; a glass index change is yet another scale. Penalizing all of them with the *same* λ‖Δx‖² is using a round ball as the trust region when the geometry is a long thin ellipsoid — I over-damp the insensitive variables and under-damp the sensitive ones, just transplanting the conditioning problem into the damping. The cleaner statement is that I want the *length* I'm penalizing to be measured in a metric that respects each variable's sensitivity. Two ways to get there, and they're the same idea. One: nondimensionalize the variables up front — rescale each xⱼ so a unit change has a comparable effect on the residuals (a linear scaling of each variable, e.g. curvatures in units of ~1/100 mm⁻¹), so that after scaling λI is roughly isotropic in the right metric. Two: instead of λI, damp with λ·diag(JᵀJ) — give each variable a damping proportional to its own sensitivity σ-scale, so the solve becomes (JᵀJ + λ·diag(JᵀJ))Δx = −Jᵀf. Either way each parameter gets a damping that compensates for its relative sensitivity, the trust region becomes an ellipsoid aligned with the problem's natural scales, and the convergence speeds up markedly compared to a single blunt λ on raw variables. I'll keep both levers: normalize the variables, and let the damping be applied in those normalized units.

One more guard from the optical reality: rays fail. Push a curvature too far and a ray totally-internally-reflects or misses a surface, and an operand comes back undefined — a NaN. I must not let that poison the merit comparison. So whenever the residual evaluation hits a non-finite value, I treat that whole candidate x as a large, finite penalty — a wall the optimizer feels and backs away from — which in the λ loop simply reads as "merit went up, reject, increase damping." That keeps the ray-failure regions out of bounds without special-casing them.

Let me also be honest about the global picture, because it sets expectations rather than changing the step. This whole construction is *local*: it finds the minimum in whose basin the starting design sits. The optical merit landscape has many minima — different lens forms — and damped least squares walks downhill to the nearest decent one. The starting configuration, the order in which I free variables, even the implementation details of the damping, decide which basin I land in. That's not a flaw in the step rule; it's a property of the problem, and it's why the starting point and the designer's staging still matter. The step rule's job is just to get reliably to the bottom of whatever basin I'm in, fast, without diverging — and that's what the damping buys me.

Now let me put it down as code, grounded in how a real lens-optimization engine is wired. The pieces map one-to-one to the reasoning. A lens is an `Optic`: an ordered stack of surfaces, each with a radius (curvature 1/R), a thickness, and a glass whose index depends on wavelength; plus an aperture, field points, and wavelengths. The defects are *operands*: each one traces what it needs and returns a value, its delta from target, and the weighted residual wᵢ(aᵢ − ãᵢ). The freed parameters are *variables* with their own linear scaling (the per-variable normalization above). An `OptimizationProblem` collects operands and variables and can hand back either the residual vector f (the unsquared weighted deltas — what least squares needs) or the scalar merit Σfᵢ². And the optimizer takes the residual vector, builds the damped step, and iterates. The damped-least-squares solver is exactly Levenberg–Marquardt — the standard library's `least_squares(method="lm")` implements the (JᵀJ + λI)Δx = −Jᵀf solve, the finite-difference Jacobian, and the λ-adaptation loop — so I feed it the residual function and let it run.

```python
import numpy as np
from scipy import optimize


class Operand:
    """One image-defect condition: a_i with a target. residual = w_i (a_i - target)."""
    def __init__(self, kind, target, weight, input_data):
        self.kind, self.target, self.weight = kind, target, weight
        self.input_data = input_data

    def value(self):
        # trace what this operand needs (a real-ray transverse aberration / OPD,
        # a Seidel coefficient, a paraxial quantity like focal length) and return it
        return evaluate_operand(self.kind, self.input_data)

    def delta(self):
        return self.value() - self.target

    def residual(self):
        return self.weight * self.delta()           # f_i = w_i (a_i - target_i)


class Variable:
    """A freed system parameter, stored in normalized units so damping is isotropic."""
    def __init__(self, optic, kind, surface_number, scale=1 / 100.0, offset=-1.0):
        self.optic, self.kind = optic, kind
        self.surface_number = surface_number
        self.scale, self.offset = scale, offset      # per-variable normalization

    def get(self):                                   # raw -> scaled
        raw = read_parameter(self.optic, self.kind, self.surface_number)
        return raw * self.scale + self.offset

    def update(self, scaled_value):                  # scaled -> raw, write back
        raw = (scaled_value - self.offset) / self.scale
        write_parameter(self.optic, self.kind, self.surface_number, raw)


class OptimizationProblem:
    def __init__(self):
        self.operands = []          # the residuals f_i (many)
        self.variables = []         # the parameters x_j (few): m >> n

    def residual_vector(self):
        # the UNsquared weighted deltas: this is f(x), what least squares consumes
        return np.array([op.residual() for op in self.operands])

    def sum_squared(self):
        r = self.residual_vector()
        return float(r @ r)         # the scalar merit f(x) = sum f_i^2

    def update_optics(self):
        for optic in {v.optic for v in self.variables}:
            optic.update()          # propagate solves/pickups after a variable change


class DampedLeastSquares:
    """Levenberg-Marquardt: Delta_x = -(J^T J + lambda I)^{-1} J^T f, lambda adapted by trust."""
    def __init__(self, problem):
        self.problem = problem

    def _residuals(self, x):
        # write the (scaled) variables, retrace, return the residual vector f(x)
        for xi, var in zip(x, self.problem.variables):
            var.update(xi)
        self.problem.update_optics()
        f = self.problem.residual_vector()
        if not np.all(np.isfinite(f)):
            # ray failure (TIR / missed surface): a finite wall the optimizer backs off
            big = np.sqrt(1e10 / max(len(f), 1))
            return np.full(len(f), big)
        return f

    def optimize(self, maxiter=None, tol=1e-3):
        x0 = np.array([v.get() for v in self.problem.variables])
        # 'lm' = damped least squares: finite-difference Jacobian J, normal equations
        # (J^T J + lambda I) Delta_x = -J^T f, and the lambda up/down trust loop.
        # Requires m >= n (we have many operands, few variables), no hard bounds.
        result = optimize.least_squares(
            self._residuals, x0, method="lm", max_nfev=maxiter, ftol=tol,
        )
        for xi, var in zip(result.x, self.problem.variables):
            var.update(xi)
        self.problem.update_optics()
        return result
```

The causal chain, start to finish: a lens is surfaces with curvatures, thicknesses, and glasses, and image quality is a long vector of ray-error and aberration defects, far more of them than I have parameters — so the honest goal is to minimize the sum of their squares, a nonlinear least-squares problem. Linearizing the residuals gives the Gauss–Newton step Δx = −(JᵀJ)⁻¹Jᵀf, which is fast because the sum-of-squares structure hands me JᵀJ as a free curvature model. But the optical Jacobian is ill-conditioned — variables act on the aberrations in nearly redundant ways — so (JᵀJ)⁻¹ amplifies the soft, near-singular directions and the step sprints out of the region where the linear model holds, sending the merit up or breaking the ray trace. Leashing the step length to a trust region adds a damping term, turning the solve into (JᵀJ + λI)Δx = −Jᵀf, where +λI replaces each soft direction's exploding 1/σ by the bounded σ/(σ²+λ): small λ recovers Gauss–Newton, large λ becomes a safe gradient step, and adapting λ by whether the merit actually fell slides between them per iteration — with per-variable scaling so the leash respects each parameter's sensitivity. That damped step, iterated, is the automatic lens-design loop.
