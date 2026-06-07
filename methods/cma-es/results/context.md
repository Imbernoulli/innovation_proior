# Context: Learning the metric of a black-box landscape

## Research question

We are handed an objective function f: R^n → R and asked to minimize it with no access to gradients, no analytic form, nothing but the ability to hand in a point x and read back the number f(x). Every such query costs; the only honest performance measure is the number of function evaluations needed to reach a target value. The landscapes that matter in practice are non-linear, non-convex, and — the two properties that wreck naive search — *ill-conditioned* and *non-separable*.

A function is ill-conditioned when curvature varies wildly across directions: along one axis a small step changes f enormously, along another a large step barely moves it. It is non-separable when the variables are coupled, so that the favorable search directions are rotated off the coordinate axes and the problem cannot be split into n independent one-dimensional problems. On such landscapes the contour lines of f are long, thin, tilted ellipsoids.

The goal a solution must achieve: progress efficiently on these tilted, ill-scaled ellipsoids, and — because the right scaling and orientation are unknown a priori and change during the search — *learn* the appropriate local geometry from the evaluations themselves. Ideally the search behaves identically on a problem and on any rotated/rescaled version of it (affine invariance) and depends only on the ranking of f-values, never their magnitudes. That invariance is what lets empirical success on one problem generalize to a whole class.

## Background

**The evolution-strategy loop.** A stochastic search that carries a Gaussian sampling distribution. Each generation it samples λ offspring, evaluates them, keeps the best μ, and recombines them into the next center. In the notation of the field this is a (μ/μ,λ)-ES: μ parents recombined, λ offspring, comma meaning non-elitist truncation selection (parents are discarded each generation). Rechenberg (1973) and Schwefel (1981) built this machinery; the mutation is a normally distributed random vector added to the current center.

**Isotropic Gaussian mutation and its failure.** The simplest distribution is N(0, σ²I): one global step-size σ, surfaces of equal density are spheres. On a sphere f(x)=Σx_i² this is optimal. But take an ill-conditioned quadratic f(x)=Σ h_i x_i² with the h_i spanning several orders of magnitude. A single σ must be small enough not to overshoot the steep direction, and that same σ makes progress along the flat direction crawl — the convergence rate degrades with the condition number. And on a non-separable problem the productive direction is a rotated combination of coordinates; an isotropic (or even axis-parallel) distribution cannot point its samples preferentially along it. This is the documented failure mode that motivates everything: the sampling distribution's shape must match the landscape's, and a sphere matches almost nothing.

**The covariance–Hessian correspondence.** For a convex-quadratic f(x)=½xᵀHx with positive-definite Hessian H, sampling from N(m, C) with C = H⁻¹ is equivalent (up to transforming m) to optimizing the plain sphere with C = I: the inverse Hessian rescales the tilted ellipsoid into a circle. So the *optimal* covariance of the search distribution is the inverse Hessian, up to a scalar. Adapting C toward H⁻¹ is the continuous-optimization analogue of a quasi-Newton method that learns the metric — but using only ranked function values, no derivatives. The condition number cond(C) = λ_max/λ_min measures how far the ellipsoid is from a sphere.

**Choosing a covariance is choosing a coordinate transformation.** For any full-rank A, ½(Ax)ᵀ(Ax) = ½xᵀ(AᵀA)x, so picking a covariance is equivalent to picking an affine transformation of the search space. Hence a method that adapts an arbitrary normal distribution is implicitly learning a general linear problem encoding, and getting it right yields invariance to that linear transformation.

**Mutative strategy-parameter control and its shortcomings.** The pre-existing way to adapt the distribution (Schwefel 1981) was *mutative*: also mutate the strategy parameters (the step-sizes, the orientation), generate the offspring with the mutated parameters, and let selection on the object variables implicitly pick good strategy parameters. Several diagnosed problems with this scheme stand out as the pain points of the time. (a) The selection of a strategy-parameter setting is *indirect* — a setting is favored only because the individual it produced happened to be selected, so the signal driving the adaptation is heavily disturbed. (b) There is a conflict between the mutation strength needed to make a *significant selection difference* between competing settings and the mutation strength that yields an *optimal change rate*; the optimal-change-rate strength is usually the smaller, and the gap grows with dimension and with the number of strategy parameters being adapted. (c) The only way to tune the change rate down to a reliable level is to enlarge the parent number μ and use recombination, so a working mutative scheme needs a population that scales linearly with the number of strategy parameters — costly, and crippling for small populations. The recurring observation is also that mutative global-step-size control tends to adapt σ *too small*, because maximizing selection probability is not the same as maximizing progress.

**Estimating a covariance from a selected sample — the reference-mean trap.** A natural idea: after selecting the best μ points, estimate a covariance from them and sample from it next time. But *which mean* you subtract matters enormously. Estimating the variance of the selected points around *their own* mean (as the EMNA / cross-entropy estimation-of-distribution algorithms of Larrañaga et al. 2001, 2002 do) measures the spread *within* the selected cloud; its reference is the minimizer of that spread, so it systematically *shrinks* the variance — and on a slope it shrinks precisely along the gradient direction, geometrically fast, driving the search to premature convergence. Subtracting instead the *old* center (the true mean the offspring were sampled around) measures the selected *steps*; on a slope this *grows* the variance along the productive direction. Same data, opposite behavior, decided entirely by the reference mean.

## Baselines

**Isotropic-step-size ES (Rechenberg 1973).** Sample x_k = m + σ·N(0,I), select best, recombine; adapt the single σ (e.g. by the 1/5 success rule or mutatively). Optimal on the sphere, with a known convergence rate. Gap: one scalar cannot represent scale differences between directions, so it degrades on ill-conditioned f by a factor tied to the condition number, and it has no notion of orientation, so non-separable problems defeat it.

**Individual (axis-parallel) step-sizes.** Assign each coordinate its own variance: C diagonal, ellipsoids axis-parallel, n free strategy parameters. Handles axis-aligned scaling. Gap: it is tied to the coordinate system — rotate the problem and the favorable directions no longer align with the axes, and performance collapses. Not rotation-invariant.

**Mutative self-adaptation of the full normal distribution (Schwefel 1981).** Mutate all O(n²) strategy parameters and let selection sort them out, in principle reaching arbitrarily oriented ellipsoids. Gap: the three MSC shortcomings above — disturbed indirect selection, the change-rate/selection-difference conflict, and the population size that must scale with the number of strategy parameters. Slow and unreliable for the full covariance, especially with small populations.

**EMNA-global / continuous cross-entropy EDA (Larrañaga et al. 2001, 2002).** Each generation, fit a Gaussian to the μ best points using their own sample mean and covariance, then resample. Gap: by referencing the selected points' own mean it estimates within-cloud variance, which is biased downward and shrinks the variance in the gradient direction — highly prone to premature convergence, particularly with small μ.

## Evaluation settings

The natural yardstick is a suite of analytically known test functions in the continuous domain, exercising the properties the method targets. A separable, well-conditioned baseline: the sphere f_sphere(x)=Σx_i². Ill-conditioning at increasing severity: the ellipsoid f(x)=Σ (10^6)^{(i-1)/(n-1)} x_i² and the "cigar"/"tablet"/"discus" families that concentrate the condition number in one or a few directions. Non-separability is induced by composing any of these with a random rotation matrix R, optimizing f(Rx), so that a coordinate-wise method is forced off its axes. Multimodality and ruggedness: Rastrigin, Ackley, Rosenbrock (a curved, non-separable valley). The dimensionality n is swept (e.g. from a few up to hundreds), and runs are repeated over random seeds and random rotations. Performance is the number of function evaluations to reach a target f-value (or the best f after a fixed budget), reported as a function of n and of condition number. Because the method is meant to be rank-based, monotonic transformations of f-value are also a natural axis of the test.

## Code framework

The available primitives are a sampler of standard normal vectors, basic linear algebra (an eigendecomposition to factor a symmetric positive-definite matrix and to form its inverse square root), and the generic randomized-black-box loop: initialize distribution parameters, then repeatedly sample → evaluate → rank → update parameters until a termination test fires. What does not yet exist is the parameter-update rule for an adaptive multivariate Gaussian; that is the empty slot.

```python
import numpy as np


def f(x):
    # black-box objective: only the returned value is observable
    raise NotImplementedError


class SearchDistribution:
    """An adaptive sampling distribution over R^n.

    Holds whatever parameters describe where and how the search samples,
    and exposes ask() to produce candidates and tell() to update from
    their ranked evaluations. The update rule is what we must design.
    """

    def __init__(self, x0, n):
        self.n = n
        # TODO: the parameters that describe the sampling distribution
        #       (a center, an overall scale, a shape, and whatever
        #        accumulated state the update will need)
        pass

    def ask(self):
        # TODO: sample a population of candidate solutions from the
        #       current distribution
        pass

    def tell(self, candidates, fitnesses):
        # TODO: rank the candidates by fitness and update the
        #       distribution parameters so that good directions become
        #       more likely next time
        pass

    def stop(self):
        # TODO: termination tests
        return False


def optimize(f, x0, n):
    dist = SearchDistribution(x0, n)
    while not dist.stop():
        X = dist.ask()
        fits = [f(x) for x in X]
        dist.tell(X, fits)
    return dist
```
