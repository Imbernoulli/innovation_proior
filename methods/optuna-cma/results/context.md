# Context: rank-based Gaussian search for continuous black-box optimization

## Research question

We are given a black-box objective `f: R^n -> R` to minimize. The only thing we can do is
choose a point `x` and read back `f(x)`; there is no gradient, no Hessian, and no algebraic
form to exploit. Every evaluation is expensive (in the applications that motivate this work a
single `f`-call is a simulation or a physical experiment), so the only performance measure
that matters is the *number of function evaluations* needed to reach a target value. The
landscapes are the hard ones: non-linear, non-convex, ill-conditioned (the favorable
directions can be orders of magnitude narrower than the unfavorable ones), strongly
non-separable (the variables interact, so the problem cannot be decomposed into `n`
one-dimensional searches), and often rugged or noisy — exactly the regime where classical
local methods such as BFGS or conjugate gradients stall or fail outright.

The precise goal is a randomized search procedure that (1) needs only the *ranking* of
sampled points, not their absolute `f`-values, so that it is unchanged by any strictly
monotone rescaling of the objective; (2) is invariant to rotation of the search space, so
that its performance on a separable problem transfers verbatim to an arbitrarily rotated
(non-separable) version of the same problem; (3) automatically discovers and exploits the
*scale and orientation* of the favorable directions — i.e. learns the conditioning of the
problem rather than requiring the user to pre-scale the variables; (4) controls the overall
length of its steps so that it neither stalls (steps too small) nor random-walks (steps too
big); and (5) does all of this *reliably with a small number of samples per iteration*, so
that the function-evaluation budget is spent on progress, not on estimating internal
parameters. Each existing approach below achieves a subset; none achieves all five at once.

## Background

The search distribution. The natural object for randomized continuous search is a
multivariate normal distribution `N(m, C)`: given all variances and covariances it is the
maximum-entropy distribution on `R^n`, and it singles out no coordinate direction, which is
what we want from a *general-purpose* sampler. A normal distribution factorizes through its
covariance. If `C = B D^2 B^T` is the eigendecomposition (`B` orthonormal columns = principal
axes, `D` diagonal of axis lengths), then

```
N(m, C)  ~  m + B D N(0, I),
```

so sampling is: draw `z ~ N(0, I)`, scale the coordinates by `D`, rotate by `B`, shift by
`m`. The one-`sigma` contour of `N(m, sigma^2 C)` is the ellipsoid
`{x : (x-m)^T C^{-1} (x-m) = sigma^2}`; its principal axes are the eigenvectors of `C` and
its axis lengths are `sigma` times the square roots of the eigenvalues.

Why the covariance is the prize. Consider a convex-quadratic model
`f(x) = 1/2 (x-x*)^T H (x-x*)` with positive-definite Hessian `H`. Sampling with `C = H^{-1}`
turns the ellipsoidal level sets of `f` into spheres: in the transformed coordinates every
direction is equally good, and the search becomes the trivial isotropic one. So on the
quadratic model the *ideal* covariance of the search distribution is the inverse Hessian (up
to a scalar), and learning a good `C` is the black-box analogue of building the inverse-Hessian
preconditioner that a quasi-Newton method maintains from gradients. The condition number of
`C` (largest over smallest eigenvalue) measures how anisotropic the distribution is; matching
it to the conditioning of `f` is what lets a search make progress on ill-conditioned problems
in a number of steps that does not blow up with the condition number.

The three levels of mutation distribution (Rechenberg 1973; Schwefel 1981). The evolution-strategy
tradition adapts the *shape* of the sampling distribution by degrees. Level one: an isotropic
Gaussian, one free strategy parameter — the global step size `sigma` (contours are spheres).
Level two: a diagonal covariance, `n` individual per-coordinate step sizes (axis-parallel
ellipsoids); this already breaks rotation invariance, because it privileges the coordinate
axes. Level three: an arbitrary normal distribution with zero mean, `(n^2+n)/2` free
parameters (arbitrarily oriented ellipsoids); any zero-mean normal can be produced, and if
the adaptation mechanism is formulated independently of the coordinate system, rotation
invariance is restored. Moving up the levels buys the ability to fit ill-conditioned,
non-separable problems — at the cost of having many more strategy parameters to set, and no
gradient with which to set them.

Mutative self-adaptation and its diagnostic failure modes.
The classical way to set strategy parameters without gradients is *mutative strategy parameter
control* (MSC): attach the strategy parameter to each individual, mutate it, generate the
object point with the mutated value, and let selection on the object point decide. For a
single global step size this reads

```
sigma_k^{(g+1)} = sigma^{(g)} * exp(xi_k),         xi_k a small zero-mean random number,
x_k^{(g+1)}     = x^{(g)} + sigma_k^{(g+1)} * z_k,  z_k ~ N(0, I),
```

select the best offspring, repeat. The implicit hope is that strategy-parameter settings that
*produced* selected steps are good settings for the near future. This works, but several
limitations are well documented in the evolution-strategy literature and are observable in
simulation:

- The selection of a strategy-parameter value is *indirect*: a value is judged only through
  the one object point it happened to generate, so two values differ in selection probability
  by a small, noisy margin. The selection signal on the strategy-parameter level is heavily
  disturbed.
- Maximizing selection probability is not the same as maximizing progress rate; the two come
  apart, and a much-observed consequence is that MSC tends to drive the global step size
  *too small*.
- The realizable change rate of a strategy parameter per generation is bounded by the finite
  amount of selection information. When `n` or `(n^2+n)/2` strategy parameters must be set,
  the change rate per parameter must drop, so adaptation slows; and the only MSC knob that
  decouples change rate from mutation strength is the parent number `mu`, which forces the
  population size to scale roughly linearly with the number of strategy parameters. Fitting a
  full covariance with MSC therefore demands large populations and many evaluations.

The setup is now clear: a Gaussian sampler whose ideal covariance is the inverse Hessian, a
ladder of distribution shapes that buys conditioning at the price of more parameters, and a
self-adaptation mechanism whose selection signal is too noisy and too slow to set those
parameters from few samples.

Two estimation viewpoints that already exist. There is a separate line that fits the Gaussian
directly to the good points rather than mutating strategy parameters: estimation-of-distribution
algorithms (EDAs) and the cross-entropy method. Given the `mu` best of `lambda` sampled points,
they re-estimate the covariance as the empirical covariance *of those selected points*,

```
C_EMNA = (1/mu) sum_{i=1}^{mu} (x_{i:lam} - m_new)(x_{i:lam} - m_new)^T,
         m_new = (1/mu) sum_{i=1}^{mu} x_{i:lam},
```

and resample from `N(m_new, C_EMNA)`. This is clean and gradient-free. Its limitation is a
diagnostic phenomenon worth stating precisely, because any covariance-fitting scheme has to
contend with it: since the reference point is the mean *of the selected points themselves*,
`C_EMNA` measures the spread *within* the surviving cluster, which on a slope is narrower than
the spread of the steps that got there. On a linear (or locally linear) function this estimator
shrinks the variance in the very direction the search should move, geometrically fast, so the
distribution collapses and the search converges prematurely — most severely with small
populations, which cannot bracket the optimum.

## Baselines

These are the prior methods a new continuous black-box optimizer would be measured against and
would react to.

Pure random search (Brooks 1958; Rastrigin 1963). Sample points i.i.d. (uniformly in a box, or
from a fixed Gaussian) and keep the best. Trivially rotation- and rank-invariant, embarrassingly
parallel, immune to ruggedness. Gap: no learning whatsoever — it ignores the geometry of `f`
entirely, so on an `n`-dimensional ill-conditioned problem the fraction of samples that improve
shrinks like the volume ratio of the level sets, and the evaluation count to reach a target grows
without any benefit from past evaluations.

The (1+1) and (mu, lambda) evolution strategies with the 1/5 rule (Rechenberg 1973; Schwefel 1981).
Sample around the current point with an isotropic Gaussian of step size `sigma`; Rechenberg's 1/5
success rule increases `sigma` if more than one-fifth of mutations improve and decreases it
otherwise, targeting the empirically optimal success rate on the sphere. This is genuine,
cheap step-size control and is rotation-invariant. Gap: it controls only one scalar — the overall
scale — and assumes the distribution stays isotropic; on an ill-conditioned or non-separable
problem an isotropic Gaussian must shrink to the *narrowest* favorable direction, so progress
along the broad directions crawls. The success rate also depends on the landscape, so the rule
mis-tunes off the sphere, and on noisy functions the success-counting is unreliable.

Mutative self-adaptation of individual step sizes (Schwefel 1981). Level-two MSC: attach a
per-coordinate step-size vector to each individual, mutate it log-normally, generate offspring,
select. This can fit axis-aligned anisotropy. Gap: it is axis-parallel, hence not
rotation-invariant — it helps only when the favorable directions happen to align with the
coordinate axes, and a rotation of the problem destroys the benefit. And it inherits MSC's
disturbed, slow strategy-parameter selection, so even the `n` diagonal entries need a population
that grows with `n`.

Correlated mutations / arbitrary-normal MSC (Schwefel 1981; the rotation-angle parameterization).
Level-three MSC: parameterize a full rotation by `n(n-1)/2` angles plus `n` axis scalings and
mutate all of them log-normally / additively. In principle this fits an arbitrary oriented
ellipsoid and is the right *expressiveness*. Gap: the parameterization is coordinate-dependent
and clumsy (the angles do not commute, the mutation of an angle does not correspond to a clean
geometric move), the invariance demand is not met, and adapting `(n^2+n)/2` parameters through the
indirect, noisy MSC selection signal is so slow that it needs impractically large populations; it
does not scale.

Direct covariance estimation: EDA / EMNA / cross-entropy (Larrañaga & Lozano 2001; Rubinstein 1999).
Re-estimate the Gaussian from the selected points each generation (the `C_EMNA` update above) and
resample. Gradient-free, simple, and at large populations it fits the right shape. Gap: as the
diagnostic above shows, estimating around the *selected-points' own mean* measures within-cluster
spread, which on a slope is smaller than the spread of the productive steps, so the variance in the
direction of improvement contracts and the search converges prematurely — and small populations,
which are exactly what an expensive `f` forces, make this worst.

Nelder–Mead downhill simplex (Nelder & Mead 1965). Maintain a simplex of `n+1` points and
reflect/expand/contract it downhill. Derivative-free and popular. Gap: not rotation-invariant in
its standard form, no probabilistic model of the landscape, and the simplex is known to collapse
into a lower-dimensional subspace in dimension beyond roughly ten, after which it can no longer
make progress in the lost directions.

## Evaluation settings

The natural yardsticks already in use for derivative-free continuous optimizers:

- The sphere `f_sphere(x) = sum_i x_i^2` — the trivial isotropic baseline; used to calibrate
  step-size control and to read off the optimal step length and the optimal change rate of
  `sigma` per `n` evaluations.
- Ill-conditioned convex-quadratics — the ellipsoid `sum_i (10^6)^{(i-1)/(n-1)} x_i^2` and the
  cigar / tablet / discus families, with condition numbers up to `10^6`, in both axis-parallel
  and *randomly rotated* coordinate systems. The rotated versions are the test of rotation
  invariance and of whether a method can learn off-axis anisotropy.
- Non-separable and non-convex test functions — the rotated Rosenbrock valley, Ackley, Rastrigin,
  Griewank and similar, which combine curved or ridged geometry with (for the multimodal ones)
  many local optima.
- Protocol: report the mean or median over independent restarts of the number of function evaluations
  to reach a fixed target `f`-value, or the best `f`-value reached within a fixed evaluation
  budget; identical initialization across the compared methods; the optimum placed so that it is
  not at the initial mean. Internal strategy parameters are meant to be set by default from the
  dimension `n` and the population size, not hand-tuned per problem.

## Code framework

A randomized black-box optimizer plugs into a generic ask/tell loop that already exists: an
optimizer object owns the search-distribution state and exposes `ask()` to emit a candidate and
`tell()` to receive the evaluated population; an outer loop evaluates `f` and feeds the results
back. The Gaussian-sampling primitive (eigendecompose the covariance, draw `z ~ N(0, I)`, map to
`m + sigma B D z`) is standard and known. What is *not* settled is the entire rule that turns one
generation of ranked points into the next distribution `(m, sigma, C)` — that update is exactly
what is to be designed — so the substrate is only the loop, the sampler, and an empty update slot.

```python
import numpy as np


class DistributionSearch:
    """Generic randomized black-box optimizer over R^n. Owns the search
    distribution N(m, sigma^2 C) and updates it from ranked samples each
    generation. Uses only the ranking of f-values (rank-based selection),
    never their absolute magnitudes."""

    def __init__(self, mean, sigma, population_size, seed=None):
        self.dim = len(mean)
        self.m = np.asarray(mean, dtype=float)   # distribution mean
        self.sigma = float(sigma)                # overall step size
        self.C = np.eye(self.dim)                # covariance shape (eigendecomposable)
        self.lam = population_size               # offspring per generation
        self.rng = np.random.RandomState(seed)
        # TODO

    def _sample_one(self):
        # standard Gaussian sampling: z ~ N(0,I), x = m + sigma * B D z ~ N(m, sigma^2 C)
        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D2, 1e-20))
        z = self.rng.randn(self.dim)
        y = B @ (D * z)
        return self.m + self.sigma * y

    def ask(self):
        """Emit one candidate point to be evaluated."""
        return self._sample_one()

    def tell(self, solutions):
        """Receive the evaluated generation and update the distribution.

        solutions: list of (x, f(x)) for the lambda sampled points.
        """
        solutions = sorted(solutions, key=lambda s: s[1])   # rank by f, best first
        # TODO
        pass


# existing outer black-box loop the optimizer plugs into
def optimize(f, opt, n_generations):
    for _ in range(n_generations):
        solutions = []
        for _ in range(opt.lam):
            x = opt.ask()             # sample a candidate
            solutions.append((x, f(x)))   # evaluate the expensive objective
        opt.tell(solutions)           # update the search distribution
    return opt.m
```

The outer loop supplies one ranked generation per call; `tell()` is where the rule that produces
the next `(m, sigma, C)` will live.
