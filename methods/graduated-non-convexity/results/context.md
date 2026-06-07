# Context: fitting through outliers and reconstructing across discontinuities

## Research question

We are given measurements and asked to estimate an unknown quantity from them: the parameters of
a model fit to data points, or a signal/surface reconstructed from a noisy field. The catch is that
the measurements are not merely perturbed by small well-behaved noise — a fraction of them are
*gross errors*: points that do not belong to the model at all (a mismatched correspondence, a
sensor glitch, a depth reading that straddles two surfaces). These are *outliers*. A second, related
form of the same difficulty appears in reconstruction: the underlying field is *piecewise* smooth,
varying continuously almost everywhere but jumping at boundaries, so the "smoothness" we want to
impose is violated at a sparse set of *discontinuities* — which behave, locally, exactly like
outliers of the smoothness assumption.

The goal is an estimator that (1) fits the bulk of the data well, (2) is essentially *unaffected* by
the outliers — it must not let a small number of arbitrarily-bad points drag the solution — and (3)
preserves the genuine discontinuities rather than smearing across them. The obstacle is structural:
the cost function that achieves (1)–(3) is non-convex, with many local minima, and the global
minimum (the one we actually want) is hard to reach by descent from an arbitrary starting point.
A usable method has to deliver the robust solution *without* relying on a good initial guess, and at a
cost far below brute-force global search.

## Background

**Ill-posed reconstruction and regularization.** Many estimation problems in early vision are
ill-posed — underconstrained and noise-sensitive — and are made well-posed by Tikhonov
regularization (Tikhonov & Arsenin): add a stabilizing term that prefers smooth solutions. For
surface recovery from data `d` this reads `E(u) = Σ_s (u_s − d_s)² + λ Σ_{s~t} (u_s − u_t)²`, a
quadratic, convex energy with a unique minimizer. Its defect is exactly that it is everywhere
smooth: it blurs across depth discontinuities, because a large gradient `(u_s − u_t)²` is penalized
without bound, so the optimizer would rather round off a true edge than pay the quadratic price.

**Line processes (Geman & Geman 1984).** To preserve edges, introduce a binary *line process*
`l_{st} ∈ {0,1}` on the dual lattice of neighbor pairs, and write the smoothness term as
`(u_s − u_t)²(1 − l_{st}) + α l_{st}`. Setting `l_{st}=1` *cuts* the bond between `s` and `t`: the
smoothness penalty is replaced by a fixed cost `α`, so a discontinuity can be introduced wherever the
gradient is too steep to be worth smoothing. This recovers piecewise-smooth surfaces, but the
binary variables make the energy non-convex; Geman and Geman minimized it with stochastic
simulated annealing — correct in the limit but very expensive. Hinton's (1978) earlier "weak
constraint" expresses the same idea: enforce smoothness only while neighbors are similar enough,
and drop it past a threshold.

**Eliminating the line process → the truncated quadratic.** If no spatial constraints couple the
line variables, one can minimize over `l_{st}` analytically and remove it. For each bond,
`min_{l∈{0,1}} [ (u_s−u_t)²(1−l) + α l ] = min( (u_s−u_t)², α )`. The smoothness term becomes a sum
of *truncated quadratics* `g(t) = min(t², α)` in the gradient `t = u_s−u_t`: quadratic for small gradient, but capped at `α`
once the gradient exceeds a threshold, at which point the bond contributes a constant and the edge is
free. This is the *weak string* (1D) / *weak membrane* (2D) energy. It is the right cost — and it is
non-convex: the cap is where the local minima come from.

**Robust statistics and influence functions (Huber 1981; Hampel et al. 1986).** Independently, the
robust-statistics literature studies M-estimators: fit parameters `a` by `min_a Σ_i ρ((d_i −
u(i;a))/σ)` for an *error norm* `ρ`. The quadratic `ρ(x)=x²` is least squares; its *influence
function*, proportional to `ψ = ρ'`, is `ψ(x)=2x`, growing without bound — so a single outlier exerts
unbounded pull. A goal of robust statistics (Hampel) is to "describe the structure best fitting the
bulk of the data" while identifying deviating points. To bound the pull, `ρ` must rise *more slowly*
than `x²`. Huber's minimax norm is quadratic for `|x|<ε` and linear beyond, giving a bounded but
non-vanishing influence; it is convex, hence easy to optimize, but its influence never returns to
zero, so it tolerates only moderate contamination (low breakdown point). *Redescending*
estimators push further: their influence `ψ(x) → 0` as `|x|→∞`, so a sufficiently gross outlier is
*ignored*. Examples include the skipped mean (whose `ρ` is exactly the truncated quadratic), the
Lorentzian `ρ = log(1 + ½(x/σ)²)`, Tukey's biweight, and Geman–McClure `ρ(x)=x²/(1+x²)`. All
redescending norms are non-convex.

So the two traditions meet on the same object. Preserving discontinuities (line process eliminated)
and rejecting outliers (redescending M-estimator) both demand a cost that is quadratic near zero and
flattens far away — and both pay for the strength of that rejection with non-convexity.

**The IRLS view (Beaton & Tukey 1974).** An M-estimate is commonly computed by iteratively
reweighted least squares: at each step, weight residual `x` by `z = ρ'(x)/(2x)` and solve a weighted
least-squares problem `min Σ x² z`. The weight `z` down-weights large residuals automatically. But
in IRLS `z` is a *function of* `x`, not a free variable; it offers no explicit handle for declaring a
point an outlier, and no place to attach prior assumptions about *where* outliers occur.

**Continuation / deterministic annealing as an alternative to stochastic search.** Stochastic
annealing escapes local minima by accepting uphill moves with a temperature-controlled probability,
cooling slowly. A deterministic alternative is *continuation*: deform the objective itself. Start
from a smoothed/convexified version of the cost whose minimizer is easy and unique, then slowly
deform the cost back toward the true (non-convex) one, re-minimizing from the previous solution at
each step — tracking a minimizer from the easy landscape into the hard one. This is the same spirit
as scale-space and coarse-to-fine analysis. The open question such a method must answer is concrete:
what one-parameter family of costs starts convex and ends at the true cost, and how slowly must the
parameter move so the tracked minimizer stays near the global one?

## Baselines

**Least squares (quadratic `ρ`).** `min_a Σ_i (d_i − u(i;a))²`, or with a smoothness term as above.
Convex, unique global minimum, closed-form for linear models. *Gap:* influence `ψ=2x` is unbounded,
so a single gross outlier biases the fit arbitrarily; a handful of outliers wreck it. In
reconstruction the quadratic smoothness term blurs every discontinuity.

**Line-process regularization (Geman & Geman 1984).** Binary `l_{st}` cuts smoothness bonds at
edges; minimized by stochastic simulated annealing. *Gap:* non-convex with combinatorial binary
variables; stochastic minimization is slow, and the result depends on the cooling schedule.

**Convex robust M-estimators (Huber 1981; Shulman & Hervé 1989).** Replace the quadratic with a
convex robust norm (Huber, L1). *Gap:* a convex `ρ` cannot have a redescending influence, so its
breakdown point is low — it tames moderate outliers but is still pulled by gross ones. Convexity
buys tractability at the cost of robustness.

**Redescending M-estimators (skipped mean / truncated quadratic, Lorentzian, Tukey, Geman–McClure).**
Influence `ψ → 0` for large residuals, so gross outliers are rejected; high breakdown point. *Gap:*
non-convex `ρ`, so plain gradient or coordinate descent gets trapped in a local minimum and the
result depends heavily on initialization — there is no built-in mechanism to find the *global*
minimum without a good starting guess.

**Iteratively reweighted least squares (Beaton & Tukey 1974).** Alternate: reweight residuals by
`z = ρ'(x)/(2x)`, solve weighted least squares. *Gap:* converges to a local minimum of the chosen
(possibly non-convex) `ρ` near the initialization; `z` is slaved to `x` with no explicit
outlier variable and no way to impose spatial structure on the outliers.

## Evaluation settings

The natural testbeds are estimation problems with a controllable outlier fraction and a known
ground-truth fit. Linear regression / line fitting is the clean quantitative stand-in: generate
`N` points from a known model `y = a t + b` with small Gaussian noise, then corrupt a prescribed
*fraction* (e.g. up to 70–90%) by replacing their values with gross errors drawn from a wide range;
the metric is the error of the recovered parameters against ground truth, and the precision/recall of
which points the method flags as inliers vs outliers. The reconstruction analogue is the
piecewise-smooth surface (the "wedding cake": several flat layers at distinct depths, corrupted by
noise and a few gross errors), where the metric is fidelity of the recovered surface to the original
and correct placement of the discontinuities. The inlier noise scale (and hence a residual threshold
separating inliers from outliers) is assumed known — e.g. set from the measurement noise standard
deviation via a chi-squared quantile. The natural yardstick is the quality reached by plain least
squares and by convex robust estimators under the same contamination.

## Code framework

The pieces that already exist: a way to assemble and solve a (weighted) least-squares problem for the
outlier-free model, a residual evaluation, a robust error norm with a control on its shape, and an
outer loop that adjusts that control. The open slots are the one-parameter family of costs (from a
convex member to the true robust cost), the per-point weight/outlier update, and the continuation
loop that ties them together.

```python
import numpy as np


def weighted_least_squares(A, y, w):
    """Global solution of the outlier-free model, weighted per measurement:
    argmin_x  sum_i w_i (A_i x - y_i)^2.  Closed form for a linear model."""
    W = np.sqrt(w)[:, None]
    Aw, yw = A * W, y * np.sqrt(w)
    return np.linalg.solve(Aw.T @ Aw, Aw.T @ yw)


def robust_weight_update(r2, mu, barc2):
    """Per-measurement weight w_i in [0,1] minimizing the dual outlier process
    for the surrogate cost at control value `mu`, given squared residuals r2."""
    # TODO: closed-form weight for the chosen robust cost / surrogate
    raise NotImplementedError


def initial_control_value(r2, barc2):
    """Pick the control value at which the surrogate is convex over the data."""
    # TODO: convexifying initial control
    raise NotImplementedError


def continuation_robust_fit(A, y, barc2, factor=1.4, max_iter=1000, cost_threshold=0.0):
    """Recover the robust fit by continuation in the control parameter."""
    w = np.ones(A.shape[0])
    x = weighted_least_squares(A, y, w)
    r2 = (A @ x - y) ** 2
    mu = initial_control_value(r2, barc2)
    prev_cost = 0.0
    # TODO: outer continuation loop:
    #   repeat { weight update from current residuals; weighted variable update;
    #            residual/cost check; move `mu` one step toward the true cost }.
    raise NotImplementedError
```

The open question the loop must answer is how to make each inner minimization easy (weighted least
squares plus a closed-form weight), how to set the first control value so the landscape starts convex,
and how to step the control toward the true non-convex cost slowly enough that the tracked minimizer
stays on the desired continuation path.
