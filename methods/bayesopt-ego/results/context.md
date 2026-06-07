# Context: global optimization of expensive black-box functions (circa 1998)

## Research question

In the automotive and semiconductor industries — and across engineering generally — designs
are increasingly explored through math/computer models rather than hardware prototypes. The
trouble is that one evaluation of such a model is *expensive*: an automotive crash simulation
can take twenty hours; a circuit or finite-element code can take minutes to hours per run. The
function `y(x)` we want to minimize over a box of design variables is therefore an expensive,
deterministic black box. We can call it, but only a few dozen times. We get no gradient (the
code is a black box), the function is typically nonlinear and may be multimodal, and we have no
analytic form for it — if we did, we would not have built the expensive code in the first place.

The precise goal: find the global minimum of `y(x)` in as *few evaluations as possible*, and,
ideally, know when to stop — a credible, self-contained estimate of how much more there is to
gain from further sampling. A method that needs hundreds or thousands of evaluations, or that
silently converges to a local minimum, is unusable here. The challenge is that the two things
one would naturally do — sample where the function looks lowest, and sample where the function
is poorly understood — pull in opposite directions, and neither alone solves the problem. Any
solution must balance *exploiting* what is already known about the surface against *exploring*
regions where the surface is uncertain, and must do so automatically, without hand-tuning a
trade-off knob per problem.

## Background

The dominant pain points of the time, and the load-bearing concepts a solution rests on:

- **Function evaluations are the budget.** Because each call to `y(x)` is so costly, the figure
  of merit is the number of evaluations to reach the optimum, not CPU time spent reasoning
  *about* where to evaluate. It is worth solving a hard auxiliary optimization at each step if it
  saves one call to the real function. This inverts the usual economics of optimization.

- **Deterministic codes have no measurement noise — but their "error" is correlated.** A
  deterministic computer model returns the same output for the same input every time. So when a
  simple model `Σ_h β_h f_h(x) + ε` fails to fit, the residual `ε(x)` is not noise; it is the
  collection of left-out terms in `x`. If `y(x)` is continuous, so is that residual, which means
  the residuals at two nearby points `x_i` and `x_j` are nearly equal — *correlated*, with the
  correlation high when the points are close and low when they are far apart. Treating those
  residuals as independent (as classical regression does) is a structurally false assumption for
  a deterministic code, and it is the conceptual wedge that the whole approach turns on.

- **A surrogate must do two jobs at once.** To choose evaluations wisely we need a cheap stand-in
  for `y(x)` that supplies, at any candidate `x`, both a prediction `μ(x)` of the function value
  *and* an honest measure `σ(x)` of how uncertain that prediction is. The prediction lets us
  exploit; the uncertainty lets us explore. A surrogate giving only a point prediction (ordinary
  regression, splines) cannot drive global search, because it cannot say where it is ignorant.

- **Multimodality defeats local and surrogate-greedy search.** It is well understood that on a
  multimodal objective, local methods (and the naive scheme of fitting a surface, jumping to its
  minimum, resampling, and iterating) converge to whichever basin they start in. The minimum of
  a fitted surface sits at a *local* minimum of the data; chasing it just refines that local
  minimum. This failure is the motivating phenomenon: pure exploitation of a surrogate is not
  global optimization.

- **The opposite extreme is just as bad.** Sampling wherever the surrogate is most uncertain
  spends the entire (tiny) budget on charting empty regions of the box and never converges on the
  optimum either. So the real problem is the *balance*, not either pole.

Two bodies of theory sit in the background and supply the pieces. First, **geostatistics /
kriging** (Matheron 1963, *Principles of geostatistics*; Krige; surveyed by Cressie): model an
unknown spatial field as a realization of a correlated random process, with a correlation that
decays with distance, and derive the best linear unbiased predictor of the field at an unsampled
location together with its mean-squared prediction error. This is exactly a surrogate that emits
both `μ(x)` and `σ(x)`, and it *interpolates* the data. Second, the **Bayesian / random-function
view of global optimization**: treat the unknown objective as a draw from a stochastic process,
so that the value at an unevaluated point is a random variable with a computable distribution,
and choose the next evaluation by a decision-theoretic criterion over that distribution.

## Baselines

The prior methods a new global optimizer would be measured against or built upon:

- **Linear-regression response surfaces / classical RSM** (Box, Hunter & Hunter 1978). Fit
  `y(x_i) = Σ_h β_h f_h(x_i) + ε_i` with independent zero-mean errors of variance `σ²`, estimate
  the `β_h` by least squares, optimize the fitted surface. Two gaps for an expensive deterministic
  code: (1) one must *choose* the regressors `f_h`, but the right functional form is unknown
  (that is why the expensive code exists); a flexible form has many parameters and so needs many
  evaluations to fit. (2) The independent-error assumption is false for a deterministic code — the
  "errors" are correlated continuous residuals — so the fit does not interpolate the data and the
  model gives no trustworthy local measure of its own uncertainty. It cannot tell you where it is
  ignorant, so it cannot drive global search.

- **Kriging / DACE — the "Design and Analysis of Computer Experiments" stochastic-process model**
  (Sacks, Welch, Mitchell & Wynn 1989; building on Matheron 1963 and the kriging literature).
  Replace the false independent-noise model with a constant mean plus a *correlated* random
  field: `y(x_i) = μ + ε(x_i)`, where `ε(x)` is a zero-mean Gaussian process with variance `σ²`
  and a correlation that decays with a special weighted distance,
  `d(x_i, x_j) = Σ_h θ_h |x_{ih} − x_{jh}|^{p_h}` (with `θ_h ≥ 0`, `p_h ∈ [1,2]`) and
  `Corr(ε(x_i), ε(x_j)) = exp(−d(x_i, x_j))`. Here `θ_h` encodes the *activity* / relevance of
  variable `h` (large `θ_h` ⇒ correlation falls off fast in that coordinate ⇒ the function is
  sensitive to it) and `p_h` encodes *smoothness* (`p_h = 2` smooth, near 1 rougher). The model
  has `2k + 2` parameters `μ, σ², θ_1..θ_k, p_1..p_k`, fit by maximum likelihood; given the
  correlation parameters, `μ̂` and `σ̂²` have closed forms, leaving a concentrated likelihood in
  the `θ_h, p_h` to maximize. The resulting best linear unbiased predictor
  `ŷ(x*) = μ̂ + r' R⁻¹ (y − 1μ̂)` interpolates the data, and its mean-squared error `s²(x*)` is
  zero at sampled points and rises to about `σ²` far from them. This *is* the `μ(x), σ(x)`
  surrogate the problem demands; what it does not yet supply is a rule for *where to sample next*.

- **Probability-of-improvement search** (Kushner 1964). Model the 1-D objective as a Wiener
  process; at each step sample the point that maximizes the probability that the new value beats
  the current best, `P(Y(x) < f_min)`, with a parameter that nudges the search "more global" or
  "more local". Gap: probability of improvement counts *whether* you improve, not *by how much*,
  so it is biased toward exploitation — it favors points clustered just below the incumbent where
  a small gain is nearly certain, and the global/local trade-off has to be set by hand and per
  problem. And it is posed in one dimension on a Wiener model.

- **Expected-improvement / Bayesian extremum-seeking** (Mockus, Tiesis & Zilinskas 1978;
  Mockus 1994). Pose global optimization in the random-function framework and score a candidate
  by the *expected value* of the improvement it would bring, not merely the probability of some
  improvement — so the magnitude of a potential gain is counted. This is the conceptual seed of an
  acquisition criterion that balances local and global search on its own. What was missing was a
  practical surrogate to evaluate it on and a way to optimize the (multimodal) criterion to
  guaranteed optimality, so the idea had not become a turnkey algorithm with a stopping rule.

## Evaluation settings

The natural yardstick is a small set of standard multimodal global-optimization test functions
over boxes (Dixon & Szegő 1978): the Branin function (2-D), the Goldstein–Price function (2-D),
and the Hartman 3 and Hartman 6 functions (3-D and 6-D). The metric is the number of expensive
function evaluations needed to locate the global minimum to a target relative accuracy (e.g.
within 1% of the true optimal value), counting the space-filling initial design plus the
sequentially chosen points. Diagnostics for the surrogate itself use cross-validated standardized
residuals (a well-fit model keeps them below about 3 in magnitude), with a log or `−1/y`
transformation of the response considered when the diagnostics look poor. The space-filling
initial designs are Latin hypercube designs (McKay, Conover & Beckman 1979) chosen so that their
one- and two-dimensional projections cover the box nearly uniformly.

## Code framework

The pieces that already exist: a numerical-linear-algebra stack (for `R⁻¹`, determinants, and an
SVD to cope with a near-singular correlation matrix), a nonlinear optimizer to maximize a
likelihood and to maximize an auxiliary criterion over a box, and a Latin-hypercube design
generator. The surrogate-driven optimization loop is the scaffold to be filled in:

```python
import numpy as np

def latin_hypercube(n_points, bounds, rng):
    """Space-filling initial design over the box `bounds`."""
    # ... draw an LHS with good low-dimensional projections ...
    raise NotImplementedError

class CorrelatedSurrogate:
    """A cheap stand-in for the expensive objective that emits, at any x,
    a prediction AND an honest uncertainty."""

    def fit(self, X, y):
        # TODO: fit the correlated random-field model to the data
        #       (estimate the correlation parameters by maximum likelihood,
        #        close the mean/variance, form R and its inverse).
        pass

    def predict(self, x, return_std=True):
        # TODO: return mu(x) [interpolating the data] and sigma(x)
        #       [zero at sampled points, growing away from them].
        pass

def acquisition(x, surrogate, f_min):
    # TODO: the figure of merit that balances exploiting low mu(x)
    #       against exploring high sigma(x). This is the slot the method fills.
    pass

def maximize_acquisition(acq_fn, bounds, rng):
    """Globally maximize a cheap multimodal criterion over the box."""
    # TODO: combine broad search over the box with local polishing.
    pass

def optimize_expensive(objective, bounds, n_init, max_evals, seed=0):
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n_init, bounds, rng)
    y = [objective(x) for x in X]          # the only expensive calls
    surrogate = CorrelatedSurrogate()
    for _ in range(max_evals - n_init):
        surrogate.fit(X, y)
        f_min = min(y)
        # pick the next point by maximizing the acquisition over the box
        acq_fn = lambda Xcand: acquisition(Xcand, surrogate, f_min)
        x_next, acq_value = maximize_acquisition(acq_fn, bounds, rng)
        # TODO: stopping rule based on the acquisition's own value
        y_next = objective(x_next)         # one expensive call
        X = np.vstack([X, x_next]); y.append(y_next)
    return X[np.argmin(y)], min(y)
```
