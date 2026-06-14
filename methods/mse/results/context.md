## Research question

A recurring situation in practical astronomy and geodesy is this. We want a small number of
unknown quantities — orbital elements of a comet or planet, the ellipticity of the Earth, the
length of a meridian arc. Each observation gives us, after the geometry is worked out, one
(approximately linear) equation relating those unknowns. We can take more observations than we
have unknowns, so the system is *over-determined*: there are more equations than degrees of
freedom. Every observation carries some error, however small, so in general no single choice of
the unknowns satisfies all the equations at once; the residuals cannot all be driven to zero.

The problem is then to combine the discordant equations into one definite answer. We need a
*principled, unique, and computable* rule for "distributing the errors" among the equations —
not an ad-hoc choice of which observations to trust. A satisfactory rule would have to: (1)
use *all* the data rather than discarding equations down to an exactly solvable subset; (2)
produce a single determinate system to solve, for any number of unknowns, not just two; (3)
reduce, in the simplest case of many equally trustworthy direct measurements of one quantity,
to the answer everyone already accepts there — the arithmetic mean; and (4) ideally come with
some justification for *why* it is the right rule, beyond mere convenience. Each existing
approach below meets some of these and fails others. Closing that gap is the problem.

## Background

The state of the art rests on a handful of load-bearing ideas.

**The additive-error observation model.** An observation is taken to be the true value of some
function of the unknowns, corrupted by a random error: `observed = true + Δ`. The errors are
assumed symmetric (a positive error of a given size is as likely as the equal negative one),
and small errors are assumed more frequent than large ones, with the chance of an error of
magnitude `Δ` described by some density `φ(Δ)` peaked at `Δ = 0` and decaying toward zero as
`|Δ|` grows. This is the framework within which "how probable is this set of observations" can
even be asked.

**Independence and the product rule.** Distinct observations are treated as independent, so the
joint probability of a whole set of residuals is the *product* of their individual densities,
`Π_i φ(Δ_i)`. This converts "find the most likely values of the unknowns" into "maximize a
product of `φ`'s" — equivalently, with a flat prior over the unknowns, the values that make the
observed data most probable are the values that maximize that product.

**Laplace's probability-of-causes (1774).** Laplace had already framed inference as inverting
from observed effects to the probability of their causes, and had studied candidate error laws
(for instance a density falling off like `exp(−m|x|)`). The framework was in place; what it
produced, though, depended entirely on which `φ` one chose, and the laws on offer led to
awkward estimators (medians and the like) rather than to anything as clean as a single linear
system. There was no canonical choice of `φ`, hence no canonical estimator.

**The arithmetic mean as the trusted anchor.** For repeated, equally careful *direct*
measurements of a single quantity, the accepted "most probable value" is the arithmetic mean of
the readings. This is a deeply held, near-universal practice — but it is a recipe for one
special case, with no agreed extension to systems of several coupled unknowns.

**Laplace's normalization integral.** The definite integral `∫_{−∞}^{∞} exp(−h²Δ²) dΔ = √π / h`
was already known (Laplace). Any candidate error density of Gaussian shape can therefore be
normalized to integrate to one, fixing its leading constant.

In the over-determined case, if a candidate solution exactly satisfies a chosen subset of the
equations, it will in general *disagree* with the remaining ones by amounts on the order of the
observation errors, and which subset one privileges changes the answer. Forcing some equations
to hold exactly makes the result hostage to the small errors in precisely those equations.

## Baselines

The prior approaches a new principle would be measured against.

**Solve an exact square subsystem, discard the rest.** With `v` unknowns, select `v` of the
equations, solve that square system exactly, ignore the surplus. Concretely simple, and it
gives a unique answer once the subset is fixed. **Gap:** the choice of subset is arbitrary and
the answer depends on it; the discarded equations carry real information that is thrown away;
and the result is dominated by whatever errors happen to sit in the chosen equations, since
those are forced to be satisfied exactly. It uses part of the data, not all of it.

**Boscovich's method / minimizing total absolute error (1757).** Choose the unknowns to make
the sum of the absolute residuals `Σ|Δ_i|` a minimum, often under the side condition that the
residuals sum to zero. A genuine optimization principle that uses all the data and has an
appealing robustness to large outliers. **Gap:** in practice it was carried through only for a
single unknown (or two), because the objective is non-smooth — `|·|` has no derivative at zero —
so there is no clean stationarity equation to solve; the minimization is combinatorial,
hinging on which residuals are zero at the optimum, and does not reduce to a uniform linear
system in many unknowns.

**Minimax (smallest maximum error).** Choose the unknowns to make the largest absolute residual
`max_i |Δ_i|` as small as possible. Also a real principle, also robust in its own way. **Gap:**
likewise limited in practice to very few unknowns; the objective is non-smooth and its optimum
is governed by a small set of "active" equations where the maximum is attained, again giving a
combinatorial problem rather than one determinate linear system, and again with no derivative
condition to exploit.

**The arithmetic mean, taken as a primitive.** For `n` equally good direct observations
`M, M', …` of one quantity, take `(M + M' + ⋯)/n`. Trusted, optimal-feeling, and the thing any
general method must agree with in this case. **Gap:** it is defined only for the single-quantity
direct-observation case; it offers no route to several coupled unknowns observed through
different functions, and on its own it gives no reason *why* it is the most probable value — it
is accepted by practice, not derived.

## Evaluation settings

The natural proving grounds are:

- **Orbit determination of a comet or minor planet.** A handful of unknown orbital elements,
  many more telescopic position measurements than elements, each measurement reduced to a
  conditional equation in the elements. The yardstick is how consistently the fitted orbit
  represents *all* the observations, including ones not used to pin it down, and how stable the
  fit is to small observation errors.
- **Geodesy — figure of the Earth and the length of the meter.** Several arc-length / latitude
  measurements constraining the Earth's ellipticity and the meridian-quadrant length; again
  more measurements than unknowns, combined into a few parameters.
- **Repeated direct measurement of a single quantity.** The degenerate but decisive case: many
  equally careful readings of one number. Any general combination rule must here return the
  accepted answer; this case is the sanity check, not a hard test.

The experimental protocol is the same throughout: reduce each observation to a (linear)
equation in the unknowns, treat observations as equally good unless there is reason otherwise,
and report the combined estimate; reliability is judged by agreement with the held-out
equations and by insensitivity to small perturbations of the data.

## Code framework

The prediction cost plugs into a setting where target feature maps and predicted feature maps
already have the same shape. An outer routine calls a small module to turn their elementwise
discrepancy into a single scalar to be made small. The available substrate is only the generic
machinery: same-shaped tensors, elementwise arithmetic, and a slot that collapses the discrepancy
to one number.

```python
import torch.nn as nn


class CustomPredictionLoss(nn.Module):
    """Prediction cost. Maps the discrepancy between target and predicted
    feature maps to a scalar, lower meaning the prediction agrees better with
    the target."""

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        # state, predicted: same-shaped arrays of corresponding quantities.
        residual = state - predicted
        # TODO: collapse the residual array to one scalar discrepancy.
        raise NotImplementedError


# existing outer routine the rule plugs into
def fit(predict_fn, observations, unknowns, optimizer, predcost, num_steps):
    # predict_fn(unknowns) yields, for the current unknowns, the predicted
    # counterpart of each observation (same shape as observations).
    for _ in range(num_steps):
        optimizer.zero_grad()
        predicted = predict_fn(unknowns)          # current predictions
        loss = predcost(observations, predicted)  # collapse residuals to a scalar
        loss.backward()                           # sensitivities of the scalar
        optimizer.step()                          # adjust the unknowns
```

The outer routine supplies the pair of same-shaped arrays; the body of
`CustomPredictionLoss.forward` supplies the scalar consumed by the optimizer.
