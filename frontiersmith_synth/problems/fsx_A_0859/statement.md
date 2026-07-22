# The Jeweler's Wobbling Scale

A jeweler's balance scale has been freshly serviced and precisely calibrated:
no matter what load `x` (a positive normalised weight reading) you place on
the pan, the **mean** reading over many trials equals a fixed, known constant
`mu`. That's boring and given to you outright. What is NOT boring is the
**wobble**: repeated readings at the same load scatter around `mu` by an
amount that depends on the load through the scale's aging mechanism. Your job
is to recover the hidden **variance law** `g(x) = Var[reading | load = x]`
from a logbook of repeated readings taken only at **light** loads, then
predict the wobble at **heavier** loads the logbook never tested.

## Input (stdin)

```
n R mu t
x_1  y_1_1  y_1_2  ...  y_1_R
x_2  y_2_1  y_2_2  ...  y_2_R
...
x_n  y_n_1  y_n_2  ...  y_n_R
```

`t` is the test id. Each of the `n` lines gives one load `x_i` and `R`
repeated readings taken at that load. All loads are in a light range and all
readings share the same known mean `mu`.

## Output (stdout): one closed-form expression

Print **exactly one line**: an arithmetic expression for `g(x)` in the single
variable `x`, built from `+ - * / **`, parentheses, numeric constants, and the
unary functions `abs`, `sqrt`, `exp`, `log`, and `step` (Heaviside: `step(u)`
is `1` if `u > 0` else `0`). At most 40 expression nodes.

**Illustrative FORM only — NOT the hidden law:**

```
0.4 + 0.15 * exp(-x) + 0.05 * step(x - 2.0)
```

This just shows the allowed syntax; the real wobble law has a different shape
and you must discover it from the data.

## Feasibility

Your expression must parse under the grammar above (known functions/names
only, finite constants, node budget respected) and must evaluate to a
**finite, strictly positive** number at every load the grader checks — a
variance cannot be zero or negative. Any violation scores `0`.

## Objective (minimise)

The grader draws a fresh batch of readings at **heavier** loads, strictly
beyond the light range you were shown, and scores your `g` by the mean
Gaussian negative log-likelihood of the true held-out readings under
`Normal(mu, g(x))`:

```
F = mean over held-out readings of  [ 0.5*log(2*pi*g(x)) + (y - mu)^2 / (2*g(x)) ]
```

The internal baseline `B` uses the SAME formula but with a single constant
variance — the pooled empirical variance of your own training residuals about
`mu`, i.e. the "the wobble doesn't depend on load at all" guess:

```
Ratio = min(1000, 100 * B / F) / 1000
```

Matching the baseline exactly scores `Ratio ≈ 0.1`; a well-calibrated `g`
that tracks the true wobble at the heavier loads scores higher. Report the
lowest `F` (equivalently, highest `Ratio`) you can.

## Why the light-load fit is a trap

Over the light-load range you're shown, the scale's spring behaves
**elastically**: the wobble grows *multiplicatively* with load, roughly like
a power law. Past a **knee** load — never marked, and different for every
test — a worn bearing takes over and the wobble instead grows *additively*,
at a fixed extra amount per unit of extra load, starting from whatever wobble
already existed exactly at the knee. The two regimes' formulas **intersect**
at the knee by construction; that intersection point is what pins the correct
extrapolation. A single smooth curve fit to all the (load, variance) pairs at
once — the obvious move once you notice the wobble depends on load — blends
the two regimes into one compromise shape. That compromise tracks the light
loads you were shown tolerably well, then drifts onto the wrong curvature the
moment you're asked about a heavier load it never had to explain.

## Worked toy example (illustrative numbers, not the real law)

Suppose two training loads give empirical variances `v(1.0) = 0.30` and
`v(4.0) = 0.30 + 2*(4.0-2.0) = 4.30` under some hidden knee at `x0=2.0`.
Guessing the pooled constant `~1.5` scores `Ratio ≈ 0.1`; guessing the exact
piecewise law scores near the noise ceiling (never exactly `1.0` — sensor
noise keeps a floor on `F`).

## Constraints

Time limit 5 s, memory 512 MB. `n` is a few dozen loads with `R` repeats
each. Scoring is fully deterministic.
