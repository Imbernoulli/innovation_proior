# Ordinary Tides Plus the Storm That Is Coming

## Problem

A coastal water-level station logs two decoupled drivers at every timestamp:
the astronomic **tide** `T(t)` (a fixed sum of harmonic constituents --
frequencies fixed for the season, amplitudes and phases specific to this
station) and a **surge-forcing proxy** `S(t)` (wind/pressure driven). What
the gauge actually *records*, though, is not their plain sum. In shallow
water the tide and the storm surge interact nonlinearly: the surge peak is
suppressed when it coincides with high tide. The true water level is

```
y = T + S - kappa * T * S + (sensor noise)
```

where `kappa` is a shallow-water interaction coefficient, independently
surveyed from this station's bathymetry -- it is **not** something you can
refit from a calm week (see below).

You are given a log recorded during ordinary, calm weather. In that regime
the surge `S` stays small, so the `kappa*T*S` correction is far smaller
than the sensor noise: it is statistically invisible in the very data you
are holding. You will be graded on a **different, held-out event**: a
storm surge, timed to peak near a **high tide** -- exactly where the
interaction term stops being negligible. That event, and its true water
levels, are never shown to you.

## Input (stdin)

```
testId n_train
kappa
T_1 S_1 y_1
T_2 S_2 y_2
...
T_{n_train} S_{n_train} y_{n_train}
```

`n_train` calm-period rows follow the header, each the tide value, the
surge-forcing value, and the observed (noisy) water level. `kappa` is the
station's surveyed interaction coefficient (a float, given directly --
use it or not).

## Output (stdout)

Print **one line**: a single arithmetic expression, over the variables `T`
and `S`, that predicts the water level. Allowed tokens: `+ - * / **`,
parentheses, numeric constants, and the unary functions `abs`, `exp`,
`sin`, `cos`, `sqrt`, `tanh`. No other names, no conditionals or
comparisons.

**Illustrative FORM only -- NOT the hidden law:**

```
2 * T - 0.5 * T ** 2
```

This shows only the syntax (a single-variable quadratic). The predictor
you actually need generally depends on **both** `T` and `S`, and must be
derived from the data and the stated mechanism, not copied from here.

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, at most 60 expression nodes). Any violation, or
any non-finite value produced while evaluating it on the held-out rows,
scores `0`.

## Objective (maximise)

The grader regenerates the held-out storm event deterministically from
the test id (never from your output or the training rows), evaluates
your expression at each held-out `(T, S)`, and forms

```
F = mean squared error of your prediction on the held-out storm
B = mean squared error of the internal baseline "predict T, ignore the surge entirely"
Ratio = min(920, 100 * B / F) / 1000
```

Reproducing the baseline scores about `0.1`. Folding in the surge (even
without the interaction) does much better. Applying the interaction
correction generally does best of all -- but the ratio is capped below
`1.0` (sensor noise plus the cap keep the ceiling open above any
reference solution).

## Why the calm log is a trap

On the calm log, `S` is small everywhere, so `kappa*T*S` is swamped by
noise: nothing in the training residuals will ever look like an
interaction term, no matter how carefully you regress. A model fit purely
to minimise training error therefore has no statistical evidence to
include one -- and reproduces the plain sum `T+S`, which looks excellent
on calm data and is materially wrong exactly where a large surge lines up
with a high tide, which is precisely the event you are graded on.

## Constraints

Time limit 5 s, memory 512 MB. `n_train = 150`. All scoring is fully
deterministic; the storm event, the sensor noise, and the harmonic
constituents themselves never depend on wall time, run-time randomness,
or your output.
