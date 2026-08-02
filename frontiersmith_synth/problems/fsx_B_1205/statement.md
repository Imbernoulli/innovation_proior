# Harvest After a Heat Wave at the Wrong Moment

## Problem

A breeding station keeps a season-by-season yield log for one field. For
every recorded season it distills the daily weather down to two
summary numbers, both derived from that season's own temperature trace:

- **G** -- total season **growing-degree-days**: the standard heat
  accumulation used to predict crop growth, summed over every day of the
  season with each day's contribution capped once temperature exceeds a
  physiological ceiling (extra heat beyond the ceiling helps growth no
  further).
- **H** -- heat **exceedance accumulated only across that season's own
  FLOWERING window**: a stretch of days located by that season's own
  thermal history (a warm spring pulls flowering earlier, a cool spring
  pushes it later), NOT a fixed calendar date. Unlike G, exceedance
  degrees above the extreme-heat threshold are never capped.

The true agronomic law behind the logged yield is

```
yield = Y0 + a*G - BETA*H**2 + (measurement noise)
```

`BETA` is a physiological flowering heat-sensitivity coefficient for this
cultivar, independently measured in a growth-chamber assay -- **not**
something you can refit from this log (see below). It is given directly
in the header.

You are given a log of ordinary seasons. In an ordinary season the
flowering window essentially never sees an extreme-heat excursion, so
`H` stays tiny and `BETA*H**2` is far smaller than the measurement
noise: statistically invisible in the data you are holding. You will be
graded on a **different, held-out season**: a heat wave, timed to land
on top of THAT season's own flowering window -- exactly where the
squared penalty stops being negligible. That event, and its true yield,
are never shown to you.

## Input (stdin)

```
testId n_train
BETA
G_1 H_1 y_1
G_2 H_2 y_2
...
G_{n_train} H_{n_train} y_{n_train}
```

`n_train` historical-season rows follow the header, each giving the
season's total GDD, its flowering-window heat exceedance, and the
observed (noisy) yield. `BETA` is the survey constant (a float, given
directly -- use it or not).

## Output (stdout)

Print **one line**: a single arithmetic expression, over the variables
`G` and `H`, that predicts yield. Allowed tokens: `+ - * / **`,
parentheses, numeric constants, and the unary functions `abs`, `exp`,
`sin`, `cos`, `sqrt`, `tanh`. No other names, no conditionals or
comparisons.

**Illustrative FORM only -- NOT the hidden law:**

```
50 + 0.01 * G - 0.4 * sqrt(abs(H))
```

This shows only the syntax. The predictor you actually need generally
depends on **both** `G` and `H`, and must be derived from the data and
the stated mechanism, not copied from here.

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, at most 60 expression nodes). Any violation, or
any non-finite value produced while evaluating it on the held-out rows,
scores `0`.

## Objective (maximise)

The grader regenerates the held-out stress season deterministically from
the test id (never from your output or the training rows), evaluates
your expression at each held-out `(G, H)`, and forms

```
F = mean squared error of your prediction on the held-out season
B = mean squared error of the internal baseline "fit yield on total
    season GDD alone, ignore exactly when within the season any
    extreme heat fell"
Ratio = min(920, 100 * B / F) / 1000
```

Reproducing the baseline scores about `0.1`. Folding in `H` (even
linearly) does somewhat better. Applying the given quadratic,
flowering-restricted correction generally does best of all -- but the
ratio is capped below `1.0` (measurement noise plus the cap keep the
ceiling open above any reference solution).

## Why the historical log is a trap

`H` is small almost everywhere in the log, so `BETA*H**2` is swamped by
noise: training residuals never show a strong, curved flowering-stress
signal. A model fit purely to minimise training error has little reason
to expect anything beyond a mild linear nuisance term -- and badly
under-predicts the loss exactly when a large, concentrated heat wave
lines up with the flowering window, which is precisely the event you
are graded on.

## Constraints

Time limit 5 s, memory 512 MB. `n_train = 60`. All scoring is fully
deterministic; the stress season, the sensor noise, and the field's
weather parameters themselves never depend on wall time, run-time
randomness, or your output.
