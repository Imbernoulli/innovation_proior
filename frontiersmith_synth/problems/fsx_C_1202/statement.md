# How Much Data Until It's Good Enough? — Forecasting Past the Floor

## Problem

A lab wants to know how much more training data to buy. You are given
**validation error vs. training-set size** `n` at a handful of
small-to-moderate sizes. Predict the error **far beyond** anything logged.
Overspending on data that buys nothing is as costly as under-buying.

Real learning curves shrink like a power law but never reach zero: label
noise and task ambiguity impose a strictly positive **irreducible error
floor** no amount of data can erase. Worse, the sizes you can afford to log
are rarely one clean regime — an early, fast representation-learning burst
decays much **more steeply** than the true long-run trend, before the curve
bends into its real asymptotic shape. Your table may straddle that bend
without telling you where it is.

**Illustrative FORM only — NOT the hidden law's shape:** `2.0 - 3.0 / (1.0 +
n) + 0.05 * sqrt(n)` is syntactically valid output (it shows the grammar
only — note it even *grows* with `n`); the real error law always decays
toward a positive floor, and its shape (floor, exponents, break point) is
different every test case and must be discovered from the data.

## Input (stdin)

```
m t
n_1 err_1
n_2 err_2
...
n_m err_m
```

`t` is the test id. `m` logged rows follow: a training-set size `n_i`
(positive integer) and the measured validation error `err_i` (a noisy
positive float) at that size. Rows are sorted by increasing `n`.

## Output (stdout)

One line: a closed-form Python expression for predicted error as a function
of `n`. Allowed: `+ - * / **`, unary `-`, parentheses, numeric constants, the
variable `n`, and the functions `sqrt`, `log`, `exp`, `abs`. No other names,
at most 40 syntax nodes.

## Feasibility

The output must be a single valid expression in `n` only, under the grammar
above, with finite numeric constants, at most 40 nodes. It is evaluated at
several held-out sizes; any parse failure, or any non-finite/undefined value
at any of them, scores the whole submission `0`.

## Objective (maximise a held-out fit quality)

The grader regenerates, from `t` alone, held-out sizes 3x to 2000x larger
than the largest `n` you were shown — genuine extrapolation, never logged —
together with the true (noiseless) error at each. Let `MSE` be your
expression's mean squared error there, and `B` the MSE of the constant
predictor `mean(err_1..err_m)` (the checker's own do-nothing baseline).
With a small fixed `EPS`:

```
L(x) = log(1 + max(x, EPS) / EPS)
Ratio = min(1000, 100 * L(B) / L(MSE)) / 1000
```

The constant baseline reproduces `Ratio ~= 0.1`. Lower held-out `MSE` raises
the score (capped at `1.0`); the log-compression keeps the scale sane even
though a floor-blind fit's error can be orders of magnitude worse than a
floor-aware one's, and clamping at `EPS` means beating the measurement-noise
floor by an ever finer margin does not keep buying score.

## Why the obvious fit is a trap

The logged rows look like a straight line on log-log paper, so the natural
first move is an ordinary least-squares power-law fit `err = A*n^(-alpha)`
over every row. That line is forced toward `0` as `n -> infinity` — but the
true error asymptotes to a strictly positive floor, so at 2000x scale the
fit is not just imprecise, it is qualitatively wrong. It is doubly wrong
because pooling all rows together also blends the steep early-burst exponent
into `alpha`, biasing even the decaying part of the prediction. The fix is
to notice the floor from how the curve's *curvature* bends upward relative
to a straight power law, and to fit only the largest-n rows that are
self-consistently past the early burst.

## Constraints

`1 <= t <= 10^5`. Time limit 5s, memory 512MB. `m` is 9 to 14 rows. Scoring
is fully deterministic.

## Example (worked score)

Suppose the baseline gives `B = 0.00841`, so `L(B) = log(1+8.41) = 2.242`.
A submission achieves `MSE = 0.00019` (below `EPS`, so clamped to `EPS`),
giving `L(MSE) = log(1+1) = 0.693`. Then `Ratio = min(1000,
100*2.242/0.693)/1000 = min(1000, 323.5)/1000 = 0.323`. (The baseline itself
always reports `Ratio ~= 0.1`.)
