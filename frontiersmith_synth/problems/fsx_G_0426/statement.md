# Period-Luminosity-Color Calibration from a Variable-Star Survey

## Problem
A wide-field time-domain survey has catalogued a population of pulsating
variable stars. For each star the pipeline has measured three observables and,
for the nearby calibration sample, an absolute magnitude derived from an
independent distance:

- `x1` = log10(pulsation period / day)
- `x2` = mean color index (V - I)
- `x3` = metallicity [Fe/H]
- `y`  = absolute magnitude `M_V` (brighter stars are MORE NEGATIVE)

There exists a smooth ground-truth **period-luminosity-color-metallicity (PLZ)
relation** `M_V = f(x1, x2, x3)` (unknown to you). You are given noisy training
measurements sampled ONLY from the well-populated short/medium-period regime
`x1 in [0, 1]` (periods 1 to ~10 days). Your calibration will be judged on a
held-out sample of **long-period stars** `x1 in [1.0, 1.8]` (periods up to ~63
days) - a genuine extrapolation beyond the training range. Recover a closed-form
relation that generalizes to that long-period tail.

## Input (stdin)
The first line contains two integers `n_train test_id`. Each of the next
`n_train` lines contains four floats:
```
x1 x2 x3 y
```
Only the training sample is provided. The long-period held-out split is never
shown to you.

## Output (stdout)
A **single line**: one closed-form expression in the variables `x1`, `x2`, `x3`
that predicts `y`. You may use `+ - * / ** ( )`, numeric constants, and the
functions `exp, log, sin, cos, sqrt, tanh, abs`. No other names are allowed.

Example of the required FORM only (this is an illustrative shape, NOT the hidden
law and not expected to score well):
```
-2.0 * x1 + 0.5 * exp(x2) - 4.0
```

## Feasibility
The expression must parse under the whitelist above, reference only
`x1, x2, x3` and the allowed functions, and evaluate to a **finite** real number
at every held-out point. Any parse error, disallowed name/call, oversized input,
multi-line output, or `nan`/`inf` result scores `0.0`.

## Objective (minimize)
Let `heldout_MSE` be the mean squared error of your expression against the
long-period held-out magnitudes, and `complexity` the number of operator /
call / variable / constant nodes in your expression. The grader forms
```
F = heldout_MSE * (1 + LAMBDA * complexity)
```
with `LAMBDA = 0.004` and minimizes `F` (lower held-out error and simpler
expressions are better).

## Scoring
The grader builds an internal constant baseline `B = baseline_MSE * (1 + LAMBDA)`,
where `baseline_MSE` uses the training-mean magnitude on the held-out split, and
reports
```
Ratio = min(1000, 100 * B / F) / 1000
```
A constant prediction reproduces the baseline and scores about `0.1`. Recovering
the period-luminosity curvature, the color term, and the metallicity-dependent
slope drives the held-out error toward the irreducible photometric-noise floor
and raises the ratio, but that noise floor and the hidden metallicity coupling
keep it well below `1.0`. Higher is better.

## Constraints
- `test_id` fixes the hidden relation; larger ids add photometric noise and
  fewer training stars.
- Expression source at most 200000 bytes; output must be one line.
- Scoring is fully deterministic (all randomness is seeded).

## Example
For a submission `-2.0 * x1 + 0.5 * exp(x2) - 4.0`, the grader regenerates the
hidden relation and long-period held-out split for the given `test_id`,
evaluates the expression there, computes `heldout_MSE`, penalizes by the node
count, and prints e.g. `... Ratio: 0.34xxxx`.
