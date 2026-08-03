# Fresh-Yield Law of an Artisan-Bakery Supply Line

## Problem

An artisan bakery runs a supply line whose **next-day sellable-loaf yield** is
governed by a single hidden, smooth **closed-form law** of four normalised
supply-chain readings logged each production day:

- `x1` = flour-silo inventory level
- `x2` = proofing-room temperature
- `x3` = forecast order volume
- `x4` = ingredient delivery lead-time (staleness of the incoming stock)

You are given a noisy sample of production days, each a row `x1 x2 x3 x4 y`,
where `y` is the observed normalised fresh-yield index. The line manager only
ever samples inside the **safe operating core** `x_i in [0,1]`. Your job is to
**recover a closed-form expression** `f(x1,x2,x3,x4)` that predicts the yield —
and, crucially, keeps predicting it out on the **over-range frontier** the line
has not yet been run at.

This is a **generalisation** task, not a curve-memorisation one: the grader
scores your expression on a held-out **extrapolation** split drawn from a
*different* input region than your training rows (larger demand, longer
lead-time, hotter room), regenerated deterministically inside the grader. It is
never shown to you.

## Input (stdin)

The first line is `n_train test_id`. The next `n_train` lines each contain five
floats: `x1 x2 x3 x4 y`, all training readings with `x_i in [0,1]`.

## Output (stdout)

A **single line**: a closed-form Python expression string in the variables
`x1, x2, x3, x4`. Allowed operators: `+ - * / ** %` and unary `-`. Allowed
function calls: `exp, log, sin, cos, sqrt, tanh, abs`. No other names, no
attributes, no imports, no assignments.

Example (illustrative FORM only — **NOT** the hidden law; the true family must
be discovered from the data):

```
0.3 + 1.2 * x1 * x4 - 0.5 * sin(x2) + 0.9 * sqrt(x3)
```

## Feasibility

The output must be a single parseable expression over the allowed
names/operators that evaluates to a **finite real number** at every grading
point. Empty output, multiple lines, disallowed names/calls, parse errors, or
any `nan`/`inf`/non-finite evaluation score **0**.

## Objective

Maximise a **complexity-penalised held-out coefficient of determination**
(`R^2`) on the frontier extrapolation split. Simpler expressions that
generalise beat both trivial constants and over-parameterised memorisers.

## Scoring

Let the grader regenerate the hidden law and the held-out frontier split
(`x_i in [1, 1.4]`, with irreducible noise). With `SS_tot` fixed by the split:

```
SS_res = sum( (y_held - f(x_held))^2 )
R2     = 1 - SS_res / SS_tot
complexity = number of expression nodes (operators, calls, names, constants)
F  = SS_res  * (1 + LAMBDA * complexity)      # LAMBDA = 0.003
B  = SS_base * (1 + LAMBDA * 1)               # SS_base = constant train-mean predictor
Ratio = min(1000, 100 * B / F) / 1000
```

A constant predictor reproduces the baseline `B` and scores `~0.1`. Recovering
the hidden demand/staleness envelope drives held-out `R^2` up and lifts the
ratio, but the irreducible frontier noise plus the hidden exponential
staleness-decay rate keep even a strong recovery **well below 1.0** — there is
no reachable perfect score.

## Constraints

- `1 <= test_id <= 10000`; the difficulty ladder `test_id = 1..10` adds logging
  noise and removes sampled days.
- Expression at most 200000 bytes, a single line.
- Deterministic scoring: the hidden law, train sample and held-out frontier are
  all functions of `test_id` only.

## Example (worked score)

Suppose on some test the constant train-mean predictor yields `SS_base = 900`
on the frontier and your expression yields `SS_res = 300` with `complexity = 20`.
Then `B = 900*(1+0.003) = 902.7`, `F = 300*(1+0.06) = 318.0`, and
`Ratio = min(1000, 100*902.7/318.0)/1000 = min(1000, 283.9)/1000 = 0.2839`.
Halving your residual again would roughly double the ratio, until the noise
floor caps it.
