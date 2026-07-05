# Rooftop-Garden Water-Stress Scaling Law

## Problem
A city urban-farming program is instrumenting a fleet of small pilot **rooftop garden beds**.
Each bed has a soil area `A` (square metres) and receives `W` litres of irrigation per day.
An overhead multispectral sensor reduces each bed to a single dimensionless **canopy
water-stress index** `L` (lower = a healthier, better-watered canopy). Adding either bed area
or daily water lowers stress, with diminishing returns, toward an irreducible floor set by the
rooftop's wind and heat exposure.

You are given a monitoring table of noisy `(A, W, L)` triples from the small pilot beds. Your
job is to **propose and fit a closed-form scaling law** `L ≈ f(A, W)` that will **extrapolate**
to a planned, much larger rooftop farm (larger `A` and larger `W` than anything measured). You
are scored purely on how well your law predicts stress in that unseen larger-resource region —
so you must recover the underlying functional shape, not memorise the training noise.

## Input (stdin)
```
N                      # number of training rows
A_1 W_1 L_1
A_2 W_2 L_2
...
A_N W_N L_N
```
`A_i`, `W_i` are positive integers; `L_i` is a positive real (full precision). The rows are the
TRAIN split only. The held-out extrapolation rows (larger A and W) are NOT given.

## Output (stdout)
A single line: a **closed-form Python expression** in the variables `A` and `W`, e.g. a sum of
powers. Allowed: `+ - * / **`, parentheses, numeric constants, the names `A`, `W`, `pi`, `e`,
and the functions `exp, log, log10, sqrt, abs, pow, sin, cos`. No assignments, no other names,
no `nan`/`inf`, no `__`. The expression must evaluate to a finite real for every held-out
`(A, W)`.

## Feasibility
The output must parse under the whitelist above and evaluate to a finite number on every
held-out point. Any violation (unparseable, disallowed token/name, non-finite result) scores 0.

## Objective (minimise)
Let `held_out_rmse` be the root-mean-square error of your expression against the true stress on
the hidden larger-rooftop grid. A gentle complexity penalty multiplies the error for very large
expressions (AST nodes above 40). Lower is better.

## Scoring
The checker regenerates the held-out extrapolation region deterministically from the same hidden
law that produced your training data, evaluates your expression there, and compares against an
internal baseline `B` (a single product power-law `k*A^c1*W^c2` fitted by log-linear regression
on the train rows):
```
Ratio = min(1.0, 0.1 * baseline_rmse / (held_out_rmse * penalty))
```
Reproducing the baseline shape scores ≈ 0.10; a law that extrapolates ~10× better than the
product power-law saturates at 1.0. Irreducible measurement noise keeps the ceiling below 1.0.

## Constraints
- `N = 48` training rows (8 area levels × 6 water levels).
- Deterministic scoring; expression length ≤ 5000 chars; evaluated only on `A`, `W`.
- Runtime well under the time limit (pure closed-form evaluation).

## Example (illustrative FORM only — NOT the hidden law)
Suppose (in a different, unrelated world) the true relationship were a saturating exponential
`L = 0.5 + 3.0 * exp(-0.2 * (A + W))`. A submission of `0.5 + 3.0*exp(-0.2*(A + W))` would then
be evaluated on the hidden larger-rooftop grid and scored by its extrapolation RMSE against the
product-power-law baseline. This example only shows the OUTPUT FORMAT; the actual hidden law in
this problem has a different shape that you must discover from the data.
