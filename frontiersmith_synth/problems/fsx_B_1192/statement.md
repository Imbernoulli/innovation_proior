# Calling the Knee: Leading-Indicator Battery-Aging Forecast

## Title
Forecast a battery cell's capacity across an aging "knee" you have never observed.

## Problem
A battery-testing lab cycles cells at various fixed operating conditions:
normalized cycling **temperature** and **depth-of-discharge (DoD)**. Every
cell's capacity fraction (1.0 = brand new) fades in two very different
regimes: a long **near-linear** decline, followed by an abrupt onset cycle
after which it **collapses** multiplicatively toward zero over a comparatively
short span. The onset cycle is not the same for every cell -- cells run
hotter and deeper age faster and hit their knee sooner -- and, critically,
the near-linear pre-knee capacity trend itself gives **no visible warning**
of when the knee will arrive. However, each cell's **internal-resistance
growth rate**, measured during the same early pre-knee window, is a noisy
but genuine early signal of the same underlying degradation state that
governs the knee's timing.

You are given a TRAINING log of cells, each described by its operating
conditions plus two early-window measurements -- resistance-growth rate and
capacity-fade rate -- and a target cycle count at which the lab wants a
capacity forecast. Every training row's target cycle is still pre-knee for
that cell (nothing in the log has aged past its own knee yet). Your job is
to forecast capacity at target cycles far beyond the training log -- a
horizon where many cells (especially the harsher-condition ones the grader
will test) HAVE already passed their knee.

## Input (stdin)
Plain text. Each line holds one training row as six whitespace-separated
floating-point numbers:

```
x0 x1 x2 x3 x4 y
```

`x0` = temperature, `x1` = depth-of-discharge, `x2` = resistance-growth rate,
`x3` = capacity-fade rate, `x4` = target cycle count, `y` = observed capacity
fraction. There are between 105 and 240 rows. No header line. Read until EOF.

## Output (stdout)
A single line: a Python-syntax arithmetic **expression** in the variables
`x0, x1, x2, x3, x4` that estimates `y`.

- Allowed operators: `+  -  *  /  **` and unary minus.
- Allowed function calls: `exp, log, sqrt, sin, cos, tan, tanh, abs, pow`.
- Numeric literals are allowed. No variable names other than `x0..x4`.
- No assignments, no other identifiers, no attribute access.

Example of the required *form* (this is an ILLUSTRATIVE shape only -- it is
NOT the hidden law and shares nothing with it):

```
0.9 - 0.1*x0 + tanh(0.05*(x2 - x4))*0.2
```

## Feasibility
An output is feasible iff it parses under the whitelist above and evaluates
to a **finite real number** at every held-out point. Any parse error,
disallowed token, or a `nan`/`inf`/exception at any evaluation point makes
the submission infeasible (score `0`).

## Objective
Maximize forecast accuracy on a held-out set, i.e. minimize the held-out
root-mean-squared error `E`, mildly inflated by an expression-complexity
factor:

```
E = RMSE_heldout * (1 + 0.0012 * C)
```

where `C` is the number of nodes in your expression's syntax tree.

## Scoring
Let `E_base` be the same complexity-adjusted held-out error of the trivial
constant predictor (the mean of the training `y`, `C = 1`). Convert both
errors to a log-accuracy score `F = max(0, -ln(E))` (smaller error -> larger
`F`), and report

```
Ratio = min(1000, 100 * F / F_base) / 1000
```

so the constant-mean baseline always scores ~0.1 by construction, and each
further multiplicative reduction of the held-out error keeps paying out
(diminishing, not flat) score, with no exact ceiling below 1.0. The held-out
cells run **hotter and deeper** than anything in training, and their target
cycles are drawn from a **forward horizon** beyond the training range -- so
most of them have, by evaluation time, already passed a knee never visible
in training. Held-out sampling and its (seeded) noise are fixed inside the
grader; nothing about the hidden law, its coefficients, or the grader seed
appears in the training data.

## Constraints
- Deterministic scoring; grader runs in well under the time limit.
- `105 <= rows <= 240`; five input variables; single scalar output.

## Example (worked score)
Suppose the held-out baseline error is `E_base = 0.72` (`F_base = -ln(0.72) =
0.329`) and your expression achieves `RMSE_heldout = 0.10` with `C = 60`
nodes, so `E = 0.10 * (1 + 0.0012*60) = 0.1072` and `F = -ln(0.1072) =
2.233`. Then `Ratio = min(1000, 100 * 2.233 / 0.329)/1000 = min(1000,
678.7)/1000 = 0.679`.
