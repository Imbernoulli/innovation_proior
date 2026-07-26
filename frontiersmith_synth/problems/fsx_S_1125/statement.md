# Interior-Band Growth Chamber

## Problem
A growth chamber grows a simulated organism under a fixed **control
parameter** `p` (a nutrient index, `0 < p < 1`). At each integer time step `t`
the chamber logs the organism's **size** `S`, its aspect ratio `AR`, and its
branch-tip count `TC`. Size follows a fixed but **unknown** law: it saturates
toward a capacity as `t` grows, and both the capacity and the approach rate
depend on `p` through their own fixed (unknown) relationships. The three
descriptors, the exponents, and every constant of the law are never written
down — only the logged numbers are.

The catch: every logbook only ever runs the chamber at `p` inside a **narrow
interior band**, `p in [0.42, 0.58]`. In that band the size curve looks almost
smooth and simple in `p` — its true curvature is barely visible. The lab now
needs to predict size `S` for **extreme** nutrient levels, far outside that
band (near the low or high end of `(0,1)`). You must produce a closed-form
model of `S(t, p)` that still holds in that extreme regime, not one that
merely memorises the interior band.

Each test id is a different logbook (different exponents, different noise).

## Input (stdin)
```
line 1:            n_rows   test_id
next n_rows lines:  t  p  S  AR  TC
```
Each row is one logged measurement at time `t` under nutrient level `p`:
size `S`, aspect ratio `AR`, and branch-tip count `TC` (all recorded together;
only `S` is your prediction target — `AR`/`TC` are extra context that may help
you cross-check the shape of the growth curve, or may be ignored). `test_id`
is for reference only.

## Output (stdout)
A **single line**: a Python expression for `S` in the variables `t, p`.
Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals allowed. No other
names, attributes, calls, or imports.

Example output line (illustrative FORM only — **not** the hidden law):
```
0.5*sin(t) + 2*p - 0.1*t*p
```

## Feasibility
The output must be exactly one line, parse under the allowed grammar, and
evaluate to a finite real number on every held-out point. Anything else
scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out extreme-p split**
(nutrient levels well outside `[0.42, 0.58]`, crossed with the same time
range) plus irreducible measurement noise, then evaluates your expression
there. Let `heldout_MSE` be the mean squared error on `S` and `complexity`
the node count of your expression. With `LAMBDA = 0.0002`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train-mean S
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ~ 0.1`). Driving
held-out error toward the irreducible-noise floor raises the ratio, but that
floor keeps even a strong recovery below `1.0`. A flexible surface fit that
interpolates the interior band directly (without assuming a particular shape
in `p`) fits the logbook well but extrapolates its own curvature, not the
organism's — it diverges hard past the band. A model that instead recovers
the true saturating shape in `t` and the two power-law relationships in `p`
extrapolates correctly. Simpler expressions with the same error score higher
via the complexity term. The per-test score is `Ratio`; the final score
averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_rows = 50` (5 logged nutrient levels x 10 time
  steps).
- The true power-law curvature in `p` grows stronger, and measurement noise
  grows, as `test_id` increases.
- Expression output <= 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression gives `heldout_MSE = 0.40` with
`complexity = 50`, while the constant baseline has `baseline_MSE = 4.00`.
Then `F = 0.40*(1+0.0002*50) = 0.404`, `B = 4.00*(1+0.0002) = 4.0008`,
`Ratio = min(1000, 100*4.0008/0.404)/1000 = 0.990` — but realistic
extrapolation noise leaves a strong recovery well under that, and a constant
prediction gives `F ~= B`, `Ratio ~= 0.1`.
