# The Alchemist's Limiting Reagent

## Problem
A guild alchemist brews a potion from four reagents — moonpetal dust (`q1`),
sulfur bloom (`q2`), quicksilver tears (`q3`) and ember root (`q4`), each
measured in grams. Every recipe has a fixed but **unknown** true stoichiometry:
each reagent `i` is consumed at a hidden small-integer rate `s_i`, and the
potion's yield is capped by whichever reagent runs out first (a *limiting
reagent* law), scaled by a hidden gain `g`. Each test id is a different recipe
(different hidden `s_i` and `g`).

You only get to watch the alchemist's **calibration brews**: mixtures poured
close to a *balanced* ratio, where no single reagent is ever drastically
scarcer than the others, and every yield reading is corrupted by measurement
noise. The guild then wants your formula to predict yield on **unbalanced
brews**, where one reagent is deliberately kept scarce relative to the rest —
a regime the calibration data barely visits. Recover a closed-form expression
that generalises there, not one that merely fits the calibration noise.

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: q1 q2 q3 q4 y      (space-separated floats, grams / yield units)
```
`test_id` is for reference only; the recipe must be inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `y` in the variables
`q1, q2, q3, q4`. Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs, min, max`; numeric literals allowed. No
other names, attributes, calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
0.5*q1 + 0.2*q2*q3 - sqrt(q4)
```

## Feasibility
The output must be exactly one line, parse under the allowed grammar, and
evaluate to a finite real number on every held-out point. Anything else
(empty output, disallowed syntax, non-finite result) scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out extrapolation split**:
mixtures where one reagent is pushed scarce relative to the others (unlike
the balanced calibration data), plus an irreducible noise floor, then
evaluates your expression there. Let `heldout_MSE` be the mean squared error
and `complexity` the node count of your expression. With `LAMBDA = 0.003`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). The true yield
law is `y = g * min(q1/s1, q2/s2, q3/s3, q4/s4)` for hidden small positive
integers `s_i` and a hidden gain `g` — but the calibration data alone gives
only weak, noisy evidence about each `s_i` individually. A model that fits the
calibration cluster with a smooth combination of the reagents (rather than a
true minimum) will match well there yet diverge sharply once one reagent
becomes genuinely limiting. Recovering the right *shape*, including snapping
ratio estimates to the correct small integers, drives held-out error toward
the irreducible-noise floor and raises the ratio — but the noise floor and
the coarseness of any finite integer search keep even a strong recovery below
`1.0`. The per-test score is `Ratio`; the final score averages over the
difficulty ladder (`test_id` 1..10, noise growing and sample count shrinking
with difficulty).

## Constraints
- `test_id` in `1..10`; `n_train` between 76 and 220 (shrinks with difficulty).
- All `q_i > 0`. Hidden `s_i` are integers in `1..6`; hidden `g` is a positive
  real.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose your expression yields `heldout_MSE = 2.85` with `complexity = 15`,
while the constant baseline has `baseline_MSE = 25.0`. Then
`F = 2.85*(1+0.003*15) = 2.978`, `B = 25.0*1.003 = 25.075`,
`Ratio = min(1000, 100*25.075/2.978)/1000 ≈ 0.842`. A plain constant gives
`F ≈ B` and `Ratio ≈ 0.1`; the noise floor and the coarseness of any finite
integer search keep even a strong recovery below `1.0`.
