# Buried Artifact-Density Law

## Problem
An archaeological survey grids a large excavation into thousands of cells. For
one dig site, the recovered **artifact density** `y` (finds per cubic metre,
normalised) is governed by a fixed but **unknown** closed-form law over four
normalised survey readings taken at each cell:

- `x1` — stratigraphic depth index
- `x2` — soil phosphate concentration
- `x3` — magnetometer gradient
- `x4` — distance to a buried watercourse

You have only fully excavated the **well-dug core of the trench**, so you can
sample readings in the normal operating region `x_i ∈ [0, 1]`, and every reading
carries instrument + counting noise. The field director needs a compact analytic
model that still predicts artifact density correctly in the **unexcavated
frontier region**, where depth, phosphate, magnetometry and distance all push
**beyond** the sampled box. Your job is to recover a closed-form expression for
`y` that **generalises to that extrapolation region**, not one that merely
memorises the noisy training cells.

Each test id corresponds to a different dig site (a different hidden law).

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: x1 x2 x3 x4 y      (space-separated floats)
```
`test_id` is provided for reference only; the law must be inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `y` in the variables
`x1, x2, x3, x4`. Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
1.5 + 0.7*x2*x4 - cos(x1)
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a finite real number on every held-out point.
Anything else scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out frontier split** in the
deeper / farther region (a different input box than the training data) plus
irreducible measurement noise, then evaluates your expression there. Let
`heldout_MSE` be the mean squared error and `complexity` the node count of your
expression. With `LAMBDA = 0.003`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). Reducing held-out
error toward the irreducible-noise floor raises the ratio, but the noise floor
and the fact that the exponential (magnetometer) rate is hidden keep even a
strong recovery below `1.0`. Simpler expressions with the same error are
rewarded via the complexity term. The per-test score is `Ratio`; the final score
averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 76 and 220 (shrinks with difficulty).
- Noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 3.0` with
`complexity = 24`, while the constant baseline has `baseline_MSE = 33.0`. Then
`F = 3.0*(1+0.003*24) = 3.216`, `B = 33.0*1.003 = 33.099`,
`Ratio = min(1000, 100*33.099/3.216)/1000 = 1.000` capped, but with realistic
frontier noise the strong model lands well under that. A plain constant gives
`F ≈ B` and `Ratio ≈ 0.1`.
