# Adaptive Street-Lighting Power Law

## Problem
A smart-city grid runs thousands of adaptive LED luminaires. For one lamp model,
the controller's measured power draw `y` (watts, normalised) is governed by a
fixed but **unknown** closed-form law over four normalised sensor readings:

- `x1` — ambient-darkness index
- `x2` — pedestrian / traffic pressure
- `x3` — wind exposure
- `x4` — fixture ageing factor

During daytime calibration you can only sample the **normal operating region**
`x_i ∈ [0, 1]`, and every measurement is corrupted by sensor noise. The grid
operator needs a compact analytic model that still predicts power draw correctly
in the **heavy-load evening region**, where the readings push **beyond** the
calibration box. Your job is to recover a closed-form expression for `y` that
**generalises to that extrapolation region**, not one that merely memorises the
noisy training points.

Each test id corresponds to a different lamp model (a different hidden law).

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
2.0 + 0.5*x1*x3 - sin(x2)
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a finite real number on every held-out point.
Anything else scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out extrapolation split** in a
heavier-load region (a different input box than the training data) plus
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
and the fact that the exponential rate is hidden keep even a strong recovery
below `1.0`. Simpler expressions with the same error are rewarded via the
complexity term. The per-test score is `Ratio`; the final score averages over
the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 65 and 200 (shrinks with difficulty).
- Noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 4.0` with
`complexity = 26`, while the constant baseline has `baseline_MSE = 40.0`. Then
`F = 4.0*(1+0.003*26) = 4.312`, `B = 40.0*1.003 = 40.12`,
`Ratio = min(1000, 100*40.12/4.312)/1000 = 0.930`. A plain constant would give
`F ≈ B` and `Ratio ≈ 0.1`.
