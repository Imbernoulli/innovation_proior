# Wind-Tunnel Multi-Regime Drag Law

## Problem
A wind-tunnel campaign calibrates the drag of a bluff-body model. The measured
drag coefficient `Cd` follows a fixed but **unknown** multi-regime closed-form
law over two inputs:

- `Re`  — Reynolds number of the flow (speed × chord / kinematic viscosity)
- `eps` — relative surface roughness of the model

Real drag laws stitch together several regimes: a viscous creeping-flow term, a
boundary-layer term, a bluff-body pressure-drag plateau, and a high-Reynolds
**drag-crisis** drop where the boundary layer trips turbulent and the wake
narrows. Your tunnel can only be run in its **mid-Reynolds calibration band**
(`Re` ≈ `10^1.5 .. 10^3.5`), and every balance reading carries transducer noise.
Operations needs a compact analytic law that still predicts drag in the
**high-Reynolds region** (`Re` up to `10^5`) that the tunnel cannot reach.
Recover a closed-form expression for `Cd` that **generalises to that high-Re
region**, not one that merely memorises the noisy mid-Re points.

Each test id corresponds to a different model geometry (a different hidden law).

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: Re eps Cd      (space-separated floats)
```
`test_id` is provided for reference only; the law must be inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `Cd` in the variables
`Re, eps`. Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
0.5 + 3.0/sqrt(Re) - 0.2*tanh(eps)
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a **finite real number** on every held-out point.
Anything else (empty, multi-line, unknown name, non-finite, oversized) scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **high-Re extrapolation split**
(a different Reynolds band than the training data) plus irreducible balance
noise, then evaluates your expression there. Let `heldout_MSE` be the mean
squared error and `complexity` the node count of your expression. With
`LAMBDA = 0.003`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). Capturing the
viscous `1/Re`, boundary-layer `1/sqrt(Re)` and plateau structure drives the
held-out error down and raises the ratio, but the high-Re drag-crisis drop and
the roughness coupling are only **partly visible** from the mid-Re band, and the
irreducible balance noise keeps even a strong recovery below `1.0`. Simpler
expressions with the same error are rewarded via the complexity term. The
per-test score is `Ratio`; the final score averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 76 and 220 (shrinks with difficulty).
- Measurement noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 0.030` with
`complexity = 12`, while the constant baseline has `baseline_MSE = 0.220`. Then
`F = 0.030*(1+0.003*12) = 0.03108`, `B = 0.220*1.003 = 0.22066`,
`Ratio = min(1000, 100*0.22066/0.03108)/1000 = 0.710`. A plain constant would
give `F ≈ B` and `Ratio ≈ 0.1`.
