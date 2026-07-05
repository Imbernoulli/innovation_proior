# Wind-Tunnel Sensor Scaling Law — extrapolating to large compute & resolution

## Problem
A pressure-tap sensor array in a wind tunnel is calibrated against a reference
flow field. Two knobs govern calibration quality:

- `x1 = C` — the **sampling-compute budget** (number of averaged flow snapshots),
- `x2 = R` — the **sensor-grid resolution** (effective tap count).

Raising either knob lowers the residual calibration error `L`, but only up to an
**irreducible instrument-noise floor** that no amount of compute or resolution can
beat. You are given noisy `L` measurements taken **only in the cheap calibration
regime** (small `C`, `R`). Your job is to discover the *functional form* of the
scaling law and fit it, so that it predicts `L` in a far more expensive regime
you were never shown.

## Input (stdin)
The first line is `n t` — the number of training rows `n` and the test id `t`.
Each of the next `n` lines has three floats:
```
C R L
```
the compute budget, resolution, and measured calibration error. `C` and `R` lie
in `[1, 25]` (log-uniform). The hidden law and its noise are not disclosed.

## Output (stdout)
A single line: a closed-form Python expression for `L` as a function of the
variables `x1` (=`C`) and `x2` (=`R`). Allowed operators: `+ - * / **`; allowed
functions: `exp, log, sqrt, sin, cos, tanh, abs, pow`. Numeric literals allowed.
No other names. Example of the required *format only* (this is an ILLUSTRATIVE
FORM, NOT the hidden law):
```
0.3 + 1.1*exp(-0.5*x1) + sin(x2)
```

## Feasibility
The expression must parse under the whitelist, use only `x1`,`x2` and the allowed
functions, and evaluate to a finite real number on every graded point. Any
violation, or a `nan`/`inf` result, scores 0.

## Objective (minimise held-out error)
The grader regenerates a **held-out EXTRAPOLATION split** in a much larger regime
`C, R in [25, 160]` (never shown in training), evaluates your expression there,
and computes the mean squared error against the true (noise-limited) values.

## Scoring
Let `F = heldout_MSE * (1 + LAMBDA*complexity)` and
`B = baseline_MSE * (1 + LAMBDA)`, where `baseline_MSE` is the error of the
constant "mean training `L`" predictor on the held-out split and `complexity`
counts expression nodes (`LAMBDA = 0.003`). The reported score is
```
Ratio = min(1000, 100 * B / F) / 1000
```
A constant predictor reproduces the baseline (`~0.1`). Fitting the true scaling
law — including the irreducible floor — drives held-out error down toward the
noise floor and raises the ratio, but that floor keeps it below `1.0`.

## Constraints
Deterministic scoring. `1 <= t <= 10`. Expression at most 200000 bytes, single
line.

## Example (worked score)
Suppose on some test `baseline_MSE = 0.50` and your expression achieves
`heldout_MSE = 0.05` with `complexity = 15`. Then `B = 0.50*(1.003) = 0.5015`,
`F = 0.05*(1 + 0.003*15) = 0.05225`, and
`Ratio = min(1000, 100*0.5015/0.05225)/1000 = min(1000, 959.8)/1000 = 0.9598`.
