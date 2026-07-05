# Tidal-Gauge Harmonic Recovery

## Problem
A coastal **tide gauge** records the sea-surface height `h` (metres, relative to
a fixed datum) once per hour. The true water level obeys a fixed but **unknown**
law that is **Fourier-sparse**: a mean sea level plus a *small* number of tidal
constituents,

```
h(t) = mu + sum_k  A_k * cos( omega_k * t + phi_k )
```

where `t` is time in hours. The angular frequencies `omega_k` lie in the
astronomical tidal band (semidiurnal near a 12-hour period, diurnal near
24 hours) but are perturbed **off** the exact astronomical values; the mean
level `mu`, the amplitudes `A_k`, the phases `phi_k`, the number of constituents,
and the exact frequencies are all **hidden** and differ from gauge to gauge.
Every reading carries instrument noise.

You are given only a **short recent window** of hourly readings. The harbour
authority needs a compact analytic tide model that predicts the **future** water
level — a later window of hours that lies **beyond** your observation window.
Recover a closed-form law in the single variable `t` that **generalises to that
future window**, not one that merely memorises the noisy readings you were given.

Each test id corresponds to a different gauge (a different hidden law).

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: t  h        (space-separated floats; hourly, t = 0,1,2,...)
```
`test_id` is provided for reference only; the law must be inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `h` in the variable `t`.
Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
1.3 + 0.6*cos(0.51*t + 2.0) - 0.2*sin(0.26*t)
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar in `t`, and evaluate to a finite real number on every held-out point.
Anything else scores `0`.

## Objective (minimise future-window error, complexity-penalised)
The grader deterministically regenerates a **future extrapolation window** (a
later, disjoint stretch of hours) from the same hidden law plus irreducible
instrument noise, then evaluates your expression there. Let `heldout_MSE` be the
mean squared error and `complexity` the node count of your expression. With
`LAMBDA = 0.002`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
Predicting a constant (the train mean) reproduces the baseline (`Ratio ≈ 0.1`).
Recovering the hidden constituents drives the future-window error down toward
the irreducible-noise floor and raises the ratio, but that noise floor and the
fact that a short window cannot perfectly resolve the off-grid frequencies keep
even a strong recovery below `1.0`. Simpler expressions with the same error are
rewarded via the complexity term. The per-test score is `Ratio`; the final score
averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 61 and 169 (the observation window
  shrinks with difficulty).
- The number of hidden constituents (2 to 4) and the instrument noise both grow
  with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 0.05` with
`complexity = 18`, while the constant baseline has `baseline_MSE = 0.50`. Then
`F = 0.05*(1+0.002*18) = 0.0518`, `B = 0.50*1.002 = 0.501`,
`Ratio = min(1000, 100*0.501/0.0518)/1000 = 0.967` (capped at `1.0`). A plain
constant would give `F ≈ B` and `Ratio ≈ 0.1`.
