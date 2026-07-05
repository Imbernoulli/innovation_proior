# Antenna Resonance — Extrapolating a Rational Transfer Function

## Problem
A receiving antenna is characterised on a network analyser. Around its fundamental
resonance the antenna behaves like a second-order band-pass network, so its measured
**power response** is a *rational transfer function* of the (normalised) frequency `f`:
a smooth peak near the resonant frequency with power-law roll-off skirts on either side.

You are given a noisy **in-band** frequency sweep (samples clustered around resonance).
You must output a single closed-form expression `P(f)` that predicts the response. You
will be graded on **out-of-band** frequencies — the low- and high-frequency roll-off
skirts that were never swept — so a model that merely interpolates the passband will
extrapolate badly. Recovering the true rational shape (its roll-off rate and gain) is
what generalises.

The hidden law, its parameters and the RNG seed are never revealed; you must infer the
functional family from the data.

## Input (stdin)
```
<n_train> <test_id>
f_1  P_1
f_2  P_2
 ...
f_n  P_n
```
The first line gives the number of training rows and the test id. Each subsequent row is
an in-band frequency `f` (in `[0.15, 2.20]`) and its noisy measured power response `P`.

## Output (stdout)
A **single line**: a closed-form expression in the one variable `f`, using
`+ - * / **`, parentheses, numeric constants, and the functions
`exp, log, sin, cos, sqrt, tanh, atan, abs`. No other names are allowed.

Example of the required OUTPUT FORMAT (this is an **illustrative form only — NOT the
hidden law**, whose shape you must discover from the data):
```
2.0 + 0.5*f - exp(-f)
```

## Feasibility
The expression must parse under the whitelist, use only the variable `f` and the allowed
functions, and evaluate to a **finite** real number at every graded frequency. Any parse
error, disallowed name, non-numeric result, division-by-zero, or `nan`/`inf` scores `0`.

## Objective (minimize)
Minimise the mean-squared error on the hidden **out-of-band** grading split, plus a small
complexity penalty on the expression size.

## Scoring
The grader regenerates the hidden antenna, the in-band train sample, and the out-of-band
extrapolation split (both roll-off skirts, with irreducible measurement noise) purely from
`test_id`. With held-out MSE `F_mse`, expression complexity `cx`, constant-baseline MSE
`B_mse`, and `LAMBDA = 0.003`:
```
F = F_mse * (1 + LAMBDA*cx)
B = B_mse * (1 + LAMBDA*1)
Ratio = min(1000, 100 * B / F) / 1000
```
Predicting the constant training mean reproduces the baseline (`Ratio ~ 0.1`). Driving the
out-of-band error toward the irreducible-noise floor raises the ratio; the hidden `(f0, Q)`
and that noise floor keep it below `1.0`. Ten test cases; the score is averaged.

## Constraints
- `1 <= test_id <= 10000`; `48 <= n_train <= 120`.
- Output at most 200000 bytes, exactly one non-empty line.
- Deterministic scoring: identical output always yields an identical score.

## Example (worked scoring)
Suppose on a case `B_mse = 16.0`. Emitting the constant mean gives `F_mse = 16.0`,
`cx = 1`, so `F = 16.0*(1.003)`, `B = 16.0*(1.003)`, `Ratio = 0.100`. A rational fit that
reduces held-out error to `F_mse = 2.5` with `cx = 20` gives `F = 2.5*(1.06) = 2.65`,
`B = 16.05`, `Ratio = min(1000, 100*16.05/2.65)/1000 = 0.605`.
