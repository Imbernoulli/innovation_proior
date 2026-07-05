# Climate Sensor: Tipping-Point Regime-Break Law Discovery

## Problem
A remote climate field station logs a scalar sensor response `y` (a normalized
melt / heat-flux index) against four normalized environmental drivers:

- `x0` — temperature anomaly (the regime-controlling driver)
- `x1` — surface humidity index
- `x2` — incident-solar index
- `x3` — wind-shear index

The physical response is **piecewise**: below a hidden temperature threshold the
sensor tracks a mild, roughly linear law; once the anomaly crosses the threshold
an **accelerating feedback** (a "tipping" term) switches on and the response bends
upward. The threshold location, the functional form, and all coefficients are
hidden. Your job is to **discover a closed-form law from the training log** that
generalizes to the **far side of the break** — temperature anomalies well beyond
anything in the training data.

## Input (stdin)
Whitespace-separated training rows, five floats per line:
```
x0 x1 x2 x3 y
```
All training `x0` lie in the near/observed band (roughly `[-1, 1]`); the drivers
`x1,x2,x3` lie in `[-1,1]`. Larger test ids give fewer, noisier rows and reveal a
smaller slice of data past the break.

## Output (stdout)
A single closed-form expression in the variables `x0, x1, x2, x3`, written as a
Python expression on the first non-empty line (an optional `y = ` prefix is
allowed). Allowed operators: `+ - * / ** %` and unary minus. Allowed function
calls: `exp, log, sqrt, sin, cos, tan, tanh, abs, pow`. Numeric literals are
allowed. No variable other than `x0..x3`, no other names, no conditionals or
comparisons.

## Feasibility
The expression must parse under the whitelist above and evaluate to a finite real
number at every held-out point. Any parse error, disallowed token, or
non-finite / non-numeric value anywhere ⇒ score `0`.

## Objective (minimize)
Minimize the **held-out extrapolation error**. The grader regenerates a private
held-out split on the **far side of the regime break** (temperature anomalies
pushed well past the train edge), evaluates your expression there, and reports the
RMSE, lightly inflated by an expression-complexity penalty.

## Scoring
Let `E` be your complexity-inflated held-out RMSE and `B` the same quantity for a
constant-mean baseline predictor built by the grader. The reported ratio is
```
Ratio = min(1000, 100 * B / E) / 1000
```
so reproducing the baseline scores ≈ `0.1` and an ~10× error reduction caps at
`1.0`. Irreducible measurement noise on the far side keeps the maximum below `1`.

## Constraints
- Deterministic scoring; the held-out split is fixed by a private seed.
- Expression length ≤ 4000 characters.

## Example (illustrative FORM only — NOT the hidden law)
A submission for a *different, unrelated* problem might look like:
```
y = 0.3 + 1.2*x0 - 0.7*sin(x1) + 0.5*sqrt(abs(x2))
```
This shows the output syntax only. The true law here is piecewise with a hidden
break on `x0`; you must discover its shape from the data. If this expression were
scored, a possible line is:
```
rmse=1.842100 C=17 B=2.51 E=1.84 Ratio: 0.136000
```
