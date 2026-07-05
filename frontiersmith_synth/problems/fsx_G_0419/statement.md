# Bioreactor Growth-Law Extrapolation: Predicting the Stationary Plateau

## Problem

You operate a fed-batch mammalian **cell-culture bioreactor**. An online probe logs the
**viable cell density (VCD)** `N` (in units of 1e6 viable cells / mL) every 2 hours during the
**early** part of the run — the lag phase, the exponential growth phase, and the very start of
deceleration (all at time `t <= 30` hours).

From this early series alone you must predict the **late-time saturation plateau**: as nutrients
deplete, the culture enters stationary phase and the density levels off. The late readings
(`t = 40 .. 90` hours) are **withheld** — the grader owns them and uses them to score you.

The true density follows a single, smooth **saturating growth law** drawn from the classic
sigmoidal family (logistic / Gompertz type): density rises from a small inoculum, accelerates,
then bends over to a carrying-capacity plateau. The exact law, its coefficients, and the
carrying capacity are **hidden**. You must **discover the functional form from the numbers** —
a fit that ignores saturation (e.g. a pure exponential) will over-shoot badly at late times.

Your submission is a single **closed-form expression in the variable `t`** that predicts `N(t)`.

## Input (stdin)

```
M
t_1 N_1
t_2 N_2
...
t_M N_M
```

The first line is the number of early samples `M`. Each following line gives a sample time
`t_i` (hours, integer) and the noisy measured density `N_i > 0`. Times are the fixed early grid
`0, 2, 4, ..., 30`.

## Output (stdout)

A single line: a Python expression string for the predicted density as a function of `t`.

- Allowed variable: `t`. Allowed constants: numeric literals, `pi`, `e`.
- Allowed operators: `+ - * / ** ` and unary `+ -`.
- Allowed functions: `exp, log, log10, sqrt, abs, pow, sin, cos, tanh`.
- No other names, attributes, comprehensions, or tokens. The strings `nan`, `inf`, `__` are
  forbidden. Expression length <= 5000 characters.

Example output line (**illustrative FORM only — this is NOT the hidden law**, just a legal
expression to show the syntax):

```
3.0 / (1.0 + t*t) + 0.5*sin(t)
```

## Feasibility

The expression must parse, use only the allowed tokens/variable, and evaluate to a **finite
real number** at every held-out time. Any violation (parse error, disallowed token, non-finite
or non-numeric result, empty output) scores **0**.

## Objective (minimize)

Let `F` be the root-mean-square error of your expression against the true densities on the
withheld late region `t = 40 .. 90`. Smaller held-out RMSE is better. A gentle penalty is
applied for very large expressions (to discourage memorising rather than modelling).

## Scoring

The grader regenerates the hidden law and the withheld late-time samples deterministically,
evaluates your expression there to get held-out RMSE `F` (times a mild complexity penalty), and
compares against an internal baseline `B` — a crude **saturating logistic** fit whose plateau is
fixed at `1.05 x max(train)`. The reported score is

```
Ratio = clamp( 100 * B / F , 0, 1000 ) / 1000
```

Reproducing the baseline scores about **0.10**. Recovering the correct saturating family and its
plateau drives the held-out error well below the baseline and raises the score; irreducible
measurement noise keeps it below 1.0. A non-saturating fit that blows up at late times scores
**below** the baseline.

## Constraints

- `M = 16` early samples per instance; times on the fixed grid `0,2,...,30` hours.
- Densities are positive with multiplicative measurement noise; the noise level and the hidden
  coefficients `(K, b, c)` vary by test instance.
- Deterministic scoring: no randomness, wall-time, or hardware is involved.

## Example (worked score)

Suppose your expression predicts the withheld late densities with RMSE `F = 2.0` (1e6 cells/mL)
after the complexity penalty, and the internal saturating-logistic baseline achieves `B = 6.0`
on the same withheld points. Then `Ratio = 100 * 6.0 / 2.0 / 1000 = 0.30`. Matching the baseline
(`F = B`) would instead give `Ratio = 0.10`.
