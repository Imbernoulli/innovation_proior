# Scrambled Sensor: Recover a Sparse Calibration Law Through a Hidden Affine Change of Variable

## Problem
A field sensor reports a raw reading `x`; the true physical quantity `y` is produced
by an **unknown internal calibration**. Inside the instrument the electronics first
apply a hidden affine transform to the reading,

```
u = a * x + b          (unknown gain a, unknown offset b)
```

and then a genuine physical response that is a **sparse polynomial** in `u`:

```
y = S(u) = sum_k  c_k * u^{e_k}      (only a few active powers e_k, unknown c_k)
```

You are handed the sensor's **commissioning log**: noisy `(x, y)` pairs recorded
entirely inside a **narrow calibration window** of raw readings. Inside that window
the curve looks like a bland low-order trend, so a naive fit "works" there and then
fails catastrophically once the sensor operates in the wider field range it was never
calibrated on. Your job is to recover a closed-form calibration law that
**extrapolates** to raw readings outside the window.

## Input (stdin)
```
N
x y        (row 1)
...        (N rows total, space-separated floats)
```
`N` grows with the difficulty ladder (tens to a few hundred rows). All rows lie in
one narrow window of `x`.

## Output (stdout)
A **single line**: one closed-form Python expression in the variable `x`.
Allowed tokens only:
- arithmetic: `+ - * / ** %`, parentheses, numeric literals;
- unary functions (each takes exactly one argument): `exp, log, sqrt, sin, cos, tanh, abs`;
- constants `pi`, `e`.

Any other name, attribute access, indexing, or multi-line output is rejected. The
expression may use at most 400 AST nodes.

## Feasibility
The expression must parse under the whitelist above and evaluate to a **finite** real
value on every held-out point. Anything else (empty, non-parseable, unknown symbol,
`nan`/`inf`, shape mismatch) scores `0`.

## Objective (minimize)
The checker regenerates a deterministic **held-out EXTRAPOLATION split** — raw
readings in a region **outside** the calibration window, with their own irreducible
measurement noise — evaluates your expression there, and forms the loss
```
F = RMSE_heldout + ALPHA * complexity      (ALPHA = 0.002, complexity = # AST nodes)
```
so held-out accuracy is traded against expression simplicity: there is no free lunch
for overfitting the window with an enormous formula.

## Scoring
Let `B` be the held-out RMSE of the internal constant predictor (`y = mean(train y)`).
```
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Reproducing the constant baseline gives `Ratio ≈ 0.1`. Recovering the calibration law
drives the held-out RMSE down and pushes the ratio up, but the irreducible held-out
noise and extrapolation-amplified coefficient error keep it well below `1.0`.

## Why it is hard
Expanded in the raw reading `x`, the true law is a **dense, high-degree** polynomial:
every power is active, so a direct low-order or dense fit on the narrow window is
either the wrong shape or badly ill-conditioned, and it extrapolates poorly. The
sparsity is only visible in the **hidden coordinate** `u = a*x + b`. Sparsity is
**invariant** under this affine change of variable, so the leverage is to search the
tiny 2-parameter `(a, b)` family, pull the data back to `u`, and test whether the
pulled-back data admits a few-term polynomial fit — a small algebraic search that
recovers a low-dimensional, well-conditioned model.

## Constraints
- `50 ≤ N ≤ 260`.
- Single-line expression, ≤ 5000 characters, ≤ 400 AST nodes.
- Deterministic scoring only; no wall-time or randomness in the score.

## Example (illustrative FORM only — NOT the hidden law)
A *valid-shaped* submission (unrelated shape, shown only for the output format and
allowed tokens):
```
0.7 * ( 3 * x + -1 ) ** 3 + 0.4 * ( 3 * x + -1 ) - 1.2
```
The actual gain, offset, active powers and coefficients are different and must be
discovered from the data.
