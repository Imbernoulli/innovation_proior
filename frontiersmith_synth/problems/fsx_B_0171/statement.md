# Inverter Response Law: Symbolic Regression with Extrapolation Hold-out

## Problem
A utility-scale solar farm logs per-string telemetry from its inverters. Each row is a
noisy measurement of an AC-power surrogate `y` as a function of four normalized operating
variables (all equal to 1.0 at the nominal operating point):

- `x0` — plane-of-array irradiance
- `x1` — module temperature rise
- `x2` — DC bus voltage ratio
- `x3` — load / clipping fraction

There is a fixed but **hidden** closed-form "response law" `y = f(x0,x1,x2,x3)` (a mix of
polynomial and exponential terms). The commissioning logs you are given were recorded
entirely inside the *low* operating box (roughly `x_i ∈ [0,1]`). You must recover a
closed-form expression that **extrapolates** to the *high* operating region the plant only
reaches on peak-production days — a region you never see in training.

## Input (stdin)
```
N
x0 x1 x2 x3 y      (row 1)
...                (N rows total, space-separated floats)
```
`N` grows with the difficulty ladder (hundreds to ~1600 rows).

## Output (stdout)
A **single line**: one closed-form Python expression in the variables `x0,x1,x2,x3`.
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
The checker regenerates a deterministic **held-out EXTRAPOLATION split** in the high
operating region (a different input box than training, with its own irreducible
measurement noise), evaluates your expression there, and forms the loss
```
F = RMSE_heldout + ALPHA * complexity        (ALPHA = 0.004, complexity = # AST nodes)
```
so accuracy is traded against expression simplicity — there is no free lunch for
overfitting the training rows with a huge formula.

## Scoring
Let `B` be the held-out RMSE of the internal constant predictor (`y = mean(train y)`).
```
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Reproducing the constant baseline gives `Ratio ≈ 0.1`; recovering the true law drives the
held-out RMSE down and pushes the ratio up, but the irreducible held-out noise and the
complexity penalty keep it well below `1.0`.

## Constraints
- `430 ≤ N ≤ 1600`.
- Single-line expression, ≤ 5000 characters, ≤ 400 AST nodes.
- Deterministic scoring only; no wall-time or randomness in the score.

## Example (illustrative FORM only — NOT the hidden law)
Suppose (hypothetically) the data looked like a damped oscillation. A *valid-shaped*
submission would be:
```
1.3*x0*sin(2.0*x2) + 0.5*x1**2 - x3
```
This shows the required output format and allowed tokens. It is an **unrelated example
shape** — the actual response law is different and must be discovered from the data.
