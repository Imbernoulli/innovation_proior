# Polar Research Base — Reactor Load Law Discovery

## Title
Symbolic regression of a hidden 4-D poly-exp sensor law under extrapolation.

## Problem
The overwinter crew of an autonomous polar research base logs four normalized
telemetry channels from the reactor room:

- `x0` — coolant loop pressure deviation
- `x1` — turbine shaft torque
- `x2` — bus voltage ripple
- `x3` — outside-air thermal load index

The base controller predicts the reactor **load response** `y` from these four
channels through a single fixed closed-form law `y = F(x0, x1, x2, x3)`. The law
is a *polynomial-exponential* combination (sums of powers, pairwise products and
an exponential term). You are given a TRAINING log of `(x0, x1, x2, x3, y)` rows
sampled while the base ran in its **nominal** regime (all channels roughly in
`[-1, 1]`).

Your job is to recover a **closed-form expression** for `F` that will be used to
predict the load during the coming **storm regime**, when every channel is driven
into a *higher, previously unobserved* range. The grader therefore evaluates your
expression on a held-out **extrapolation** set drawn from that higher range — so
you must recover the true functional *form*, not memorize the training points.

## Input (stdin)
Plain text. Each line holds one training sample as five whitespace-separated
floating-point numbers:

```
x0 x1 x2 x3 y
```

There are between 100 and 280 rows (more rows / less noise on easier instances).
No header line. Read until EOF.

## Output (stdout)
A single line: a Python-syntax arithmetic **expression** in the variables
`x0`, `x1`, `x2`, `x3` that estimates `y`.

- Allowed operators: `+  -  *  /  **` and unary minus.
- Allowed function calls: `exp, log, sqrt, sin, cos, tan, tanh, abs, pow`.
- Numeric literals are allowed. No variable names other than `x0..x3`.
- No assignments, no other identifiers, no attribute access.

Example of the required *form* (this is an ILLUSTRATIVE shape only — it is NOT the
hidden law and shares nothing with it):

```
sin(x0) + x1/(1.0 + abs(x2)) - 0.3*x3
```

## Feasibility
An output is feasible iff it parses under the whitelist above and evaluates to a
**finite real number** at every held-out point. Any parse error, disallowed
token, or a `nan`/`inf`/exception at any evaluation point makes the submission
infeasible (score `0`).

## Objective
Minimize the held-out root-mean-squared error `E`, mildly inflated by an
expression-complexity factor:

```
E = RMSE_heldout * (1 + 0.002 * C)
```

where `C` is the number of nodes in your expression's syntax tree. The complexity
term rewards parsimonious laws over bloated over-fits.

## Scoring
Let `B` be the held-out error of the trivial constant predictor (the mean of the
training `y`), inflated the same way with `C = 1`. The reported ratio is

```
Ratio = min(1000, 100 * B / E) / 1000
```

so the constant-mean baseline scores ≈ `0.1` and you must be ~10× better to
saturate. The held-out set carries irreducible measurement noise, so a perfect
`1.0` is not attainable. The held-out region and its (seeded) noise are fixed
inside the grader; nothing about the hidden law, its coefficients, or the grader
seed appears in the training data.

## Constraints
- Deterministic scoring; grader runs in well under the time limit.
- `100 ≤ rows ≤ 280`; four input variables; single scalar output.

## Example (worked score)
Suppose the held-out constant-mean baseline error is `B = 4.2` and your expression
achieves `RMSE_heldout = 0.9` with `C = 45` nodes. Then
`E = 0.9 * (1 + 0.002*45) = 0.9 * 1.09 = 0.981` and
`Ratio = min(1000, 100 * 4.2 / 0.981)/1000 = min(1000, 428.1)/1000 = 0.428`.
