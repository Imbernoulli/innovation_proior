# ISP Telemetry — Congestion-vs-Load Law Discovery

## Title
Symbolic regression of a hidden queueing-style congestion law, recovered from
low-load telemetry and graded under high-load extrapolation.

## Problem
A regional ISP instruments one backbone link and logs, per measurement window,
three normalized telemetry channels together with a measured **congestion
metric** (a mean normalized queueing-delay index):

- `rho` — link **utilization** (offered load / capacity), in `[0, 1)`.
- `cv`  — a **service-time variability** index (driven by the packet-size mix).
- `hop` — normalized **path length / propagation** share on the path.

The measured congestion `y` follows a single fixed closed-form law
`y = F(rho, cv, hop)`. You are given a TRAINING log of `(rho, cv, hop, y)` rows,
all sampled while the link ran in its **nominal, lightly-loaded** regime
(`rho` never exceeds `0.60`).

Capacity planners need to predict congestion during the coming **peak** period,
when utilization is driven far higher than anything the link has yet
experienced. The grader therefore evaluates your expression on a held-out
**high-load extrapolation** set with `rho` in a range strictly above the training
range. Because a congestion law typically has a saturation pole as `rho -> 1`,
that pole is nearly invisible at low load — you must recover the true functional
*form*, not memorize the low-load points.

## Input (stdin)
Plain text. Each line holds one training sample as four whitespace-separated
floating-point numbers:

```
rho cv hop y
```

There are between 120 and 354 rows (more rows / less noise on easier instances).
No header line. Read until EOF.

## Output (stdout)
A single line: a Python-syntax arithmetic **expression** in the variables
`rho`, `cv`, `hop` that estimates `y`.

- Allowed operators: `+  -  *  /  **` and unary minus.
- Allowed function calls: `exp, log, sqrt, sin, cos, tan, tanh, abs, pow`.
- Numeric literals are allowed. No variable names other than `rho`, `cv`, `hop`.
- No assignments, no other identifiers, no attribute access. An optional
  leading `y =` prefix is stripped.

Example of the required *form* (this is an ILLUSTRATIVE shape only — it is NOT
the hidden law and shares nothing with it):

```
sin(hop) + cv/(1.0 + abs(rho)) - 0.3*hop
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
saturate. The held-out high-load region carries irreducible measurement noise, so
a perfect `1.0` is not attainable. The held-out region and its (seeded) noise are
fixed inside the grader; nothing about the hidden law, its coefficients, or the
grader seed appears in the training data.

## Constraints
- Deterministic scoring; grader runs in well under the time limit.
- `120 ≤ rows ≤ 354`; three input variables; single scalar output.

## Example (worked score)
Suppose the held-out constant-mean baseline error is `B = 5.0` and your
expression achieves `RMSE_heldout = 1.8` with `C = 20` nodes. Then
`E = 1.8 * (1 + 0.002*20) = 1.8 * 1.04 = 1.872` and
`Ratio = min(1000, 100 * 5.0 / 1.872)/1000 = min(1000, 267.1)/1000 = 0.267`.
