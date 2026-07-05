# Gravitational Lensing: Recovering a Radial Deflection Law from the Inner Band

## Problem
A cluster-scale gravitational lens bends light by an amount set by a hidden radial
force / deflection law `F(r)`, where `r` is the projected radius from the lens
centre (in arbitrary scaled units). Your telescope resolves multiple lensed images
only in the **inner strong-lensing band** (small `r`), where a **short-range
structural correction** to the smooth outer profile is non-negligible.

You are given noisy measurements `(r, F)` sampled **only in the inner band**. Your
job is to rediscover a closed-form expression for `F(r)` — a smooth power-law
profile plus its short-range correction — that **extrapolates** to the
**outer weak-lensing band** (large `r`), which is held out and never shown to you.

The grader scores your expression on that held-out outer band, so fitting the
inner-band noise is useless: only recovering the true functional *form* generalizes.

## Input (stdin)
Whitespace-separated data rows, two floats per line:
```
r F
r F
...
```
Each row is one noisy measurement of the deflection `F` at projected radius `r`.
All training radii lie in the inner band. Neither the hidden law, its coefficients,
nor the held-out band are provided.

## Output (stdout)
A single line: a closed-form expression for `F` as a function of the variable `r`,
written in Python syntax. Example of the **FORM ONLY** (this is an illustrative,
UNRELATED shape — NOT the hidden law):
```
2.0*sin(r) + 0.3*r**2 - 1.5
```
An optional leading `F =` or `y =` is allowed and stripped.

Allowed variable: `r`. Allowed operators: `+ - * / ** %` and unary sign.
Allowed functions: `exp, log, sqrt, sin, cos, tan, tanh, abs, pow`. Numeric
literals are allowed. Any other identifier makes the submission infeasible.

## Feasibility
The expression must parse under the whitelist above and evaluate to a **finite**
real number at every held-out radius. Any parse error, disallowed identifier, or
`nan`/`inf` prediction anywhere ⇒ score `0`.

## Objective (minimize)
Minimize the root-mean-square error of your expression against the true deflection
on the **held-out outer band**, lightly inflated by an expression-complexity
penalty. Lower held-out error ⇒ higher score.

## Scoring
Deterministic. The grader:
1. builds an internal **baseline** predictor `B` = the naive leading-only
   inverse-square fit `c/r**2` (least squares on your training band, short-range
   correction ignored);
2. regenerates the held-out outer band deterministically (fixed seed, fixed
   irreducible noise) from the private ground truth;
3. computes your held-out RMSE `E` (with the complexity penalty) and the baseline
   RMSE `B`, then reports
   ```
   Ratio: min(1000, 100 * B / E) / 1000
   ```
Reproducing the baseline scores ≈ `0.1`. Recovering the true form scores well above
that; the irreducible held-out noise keeps the maximum below saturation.

## Constraints
- Inner training band: `r ∈ [0.2, 0.6]`; held-out outer band: `r ∈ [1.5, 4.0]`.
- Training rows per instance: 40–148 (fewer, noisier for higher test ids).
- Expression length ≤ 4000 characters.

## Example (worked score)
This example uses an UNRELATED illustrative law purely to show the scoring
mechanics — it is NOT the hidden law and NOT its functional family. Imagine the
truth were `F = 5*exp(-r) + 0.4*sqrt(r)`. A submission matching only its coarse
scale reproduces the grader's baseline predictor, giving `E ≈ B` and
`Ratio ≈ 0.1`. A submission that recovers the true additive structure drives `E`
well below `B`, e.g. `B/E ≈ 5` ⇒ `Ratio ≈ 0.5`. A submission using a disallowed
identifier (e.g. `mass/r**2`) is infeasible ⇒ `Ratio = 0.0`. You must discover the
actual family — power law plus short-range correction — from the data alone.
