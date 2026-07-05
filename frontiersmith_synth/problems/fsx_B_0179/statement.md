# Clutch-Rating Law: Extrapolating a Hidden Poly-Exp Formula

## Problem

An e-sports analytics team scores every player's round with a hidden **clutch-rating**
law that maps four per-round features to a scalar:

- `x1` — aim precision
- `x2` — reaction delay
- `x3` — team synergy
- `x4` — economy pressure

You are given a **training sample** of rounds drawn from the *ordinary competitive
range* of every feature (each feature in `[0, 1]`), together with the observed
clutch-rating `y` (which carries measurement noise).

Your job is to **recover a closed-form expression** `f(x1, x2, x3, x4)` that predicts
the clutch-rating. You will be judged NOT on the training rounds but on a **held-out
set of extreme rounds** whose features lie in a *different, more intense region*
(`[1.0, 1.8]` per feature). So the law you propose must **generalize / extrapolate**,
not merely memorize the training points.

## Input (stdin)

```
N 4
x1 x2 x3 x4 y      (row 1)
...                (N rows total)
```

`N` training rows, each with the four features and the noisy clutch-rating `y`.
All values are floating point. The hidden law and its coefficients are **not** given.

## Output (stdout)

A **single line**: one closed-form expression string over the variables
`x1, x2, x3, x4`.

- Allowed operators: `+  -  *  /  **` and unary `-` / `+`.
- Allowed functions (one argument each): `exp, log, sqrt, sin, cos, tanh, abs`.
- Allowed names: only `x1, x2, x3, x4` and numeric literals. Any other name,
  attribute access, function call, or exponent that is not a bounded numeric
  constant makes the output **infeasible** (score 0).
- Powers must use a numeric constant exponent with magnitude `<= 6`.

Illustrative FORM only (this is **not** the hidden law — its shape is unrelated;
you must discover the real family from the data):

```
0.3 + 1.2 * x2 * sqrt( x4 ) - x1 / ( 1.0 + x3 )
```

## Feasibility

The expression must parse under the whitelist above and evaluate to a **finite**
real number at every held-out point (no `nan`, no `inf`, no domain errors such as
`log` of a non-positive value). Any violation scores `Ratio: 0.0`.

## Objective (maximize)

Let `Q` be the held-out fit quality with a parsimony penalty:

```
nmse   = mean_heldout( (f - y)^2 ) / var_heldout( y )
Q      = ( 1 / (1 + nmse) ) / ( 1 + 0.02 * complexity )
```

where `complexity` is the number of nodes in your parsed expression. The checker
regenerates the held-out extrapolation set deterministically and compares your `Q`
against its own trivial baseline `B` = predicting the constant training mean:

```
Ratio = min(1000, 100 * Q / B) / 1000
```

Reproducing the baseline scores about `0.1`; better-generalizing, parsimonious laws
score higher. Held-out noise and the complexity penalty keep a perfect score out of
reach, so the problem stays open-ended.

## Constraints

- `40 <= N <= 200`.
- Deterministic scoring only; the held-out region is disjoint from the training region.
- Expression length `<= 8000` characters.

## Example (worked score)

Suppose you submit the constant `3.8` (roughly the training mean). On the held-out
extreme rounds this is far off, giving a large `nmse`, so `Q ~= B` and `Ratio ~= 0.1`.
If instead you submit a poly-exp law that matches the true structure, `nmse` drops
sharply and `Ratio` climbs well above `0.1` (typically `0.3`–`0.6`), the remaining
gap being irreducible held-out noise plus the complexity penalty.
