# Grid Efficiency Scaling Law: Extrapolate to Continental Scale

## Problem

A national power authority is coupling regional grids into one interconnection.
For a grid of **scale `x`** (the number of interconnected substations) they can
measure the **aggregate delivery efficiency** `eta(x)` — the fraction of
generated power that reaches load after transmission and switching losses.

Efficiency rises with scale (more parallel paths, better load balancing) but
obeys a smooth, saturating **scaling law** with an asymptotic ceiling. You are
given noisy efficiency measurements at a range of **small-to-medium** scales.
Your job is to discover the functional form of the law and fit it, so it
**extrapolates** to the far larger continental scales that cannot yet be built.

You must output a single **closed-form expression** in the variable `x`. The
grader evaluates your expression on a held-out set of **much larger scales**
(a region your training data never covers) and scores the extrapolation error.

## Input (stdin)

Several lines, each a training measurement:

```
<x> <eta>
```

`x` is the (positive) grid scale and `eta` the measured efficiency at that
scale. All training `x` lie in a small-to-medium band; the held-out grading
scales are strictly larger (genuine extrapolation).

## Output (stdout)

One line: a closed-form expression in the single variable `x`, e.g.

```
0.94 - 0.5*x**(-0.5)
```

Allowed tokens: the variable `x`; numeric constants; `+ - * / ** %`; unary `-`;
the constants `pi`, `e`; and the functions
`log, log10, exp, sqrt, sin, cos, tanh, abs, pow`. A leading `y =` is tolerated.

## Feasibility

The expression must parse under the whitelist above, evaluate to a **finite**
real number of moderate magnitude at every held-out scale, and use only the
allowed tokens. Any violation (unparseable, disallowed name/function,
`nan`/`inf`, or absurd magnitude) scores **0**.

## Objective

Minimize the root-mean-square extrapolation error of your expression on the
hidden held-out large-scale set, plus a small penalty on expression complexity.

## Scoring

Let `err` be the RMSE of your expression on the held-out scales and
`eff = err * (1 + 0.004 * complexity)` (complexity = size of your expression's
syntax tree). Let `B` be the RMSE of the internal **constant baseline** (predict
the training mean everywhere). Then

```
Ratio = min(1000, 100 * B / eff) / 1000
```

Reproducing the constant baseline scores about `0.1`; you must beat it by
extrapolating the true saturating shape. Irreducible measurement noise keeps a
perfect score out of reach.

## Constraints

- Training points per instance: 24. Held-out grading points: 14.
- Training scales lie in `[8, 250]`; held-out scales in `[400, 4000]`.
- Deterministic scoring; the held-out region and ground-truth law are fixed per
  instance and regenerated inside the grader.

## Example (worked score — illustrative FORM only, NOT the hidden law)

Suppose (for illustration) the hidden law were `eta = 3.0 + 1.2*exp(-x/50)`
(it is **not** — the real family is different; you must discover it from data).
If your submitted expression achieved held-out RMSE `err = 0.01` with a tree of
complexity `9`, then `eff = 0.01 * (1 + 0.004*9) = 0.01036`. If the constant
baseline had `B = 0.06`, then `Ratio = min(1000, 100*0.06/0.01036)/1000 =
min(1000, 579.2)/1000 = 0.5792`.
