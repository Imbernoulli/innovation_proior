# Planetary-Ellipse Orbit Recovery from a Sky-Plane Point Cloud

## Problem
A small telescope has logged the sky-plane positions `(x, y)` of a single
planet as it moves along its orbit. By Kepler's first law the orbit is an
ellipse with the **Sun at a focus**, and the coordinate system is centred on the
Sun, so the **origin is that focus**. Every logged position therefore lies on
one hidden **conic**

```
F(x, y) = 0
```

(the orbital ellipse). The telescope was pointed elsewhere during one
contiguous stretch of the orbit, so a whole **arc** of the orbit is MISSING from
your training log. Recover a closed-form implicit relation `F(x, y)` whose zero
curve `F = 0` matches the orbit - and in particular extrapolates through the
never-observed arc.

## Input (stdin)
The first line contains two integers `n_train test_id`. Each of the next
`n_train` lines contains two floats:
```
x y
```
These are the noisy logged positions. The withheld-arc points are never shown.

## Output (stdout)
A **single line**: one closed-form expression in the variables `x` and `y`
describing the implicit curve `F(x, y) = 0`. You may use `+ - * / ** ( )`,
numeric constants, and the functions `exp, log, sin, cos, sqrt, tanh, abs`. No
other names are allowed. The scale of `F` does not matter (the curve is `F = 0`);
only the shape of its zero set is judged.

Example of the required FORM only (an illustrative shape, NOT the hidden law and
not expected to score well):
```
x**2 + 3.0*y**2 + 0.5*x*y - 2.0
```

## Feasibility
The expression must parse under the whitelist above, reference only `x, y` and
the allowed functions, and evaluate to a **finite** real number at every graded
point. Any parse error, disallowed name/call, oversized input, multi-line
output, `nan`/`inf` result, or a **numerically degenerate** curve (a gradient
`grad F` that vanishes, e.g. the trivial `F = 0`) scores `0.0`.

## Objective (minimize)
The recovered curve is judged on the **withheld arc** by the scale-invariant
first-order (Taubin) distance from each arc point `p` to the zero set:
```
d(p) = |F(p)| / || grad F(p) ||
```
which approximates the geometric distance from `p` to the curve `F = 0` and is
invariant to rescaling `F`. Let `arc_taubin` be the mean of `d(p)` over the
withheld-arc points and `complexity` the number of operator / call / variable /
constant nodes in your expression. The grader forms
```
F_obj = arc_taubin * (1 + LAMBDA * complexity)
```
with `LAMBDA = 0.010` and minimizes it (a tighter fit through the withheld arc
and a simpler expression are both better).

## Scoring
The grader builds an internal baseline `B` from its own trivial construction -
the best-fit **circle** through the training centroid - evaluated by the same
functional. It reports
```
Ratio = min(1000, 100 * B / F_obj) / 1000
```
A circle reproduces the baseline and scores about `0.1`. Recovering the
eccentric conic - and especially exploiting the focus-at-origin structure -
drives the withheld-arc distance toward the irreducible positional-noise floor
and raises the ratio, but that noise floor keeps it well below `1.0`. Higher is
better.

## Constraints
- `test_id` fixes the hidden orbit; larger ids add positional noise and log
  fewer points.
- Expression source at most 200000 bytes; output must be one line.
- Scoring is fully deterministic (all randomness is seeded).

## Example
For a submission `x**2 + 3.0*y**2 + 0.5*x*y - 2.0`, the grader regenerates the
hidden orbit and its withheld arc for the given `test_id`, measures the mean
Taubin distance of your zero curve to the arc points, penalizes by the node
count, compares against its circle baseline, and prints e.g.
`... Ratio: 0.43xxxx`.
