# Aquarium Manifold: Maximum-Bore Pipe Packing

## Problem
You are plumbing the back panel of a rectangular aquarium sump of width `W` and height
`H`. Through this panel you must run `N` circular pipes (bulkhead fittings). Each pipe `i`
is a circle of center `(x_i, y_i)` and radius `r_i` (the pipe's outer bore radius). To keep
turbulence low and the panel structurally sound:

- every pipe circle must lie **entirely inside** the panel rectangle `[0,W] x [0,H]`;
- no two pipe circles may **overlap** (they may touch);
- every pipe radius is capped by the fitting stock on hand: `0 < r_i <= rmax`.

Total flow capacity grows with pipe bore, so you want the bores as large as possible.
**Maximize the sum of the pipe radii** `sum_i r_i`.

There is no known closed-form optimum for this packing; many layouts (regular grids,
staggered/hex offsets, corner-biased growth, iterative relaxation) trade off against each
other. Your job is to construct a good feasible layout.

## Input (stdin)
One line with four numbers:
```
N W H rmax
```
`N` is a positive integer (number of pipes). `W`, `H`, `rmax` are positive reals.

## Output (stdout)
Exactly `N` lines. Line `i` gives three reals:
```
x_i y_i r_i
```
the center and radius of pipe `i`. Order does not matter.

## Feasibility (tolerance 1e-6)
Let `tol = 1e-6`. A layout is feasible iff:
- there are exactly `N` output triples, all finite;
- for each `i`: `r_i > 0` and `r_i <= rmax + tol`;
- containment: `x_i - r_i >= -tol`, `x_i + r_i <= W + tol`, `y_i - r_i >= -tol`,
  `y_i + r_i <= H + tol`;
- non-overlap: for all `i < j`, `dist((x_i,y_i),(x_j,y_j)) >= r_i + r_j - tol`.

Any violation (wrong count, non-finite value, out-of-range radius, protrusion, overlap)
makes the whole layout infeasible and scores `0`.

## Objective
`F = sum_i r_i`, to be **maximized**.

## Scoring
The checker builds its own feasible baseline `B` (an equal-radius grid packing capped at
`rmax`). Your score is
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
So reproducing the baseline yields `Ratio ≈ 0.1`, and a layout with ten times the baseline
total radius caps at `Ratio = 1.0`.

## Constraints
- `10 <= N <= 40`, `1.0 <= W,H <= 2.0`, `0.1 <= rmax <= 0.5`.
- All arithmetic in the checker uses a fixed `1e-6` tolerance; scoring is deterministic.

## Example (illustrative)
Suppose `N=4, W=1, H=1, rmax=0.3`. Placing four circles centered at
`(0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75)` each with `r=0.245` is feasible
(each circle fits its quadrant and none overlap), giving `F = 4 * 0.245 = 0.98`. Growing the
circles until they touch their neighbors and the walls would raise `F` further — that search
for the tightest feasible bores is the problem.
