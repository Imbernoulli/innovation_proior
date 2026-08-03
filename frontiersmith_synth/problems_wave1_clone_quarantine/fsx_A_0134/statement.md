# Wind Tunnel Sensor Spread: Maximize the Smallest Triangulation Cell

## Problem
A square wind-tunnel test section (modelled as the unit square `[0,1] x [0,1]`) must be
instrumented with `n` point pressure sensors. Downstream flow reconstruction triangulates the
measurements: any three sensors form a triangular sensing cell, and the reconstruction is only
as robust as its *worst* (thinnest, smallest-area) cell. Three nearly-collinear sensors give a
degenerate cell and a blind spot.

Your job: place the `n` sensors so that the **minimum triangle area over every triple of
sensors is as large as possible**. This is a Heilbronn-type extremal placement problem: there
is no known simple optimum, and many different geometric strategies compete.

## Input (stdin)
A single integer `n` (`n >= 3`), the number of sensors.

## Output (stdout)
`n` lines. Line `i` contains two space-separated real numbers `x_i y_i` — the coordinates of
sensor `i`. Exactly `2*n` numbers total must be printed.

## Feasibility
- Exactly `n` sensors (`2*n` real numbers) must be emitted.
- Every coordinate must lie in the unit square: `0 <= x_i, y_i <= 1` (tolerance `1e-6`).
- No two sensors may coincide.
Any violation scores `0`.

## Objective (maximize)
Let `F` be the minimum, over all `C(n,3)` triples of sensors, of the triangle area
`0.5 * |(B-A) x (C-A)|`. Larger `F` is better.

## Scoring
The checker builds an internal baseline `B`: the same `n` sensors placed evenly on a thin
inscribed ellipse (semi-axes `0.5` and `0.10`). Your score is
```
Ratio = min(1.0, 0.1 * F / B)
```
so reproducing the baseline scores about `0.1`, and reaching `10x` the baseline min-area caps
at `1.0`. Scoring is exact and deterministic (fixed geometric tolerance only).

## Constraints
`3 <= n <= 30` across the test ladder. Time limit 5s, memory 512m.

## Example
For `n = 3`, any triangle with positive area is feasible; the three corners
`0 0`, `1 0`, `0 1` give `F = 0.5`, far above the thin-ellipse baseline, so `Ratio` caps at `1.0`.
