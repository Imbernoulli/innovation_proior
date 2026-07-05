# Reservoir Gauge Network: Maximize the Smallest Sensing Triangle in a Triangular Catchment

## Problem
A dam authority manages a triangular catchment (modelled as the unit triangle with
vertices `(0,0)`, `(1,0)`, `(0,1)`) drained by a single reservoir. It must install `n`
telemetered gauge stations (water-level / sediment sensors) inside the catchment. Flood
reconstruction interpolates the catchment surface over triangles formed by triples of
gauges: any three gauges define a sensing triangle, and the reconstruction is only as
reliable as its *worst* (thinnest, smallest-area) triangle. Three nearly collinear gauges
give a degenerate cell and a hydrological blind spot.

Your job: place the `n` gauges so that the **minimum triangle area over every triple of
gauges is as large as possible**. This is a Heilbronn-type extremal placement problem
inside a triangle: there is no known simple optimum, and many different geometric
strategies compete.

## Input (stdin)
A single integer `n` (`n >= 3`), the number of gauge stations.

## Output (stdout)
`n` lines. Line `i` contains two space-separated real numbers `x_i y_i` — the coordinates
of gauge `i`. Exactly `2*n` numbers total must be printed.

## Feasibility
- Exactly `n` gauges (`2*n` real numbers) must be emitted.
- Every gauge must lie in the unit triangle: `x_i >= 0`, `y_i >= 0`, and
  `x_i + y_i <= 1` (tolerance `1e-6`).
- No two gauges may coincide.
Any violation scores `0`.

## Objective (maximize)
Let `F` be the minimum, over all `C(n,3)` triples of gauges, of the triangle area
`0.5 * |(B-A) x (C-A)|`. Larger `F` is better.

## Scoring
The checker builds an internal baseline `B`: the same `n` gauges spaced evenly around a
thin inscribed ellipse (x semi-axis `0.28`, y semi-axis `0.028`, centred at the triangle's
incentre). Your score is
```
Ratio = min(1.0, 0.1 * F / B)
```
so reproducing the thin-arc baseline scores about `0.1`, and reaching `10x` the baseline
min-area caps at `1.0`. Scoring is exact and deterministic (fixed geometric tolerance only).

## Constraints
`3 <= n <= 30` across the test ladder (anchor instance `n = 11`). Time limit 5s, memory 512m.

## Example
For `n = 3`, any triangle with positive area is feasible; the three corners
`0 0`, `1 0`, `0 1` give `F = 0.5`, far above the thin-ellipse baseline, so `Ratio`
caps at `1.0`.
