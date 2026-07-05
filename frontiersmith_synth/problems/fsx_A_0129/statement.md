# Aquarium Floor Probe Placement (Minimum Star Discrepancy)

## Problem
You are plumbing a large aquarium. Its rectangular floor is modelled as the
unit square `[0,1]^2`. You must install **M** identical water-quality probe
fittings on the floor, one point each. To sample the water evenly, the
fittings must be spread so that *every* axis-aligned corner region anchored at
the intake corner `(0,0)` holds a fraction of the fittings close to its area.

The imbalance of a placement is its **star discrepancy**

```
D*(P) = sup over boxes B = [0, b1) x [0, b2), 0 <= b1,b2 <= 1  of
        | (#fittings inside B) / M  -  area(B) | .
```

`D*` measures the worst-case gap between "how many probes fall in a corner
region" and "how big that region is". You want the plumbing to be as uniform
as possible, so you **minimize** `D*`.

## Input (stdin)
One line: `d M`

* `d` — number of axes, always `2`.
* `M` — number of probe fittings you must place.

## Output (stdout)
Print exactly `M` lines (whitespace between numbers is flexible; the checker
reads exactly `2*M` numbers in row-major order). Line `i` gives the
coordinates of fitting `i`:

```
x_i y_i
```

with `0 <= x_i, y_i <= 1`.

## Feasibility
* Exactly `M` fittings (exactly `2*M` numbers) must be emitted.
* Every coordinate must be a finite real in `[0,1]` (values marginally outside
  are clamped; `nan`/`inf`, out-of-range, wrong count, or non-numeric tokens
  are rejected). Duplicate points are allowed.

Any violation scores `Ratio: 0.0`.

## Objective
Minimize the exact star discrepancy `D*(P)`. The supremum is attained on the
finite grid formed by the fittings' own coordinates, so the checker evaluates
it **exactly** — there is no sampling and no randomness in the score.

## Scoring
Let `F = D*(your placement)` and let `B = D*` of the checker's own baseline
(all fittings on the single vertical line `x = 1/2`). Because this is a
minimization:

```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```

Reproducing the baseline scores `Ratio ~ 0.1`; a placement with one-tenth the
baseline discrepancy caps at `Ratio = 1.0`. Optimal low-discrepancy sets for a
given `M` are **not known in closed form**, so there is no reachable optimum —
grids, Hammersley/Halton sets, Fibonacci lattices, and local optimization are
all viable and give different scores.

## Example
Instance:
```
2 4
```
A placement:
```
0.125 0.125
0.375 0.625
0.625 0.375
0.875 0.875
```
The checker computes `D*` of these 4 points exactly, computes `B` for the
`x=1/2` baseline, and prints `Ratio: 100*B/F / 1000`. (Illustrative only — this
4-point set is not optimal.)

## Constraints
* `d = 2`, `5 <= M <= 32` across the test ladder.
* Deterministic scoring; 5s / 512 MB per test.
