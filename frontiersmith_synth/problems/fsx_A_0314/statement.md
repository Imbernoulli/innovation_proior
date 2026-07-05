# Wide-Wake Turbine Siting: Maximize the Thinnest Wake Triangle

## Problem

You are siting the turbines of an offshore wind farm on a square parcel of open water,
the unit square field `[0,1] x [0,1]`. Every unordered TRIPLE of turbines forms a "wake
triangle". When three turbines are nearly collinear (a very thin triangle), a single wind
direction can line their wakes up so the downwind machines sit in the turbulent shadow of
the upwind ones — the worst case for annual energy yield. The robustness of a whole layout
against this failure mode is governed by its **thinnest** wake triangle: the triple of
turbines whose triangle has the smallest area.

Place all `n` turbines so that the **minimum triangle area over every triple** is as large
as possible. This is a fresh, wind-farm instance of the extremal Heilbronn-type
point-configuration problem: there is no known closed-form optimum, and many different
spatial strategies (rings, jittered lattices, incremental spacing, annealed relaxations)
are viable.

## Input (stdin)

```
n
xmin xmax ymin ymax
```

- Line 1: integer `n`, the number of turbines to place (`12 <= n <= 21`).
- Line 2: four reals giving the field bounds; always `0 1 0 1` (the unit square).

## Output (stdout)

Print exactly `n` lines, each `x y`: the coordinates of one turbine, in any order.
Coordinates must be finite and satisfy `0 <= x <= 1`, `0 <= y <= 1`.

```
x_1 y_1
x_2 y_2
...
x_n y_n
```

## Feasibility

An output is feasible only if it contains exactly `n` coordinate pairs, every value is
finite (no `nan`/`inf`), and every turbine lies inside the field `[0,1]^2` (a tolerance of
`1e-9` is allowed on the boundary). Any violation scores `Ratio: 0.0`.

## Objective (maximize)

Let the turbines be `p_1..p_n`. For a triple `(i,j,k)` the wake-triangle area is
`A(i,j,k) = 0.5 * |(p_j - p_i) x (p_k - p_i)|`. Your layout's quality is

```
F = min over all triples (i<j<k) of A(i,j,k).
```

Larger `F` is better. Coincident or collinear turbines drive `F` to 0.

## Scoring

The checker builds an internal baseline layout `B` = `n` turbines naively clustered on a
small central ring (radius `0.26`, centred at the field centre) and computes that layout's
thinnest wake triangle. Your score is

```
sc    = min(1000, 100 * F / max(1e-9, B_area))
Ratio = sc / 1000        # printed as "Ratio: <value>"
```

Reproducing the ring baseline scores about `0.1`; a layout with a ten-times larger thinnest
triangle caps the score at `1.0`.

## Constraints

- `12 <= n <= 21`.
- Field is the unit square. Time limit 5 s, memory 512 MB.
- Scoring is exact and deterministic (fixed `1e-9` geometric tolerance).

## Example (worked score)

Suppose `n = 3` (illustrative FORM only). Output the three corners
`(0,0), (1,0), (0,1)`. The single triangle has area `0.5`, so `F = 0.5`. If the ring
baseline's thinnest triangle were `B_area = 0.05`, then `sc = 100 * 0.5 / 0.05 = 1000`
and `Ratio = 1.0`. (The real instances use `n >= 12`, where no such easy optimum exists.)
