# Apiary Landing-Pad Spread: Maximize the Faintest Foraging Triangle

## Problem

A beekeeper is laying out `n` numbered landing pads inside a single triangular
apiary field. Whenever three pads are visited in sequence, a scout bee traces the
triangle they form; the **faintest** such triangle (the one with the *smallest area*)
is the weakest link in the colony's navigation — bees confuse pads whose triangle is
too thin. You must place the pads so that even the faintest foraging triangle is as
large as possible.

The field is the closed unit triangle with corners `(0,0)`, `(1,0)`, `(0,1)`
(total area `1/2`). This is a Heilbronn-type extremal point-configuration problem:
there is no known closed-form optimum for these sizes, and many spreading strategies
compete.

## Input (stdin)

One line containing a single integer `n` — the number of landing pads to place
(`8 <= n <= 17`).

## Output (stdout)

Exactly `n` lines. Line `i` contains two real numbers `x_i y_i` (space separated),
the coordinates of pad `i`. Print full precision (e.g. `%.17g`). All pads must lie
inside the closed unit triangle:

```
x_i >= 0 ,  y_i >= 0 ,  x_i + y_i <= 1
```

(a tolerance of `1e-6` is allowed on each of these three inequalities).

## Feasibility

Your output is **feasible** iff it contains exactly `n` finite coordinate pairs and
every pad lies inside the triangle (within tolerance). Any violation — wrong count,
non-numeric / non-finite token, or a pad outside the field — scores `0`.

## Objective (maximize)

Let `A(i,j,k)` be the area of the triangle formed by pads `i, j, k`:

```
A(i,j,k) = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
```

Your raw score is the **minimum** triangle area over all `C(n,3)` triples:

```
F = min over all i<j<k of A(i,j,k)
```

Larger `F` is better. Three (nearly) collinear pads make `F` tiny, so avoid thin
triangles.

## Scoring

The checker builds an internal reference layout `B` (a fixed pseudo-random spread
of `n` pads in the field) and reports a normalized ratio

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

Reproducing the reference layout scores `Ratio = 0.1`; a layout whose faintest
triangle is `10x` the reference caps at `Ratio = 1.0`. An infeasible layout scores
`Ratio = 0.0`.

## Constraints

- `8 <= n <= 17`.
- Deterministic scoring; geometry compared with tolerance `1e-6`.
- Time limit 5 s, memory 512 MB.

## Example (worked score)

Suppose `n = 8` and the reference layout `B` has faintest-triangle area
`0.0100`. If your layout achieves `F = 0.0250`, then
`sc = 100 * 0.0250 / 0.0100 = 250`, so `Ratio = 0.250`. If instead your three
pads `0,1,2` are collinear, then `A(0,1,2)=0`, so `F = 0` and `Ratio = 0.0`.
