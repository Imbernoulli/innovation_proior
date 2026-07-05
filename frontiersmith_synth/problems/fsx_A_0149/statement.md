# Geothermal Field: Even Well Siting by Star Discrepancy

## Problem
A geothermal reservoir is surveyed as the unit square field `[0, 1]^2`. To characterize the
subsurface temperature and permeability without bias, you must site exactly `N` monitoring
wells so that they **sample the field as uniformly as possible**. A well pattern that clumps
in one corner or lines up along a fault leaves large unsampled sub-regions, so mission
geophysicists measure the *worst-case sampling imbalance* of your layout with the
**star discrepancy**.

For an axis-aligned test box `[0, q) = [0, q_1) x [0, q_2)` anchored at the origin, the
imbalance is the gap between the fraction of wells that fall inside the box and the box's area:

```
g(q) = ( #{ wells strictly inside [0, q) } / N )  -  q_1 * q_2
```

The **star discrepancy** of your layout is the largest such imbalance over every anchored box:

```
D*  =  sup over q in [0,1]^2  of  | g(q) |
```

A smaller `D*` means every anchored sub-region holds close to its fair share of wells.
Output a set of `N` well coordinates that makes `D*` as small as possible.

## Input (stdin)
One line with two integers `N` and `D`:
```
N D
```
`N` = number of wells to site, `D` = field dimension (always `2`).

## Output (stdout)
First line: an integer `M`, the number of wells you place; you must place all of them, so
`M = N`.
Then `M` lines, each `x y` (floats): the coordinates of one well in the field.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `M = N` exactly.
- Every coordinate is finite and lies in the field: `-tol <= x <= 1 + tol`, `-tol <= y <= 1 + tol`.

Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`F = D*`, the exact star discrepancy of your `N` wells. The checker computes `D*` exactly:
the supremum is attained on the finite grid of candidate boxes whose upper corner coordinates
are well coordinates (and `1`), so it evaluates the two one-sided imbalances
- over-count on closed boxes `#{ well <= q } / N - area(q)`, and
- under-count on open boxes `area(q) - #{ well < q } / N`

over that grid and takes the maximum.

## Scoring
Let `B` be the checker's internal trivial baseline: the star discrepancy of `N` wells placed
evenly along the field diagonal, `(k + 0.5)/N` in both coordinates. With `F` the star
discrepancy of your feasible layout,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
So reproducing the diagonal baseline scores about `0.1`, and a layout with ten times lower
discrepancy caps at `1.0`. Lower `D*` is better.

## Constraints
`8 <= N <= 56`, `D = 2`. The exact discrepancy is `O(N^3)`; well under the time limit here.

## Example
For `N = 4`, four wells at the cell centres of a 2x2 grid
`(0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75)` have star discrepancy `D* = 0.25`
(box `q = (0.5, 0.5)` holds one well: `1/4 - 0.25 = 0.0`; box `q = (1, 0.5)` holds two:
`2/4 - 0.5 = 0.0`; the extremum is a closed box at a well corner). Placing all four wells
on the diagonal instead gives the larger baseline discrepancy, scoring about `0.1`.
