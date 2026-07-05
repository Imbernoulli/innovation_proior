# Fairground Anchors: Minimum-Discrepancy Carnival Ride Circuit

## Problem
A travelling carnival is laying out its ride circuit on a square fairground, modelled as the
unit cube `[0,1]^d`. You must choose the ground **anchor** positions for `M` rides. The circuit
electrical loop then connects the anchors, but for scoring only the *positions* matter: the
organiser wants the crowd spread as evenly as possible so that no rectangular slice of the
fairground is over- or under-served.

"Evenness" is measured by the **star discrepancy** of the anchor set. Your task is to place the
`M` anchors so that the star discrepancy is as small as possible.

This is genuinely open-ended: for almost all `(d, M)` the point set of minimum star discrepancy
is **unknown**. Classical constructions (lattices, Hammersley/Halton sequences) are strong but
not optimal, and local optimisation can beat them.

## Input (stdin)
A single line with two integers:
```
d M
```
`d` is the fairground dimension (2 or 3) and `M` is the number of ride anchors to place.

## Output (stdout)
`M` lines, each with `d` space-separated real numbers in `[0,1]` — the coordinates of one anchor.
Output exactly `M * d` numbers. All coordinates must be finite and lie in `[0,1]`.

## Feasibility
The output is rejected (score `0`) unless it contains exactly `M * d` finite real numbers, each
in `[0,1]`. Anchors may coincide; there is no minimum-separation requirement.

## Objective (minimize)
Let `P = {x_1, ..., x_M}` be your anchors. The **star discrepancy** is
```
D*(P) = sup over anchored boxes [0,c) of | (#points in the box)/M - vol([0,c)) |,
```
where `c` ranges over `[0,1]^d` and `vol([0,c)) = prod_k c_k`. The checker computes `D*(P)`
**exactly** by enumerating the finite grid of candidate box corners induced by your coordinates
(each corner uses both the open `<` count and the closed `<=` count). Smaller is better.

## Scoring
The checker builds an internal baseline `B` = the star discrepancy of the diagonal ride-line
`x_i = ((i+0.5)/M, ..., (i+0.5)/M)`. With `F = D*(P)` your submission's discrepancy, the score is
```
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Reproducing the diagonal baseline gives `Ratio ~= 0.1`; halving the discrepancy relative to the
baseline roughly doubles the ratio; a set ten times better than the baseline caps at `Ratio = 1.0`.

## Constraints
- `d in {2, 3}`, `8 <= M <= 40`.
- Deterministic scoring; no randomness, wall-time or hardware is involved.

## Example
For `d = 2, M = 4`, the diagonal baseline places anchors at
`(0.125,0.125), (0.375,0.375), (0.625,0.625), (0.875,0.875)`; these cluster along one line and
have a large star discrepancy (the box `[0,0.5)^2` contains 2/4 of the points but has volume 0.25,
already a local gap of 0.25). Spreading the four anchors to
`(0.125,0.625), (0.375,0.125), (0.625,0.875), (0.875,0.375)` lowers the discrepancy and raises the
ratio. The numbers above are illustrative only.
