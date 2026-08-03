# Quantum Lab Wiring: Shielded Spool Placement

## Problem
The floor of a dilution-refrigerator wiring lab is modelled as the unit square container
`[0, S] x [0, S]`. A set of **fixed forbidden zones** — the central cryostat and several
instrument pillars — occupy circular regions you must route around. Zone `j` is the disk of
radius `g_j` centred at `(ox_j, oy_j)`.

You must position up to `N` circular **cable-spool pads**. A pad centred at `(x, y)` with
radius `r` bundles all coax that fans out within radius `r`. To prevent electromagnetic
cross-talk and mechanical clash, the pads must be **pairwise non-overlapping**, must lie
**entirely inside** the lab floor, and must **not intrude on any forbidden zone**. Larger pads
route more channels, and the commissioning reward scales with total routed reach, so you want to
make the **sum of the pad radii as large as possible**.

Output a placement of at most `N` pads that maximizes the sum of their radii.

## Input (stdin)
```
N S K
ox_1 oy_1 g_1
...
ox_K oy_K g_K
```
`N` = number of available spool pads, `S` = side length of the lab floor (`S = 1.0`),
`K` = number of forbidden zones, followed by `K` zone descriptors. Every forbidden zone lies
fully inside `[0.15, 0.85] x [0.15, 0.85]`, so the outer margin of the floor is always clear.

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of pads you deploy.
Then `M` lines, each `x y r` (floats): a pad centred at `(x, y)` with radius `r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite (no `nan`/`inf`).
- Containment: `x - r >= -tol`, `x + r <= S + tol`, `y - r >= -tol`, `y + r <= S + tol`.
- Pad non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.
- Zone clearance: for every pad `i` and zone `j`, `dist(center_i, zone_j) >= r_i + g_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the deployed pads.

## Scoring
Let `B` be the checker's internal trivial baseline: `N` equal pads laid in a single row along
the clear bottom margin, giving `B = S / 2` (independent of the forbidden zones). With `F` your
feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the bottom-row baseline scores about `0.1` and a placement ten times better caps at `1.0`.

## Constraints
`8 <= N <= 40`, `S = 1.0`, `2 <= K <= 7`. Every zone is fully contained in the central box
`[0.15, 0.85]^2` with `0.03 <= g_j <= 0.10`. Runs well under the time limit for these sizes.

## Example
Suppose `N = 8`, `S = 1.0`, with a single conceptual zone near the centre. Eight equal pads of
radius `0.125` placed at the cell centres of a 4x2 grid that stays clear of the zone can be
feasible with `F = 1.0`, giving `Ratio = min(1000, 100*1.0/0.5)/1000 = 0.2`. The bottom-row
baseline (eight pads of radius `1/16` along `y = 1/16`) gives `F = 0.5`, `Ratio = 0.1`. (These
numbers are illustrative of the scoring only — the actual instance has its zones supplied on
stdin.)
