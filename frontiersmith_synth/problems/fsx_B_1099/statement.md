# Vine on the Wall: Multi-Tropism Maze Climb

## Problem
A vine seedling roots at the bottom of a walled `R x C` grid and grows one
cell per step, chasing the sun. `#` cells are opaque, impassable wall
segments; every other cell is a candidate for growth. At each growth step
a tip's next cell is chosen by integrating **three tropisms**:

- **Gravitropism**: a constant pull straight "up" (toward larger row
  index).
- **Phototropism**: sum, over every light source that (a) has not yet been
  reached by the vine and (b) has an unobstructed straight-line path (no
  `#` strictly between) to the tip's current cell, the vector
  `brightness/(1+distance) * unit_vector(tip -> source)`. A source stops
  contributing the instant the vine occupies its own cell (it has been
  reached); an occluded source contributes nothing at all.
- **Thigmotropism** (touch): look at the tip's 8 neighboring cells. Sum
  the offsets of every one that is wall or off-grid into a "wall-mass"
  vector and rotate it 90 degrees clockwise; this is the tangent that
  hugs the wall surface. Zero if no neighbor is wall/off-grid.

The resultant `V = wg*gravi + wp*photo + wt*thigmo` is dotted against the
unit offset to each of the (up to) 4 orthogonal neighbors (N, E, S, W)
that are in-bounds, non-wall, and not yet grown; the tip advances to
whichever maximizes the dot product (ties broken by the fixed priority
N, E, S, W). A tip with no such neighbor dies. Growth proceeds in rounds:
every currently-alive tip attempts one step per round, in creation order,
sharing one global step budget `STEPS`.

**Branching.** You may designate up to `K` grid cells as branch triggers.
The first time ANY tip's step lands on a designated cell (and the branch
budget is not exhausted), a **new tip is spawned there**: on its own first
move it takes the *second-best* scoring neighbor instead of the best one
(so it diverges from what the original tip does next), then behaves
normally. Branches do not cost extra step budget beyond the new tip's own
future moves, which still draw from the shared `STEPS`.

## Input (stdin)
```
R C
STEPS K
<R lines, the i-th describing row i (0 = ground); '.'=open, '#'=wall,
 exactly one 'S' = the root, in row 0>
M
<M lines, each "r c brightness" -- an integer-brightness light source cell>
```

## Output (stdout)
```
wg wp wt
Kp
<Kp lines, each "r c" -- a distinct branch-trigger cell, 0 <= Kp <= K>
```
`wg, wp, wt` are non-negative reals (not all zero).

## Feasibility
Checker prints `Ratio: 0.0` on any violation: a negative or non-finite
weight, all three weights zero, a weight exceeding 1000, `Kp` outside
`[0, K]`, a branch cell out of bounds / on a wall / duplicated, or trailing
garbage after the declared tokens.

## Objective
Run the deterministic growth simulation above with your `(wg,wp,wt)` and
branch set. For each source, let its *reach value* be the largest
`brightness/(1+distance)` achieved by ANY single cell the vine ever
occupies with unobstructed line-of-sight to it (0 if never reached).
Maximize
```
F = sum of reach values over every source
```
This rewards genuinely reaching toward each source, not padding the score
by wandering many mediocre cells near one already-approached source.
Growing into a dead end reachable by only one tip forfeits every source
that only a *different* path could have reached -- unless a branch is
spent to send a second tip down that path while the first keeps climbing.

## Scoring
The checker also runs its own internal reference (pure gravitropism,
`wg=1,wp=0,wt=0`, no branches) to get baseline `B > 0`, then reports
```
Ratio = min(1000, 100 * F / B) / 1000
```
so matching that blind climb scores about `0.1`; reaching more sources,
and reaching them more closely, scores higher.

## Constraints
- `9 <= R <= 45`, `9 <= C <= 19`, `2 <= K <= 5`.
- `3 <= M <= 5` light sources, brightness in `[14, 60]`.
- Deterministic exact scoring; time limit covers the largest instance
  comfortably.

## Example
Just above the root, a one-cell-wide dead-end shaft climbs straight up,
capped by a solid ceiling; the real corridor turns sideways instead. A tip
using gravity alone always tries "up" first -- the shaft satisfies that
immediately, and once inside, the flanking walls make it a one-way trip: a
few cells gained, nothing more. A tip that also reads touch senses the
asymmetric wall mass right at the entrance and turns the other way before
ever committing to the shaft, reaching everything beyond it instead.
