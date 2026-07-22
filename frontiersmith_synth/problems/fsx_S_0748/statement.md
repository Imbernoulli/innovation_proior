# Reef Cradle: Growing Coral to Hold a Lighthouse

## Problem
A lighthouse stands on a shallow reef. Its base rests on several **mounds** of the
seabed, each of which pushes down on the structure with a known force. Below, the
seabed floor is a grid of `W` columns (`0..W-1`) and `H` rows (`0..H-1`); row `0` is
solid bedrock (a set of fixed **anchor** columns). You may grow **coral** into any
other cell to try to hold the mounds still. You have a fixed budget of `M` coral
cells; coral is expensive, so you cannot fill the seabed solid.

Coral only transmits force between cells that share a **face** (up/down/left/right).
Two coral cells that merely touch at a corner do **not** interact at all — a diagonal
"shortcut" is structurally worthless.

To grade how still a candidate coral layout holds the mounds, a deterministic
**displacement relaxation** is run: every non-anchor coral cell has a
*displacement* value, all starting at `0`. Anchor cells are pinned at displacement
`0` forever. The relaxation runs for a **fixed** `K` iterations (never to full
convergence); each iteration, every non-anchor coral cell `i` with `d` face-adjacent
coral neighbours updates synchronously to `(sum of neighbours' current displacements
+ injected_force(i)) / d` (integer floor division on a fixed-point scale). A load
mound's own `f` value is its injected force. A coral cell with **no** face-adjacent
neighbour at all (e.g. reached only through a corner touch) has no averaging
reference and simply drifts upward every iteration — this is what makes
corner-only "connections" catastrophic. This mechanism rewards *hollow, branching,
tapering* coral — not solid fill — because parallel branches near a merge point
carry combined force with much less drift than a single thin, uniformly-thick path.

## Input (stdin)
```
W H
A
a_1 a_2 ... a_A
L
r_1 c_1 f_1
...
r_L c_L f_L
M
K
```
`a_1..a_A` are the anchor columns (row `0`). `r_i c_i f_i` are a load mound's row,
column and (positive integer) force. `M` is the coral-cell budget; `K` is the
fixed number of relaxation iterations.

## Output (stdout)
```
k
r_1 c_1
...
r_k c_k
```
`k` is the number of coral cells you place, then their `(row, col)` positions.

## Feasibility
- `0 <= k <= M`.
- Every cell has `1 <= row <= H-1` and `0 <= col <= W-1` (never row `0`).
- The `k` cells are pairwise distinct.
- Every load mound's `(r_i, c_i)` must be among your placed cells.
Any violation scores `Ratio: 0.0`.

## Objective
Minimize the **compliance** `F = sum over mounds of f_i * u_i`, where `u_i` is
mound `i`'s displacement after the `K`-iteration relaxation over the graph formed
by your coral cells plus the anchors.

## Scoring
The checker builds its own simple reference layout `B`: for each mound
independently, a straight coral column from its nearest anchor column up to the
mound's row, then straight over to the mound — every mound wired on its own, with
no sharing between mounds. `B` is that layout's compliance (always feasible,
independent of your budget). With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the reference layout scores `0.1`; cutting compliance to a tenth of it
caps the ratio at `1.0`.

## Constraints
- `1 <= A <= 2`, `1 <= L <= 3`, `10 <= W <= 30`, `8 <= H <= 18`.
- `M` is always large enough that at least one fully feasible layout exists.
- Time limit 5s, memory 512m.

## Example
`W=6 H=5`, one anchor at column `2`, one mound at `(4, 2, f=3)` (directly above the
anchor), `M=6`, `K=10`. The reference layout is the straight column
`(1,2),(2,2),(3,2),(4,2)`; reproducing exactly that scores `Ratio = 0.1`. Spending
the extra 2 cells of budget to widen that column (e.g. adding `(2,3)` and `(3,3)`
as a parallel face-adjacent branch) lowers the mound's displacement and scores
higher than `0.1`. (This worked example's shape is illustrative only — it is not
the family of mound/anchor layouts used to score submissions.)
