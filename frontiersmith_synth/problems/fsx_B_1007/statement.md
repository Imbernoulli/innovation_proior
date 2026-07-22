# Frame Mill: Clamp-Margin Sequencing

## Problem
A rectangular billet occupies an `R x C` grid of cells. A permanent **frame**
of material (row `0`, row `R-1`, column `0`, column `C-1`) is the finished
product and must never be removed. Every interior cell (`1<=r<=R-2`,
`1<=c<=C-2`) is **waste** and must eventually be milled away, exactly once.

The billet is held by exactly `K` clamps. Each clamp occupies one cell of
*currently existing* material (frame or not-yet-removed waste); no two clamps
may ever share a cell. **The workpiece is simultaneously the product and the
fixture**: as material disappears, whatever remains is all you have left to
clamp onto.

**Stability.** Before you remove a waste cell `c`, compute the *margin*: the
signed distance from `c` to the convex hull of the `K` current clamp
positions -- positive if `c` is strictly inside the hull, negative (and
therefore unsafe) if `c` lies outside it. You may output two kinds of moves:
- `R r c` -- mill away cell `(r,c)`. It must currently be an un-removed waste
  cell, and no clamp may currently sit on it. This records that removal's
  margin (computed from the clamp positions *before* the removal).
- `M k r c` -- reposition clamp `k` (0-indexed) onto cell `(r,c)`. That cell
  must currently exist (frame, or waste not yet removed) and be free of any
  other clamp. Every reposition costs a fixed **refixture penalty**.

You must output an interleaved sequence of these two move types that removes
every waste cell exactly once, ending with all waste gone and only the frame
remaining.

## Input (stdin)
```
R C K PEN
<R lines of the grid, '.'=frame (permanent), '#'=waste>
K lines, each "r c" -- the starting position of clamp k (0-indexed)
```
`PEN` is the refixture penalty charged per `M` move, in the same units as
margin.

## Output (stdout)
```
T
<T lines, each "R r c" or "M k r c">
```

## Feasibility
Every violation below makes the checker print `Ratio: 0.0`:
- an `R` targets a cell that is not waste, already removed, or currently
  occupied by a clamp;
- an `M` targets an out-of-bounds cell, a cell that does not currently exist
  (frame is always fine; waste only if not yet removed), or a cell already
  occupied by a *different* clamp;
- any waste cell is left un-removed when the sequence ends.

## Objective
Let `m_1, ..., m_w` be the margins recorded at each of the `w` removals
(`w` = number of waste cells), and let `mv` be the total number of `M` moves
you performed. Maximize
```
F = ( min(m_1, ..., m_w) + 6.5 ) - PEN * mv
```
The `+6.5` is a fixed constant (same for every test case) that keeps the
score comparable across corridors of very different lengths; it does not
change which strategy is best. Larger minimum margin is better; every
reposition costs you `PEN`.

## Scoring
The checker also builds its own internal reference construction (a cheap,
always-feasible one that never plans ahead) to get a baseline value `B > 0`,
then reports
```
Ratio = min(1000, 100 * F / B) / 1000
```
so a construction matching that cheap reference scores about `0.1`, and one
that finds a far better margin/refixture trade-off scores higher.

## Constraints
- `3 <= R <= 6`, `8 <= C <= 56`, `K = 4` across the test ladder.
- `0.001 <= PEN <= 0.01`.
- Deterministic exact scoring; time limit covers even the largest corridor
  comfortably.

## Example
`R=3,C=8`: clamps start near the left edge. Removing the interior cell
`(1,1)` right away (clamps still at their starting bracket, which comfortably
contains column 1) records a solidly positive margin. Removing `(1,6)`
*without ever moving a clamp* would record a strongly negative margin,
because column 6 is far outside the starting bracket's hull -- exactly the
kind of removal a plan should either avoid until the clamps have moved, or
schedule after repositioning.
