# Locator Beacons: One Anchor Layout, Five Viewpoint Regimes

## Problem
A site (a grid of open floor `.` and wall `#` cells) must be covered by a
single fixed deployment of `A` locator beacons ("anchors"). The site
contains several **regimes** of intended use: open plazas, a bent, narrow
connecting corridor, a perimeter strip, and the corridor's two mouths where
it meets the plazas. Each regime `k` is described by a list of representative
**target points** inside it. You choose the `A` anchor positions **once**;
the same layout must serve every regime.

For a target `t` and an anchor `a` with clear line-of-sight to `t` (a straight
grid segment through no wall cell), let `u = (cos th, sin th)` be the unit
bearing vector from `t` to `a`. Form `M = sum_i u_i u_i^T` over all anchors
visible from `t`. Since `trace(u u^T) = 1` for any unit vector, `trace(M)`
equals `n_vis`, the number of visible anchors, exactly. Define
```
GDOP(t) = sqrt(n_vis / det(M))     if n_vis >= 2 and det(M) >= 1e-6
GDOP(t) = 30.0 (PEN)               otherwise
```
`det(M)` measures **angular diversity**: it is large when the visible anchors
spread across many bearings and collapses toward 0 when they cluster near one
direction (or its opposite -- two anchors exactly ahead and behind a target
are just as collinear as one). More visible anchors alone never fixes a
narrow angular spread; only genuinely different bearings raise `det(M)`.

A regime's cost is the **mean** `GDOP` over its target list. The instance's
cost is the **max over regimes** of that mean -- the single anchor layout is
judged by its worst-served regime, not its average performance.

## Input (stdin)
```
W H A K T
<H lines, W chars each: the grid ('.' open, '#' wall)>
<repeated K times:>
  <scenario name>
  <T lines: r c   (an open target cell)>
```

## Output (stdout)
`A` anchors given as `r c` pairs (`2*A` whitespace-separated integers in
total; one anchor per line is conventional but any whitespace layout is
accepted) -- the integer grid coordinates of each anchor, in order.

## Feasibility
- Exactly `A` coordinate pairs, all finite integers.
- Every `(r, c)` inside `[0,H) x [0,W)` and on an open ('.') cell.
- All `A` positions pairwise distinct.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`F` = the max-over-regimes mean `GDOP`, computed exactly as above, over your
submitted anchor set.

## Scoring
The checker also builds its own internal baseline `B`: spread `A` anchors over
the open cells by iterated farthest-point sampling (maximize pairwise
Euclidean distance) -- a purely geometric spread with no notion of
line-of-sight or bearing at all. With your `F`,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so matching `B`'s quality scores about `0.1`; a smaller `F` scores higher
(capped at `1.0`). There is no known closed-form optimum: the corridor is
only two cells wide, so almost every straight stretch of it can only see
anchors near its own axis (both ends look the same to `det(M)`), while the
plazas are wide open and reward a totally different spread -- one fixed
layout must trade these off, and a construction tuned for raw spatial spread
routinely serves the plazas well while leaving the corridor's targets
`GDOP`-blind.

## Example (illustrative FORM only -- not the scored geometry)
For a target `t` with two visible anchors at bearings `0 deg` and `90 deg`:
`M = [[1,0],[0,1]]`, `det(M)=1`, `n_vis=2`, so `GDOP = sqrt(2)`. If instead
both anchors are seen at bearings `0 deg` and `180 deg` (opposite along one
line): `u_2 = -u_1`, so `u_2 u_2^T = u_1 u_1^T` and `M` stays **rank 1**:
`det(M) = 0` exactly, regardless of how many more anchors join that same
line -- `GDOP = PEN`.

## Constraints
`21 <= W <= 30`, `12 <= H <= 17`, `A = 12` anchors, `K = 5` regimes, `T = 7`
targets per regime. Grid cells are `O(W*H)`; scoring is `O(A * (K*T))` per
line-of-sight check, comfortably inside the time limit.
