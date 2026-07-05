# SkyGrid Swarm: Corner-Free Drone Deployment

## Problem
A delivery company parks its drone swarm on a square rooftop **landing grid** of
`m x m` pads. A pad is addressed by a coordinate `(r, c)` with `0 <= r < m` (row) and
`0 <= c < m` (column). Some pads are **obstructed** (antennae, vents) and cannot host a
drone.

When many drones fly simultaneously, a dangerous **relay resonance** appears whenever three
*distinct* active pads form an axis-aligned right-angle "corner": pads

```
(r, c),  (r + d, c),  (r, c + d)     for some integer d >= 1
```

i.e. one pad at the pivot, one directly below it, and one directly to its right, at the
**same offset** `d`. Any such triple destabilizes the swarm.

You choose which unobstructed pads to activate. The dispatcher wants to fly **as many drones
as possible** while never creating a single corner. This is exactly a **corner-free set**
(a.k.a. corner-free / cap-corner configuration) on the grid `[0,m) x [0,m)`.

## Input (stdin)
```
m
b
<b lines, each "r c" = an obstructed pad>
```
`m` is the grid side; `b` is the number of obstructed pads. All obstructed pads are distinct,
and none lies in row `0` or column `0` (so the two border lines are always fully free).

## Output (stdout)
```
k
<k lines, each "r c" = an activated pad>
```
Print the number of activated pads `k`, then the `k` addresses, one per line.

## Feasibility
An output is valid iff **all** of the following hold:
- each printed pad has integer coordinates with `0 <= r < m` and `0 <= c < m`;
- the `k` pads are pairwise distinct;
- no activated pad is obstructed;
- no three distinct activated pads `(r,c)`, `(r+d,c)`, `(r,c+d)` with `d >= 1` are all active
  (no corner).

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = k`, the number of activated drones (a corner-free set avoiding obstructed pads).

## Scoring
Let `B` be the size of the checker's own trivial construction: the largest **single line**
(one full row or one full column) restricted to unobstructed pads. A single line can never
contain a corner (a corner needs two distinct rows *and* two distinct columns), so it is
always valid; since row `0` and column `0` are never obstructed, `B = m`.

With maximization normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing a full border line scores `Ratio = 0.1`; a corner-free set `10x` larger caps
at `1.0`.

## Constraints
- `6 <= m <= 24` (small-scale; `36 <= m*m <= 576` pads).
- Row `0` and column `0` are always fully unobstructed (the baseline line is always available,
  and the pads `(0,0)`, `(1,0)`, `(0,1)` are always free).
- Time limit 5s, memory 256m.

## Example
Suppose `m = 6` with no obstructed pads. The full row `0` = `{(0,0),(0,1),...,(0,5)}` has
`B = 6` and is corner-free, scoring `0.1`. A corner-free set of size `F = 18` gives
`sc = 100 * 18 / 6 = 300`, `Ratio = 0.300`.
