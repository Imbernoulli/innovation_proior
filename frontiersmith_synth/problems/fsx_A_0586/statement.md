# Empty Cones: Downwash-Aware Formation Morphing

A drone light show must morph a start figure **F0** of `N` drones into a target
figure **F1**. The `N` target points are an **unlabeled point cloud** — any drone
may fly to any target point, as long as every target is used exactly once. You choose
that matching *and* a descent schedule; a replay charges energy and **downwash**.

## Input (stdin)
```
N L S W K
Rbase Rslope_num Rslope_den Hmax Wpen Wmake
x y z            (N lines: F0, the start position of drone i, i = 0..N-1)
x y z            (N lines: F1, target point j, j = 0..N-1, in arbitrary order)
```
All integers. Both figures are `S` angular struts of `L` drones each (`N = S*L`);
along a strut the drones sit at distinct radii and heights. F1 uses the same struts
with the heights re-stacked, so the figure inverts.

## Output (stdout)
`N` lines, one per drone `i` in input order:
```
t_i w_i
```
`t_i` is the target point drone `i` flies to; `(t_0,…,t_{N-1})` **must be a
permutation** of `0..N-1`. `w_i` in `0..W-1` is the wave in which drone `i` moves.

## Replay & feasibility
The used wave slots are compacted to `0..u-1` in ascending order (`u` = number of
**distinct** waves used). Wave `w` owns ticks `[w*K, (w+1)*K)`; during its own wave a
drone slides from its start to its target in `K` equal integer sub-steps, waits at the
start before its wave, and rests at the target after. Any non-permutation, an
out-of-range wave, or a non-integer / non-finite token scores **0**.

## Objective (minimize)
`cost = energy + Wpen*downwash + Wmake*u`, where
- **energy** `= Σ_i |F0[i] − F1[t_i]|²` (squared straight-line travel);
- **downwash** = number of ordered drone pairs `(a,b)` over all replay ticks such that
  `a` is strictly above `b` and `b` lies in `a`'s **truncated downwash cone**:
  `z_a − z_b ≤ Hmax` and `horizontal_dist(a,b)² ≤ radius²` with
  `radius = Rbase + Rslope_num*(z_a − z_b) / Rslope_den`;
- **u** = distinct waves used (each wave lengthens the descent, so makespan is paid).

## Scoring
The checker builds baseline `B` = cost of the identity matching flown in one wave, then
`Ratio = min(1000, 100*B / cost) / 1000`. Matching the baseline scores `0.1`; a
10×-cheaper morph caps at `1.0`.

## The tension
Minimum-distance matching is the obvious move, but the struts are **tall and thin**
(`dH ≫ dR`): the cheapest match keeps each drone's height and slides it **radially** to
the mirror radius — so two drones on a strut swap radii and **cross at the same (x,y)
column at different heights**, piling into each other's cones. Spending the free
permutation the other way — keeping every drone on its own strut radius and moving it
**vertically** — makes the moves longer but leaves the cones empty; descending the
struts **layer by layer** phases the motion so the sloped cones never overlap. Balancing
the longer travel against the downwash and makespan is the problem.

## Constraints
`12 ≤ N ≤ 40`, `2 ≤ W ≤ 5`, `K = 6`. Time limit 5 s, memory 512 MB. Deterministic
scoring. Coordinates fit in 32-bit integers.

## Example (illustrative, not to scale)
With `N=4` on two struts, flying the identity match in one wave might cost `B`; a
matching that keeps each drone on its own radius and descends the top layer first can
cost `B/3`, scoring about `0.3`. The exact cone and weight constants are in the input —
read them and exploit them.
