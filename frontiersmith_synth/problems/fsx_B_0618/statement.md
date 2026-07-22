# Gemstone Sticker: Fold-Compaction Under Overlap Avoidance

## Problem

A gemstone's surface has been cut into `n` unit-square **facets**, numbered
`0..n-1`. Every facet has 4 **tabs**, labelled `N=0, E=1, S=2, W=3`. The
factory's cutting diagram lists `m` **fusion edges**: a fusion edge
`(u, su, v, sv)` means tab `su` of facet `u` is physically fused to tab `sv`
of facet `v` in the finished stone. Because `m` is usually larger than
`n-1`, the diagram is redundant — the same facets can be held together by
many different subsets of fusion edges.

You must choose which fusion edges stay **glued** (the rest are **cut**).
The kept edges must form a forest (no cycles): each tree in the forest is
one physical **piece** of the unfolded net; a facet touched by no kept edge
is its own singleton piece. Given the kept edges, each piece is laid out
flat on a sticker sheet by the following fixed rule (apply it once per
piece, independently):

- Directions `0,1,2,3` correspond to `(dx,dy) = (0,1),(1,0),(0,-1),(-1,0)`.
- Pick any facet of the piece as its root; place it at grid cell `(0,0)`
  with orientation `o=0`.
- Process the piece's kept edges outward (breadth/depth-first, either
  order gives the same final placement). If facet `u` is already placed at
  `(x,y)` with orientation `o`, and a kept edge fuses tab `su` of `u` to tab
  `sv` of unplaced facet `v`, let `g = (su + o) mod 4`. Place `v` at
  `(x + dx[g], y + dy[g])` with orientation `o' = (g + 2 - sv) mod 4`.

This rule is exact and deterministic — every facet of a connected piece ends
up at one specific integer grid cell.

## Input (stdin)

```
n m penalty
u_1 su_1 v_1 sv_1
...
u_m su_m v_m sv_m
```
`0 <= su_i, sv_i <= 3`, `0 <= u_i, v_i < n`, `1 <= n,m <= 400`.

## Output (stdout)

```
k
e_1 e_2 ... e_k
```
`k` is the number of fusion edges you keep glued; `e_1..e_k` are their
0-indexed positions in the input edge list (any order, all distinct). `k=0`
is legal (cut everything).

## Feasibility

1. `0 <= k <= m`, all `e_i` distinct and in range.
2. The kept edges must form a forest (no cycles).
3. **Self-overlap-avoidance**: within a single piece, no two facets may be
   placed at the same grid cell. (Different pieces are separate physical
   stickers and never interact.)

Any violation scores `Ratio: 0.0` for that test.

## Objective (minimize)

For each piece, its **bounding-rectangle area** = `(max_x-min_x+1) *
(max_y-min_y+1)` over its own facets. Let `P` be the number of pieces. The
cost is
```
F = sum of all pieces' bounding-rectangle areas + penalty * (P - 1)
```
Lower `F` is better: fewer, tighter, non-overlapping stickers.

## Scoring

The checker also computes its own always-feasible baseline `B = n + penalty
* (n-1)` (cut every fusion edge — `n` singleton 1x1 stickers). Your score is
```
Ratio = min(1000, 100 * B / F) / 1000
```
so matching the baseline scores `0.1`; a much tighter, still-feasible net
scores higher (capped at `1.0`).

## Constraints

`1 <= n,m <= 400`, time limit 5s, memory 512MB.

## Example (worked, illustrative shape only)

`n=3, m=3, penalty=4`:
```
0 1 1 3
1 1 2 3
0 0 2 1
```
Keeping `{0,1}` (edges `0-1` and `1-2`, both tab-`E`/tab-`W`): facet 0 at
`(0,0)`, facet 1 at `(1,0)`, facet 2 at `(2,0)` — one straight piece,
bounding area `3*1=3`, `F=3`. Baseline `B = 3 + 4*2 = 11`. Ratio
`= min(1000, 100*11/3)/1000 = 0.366667`.

Keeping nothing (`k=0`) gives `F=B=11` exactly, i.e. `Ratio=0.1` — always
achievable, never great. Edge `2` (`0-2`, tab-N/tab-E, a fold, not an
opposite-tab pair) is a redundant alternative fusion that would instead pull
facet 2 up and to the side (bounding area `4`, worse here) — a preview of
the real trade-off: on larger instances, some fold choices interact and only
a shape-aware choice among the redundant candidates avoids stacking facets
on top of each other.
