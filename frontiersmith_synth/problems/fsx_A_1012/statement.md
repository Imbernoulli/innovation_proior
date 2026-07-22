# Switchback Haul Road: Phase-Locked Convoy Pipeline

## Problem
A mine's pit and crusher are joined by one narrow switchback road, too tight for two
trucks to pass anywhere except at built passing bays ("pullouts"). The road is a chain of
`M` **single-lane blocks**; block `i` connects node `i-1` to node `i`. Node `0` is the
pit, node `M` is the crusher, nodes `1..M-1` are the pullouts. Pit and crusher have
unlimited room; every pullout and every block holds **at most one truck**, in either
direction, at a time.

You control `K` trucks that cycle forever: load at the pit, haul down to the crusher,
return empty, repeat. Crossing block `i` takes exactly `t_i` ticks (fixed — no stopping
partway). Descending (node increasing, loaded) raises a truck's brake heat by `g_i`;
ascending (node decreasing, empty) lowers it by a fixed `heat_loss`; waiting anywhere
lowers it by `idle_cool` per tick (heat floors at 0). A truck may **not** start descending
a block if that would push its heat above the cap `H_MAX` — it must already be cooled
enough, or wait first. Heat is tracked separately per truck, starting at 0.

Plan every truck's full schedule over a horizon of `T` ticks to maximize completed loads.

## Input (stdin)
```
M K T H_MAX idle_cool heat_loss
t_1 t_2 ... t_M
g_1 g_2 ... g_M
```

## Output (stdout)
```
K
L_0 tick_0 node_0 tick_1 node_1 ... tick_{L_0-1} node_{L_0-1}
L_1 ...
...
L_{K-1} ...
```
One line per truck (input order): `L_k` checkpoints `(tick, node)`, strictly increasing in
tick, describing that truck's whole journey; the first must be `(0, 0)`. Between two
consecutive checkpoints, either the node is unchanged (waiting) or it changes by exactly
`±1` with the tick gap equal to that block's `t_i` (crossed it; `+1` = descending, `-1` =
ascending). No checkpoint may exceed tick `T`.

## Feasibility
- Checkpoints start `(0, 0)`; ticks strictly increase; nodes stay in `[0, M]`; every move
  is a wait or a legal `±1` crossing of the exact right duration.
- **Heat**: replay each truck's own heat from 0; a descending crossing that would exceed
  `H_MAX` is illegal.
- **Block capacity**: per block, the crossing intervals `[enter, enter+t_i)` of all trucks
  (either direction) must be pairwise disjoint.
- **Pullout capacity**: a truck occupies pullout `p` from arrival until it next departs
  (both endpoints inclusive). These spans must be pairwise disjoint at the same pullout —
  two trucks may not even touch the same bay at once, from either direction.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F` = total number of **completed loads**: a truck completes one when it crosses into
node `M` having reached it via an unbroken, non-decreasing node run since its last visit
to node `0` (loaded at the pit, drove straight through). It must then return all the way
to `0` before another crossing into `M` counts; shuttling over just the last block without
reaching the pit earns nothing.

## Scoring
The checker's internal baseline `B` sends a single truck through repeated complete round
trips (its own just-in-time cooldown stops included) while the other `K-1` trucks never
move, and counts that lone truck's deliveries within `T`. With your feasible `F`,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so matching the one-truck baseline scores about `0.1`; using the whole fleet well scores
higher.

## Constraints
`3 <= M <= 14`, `2 <= K <= 10`, `1 <= t_i, g_i`, `1 <= idle_cool, heat_loss`, all values
positive integers fitting in 32 bits. Runs well under the time limit.

## Example
`M=3, K=1, T=20, H_MAX=100, idle_cool=1, heat_loss=1`, `t=[3,4,2]`, `g=[2,2,1]` (heat
never binds — illustrates the checkpoint format only, not the real scoring regime). Output
`1` then `7 0 0 3 1 7 2 9 3 11 2 15 1 18 0`: block 1 in `[0,3)` to node 1, block 2 in
`[3,7)` to node 2, block 3 in `[7,9)` to the crusher (`F=1`), then straight back via block
3 in `[9,11)`, block 2 in `[11,15)`, block 1 in `[15,18)` — back at the pit with no time
left in `T=20` for another delivery. The lone-truck baseline also completes exactly one
delivery over `T=20`, so `B=1`, `Ratio = min(1000, 100*1/1)/1000 = 0.1`.
