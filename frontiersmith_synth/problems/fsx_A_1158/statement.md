# Pack-Ice Convoy Leases: Riding the Refreeze Window

## Problem
A single icebreaker fleet threads cargo convoys through a pack-ice channel modelled as
cells `1..L` in a line out from the port (`cell 0`). You command `B` identical breakers.
There are `M` cargo ships; ship `j` has a per-cell pace `s_j` (ticks needed to cross one
cell), a destination cell `d_j`, and a priority weight `w_j`.

A breaker may lead an **escorted convoy** of up to `c` ships out of the port. The convoy
(breaker + its still-travelling members) advances one cell at a time in lock-step: the
tick-cost to enter the next cell equals the **slowest still-travelling member's** `s`
(mixing a fast and a slow ship in one convoy throttles the fast one to the slow one's
pace). A ship leaves the convoy the instant it reaches its own `d_j` (its arrival tick).
The instant the convoy (or breaker) enters a cell, that cell becomes **clear**; it
**refreezes** exactly `r` ticks after that clearing tick unless re-cleared. After finishing
a convoy at cell `D`, a breaker must sail back to the port (`D` more ticks, unescorted,
always possible) before it may depart with its next convoy.

A ship may instead travel **unescorted** ("lease" the channel): pick a departure tick and
move alone at its own pace `s_j` ticks/cell, with NO convoy-capacity limit — but every cell
it enters must currently be clear (cleared by some breaker's pass at or before that tick,
not yet refrozen), or the plan is infeasible. This is the only way to move more than `c`
ships through one breaker's pass: once a convoy has broken a corridor, any number of ships
can simultaneously lease that decaying clear window if each one's own timing keeps it
inside the window the whole way. Breakers/ships never collide; only capacity, clocks, and
the refreeze constant `r` govern feasibility.

## Input (stdin)
```
L M B c r
s_1 d_1 w_1
...
s_M d_M w_M
```
All values are positive integers, `1 <= d_j <= L`.

## Output (stdout)
```
T
b_1 t0_1 k_1 id_1 ... id_{k_1}
...
P
id t0
...
```
`T` escorted-trip lines: breaker index (`0..B-1`), non-negative departure tick, convoy size
`1<=k<=c`, then that many distinct ship ids (`1..M`). Then `P` unescorted-lease lines: a
ship id and its departure tick. Every ship id `1..M` must appear **exactly once**, total,
across all trips and leases.

## Feasibility
- All ids/ticks/counts must parse as integers in range; each ship id used exactly once.
- For each breaker, its trips (ordered by departure tick) may not overlap: trip `k+1`'s
  departure tick must be `>=` the previous trip's finish tick plus its own return distance.
- A leased (unescorted) ship must find every cell `1..d_j` clear at the exact tick it
  arrives there (own constant pace from its departure tick); otherwise the whole submission
  is infeasible.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`F = sum_j w_j * arrival_j` over all `M` ships (weighted total arrival time).

## Scoring
The checker's internal baseline `B` escorts every ship **alone** (its own convoy of size 1),
round-robin across the `B` breakers in ship-index order, each departing the instant its
breaker is free. With your feasible `F`:
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so the singleton baseline scores about `0.1`, and a schedule ten times better caps at `1.0`.

## Constraints
`1 <= L <= 300`, `1 <= M <= 80`, `1 <= B <= 3`, `1 <= c <= 6`, `1 <= r <= 40`,
`1 <= s_j <= 12`, `1 <= w_j <= 10`. Runs in well under the time limit.

## Example
`L=5 M=2 B=1 c=2 r=3`; ship 1: `s=1 d=3 w=2`; ship 2: `s=1 d=5 w=1`. Escort both together
(`T=1`: `0 0 2 1 2`, `P=0`): the convoy reaches cell 3 at tick 3 (ship 1 arrives) and cell 5
at tick 5 (ship 2 arrives), `F = 2*3 + 1*5 = 11`. The singleton baseline sends ship 1 alone
(arrives tick 3, breaker free at tick 6) then ship 2 alone (departs tick 6, arrives tick 11),
`B = 2*3 + 1*11 = 17`. `Ratio = min(1000, 100*17/11)/1000 = 0.154545`.
