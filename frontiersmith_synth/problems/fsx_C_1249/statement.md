# On-Chip Network Wiring: Traffic-Shaped Express Links Under a Link Budget

## Problem
`N` cores sit on a die, arranged along a physical ring wiring channel (core `i`
next to core `i+1 mod N`). You must design the on-chip network: a set of
undirected links between cores. Linking `i` and `j` costs
`cost(i,j) = min(|i-j|, N-|i-j|)` — nearby cores are cheap to wire, cores on
opposite sides of the die are expensive — and your total wiring **link
budget** is `L_max`: the sum of costs of every link you build must not exceed it.

You are also given the measured **traffic matrix** `T[i][j]` — how many
message units per cycle core `i` sends to core `j`. Traffic is routed
automatically along shortest paths in the network you built, one canonical
shortest-path tree per source found by breadth-first search over each core's
neighbors in ascending id order — a fixed rule, so routing is completely
determined once you fix the topology (you do not choose routes yourself).
Every link also has a **capacity**
`CAP`: if the total traffic funneled through one link (summed over every
routed flow that uses it, both directions) exceeds `CAP`, that link stalls,
costing `STALL_COST` per unit of traffic over capacity.

Your job: choose links (within budget) that minimize the network's total
**cost of operation** — traffic-weighted hop count plus congestion stalls.

## Input (stdin)
```
N
L_max
CAP
STALL_COST
```
then `N` lines of `N` integers each: row `i` is `T[i][0] ... T[i][N-1]`
(`T[i][i] = 0`). `4 <= N <= 200`.

## Output (stdout)
```
M
u_1 v_1
...
u_M v_M
```
`M` undirected links (`0 <= u,v < N`, `u != v`, no duplicates).

## Feasibility
- Every token must parse as an integer (no `nan`/`inf`/decimals — reject).
- `0 <= M <= N(N-1)/2`, valid endpoints, no self-loop, no duplicate link.
- `sum(cost(u,v) over your links) <= L_max`.
- The network must connect all `N` cores (every core reachable from every other).
Any violation scores `Ratio: 0.0`.

## Objective
For every ordered pair `(i,j)` with `T[i][j] > 0`, route it along its
canonical shortest path (the source-rooted BFS tree above) in your network;
let `hops(i,j)` be its length. Let `load(e)` be the total traffic (summed over
every routed flow, both directions) that crosses link `e`. Minimize:

```
F = sum_{i,j} T[i][j] * hops(i,j)  +  sum_e max(0, load(e) - CAP) * STALL_COST
```

## Scoring
The checker builds its own baseline `B`: the same `F`, computed on the
minimal ring alone (`i` linked to `i+1 mod N`). With your objective value `F`:
```
Ratio = min(1, B / F)
```
The bare ring baseline scores `0.1`. Ten times more efficient than the ring caps at `1.0`.

## Constraints
- `T[i][j] >= 0`, `T[i][i] = 0`; `L_max >= N` (a ring always fits).
- `CAP, STALL_COST >= 1`.
- Deterministic integer scoring; time limit 2–5s.

## Example (illustrative shape only, not an actual test)
`N=4`, ring links `(0,1),(1,2),(2,3),(3,0)`, all costs `1`. Suppose traffic is
`T[0][2]=T[2][0]=10`, everything else `1`. On the ring, `hops(0,2)=2`, so those
10-unit flows cost `20` hops each way — `40` total — likely pushing some ring
edge's load over `CAP` too. Adding one express link `(0,2)` (cost `2`, if
budget allows) drops `hops(0,2)` to `1`, cutting that term to `20` and
draining the ring edges that used to carry it, at a one-time budget cost of
`2` — the essence of the intended trade-off.
