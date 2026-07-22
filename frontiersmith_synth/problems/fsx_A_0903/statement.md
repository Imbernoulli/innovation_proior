# DroneScript Barrier Compiler: Pricing the Swarm's Safety Proof

## Problem
A quadcopter fleet occupies a 3D warehouse grid. Every drone must travel from its
start cell to its goal cell by way of a compiled flight **program** built from three
instructions:
```
MOVE d DIR   -- drone d steps one grid cell along DIR in {PX,NX,PY,NY,PZ,NZ}; costs 1
HOLD d       -- drone d stays put this instant; costs 0
BARRIER      -- a global synchronization fence; costs Bc
```
The program is a flat list of instructions. `BARRIER` instructions cut it into
ordered **blocks**. Everything inside one block is issued to the fleet at the same
physical instant; instructions in different blocks are never checked against each
other, because the barrier already guarantees the fleet is at rest before the next
block starts. A `MOVE`'s destination cell must be **unoccupied at the start of the
block** — held by no drone, whether that drone is staying put or is itself flying
elsewhere in the very same block — and no two `MOVE`s in one block may target the
same destination (this also rules out a direct two-drone swap, since a swapping
drone's target is the other's pre-block cell). On top of that, the fleet shares one
uplink: a block may contain **at most `Ccap` `MOVE` instructions in total**, however
many independent drones are involved. Any violation makes the whole program worth 0.

`MOVE`s are unavoidable — drones must fly. `BARRIER`s are a tax you pay for **not**
having proven that the blocks on either side are safe to run concurrently, and every
block is additionally squeezed by the shared `Ccap` budget. The cheapest program is
not the one with the shortest paths; it is the one that both proves the most
independence AND spends the shared per-block command budget on whichever drones are
actually holding up the finish line, so it needs the fewest synchronization fences.

## Input (stdin)
```
X Y Z N Bc Ccap
sx1 sy1 sz1 gx1 gy1 gz1
...
sxN syN szN gxN gyN gzN
```
`X,Y,Z` bound the grid (`0<=x<X`, `0<=y<Y`, `0<=z<Z`). `N` drones follow, each with a
start `(sx,sy,sz)` and goal `(gx,gy,gz)`. No two drones share a start cell, and no two
share a goal cell. `Ccap>=1` is the maximum number of `MOVE`s any single block may
contain.

## Output (stdout)
```
P
<P instruction lines, each "MOVE d DIR", "HOLD d", or "BARRIER">
```
`d` is a 0-indexed drone id. Emit exactly `P` lines after the first; any non-blank
content beyond those `P` lines is rejected.

## Feasibility
Replay the program, resetting the pending block at every `BARRIER` and once more at
end-of-program. Inside one block a drone may appear in at most one instruction. Every
`MOVE` must land in-bounds, on a cell nobody occupies at the block's start, and no two
movers in the block may share a destination; a block may hold at most `Ccap` `MOVE`s.
After the whole program every drone must sit exactly on its own goal cell. Any
failure -> `Ratio: 0.0`.

## Objective
Minimize total program cost = `(#MOVE instructions) + Bc * (#BARRIER instructions)`.
(`HOLD` is free and never required — omitting a drone from a block already holds it.)

## Scoring
Let `M = sum_i manhattan(start_i, goal_i)`. The checker's reference cost is the
"never batch anything" construction: move every drone alone, one `MOVE` then its own
`BARRIER`, i.e. `Base = M * (1 + Bc)`. With your program's true cost `F`:
```
Ratio = min(1, 0.1 * Base / F)
```
Routing every drone in total isolation scores near `0.1`; halving `F` roughly doubles
the ratio.

## Constraints
`1 <= N <= 1000`, `2 <= X,Y,Z <= 300`, `1 <= Bc <= 20`. Time limit 2-5s.

## Example
Drone A: `(0,0,0) -> (1,0,0)`. Drone B, on an unrelated row: `(0,5,0) -> (1,5,0)`.
`Bc = 5`, `Ccap = 2` (room for both), so `Base = 2 * 6 = 12`. Neither drone's move
touches a cell the other ever uses, and the block only needs 2 of its `Ccap=2`
slots, so a program that issues both moves in one shared block, with **no barrier
anywhere**, is fully feasible:
```
2
MOVE 0 PX
MOVE 1 PX
```
`F = 2 + 0*5 = 2`, `Ratio = min(1, 0.1*12/2) = 0.6`. Proving two drones can never
interact lets you skip synchronizing them entirely — the more of the fleet you can
prove mutually independent, the fewer fences (and the higher the score) your program
needs, no matter how many drones are actually moving.
