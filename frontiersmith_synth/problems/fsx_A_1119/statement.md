# One Coolant, Many Fronts

## Problem

A molten rod is discretized into `N` cells in a line, indexed `0..N-1`. A few
cells are pre-placed **seed grains** that are already solid, each carrying an
integer *orientation* label; every other cell starts liquid with heat `H0`.
You control a single coolant actuator. Over `K` stages you choose, each
stage, **at most one** currently-liquid cell to directly cool. The field then
diffuses and any liquid cell that gets cold enough freezes permanently,
inheriting its orientation from the nearest already-solid cell.

A stage proceeds in this fixed order:

1. **Cool (optional).** You name one liquid cell `c`; its heat drops by
   `CSTEP` (`heat[c] -= CSTEP`). You may instead cool nothing this stage.
2. **Diffuse.** Every liquid cell `i` updates *simultaneously*:
   `new_heat[i] = floor((L + R + 2*heat[i]) / 4)`, where `L = heat[i-1]` if
   cell `i-1` exists and is liquid, otherwise `L = heat[i]` (a solid cell or
   the grid edge is a perfect insulator: no heat crosses it, so a liquid
   region only exchanges heat with itself). `R` is defined symmetrically.
3. **Freeze.** Every liquid cell whose *new* heat is `<= F` solidifies. If
   several cells cross the threshold in the same stage they all freeze
   together, each independently taking the orientation of the cell that was
   **nearest to it among the cells that were already solid at the START of
   this stage** (Manhattan distance along the line; a tie between an
   equidistant solid cell to the left and one to the right is broken in
   favor of the **left** one). A frozen cell's orientation never changes
   again, and solid cells never move or re-melt.

You are given a target microstructure `T[0..N-1]`. Because solid regions
block diffusion, the direction and timing in which you push each front
changes what the *next* front even sees -- the same total number of cooling
actions, spent in a different order, can lock in a completely different
pattern.

## Input (stdin)

```
N K H0 F CSTEP
P
pos_1 orient_1
...
pos_P orient_P
T_0 T_1 ... T_{N-1}
```
`N` cells, `K` stages, initial heat `H0`, freeze threshold `F`, cooling step
`CSTEP` (all fixed for the whole instance). `P` seed grains follow, each a
`(position, orientation)` pair (`1 <= orientation <= P`, positions distinct,
sorted increasing, and `T[pos_k] = orient_k`). Finally the length-`N` target
array, each entry in `1..P`.

## Output (stdout)

Exactly `K` lines. Line `s` is a single integer: `-1` (cool nothing that
stage) or a cell index `0..N-1` naming the liquid cell to cool at stage `s`.

## Feasibility

Every one of the `K` tokens must parse as an integer in `[-1, N-1]`. Whenever
a stage names an index `c != -1`, cell `c` must be liquid **at that point in
the simulation** (you may never cool an already-solid cell). Any violation,
or a malformed/incomplete/non-finite output, scores `0`.

## Objective & Scoring

Simulate your `K` stages exactly as specified. Let `F_raw` = number of cells
whose final orientation equals `T`. The checker also simulates its own
"cool nothing, ever" baseline on the same instance to get `B_raw` (only the
seed cells match `T` there). Your score is
`Ratio = min(1.0, 0.1 * F_raw / B_raw)`. Matching every cell scores far above
the baseline (capped at `1.0`); matching only what the seeds already give you
scores `0.1`.

## Constraints

`8 <= N <= 70`, `2 <= P <= 10`, `K <= 250`, values fit in 32-bit signed
integers. Time limit 5s, memory 512MB.

## Example (worked, not to scale)

`N=5 K=6 H0=100 F=45 CSTEP=45`, seeds `(0,1)` and `(4,2)`, target
`1 1 2 2 2` (seed 0 claims only cell 1; seed 4 claims cells 2,3,4 -- an
*asymmetric* split, not the midpoint). One valid schedule: cool cell 1 for
two consecutive stages so it freezes as orientation 1, then commit to cell 3
until it freezes as orientation 2, then finish cell 2 the same way. A "fair"
schedule that alternates between cell 1 and cell 3 every stage instead lets
cell 2 end up equidistant from both fronts when it finally freezes --
landing on whichever side happened to reach it first, not necessarily the
target's chosen side.
