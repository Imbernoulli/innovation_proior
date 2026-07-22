# Pipe Maze: Stick-Congruent Routing

## Problem

A plumber must route a pipe through a 2D grid maze from a source cell **A** to a sink cell
**B**, moving only orthogonally (up/down/left/right) between adjacent free cells, never
entering an obstacle cell. The pipe is assembled on site from **stock sticks**, each of
fixed nominal length **L**. Consecutive sticks are joined end-to-end by a **weld**.

Every straight run of the pipe consumes length 1-for-1 from whichever stick it is cut from.
Every **bend** (a 90-degree turn in the route) additionally consumes a fixed **bend
allowance a** from the stick that contains it -- forming the bend uses extra material at
that joint. A stick used for `x` units of straight pipe and `k` bends carries load
`x + k*a <= L`. A bend can never be split across two sticks -- the entire allowance for one
turn must come from a single stick, so a weld can never fall in the middle of a bend (it may
fall exactly at the start or end, never strictly inside).

Formally: unroll the route into its **developed length** by walking it edge by edge (each
grid step has length 1) and inserting a virtual block of length `a` at every bend, giving a
1-D length `T`. Partition `[0, T]` into consecutive pieces (sticks) of length `<= L`; a stick
of length `L` used for a piece of length `p` wastes `L - p` (the offcut is scrap). No cut may
land strictly inside a bend's block. Consecutive pieces are joined by a weld.

**The obvious approach -- take the shortest collision-free route -- is usually a trap.**
Mazes are built so the geometric shortest path is forced through a tight corridor and must
turn constantly, scattering bends so densely that almost every stick partition is forced
into small, wasteful pieces (lots of welds, lots of scrap). A **slightly longer** route that
goes around the obstruction, with far fewer bends spaced apart congruently with multiples of
`L`, can pack cleanly into whole sticks with almost no waste -- and wins on total cost.

## Input (stdin)

```
R C
<R lines, each a string of length C over '.'(free) and '#'(obstacle)>
Ar Ac Br Bc
L a W
```
`(Ar,Ac)` = A, `(Br,Bc)` = B (0-indexed row, col; both free cells). `1 <= L`, `1 <= a < L`.

## Output (stdout)

```
K
r_1 c_1
r_2 c_2
...
r_K c_K
```
A polyline of `K` grid cells: `(r_1,c_1) = A`, `(r_K,c_K) = B`, every consecutive pair a
unit orthogonal step, every cell free. The route need not be simple (it may revisit a cell).

## Feasibility

The submitted polyline must: start at A, end at B, take only unit orthogonal steps, stay in
bounds, and never touch an obstacle cell. Any violation scores `Ratio: 0.0`.

## Objective

Given a feasible polyline, its developed length `T` and bend-blocks are computed as above.
The checker finds the **optimal** stick partition of `[0,T]` (minimizing total cost) subject
to the no-cut-inside-a-bend rule -- you output only the route, not the partition. The cost of
a partition into sticks with waste `L-p_i` each and `m` welds is:
```
cost = m * W + sum(waste_i)
```
**Objective: minimize cost.** (Fewer, cleanly-spaced bends and less scrap both help.)

## Scoring

Let `F` be your optimal cost and `B` the checker's own reference-construction cost (a naive
walk toward B, oblivious to L/a/W). Score `= min(1000, 100*B/F) / 1000`, so matching the
naive reference scores ~0.1; a properly stick-aware route scores substantially higher.

## Constraints

`R, C <= 25`; instances fit comfortably under the 5s / 512MB limits.

## Example (worked score, illustrative only)

A path with 6 unit steps and 1 bend, `a=2`: developed length `T = 6+2 = 8`. With `L=8, W=5`
the whole thing fits one stick (no weld) with 0 waste: `cost = 0`. If instead the route had 3
bends (`T = 6+6 = 12`) forcing a 2-stick split with 1 weld and, say, 3 total waste:
`cost = 1*5 + 3 = 8` -- far worse, illustrating why bend count and placement (not raw length)
dominate the score.
