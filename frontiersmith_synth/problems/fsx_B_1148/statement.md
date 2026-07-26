# Wings as Registers: Blackout Scene-Swap Choreography

## Problem
A theater stage has `N` numbered cells `0..N-1` in a row, plus two side
**wings**, `L` and `R`, each a LIFO stack (last prop pushed in is the first
one that can come back out) with capacity `w` props. There are `M` props,
each with a fixed **precedence value** `prec[p]` (a permutation of
`1..M` — no two props share a value).

The show has `S` scenes; scene `s` requires an exact layout: for every
cell, either a specific prop must sit there or the cell must be empty.
Between consecutive scenes the lights go out for one **blackout window**;
during a window stagehands may only do two kinds of move, any number of
times, in the order you list them:

- `PUSH src wing p` — prop `p` must currently occupy stage cell `src`;
  it is lifted off stage and placed on top of `wing` (`L` or `R`). Illegal
  if `wing` already holds `w` props. Costs **1 tick** if `wing` is empty
  or `prec[p] >= prec[top of wing]`; otherwise the stagehand has to wedge
  it in carefully and it costs **2 ticks**.
- `POP wing dest p` — the top prop of `wing` must be exactly `p`; it is
  carried to stage cell `dest`, which must currently be empty. Costs
  **1 tick**. You cannot pop anything but the current top.

A prop that is not moved stays exactly where it is (free). At the end of
a window the on-stage layout (cells only — wing contents are backstage
and never checked) must match the next scene exactly.

## Input (stdin)
```
N M S w
prec[1] prec[2] ... prec[M]
Lcount id id ...              (initial wing L contents, bottom..top)
Rcount id id ...              (initial wing R contents, bottom..top)
scene_1 (N ints, cell 0..N-1; 0 = empty, else a prop id 1..M)
scene_2
...
scene_S
B_1 B_2 ... B_{S-1}           (tick budget for each of the S-1 windows)
```
`scene_1` together with the initial wing contents accounts for every prop
exactly once. Guaranteed `N >= M` and `2w >= M` (there is always somewhere
to put every prop).

## Output (stdout)
For each of the `S-1` windows, in order: a line with the move count `k`,
then `k` lines, each one move in the exact form shown above
(`PUSH src wing p` or `POP wing dest p`).

## Feasibility
Any of the following scores `Ratio: 0.0`: an unparsable/garbage/wrong-shaped
output; a `PUSH`/`POP` referencing a prop not currently at the claimed
source (stage cell or wing top); pushing into a full wing; popping a wing
that isn't holding the claimed prop on top; popping onto a non-empty cell;
an out-of-range cell/wing/prop id; or a window whose final on-stage layout
does not exactly match the next scene.

## Objective (minimize)
Let `ticks` be the total tick cost of all your moves, and let
`exceeded` be the number of windows whose own tick total exceeds its
budget `B_i` (this is a penalty, not infeasibility). Minimize
```
F = ticks + P * exceeded,   where P = 10 * N
```

## Scoring
The checker builds its own reference plan `B`: every window it evacuates
the *entire* occupied stage (even props that need not move), fully
empties both wings, refills what the next scene needs, and stows whatever
is left over — always feasible, but wasteful. `B` is `F` computed on that
plan. With `F` your own score:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the reference plan scores `0.1`; a plan `10x` cheaper caps at `1.0`.

## Constraints
`5 <= N <= 9`, `4 <= M <= 7`, `4 <= S <= 8`, `1 <= w <= 5`. Time limit 5s,
memory 256m.

## Example
`N=3, M=2, S=2, w=1`, `prec = [2, 1]`, wings start empty, `scene_1 = [1,0,2]`,
`scene_2 = [2,0,1]`, `B_1 = 10`. One valid plan for the single window:
`PUSH 0 L 1` (wing empty, 1 tick), `PUSH 2 R 2` (wing empty, 1 tick),
`POP L 2 1` (1 tick), `POP R 0 2` (1 tick) — `ticks = 4`, layout now
`[2,0,1]`, matches `scene_2`, `exceeded = 0`, so `F = 4`.
