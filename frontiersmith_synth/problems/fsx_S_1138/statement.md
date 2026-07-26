# Paper Snowflake: Minimal Fold/Punch Instruction Tape

## Problem
You are given a square sheet of paper, `N x N` unit cells (`N` a power of two), and a
target set of cells that must end up as holes -- a paper snowflake pattern. You cut the
snowflake with an **instruction tape** of four operation kinds:

- `FOLD_X` -- fold the sheet's current *row*-extent (its first coordinate, `i`) exactly
  in half: the far half is laid exactly on top of the near half, cell for cell. Every
  current row index `x` in the new (halved) extent now has, stacked on top of it,
  whatever layers were at `x` **and** whatever layers were at the mirror row (the far
  half's counterpart). Requires the current row-extent to be even and at least 2.
- `FOLD_Y` -- the same fold, but on the current *column*-extent (`j`).
- `PUNCH x y` -- drive a punch straight through the sheet at the *current* position
  `(x,y)`. It pierces **every layer** currently stacked at that position, i.e. every
  original cell that has, through the folds performed so far, come to lie at `(x,y)`.
- `UNFOLD_ALL` -- open the sheet back out. This must be the **final** instruction of
  the tape (nothing may follow it), and it must appear exactly once.

Each instruction, of any kind, costs exactly **one unit** on the tape. Your job is to
produce the shortest tape that punches exactly the target holes -- no more, no fewer.

## Input (stdin)
```
N
H
i_1 j_1
i_2 j_2
...
i_H j_H
```
`N` is a power of two (`4 <= N <= 64`). `H` is the number of target hole cells,
followed by their `(row, col)` coordinates, `0 <= i,j < N`, each appearing once.

## Output (stdout)
A whitespace/newline-separated instruction tape using exactly the four tokens above,
e.g.:
```
FOLD_X
FOLD_Y
PUNCH 0 0
PUNCH 1 0
UNFOLD_ALL
```
`PUNCH` always takes two integer arguments, given in the *current* (already-folded)
coordinate system, which shrinks every time a fold happens.

## Feasibility
Feasible iff: every `FOLD_X`/`FOLD_Y` is applied while its dimension is even and `>= 2`;
every `PUNCH x y` satisfies `0 <= x < current row-extent`, `0 <= y < current
column-extent`; `UNFOLD_ALL` appears exactly once, as the last instruction; and the set
of original cells pierced by all punches together equals the target hole set **exactly**
(no extra hole, no missing hole). Any violation, parse error, or non-integer/non-finite
`PUNCH` argument scores 0.

## Objective
Minimize the total instruction count of a feasible tape.

## Scoring
Let `B = H + 1` (the naive tape: `PUNCH` every target cell individually with zero folds,
then `UNFOLD_ALL` -- always feasible, always costs exactly `H + 1`). With your feasible
tape's instruction count `F`,
```
Ratio = min(1, 0.1 * B / F)
```
The naive per-cell tape scores `0.1`. Halving your instruction count doubles the ratio;
using a tenth of `B` instructions caps the ratio at `1.0`.

## Constraints
- `4 <= N <= 64`, `N` a power of two; `0 <= H <= N*N`.
- Instruction tape length `<= 5000`.
- Deterministic exact integer simulation; no timing.

## Example
`N=4`, target holes `{(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)}` (`H=8`, `B=9`).
`FOLD_X` merges row `0` with row `3` and row `1` with row `2` (both merges land on
identical rows, so nothing is lost); the folded `2 x 4` sheet has row `0` = holes at
`j=1,2` and row `1` = holes at `j=0,3`. `FOLD_Y` then merges column `0` with column `3`
and column `1` with column `2` inside each of those rows, giving a `2 x 2` sheet with
exactly two hot cells: `(0,1)` and `(1,0)`. The tape
`FOLD_X / FOLD_Y / PUNCH 0 1 / PUNCH 1 0 / UNFOLD_ALL` (5 instructions) reconstructs all
8 holes exactly: `Ratio = min(1, 0.1*9/5) = 0.18`. Punching each of the 8 holes
individually (the naive tape) instead scores exactly `0.1`.

## Notes
Target patterns are the orbit of a small "core" pattern under a chain of fold
subgroups, so deep compressions are usually available -- but real snowflakes are only
*approximately* symmetric: a handful of cells may break the sheet's whole-grid mirror
symmetry while leaving almost all of its local fold structure intact. A tape that
insists on perfect global symmetry before folding at all will miss this; a tape that
folds anyway and patches the few broken cells with individual corrections at the right
depth will not.
