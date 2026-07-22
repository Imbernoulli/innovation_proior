# Self-Assembling Chimney: Fewest Brick Types

## Problem
You are designing LEGO-like bricks that assemble *themselves*: a brick type
is a unit square whose four sides (N/E/S/W) each carry a **glue** -- either
"null" (no glue) or a positive integer *label* with an integer *strength*
in {1, 2}. You may stamp out as many copies of each brick **type** as you
like, but every type you design costs you (fewer types = better). One
designated **seed** brick is placed at grid cell (0,0). Growth then
proceeds automatically, following one fixed physical rule (the abstract
Tile Assembly Model at **temperature 2**):

> A brick of some type may snap into an empty cell adjacent to already-
> placed bricks **iff** the sum, over its already-filled neighbors, of the
> strengths of glue matches on the touching sides is **>= 2**. Two touching
> sides match only if they carry the *same label* **and** the *same
> strength*. A strength-2 match alone is enough; two strength-1 matches
> from two different neighbors (e.g. South and West) also add up and
> together are enough -- this is *cooperative* binding. A single
> strength-1 match alone is never enough.

Growth must be **unambiguous**: at every moment, every empty cell touching
the structure admits *at most one* addable brick type (two types
simultaneously qualifying at the same cell is rejected). The process runs
until no more bricks can be added (a **terminal**, finite structure).

## Input (stdin)
One line with one integer `T` (0 <= T <= 34 on graded data). Let `k = max(1,
bitlength(T))`. Your **target shape** is:
- if `k = 1`: the single column `{(0,y) : 0 <= y <= T}` (T+1 cells);
- if `k >= 2`: the solid rectangle `{(x,y) : 0 <= x < k, 0 <= y <= 2T}`
  (`k` columns wide, `2T+1` rows tall) **plus** the extra cells
  `{(x, 2T+1) : 1 <= x < k}` -- one short "flag" row sitting on top of
  every column except the leftmost.

Your terminal structure must occupy **exactly** this cell set: no missing
cell, no extra cell anywhere, ever -- including the flag row, which is
part of the required shape like any other cell.

## Output (stdout)
```
M
N1_label N1_strength E1_label E1_strength S1_label S1_strength W1_label W1_strength
... (M such lines, one per brick type, 1-indexed)
seed_type_index
```
`M` is how many distinct brick types you declare (1 <= M). Each of the
next `M` lines gives one type's four sides in N,E,S,W order. `label = 0`
means null glue and must pair with `strength = 0`; a non-null glue has
`label` in `[1, 10^6]` and `strength` in `{1, 2}`. The last line names
which of your `M` types (1-indexed) is placed as the seed at (0,0).

## Feasibility
Your output is rejected (score 0) if: any token is malformed, `M` or a
label/strength is out of range, a null/strength pairing is violated, the
seed index is invalid, growth is ever ambiguous, growth does not reach a
terminal structure within a generous step budget, or the terminal
structure is not *exactly* the target shape.

## Objective and Scoring
Minimize `M`, the number of distinct brick types. Let `B` be the target
shape's total cell count (always achievable by giving every cell its own
private type). If your submission is feasible with `F = M` types,
the checker computes `sc = min(1000, 100*B/F)` and prints `Ratio: sc/1000`.
So a construction that matches `B` (one type per cell) scores 0.1, and a
construction using 10x fewer types than `B` saturates the score. Fewer
types is strictly better; there is no bound relating `M` to `T` other
than `M >= 1`.

## Example (illustrative form only, not the intended target)
A single-column shape with `T=1` (2 cells stacked, k=1) terminates with
just **one** non-seed type: the seed has `North=(label 5, strength 2)`,
all other sides null; the second type has `South=(label 5, strength 2)`,
all other sides null, so growth stops after one more row -- well above
the baseline `B=2`. Real instances use much larger `T`; the shape of a
good design there, not the number "5", is what generalizes.

## Constraints
`0 <= T <= 34` on the graded test data (chosen so the ceiling stays open
for large `T`), time limit 5s, memory 256MB.
