# Twin-Row Tally Tableau

## Problem
You build a digit tableau: `R` rows by `C` columns, every cell holding a digit in
`{0, ..., base-1}`. A subset of cells are **seed hints**: their digit is fixed by the
input and must not be changed. The remaining cells are yours to fill.

A subset of the (non-hint) cells are **counter cells**. Each counter cell `(r, c)` is
tagged with a target digit `d` and a **scope**: either its own row `r`, or its own row
`r` together with a fixed **partner row** `partner(r) = (r + R/2) mod R` (`R` is always
even, so rows split into `R/2` disjoint partner pairs). A counter cell is **satisfied**
if the digit written in it equals the number of occurrences of digit `d` among all cells
of its scope (the counter cell itself is part of its own scope, so this is a genuine
self-reference, not a one-shot count of something external). Note that a row-scope
counter only "sees" its own row, while a pair-scope counter's tally depends on both rows
in its pair — edit either row and every pair-scope counter anchored in that pair may
flip.

Each counter cell also carries a positive integer weight `w`. Your job is to choose
digits for all non-hint cells to make as much total weight of *simultaneously* satisfied
counters as possible — you are not required to satisfy every counter, and forcing one
counter to hold rarely leaves every other counter's tally unchanged, since row and
pair scopes overlap heavily.

## Input (stdin)
```
R C base
K
r_1 c_1 d_1 w_1 scope_1
...
r_K c_K d_K w_K scope_K
H
r_1 c_1 v_1
...
r_H c_H v_H
```
`scope_i` is the literal token `ROW` or `PAIR`. Counter cells and hint cells are always
disjoint sets of `(r, c)` positions; no cell carries two constraints.

## Output (stdout)
`R` lines, each with `C` space-separated integers in `[0, base-1]`: the completed
tableau, row by row.

## Feasibility
- Output must contain exactly `R` lines of exactly `C` integer tokens each (whitespace
  separated overall is fine), every token parsing as a plain base-10 integer in
  `[0, base-1]`.
- For every seed hint `(r, c, v)`, the output digit at `(r, c)` must equal `v` exactly.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F`, the sum of `w_i` over all counters `i` whose written digit equals the true
occurrence count of `d_i` in its scope, evaluated on your final tableau.

## Scoring
The checker builds its own reference tableau `T0`: every non-hint cell gets digit
`(column index) mod base` (hint cells keep their fixed value), independent of any
counter. Let `B` be `F` evaluated on `T0` (always `>= 1` after flooring). With
maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing `T0` scores `Ratio = 0.1`; achieving 10x `T0`'s weighted-satisfied total caps
at `1.0`.

## Constraints
- `4 <= R <= 14` (even), `6 <= C <= 16`, `8 <= base <= 12`.
- `1 <= K`, `0 <= H`, `K + H <= R*C`.
- Weights `1 <= w_i <= 9`.
- Time limit 5s, memory 512m.

## Example
`R=4, C=4, base=4`, one `ROW` counter at `(0,0)` with `d=1, w=5` (scope = row 0), no
hints. Row 0 written as `2 1 0 0`: the count of digit `1` in row 0 is 1 (position 1
only), but the written digit at `(0,0)` is `2` — not satisfied. Instead write row 0 as
`1 0 2 3`: digit `1` occurs exactly once in row 0 (position 0, the counter cell itself),
and the counter cell holds `1` — satisfied, contributing `F += 5`. (This tiny example is
illustrative only; real instances have many interacting counters.)
