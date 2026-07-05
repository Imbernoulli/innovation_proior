# Traffic Signal Phase Grid Completion

## Problem
A city district is an `N x N` grid of intersections. Every intersection must be
assigned one of `N` **signal phases** (integer phase codes `0 .. N-1`). To keep
traffic waves coherent, the traffic authority requires that **within any row of
intersections, no phase code repeats**, and likewise **within any column, no
phase code repeats** — exactly the row/column constraints of a Latin square.

Some intersections have already been commissioned with a fixed phase (these
prefilled assignments never conflict with one another). Your job is to program
the remaining intersections so that as many intersections as possible carry a
**valid, non-conflicting** phase. Dense pre-commissioning can make a fully
conflict-free assignment impossible, so this is a maximization problem, not a
feasibility puzzle.

## Input (stdin)
```
N
row 0:  N integers
row 1:  N integers
...
row N-1: N integers
```
Each grid entry is either a prefilled phase code in `0 .. N-1`, or `-1` for an
un-commissioned intersection. The prefilled cells form a valid partial Latin
square.

## Output (stdout)
`N` lines of `N` integers — your completed grid. Each entry must be a phase code
in `0 .. N-1`, or `-1` to leave an intersection dark (unassigned).
**Every prefilled cell must keep exactly its given phase code.**

## Feasibility
The output is rejected (score 0) if it is not exactly `N` lines of `N` integers,
if any entry is outside `{-1, 0, ..., N-1}` (non-integer / `nan` / `inf`
included), or if any prefilled intersection's phase was changed.

## Objective
Let a cell be **valid** if it holds a phase code (not `-1`) that appears exactly
once in its row and exactly once in its column, counting all non-empty cells.
Your raw score `F` is the number of **newly assigned** valid cells (prefilled
cells are excluded from `F`; leaving a cell `-1` scores nothing for it). A phase
you add that collides with another cell in its row or column makes both cells
invalid, so reckless filling can hurt.

## Scoring
The checker builds a weak internal baseline `B` (a sparse "stripe-cyclic" fill)
and reports
```
sc    = min(1000, 100 * F / max(1, B))
Ratio = sc / 1000
```
Reproducing the baseline scores about `0.1` (the baseline itself validly fills a
sparse "stripe" of intersections); filling roughly ten times as many valid
intersections as the baseline saturates at `1.0`.

## Constraints
`19 <= N <= 46`. About one third of the intersections are prefilled. Deciding
the true maximum number of completable cells is NP-hard, so no exact optimum is
expected — better constructive strategies simply score higher.

## Example (worked score)
Suppose `N = 4` and the baseline validly fills `B = 3` cells. If your completion
adds `F = 12` valid cells, then `sc = min(1000, 100 * 12 / 3) = 400` and
`Ratio = 0.400`. Adding `F = 30` (were the grid large enough) would give
`sc = min(1000, 1000) = 1000`, i.e. `Ratio = 1.000`.
