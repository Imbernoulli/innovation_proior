# Ledger Recovery Under Blackout Blocks

## Problem
A quartermaster keeps an `n x m` ledger of resupply costs: row `i` is a depot, column `j`
is a route, and cell `(i, j)` is the (real-valued) cost of shipping depot `i`'s goods over
route `j`. Depots and routes are secretly organized into a small number of unlabeled
*groups* (e.g. depots on the same rail line, routes through the same terrain); the cost of
a cell is driven mainly by which depot-group meets which route-group, plus small
depot-specific and route-specific surcharges.

Fire has destroyed part of the ledger. Some destroyed cells are scattered at random.
Others are destroyed in **blackout blocks**: every cell where a whole subset of one
depot-group crosses a whole subset of one route-group is gone at once — a structural
gap, not a random one. You do NOT get the group labels directly, but you are given two
**similarity graphs**: an edge between two depots means they belong to the same
depot-group, and likewise for routes. Reconstruct the missing cells as accurately as
possible.

## Input (stdin)
```
n m K L seed
p
p lines: i j v          (observed cell (i,j) has cost v)
re
re lines: a b           (depot a and depot b are in the same depot-group)
ce
ce lines: a b            (route a and route b are in the same route-group)
q
q lines: i j             (predict the cost of this destroyed cell, in this order)
```
`K`, `L` are the (unused-by-you-directly) group counts; `seed` is bookkeeping for the
grader. Rows/columns are 0-indexed. The similarity edges within each group always form a
connected sub-graph, so connected components recover group membership exactly.

## Output (stdout)
Exactly `q` numbers (any whitespace layout), the predicted cost for each queried cell, in
the same order as the input's query list.

## Feasibility
The output is valid iff it contains exactly `q` finite, parseable real numbers, each with
absolute value `<= 1e6`. Any violation scores `Ratio: 0.0`.

## Objective
Let `RMSE` be the root-mean-squared error of your predictions against the true (hidden)
costs of the `q` queried cells. Define `F = 1 / (1 + RMSE)` (so `F` is in `(0, 1]` and
rises as your reconstruction improves). Maximize `F`.

## Scoring
The checker also builds its own baseline prediction: for each queried cell, the mean of
the *observed* costs in that same row (or the global observed mean if the row has no
observed cells). Let `B = 1 / (1 + RMSE_baseline)` from that baseline. Then
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the row-mean baseline scores `Ratio = 0.1`; cutting the RMSE enough to make
`F` ten times `B` caps the score at `1.0`. A pattern-blind low-rank fit that ignores the
two graphs can match this baseline on scattered damage but is not obligated to do any
better inside a blackout block — the graphs are what let you transfer known cost
structure from a group's surviving members across the gap.

## Constraints
`12 <= n, m <= 30`; each `.in <= 5MB`; time limit 5s, memory 512m. Every depot-group and
route-group involved in a blackout block still has other members with observed cells
elsewhere in the ledger, so the group-pair's typical cost is always recoverable in
principle from the rest of the data.

## Example (illustrative arithmetic only)
Suppose 4 queried cells all belong to one depot-group x route-group pair whose true costs
are `40, 40, 40, 40`, while the row-mean baseline (built from other, cheaper routes in
those rows) predicts `30` for all four: `RMSE_baseline = 10`, `B = 1/11 = 0.0909`.
A submission predicting `38, 38, 42, 42` has `RMSE = 2`, `F = 1/3 = 0.3333`, giving
`sc = 100 * 0.3333 / 0.0909 = 366.7`, so `Ratio = 0.3667`. A submission that recovers the
group's true typical cost exactly (`RMSE = 0`, `F = 1`) would score `Ratio = 1.0` (capped).
