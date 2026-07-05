# Recycling Depot Routes: Maximum Partial-Latin Schedule Completion

## Problem
A city runs an `N`-truck recycling fleet over `N` time slots and `N` depots. A weekly
plan is an `N x N` grid: cell `(i, j)` records which **depot** truck `i` visits in slot
`j`. Two hard rules keep the plan legal:

- **Row rule:** a truck may visit any given depot **at most once** across the day
  (no depot repeats within a row).
- **Column rule:** a depot may receive **at most one** truck in any given slot
  (no depot repeats within a column).

Dispatch has already locked in some assignments (the *prefilled* cells). Your job is to
schedule as many of the remaining `(truck, slot)` cells as possible **without breaking
either rule**. Unassigned cells are allowed — you simply leave them empty.

## Input (stdin)
```
N
row 0: N integers
row 1: N integers
...
row N-1: N integers
```
Each integer is `0` (cell currently empty) or a depot id in `1..N` (a prefilled,
locked assignment). The prefilled cells never conflict with each other.

## Output (stdout)
`N` lines of `N` integers each — your completed grid. Use `0` for a cell you leave
empty and a depot id in `1..N` for a cell you schedule. Every prefilled cell **must
be reproduced with its original value**.

## Feasibility
The output is rejected (score `0`) if any of the following holds:
- fewer than `N*N` integers are provided, or any value is outside `0..N`;
- a prefilled cell is changed;
- some depot id repeats within a row;
- some depot id repeats within a column.

## Objective (maximize)
`F` = the number of **non-zero** cells in your grid (prefilled plus newly scheduled).

## Scoring
Let `B` = the number of prefilled cells (the "schedule nothing new" baseline the checker
builds itself). The reported ratio is
```
Ratio = min(1000, 100 * F / B) / 1000
```
Echoing the input unchanged gives `F = B` and `Ratio = 0.1`. Filling `10x` the baseline
caps the ratio at `1.0`. A full legal completion (`F = N*N`) is always mathematically
possible because the revealed cells are a subset of a valid weekly plan — but finding a
large completion is the hard part (completing a partial Latin square is NP-complete).

## Constraints
- `6 <= N <= 18`.
- Deterministic scoring; all instance randomness is seeded by the test id.

## Example
For `N = 3` with input
```
3
1 0 0
0 0 0
0 0 2
```
`B = 2`. Echoing it back scores `Ratio = 0.1`. A full legal completion, e.g.
```
1 2 3
3 1 2
2 3 1
```
Wait — this changes a prefilled cell, so instead a valid full completion respecting the
locks (`1` at `(0,0)`, `2` at `(2,2)`) such as
```
1 2 3
2 3 1
3 1 2
```
gives `F = 9`, `Ratio = min(1000, 100*9/2)/1000 = 0.45`.
