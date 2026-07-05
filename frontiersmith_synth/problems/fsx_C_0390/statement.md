# Warehouse SKU Stocking: Maximum Diagonal-Restricted Latin Completion

## Problem
An automated warehouse is an `N x N` grid of storage slots: `N` aisles (rows) by
`N` rack columns. There are `N` distinct SKUs (item types), numbered `1..N`.
A slot may hold **one** SKU or be left **empty**.

Three distinctness rules must hold for the SKUs that are stocked:

1. **Aisle rule** — no SKU appears twice in the same row.
2. **Rack rule** — no SKU appears twice in the same column.
3. **Conveyor-loop rule** — the slots are threaded onto `N` circular conveyor
   loops, where slot `(r, c)` belongs to loop `(r + c) mod N`. No SKU appears
   twice on the same loop.

Each slot has a **reachability list**: because of robot geometry, the robot that
services slot `(r, c)` can only place a SKU drawn from a small allowed set
`A[r][c]`. Some slots are already **prefilled** (givens); every given already
respects the three rules and lies inside its own reachability list.

You must decide what (if anything) to stock in every remaining slot.

## Input (stdin)
```
N
N lines : the prefilled grid, N integers per line (0 = empty slot)
N*N lines (row-major, r = 0..N-1 then c = 0..N-1):
        k  a_1 a_2 ... a_k      = the reachability list A[r][c]
```

## Output (stdout)
`N` lines of `N` integers: your final grid `S`. Each `S[r][c]` is either `0`
(leave empty) or a SKU in `1..N`. Every prefilled slot must be reproduced
exactly.

## Feasibility
The board is rejected (score `0`) unless: exactly `N*N` integer tokens are
present, every token lies in `[0, N]`, every prefilled slot is preserved, every
stocked SKU is in that slot's reachability list, and all three distinctness
rules hold.

## Objective (maximize)
`F` = the number of **stocked** slots (non-empty cells).

## Scoring
Let `B` be the number of prefilled slots (the checker rebuilds this trivial
feasible baseline itself). With `sc = min(1000, 100 * F / B)` the reported score
is `Ratio = sc / 1000`. Echoing only the givens yields `Ratio = 0.1`; stocking
ten times as many slots as were given caps at `1.0`. The tight lists plus the
conveyor-loop rule make a full completion impossible in general, so the maximum
is genuinely open — many completions of different sizes exist.

## Constraints
`9 <= N <= 18`. Each reachability list has 3 SKUs. Deterministic scoring; exact
integer arithmetic.

## Example (worked score)
Suppose `N = 9`, there are `B = 12` givens, and your board stocks `F = 48` slots
in total (all rules satisfied). Then `sc = min(1000, 100 * 48 / 12) = 400` and
`Ratio = 0.400`. A board that only echoes the 12 givens scores
`100 * 12 / 12 / 1000 = 0.100`.
