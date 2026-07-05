# Wind-Farm Turbine Micrositing: Maximum Wake-Free Completion

## Problem
A new offshore wind farm is laid out on an `n x n` grid of turbine slots. There are `n`
interchangeable turbine **models** `0, 1, ..., n-1` (each differs in hub height / rotor
diameter). To suppress wake interference, the utility enforces a strict rule:

> No two turbines of the **same model** may share a grid row or a grid column.

Equivalently, the completed layout must be a (partial) **Latin square** over the `n` models.

Some slots have already been surveyed and **pre-installed** (fixed givens); the rest are empty.
Your job is to install as many additional turbines as possible **without ever violating the
wake rule** and **without moving or removing any pre-installed turbine**. You may leave slots
empty.

Deciding whether a partial Latin square can be completed at all is NP-complete, and
maximizing the number of filled cells is harder still -- there is no simple closed-form
optimum. Greedy fills dead-end; smarter ordering and backtracking install strictly more.

## Input (stdin)
```
n
row 0: n tokens
row 1: n tokens
...
row n-1: n tokens
```
Each token is either a model index `0..n-1` (a pre-installed turbine) or `.` (an empty slot).
The givens form a valid partial Latin square (no givens conflict).

## Output (stdout)
Print the completed grid: `n` lines, each with `n` whitespace-separated tokens. Each token is
either a model index `0..n-1` (installed turbine) or `.` / `-1` (slot left empty).

## Feasibility
Your layout is rejected (score `0`) if any of these hold:
- the output does not contain exactly `n*n` tokens;
- any installed value is not an integer in `[0, n-1]`;
- any pre-installed given slot is altered or emptied;
- any model repeats within a row or within a column (among installed slots).

## Objective (maximize)
`F` = the number of installed (non-empty) slots in your feasible layout (givens included).

## Scoring
Let `B` = the number of pre-installed givens (the baseline "install nothing new" layout).
The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
so echoing only the givens scores `0.1`, and installing `10x` the givens caps the ratio at `1.0`.
Scoring is exact integer arithmetic and fully deterministic.

## Constraints
- `4 <= n <= 9` (small scale).
- Given density roughly `0.42 - 0.50` of the slots.

## Example
Input:
```
4
0 . . 1
. . . .
. . . .
1 . . 0
```
Here `B = 4` givens. Echoing the input unchanged gives `F = 4`, `Ratio = 0.1`. A full wake-free
completion installs all `16` slots (`F = 16`), giving `sc = min(1000, 100*16/4) = 400`,
`Ratio = 0.4`.
