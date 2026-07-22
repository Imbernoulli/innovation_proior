# Pit-Lane Arm Ballet: Minimal-Tick Service Plan

## Problem
A pit crew's `K` identical robot arms service one car during a single stop. There are `N`
service tasks. Task `i` has an integer duration `d_i` (ticks) and works in one of **6**
physical sectors (0..5) arranged around the car (e.g. front-left tyre, rear jack, fuel port,
...). Two structural rules govern any valid plan:

- **Precedence DAG.** Some tasks must finish before others can start (e.g. the jack must lift
  before a wheel comes off). You are given a set of directed edges `(u, v)`, always with
  `u < v`, meaning task `v` may not *start* before task `u` has *finished*.
- **Sector mutex.** Only one robot arm fits inside a given sector's physical envelope at a
  time. Regardless of how many arms exist, at most one task belonging to a given sector may
  be "in progress" at any tick — two same-sector tasks can never overlap in time, even on
  different arms.

Each arm processes at most one task at a time (no overlap on the same arm either). A plan is
a **straight-line schedule**: an assignment, for every task, of which arm runs it and the
exact tick it starts on. Time is discrete and exact (ticks), and a task occupies
`[start_i, start_i + d_i)`.

## Input (stdin)
```
N K
d_1 d_2 ... d_N
s_1 s_2 ... s_N
E
u_1 v_1
...
u_E v_E
```
`d_i` (1..9) is task `i`'s duration, `s_i` (0..5) its sector. Each edge has `u < v`, so
input order is already a valid topological order. `1 <= N <= 120`, `1 <= K <= 5`.

## Output (stdout)
Exactly `N` integer pairs, in task-index order (whitespace/newlines both fine):
```
arm_1 start_1
arm_2 start_2
...
arm_N start_N
```
`arm_i` in `[0, K)`, `start_i >= 0`.

## Feasibility
A plan is feasible iff **all** hold:
1. For every edge `(u, v)`: `start_v >= start_u + d_u`.
2. For every arm, its assigned tasks have pairwise non-overlapping `[start, start+d)`
   intervals.
3. For every sector, ALL tasks in that sector (across every arm) have pairwise
   non-overlapping `[start, start+d)` intervals.

Any parse error, out-of-range arm/start, non-finite/non-integer token, wrong token count, or
violation of 1-3 scores `Ratio: 0.0`.

## Objective
Minimize the makespan `F = max_i (start_i + d_i)` — the exact tick the last task finishes.

## Scoring
The checker builds its own reference plan: run every task strictly sequentially, one at a
time, in input order, on a single arm (always feasible, since input order already respects
precedence and a fully sequential plan trivially respects both arm capacity and sector
mutex). Its makespan is `B = sum(d_i)`. With your makespan `F`:
```
Ratio = min(1, B / (10*F))
```
Matching the sequential reference exactly scores `0.1`; a 10x-shorter makespan caps at
`1.0`. Because sector mutex only bounds *concurrent* use, not *order*, a smart plan can pack
far more true parallelism than the sequential reference — but the true minimum makespan is
not known to be reachable in polynomial time, so real headroom above any given plan remains.

## Constraints
- `1 <= N <= 120`, `1 <= K <= 5`, 6 sectors (0..5), `1 <= d_i <= 9`.
- Edges always satisfy `u < v`. Time limit 5s, memory 512MB, deterministic scoring only.

## Example
Suppose 3 tasks, `d = [4, 2, 3]`, all sector 0, no edges, `K = 2`. Sector mutex forces them
serial regardless of arms: a feasible optimum is starts `[0, 4, 6]`, `F = 9`. The sequential
reference gives the same `B = 9`, so this plan scores `min(1, 9/90) = 0.1`. (This tiny example
is illustrative only — real instances mix sectors and precedence so parallelism pays off.)
