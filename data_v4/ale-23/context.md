# Resource-Constrained Project Scheduling (RCPSP)

## Research question

You are given a project of `n` tasks (activities). Time is discrete and starts at `0`. There are `R`
**renewable** resources; resource `k` has a constant capacity `cap_k` available at *every* time unit.
Task `i` runs without interruption for `dur_i` time units and, while running over the half-open
interval `[s_i, s_i + dur_i)`, consumes `d_{i,k}` units of resource `k`. The tasks are linked by
**finish-to-start precedence**: a task may start only after all of its predecessors have finished,
i.e. `s_i >= s_j + dur_j` for every predecessor `j` of `i`.

Choose a start time `s_i >= 0` for every task so that

- every precedence constraint holds, and
- at no time `t` does the total demand of the tasks running at `t` exceed any resource capacity,

while **minimizing the makespan** `max_i (s_i + dur_i)` — the time at which the whole project
finishes. RCPSP is strongly NP-hard; there is no exact answer to read off, so a solution is judged by
a continuous score and an infeasible schedule scores `0`. The only lever is the heuristic that decides
*when* to start each task.

## Input / output contract

- Input (stdin):
  - first line: `n R` — number of tasks `n` (`0 <= n <= 200` in the seed regime, larger possible) and
    number of renewable resources `R` (`1 <= R <= 4`);
  - second line: `cap_1 cap_2 ... cap_R` — the integer capacity of each resource
    (`1 <= cap_k`), constant over all time;
  - then `n` lines, the `i`-th describing task `i` (tasks are **1-indexed**):
    `dur  d_{i,1} d_{i,2} ... d_{i,R}  p  pred_1 ... pred_p`
    where `dur` is the duration (`1 <= dur`), `d_{i,k}` is the per-unit demand of resource `k`
    (`0 <= d_{i,k} <= cap_k`), `p` is the number of predecessors, and `pred_1..pred_p` are their
    1-indexed task ids. Predecessors always have a strictly smaller id, so the precedence graph is a
    DAG.
- Output (stdout): `n` integers `s_1 ... s_2 ... s_n` (whitespace-separated, any layout), the start
  time of each task in **input order**.
- Time limit: about 2 seconds. Memory: 256 MB.

A schedule is the vector of start times. Task `i` occupies `[s_i, s_i + dur_i)` and books `d_{i,k}`
of each resource `k` over that window.

## Background

RCPSP is the canonical resource-constrained scheduling problem and one of the most studied NP-hard
combinatorial problems. Two reference points frame the design.

- **Earliest-start list scheduling.** Process tasks in a fixed priority order (here the input order);
  repeatedly take the next task whose predecessors are all scheduled and place it at the earliest time
  `>= its predecessor finish` at which every resource still has room for its whole duration. This is the
  **serial schedule-generation scheme (SGS)** driven by the trivial priority list. It is fast, always
  feasible, and is the standard baseline — but a poor list order leaves resources idle and inflates the
  makespan.
- **Priority-rule construction (e.g. by latest-finish or most-successors).** A smarter static priority
  order fed to the same SGS does better, but a single fixed rule still gets stuck: the best order is
  instance-specific, so no hand rule dominates.

The established strong family is **activity-list metaheuristics**: encode a schedule as a *priority
list* (a topological order of the tasks), decode it with the serial SGS, and search the space of lists
with simulated annealing, a genetic algorithm, or tabu search, typically boosted by **forward–backward
improvement (double justification)**. The list encoding is what makes the search clean: every list is a
topological order, so the SGS decode is always precedence- *and* resource-feasible — feasibility is
free, and the optimizer only ever moves between valid schedules.

## Evaluation settings

The harness is a faithful local RCPSP reproduction. A **generator** (`verify/gen.py`, parameter: an
integer seed) builds a random precedence DAG on a topological order, random durations and per-resource
demands, and resource capacities scaled so resources are genuinely scarce (a task often cannot start the
instant its predecessors finish). A **scorer** (`verify/score.py`) reads the instance and a solution and:

1. **Feasibility (any violation → score `0`):** the output parses as exactly `n` non-negative integers;
   every precedence `s_i >= s_j + dur_j` holds; and at no integer time `t` does the summed demand of the
   tasks running at `t` exceed any `cap_k` (checked by a per-resource difference array over the schedule
   horizon).
2. **Makespan (lower is better):** `makespan = max_i (s_i + dur_i)` (`0` if `n = 0`).
3. **Score (higher is better), normalized against the earliest-start baseline** the scorer recomputes
   itself (serial SGS with the input-order priority list):
   `score = round(1_000_000 * baseline_makespan / max(1, solver_makespan))`.
   The earliest-start baseline scores `~1_000_000`; a shorter (better) schedule scores more. An
   infeasible schedule scores `0`.

We report the mean score over a fixed seed set (seeds `1..20`). Each rung is run on the *same*
instances so the numbers are directly comparable. Nothing about the generator or the scoring is
learnable; the only lever is the schedule the solver emits.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible schedule
to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, R;
    if (!(cin >> n >> R)) return 0;
    vector<int> cap(R);
    for (auto &c : cap) cin >> c;
    vector<int> dur(n);
    vector<vector<int>> dem(n, vector<int>(R));
    vector<vector<int>> preds(n);
    for (int i = 0; i < n; i++) {
        cin >> dur[i];
        for (int k = 0; k < R; k++) cin >> dem[i][k];
        int p; cin >> p;
        preds[i].resize(p);
        for (int t = 0; t < p; t++) { int j; cin >> j; preds[i][t] = j - 1; }
    }

    // TODO heuristic: build a priority list (topological order), decode it with
    // the serial schedule-generation scheme into feasible start times, and search
    // the space of lists to minimize the makespan. Must always output a feasible
    // schedule.
    vector<int> start(n, 0);

    for (int i = 0; i < n; i++)
        cout << start[i] << (i + 1 < n ? ' ' : '\n');
    return 0;
}
```
