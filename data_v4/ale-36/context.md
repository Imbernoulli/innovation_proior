# Sequencing to Minimize Total Weighted Tardiness

## Research question

A single machine must process `n` jobs, one at a time, without preemption and without idle
time. Job `i` has a processing time `p_i`, a weight (penalty rate) `w_i`, and a due date `d_i`.
You choose the order in which the jobs run. If the jobs are run in some permutation, the
**completion time** `C_i` of a job is the sum of the processing times of every job up to and
including it in that order. Its **tardiness** is `T_i = max(0, C_i - d_i)` — how late it finishes,
or `0` if it finishes on time. The cost of an order is the **total weighted tardiness**

```
WT = sum over i of  w_i * max(0, C_i - d_i).
```

The task is to output a permutation of the job ids that makes `WT` as small as possible. This is
the classic single-machine weighted-tardiness sequencing problem, written `1 || sum w_j T_j` in
scheduling notation. It is strongly NP-hard, so there is no known efficient exact algorithm; the
problem is judged as a heuristic optimization with a continuous score.

## Input / output contract

- **Input (stdin):** the first token is `n` (`1 <= n <= 100`). Then `n` lines follow, line `i`
  (for `i = 0 .. n-1`) holding three integers `p_i w_i d_i`:
  - `p_i` — processing time, `1 <= p_i <= 100`,
  - `w_i` — weight, `1 <= w_i <= 10`,
  - `d_i` — due date, `1 <= d_i <= sum of all p`.
- **Output (stdout):** a permutation of the job ids `0 .. n-1` — each id printed exactly once, one
  id per line (whitespace-separated is also accepted). The order of the ids is the order in which
  the machine runs the jobs.
- **Time limit:** 2 seconds. **Memory:** 256 MB.

A solution is **feasible** iff the printed ids are exactly a permutation of `0 .. n-1` (each id in
range, each appearing once). Anything else — a missing id, a duplicate, an out-of-range token, the
wrong count, a non-integer — is infeasible.

## Background

`1 || sum w_j T_j` is one of the most-studied single-machine scheduling problems. The exact methods
(branch and bound, dynamic programming) blow up past a few tens of jobs, so practice relies on a
two-stage heuristic recipe that is the established strong baseline for this structure:

- **Construction by a composite dispatching rule.** The standard choice is the **Apparent
  Tardiness Cost (ATC)** rule of Rachamadugu and Morton. Scheduling forward in time, at each step
  it picks the unscheduled job maximizing the priority index

  ```
  I_j(t) = (w_j / p_j) * exp( - max(0, d_j - p_j - t) / (k * p_bar) ),
  ```

  where `t` is the current time, `p_bar` is the average processing time of the remaining jobs, and
  `k` is a lookahead parameter. ATC blends the weighted-shortest-processing-time term `w_j / p_j`
  (good when everything is already late) with an exponential urgency term on the slack `d_j - p_j - t`
  (good when due dates still matter). `k` is not universal, so a few values are tried and the best
  construction kept.

- **Improvement by local search.** Simple dispatch is then polished with neighbourhood moves:
  **adjacent pairwise interchange** (swap two neighbours when it lowers cost — this is exactly the
  move sanctioned by the classical adjacency dominance rule) and **job insertion / reinsertion**
  (lift one job out and drop it into the best slot). Insertion is the strong move because a single
  job can travel far. Wrapping the local search in an **iterated local search** (perturb the
  incumbent with a random block move, re-optimize, keep the best) is the established way to escape
  local optima within a fixed time budget.

The **lever** that makes the insertion neighbourhood affordable is incremental evaluation. A
straightforward "move job, re-score the whole order in `O(n)`" makes one local-search pass cost
`O(n^3)`. But relocating a job only changes the completion times of the jobs *between* its old and
new positions; everything before is untouched and everything after keeps its completion time
because the total processing time inside the moved interval is invariant. So each candidate move is
evaluated in `O(interval length)` by recomputing weighted tardiness over that window only, against
the same window's old cost. That is the difference between a toy that does one pass and a solver
that runs thousands of insertion sweeps inside the time budget.

## Evaluation settings

The score rewards low weighted tardiness on a continuous scale, with an infeasibility floor of `0`.

- The scorer reads the instance and the solution. If the solution is **not** a permutation of
  `0 .. n-1`, the score is **`0`** (the feasibility floor).
- Otherwise it computes completion times `C_i` along the printed order and the total weighted
  tardiness `WT = sum_i w_i * max(0, C_i - d_i)`.
- It also computes `WT_edd`, the weighted tardiness of the **earliest-due-date (EDD)** order (jobs
  sorted by `d_i`, ties broken by id). EDD is the trivial reference schedule.
- The reported score is

  ```
  score = 1e6 * (WT_edd + 1) / (WT + 1).
  ```

  Lower `WT` gives a higher score. The EDD order scores exactly `1e6`; any schedule strictly better
  than EDD scores above `1e6`; an order achieving `WT = 0` approaches `1e6 * (WT_edd + 1)`. The
  `+1` terms keep the ratio finite when a weighted tardiness is `0`.

**Instance generation.** Instances follow the standard weighted-tardiness benchmark scheme
(Crauwels-Potts-Van Wassenhove / OR-Library style), parameterized by a single integer seed.
`n` is drawn from `{40, 50, 60, 75, 90, 100}`; `p_i ~ U[1,100]`; `w_i ~ U[1,10]`. Due dates are set
from the total processing time `P = sum p_i` via a relative range of due dates `RDD` (from
`{0.2,0.4,0.6,0.8,1.0}`) and an average tardiness factor `TF` (from `{0.2,0.4,0.6,0.8}`):
`d_i ~ U[ P*(1 - TF - RDD/2), P*(1 - TF + RDD/2) ]`, rounded and clamped to `>= 1`. Tight,
bunched due dates (small `RDD`, large `TF`) make the instance hard. The generator, the scorer, and
the seed set are frozen.

## Code framework

A single self-contained C++17 program reads the instance from stdin and writes a feasible
permutation to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> p(n), w(n), d(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> w[i] >> d[i];

    // A trivially feasible order: identity (jobs in input order).
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);

    // TODO heuristic: ATC construction, then dominance-pruned insertion /
    // adjacent-interchange local search with O(window) incremental scoring,
    // wrapped in iterated local search under the time budget. Must keep `order`
    // a permutation of 0..n-1 at all times.

    for (int j : order) cout << j << '\n';
    return 0;
}
```
