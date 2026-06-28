# Machine Assignment with Sequence-Dependent Setups

## Research question

You schedule `n` jobs on `M` identical parallel machines. Job `j` has a processing duration `d_j`
and a **type** `c_j` in `[0, T)`. Each machine processes the jobs assigned to it in some order; the
order matters because of **changeovers**:

- before its **first** job (of type `a`) a machine pays an initial setup `init[a]` (set-up from a
  clean / ground state);
- between two **consecutive** jobs of types `a` then `b` on the same machine, the machine pays a
  setup `s[a][b]` from a `T x T` matrix.

The setup matrix is sequence- and type-dependent, asymmetric (`s[a][b] != s[b][a]` in general), and
non-metric: same-type changeovers are cheap, cross-type changeovers are moderate to large. You must
assign **every** job to exactly one machine and choose each machine's order so as to **minimize the
total over all machines of (sum of durations + all setups)**. This is an assignment problem wrapped
around, on each machine, an open asymmetric **TSP-on-types** — it is NP-hard, has no known closed
form, and is judged by a continuous score.

## Input / output contract

- Input (stdin):
  - Line 1: `n M T` (`1 <= n <= 260`, `4 <= M <= 10`, `4 <= T <= 9`).
  - Line 2: `n` durations `d_0 ... d_{n-1}` (`10 <= d_j <= 100`).
  - Line 3: `n` types `c_0 ... c_{n-1}` (each in `[0, T)`).
  - Line 4: `T` initial setups `init_0 ... init_{T-1}`.
  - Next `T` lines: row `a` is `s[a][0] ... s[a][T-1]`, the setup matrix.
- Output (stdout): exactly `M` lines, one per machine. Line `m` (0-based machine `m`) is
  `k j_1 j_2 ... j_k`: the count `k >= 0` of jobs on machine `m` followed by their `0`-based indices
  **in the order they run** on that machine. A machine may be empty (`k = 0`). The set of all listed
  jobs must be exactly `{0, ..., n-1}` (every job exactly once).
- Time limit: 2 seconds. Memory: 256 MB.

## Background

Two families of approach are on the table before committing to one.

- **Balanced round-robin (the trivial baseline).** Send job `j` to machine `j mod M` in input order.
  This balances counts but ignores types entirely, so almost every adjacency is a cross-type
  changeover — it pays the setup matrix at nearly its average value on every edge. It is the reference
  the score normalizes against, and it is what any type-aware method should crush.
- **Assignment + per-machine sequencing.** The processing time `sum_j d_j` is a **constant** (every
  job runs exactly once on some machine), so the only thing under our control is the total **setup**
  cost. Setup depends only on the type at each position and its predecessor. Hence each machine's
  contribution is the cost of an **open path** through the types of its jobs, started from a virtual
  "ground" type whose out-edge to type `a` costs `init[a]`: an open, asymmetric **TSP on types**,
  embedded inside the choice of which jobs go to which machine.

The non-obvious lever is to recognize this **"TSP-on-types per machine inside an assignment"**
structure and to optimize it with **two coupled local-search neighborhoods whose cost deltas are
`O(1)`** — so the metaheuristic can take millions of steps and the per-step work does not grow with
`n`.

## Evaluation settings

A deterministic local scorer (`verify/score.py`) reads the instance and a candidate solution and
prints an integer score; higher is better.

- **Feasibility (any violation floors the score to 0):** the output is exactly `M` token-bearing
  lines; each line is a non-negative `k` followed by exactly `k` integers; every listed job index is
  in `[0, n)`; and the multiset of all listed jobs is exactly `{0, ..., n-1}` (no job unassigned, no
  job duplicated). Any failure makes the whole solution INFEASIBLE and scores `0`.
- **Cost of a feasible solution (lower is better).** For each machine `m` with order
  `(j_1, ..., j_k)`,
  `load(m) = sum_t d[j_t] + init[c[j_1]] + sum_{t=2..k} s[c[j_{t-1}]][c[j_t]]`
  (an empty machine has `load 0`). The objective is the **total**:
  `cost = sum_m load(m)`. (The stated objective is total completion + setup time; the makespan /
  max-load variant is **not** used.)
- **Score (normalized, higher better).** The scorer recomputes the cost of the deterministic
  **balanced round-robin** baseline (`baseline_cost`) and reports
  `score = round(1_000_000 * baseline_cost / max(1, solver_cost))`. The baseline scores about
  `1_000_000`; a lower-cost schedule scores more. Infeasible output scores `0`.
- **Instances.** `verify/gen.py <seed>` draws `n in [120, 260]`, `M in [4, 10]`, `T in [4, 9]`;
  durations in `[10, 100]`; types with a **skewed** multiplicity (some common, some rare); `init[a]`
  in `[15, 60]`; and an asymmetric, non-metric setup matrix with cheap same-type entries
  (`s[a][a] in [0, 8]`), moderate-to-large cross-type entries (`[20, 90]`), and an occasional
  expensive pair. The skew and asymmetry make grouping same-type jobs and ordering the type-runs well
  the dominant lever — exactly the regime where a type-blind round-robin is wasteful.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, M, T;
    if (scanf("%d %d %d", &n, &M, &T) != 3) return 0;
    vector<int> d(n), c(n);
    for (int j = 0; j < n; j++) scanf("%d", &d[j]);
    for (int j = 0; j < n; j++) scanf("%d", &c[j]);
    vector<int> init(T);
    for (int a = 0; a < T; a++) scanf("%d", &init[a]);
    vector<int> s(T * T);
    for (int a = 0; a < T; a++)
        for (int b = 0; b < T; b++) scanf("%d", &s[a * T + b]);

    // TODO heuristic: assign every job to a machine and order each machine so as
    // to minimize  sum_m [ sum durations + init[first type]
    //                      + sum consecutive s[prev type][type] ].
    // Idea: durations are constant -> minimize total SETUP. Each machine is an
    // open asymmetric TSP-on-types from a "ground" start; optimize assignment +
    // order with O(1)-delta relocate/swap moves under simulated annealing.

    // Print a FEASIBLE schedule: M lines "k j_1 ... j_k", every job exactly once.
    for (int m = 0; m < M; m++) {
        // trivial round-robin partition so the output always parses:
        printf("0\n");
    }
    return 0;
}
```
