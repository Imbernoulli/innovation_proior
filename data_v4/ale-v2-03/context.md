# Factory Job-Shop Scheduling

## Research question

A factory must process `n` jobs on `m` machines. Each job is a fixed **chain of `m` operations** that
must be performed in a given order; operation `t` of job `j` runs on a specified machine and occupies
it for a given processing time. Every job visits every machine **exactly once**, so each machine has
exactly `n` operations queued for it (one per job). A machine processes one operation at a time with no
preemption, and an operation cannot start until both its job-predecessor has finished *and* the machine
is free. The task is to decide, for every machine, the **order** in which it runs its `n` operations so
that the **makespan** — the completion time of the last operation, i.e. the time the whole factory goes
idle — is as small as possible.

Phrased structurally: fix the per-job chain arcs and choose, on each machine, a total order of its
operations (a *disjunctive* orientation). Those choices induce a directed graph on the `n·m`
operations; the makespan is the **length of the longest source-to-sink path** in that graph. Minimizing
it is the classic `J||C_max` job-shop scheduling problem — NP-hard, with no efficient exact solution at
this scale, judged by *how short* the makespan is rather than by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first line holds two integers `n m` (`15 ≤ n ≤ 30` jobs, `10 ≤ m ≤ 20`
  machines). Then `n` lines follow, line `j` (0-indexed) describing job `j` as `m` operation pairs
  `mach_0 proc_0 mach_1 proc_1 … mach_{m-1} proc_{m-1}`: operation `t` of job `j` runs on machine
  `mach_t` for `proc_t` time units (`1 ≤ proc_t ≤ 99`). The machine list of every job is a permutation
  of `0…m-1`, so each machine receives exactly one operation from each job.
- **Output (stdout):** `m` lines, line `k` (0-indexed, in increasing `k`) giving the order in which
  machine `k` processes the jobs — a **permutation of `0 … n-1`** (the `n` job indices). Line `k` means:
  machine `k` first runs the operation of `order[k][0]`, then that of `order[k][1]`, and so on.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff (1) the output is exactly `m` lines, each a permutation of `{0,…,n-1}`,
**and** (2) the chosen machine orders together with the job chains form an **acyclic** precedence graph
(a consistent set of start times exists — no deadlock where two machines each wait on the other). Wrong
line/token count, an out-of-range or repeated index, a non-integer token, a missing file, or a cyclic
(deadlocked) orientation are all **infeasible**.

## Background

The decision variables are the machine orders; everything else (which operation precedes which inside a
job, and each processing time) is fixed by the instance. This is exactly the **disjunctive-graph** view
of the job shop: nodes are operations, *conjunctive* arcs encode the immutable job chains, and on each
machine we must *orient* the clique of its operations into a line. Once oriented, start times are forced
by the longest path, so the search space is "one permutation per machine" and the cost of any point in
it is a single longest-path computation.

Several approaches sit on the table before committing:

- **List scheduling (the construction baseline).** Build a schedule greedily: repeatedly dispatch the
  operation that can *start earliest* (a non-delay / active-schedule rule), which fixes the machine
  orders as a side effect. `O(n·m·n)` and always feasible (an active schedule is acyclic by
  construction), but it commits early to orderings it cannot revisit and typically lands 20–60% above
  optimal.
- **Permutation-encoded metaheuristics** (GA / SA on operation-priority lists). General but they spend
  most evaluations far from the critical path and waste the structure of the problem.
- **Critical-path local search.** The makespan is a longest path; only operations *on* a critical path
  can be reordered to shorten it. Swapping two adjacent operations that are *not* on a critical path
  cannot reduce the makespan. So the search should propose moves only along critical paths — this is
  the lever that makes local search both fast and effective for the job shop.

The decisive structure is the **N5 critical-block neighbourhood** of Nowicki & Smutnicki. A critical
path, cut into maximal runs of operations that share a machine ("blocks"), can only be shortened by
reordering operations *within* a block, and (a theorem of the disjunctive graph) it suffices to consider
swapping the **first two** or the **last two** operations of each block. These swaps are provably
**makespan-non-increasing-or-neutral candidates that never create a cycle**, so feasibility is automatic.
Combined with **tabu search** (forbid immediately reversing a just-reversed arc, with an aspiration
override when a move beats the incumbent) and **incremental longest-path re-evaluation that only touches
the critical structure**, this is the established strong-yet-simple metaheuristic for `J||C_max`.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  picks `n ∈ [15, 30]` and `m ∈ [10, 20]`, gives every job a uniformly random machine permutation, and
  draws each processing time uniformly in `[1, 99]` (the standard Taillard band). This size band is where
  list scheduling leaves substantial slack for critical-block search to recover while the longest-path
  makespan stays cheap to recompute.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted machine
  orders.
  - **Feasibility floor:** if the output is not exactly `m` permutations of `{0,…,n-1}`, **or** the
    induced precedence graph contains a cycle (deadlock), the score is **`0`**.
  - Otherwise compute the **makespan** `C` as the longest source-to-sink path in the disjunctive graph
    (job-chain arcs plus the machine-order arcs), via a topological longest-path pass. Let `B` be the
    makespan of the scorer's own deterministic **list-scheduling** baseline (an earliest-start non-delay
    schedule, recomputed inside the scorer so the reference is reproducible and independent of the
    solver). The score is

    ```
    score = round( 1 000 000 × B / C )      (feasible, C > 0)
    score = 0                               (infeasible)
    ```

    A higher score is better. The list-scheduling reference scores exactly `1 000 000`; a shorter
    makespan scores strictly more; a longer one scores less but stays positive.
- **Reported metric.** The mean score over a fixed seed set. A genuine critical-block tabu solver should
  land well above `1 000 000` (≈ 1.2–1.6× the list-scheduling reference on these instances); the trivial
  *identity-order* output (every machine runs jobs `0,1,…,n-1`) is feasible but poor and scores only
  ~150 000 — the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing `m` feasible machine
orders to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    // M[j][t], P[j][t]: machine and processing time of operation t of job j.
    vector<vector<int>> Mc(n, vector<int>(m)), Pr(n, vector<int>(m));
    for (int j = 0; j < n; j++)
        for (int t = 0; t < m; t++)
            scanf("%d %d", &Mc[j][t], &Pr[j][t]);

    // A feasible answer is ANY set of m permutations of 0..n-1 that does not
    // deadlock. The non-delay list schedule (dispatch the op that can start
    // earliest) is always acyclic, so start there to guarantee a legal output.
    vector<vector<int>> order(m);   // order[k] = job order on machine k

    // TODO heuristic: build the disjunctive graph, compute the makespan as the
    // longest source->sink path, find a critical path, decompose it into machine
    // BLOCKS, and run tabu search over the N5 neighbourhood (swap the first-two /
    // last-two operations of each critical block), recomputing heads/tails to
    // evaluate each move and keeping the best acyclic schedule under a ~2s budget.

    string out;
    for (int k = 0; k < m; k++) {
        for (int i = 0; i < (int)order[k].size(); i++)
            out += to_string(order[k][i]) + (i + 1 < (int)order[k].size() ? ' ' : '\n');
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
