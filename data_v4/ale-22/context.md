# Job-Shop Scheduling (makespan)

## Research question

You run a workshop with `m` machines and a queue of `n` jobs. Each job is a fixed recipe: a sequence
of `m` operations that must be done **in order**, the `k`-th operation of job `j` running on a
specified machine for a given duration. Every job visits every machine exactly once. A machine
processes at most one operation at a time, and once an operation starts it runs to completion (no
preemption). You must decide, **for each machine, the order in which it processes the operations
queued on it**. The *makespan* is the time the last operation finishes. You want to **minimize the
makespan**.

This is the classic `n x m` job-shop scheduling problem (JSP). It is strongly NP-hard, has no known
closed-form or efficiently-computable optimum, and is judged here by a continuous score: a better
(smaller) makespan scores higher, and any infeasible schedule scores `0`.

## Input / output contract

- Input (stdin): the first line is `n m` (`10 <= n <= 20` jobs, `8 <= m <= 15` machines). Then `n`
  job rows follow. Row `j` lists the `m` operations of job `j` as `m` consecutive
  `machine duration` pairs (so `2*m` integers): the `k`-th pair `(M[j][k], D[j][k])` means operation
  `k` of job `j` runs on machine `M[j][k]` for `D[j][k]` time units (`1 <= D <= 200`). Within a row
  every machine index in `[0, m)` appears exactly once.
- Output (stdout): a **machine order** — exactly `m` lines. Line `i` is the order in which machine `i`
  processes its operations, written as a permutation of the job indices whose operation lies on
  machine `i`. Because every job visits every machine exactly once, the job index on machine `i`'s
  line uniquely identifies the operation `(j, k)` with `M[j][k] == i`. Each of the `m` lines must be a
  permutation of `{0, 1, ..., n-1}`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with `n = 2`, `m = 2`, job 0 = `[(m0,3),(m1,4)]`, job 1 = `[(m1,2),(m0,5)]`, the output

```
0 1
1 0
```

means machine 0 runs job 0 then job 1, and machine 1 runs job 1 then job 0; decoding gives a makespan
of `9`.

## Background

The schedule is fully determined by the per-machine orders, but evaluating one — and improving one —
is where the structure lives. The standard model is the **disjunctive graph**: a node per operation,
fixed *conjunctive* arcs chaining each job's operations in order, and, once we fix a machine order,
*disjunctive* arcs chaining consecutive operations on each machine. Start times are the **longest
path** to each node; the makespan is the longest path in the whole graph. A machine order is feasible
iff this graph is **acyclic** (a cycle is a scheduling deadlock).

Two families of approach are on the table before committing to one.

- **Dispatch / priority rules.** Simulate the shop, and whenever a machine is free pick the waiting
  operation with the highest priority (Shortest Processing Time, Most Work Remaining, etc.). This is
  `O(n*m)`, always feasible, and gives a decent schedule in one pass — but a single greedy pass leaves
  large bottleneck gaps and is well known to sit far from optimal.
- **Neighborhood search over machine orders.** Start from a feasible order and repeatedly apply small
  changes — swapping the processing order of two operations on a machine — keeping improvements. The
  obvious version "try every adjacent swap on every machine" is `O(n*m^2)` re-evaluations per step and
  most swaps change nothing, because a swap can only shorten the makespan if it touches a longest
  (critical) path.

The non-obvious lever is the **critical-path (critical-block) neighborhood**: from the disjunctive
graph, extract a longest path realizing the makespan; the only swaps worth trying are of two
machine-consecutive operations that lie on that path. Swapping two adjacent operations of a critical
block can never create a cycle, so every move stays feasible, and the neighborhood is tiny (length of
the critical path) instead of `O(n*m^2)`. This is the established engine of strong JSP heuristics.

## Evaluation settings

A deterministic local scorer (`verify/score.py`) reads the instance and a candidate solution and
prints an integer score; higher is better.

- **Feasibility (any violation floors the score to 0).** The output must parse as exactly `m` lines,
  line `i` a permutation of the job set on machine `i` (here all `n` jobs). The implied disjunctive
  graph — job-precedence arcs plus the given machine-order arcs — must be **acyclic**: the scorer
  builds start times by a longest-path (Kahn topological) pass and rejects the solution if not all
  operations can be ordered (a deadlock cycle). As a defensive re-check it then verifies job
  precedence (`start(j,k) >= start(j,k-1) + D(j,k-1)`) and machine exclusivity (operations on a
  machine, in the stated order, occupy disjoint time intervals). Any failure scores `0`.
- **Makespan (lower is better).** For a feasible schedule, `start(j,k)` is the longest-path value in
  the disjunctive graph and the makespan is `max over (j,k) of start(j,k) + D(j,k)`.
- **Score (normalized, higher better).** The scorer recomputes the makespan of a deterministic
  **Shortest-Processing-Time (SPT) list-scheduling baseline** (`base`) and reports
  `score = round(1_000_000 * base / max(1, makespan))`. The SPT baseline scores about `1_000_000`; a
  shorter makespan scores more. Infeasible output scores `0`.
- **Instances.** `verify/gen.py <seed>` draws `n` in `[10, 20]`, `m` in `[8, 15]`, and for each job a
  random permutation of the machines (every job visits every machine once) with durations in
  `[10, 99]`, where about 18% of operations are made "heavy" (`[100, 200]`) to create bottleneck
  machines. This square-ish, mixed-duration regime is exactly where a critical-path local search beats
  a one-pass dispatch rule by a wide margin.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<vector<int>> mc(n, vector<int>(m)), du(n, vector<int>(m));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++)
            scanf("%d %d", &mc[j][k], &du[j][k]);

    // TODO: choose, for each machine, the order in which it runs its operations
    // so as to MINIMIZE the makespan (last completion time).
    // Idea: model as a disjunctive graph; decode start times by longest path;
    // local-search the machine orders, restricting swaps to adjacent operations
    // on a CRITICAL path (a swap off the critical path cannot help) and keeping
    // the graph acyclic so every schedule stays feasible.

    // Print a feasible solution: m lines, line i = machine i's job order.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) printf("%d%c", j, j + 1 < n ? ' ' : '\n');
    }
    return 0;
}
```
