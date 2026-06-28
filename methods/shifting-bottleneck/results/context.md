## Research question

We are given a job shop: a set of `n` jobs and `m` machines. Each job is a fixed *route* — an
ordered sequence of operations, one per machine in this classical case — where operation `i` must be
processed on a prescribed machine for a fixed, uninterruptible duration `d_i`. A machine handles one
operation at a time. We must decide, for every machine, the order in which it processes the
operations assigned to it, so as to minimize the **makespan** — the time at which the last job
finishes.

Stated as a math program with `t_i` the (variable) start time of operation `i`, and `0`, `n` two
dummy operations "start" and "finish":

    min  t_n
    s.t. t_j - t_i >= d_i                         for (i,j) in A          (job precedence)
         t_i >= 0                                  for i in N
         t_j - t_i >= d_i   OR   t_i - t_j >= d_j  for (i,j) in E_k, k in M  (machine capacity)

The first family of constraints (`A`) fixes the order *within* each job and is non-negotiable. The
third family is the hard part: for every pair of operations `(i,j)` that share a machine, *one* of
them must come first, but which one is ours to choose. Those *disjunctions* are the entire
combinatorial difficulty.

The question is how to choose the machine orderings — orienting the disjunctive edges — so as to
minimize the makespan on instances the size of a real shop (on the order of ten machines and
ten-to-fifty jobs).

## Background

**The disjunctive graph (Roy & Sussman 1964).** The standard representation of this problem is a
graph `G = (N, A, E)`. Nodes `N` are the operations (plus the two dummies `0`, `n`). The
*conjunctive* arcs `A` are the job-precedence relations: a directed arc `i -> j` whenever `j`
immediately follows `i` in some job's route, carrying weight `d_i`. The *disjunctive* edges `E` are
undirected pairs `{i,j}` for every two operations on the same machine; `E` partitions into cliques
`E_k`, one per machine `k`, since every pair on machine `k` conflicts. To *schedule* is to *orient*
every disjunctive edge — decide `i -> j` or `j -> i`. A choice of orientation for one machine's
clique `E_k` is a *selection* `S_k`; it is *acyclic* exactly when it encodes a valid linear order of
that machine's operations. A *complete selection* `S = ∪_k S_k` orients all machines.

**Makespan equals a longest path.** Once a complete acyclic selection `S` is fixed, the disjunctive
graph becomes an ordinary directed acyclic graph `D_S = (N, A ∪ S)`, and the earliest each operation
can start is the length of the longest path reaching it from the source `0`. The makespan is then
the longest path from `0` to `n` in `D_S`. This is the load-bearing fact: scheduling = choosing an
acyclic orientation of all the machine cliques to minimize the longest path. Computing the longest
path in a DAG is linear; the difficulty is purely in the orientation.

**Hardness.** Job shop makespan minimization is NP-hard (Garey & Johnson 1979 place machine
sequencing among the strongly NP-hard problems). It is notoriously hard *in practice*, not just in
theory: by the mid-1980s one could solve traveling-salesman instances of 300–400 cities or
set-covering problems with over 100,000 variables exactly, yet a ten-job/ten-machine job shop was
routinely beyond reach. The single 10×10 instance of Muth & Thompson (1963) had resisted every
algorithm for over twenty years; its optimum of 930 was only settled around 1986.

**Single-machine relaxation with heads and tails.** A single machine with release dates and
delivery times asks for an order in which job `j` cannot start before its release (head) `r_j`,
runs for `p_j`, and then carries a delivery tail `q_j`; the objective is
`min max_j(C_j + q_j)`, the problem `1|r_j,q_j|C_max`. It is equivalent to maximum lateness
`1|r_j|L_max` after choosing a fixed offset `H` and due date `f_j = H - q_j`, because
`C_j - f_j = C_j + q_j - H`; subtracting the same `H` does not change the best order. The problem
is itself NP-hard in the strong sense, but small instances solve in milliseconds by branch and
bound. A polynomially solvable *relaxation* of it — allow preemption, `1|r_j,pmtn|L_max` — is solved
exactly by the preemptive earliest-due-date rule and gives a valid lower bound on the
non-preemptive optimum, since preemption can only help.

**Active and non-delay schedules; priority dispatching.** Giffler & Thompson's classical result is
that an optimal schedule can always be found among *active* schedules — ones in which no operation
can be shifted earlier without delaying some other operation. Greedy *priority dispatching rules*
build one active (often non-delay) schedule in a single pass: repeatedly, among the operations
currently eligible, pick one by a priority and commit it, never reconsidering. This is fast and the
backbone of practical scheduling.

## Baselines

**Priority dispatching rules.** The dominant practical methods. At each step a rule chooses the next
operation to dispatch from the set of currently schedulable ones by a local criterion: SPT (shortest
processing time), MWKR (most work remaining on the job), FCFS, LST (late start time), MINSLK
(minimum slack), and others. The eligible set is restricted so the result is an active or non-delay
schedule. These are one-pass greedy procedures: decisions are based on what looks locally best and
are never undone.

**Implicit enumeration on the disjunctive graph (Balas 1969).** Machine sequencing via the
disjunctive graph, solved by branch and bound that orients disjunctive edges and prunes by longest
paths. A machine `k` is *critical* for a selection `S` if its selection `S_k` contributes an arc to
a longest (critical) path in `D_S`; any schedule strictly better than `S` must reverse at least one
arc on every critical path.

**Branch and bound for the one-machine problem (McMahon & Florian 1975; Carlier 1982).** Exact
algorithms for `1|r_j,q_j|C_max`. McMahon & Florian solved instances up to ~80 jobs; Carlier (1982)
reports solving up to 1,000 jobs. Carlier's scheme: a Schrage-style dispatch heuristic (at the
current time, among released jobs pick the one with largest tail `q_j`) produces a feasible schedule
and its critical path; from a *critical block* `J` of jobs one reads a lower bound
`h(J) = min_{i∈J} r_i + Σ_{i∈J} d_i + min_{i∈J} q_i`, and a single "critical job" `j(k)` whose
placement before or after `J` defines a two-way branch. Trees stay small (rarely above `2n` nodes).

## Evaluation settings

The natural yardstick is the bank of benchmark job-shop instances accumulated over the preceding
decades. The Fisher–Thompson instances (Muth & Thompson 1963) include the small 6×6 and the
notorious 10×10. Lawrence (1984) contributes a graded family of instances spanning 5–10 machines and
10–30 jobs, with random machine routes and processing times drawn uniformly from `[5,99]`. Instances
range from a handful of operations up to several hundred (e.g. 10 machines × 50 jobs = 500
operations). The metric is makespan (lower is better), reported alongside the best lower bound
available for the instance and the wall-clock time to produce the schedule. The relevant comparison
is against the priority-dispatching rules in both straight and randomized form, run on the same
instances, at comparable computing budgets.

## Code framework

The deliverable is a single self-contained C++17 program. It reads a job-shop instance from stdin in
the standard OR-Library format: first line `n m` (jobs, machines), then `n` job rows, each containing
`m` pairs `machine duration` in processing order. It writes one makespan value followed by a newline
to stdout. Operations may be numbered `op = job*m + position`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) {
        return 0;
    }

    vector<vector<int>> machine(n, vector<int>(m));
    vector<vector<long long>> duration(n, vector<long long>(m));

    for (int job = 0; job < n; ++job) {
        for (int position = 0; position < m; ++position) {
            cin >> machine[job][position] >> duration[job][position];
        }
    }

    long long makespan = 0;
    // TODO: Compute the requested makespan from the parsed instance.

    cout << makespan << '\n';
    return 0;
}
```
