# Production Line Scheduling (permutation flow shop, makespan)

## Research question

A factory runs a single production line of `m` machines wired in series:
`machine 0 -> machine 1 -> ... -> machine m-1`. There are `n` jobs to produce.
**Every job must visit all `m` machines, in that fixed order**, and job `j` needs
`p[j][k]` time units on machine `k`. A machine processes one job at a time, and
once it starts a job it runs that job to completion (no preemption).

The line is operated as a **permutation flow shop**: we pick *one* ordering of
the `n` jobs and feed the jobs into machine 0 in that order; no job is allowed to
overtake another further down the line, so the *same* order is used on every
machine. The only decision is therefore a single permutation `pi` of the jobs.

We want to finish the whole batch as early as possible: **minimize the makespan**,
the completion time of the last job on the last machine. This is the classic
`Fm | prmu | Cmax` problem. It is strongly NP-hard for `m >= 3`, has no known
closed-form optimum at these sizes, and is judged by a continuous score that
rewards a smaller makespan. The lever is purely the job order; nothing else is
editable.

## Input / output contract

- **Input (stdin), the instance.**
  - Line 1: two integers `n m` with `40 <= n <= 80` jobs and `5 <= m <= 20`
    machines.
  - Next `n` lines: line `j` (0-based) holds `m` integers
    `p[j][0] p[j][1] ... p[j][m-1]` with `1 <= p[j][k] <= 99` — the processing
    time of job `j` on each machine, in machine order `0..m-1`.
- **Output (stdout), the solution.**
  - A permutation of the `n` job ids `0..n-1`, whitespace-separated (any layout
    of spaces/newlines). The scorer also accepts an optional leading header token
    equal to `n` before the permutation; it is not required.
  - The permutation is interpreted as the order in which jobs enter machine 0.
- **Time limit:** 2 seconds wall-clock. **Memory:** 256 MB.

## Background

The makespan of a fixed order `pi` is computed by the flow-shop completion-time
recurrence. With `C[i][k]` the completion time of the `i`-th job of `pi` on
machine `k`:

```
C[0][0] = p[pi[0]][0]
C[i][0] = C[i-1][0] + p[pi[i]][0]
C[0][k] = C[0][k-1] + p[pi[0]][k]
C[i][k] = max(C[i-1][k], C[i][k-1]) + p[pi[i]][k]
Cmax    = C[n-1][m-1]
```

Two reference approaches frame the problem before committing to one:

- **Dispatch rules (SPT / LPT).** Order jobs by total processing time
  (shortest-processing-time first, or longest first). This is `O(n log n)`,
  trivially feasible, and the natural baseline — but it ignores the interaction
  between machines and leaves large makespans on the table.
- **NEH + insertion local search.** NEH (Nawaz–Enscore–Ham) is the strongest
  *constructive* heuristic for this objective: sort jobs by descending total
  time, then insert them one at a time into the best position of the partial
  order. A naive NEH that re-simulates the whole partial schedule for each of the
  `O(n)` candidate positions of each job costs `O(n^3 m)` overall, which is the
  bottleneck the strong method removes.

The open question is how to evaluate the *insertion* move — "where in the current
order does this job go?" — cheaply enough to run many thousands of
reconstructions inside two seconds.

## Evaluation settings

- **Scoring (what the judge reports; higher is better).** A solution is feasible
  iff its body is exactly a permutation of `{0,1,...,n-1}` (the right count, every
  id in range, all distinct). Then

  ```
  score = 0                         if the solution is infeasible
  score = round(10^9 / Cmax)        otherwise,
  ```

  where `Cmax` is the makespan of the permutation under the recurrence above. A
  smaller makespan yields a strictly larger score; any malformed, wrong-length,
  out-of-range, duplicate, or non-permutation output **floors the score to
  exactly 0**. The underlying objective is to **minimize `Cmax`**; the
  `10^9 / Cmax` wrapper just turns it into a bounded, maximize-style continuous
  score with a hard feasibility floor.

- **Instances.** A frozen generator (`gen.py`) draws `n in [40,80]`,
  `m in [5,20]`, and each `p[j][k]` independently uniform in `[1,99]` — exactly
  Taillard's standard flow-shop regime, so no machine trivially dominates.
  Everything — `n`, `m`, every processing time — is a deterministic function of an
  integer seed. We report the mean score over a fixed seed set (seeds `1..20`),
  each rung run on the *same* instances so numbers are directly comparable. The
  trivial baseline is the **shortest-processing-time (SPT)** dispatch order.

## Code framework

A single self-contained C++17 program that reads the instance on stdin and writes
a feasible permutation on stdout within the time budget.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> P(n, vector<int>(m));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) cin >> P[j][k];

    // A feasible solution is ANY permutation of 0..n-1.
    // Start from the identity so we can always print something legal.
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 0);

    // TODO: heuristic. Improve `perm` to minimise the flow-shop makespan,
    // e.g. NEH construction with Taillard's accelerated insertion (head/tail
    // completion-time arrays evaluate all n insertion positions in O(n*m)),
    // then Iterated Greedy (destruct d jobs, greedily reinsert, accept by a
    // constant-temperature rule) with an insertion-neighbourhood local search.

    for (int i = 0; i < n; i++) cout << perm[i] << (i + 1 < n ? ' ' : '\n');
    return 0;
}
```
