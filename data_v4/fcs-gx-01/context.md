# Minimizing total weighted completion time on one machine

## Research question

You schedule `n` jobs on a single machine that runs exactly one job at a time, with no
idle gaps and no preemption. Job `i` needs `t[i]` units of processing time and carries a
weight `w[i]`. If you run the jobs in some order, the **completion time** `C[i]` of a job
is the moment it finishes — the sum of the processing times of every job that runs at or
before it. The cost of job `i` is `w[i] * C[i]`, and the cost of a schedule is the sum
over all jobs:

```
cost = sum_i  w[i] * C[i]
```

Find the ordering that **minimizes** the total weighted completion time, and output that
minimum cost.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then `n` lines (or, equivalently,
  `2*n` whitespace-separated tokens) follow, each giving two integers `t[i]` and `w[i]`
  (`1 <= t[i] <= 10^4`, `1 <= w[i] <= 10^4`).
- Output (stdout): a single line with the minimum achievable total weighted completion time.
- Time limit: 1 second. Memory: 256 MB.

Worked example. Input

```
3
1 1
3 5
2 1
```

The three jobs are `A=(t=1,w=1)`, `B=(t=3,w=5)`, `C=(t=2,w=1)`. The optimal order is
`B, A, C`: completion times `3, 4, 6`, cost `5*3 + 1*4 + 1*6 = 15 + 4 + 6 = 25`. The
answer is `25`. (Note that ordering by ascending processing time — `A, C, B` — gives
`1*1 + 1*3 + 5*6 = 34`, much worse; so the obvious key is not the right key.)

## Background

Two cheap-looking keys present themselves immediately, and both are wrong in general:

- **Sort by processing time `t[i]` ascending** (shortest-job-first). This is optimal when
  all weights are equal (it minimizes the *unweighted* sum of completion times), but a job
  with a large weight should be pulled earlier even if it is long.
- **Sort by weight `w[i]` descending** (heaviest-first). This ignores that a long heavy job
  delays everyone behind it; a light, very short job often belongs in front of it.

The right ordering compares two jobs through *both* of their numbers at once. Establishing
which pairwise comparison is correct — and proving it gives a globally optimal schedule, not
just a locally good swap — is the content of the problem.

## Evaluation settings

Judged on hidden tests covering: equal weights, equal processing times, many jobs that tie
under a naive ordering key (the comparator must order such ties consistently so the cost is
well defined), strongly skewed inputs (one long-light job versus many short-heavy jobs), the
empty instance (`n = 0`), a single job, and large `n = 2*10^5` with values near `10^4`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // TODO: order the jobs to minimize sum_i w[i] * C[i], where C[i] is the
    // completion time (prefix sum of processing times) of job i in that order,
    // then accumulate and print the minimum cost.
    long long cost = 0;

    cout << cost << "\n";
    return 0;
}
```
