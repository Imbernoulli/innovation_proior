# Assignment with Per-Worker Quotas and Convex Overtime Cost

## Research question

A dispatcher must assign every task in a batch to some worker. Each task goes to exactly one
worker, and a worker may take any number of tasks (including none). Two costs are in play:

- a **base cost** `c[i][j]` for letting worker `i` handle task `j` (skill/affinity), and
- an **overtime cost** that grows the more tasks a single worker is given.

Each worker `i` has a *regular quota* `q[i]`: the first `q[i]` tasks that worker handles carry no
overtime surcharge. Beyond the quota, surcharges escalate. Concretely, the `m`-th task handed to
worker `i` (`m = 1, 2, ...`) carries a marginal overtime surcharge

```
s_i(m) = base[i] * max(0, m - q[i]).
```

So the first `q[i]` tasks add `0` overtime, the `(q[i]+1)`-th adds `base[i]`, the next adds
`2*base[i]`, and so on. The **total** overtime a worker pays for `m` tasks is the running sum of
these marginals, `base[i] * (m - q[i])(m - q[i] + 1)/2` once `m > q[i]` — a **convex** function of
the load `m`. The goal is to assign all tasks to **minimize the total cost** (sum of base costs plus
every worker's overtime).

This convexity is the whole story. If overtime were a flat per-task fee, or if each worker could take
at most one task, the problem would be a plain (linear) assignment. The escalating surcharge makes the
cost of a worker depend non-linearly on *how many* tasks it accumulates, which couples the assignment
decisions in a way linear assignment cannot express.

## Input / output contract

- Input (stdin):
  - line 1: two integers `W T` — number of workers and number of tasks (`1 <= W <= 100`,
    `0 <= T <= 100`, and `W + T <= 198` so the flow network has at most ~200 nodes).
  - next `W` lines: row `i` holds `T` integers `c[i][0..T-1]` — base cost of worker `i` on each task
    (`0 <= c[i][j] <= 10^9`). When `T = 0` these lines are empty.
  - next line: `W` integers `q[0..W-1]` — regular quota of each worker (`0 <= q[i] <= T`).
  - next line: `W` integers `base[0..W-1]` — overtime slope of each worker (`0 <= base[i] <= 10^6`).
- Output (stdout): a single line with the minimum achievable total cost.
- Time limit: 2 seconds. Memory: 256 MB.

Worked example:

```
2 3
4 2 8
3 5 1
1 1
10 10
```

Answer: `16`. Assign task 0 to worker 1 (cost 3), task 1 to worker 0 (cost 2), task 2 to worker 1
(cost 1). Worker 1 now holds two tasks; its quota is 1, so the second task adds `10*(2-1)=10` overtime.
Total `3 + 2 + 1 + 10 = 16`. Every alternative (e.g. splitting one task each onto both workers and
dumping the third on someone) costs at least as much.

## Background

The shape "workers to tasks, minimize cost" is the textbook **assignment problem**, classically solved
by the Hungarian algorithm in `O(n^3)`. Hungarian assumes the cost is a *sum of independent
worker-task pairings*: the price of a matching is just the sum of its chosen `c[i][j]` entries. That
assumption is exactly what the convex overtime breaks — the price of giving worker `i` its fifth task
is not a fixed number, it depends on how many tasks `i` already has. There is no single weight you can
put on a worker-task edge that captures "this is the m-th task for this worker", because m is decided
by the global assignment, not by the edge.

Two families of approach are on the table:

- **Linear assignment / Hungarian (or a min-cost matching with fixed edge weights).** Fast and clean,
  but it can only encode a cost that is additive over independently-priced pairings. To use it here you
  would have to bake a fixed overtime price into each worker-task edge, which mis-prices the convex
  load — the model would either over- or under-count overtime depending on the eventual load.
- **Min-cost flow.** Model tasks and workers as a bipartite flow network with a source feeding tasks
  and a sink draining workers; the assignment is a flow of value `T`. A flow network can route *several*
  units through one worker node, and crucially it can price those units **separately**. That is the
  hook the convex cost needs.

## Evaluation settings

Judged on hidden tests covering: tiny instances (brute-forceable), `T = 0` (no tasks, answer 0),
`W = 1` (one worker forced to absorb all tasks, pure convex overtime), `base[i] = 0` (no overtime, a
pure linear assignment), `q[i] = 0` (every task is overtime from the first), `q[i] >= T` (overtime
never triggers), heavily asymmetric quotas/slopes that force load balancing across workers, and
maximum-size instances (`W + T` near 200, base costs near `10^9`) where the total can approach `10^14`,
so 64-bit accumulators are mandatory.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int W, T;
    if (!(cin >> W >> T)) return 0;

    vector<vector<long long>> c(W, vector<long long>(T));
    for (int i = 0; i < W; i++)
        for (int j = 0; j < T; j++)
            cin >> c[i][j];

    vector<long long> q(W), base(W);
    for (int i = 0; i < W; i++) cin >> q[i];     // regular quota of worker i
    for (int i = 0; i < W; i++) cin >> base[i];  // overtime slope of worker i

    // TODO: assign every task to a worker minimizing
    //   sum of base costs + sum over workers of the convex overtime cost.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
