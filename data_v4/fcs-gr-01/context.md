# Project Selection with Shared Prerequisite Machines

## Research question

A factory is deciding which of `n` projects to run. Running project `i` earns a fixed
profit `p[i]` (which may be negative — some projects lose money on their own). Running a
project requires a set of *machines*: project `i` can only run if the factory owns every
machine it lists as a prerequisite. There are `m` machines; buying machine `j` costs
`c[j] >= 0`, and a machine, once bought, is available to **every** project that needs it
(you pay for it at most once). You may also choose to run no projects at all.

Choose a subset of projects to run and a set of machines to buy so that

```
  (sum of profits of chosen projects)  -  (sum of costs of bought machines)
```

is **maximized**. Because running nothing is allowed, the answer is at least `0`. Output
that maximum net profit.

The structural catch is that prerequisites are *shared*: two projects can split the cost
of a machine they both need, so projects are not independent and you cannot decide them one
at a time. This is the classic **project-selection / maximum-weight-closure** problem, and
getting it right at the stated scale (up to 500 projects, 500 machines, and a dense web of
prerequisites) is the task.

## Input / output contract

- Input (stdin):
  - Line 1: two integers `n m` (`0 <= n <= 500`, `0 <= m <= 500`).
  - Line 2: `n` integers `p[0..n-1]` (`-10^9 <= p[i] <= 10^9`), the project profits.
  - Line 3: `m` integers `c[0..m-1]` (`0 <= c[j] <= 10^9`), the machine costs.
  - Line 4: one integer `E` (`0 <= E <= n*m`), the number of prerequisite edges.
  - Next `E` lines: two integers `i j` (`1 <= i <= n`, `1 <= j <= m`) meaning
    "project `i` requires machine `j`". No edge is repeated.
  - (When `n = 0` the profit line is empty; likewise for `m = 0`. Whitespace, including
    line breaks, is not significant — tokens may be split across lines arbitrarily.)
- Output (stdout): a single line with the maximum achievable net profit.
- Time limit: 2 seconds. Memory: 256 MB.

### Worked sample

Input:

```
3 2
10 12 8
6 7
4
1 1
2 1
2 2
3 2
```

Output:

```
17
```

Project 1 needs machine 1; project 2 needs machines 1 and 2; project 3 needs machine 2.
Running all three projects earns `10 + 12 + 8 = 30` and requires both machines, costing
`6 + 7 = 13`, for a net of `17`. No other choice does better — notice that once machines 1
and 2 are bought for projects 1 and 3, adding project 2 contributes its full `+12` at no
extra machine cost, which is exactly the kind of shared-cost coupling that makes deciding
projects independently fail.

## Background

Two framings are natural before committing to an algorithm.

- **It looks like a knapsack / per-project greedy.** For each project compute "profit minus
  the cost of the machines it needs" and take the profitable ones. This ignores that machine
  costs are *shared*: a machine paid for by one project is then free for another, so a project
  that looks unprofitable in isolation can be worth running once its machines are already
  bought. Charging each project the full cost of its machines double-counts, and the greedy
  has no consistent way to attribute shared costs.

- **It is a selection-with-implications problem.** "Choosing project `i` forces choosing
  (paying for) every machine it requires" is a *closure* condition on a directed graph:
  if a node is selected, all of its out-neighbours must be selected too. We want the
  selected set (a closure) whose total weight — profits of projects, *minus* costs of
  machines — is maximum. That is the **maximum-weight closure** problem.

The scale (`n, m <= 500`, up to `n*m = 250000` prerequisite edges) rules out enumerating
subsets and demands a polynomial method that the closure framing supplies.

## Evaluation settings

Judged on hidden tests covering: all-positive profits with cheap shared machines; profits
that are individually unprofitable but worth running jointly; negative-profit projects;
zero-cost and very expensive machines; the empty instance (`n = 0` and/or `m = 0`); a single
project with and without a satisfiable prerequisite; dense prerequisite graphs near
`E = n*m`; and large values near `10^9` so the net profit overflows 32-bit arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> profit(n), cost(m);
    for (int i = 0; i < n; ++i) cin >> profit[i];
    for (int j = 0; j < m; ++j) cin >> cost[j];

    int E;
    cin >> E;
    // read E prerequisite edges (project i requires machine j, both 1-based)

    // TODO: maximize (sum of chosen profits) - (sum of bought machine costs),
    //       subject to: a chosen project's machines must all be bought.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
