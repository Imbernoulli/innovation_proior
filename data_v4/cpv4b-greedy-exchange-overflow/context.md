# Festival booth scheduling: maximize total profit under per-act deadlines

## Research question

A street festival has a single performance stage and runs for a long string of equal-length time
slots numbered `1, 2, 3, ...`. You are pitched `n` acts. Act `i` will pay the festival a fee
`p[i]` if and only if it is staged, and it must be staged in **one** time slot whose number is
**at most** its deadline `d[i]` (the act's contract expires after slot `d[i]`). Each slot can hold
**at most one** act, and each act occupies **exactly one** slot. You may leave any act unscheduled
(staging nothing in some slots is fine, and turning an act away is fine).

Choose which acts to stage and in which slots so that the **total fee collected is maximized**, and
output that maximum total fee. This is the classic "job sequencing with deadlines" selection problem
dressed as a festival; the point of interest is that the greedy that solves it has to be justified by
an exchange argument, and that the total fee is large enough to break a careless accumulator.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then follow `n` lines, each with two
  integers `p[i]` and `d[i]` (`0 <= p[i] <= 10^9`, `1 <= d[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum total fee that can be collected.
- Time limit: 1 second. Memory: 256 MB.

Example: for the five acts `(p, d) = (100, 2), (60, 1), (70, 2), (40, 1), (90, 3)` the answer is
`260` — stage the `100` act in slot 2, the `90` act in slot 3, and the `70` act in slot 1; the `60`
and `40` acts have deadline 1 but slot 1 is taken, so they are turned away.

## Background

The selection is constrained two ways at once: a slot holds one act, and an act can only go in slots
up to its deadline. Two routes are on the table before committing to one:

- **Greedy by fee, latest-slot assignment.** Sort the acts by fee descending; walk down the list and
  put each act in the *latest still-free slot that is `<= d[i]`*, skipping the act if no such slot
  exists. This is `O(n log n)` plus the cost of finding latest free slots. The open question is
  whether grabbing the biggest fees first and pushing each act as late as legally possible is
  actually optimal — that needs an exchange argument, not a hunch.

- **Bipartite matching / DP.** Model acts-versus-slots as a weighted bipartite graph and solve a
  maximum-weight matching, or run a DP over slots. Correct but far heavier than necessary if the
  greedy can be proven.

Deadlines can be as large as `10^9`, so the slots cannot be materialized one-per-deadline; only the
deadlines that actually occur matter, and "latest free slot `<= d`" has to be answered without an
array indexed by raw deadline value.

## Evaluation settings

Judged on hidden tests covering: all acts fitting (large distinct deadlines), heavy collisions
(many acts sharing one small deadline), `p[i] = 0` acts, the empty instance (`n = 0`), a single act,
deadlines far larger than `n`, and large `n = 2*10^5` with fees near `10^9` so the collected total
reaches about `2*10^14` and overflows a 32-bit integer.

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
    vector<long long> p(n);
    vector<int> d(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> d[i];

    // TODO: schedule acts to maximize total collected fee under the deadline/slot constraints.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
