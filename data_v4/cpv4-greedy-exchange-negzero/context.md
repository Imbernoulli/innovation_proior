# Freelance gig scheduling with deadlines (maximum total payout)

## Research question

You are a freelancer planning the next stretch of work. There are `n` gigs on offer. Gig `i` has a
**deadline** `d[i]` (a day index: the gig must be *finished* on some day in `1, 2, ..., d[i]`) and a
**payout** `v[i]`. You can finish **at most one gig per day**, you cannot split a gig across days, and
you may **decline** any gig you like. Some gigs are loss-leaders or carry penalties, so a payout may be
**negative or zero**; some deadlines may be `0`, meaning there is no valid day on which the gig could
ever be finished.

Choose which gigs to accept and on which day to finish each, respecting the one-gig-per-day rule and
every accepted gig's deadline, so as to **maximize the total payout**. Because you may decline
everything, the answer is always at least `0`.

This is the classic *job sequencing with deadlines* objective. It shows up inside packing,
single-machine scheduling, and resource-allocation problems, so getting the one-dimensional version
exactly right — including the negative/zero-payout and all-decline corners — matters.

## Input / output contract

- Input (stdin):
  - the first token is `n` (`0 <= n <= 2*10^5`);
  - then `n` integers `d[0..n-1]` (`0 <= d[i] <= 10^9`), the deadlines;
  - then `n` integers `v[0..n-1]` (`-10^9 <= v[i] <= 10^9`), the payouts.
  - All tokens are whitespace-separated; line breaks are not significant.
- Output (stdout): a single line with the maximum achievable total payout.
- Time limit: 1 second. Memory: 256 MB.

Example: for `d = [2, 1, 2, 1, 3]`, `v = [20, 10, 40, 30, 50]` the answer is `120` (finish gig 4 on
day 3, gig 2 on day 2, gig 3 on day 1; gigs 0 and 1 are declined because no free day remains).

## Background

A feasible plan assigns each *accepted* gig a distinct day no later than its deadline. A well-known
fact makes feasibility easy to reason about: a set of gigs is schedulable (one per day, each within
its deadline) **iff**, sorting the accepted deadlines ascending `d_1 <= d_2 <= ... <= d_m`, we have
`d_j >= j` for every `j` — the `j`-th-earliest-deadline gig can sit on day `j`. Two families of
approach are on the table before committing to one:

- **Greedy exchange (deadline scheduling).** Consider gigs in **decreasing payout** order and assign
  each to the **latest still-free day at or before its deadline**; if no such day exists, drop the
  gig. The exchange argument: putting a higher-payout gig as late as legally possible never blocks a
  choice that a different placement would have kept open, so the greedy set is optimal. With a
  disjoint-set "latest free day" structure this is `O(n alpha(n))` after the sort. The open questions
  are (a) whether to ever schedule a non-positive gig and (b) what to do when a deadline is `0` or
  exceeds `n`.
- **Subset search / feasibility DP.** Enumerate subsets of gigs and test each with the Hall condition
  above, keeping the max feasible total. This is `O(2^n n)` and only usable as an independent
  correctness oracle on tiny `n`; it is hopeless at `n = 2*10^5`.

## Evaluation settings

Judged on hidden tests covering: all-positive payouts; a mix of negatives, zeros, and positives
(where accepting a non-positive gig must never help); the empty instance (`n = 0`, answer `0`);
`n = 1`; all-negative and all-zero payouts (answer `0`); deadlines equal to `0` (a gig that can never
be placed); deadlines far larger than `n` (which must be capped); many gigs competing for the same
early day (ties); and large instances with `n = 2*10^5` and `|v[i]|` near `10^9`, so the accumulated
answer reaches `~2*10^14`, far outside 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> d(n), v(n);
    for (int i = 0; i < n; i++) cin >> d[i];
    for (int i = 0; i < n; i++) cin >> v[i];

    // TODO: accept gigs greedily by decreasing payout, placing each on the latest
    //       free day <= its deadline; skip non-positive payouts and invalid days.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
