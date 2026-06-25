# Sequencing repair jobs on a single bench to minimize total weighted wait

## Research question

A single repair bench processes `n` jobs one at a time, with no idle gaps and no preemption: once
the bench starts a job it runs it to completion, then immediately starts the next. Job `i` needs
`t[i]` minutes of bench time, and each minute that job `i`'s customer spends *waiting for their job
to finish* costs `w[i]` units of goodwill. If you process the jobs in some order, job `i` finishes at
the moment `C[i]` = (sum of `t` of every job processed up to and including `i`), and that job
contributes `w[i] * C[i]` to the total goodwill cost (the customer waits the whole time until their
own job is done, weighted by how impatient they are).

You choose the processing order. **Minimize the total weighted completion time**
`sum over i of w[i] * C[i]`, and output that minimum.

This is the single-machine total weighted completion time problem, `1 || sum w_j C_j`. It is a clean
setting in which the *locally obvious* orderings — "do the quickest job first" or "do the most
impatient customer first" — are both wrong, and the correct rule only falls out of an exchange
argument.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^5`). Then `n` lines (or just `n` whitespace
  separated pairs) follow, each giving two integers `t[i]` and `w[i]`
  (`1 <= t[i] <= 10^4`, `1 <= w[i] <= 10^4`).
- Output (stdout): a single line with the minimum achievable total weighted completion time.
- Time limit: 1 second. Memory: 256 MB.

Example: for the three jobs `A = (t=3, w=4)`, `B = (t=1, w=1)`, `C = (t=2, w=3)` the answer is `32`.

## Background

The cost of an order is fully determined by it: process the jobs in some sequence, accumulate the
running completion time, and sum `w[i] * C[i]`. There are `n!` orders, so brute force is out for any
real `n`; the question is which single sort key — if any — produces an optimal order. Three keys are
on the table before committing:

- **Shortest processing time first** (sort by `t` ascending). Intuition: finishing quick jobs early
  keeps every later completion time small. `O(n log n)`. Open question: does ignoring the weights
  ever cost more than it saves?
- **Most impatient first** (sort by `w` descending). Intuition: get the high-penalty customers off
  the bench as early as possible. `O(n log n)`. Open question: does ignoring the durations ever
  backfire?
- **Ratio rule** (sort by `t[i] / w[i]` ascending, i.e. by some combined key). `O(n log n)`. Open
  question: what exactly is the key, and why is *that* order globally optimal rather than just
  locally plausible?

## Evaluation settings

Judged on hidden tests covering: `n = 0` (empty, answer `0`); `n = 1`; jobs with equal durations,
equal weights, and equal `t/w` ratios (so any correct tie handling must agree); adversarial cases
constructed to defeat shortest-first and impatient-first simultaneously; and large `n = 10^5` with
`t, w` near `10^4`, where the accumulated completion times reach `~10^9` and the weighted sum reaches
`~5 * 10^17`, far past the 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // TODO: order the jobs to minimize sum of w[i] * C[i], where C[i] is the
    // completion time of job i, then accumulate and print that minimum.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
