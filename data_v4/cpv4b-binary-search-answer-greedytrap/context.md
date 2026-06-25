# Balancing the night shift at a screen-printing shop

## Research question

A screen-printing shop runs `m` identical presses overnight. There are `n` print jobs queued; job
`i` needs `t[i]` minutes of press time and must run on **exactly one** press from start to finish (a
job is never split or paused across presses). Every press can run any number of jobs, one after
another, and a press's finishing time is the sum of the `t[i]` of the jobs assigned to it. The shop
closes when the **last** press finishes, so the closing time equals the largest press load over the
assignment.

You decide which jobs go on which press. Choose the assignment that **minimizes the closing time**
(the makespan), and output that minimum.

This is the identical-machines makespan-minimization problem, `P || C_max`. It is the canonical place
where a beginner reaches for a load-balancing greedy, and the canonical place that greedy quietly
returns a wrong number. The right tool is to binary-search the closing time and test feasibility, but
the feasibility test is itself the subtle part.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`0 <= n <= 14`, `1 <= m <= 14`). The second line has `n` integers `t[i]`
  (`1 <= t[i] <= 10^9`), whitespace-separated. If `n = 0` the second line is empty or absent.
- Output (stdout): a single line with the minimum possible closing time (makespan).
- Time limit: 1 second. Memory: 256 MB.

Example: for `m = 3` presses and jobs `t = [5, 5, 4, 4, 3, 3, 3]` the answer is `9` — put `{5,4}`,
`{5,4}`, `{3,3,3}` on the three presses, giving loads `9, 9, 9`.

## Background

The closing time is bounded below by two easy quantities — the largest single job `max t[i]` (some
press must run it) and the ceiling of the average load `ceil(sum t[i] / m)` (the total work cannot
shrink) — and bounded above by `sum t[i]` (one press does everything). The answer lives in
`[max t[i], sum t[i]]`. Two routes are on the table before committing:

- **A load-balancing greedy (LPT).** Sort jobs from longest to shortest and drop each onto the press
  that is currently least loaded. This is `O(n log n)`, famously a good *approximation* (within 4/3 of
  optimal), and very tempting to ship as "the answer." The open question is whether it is ever *exactly*
  optimal — and whether the test data can tell.
- **Binary search on the closing time.** The predicate "can all jobs finish by time `T`?" is monotone
  in `T`: if everything fits under `T`, it certainly fits under any `T' > T`. So binary-search the
  smallest feasible `T`. The open question is how to *decide feasibility* — partitioning jobs into at
  most `m` groups each summing to at most `T` is itself a bin-packing decision, and a sloppy feasibility
  check is its own trap.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (answer `0`); `n = 1`; `m = 1` (answer is the total);
`m >= n` (each job can have its own press, answer is `max t[i]`); clustered job sizes engineered so the
LPT greedy is strictly suboptimal; the average-bound trap (`ceil(sum/m)` not achievable because one job
dominates); and full-size `n = 14` with `t[i]` near `10^9` (so loads and the search range exceed 32
bits).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    // TODO: minimize the makespan = assign each job to one of m presses so that
    // the largest press load is as small as possible; output that minimum.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
