# Interval Scheduling on Few Rooms

## Research question

You are given `n` weighted time intervals and `K` identical rooms. Each interval `i` has an integer
start `s_i`, an integer end `e_i` (with `s_i < e_i`) and a positive integer weight `w_i`. You must
assign each interval to one of the `K` rooms or **reject** it. Two intervals placed in the **same**
room may not overlap: intervals are half-open `[s, e)`, so two of them conflict iff
`s_a < e_b` and `s_b < e_a` — sharing only an endpoint (`e_a == s_b`) is allowed. The objective is to
**maximize the total weight of the accepted (assigned) intervals**.

With `K = 1` this is the classic weighted interval scheduling problem, solvable exactly by dynamic
programming. For general `K` and arbitrary weights it is the *weighted job interval selection /
k-track interval packing* problem, which is NP-hard, so there is no fast exact answer at the sizes
here — it is a heuristic optimization problem judged by a continuous score.

## Input / output contract

- **Input (stdin).** The first line is `n K` (`1 <= n <= 600`, `1 <= K <= 5`). Then `n` lines
  follow, the `i`-th being `s_i e_i w_i` with `0 <= s_i < e_i <= 100000` and `1 <= w_i <= 1000`.
  Intervals are given in an arbitrary order; that input order is the index `i = 0 .. n-1`.
- **Output (stdout).** Exactly `n` lines (or `n` whitespace-separated integers). The `i`-th value is
  the room id in `{0, 1, ..., K-1}` to which interval `i` is assigned, or `-1` to reject interval
  `i`. Any output with the wrong count, or a room id outside `{-1, 0, ..., K-1}`, is infeasible.
- **Time limit:** 2 seconds. **Memory:** 256 MB.

## Background

Two reference approaches frame the problem before committing to one.

- **Per-room dynamic programming is exact only for `K = 1`.** Sorting by end time and taking, for
  each interval, the best of "skip it" or "take it plus the best compatible prefix" solves a single
  room optimally in `O(n log n)`. But there is no clean DP that splits `n` weighted intervals across
  `K > 1` rooms optimally in polynomial time; greedily filling room after room with the single-room
  DP is not optimal because an interval that is best left out of room 0 might be exactly what room 1
  needed, and the choice interacts globally.

- **Greedy by weight (descending) with a feasibility index.** Process intervals from heaviest to
  lightest; place each one in any room where it currently fits, else reject it. This is the standard
  strong constructive heuristic for weighted packing: heavy intervals get first claim on the scarce
  room-time. The only subtlety is testing "does this interval fit in room r" quickly; with a balanced
  ordered set of the room's placed intervals it is an `O(log n)` predecessor/successor probe.

Greedy-by-weight is a strong start but it is myopic: once a medium interval is placed it can block a
cluster of cheaper intervals whose combined weight exceeds it, or it can sit where a slightly heavier
late-arriving interval would have wanted to go. The lever is a local search that can **eject**
already-placed intervals to make room for a better one, accepted under a simulated-annealing
criterion, with feasibility re-checked only inside the one affected room.

## Evaluation settings

For a fixed seed the generator (below) produces one instance. A solver's output is scored as follows
(this is exactly what `verify/score.py` computes):

- **Feasibility / floor.** If the output does not have exactly `n` values, or any value is outside
  `{-1, 0, ..., K-1}`, or any room contains two overlapping accepted intervals (half-open overlap as
  defined above), the score is **0**.
- **Objective.** Otherwise the raw score is `Σ w_i` over all accepted intervals (`assign[i] != -1`).
- **Normalized score.** The reported metric is `raw / baseline`, where `baseline` is the raw
  objective of the deterministic **first-fit-by-start** assignment: sort all intervals by
  `(start, end)`, and place each into the lowest-indexed room whose last placed end `<= ` this
  interval's start, otherwise reject it. (If that baseline is 0, the raw objective is reported
  directly, so we never divide by zero.) A higher normalized score is better; an infeasible output
  scores 0.

**How instances are generated** (`verify/gen.py`, parameter = integer seed). `n` is drawn in
`[400, 600]` and `K` in `[2, 5]`. Interval starts are clustered around a handful of random temporal
"hot spots" (so contention for room-time is real), durations are mostly short with a heavy tail of
long hard-to-pack jobs, and weights are heavy-tailed (most intervals modest, a rare few very
valuable). This clustering is what makes a pure first-fit baseline leave substantial weight on the
table and what makes the heavy-interval-protecting, ejection-driven search pay off.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
assignment to stdout. The scaffold below already emits a valid solution (reject everything is legal,
scoring 0); the method replaces the TODO with the construction + ejection-based local search.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, K;
    if (!(cin >> n >> K)) return 0;
    vector<int> S(n), E(n), W(n);
    for (int i = 0; i < n; i++) cin >> S[i] >> E[i] >> W[i];

    vector<int> assign(n, -1);   // assign[i] in {-1, 0, ..., K-1}

    // TODO: greedy-by-weight construction (per-room ordered set for O(log n)
    //       feasibility) + simulated annealing with targeted ejection moves,
    //       feasibility re-checked only inside the affected room.

    string buf;
    for (int i = 0; i < n; i++) { buf += to_string(assign[i]); buf += '\n'; }
    cout << buf;
    return 0;
}
```
