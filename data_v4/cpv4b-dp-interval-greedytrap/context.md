# One telescope, one night: maximum-value non-overlapping observations

## Research question

A research observatory has a single telescope and one usable night. Astronomers have submitted
`n` candidate observations. Observation `i` would occupy the telescope for the half-open time
interval `[s_i, e_i)` (in integer minutes from the start of the night) and, if carried out, would
yield scientific value `v_i >= 0`. The telescope can run only one observation at a time, so two
chosen observations may not overlap in time. Two observations whose intervals merely *touch* — one
ending exactly when the next begins (`e_i == s_j`) — do **not** overlap and may both be scheduled.

Choose a subset of the observations, no two overlapping, that **maximizes the total scientific
value**. You may schedule none (value `0`). Output that maximum total value.

This is the weighted version of single-machine interval selection. It is the kind of subproblem
that sits inside telescope scheduling, CPU job admission, and ad-slot allocation, so the exact
handling — the half-open touching convention, zero-value and empty corners, and large sums —
matters.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then `n` lines (or simply `n`
  triples of whitespace-separated tokens) follow, each giving `s_i e_i v_i` with
  `0 <= s_i < e_i <= 10^9` and `0 <= v_i <= 10^9`.
- Output (stdout): a single line with the maximum achievable total value.
- Time limit: 1 second. Memory: 256 MB.

Example: for the five observations

```
0 30 20
0 90 50
30 60 25
60 90 25
40 50 5
```

the answer is `70` — run `[0,30)` (value 20), then `[30,60)` (value 25), then `[60,90)`
(value 25). Grabbing the single highest-value run `[0,90)` (value 50) instead would block the whole
night and score only 50.

## Background

The single non-overlap constraint makes this a constrained selection problem, part of the broader
family of interval-scheduling problems that also includes job admission and resource allocation.
A range of algorithmic tools — greedy heuristics and dynamic programming among them — have been
used on variants of this family, with correctness and efficiency depending sensitively on the
details, including the half-open touching convention.

## Evaluation settings

Judged on hidden tests covering: the empty instance (`n = 0`), a single observation, observations
that merely touch at endpoints (must be allowed together), fully nested observations (all mutually
overlapping, so at most one is chosen), many identical intervals, zero-value observations, adversarial
instances, and large `n = 2*10^5` with values near `10^9`.

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
    struct Obs { long long s, e, v; };
    vector<Obs> obs(n);
    for (int i = 0; i < n; i++) cin >> obs[i].s >> obs[i].e >> obs[i].v;

    // TODO: compute the maximum total value of a subset of non-overlapping
    //       half-open observations (the empty subset, value 0, is allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
