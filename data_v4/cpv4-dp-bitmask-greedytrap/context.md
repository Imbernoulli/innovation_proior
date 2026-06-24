# Cheapest contractor bundle to cover every required skill

## Research question

You are staffing a project that needs `m` distinct skills, numbered `0 .. m-1`. There are `k`
contractors available for hire. Contractor `i` charges a fixed fee `cost[i]` and brings a known set
of skills, given as an `m`-bit bitmask `mask[i]` (bit `j` set means contractor `i` has skill `j`).
You may hire any subset of contractors; the skills they bring are pooled. Choose a subset whose
pooled skills cover **all** `m` required skills while **minimizing the total fee paid**. Output that
minimum total fee, or `-1` if no subset can cover every skill.

This is weighted set cover over a small universe. It is exactly the kind of subproblem that hides
inside team-formation, feature-bundling, and "buy a set of items so the union has property X"
questions, so nailing the exact optimum — not an approximation — is the point.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `k` (`1 <= m <= 18`, `1 <= k <= 100`).
  Then `k` lines follow; line `i` has two integers `cost[i]` (`1 <= cost[i] <= 10^6`) and `mask[i]`
  (`0 <= mask[i] <= 2^m - 1`), the fee and the skill bitmask of contractor `i`.
- Output (stdout): a single line with the minimum total fee that covers all `m` skills, or `-1` if
  covering all skills is impossible.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `m = 6` and the three contractors

```
6 3
3 62
2 7
2 56
```

the answer is `4`. (Mask `62 = 111110b` covers skills `1..5`; mask `7 = 000111b` covers skills
`0,1,2`; mask `56 = 111000b` covers skills `3,4,5`. Hiring the last two for `2 + 2 = 4` covers all
six skills `0..5`.)

## Background

The union-coverage requirement makes this a global combinatorial optimization, not a per-contractor
decision. Two families of approach are on the table before committing to one:

- **Greedy by cost-efficiency.** Repeatedly hire the contractor with the best ratio of fee to number
  of *newly* covered skills, updating the covered set, until everything is covered. It is fast and a
  few lines; the open question is whether the locally cheapest-per-new-skill pick is globally
  optimal under the union constraint.
- **Bitmask dynamic programming over coverage states.** Treat the set of already-covered skills as a
  state in `{0, 1}^m` and compute, for every coverage state, the minimum fee to reach it. This is
  `O(2^m * k)`; the open questions are the exact transition, the processing order, and how to report
  impossibility.

## Evaluation settings

Judged on hidden tests covering: instances where the cost-efficiency greedy diverges from the
optimum, impossible instances (some skill appears in no contractor's mask, answer `-1`), single-skill
universes (`m = 1`), heavily overlapping masks where buying small pieces beats one big bundle,
duplicate masks at different fees, and large instances (`m = 18`, `k = 100`) where the `2^m` state
space must be handled within the time limit and total fees can exceed a 32-bit accumulator only mildly
(use a wide integer to be safe).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, k;
    if (!(cin >> m >> k)) return 0;
    vector<long long> cost(k);
    vector<int> mask(k);
    for (int i = 0; i < k; i++) cin >> cost[i] >> mask[i];

    // TODO: minimum total fee to hire a subset whose pooled skills cover all m skills,
    //       or -1 if impossible.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
