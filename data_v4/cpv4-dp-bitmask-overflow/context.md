# Relay roster: maximum total synergy of a runner-to-leg assignment

## Research question

A relay event has `n` legs (numbered `0 .. n-1`, run in that order) and a squad of exactly `n`
runners (numbered `0 .. n-1`). You are given an `n x n` table `s`, where `s[i][j]` is the
**synergy** the squad gains if runner `i` is the one who runs leg `j`. Every runner must be used
exactly once and every leg must be filled exactly once — that is, you must choose a **perfect
assignment** (a permutation) of runners to legs. The squad's total synergy is the sum of `s[i][j]`
over the chosen (runner, leg) pairs. Output the **maximum total synergy** over all assignments.

This is the assignment / perfect-matching problem on a complete bipartite graph, solved by a DP over
subsets of runners (a "bitmask DP"). It is the kind of subproblem buried inside scheduling,
crew-rostering, and task-allocation systems, so getting the exact-optimum version right — including
the value scale, where a 32-bit accumulator silently overflows — matters.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 18`). Then `n` lines each containing `n`
  integers; the `j`-th integer on line `i` is `s[i][j]` (`0 <= s[i][j] <= 10^9`),
  whitespace-separated. (The reader is whitespace-agnostic, so any spacing/newlines are accepted.)
- Output (stdout): a single line with the maximum total synergy.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 3` and

```
1000000000  900000000  800000000
 850000000 1000000000  950000000
 700000000  950000000 1000000000
```

the answer is `3000000000` (assign runner 0 -> leg 0, runner 1 -> leg 1, runner 2 -> leg 2, each
worth `10^9`). Note this exceeds the 32-bit signed range.

## Background

We must pick one runner per leg, all distinct, maximizing the summed synergy. Two families of
approach are on the table before committing to one:

- **Brute force over permutations.** Enumerate all `n!` runner-to-leg assignments and keep the best
  sum. Trivially correct, but `18!` is about `6.4 * 10^15`, far beyond any time limit. This is the
  reference oracle for small `n`, not a contender for the real bounds.
- **Bitmask DP over the set of used runners.** Fill legs in index order. A subproblem is identified
  by the *set* of runners already placed; the number of placed runners equals the next leg to fill.
  This is `O(2^n * n)` time and `O(2^n)` space; the open questions are the exact state definition,
  the transition, and — crucially — the integer width of the accumulator.

## Evaluation settings

Judged on hidden tests covering: `n = 1`; small `n` checkable against the permutation oracle; the
maximum `n = 18`; tables with many ties (so the argmax is non-unique); tables of all zeros; and
adversarial large-value tables where the optimal total reaches `n * 10^9` (up to `1.8 * 10^10`),
which overflows a signed 32-bit integer. A solution using `int` for the DP table or the running sum
is a silent wrong answer on those.

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
    vector<vector<long long>> s(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> s[i][j];

    // TODO: bitmask DP over the set of used runners; legs are filled in index order,
    //       so the next leg to fill is popcount(mask). Maximize the total synergy.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
