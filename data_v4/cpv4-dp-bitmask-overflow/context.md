# Relay roster: maximum total synergy of a runner-to-leg assignment

## Research question

A relay event has `n` legs (numbered `0 .. n-1`, run in that order) and a squad of exactly `n`
runners (numbered `0 .. n-1`). You are given an `n x n` table `s`, where `s[i][j]` is the
**synergy** the squad gains if runner `i` is the one who runs leg `j`. Every runner must be used
exactly once and every leg must be filled exactly once — that is, you must choose a **perfect
assignment** (a permutation) of runners to legs. The squad's total synergy is the sum of `s[i][j]`
over the chosen (runner, leg) pairs. Output the **maximum total synergy** over all assignments.

This is the assignment / perfect-matching problem on a complete bipartite graph. It is the kind of
subproblem buried inside scheduling, crew-rostering, and task-allocation systems, so getting the
exact-optimum version right matters.

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
worth `10^9`).

## Evaluation settings

Judged on hidden tests covering: `n = 1`; small `n` checkable against the permutation oracle; the
maximum `n = 18`; tables with many ties (so the argmax is non-unique); tables of all zeros; and
adversarial large-value tables that stress the range of the accumulated sum.

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
