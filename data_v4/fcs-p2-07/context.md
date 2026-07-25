# Egg drop: minimum worst-case trials with a limited number of eggs

## Research question

There is a building with `m` floors numbered `1..m`. There is a *critical floor* `c` with the
property that an egg dropped from any floor `>= c` breaks, and an egg dropped from any floor `< c`
survives (`c` may be `m+1`, meaning no drop ever breaks; the task is only to *determine* the
threshold, so you never need to name `m+1` explicitly — distinguishing among `m` floors covers it).
A broken egg cannot be reused; a survived egg can.

You start with `k` identical eggs. In one *trial* you choose a floor and drop an egg from it,
observing break/survive. You must design a dropping strategy that is **guaranteed** to identify the
critical floor, and you want the strategy whose **worst case uses the fewest trials**.

Output that minimum worst-case number of trials as a function of `k` and `m`.

This is the "super egg drop" problem.

## Input / output contract

- Input (stdin): a single line (or whitespace-separated) with two integers `k` and `m`.
  - `1 <= k <= 100` (number of eggs)
  - `1 <= m <= 10000` (number of floors)
- Output (stdout): a single line with one integer — the minimum number of trials that guarantees
  identifying the critical floor in the worst case.
- Time limit: 1 second. Memory: 256 MB.

Examples:

- `k = 1, m = 2` -> `2`. With one egg, two trials are needed in the worst case.
- `k = 2, m = 6` -> `3`. Two eggs over six floors can be guaranteed in three trials.
- `k = 3, m = 14` -> `4`. With three eggs, four trials suffice for fourteen floors.
- `k = 2, m = 100` -> `14`. The classic "100 floors, 2 eggs" puzzle.

## Evaluation settings

Judged on hidden tests covering: `k = 1`; `k = 2`; small `k` with `m` ranging across its full span;
large `k`; the extreme corners `m = 1`, `m = 10000`, `k = 100`; and values of `m` and `k` chosen so
that the minimum trial count is checked at many different points, not just a few round numbers.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int k;       // number of eggs, 1 <= k <= 100
    long long m; // number of floors, 1 <= m <= 10000
    if (!(cin >> k >> m)) return 0;

    // TODO: compute the minimum worst-case number of trials that guarantees
    // identifying the critical floor among m floors using k eggs.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
