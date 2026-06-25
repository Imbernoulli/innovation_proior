# Counting loadable container runs on a weight-limited conveyor

## Research question

A dock conveyor carries `n` shipping containers in a fixed line; container `i` has weight `w[i]` (a
non-negative integer, in kilograms). A crane can lift any **contiguous run** of containers `[i..j]`
in one pass, but only if the run's **total weight** does not exceed the crane's capacity `S`. The
operations team wants to know how many distinct contiguous runs are liftable at all — i.e. count the
number of pairs `(i, j)` with `0 <= i <= j <= n-1` such that `w[i] + w[i+1] + ... + w[j] <= S`.

Output that count. A single container whose own weight already exceeds `S` contributes nothing; an
all-light array where even the full line fits contributes every one of its `n*(n+1)/2` runs.

This is the canonical "count subarrays with sum at most `S`" problem on non-negative values, the kind
of windowed-counting subroutine that appears inside throughput planning, rate limiting, and streaming
analytics. The point of interest is that **both** the running window sum and the final count grow far
past a 32-bit integer at the stated scale, so the data types have to be chosen deliberately.

## Input / output contract

- Input (stdin): the first line has two tokens, `n` (`0 <= n <= 2*10^5`) and `S`
  (`0 <= S <= 10^18`). The second line has `n` integers `w[i]` (`0 <= w[i] <= 10^9`),
  whitespace-separated. When `n = 0` the second line is empty or absent.
- Output (stdout): a single line with the number of contiguous runs whose total weight is `<= S`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `S = 7`, `w = [3, 1, 4, 1, 5, 2]` the answer is `12`. The twelve liftable runs
are `[0,0]=3`, `[0,1]=4`, `[1,1]=1`, `[1,2]=5`, `[1,3]=6`, `[2,2]=4`, `[2,3]=5`, `[3,3]=1`,
`[3,4]=6`, `[4,4]=5`, `[4,5]=7`, `[5,5]=2`; every run containing the `[0,1,2]` triple or longer
exceeds `7`.

## Background

Because all weights are non-negative, the prefix sums are non-decreasing, and that monotonicity is
exactly what a two-pointer (sliding-window) scan exploits. Two families of approach are on the table
before committing to one:

- **Per-right-endpoint binary search on prefix sums.** Build the prefix-sum array, then for each
  right end `j` binary-search the smallest left end `i` with `prefix[j+1] - prefix[i] <= S`. That is
  `O(n log n)` and correct, but it carries an array of `n+1` prefix sums (each up to `2*10^14`) and a
  per-step subtraction whose width must be tracked.
- **Two-pointer sliding window.** Keep a window `[left..right]` and a running `sum`. Advance `right`
  one step at a time, adding `w[right]`; whenever `sum > S`, advance `left`, subtracting weights,
  until the window is valid again. Every window ending at `right` with start in `[left..right]` is
  then admissible, contributing `right - left + 1` to the count. This is `O(n)` time, `O(1)` extra
  space; the open question is the exact contribution formula and, crucially, the widths of `sum` and
  the accumulator.

## Evaluation settings

Judged on hidden tests covering: tiny arrays and `n = 0`; arrays of all zeros with `S = 0` (every run
is liftable); single containers heavier than `S`; capacities below the lightest, between, and above
the whole-line total; and large `n = 2*10^5` with weights near `10^9` so that the running window sum
reaches roughly `2*10^14` and the answer count reaches roughly `2*10^10` — both well outside the
32-bit range of about `2.1*10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: count contiguous runs [i..j] with w[i]+...+w[j] <= S.
    long long count = 0;

    cout << count << "\n";
    return 0;
}
```
