# Strongest guaranteed floor over a long-enough relay window

## Research question

A relay chain logs `n` signal-strength readings `a[0..n-1]` along a line (each reading is an integer
that may be **negative or zero** — the sensor is calibrated around an arbitrary baseline, so a
reading below the baseline is a legitimate negative number). To stay locked, the chain must operate
over a **contiguous block of at least `k` consecutive positions**, and the quality it can guarantee
over that block is the **minimum reading inside the block** (a chain is only as strong as its weakest
link). You get to choose *which* block of length `>= k` to use.

Output the **largest guaranteed floor** achievable: the maximum, over every contiguous block of
length at least `k`, of the minimum reading in that block. If no block of length `>= k` exists
(i.e. `n < k`), the chain cannot lock at all — output the literal string `INFEASIBLE`.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k`
  (`0 <= n <= 2*10^5`, `1 <= k <= 2*10^5`). Then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated (possibly across lines).
- Output (stdout): a single line. If `n >= k`, print the maximum-over-blocks-of-the-block-minimum
  (an integer, which may be negative or zero). If `n < k`, print `INFEASIBLE`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `k = 3`, `a = [2, -1, 3, 3, 5, -4, 3]` the answer is `3`: the block at indices
`2..4` is `[3, 3, 5]` with minimum `3`, and no block of length `>= 3` has a larger minimum.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays; arrays mixing negatives, zeros, and positives;
the empty array (`n = 0`); `k = 1`; `k = n`; all-negative arrays; all-zero arrays; `n < k`
(`INFEASIBLE`); and large `n = 2*10^5` with `|a[i]|` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: if no block of length >= k exists, print INFEASIBLE.
    // Otherwise compute the largest guaranteed floor and print it.

    return 0;
}
```
