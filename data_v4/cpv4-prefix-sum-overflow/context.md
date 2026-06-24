# Auditing total flow across many gauge-window queries

## Research question

A river-monitoring station records `n` consecutive hourly **net-flow** readings `a[1..n]`, where each
reading is a signed integer: positive means net inflow into the reach during that hour, negative means
net outflow. An auditor then submits `q` queries. Each query is a window `[l, r]` (`1 <= l <= r <= n`)
and asks for the **net flow over that window**, i.e. the sum `a[l] + a[l+1] + ... + a[r]`.

The single deliverable is the **total audited volume**: the sum of the answers to *all* `q` queries,
output as one integer. Windows may overlap freely, so the same hour can be counted many times, and the
running total can be large in magnitude (and either sign). The task is to answer every query and report
the grand total efficiently — a textbook setting for prefix sums, but with constraints chosen so the
intermediate and final magnitudes do not fit in a 32-bit integer.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `q` (`1 <= n <= 10^5`, `1 <= q <= 5*10^4`).
  The second line has `n` integers `a[1..n]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated. Then `q`
  lines follow, each with two integers `l r` (`1 <= l <= r <= n`) describing one query window.
- Output (stdout): a single line with the total audited volume — the sum over all `q` queries of
  `a[l] + ... + a[r]`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 5`, readings `a = [10^9, 10^9, 10^9, -5, 10^9]`, and the three queries `[1,3]`,
`[2,5]`, `[1,5]`, the window sums are `3*10^9`, `2999999995`, `3999999995`, so the total is
`9999999990`.

## Background

The naive route is to answer each query by re-summing its window: `O(r - l + 1)` per query, hence
`O(n*q)` overall, which at `n = 10^5` and `q = 5*10^4` is `5*10^9` additions — far too slow for a
1-second limit. The standard acceleration is the **prefix-sum** array: define `prefix[0] = 0` and
`prefix[i] = a[1] + ... + a[i]`. Then any window sum is a single subtraction,
`a[l] + ... + a[r] = prefix[r] - prefix[l-1]`, computed in `O(1)` after an `O(n)` precomputation. The
two design questions before committing are:

- **Indexing.** The `prefix[l-1]` term must reference the element just *before* the window. A
  1-indexed array with a sentinel `prefix[0] = 0` makes `l = 1` work without a special case; an
  off-by-one here silently corrupts every query that starts at position 1.
- **Magnitudes / types.** A single window sum can reach `n * max|a| = 10^5 * 10^9 = 10^{14}`, and the
  grand total over `q` windows can reach `q * 10^{14} = 5*10^{18}`. The first already overflows a
  32-bit `int` (max `~2.1*10^9`); the total is roughly `4*10^9` times the 32-bit range. Every prefix
  value and the accumulator must be 64-bit.

## Evaluation settings

Judged on hidden tests covering: small hand-checkable instances; arrays mixing large positives, large
negatives, and zeros; windows of length 1 and windows spanning the whole array; many overlapping
windows that re-count the same large hours (driving the total toward its `~5*10^{18}` extreme of either
sign); and the minimal case `n = q = 1`. Tests are specifically chosen so that a 32-bit accumulator or a
32-bit prefix array yields a wrapped, wrong answer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    // Read a[1..n] and build prefix sums; answer each [l, r] in O(1); sum the answers.
    // TODO: choose data types so that neither a window sum (~10^14) nor the running
    //       total (~5*10^18) overflows, and handle the l = 1 boundary via prefix[0] = 0.
    long long total = 0;

    cout << total << "\n";
    return 0;
}
```
