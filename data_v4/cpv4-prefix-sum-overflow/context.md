# Auditing total flow across many gauge-window queries

## Research question

A river-monitoring station records `n` consecutive hourly **net-flow** readings `a[1..n]`, where each
reading is a signed integer: positive means net inflow into the reach during that hour, negative means
net outflow. An auditor then submits `q` queries. Each query is a window `[l, r]` (`1 <= l <= r <= n`)
and asks for the **net flow over that window**, i.e. the sum `a[l] + a[l+1] + ... + a[r]`.

The single deliverable is the **total audited volume**: the sum of the answers to *all* `q` queries,
output as one integer. Windows may overlap freely, so the same hour can be counted many times, and the
running total can be large in magnitude (and either sign). The task is to answer every query and report
the grand total efficiently.

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
1-second limit. Efficiently answering many range-sum queries over a fixed array is a classic setting
with several standard techniques; whichever is chosen must also be checked for correctness at the
window boundaries and for the range of values the arithmetic can produce.

## Evaluation settings

Judged on hidden tests covering: small hand-checkable instances; arrays mixing large positives, large
negatives, and zeros; windows of length 1 and windows spanning the whole array; many overlapping
windows that re-count the same large hours; and the minimal case `n = q = 1`.

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

    // Read a[1..n] and answer each query [l, r] with its window sum a[l] + ... + a[r];
    // sum all q answers into the total.
    // TODO: pick a data structure/approach that answers each query fast enough, choose
    //       integer types that hold the largest values the arithmetic can produce, and
    //       double-check the boundary behavior at l = 1.
    long long total = 0;

    cout << total << "\n";
    return 0;
}
```
