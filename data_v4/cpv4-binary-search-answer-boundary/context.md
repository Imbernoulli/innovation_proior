# Earliest day to assemble bouquets from a row of flower beds

## Research question

A greenhouse has `n` flower beds arranged in a single row, indexed `0..n-1`. Bed `i` first blooms on
day `b[i]` and stays bloomed forever after. You want to assemble exactly `m` bouquets. A single
bouquet is made from `k` **consecutive** beds that are *all bloomed at the same time*, and each bed
may be used in at most one bouquet. Concretely, on a given day, scan the row: every maximal block of
consecutive bloomed beds of length `L` can be cut into `floor(L / k)` bouquets (a block of length
`L` shorter than `k` yields nothing, and the leftover `L mod k` beds at the end of a block are
wasted).

Find the **earliest day** `t` on which at least `m` bouquets can be assembled. If it can never be
done — even with every bed bloomed at once there are not enough beds — output `-1`.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `m`, `k`
  (`1 <= n <= 2*10^5`, `1 <= m <= 10^9`, `1 <= k <= n`). The second line has `n` integers
  `b[0..n-1]` (`1 <= b[i] <= 10^9`), the bloom day of each bed, whitespace-separated.
- Output (stdout): a single line — the earliest day `t` on which at least `m` bouquets can be made,
  or `-1` if it is impossible.
- Time limit: 1 second. Memory: 256 MB.

Example: `n = 7`, `m = 2`, `k = 2`, `b = [3, 1, 4, 1, 5, 9, 2]`. The answer is `4`. On day 3 only one
bouquet is possible; on day 4 beds `0..3` (days `3,1,4,1`) are all bloomed, a block of length 4 that
cuts into `floor(4/2) = 2` bouquets.

## Evaluation settings

Judged on hidden tests covering: the impossible case; `k = 1`; a single bed; all beds blooming on
the same day; runs that are *broken* by a late-blooming bed so the answer is forced strictly later
than the first bloom; many tied bloom days; and large inputs `n = 2*10^5` with `b[i]` near `10^9`
and `m, k` near their limits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long m, k;
    if (scanf("%d %lld %lld", &n, &m, &k) != 3) return 0;
    vector<long long> b(n);
    for (int i = 0; i < n; i++) scanf("%lld", &b[i]);

    // TODO: compute the earliest day t on which at least m bouquets can be
    // assembled, or -1 if it is impossible.
    long long answer = -1;

    printf("%lld\n", answer);
    return 0;
}
```
