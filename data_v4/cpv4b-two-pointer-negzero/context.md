# Counting balanced anomaly pairs within a comfort band

## Research question

A weather station has logged `n` daily temperature **anomalies** — deviations from a seasonal
baseline, recorded in tenths of a degree. An anomaly may be negative (colder than baseline), zero
(exactly on baseline), or positive (warmer). Climatologists call two **distinct** days `i` and `j`
a **balanced pair** when the sum of their two anomalies lands inside an inclusive *comfort band*
`[lo, hi]`, i.e. `lo <= a[i] + a[j] <= hi`. Pairs are unordered: `(i, j)` and `(j, i)` are the same
balanced pair and counted once.

Given the `n` anomalies and the band endpoints `lo` and `hi`, **count how many balanced pairs
exist**. Because the anomalies span negatives, zeros, and positives, and because the band itself may
be degenerate (a single value, or even empty when `lo > hi`), the sign and base-case handling is the
whole difficulty: a naive count that forgets the empty-band or all-negative corner reports a wrong —
sometimes negative — number.

## Input / output contract

- Input (stdin): the first line holds three integers `n`, `lo`, `hi`
  (`0 <= n <= 2*10^5`, `-2*10^9 <= lo, hi <= 2*10^9`; `lo > hi` is allowed and means an empty band).
  The second line holds `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When
  `n = 0` the second line may be empty or absent.
- Output (stdout): a single line with the number of unordered balanced pairs.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, band `[-3, 4]`, anomalies `[-5, 2, -1, 0, 7, -3]`, the answer is `8`.
The qualifying pairs (by value) are `(-5,2)=-3`, `(-5,7)=2`, `(2,-1)=1`, `(2,0)=2`, `(2,-3)=-1`,
`(-1,0)=-1`, `(0,-3)=-3`, `(7,-3)=4` — eight sums, each inside `[-3, 4]`.

## Background

The brute force enumerates all `O(n^2)` pairs; at `n = 2*10^5` that is `~2*10^10` operations and far
too slow. Two faster families are on the table before committing:

- **Sort plus binary search.** Sort the anomalies; for each `a[i]`, the partners `a[j]` that land in
  the band form a contiguous slice `[lo - a[i], hi - a[i]]` of the sorted array, locatable with two
  `lower_bound`/`upper_bound` calls. This is `O(n log n)`; the open question is how to exclude the
  self-pair `j = i` and avoid double counting unordered pairs.
- **Sort plus two pointers.** Reduce "count pairs in a band" to two "count pairs with sum `<= K`"
  queries via `countLE(hi) - countLE(lo - 1)`, and answer each `countLE` with a single linear
  two-pointer sweep over the sorted array. This is `O(n log n)` to sort then `O(n)` per sweep; the
  open question is the exact pointer movement and — critically — when the subtraction `countLE(hi) -
  countLE(lo - 1)` is even valid, since with `lo > hi` it can go negative.

## Evaluation settings

Judged on hidden tests covering: all-positive anomalies, mixtures with negatives and zeros, the
empty log (`n = 0`), a single day (`n = 1`, which has no pairs), all-negative anomalies, an empty
band (`lo > hi`), a degenerate one-value band (`lo == hi`, e.g. counting pairs summing to exactly
zero), heavy duplicates (so the same value repeats and pairs of equal values must be counted), and
large `n = 2*10^5` with `|a[i]|` near `10^9` so both the pair sums (up to `2*10^9`) and the answer
(up to `~2*10^10`) overflow 32-bit integers.

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
    long long lo, hi;
    cin >> lo >> hi;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count unordered pairs (i<j) with lo <= a[i]+a[j] <= hi (handle the empty band).
    long long ans = 0;

    cout << ans << "\n";
    return 0;
}
```
