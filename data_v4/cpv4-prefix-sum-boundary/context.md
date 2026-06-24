# Counting sensor windows whose total lies in a band

## Research question

A field sensor produces `n` integer readings `a[1..n]` over `n` consecutive minutes; a reading may be
negative (the sensor reports a signed delta, so values can go either way). An analyst wants to know
how many **contiguous time windows** `[l, r]` (with `1 <= l <= r <= n`) have a window total
`a[l] + a[l+1] + ... + a[r]` that lands inside a fixed **inclusive** acceptance band `[L, R]` —
i.e. `L <= total <= R`. Count those windows and output the count.

The single-element windows (`l = r`) count too, and because both band endpoints are inclusive, a
window whose total equals exactly `L` or exactly `R` is accepted. The whole difficulty of the
problem lives on these boundaries: a window total is a difference of two prefix sums, and the band
membership test `L <= P[r] - P[l-1] <= R` rearranges into an *inclusive interval query on prefix
values*, so every inclusive/exclusive choice — both in the band and in the set of admissible prefix
indices — must be exactly right or the count is silently off.

## Input / output contract

- Input (stdin): the first line contains three integers `n`, `L`, `R`
  (`0 <= n <= 2*10^5`, `-10^18 <= L <= R <= 10^18`). The second line contains `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty/absent.
- Output (stdout): a single line with the number of windows `[l, r]` whose total lies in `[L, R]`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 6`, band `[3, 5]`, and `a = [2, -1, 3, 1, -4, 2]`, the answer is `6`. The accepted
windows are `[1,3] = 4`, `[1,4] = 5`, `[1,6] = 3`, `[2,4] = 3`, `[3,3] = 3`, `[3,4] = 4`. Three of
them hit the lower band edge `3` exactly, which is precisely the inclusive-boundary case.

## Background

Define prefix sums `P[0] = 0` and `P[i] = a[1] + ... + a[i]`. The total of window `[l, r]` is
`P[r] - P[l-1]`. For a fixed right end `r`, accepting the window means

```
L <= P[r] - P[l-1] <= R   <=>   P[r] - R <= P[l-1] <= P[r] - L,
```

with `l - 1` ranging over the prefix indices `0, 1, ..., r-1`. So the count for this `r` is the
number of *already-seen* prefix values that fall in the inclusive interval `[P[r] - R, P[r] - L]`.
Two families of approach are on the table before committing to one:

- **Quadratic enumeration.** For every `l` extend `r` and accumulate the running total, testing the
  band each step. This is `O(n^2)`, obviously correct, but far too slow at `n = 2*10^5`.
- **Prefix sums + an order-statistics structure.** Sweep `r` left to right, and for each `r` ask "how
  many previously inserted prefix values lie in `[P[r] - R, P[r] - L]`?" A Fenwick (BIT) over
  coordinate-compressed prefix values answers each query in `O(log n)` for `O(n log n)` total. The
  open questions are the inclusivity of the interval endpoints and *which* prefix indices are
  "already seen" when `r` is processed.

Because the readings can be negative, the prefix sums are **not monotone**, so a two-pointer /
sliding-window shortcut does not apply; the order-statistics structure is the natural route.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, the empty
array (`n = 0`), a single element (`n = 1`), all-non-positive arrays, constant arrays, degenerate
bands with `L = R` (so only windows hitting an exact total count), bands placed exactly on realizable
totals (to exercise both inclusive endpoints), and large `n = 2*10^5` with `|a[i]|` near `10^9` so
both the prefix sums (up to `~2*10^14`) and the answer count (up to `~2*10^10`) overflow 32 bits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> P(n + 1);
    P[0] = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        P[i] = P[i - 1] + x;
    }

    long long answer = 0;
    // TODO: for each r, count previously-seen prefix indices j in {0..r-1}
    //       with P[r]-R <= P[j] <= P[r]-L, using a Fenwick over compressed prefix values.

    cout << answer << "\n";
    return 0;
}
```
