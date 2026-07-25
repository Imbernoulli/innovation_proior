# Harvesting lots: maximum profit from non-overlapping runs of length at least L

## Research question

A vineyard logs a daily *net profit* for each of `n` consecutive days as integers `a[0..n-1]` (a day
can be a loss, so values may be negative). The owner will *harvest* the grapes in a number of
**lots**. A lot is a **contiguous block of days** `[j, i-1]`, and machine setup makes a lot worthwhile
only if it spans **at least `L` days** (`i - j >= L`). Lots may not overlap, and any day may be left
unharvested. The profit of a lot is the sum of its daily values (including the losing days inside it,
which still cost money). The owner may also harvest nothing.

Choose a set of non-overlapping lots, each of length at least `L`, to **maximize the total profit**.
Output that maximum. Because harvesting nothing is allowed, the answer is always at least `0`.

This is the "minimum-length segments, weighted, maximize" member of the prefix-sum segment family: the
sum of any candidate lot `[j, i-1]` is `P[i] - P[j]` once prefix sums `P` are precomputed, which is the
whole reason prefix sums show up here.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `L` (`0 <= n <= 2*10^5`, `1 <= L <= 2*10^5`;
  when `n >= 1` you are guaranteed `L <= n` is *not* assumed — `L` may exceed `n`, in which case no lot
  fits and the answer is `0`). Then `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable total profit.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8`, `L = 3`, `a = [3, -1, 4, -10, 2, 2, -1, 5]` the answer is `14` (take the lot
`[0,2] = 3-1+4 = 6` and the lot `[4,7] = 2+2-1+5 = 8`; the day with `-10` is left unharvested).

## Evaluation settings

Judged on hidden tests covering: all-positive days, mixes with negatives and zeros, the empty input
(`n = 0`), `L = 1`, `L = n`, `L > n` (no lot fits, answer `0`), all-negative days (answer `0`),
adversarial instances that stress the selection across lots, and large `n = 2*10^5` with values near
`10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums: P[i] = a[0] + ... + a[i-1], so sum(a[j..i-1]) = P[i] - P[j].
    vector<long long> P(n + 1, 0);
    for (int i = 0; i < n; i++) P[i + 1] = P[i] + a[i];

    // TODO: maximum total over non-overlapping lots, each of length >= L (empty allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
