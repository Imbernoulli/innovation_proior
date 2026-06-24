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
whole reason prefix sums show up here. The catch is the *selection* across lots — a local "grab the best
lot" rule is tempting and wrong.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `L` (`0 <= n <= 2*10^5`, `1 <= L <= 2*10^5`;
  when `n >= 1` you are guaranteed `L <= n` is *not* assumed — `L` may exceed `n`, in which case no lot
  fits and the answer is `0`). Then `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable total profit.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8`, `L = 3`, `a = [3, -1, 4, -10, 2, 2, -1, 5]` the answer is `14` (take the lot
`[0,2] = 3-1+4 = 6` and the lot `[4,7] = 2+2-1+5 = 8`; the day with `-10` is left unharvested).

## Background

The sum of a candidate lot `[j, i-1]` is `P[i] - P[j]` where `P[i] = a[0] + ... + a[i-1]`, so prefix
sums turn each lot's value into an `O(1)` lookup. Two families of approach are on the table before
committing:

- **Greedy by best lot.** Repeatedly find the single maximum-sum lot of length `>= L` among the
  still-free days, take it, and recurse on the free runs left and right; stop when no positive lot
  remains. It is simple and *feels* optimal. The open question is whether grabbing the globally best
  block can ever block a strictly better pair of blocks.
- **Linear prefix DP.** Sweep the day boundaries left to right; `dp[i]` is the best total over the first
  `i` days, either leaving day `i-1` unharvested (`dp[i-1]`) or closing a lot `[j, i-1]` of length
  `>= L` (`dp[j] + P[i] - P[j]`). The inner `max` over `j` is kept in `O(1)` by carrying the running
  best of `dp[j] - P[j]` as `j` becomes eligible. `O(n)` total; the open question is the exact eligibility
  offset and the data types.

## Evaluation settings

Judged on hidden tests covering: all-positive days, mixes with negatives and zeros, the empty input
(`n = 0`), `L = 1` (every length is allowed, so it reduces to "sum of all positive lots"), `L = n`
(only the whole array or nothing), `L > n` (no lot fits, answer `0`), all-negative days (answer `0`),
adversarial instances where the best single lot strictly beats greedy, and large `n = 2*10^5` with
values near `10^9` so the total can reach `~2*10^14` and exceed a 32-bit integer.

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
