# Counting "valid bursts" in an energy-pulse stream

## Research question

A sensor logs a stream of `n` energy pulses `a[0..n-1]`, each a **positive** integer (`a[i] >= 1`).
A monitoring rule flags a **burst**: any *contiguous* run of one or more pulses whose total energy is
at least `L` and at most `R`. Two bursts are different if they start or end at different positions
(even if they have equal total energy). Count how many distinct valid bursts exist.

Formally: count the number of index pairs `(i, j)` with `0 <= i <= j <= n-1` such that
`L <= a[i] + a[i+1] + ... + a[j] <= R`.

This is the canonical "count subarrays whose sum lands in a range" problem. Because the values are
positive, prefix sums are strictly increasing, which is exactly the structure that a two-pointer
sliding window exploits — but the *counting* layer on top of the window is where the off-by-one and
double-count mistakes live, so getting the boundary arithmetic exactly right is the whole game.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `L`, `R`
  (`0 <= n <= 2*10^5`, `1 <= L <= R <= 10^18`).
  The second line has `n` integers `a[i]` (`1 <= a[i] <= 10^9`), whitespace-separated.
  When `n = 0` the second line is empty (or absent).
- Output (stdout): a single line with the number of valid bursts.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 5`, `L = 3`, `R = 5`, `a = [2, 1, 3, 2, 1]` the answer is `5`. The valid bursts are
`[2,1]` (sum 3), `[1,3]` (sum 4), `[3]` (sum 3), `[3,2]` (sum 5), and `[2,1]` at the tail (sum 3).

## Background

The constraint "contiguous run with sum in `[L, R]`" is a range-membership count over subarrays.
Two families of approach are on the table before committing:

- **Brute force over all subarrays.** Fix a start `i`, extend the end `j`, accumulate the running
  sum, and count whenever it lands in `[L, R]`. This is `O(n^2)`; obviously correct but far too slow
  for `n = 2*10^5`. It is the reference oracle, not the submission.
- **Two-pointer sliding window with a difference of "at most" counts.** Because all `a[i] >= 1`, the
  prefix-sum sequence is strictly increasing, so for a fixed right endpoint the set of left endpoints
  yielding a window sum `<= X` is a *suffix* of valid starts. A single forward sweep with a shrinking
  left pointer computes `atMost(X)` = number of subarrays with sum `<= X` in `O(n)`. The range count
  is then `atMost(R) - atMost(L-1)`. This is `O(n)`; the open question is the exact window-shrink
  condition and the boundary `L-1`, both of which are easy to get subtly wrong.

## Evaluation settings

Judged on hidden tests covering: all-equal arrays (so many subarrays share a sum), exact-value
queries (`L = R`), windows above the total sum (answer `0`), windows that admit every subarray
(`L = 1`, `R >= total`), the empty stream (`n = 0`), single-element streams (in and out of range),
and large `n = 2*10^5` with values near `10^9` so prefix sums and the answer (up to `~2*10^10`
bursts, sums up to `~2*10^14`) overflow 32-bit integers.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count contiguous subarrays whose sum lies in [L, R], using a two-pointer
    //       sliding window (valid because every a[i] >= 1).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
