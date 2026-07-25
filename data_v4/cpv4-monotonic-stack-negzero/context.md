# Widest-valley score: maximize min(subarray) times its length

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be negative, zero, or positive).
For any **non-empty** contiguous subarray `a[l..r]` define its *score* as

```
score(l, r) = min(a[l..r]) * (r - l + 1)
```

— the minimum value inside the window multiplied by the window's width. The **empty subarray is
also allowed and scores `0`**. Output the maximum score over all subarrays (empty included), so the
answer is always at least `0`.

When the values are non-negative this is exactly the "largest rectangle under a skyline" shape:
widening a window can only help because the multiplier is non-negative. The contract here permits
**negative and zero values** as well, so that reasoning does not automatically carry over. Getting
this exactly right is the kind of one-dimensional kernel that appears inside histogram, skyline, and
"best window" problems.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` there are no further tokens.
- Output (stdout): a single line with the maximum achievable score.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [2, 1, 5, 6, 2, 3]` the answer is `10`, attained by the window `a[2..3] = [5, 6]`,
whose minimum is `5` and width is `2` (`5 * 2 = 10`). The full array `[2,1,5,6,2,3]` has minimum `1`
and width `6`, scoring only `6`, which is why a wider window is not automatically better.

## Background

The constraint couples a *minimum* (which shrinks as the window grows) with a *length* (which grows),
so the score is non-monotone in the window and a naive `O(n^2)` scan over all `(l, r)` is the obvious
but too-slow baseline at `n = 2*10^5`.

- **Brute force over all subarrays.** For each left endpoint, extend right while tracking the running
  minimum, scoring each window. It is `O(n^2)`, obviously correct, and is exactly the oracle to test
  against — but it is too slow for the largest inputs.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays (the histogram case), arrays mixing negatives,
zeros, and positives, the empty array (`n = 0`), a single element (`n = 1`, possibly negative),
all-negative arrays, all-zero arrays, arrays with repeated equal values, and large `n = 2*10^5` with
values near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { cout << 0 << "\n"; return 0; }
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum over all subarrays (empty allowed, scoring 0) of
    //       min(a[l..r]) * (r - l + 1), efficiently enough for n up to 2*10^5.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
