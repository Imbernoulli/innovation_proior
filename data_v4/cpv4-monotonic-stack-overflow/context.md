# Sum of subarray minimums

## Research question

You are given an array of `n` integers `a[0..n-1]` (values may be negative). Consider **every**
contiguous subarray `a[l..r]` with `0 <= l <= r < n` — there are `n*(n+1)/2` of them. For each such
subarray take its **minimum** element. Output the **sum of those minimums over all subarrays**.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 3*10^4`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the sum of the minimums over all `n*(n+1)/2` subarrays. The
  answer can be negative (e.g. all values negative), and its magnitude can be large.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 2, 4]` the answer is `17`. The ten subarrays and their minimums are
`[3]=3, [3,1]=1, [3,1,2]=1, [3,1,2,4]=1, [1]=1, [1,2]=1, [1,2,4]=1, [2]=2, [2,4]=2, [4]=4`, and
`3+1+1+1+1+1+1+2+2+4 = 17`.

## Background

The obvious approach enumerates every subarray directly.

- **Enumerate every subarray.** Two nested loops fix `l` and extend `r`, carrying a running minimum,
  accumulating each subarray's minimum. This is `O(n^2)` and obviously correct, but `n` up to
  `3*10^4` means up to `4.5*10^8` subarrays — too slow for a 1 second limit. Useful only as an
  oracle to check a faster method against.

## Evaluation settings

Judged on hidden tests covering: strictly increasing and strictly decreasing arrays, arrays with
many equal values (the tie-breaking corner), arrays with negatives and zeros, `n = 0`, `n = 1`,
all-negative arrays (answer is negative), and large `n = 3*10^4` with values near `10^9`.

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
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: sum, over all n*(n+1)/2 subarrays, of the minimum element.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
