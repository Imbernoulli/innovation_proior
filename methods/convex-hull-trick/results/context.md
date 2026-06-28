# Context

## Problem

A one-dimensional dynamic program has the transition

$$
dp[i] = \min_{0 \le j < i}\big(dp[j] + b[j]\cdot a[i]\big), \qquad i = 1,\dots,n-1,
$$

with base case $dp[0] = 0$. The array `a` supplies the query point for each
state, and `b[j]` is the coefficient attached to a finished state `j`, so state
`j` can only be used after `dp[j]` has already been computed.

Two monotonicity facts are guaranteed:

- `b[j]` is non-increasing as `j` grows;
- `a[i]` is non-decreasing as `i` grows.

The values can be negative, and products such as `b[j] * a[i]` can be large.
The direct transition scans every earlier `j` for every `i`, which is
$O(n^2)$. The target is to compute all `dp` values faster under the two
monotonicity assumptions.

## Code framework

The deliverable is a single self-contained C++17 program reading from stdin and
writing to stdout. It reads `n`, then arrays `a` and `b`, computes the dynamic
program, and prints `dp[n - 1]` followed by a newline.

```python
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n), b(n);
    for (int i = 0; i < n; ++i) cin >> a[i];
    for (int i = 0; i < n; ++i) cin >> b[i];

    vector<long long> dp(n);

    // TODO:

    cout << dp[n - 1] << "\n";
    return 0;
}
```
